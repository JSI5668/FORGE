#!/bin/bash
# Reproduce FORGE end to end: record expert data -> build datasets -> train in rounds -> evaluate.
#
#   bash run_forge.sh [record|datasets|train|eval|all]     (default: all)
#
# Each stage skips work that is already finished, so the script can be re-run after an interruption.
# Settings (environment variables):
#   ISAAC_ENV    conda env with Isaac Sim 6.0           (default: isaacsim)
#   LEROBOT_ENV  conda env with LeRobot 0.6.1           (default: lerobot)
#   PAR          simulator processes run in parallel    (default: 3)
#   CKPT         checkpoint to evaluate                 (default: the final round below)
#
# Data (the seed of batch i is fixed, so recordings are reproducible):
#   full    : 30 batches x 10 full-mission expert episodes, seeds 600-629                 -> D_base
#   sgf     : 30 batches x 10 docking-stage episodes (--dock-only), seeds 700-729          -> D_SGF
#   trp     : 15 batches x 10 full-mission episodes with steering perturbation, seeds 800-814 -> D_TRP (transport)
#   dockpert: 30 batches x 10 docking-stage episodes with stronger docking perturbation,
#             seeds 900-929                                                             -> D_TRP (docking)
# Training rounds (each round starts from the previous round's final checkpoint):
#   r1: D_base + D_SGF                 40k steps from lerobot/smolvla_base
#   r2: D_base + D_SGF                120k steps (learning-rate decay over 120k)
#   r3: + D_TRP (transport)            60k steps
#   r4: + D_TRP (docking)  = D_FORGE   60k steps  -> final FORGE policy
# Evaluation: seeds 1000-1005 x 5 episodes = 30 episodes, with localization noise.
set -u
ROOT=$(cd "$(dirname "$0")" && pwd); cd $ROOT
STAGE=${1:-all}
ISAAC_ENV=${ISAAC_ENV:-isaacsim}; LEROBOT_ENV=${LEROBOT_ENV:-lerobot}; PAR=${PAR:-3}
TASK="pick up the pallet, deliver it to the drop zone and come back"
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p captures datasets outputs
say() { echo "[forge] $(date '+%F %T') $*" | tee -a outputs/run_forge.log; }
done_run() { [ -f captures/$1/episodes.csv ] && [ $(tail -n +2 captures/$1/episodes.csv | wc -l) -ge $2 ]; }

record_set() {  # record_set <prefix> <first seed> <batches> <extra forge_sim args...>
  local P=$1 S0=$2 NB=$3; shift 3
  conda activate $ISAAC_ENV
  for i in $(seq 1 $NB); do
    local run=$(printf "%s_%03d" $P $i)
    done_run $run 10 && continue
    [ -d captures/$run ] && mv captures/$run captures/${run}_incomplete_$(date +%s)
    while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 20; done
    say "record $run (seed $((S0 + i - 1)))"
    python scripts/forge_sim.py --run $run --episodes 10 --seed $((S0 + i - 1)) --record --rand --lidar --env-rand --loc-noise \
      "$@" --headless --close > /dev/null 2>&1 &
    sleep 30
  done
  wait
}

convert() {  # convert <dataset name> <run prefixes...>
  local DS=$1; shift
  [ -f datasets/$DS/meta/info.json ] && { say "dataset $DS exists — skip"; return; }
  local RUNS=""; for p in "$@"; do RUNS="$RUNS $(ls -d captures/${p}_[0-9][0-9][0-9] | tr '\n' ' ')"; done
  conda activate $LEROBOT_ENV
  say "build datasets/$DS"
  python scripts/to_lerobot.py $RUNS --out datasets/$DS --task "$TASK" > outputs/${DS}_convert.log 2>&1 || { say "conversion failed (outputs/${DS}_convert.log)"; exit 1; }
}

train_round() {  # train_round <name> <steps> <dataset> <init> [decay steps]
  local NAME=$1 STEPS=$2 DS=$3 INIT=$4 DECAY=${5:-}
  local CK=outputs/train/$NAME/checkpoints/$(printf "%06d" $STEPS)/pretrained_model
  [ -f $CK/model.safetensors ] && { say "round $NAME done — skip"; return; }
  conda activate $LEROBOT_ENV
  say "train $NAME ($STEPS steps on $DS from $INIT)"
  TRAIN_EXTRA=${DECAY:+--policy.scheduler_decay_steps=$DECAY} bash scripts/train.sh $NAME $STEPS datasets/$DS $INIT \
    || { say "training $NAME failed (outputs/train/$NAME.log)"; exit 1; }
}

T=outputs/train
FINAL=$T/forge_r4/checkpoints/060000/pretrained_model

if [ $STAGE = record ] || [ $STAGE = all ]; then
  record_set rec_full 600 30
  record_set rec_sgf 700 30 --dock-only 5
  record_set rec_trp 800 15 --perturb
  record_set rec_dockpert 900 30 --dock-only 5 --perturb --pert-dock 1.0 --pert-dock-n 3 --pert-dock-steer 10 30 --pert-dock-dur 1.0 2.5
fi
if [ $STAGE = datasets ] || [ $STAGE = all ]; then
  convert forge_base_sgf rec_full rec_sgf
  convert forge_base_sgf_trp rec_full rec_sgf rec_trp
  convert forge_full rec_full rec_sgf rec_trp rec_dockpert
fi
if [ $STAGE = train ] || [ $STAGE = all ]; then
  train_round forge_r1 40000 forge_base_sgf lerobot/smolvla_base
  train_round forge_r2 120000 forge_base_sgf $T/forge_r1/checkpoints/040000/pretrained_model 120000
  train_round forge_r3 60000 forge_base_sgf_trp $T/forge_r2/checkpoints/120000/pretrained_model 60000
  train_round forge_r4 60000 forge_full $T/forge_r3/checkpoints/060000/pretrained_model 60000
fi
if [ $STAGE = eval ] || [ $STAGE = all ]; then
  CK=${CKPT:-$FINAL}
  [ -f $CK/model.safetensors ] || { say "checkpoint not found: $CK"; exit 1; }
  TAG=${EVAL_TAG:-eval_forge}
  conda activate $LEROBOT_ENV
  bash scripts/serve.sh $CK 8001 > outputs/${TAG}_server.log 2>&1 &
  SERVER=$!
  until grep -q "ready on port 8001" outputs/${TAG}_server.log; do sleep 5; done
  say "policy server ready ($CK)"
  conda activate $ISAAC_ENV
  for sd in 1000 1001 1002 1003 1004 1005; do
    run=${TAG}_s$sd
    done_run $run 5 && continue
    [ -d captures/$run ] && mv captures/$run captures/${run}_incomplete_$(date +%s)
    while [ $(jobs -rp | wc -l) -ge $((PAR + 1)) ]; do sleep 20; done
    say "evaluate $run"
    python scripts/forge_sim.py --run $run --episodes 5 --seed $sd --rand --lidar --loc-noise --top-gif 2.0 \
      --policy-url http://127.0.0.1:8001/act --headless --close > /dev/null 2>&1 &
    sleep 30
  done
  while [ $(jobs -rp | wc -l) -gt 1 ]; do sleep 20; done
  pkill -P $SERVER; kill $SERVER 2>/dev/null
  python scripts/summary.py captures/${TAG}_s100[0-5] | tee -a outputs/run_forge.log
fi
say "done ($STAGE)"
