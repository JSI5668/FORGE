#!/bin/bash
# Fine-tune SmolVLA on a FORGE dataset (one training round).
#
#   bash scripts/train.sh <name> <steps> <dataset dir> [init policy]
#
#   <name>         output folder outputs/train/<name> (log: outputs/train/<name>.log)
#   <steps>        number of training steps in this round
#   <dataset dir>  LeRobot dataset made by scripts/to_lerobot.py
#   [init policy]  checkpoint to start from (default: lerobot/smolvla_base);
#                  later rounds start from the previous round's pretrained_model
#
# Optional environment variables:
#   TRAIN_EXTRA   extra lerobot-train arguments for a fresh start,
#                 e.g. "--policy.scheduler_decay_steps=120000"
#   MAX_TRIES     automatic resume attempts after a crash (default 100)
#
# Checkpoints are saved every 500 steps. If the process dies (we saw sporadic CUDA
# driver segfaults), the script resumes from the last checkpoint automatically.
# Run it inside the `lerobot` environment (see README).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1; STEPS=$2; DATA=$(realpath "$3"); INIT=${4:-lerobot/smolvla_base}
[ -z "$NAME" ] || [ -z "$STEPS" ] || [ ! -d "$DATA" ] && { echo "usage: bash scripts/train.sh <name> <steps> <dataset dir> [init policy]"; exit 1; }
RENAME='{"observation.images.front":"observation.images.camera1","observation.images.fork":"observation.images.camera2","observation.images.rear":"observation.images.camera3"}'
OUT=$ROOT/outputs/train/$NAME
LOG=$ROOT/outputs/train/$NAME.log
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1   # avoided the driver segfaults on our machine (about 6% slower)
mkdir -p $ROOT/outputs/train
for try in $(seq 1 ${MAX_TRIES:-100}); do
  CFG=$OUT/checkpoints/last/pretrained_model/train_config.json
  if [ ! -f "$CFG" ] && [ -d "$OUT/checkpoints" ]; then
    # 'last' was broken by a crash during saving: relink it to the newest complete checkpoint
    GOOD=$(ls -d $OUT/checkpoints/[0-9]*/ 2>/dev/null | sort | while read d; do
             [ -f "$d/pretrained_model/train_config.json" ] && [ -f "$d/training_state/training_step.json" ] && echo "$d"; done | tail -1)
    [ -z "$GOOD" ] && { echo "[train] no complete checkpoint in $OUT — stopping" >> $LOG; exit 1; }
    ln -sfn "$(basename $GOOD)" "$OUT/checkpoints/last"
  fi
  if [ -f "$CFG" ]; then
    echo "[train] $(date '+%F %T') resume (attempt $try)" >> $LOG
    python $ROOT/scripts/train_lerobot.py --config_path=$CFG --resume=true --save_freq=500 >> $LOG 2>&1
  else
    rm -rf "$OUT"
    echo "[train] $(date '+%F %T') start from $INIT (attempt $try)" >> $LOG
    python $ROOT/scripts/train_lerobot.py --policy.path=$INIT --dataset.repo_id=local/forge --dataset.root=$DATA \
      --dataset.video_backend=pyav --tolerance_s=1e-3 ${TRAIN_EXTRA:-} \
      --rename_map="$RENAME" \
      --batch_size=32 --steps=$STEPS --log_freq=200 --save_freq=500 --num_workers=8 \
      --output_dir=$OUT --job_name=$NAME --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false >> $LOG 2>&1
  fi
  rc=$?
  echo "[train] $(date '+%F %T') exit code $rc" >> $LOG
  [ $rc -eq 0 ] && exit 0
  sleep 5
done
exit 1
