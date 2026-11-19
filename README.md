# FORGE: Forklift-Oriented Recovery and Gap-Filling for End-to-End Vision–Language–Action Policies

FORGE trains a single end-to-end Vision–Language–Action (VLA) policy for a full forklift mission in NVIDIA Isaac Sim: **dock** a pallet, **carry** it along a warehouse aisle, **unload** it in the drop zone, and **return** to the staging area.
The policy is SmolVLA fine-tuned on demonstrations from a rule-based expert.
FORGE finds the mission stage where the policy fails and adds data for that stage:
- **SGF**: stage-only expert episodes.
- **TRP**: recovery data made by perturbing the vehicle during recording.

Training then continues from the previous round's checkpoint.

| Stage | Lift | Carry | Drop-off | Return (success) | Strict |
|---|---|---|---|---|---|
| FORGE (30 episodes, seeds 1000–1005) | 28 | 27 | 27 | **26** | 23 |

## Contents

```
run_forge.sh              end-to-end reproduction: record -> datasets -> training rounds -> evaluation
scenes/factory_scene.usd  warehouse scene (Isaac Sim 6.0 assets)
scripts/
  forge_sim.py            simulator: expert controller, data recording, policy evaluation
  planner.py              Hybrid A* detour planner on an occupancy grid (used by the expert)
  mid360.py               simulated Livox Mid-360 LiDAR
  env_rand.py             per-episode appearance randomization
  to_lerobot.py           converts recorded episodes to a LeRobot v3 dataset
  train.sh                one training round (auto-resume after a crash)
  train_lerobot.py        lerobot-train wrapper used by train.sh
  policy_server.py        HTTP inference server (POST /act)
  serve.sh                starts the server and restarts it if it crashes
  summary.py              stage-wise evaluation table
```

## Requirements

Tested on Ubuntu 24.04 with an NVIDIA RTX PRO 4500 Blackwell (32 GB).

- **Isaac Sim 6.0.0** (pip install, Python 3.12). Use it in a conda env, e.g. `isaacsim`.
- **LeRobot 0.6.1** with SmolVLA in a second conda env, e.g. `lerobot`:
  `pip install "lerobot[smolvla]==0.6.1"` and `conda install -c conda-forge ffmpeg`.
  Blackwell GPUs need a CUDA 12.8+ PyTorch build.
- Python packages for the simulator env: `numpy scipy opencv-python pillow`.

## Quick start: evaluate the released policy

1. Download `forge_policy.tar.gz` from the [Releases](../../releases) page and extract it into the repository folder. This creates `checkpoints/forge/pretrained_model`.
2. Start the policy server in a first terminal, inside the `lerobot` env:
   ```bash
   bash scripts/serve.sh checkpoints/forge/pretrained_model 8001   # wait for "[server] ready on port 8001"
   ```
3. Run evaluation episodes in a second terminal, inside the `isaacsim` env:
   ```bash
   python scripts/forge_sim.py --run eval_demo_s1000 --episodes 5 --seed 1000 --rand --lidar --loc-noise \
       --top-gif 2.0 --policy-url http://127.0.0.1:8001/act
   ```
   - Add `--headless --close` to run without a window.
   - Each run folder `captures/<run>/` contains:
     - `episodes.csv`: one row per episode, with result, lift height, carried distance and drop-off error.
     - `run.log`.
     - per-episode camera frames and a top-view GIF (`ep_NNN/top.gif`).
4. Summarize the results:
   ```bash
   python scripts/summary.py captures/eval_demo_s1000
   ```

The paper's 30 evaluation episodes are seeds 1000–1005 with 5 episodes each. The same seed always gives the same pallet layout and the same obstacles.
To run them all, use `CKPT=checkpoints/forge/pretrained_model bash run_forge.sh eval`.

## Watch the expert

```bash
python scripts/forge_sim.py --run expert_demo --episodes 2 --rand --lidar
```

## Full reproduction

```bash
bash run_forge.sh          # or one stage at a time: record | datasets | train | eval
```

| Stage | What it does |
|---|---|
| record | Records expert episodes (10 per batch, fixed seeds). **full**: 300 full-mission episodes. **sgf**: 300 docking-stage episodes (`--dock-only`). **trp**: 150 full-mission episodes with steering perturbation (`--perturb`). **dockpert**: 300 docking-stage episodes with stronger docking perturbation. |
| datasets | Builds LeRobot datasets from successful episodes only. Cameras: front, fork, rear. State: lift, steering, speed, and the goal pose in the vehicle frame. Action: speed, steering, lift. |
| train | Four rounds, each starting from the previous checkpoint. r1: 40k steps on base+SGF, starting from `lerobot/smolvla_base`. r2: 120k steps on base+SGF. r3: 60k steps after adding transport TRP. r4: 60k steps after adding docking TRP; this is the final FORGE policy. |
| eval | Seeds 1000–1005 × 5 episodes with localization noise, followed by `summary.py`. |

All stages skip finished work, so the script can be re-run after an interruption.
Recording is the slowest part: each batch of 10 full-mission episodes takes about 3.5 h of wall time on our machine. `PAR` sets how many simulator processes run at once.

## Main simulator options (`scripts/forge_sim.py`)

| Option | Meaning |
|---|---|
| `--record` | save camera frames, state and expert action at 10 Hz |
| `--policy-url URL` | drive with a learned policy instead of the expert |
| `--rand` | randomize the pallet layout, aisle-side objects and crossing pedestrians |
| `--env-rand` | randomize floor, wall, rack, pallet and box colors and the lighting per episode |
| `--loc-noise` | add SLAM-like localization error to the goal pose given to the policy |
| `--lidar` | add the simulated Mid-360 (safety stops use it with the two safety scanners) |
| `--dock-only S` | stop the episode S seconds into the aisle (stage-only data) |
| `--perturb` | add temporary steering perturbations during execution; the recorded action stays the expert's correction |
| `--headless --close` | no window; exit when finished |

## Citation

The paper is under review. Citation information will be added after publication.
