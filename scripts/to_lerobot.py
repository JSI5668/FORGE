# Convert episodes recorded with forge_sim.py --record into a LeRobot v3 dataset
#
#   python scripts/to_lerobot.py captures/rec_* --out datasets/forge --task "pick up the pallet, deliver it to the drop zone and come back"
#
# - Only successful episodes are included (result == success in episodes.csv)
# - Three cameras: observation.images.front (front of the body, 20° down), observation.images.fork (between the forks, moves with the lift),
#   observation.images.rear (rear of the body, 30° down; the view used when reversing with a load)
# - observation.state = [lift position m, steering angle rad, forward speed m/s, goal forward m, goal left m, goal heading difference rad]
#   (goal = pose the forklift should reach at drop-off, in the forklift frame), action = [speed command m/s, steering command rad, lift command m]
# - 10 fps (0.1 s control period)
import argparse
import csv
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", help="captures/<run> directories")
p.add_argument("--out", default="datasets/forge")
p.add_argument("--repo-id", default="local/forge")
p.add_argument("--task", default="pick up the pallet, deliver it to the drop zone and come back")
p.add_argument("--max-episodes", type=int, default=0, help="0 = all")
p.add_argument("--overwrite", action="store_true")
args = p.parse_args()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

FPS = 10
CAMS = ["front", "fork", "rear"]
STATE_NAMES = ["lift_pos", "steer_angle", "speed", "goal_fwd", "goal_left", "goal_yaw"]
STATE_COLS = ["state_lift", "state_steer", "state_speed", "state_goal_fwd", "state_goal_left", "state_goal_yaw"]
ACTION_NAMES = ["speed_cmd", "steer_cmd", "lift_cmd"]

eps = []
for run in args.runs:
    with open(os.path.join(run, "episodes.csv"), newline="") as f:
        for r in csv.DictReader(f):
            ed = os.path.join(run, f"ep_{int(r['ep']):03d}")
            if r["result"] == "success" and os.path.exists(os.path.join(ed, "data.csv")):
                eps.append(ed)
if args.max_episodes:
    eps = eps[:args.max_episodes]
if not eps:
    sys.exit("[convert] no successful episodes found")
h, w = np.asarray(Image.open(os.path.join(eps[0], "frames", "front", "00000.jpg"))).shape[:2]
print(f"[convert] episodes {len(eps)}, images {w}×{h} -> {args.out}", flush=True)

if os.path.exists(args.out):
    if not args.overwrite:
        sys.exit(f"[convert] {args.out} already exists (use --overwrite to replace it)")
    shutil.rmtree(args.out)

features = {f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
            for c in CAMS}
features["observation.state"] = {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES}
features["action"] = {"dtype": "float32", "shape": (3,), "names": ACTION_NAMES}
ds = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features, root=args.out,
                           robot_type="forklift_b", use_videos=True, image_writer_threads=4)

t0 = time.time()
n_frames = 0
for k, ed in enumerate(eps):
    with open(os.path.join(ed, "data.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        i = int(r["frame_index"])
        frame = {f"observation.images.{c}": np.asarray(Image.open(os.path.join(ed, "frames", c, f"{i:05d}.jpg")).convert("RGB"))
                 for c in CAMS}
        frame["observation.state"] = np.array([float(r[c]) for c in STATE_COLS], dtype=np.float32)
        frame["action"] = np.array([float(r["action_speed"]), float(r["action_steer"]), float(r["action_lift"])],
                                   dtype=np.float32)
        frame["task"] = args.task
        ds.add_frame(frame)
    ds.save_episode()
    n_frames += len(rows)
    print(f"[convert] {k + 1}/{len(eps)} {ed} ({len(rows)} frames, elapsed {time.time() - t0:.0f} s)", flush=True)
ds.finalize()
print(f"[convert] done: episodes {len(eps)}, frames {n_frames} ({n_frames / FPS / 60:.1f} min) -> {args.out}", flush=True)
