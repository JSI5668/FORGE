# FORGE forklift simulator (Isaac Sim, standalone).
#
# Simulates a walkie-stacker forklift in a warehouse and drives it with either
#   - a rule-based expert that uses ground-truth poses from the simulator, or
#   - a learned VLA policy queried over HTTP (--policy-url).
#
# Episode sequence (one full cycle):
#   1. Pick: approach the target pallet in the staging row, check alignment (reverse and retry if off),
#      lower the forks, insert, and lift.
#   2. Reverse out of the row, then spin in place (steered wheel at ±90°) so the rear faces the aisle.
#   3. Reverse along the aisle centerline (fixed route with rounded corners). Objects at the aisle edge are
#      handled by the safety stop and a Hybrid A* detour planner (planner.py); for people crossing the
#      aisle the vehicle stops and waits.
#   4. Spin 180° at the branch entrance so the forks face the end wall.
#   5. Drive forks-first to the drop zone at the end wall, check alignment, lower the pallet, and reverse
#      to pull the forks out.
#   6. Reverse back down the aisle to the staging area and spin to face the next pallet.
#
# Layout (sizes are provisional): a staging row of up to three pallet slots in the south-west, an aisle
# bounded by 1 m high blocks with 1 m gaps where people cross, and an eastern branch at the north end
# with a yellow drop zone. Layout prims, cameras and sensors are added at run time only; the scene file
# is never modified.
#
# Sensors:
#   - cameras: front (cam_front), fork (on the lift link, between the forks), rear (above the rear end,
#     looking back and down); these are the policy inputs
#   - two 270° safety scanners at diagonal corners, 0.15 m above the floor; their stop/slow fields
#     override only the wheel speed, never steering or lift
#   - optional simulated Livox Mid-360 (--lidar, see mid360.py): ray-cast point cloud used for the
#     safety stop and the occupancy map
#
# State  = [lift m, steer rad, forward speed m/s, goal forward m, goal left m, goal yaw rad]
# Action = [speed command, steer command, lift command]
#
# Main modes:
#   --record       record expert episodes (10 Hz cameras + state/action); convert with to_lerobot.py
#   --policy-url   evaluate a learned policy served by policy_server.py instead of the expert
#   --dock-only N  record docking plus N s of aisle reversing only (more docking frames per hour)
#   --perturb      DART-style steering perturbations while recording: the executed steering is offset,
#                  the recorded action is the expert's command for the actual state
#   --env-rand     randomize warehouse appearance per episode (env_rand.py)
#   --loc-noise    add SLAM-like localization error to the goal state given to the policy
#   --lidar        add the simulated Mid-360
#   --layout-only  render layout snapshots and exit
#
# Usage:
#   python scripts/forge_sim.py --run demo_001 --episodes 2 --rand --lidar --close
#   python scripts/forge_sim.py --run data_001 --episodes 10 --seed 100 --record --rand --lidar --headless --close
#   python scripts/forge_sim.py --run eval_001 --episodes 5 --seed 1000 --rand --lidar \
#       --policy-url http://127.0.0.1:8001/act --close
#
# Outputs (captures/<run>/):
#   episodes.csv     one row per episode (result, timings, alignment/drop errors, safety stops, ...)
#   trace.csv        0.1 s trace of pose, commands, pallet and safety state
#   run.log          console log; the last summary line starts with "[result]"
#   ep_NNN/data.csv, ep_NNN/frames/{front,fork,rear}/NNNNN.jpg   recorded or evaluated frames
#   ep_NNN/<stage>/*.png snapshots, ep_NNN/map/*.png and map.gif (occupancy map), optional top.gif
#   code/            copy of the scripts used for the run
import argparse
import csv
import math
import os
import random
import sys
import time

# Project root = parent of this script's folder (scripts/), so paths work on any machine
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--scene", default=os.path.join(ROOT, "scenes", "factory_scene.usd"))
p.add_argument("--pallet", default="/World/loaded_pallet_02")
p.add_argument("--side", choices=["-y", "+y"], default="-y", help="Pallet face to approach, in the pallet's local frame")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--dist", type=float, nargs=2, default=[1.5, 3.5], help="Initial fork-tip to pallet-face distance range [m]")
p.add_argument("--lat", type=float, default=0.6, help="Initial lateral offset of the fork center, ± [m]")
p.add_argument("--yaw", type=float, default=15.0, help="Initial heading error, ± [deg]")
p.add_argument("--lift-low", type=float, default=-0.145, help="Lift height during insertion [m]")
p.add_argument("--insert-vmin", type=float, default=0.2,
               help="Minimum insertion speed [m/s] (at 0.06 the forks stall from friction in the last 0.1 m)")
p.add_argument("--record", action="store_true", help="Record training data (cameras + state/action, 10 Hz)")
p.add_argument("--rand", action="store_true", help="Randomize pallet placement, yaw and lighting")
p.add_argument("--pallet-x", type=float, nargs=2, default=[3.5, 5.5], help="Legacy (unused in this layout): target pallet x range for --rand")
p.add_argument("--pallet-y", type=float, nargs=2, default=[-3.0, -2.0], help="Legacy (unused in this layout): target pallet y range for --rand")
p.add_argument("--pallet-yaw", type=float, default=25.0, help="Target pallet yaw range for --rand, ± [deg]")
p.add_argument("--light", type=float, nargs=2, default=[0.5, 1.5], help="Light intensity scale range for --rand")
p.add_argument("--res", type=int, nargs=2, default=[640, 360], help="Camera resolution")
p.add_argument("--policy-url", default=None, help="Policy server URL (e.g. http://127.0.0.1:8001/act)")
p.add_argument("--exec-steps", type=int, default=10, help="Actions executed per policy query (10 = 1 s)")
p.add_argument("--max-time", type=float, default=240.0, help="Episode time limit for policy evaluation [s, sim time]")
p.add_argument("--task", default="pick up the pallet and carry it to the door", help="Task instruction sent to the policy (must match training)")
p.add_argument("--calib", action="store_true", help="Steering calibration only: drive with fixed steering and measure turn radius and fixed-axle position")
p.add_argument("--start-delay", type=float, default=0.0)
p.add_argument("--door-x", type=float, default=4.0, help="Legacy (unused in this layout): door center x [m] on the south wall y=-12.2")
p.add_argument("--layout-shift", type=float, default=7.0, help="Legacy (unused in this layout): shift of the three pallets toward +y [m]")
p.add_argument("--obstacle", choices=["none", "box", "person", "mixed"], default="none",
               help="Legacy: obstacles on the carry route (mixed = --n-boxes boxes + a walking person); here only used to load the person asset")
p.add_argument("--obstacle-mix", default="",
               help="Legacy: per-episode obstacle type sampled by probability, e.g. none:0.4,box:0.35,person:0.15,mixed:0.1 "
                    "(box = 1..--n-boxes boxes); here only used to load the person asset")
p.add_argument("--n-boxes", type=int, default=1, help="Legacy (unused in this layout): number of boxes (1 = one on the route; more = first on the route, others within ±2 m)")
p.add_argument("--box-clear", type=float, default=0.0, help="Remove the boxes after a safety stop lasting this long [s] (0 = never)")
p.add_argument("--no-detour", action="store_true", help="Disable detour planning (wait when blocked)")
p.add_argument("--handover", action="store_true",
               help="Experimental: during policy evaluation, hand control to the detour planner when the loaded carry is blocked and return it to the VLA after the detour (off = VLA + safety scanners only)")
p.add_argument("--lidar", action="store_true",
               help="Add a simulated Livox Mid-360 (mount from mid360.MOUNT: rear post at 2.2 m, inverted, tilted 20° back) and use its points for the safety stop and the detour map")
p.add_argument("--no-scanners", action="store_true",
               help="Disable the two safety scanners (with --lidar, the safety stop and map use the Mid-360 only)")
p.add_argument("--lidar-pts", type=int, default=4000,
               help="Mid-360 points per 0.1 s frame (the real sensor has ~20k; reduced because Python ray casting is slow, and the non-repetitive pattern fills in over a few frames)")
p.add_argument("--env-rand", action="store_true", help="Randomize warehouse appearance per episode (original appearance 25%%)")
p.add_argument("--loc-noise", action="store_true", help="Add SLAM-like localization error to the goal state given to the policy (recording and evaluation)")
p.add_argument("--save-lidar", action="store_true", help="Save raw Mid-360 points when recording (requires --lidar)")
p.add_argument("--dagger", action="store_true",
               help="Legacy/experimental DAgger: while the policy drives, a shadow expert (ExpertShadow) computes the expert action and writes it to the data.csv action columns (with probability --dagger-beta a 1 s chunk is executed by the expert)")
p.add_argument("--dagger-beta", type=float, default=0.2, help="DAgger: probability of executing the expert action for each 1 s chunk")
p.add_argument("--no-safety", action="store_true", help="Disable the safety stop (for comparison)")
p.add_argument("--rear-gap", type=float, default=2.0, help="Legacy (unused in this layout): forklift rear to wall distance when unloading [m]")
p.add_argument("--edge-mix", default="0:0.3,1:0.5,2:0.2", help="Probabilities for the number of aisle-edge objects (count:prob)")
p.add_argument("--person-prob", type=float, default=0.3, help="Probability that a person crosses the aisle")
p.add_argument("--dock-only", type=float, default=0.0,
               help="Docking-only recording: pick, back out, spin, then reverse along the aisle for this long [s] and end as success (0 = full cycle). Ending while moving avoids 'stopped' labels; the task text and goal state are the same as for the full cycle")
p.add_argument("--layout-only", action="store_true", help="Render layout snapshots only, then exit")
p.add_argument("--top-gif", type=float, default=0.0, help="Capture a top-down view every this many seconds [sim time] into ep_NNN/top.gif (0 = off)")
p.add_argument("--strict-drop", action="store_true",
               help="Use the strict drop criterion in policy evaluation (pallet center within ±0.3 m, yaw within ±5°). Default: the drop succeeds if the pallet center is inside the drop zone (1.6×1.8 m), yaw ignored")
p.add_argument("--perturb", action="store_true",
               help="While recording, briefly offset the steering to push the vehicle off its path (DART). The recorded action is the expert's command for the current state; "
                    "only the execution is perturbed, so the data contains drift-and-recover examples")
p.add_argument("--pert-rate", type=float, default=0.06, help="Probability per second of starting a perturbation during carry/return")
p.add_argument("--pert-steer", type=float, nargs=2, default=[10.0, 30.0], help="Carry/return perturbation magnitude range [deg] (random sign)")
p.add_argument("--pert-dur", type=float, nargs=2, default=[1.5, 3.0], help="Carry/return perturbation duration range [s]")
p.add_argument("--pert-dock", type=float, default=0.5, help="Probability per episode of perturbing the docking approach (see --pert-dock-steer/--pert-dock-dur)")
p.add_argument("--pert-dock-n", type=int, default=1, help="Number of docking-approach perturbations per episode (remaining ones also apply after a retry)")
p.add_argument("--pert-dock-steer", type=float, nargs=2, default=[8.0, 20.0], help="Docking-approach perturbation magnitude range [deg]")
p.add_argument("--pert-dock-dur", type=float, nargs=2, default=[1.0, 2.0], help="Docking-approach perturbation duration range [s]")
p.add_argument("--headless", action="store_true")
p.add_argument("--close", action="store_true")
p.set_defaults(pallet_yaw=10.0, max_time=600.0, task="pick up the pallet, deliver it to the drop zone and come back")
args = p.parse_args()
if args.dagger:
    sys.exit("[main] --dagger is not supported with this layout")

RUN_DIR = os.path.join(ROOT, "captures", args.run)
if os.path.exists(RUN_DIR):
    sys.exit(f"[expert] {RUN_DIR} already exists")
os.makedirs(RUN_DIR)
_logf = open(os.path.join(RUN_DIR, "run.log"), "w")
# Copy of the code used for this run (for later comparison or restoration)
import shutil  # noqa: E402
os.makedirs(os.path.join(RUN_DIR, "code"))
for _fn in [os.path.abspath(__file__)] + [os.path.join(os.path.dirname(os.path.abspath(__file__)), _m)
                                           for _m in ("planner.py", "mid360.py", "env_rand.py")]:
    shutil.copy2(_fn, os.path.join(RUN_DIR, "code"))
T0 = time.time()


def log(msg):
    msg = f"[{time.time() - T0:6.1f}s] {msg}"
    print(msg, flush=True)
    _logf.write(msg + "\n")
    _logf.flush()


log(f"[main] {' '.join(sys.argv)}")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading, open_stage  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

# ---- Constants ----
PHYSICS_DT = 1.0 / 60.0
CTRL_EVERY = 6                     # control period = 6 physics steps = 0.1 s
RENDER_EVERY = 4
FORK = "/World/forklift_b"
BODY = f"{FORK}/body"
WHEEL_RADIUS, DRIVE_SIGN = 0.150, -1.0
TIP_X = -2.146                     # fork tip (body-local x; the forks point toward -x)
LEG_X = -1.108                     # front end of the floor support legs -> insertion depth limit
FIX_X = -0.906                     # fixed axle = front rollers (point of zero lateral slip, measured with --calib)
STEER_L = 1.48                     # joint [rad] = atan(curvature × STEER_L); +joint = left turn (measured with --calib)
STEER_MAX = math.radians(45.0)
PALLET_HALF = 0.5                  # pallet center to wide face
LIFT_LOW, LIFT_HIGH = args.lift_low, 0.15
V_APPROACH, V_INSERT, V_REVERSE = 0.5, 0.25, 0.4
PRE_GAP = 0.25                     # pre-insertion check point: fork tip 0.25 m in front of the face
# Approach steering: for lateral error e and heading error ψ of the fixed-axle point, curvature κ = -(KP·e + KD·ψ)
#   (de/ds = sinψ, dψ/ds = κ -> critically damped, natural frequency √KP = 1.6 /m -> converges within ~3 m)
KP, KD = 2.56, 3.2
REVERSE_TO = 3.0                   # on a retry, reverse until the fork tip is this far from the face
TOL_LAT, TOL_YAW = 0.03, math.radians(1.5)   # insertion tolerance (fork-to-pocket lateral clearance is ±0.08 m)
MAX_RETRY = 3
PUSH_TOL = 0.03                    # insertion fails if the pallet is pushed this far
FORK_CAM = f"{FORK}/lift/_cam_fork"
FORK_CAM_POS = (-1.15, 0.0, 0.25)  # body-local (lift at 0): between the forks at fork height (at x -0.98 the chassis blocks the upper view)
FORK_CAM_TILT = 5.0                # downward [deg]
REAR_CAM = f"{BODY}/_cam_rear"
REAR_CAM_POS = (0.95, 0.0, 1.6)    # body-local: just behind the rear end (x 0.885), 1.6 m high
REAR_CAM_TILT = 30.0               # downward [deg]
DOOR_Y = -12.2                     # inner face of the south wall
CAMS = {"front_camera": f"{BODY}/cam_front", "fork_camera": FORK_CAM, "rear_camera": REAR_CAM}
if not args.record and not args.policy_url:
    CAMS.update(front_left_camera=f"{BODY}/cam_front_left", front_right_camera=f"{BODY}/cam_front_right")
REC_CAMS = {"front": "front_camera", "fork": "fork_camera", "rear": "rear_camera"}     # dataset name -> annotator name
BOUNDS = ((-9.3, 8.3), (-11.3, 16.0))   # allowed forklift area inside the warehouse (x, y)

log(f"[load] open {args.scene}")
open_stage(args.scene)
while is_stage_loading():
    app.update()
for _ in range(30):
    app.update()
stage = omni.usd.get_context().get_stage()
pal = stage.GetPrimAtPath(args.pallet)
if not pal.IsValid():
    log(f"[expert] pallet not found: {args.pallet}")
    app.close()
    sys.exit(1)

# ---- Physics settings applied at run time only ----
lj = stage.GetPrimAtPath(f"{FORK}/lift_joint")
lj.GetAttribute("drive:linear:physics:maxForce").Set(30000.0)
lj.GetAttribute("drive:linear:physics:stiffness").Set(1.0e6)
lj.GetAttribute("drive:linear:physics:damping").Set(1.0e5)
# Drive wheel: the scene's default position stiffness of 100 winds the wheel back as it turns (slow low-speed acceleration) -> pure velocity drive
stage.GetPrimAtPath(f"{FORK}/back_wheel_joints/back_wheel_drive").GetAttribute("drive:angular:physics:stiffness").Set(0.0)
# In-place spin: widen the steering joint limit from the scene default ±60° to ±95° (at ±90° the vehicle spins about the front rollers)
_sw = stage.GetPrimAtPath(f"{FORK}/back_wheel_joints/back_wheel_swivel")
_sw.GetAttribute("physics:lowerLimit").Set(-95.0)
_sw.GetAttribute("physics:upperLimit").Set(95.0)
SPIN_STEER = math.radians(90.0)
n_box = sum(1 for c in pal.GetChildren() if "CardBox" in c.GetName())
for pr in stage.GetPrimAtPath("/World").GetChildren():
    if "pallet" in pr.GetName() and pr.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(25.0 + 15.0 * n_box)

Mp = omni.usd.get_world_transform_matrix(pal)
PC0 = Mp.ExtractTranslation()
pyaw = math.atan2(Mp.GetRow(0)[1], Mp.GetRow(0)[0])
UX = UY = D = N = None


def set_frame(cx, cy, yaw):
    """Set the controller frame from the target pallet center and yaw: approach direction D and its left normal N."""
    global PC0, UX, UY, D, N
    PC0 = Gf.Vec3d(cx, cy, PC0[2])
    UX = np.array([math.cos(yaw), math.sin(yaw)])
    UY = np.array([-math.sin(yaw), math.cos(yaw)])
    D = UY * (1.0 if args.side == "-y" else -1.0)    # approach direction (world)
    N = np.array([-D[1], D[0]])                      # left of the approach direction


set_frame(PC0[0], PC0[1], pyaw)
log(f"[pallet] {args.pallet} center ({PC0[0]:.2f}, {PC0[1]:.2f}) yaw {math.degrees(pyaw):.1f}°, approach direction ({D[0]:+.2f}, {D[1]:+.2f})")

# Fork camera: attached to the lift link so it moves with the forks (added at run time only)
_Mb = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(BODY))
_Ml = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(f"{FORK}/lift"))
_t = math.radians(FORK_CAM_TILT)
_cam_in_body = Gf.Matrix4d(0, 1, 0, 0,                                  # camera x (right) = body +y
                           -math.sin(_t), 0, math.cos(_t), 0,            # camera y (up)
                           math.cos(_t), 0, math.sin(_t), 0,             # camera z = opposite of the view direction (-x, tilted down)
                           *FORK_CAM_POS, 1)
fcam = UsdGeom.Camera.Define(stage, FORK_CAM)
fcam.CreateFocalLengthAttr(10.0)                     # horizontal FOV ~92°
fcam.CreateHorizontalApertureAttr(20.955)
fcam.CreateVerticalApertureAttr(20.955 * args.res[1] / args.res[0])
fcam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 200.0))
UsdGeom.Xformable(fcam.GetPrim()).AddTransformOp().Set(_cam_in_body * _Mb * _Ml.GetInverse())

# Rear camera: attached to the body (independent of the lift), looking back (+x) and tilted down by REAR_CAM_TILT
_t = math.radians(REAR_CAM_TILT)
rcam = UsdGeom.Camera.Define(stage, REAR_CAM)
rcam.CreateFocalLengthAttr(10.0)                     # horizontal FOV ~92°
rcam.CreateHorizontalApertureAttr(20.955)
rcam.CreateVerticalApertureAttr(20.955 * args.res[1] / args.res[0])
rcam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 200.0))
UsdGeom.Xformable(rcam.GetPrim()).AddTransformOp().Set(
    Gf.Matrix4d(0, -1, 0, 0,                                             # camera x (right) = body -y
                math.sin(_t), 0, math.cos(_t), 0,                         # camera y (up)
                -math.cos(_t), 0, math.sin(_t), 0,                        # camera z = opposite of the view direction (+x, tilted down)
                *REAR_CAM_POS, 1))

ER = None
if args.env_rand:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import env_rand as ERM  # noqa: E402
    ER = ERM.EnvRand(stage, log)
if args.save_lidar and not args.lidar:
    sys.exit("[main] --save-lidar requires --lidar")

# Safety scanner visuals (black cylinders on the body, no collision). Positions match SCANNERS below
for _nm, _sx, _sy in ((() if args.no_scanners else (("front", -0.98, 0.58), ("rear", 0.93, -0.58)))):
    _cy = UsdGeom.Cylinder.Define(stage, f"{BODY}/_scanner_{_nm}")
    _cy.CreateRadiusAttr(0.05)
    _cy.CreateHeightAttr(0.1)
    _cy.CreateAxisAttr("Z")
    _cy.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.05)])
    UsdGeom.Xformable(_cy.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(_sx, _sy, 0.15 - float(_Mb.ExtractTranslation()[2])))

# Mid-360 visual and sensor
LIDAR = None
if args.lidar:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mid360 as MID  # noqa: E402
    MID.add_visual(stage, BODY)
    LIDAR = MID.Mid360("mid360", MID.MOUNT["pos"], MID.MOUNT["inverted"], MID.MOUNT["tilt_deg"], noise_std=0.02)

# Drop zone marking: yellow floor outline, no collision
from pxr import UsdShade, Sdf  # noqa: E402


def _box(path, center, size, color):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(c.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))


REAR_X = 0.885                     # rear end of the chassis (body-local x)
PAL_FROM_BODY = -LEG_X + PALLET_HALF + 0.11         # body origin to pallet center when unloading (forward direction) ≈ 1.72 m
                                                    # (geometry gives 1.63, but measured pallets sit 0.09 m further forward on average)
# ---- Warehouse layout (sizes are provisional) ----
V4_CX, V4_CW = 2.0, 3.5                             # aisle centerline x and width -> aisle x 0.25~3.75
V4_X0, V4_X1 = V4_CX - V4_CW / 2, V4_CX + V4_CW / 2
V4_BW, V4_BH = 0.6, 1.0                             # boundary block thickness and height
V4_TOP = 14.6                                       # north end of the aisle (inner face of the boundary blocks)
V4_BR_Y0 = 10.0                                     # inner face of the east branch's south boundary -> branch y 10.0~14.6 (4.6 m wide: room for a 180° spin)
V4_BR_CY = 12.3                                     # branch centerline y (during the spin the rear corner, radius 1.96 m, has 2.3 m to the north end; the load side, 1.45 m, has 2.3 m to the south)
V4_SPIN_X = 4.8                                     # spin point at the branch entrance (x of reference point f)
V4_LEFT_Y0, V4_RIGHT_Y0 = -3.0, -3.5                # start y of the west/east aisle boundaries (south entrance)
V4_GAPS = (0.2, 3.6, 7.0)                           # center y of the 1 m boundary gaps where people cross
V4_GAP_W = 1.0
V4_EAST_WALL = 9.41                                 # inner face of the warehouse east wall
V4_ROW_Y = -2.5                                     # staging row center y (pallet face at y -3.0; the forklift enters from the south toward +y)
V4_SLOTS_X = (-5.2, -3.4, -1.6)
# Unloading is forks-first: spin 180° at the branch entrance, then drive east and set the pallet down in front of the end wall
GOAL_D = np.array([1.0, 0.0])                       # forward (fork) direction when unloading = east (toward the end wall)
GOAL_N = np.array([-GOAL_D[1], GOAL_D[0]])
ZONE = (V4_EAST_WALL - 0.35 - PALLET_HALF, V4_BR_CY)   # drop zone center: far pallet face 0.35 m from the end wall
GOAL = np.array(ZONE) - PAL_FROM_BODY * GOAL_D      # body-origin target when unloading
ZONE_W, ZONE_L = 1.8, 1.6                           # width across and length along the approach direction (pallet 1.21×1.0 + margin)
ZONE_XY = (ZONE_L, ZONE_W) if abs(GOAL_D[0]) > 0.5 else (ZONE_W, ZONE_L)   # x, y size for map drawing
_lw = 0.1                                           # floor line width
UsdGeom.Xform.Define(stage, "/World/_carry_door")
for nm, sgn_d, sgn_n, ld, ln in [("zone_a", 1, 0, _lw, ZONE_W), ("zone_b", -1, 0, _lw, ZONE_W),
                                 ("zone_c", 0, 1, ZONE_L, _lw), ("zone_d", 0, -1, ZONE_L, _lw)]:
    c_ = np.array(ZONE) + sgn_d * ZONE_L / 2 * GOAL_D + sgn_n * ZONE_W / 2 * GOAL_N
    sx_, sy_ = (ld, ln) if abs(GOAL_D[0]) > 0.5 else (ln, ld)
    _box(f"/World/_carry_door/{nm}", (c_[0], c_[1], 0.003), (sx_, sy_, 0.004), (0.95, 0.75, 0.05))


def _seg_y(y0, y1, gaps=V4_GAPS):
    """Split y0~y1 into segments, excluding the gaps (centers in gaps, width V4_GAP_W)."""
    cuts = sorted(g for g in gaps if y0 < g < y1)
    out, a = [], y0
    for g in cuts:
        out.append((a, g - V4_GAP_W / 2))
        a = g + V4_GAP_W / 2
    out.append((a, y1))
    return [(a_, b_) for a_, b_ in out if b_ - a_ > 0.05]


# Boundary blocks (with collision, visible to scanners and lidar). Rectangles (x0, y0, x1, y1), also used for the prior map (gaps included as blocked)
V4_BLOCKS = []
for a_, b_ in _seg_y(V4_LEFT_Y0, V4_TOP):
    V4_BLOCKS.append((V4_X0 - V4_BW, a_, V4_X0, b_))
for a_, b_ in _seg_y(V4_RIGHT_Y0, V4_BR_Y0):
    V4_BLOCKS.append((V4_X1, a_, V4_X1 + V4_BW, b_))
V4_BLOCKS.append((V4_X0 - V4_BW, V4_TOP, V4_EAST_WALL, V4_TOP + V4_BW))            # north end
V4_BLOCKS.append((V4_X1 + V4_BW, V4_BR_Y0 - V4_BW, V4_EAST_WALL, V4_BR_Y0))       # south side of the branch
V4_STATIC = [(V4_X0 - V4_BW, V4_LEFT_Y0, V4_X0, V4_TOP), (V4_X1, V4_RIGHT_Y0, V4_X1 + V4_BW, V4_BR_Y0)] + V4_BLOCKS[-2:]
_bmat = UsdShade.Material.Define(stage, "/World/_v4_block_mat")
V4_BLOCK_SH = UsdShade.Shader.Define(stage, "/World/_v4_block_mat/shader")
V4_BLOCK_SH.CreateIdAttr("UsdPreviewSurface")
V4_BLOCK_SH.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.42, 0.47, 0.5))
V4_BLOCK_SH.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
V4_BLOCK_SH.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.3)
_bmat.CreateSurfaceOutput().ConnectToSource(V4_BLOCK_SH.ConnectableAPI(), "surface")
UsdGeom.Xform.Define(stage, "/World/_v4_layout")
for _k, (x0_, y0_, x1_, y1_) in enumerate(V4_BLOCKS):
    _c = UsdGeom.Cube.Define(stage, f"/World/_v4_layout/block_{_k:02d}")
    _c.CreateSizeAttr(1.0)
    _xf = UsdGeom.Xformable(_c.GetPrim())
    _xf.AddTranslateOp().Set(Gf.Vec3d((x0_ + x1_) / 2, (y0_ + y1_) / 2, V4_BH / 2))
    _xf.AddScaleOp().Set(Gf.Vec3f(x1_ - x0_, y1_ - y0_, V4_BH))
    UsdShade.MaterialBindingAPI.Apply(_c.GetPrim()).Bind(_bmat)
    UsdPhysics.CollisionAPI.Apply(_c.GetPrim())
log(f"[layout] aisle x {V4_X0:.2f}~{V4_X1:.2f} (width {V4_CW}), north end y {V4_TOP}, drop branch y {V4_BR_Y0}~{V4_TOP}, {len(V4_BLOCKS)} boundary blocks, "
    f"gaps y {V4_GAPS} | drop target body ({GOAL[0]:.2f}, {GOAL[1]:.2f}) forward ({GOAL_D[0]:+.0f}, {GOAL_D[1]:+.0f}), "
    f"drop zone center ({ZONE[0]:.2f}, {ZONE[1]:.2f}) | pallet slots x {V4_SLOTS_X} y {V4_ROW_Y}")

# Top-down check camera (snapshots only)
top = UsdGeom.Camera.Define(stage, "/World/_carry_top")
top.CreateFocalLengthAttr(10.0)
top.CreateHorizontalApertureAttr(20.955)
top.CreateVerticalApertureAttr(20.955 * 9 / 16)
top.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
UsdGeom.Xformable(top.GetPrim()).AddTransformOp().Set(
    Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.0, 2.5, 24.0), Gf.Vec3d(1.0, 2.5, 0.0), Gf.Vec3d(0, 1, 0)).GetInverse())   # whole layout

# Lights (base intensities for random scaling)
from pxr import UsdLux  # noqa: E402
LIGHTS = []
for pr in stage.Traverse():
    if pr.HasAPI(UsdLux.LightAPI) and pr.GetAttribute("inputs:intensity").IsValid():
        LIGHTS.append((pr.GetAttribute("inputs:intensity"), pr.GetAttribute("inputs:intensity").Get()))
log(f"[setup] {len(LIGHTS)} lights")

# Observer camera: oblique view from above, beside the approach path
obs = UsdGeom.Camera.Define(stage, "/World/_expert_obs_cam")
obs.CreateFocalLengthAttr(12.0)
obs.CreateHorizontalApertureAttr(20.955)
obs.CreateVerticalApertureAttr(20.955 * 9 / 16)
obs_op = UsdGeom.Xformable(obs.GetPrim()).AddTransformOp()


def place_observer():
    mid = np.array([PC0[0], PC0[1]]) - 2.0 * D
    eye = mid + np.array([4.5, -4.0])                   # from above, south-east of the staging area
    obs_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(eye[0], eye[1], 4.0), Gf.Vec3d(mid[0], mid[1], 0.2),
                                       Gf.Vec3d(0, 0, 1)).GetInverse())


place_observer()
try:
    from omni.kit.viewport.utility import get_active_viewport
    get_active_viewport().camera_path = "/World/_expert_obs_cam"
except Exception:  # noqa: BLE001
    pass

annots = {}
for name, path in {**CAMS, "observer": "/World/_expert_obs_cam", "top": "/World/_carry_top"}.items():
    rp = rep.create.render_product(path, tuple(args.res))
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rp])
    annots[name] = a

# ---- Obstacles (added at run time, parked outside the warehouse when unused). Kinematic rigid bodies + colliders -> visible to ray casts ----
OBS_AWAY = (50.0, 50.0)
BOX_SIZE = (0.6, 0.6, 0.9)
PERSON_R = 0.25
PERSON_URL = ("https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/People/Characters/"
              "female_adult_police_01_new/female_adult_police_01_new.usd")
_mat = UsdShade.Material.Define(stage, "/World/_obs_box_mat")       # brown (displayColor alone renders white)
_sh = UsdShade.Shader.Define(stage, "/World/_obs_box_mat/shader")
_sh.CreateIdAttr("UsdPreviewSurface")
_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.36, 0.18))
_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
_mat.CreateSurfaceOutput().ConnectToSource(_sh.ConnectableAPI(), "surface")
# Four aisle objects: 0,1 = boxes (size randomized per run), 2,3 = drums (0.6 m diameter, 0.9 m high). Each episode picks from these
N_BOX = 4
V4_DRUM = (2, 3)
# Sizes are set once at startup: changing scale at run time invalidates the physics view ("Simulation view object is invalidated")
_rs = random.Random(args.seed * 31 + 7)
BOX_SIZES = [(_rs.uniform(0.5, 0.8), _rs.uniform(0.5, 0.8), _rs.uniform(0.6, 1.0)) for _ in range(2)] + [(0.6, 0.6, 0.9)] * 2
_dmat = UsdShade.Material.Define(stage, "/World/_obs_drum_mat")       # drum: blue metal
_dsh = UsdShade.Shader.Define(stage, "/World/_obs_drum_mat/shader")
_dsh.CreateIdAttr("UsdPreviewSurface")
_dsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.25, 0.55))
_dsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
_dsh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
_dmat.CreateSurfaceOutput().ConnectToSource(_dsh.ConnectableAPI(), "surface")
BOX_OPS = []                                                          # per object (translate, orient, scale)
for _k in range(N_BOX):
    if _k in V4_DRUM:
        _ob = UsdGeom.Cylinder.Define(stage, f"/World/_obs_box_{_k}")
        _ob.CreateRadiusAttr(0.5)
        _ob.CreateHeightAttr(1.0)
        _ob.CreateAxisAttr("Z")
        UsdShade.MaterialBindingAPI.Apply(_ob.GetPrim()).Bind(_dmat)
    else:
        _ob = UsdGeom.Cube.Define(stage, f"/World/_obs_box_{_k}")
        _ob.CreateSizeAttr(1.0)
        UsdShade.MaterialBindingAPI.Apply(_ob.GetPrim()).Bind(_mat)
    _xb = UsdGeom.Xformable(_ob.GetPrim())
    BOX_OPS.append((_xb.AddTranslateOp(), _xb.AddOrientOp(), _xb.AddScaleOp()))
    BOX_OPS[-1][0].Set(Gf.Vec3d(OBS_AWAY[0] + 2 * _k, OBS_AWAY[1], BOX_SIZE[2] / 2))   # start outside the warehouse (rendering at the origin even briefly leaves ghosting)
    BOX_OPS[-1][2].Set(Gf.Vec3f(*BOX_SIZES[_k]))
    UsdPhysics.CollisionAPI.Apply(_ob.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(_ob.GetPrim()).CreateKinematicEnabledAttr(True)
_op = UsdGeom.Xform.Define(stage, "/World/_obs_person")
PERSON_OPS = (_op.AddTranslateOp(), _op.AddOrientOp())
PERSON_OPS[0].Set(Gf.Vec3d(OBS_AWAY[0], OBS_AWAY[1] + 3, 0.0))
UsdPhysics.RigidBodyAPI.Apply(_op.GetPrim()).CreateKinematicEnabledAttr(True)
MIX = [(k_, float(w_)) for k_, w_ in (t_.split(":") for t_ in args.obstacle_mix.split(","))] if args.obstacle_mix else []
if args.person_prob > 0 or args.obstacle in ("person", "mixed") or any(k_ in ("person", "mixed") for k_, _ in MIX):
    _vis = stage.DefinePrim("/World/_obs_person/visual", "Xform")
    _vis.GetPayloads().AddPayload(PERSON_URL)
_pc = UsdGeom.Capsule.Define(stage, "/World/_obs_person/collider")      # person torso collider (invisible)
_pc.CreateRadiusAttr(PERSON_R)
_pc.CreateHeightAttr(1.2)
_pc.CreateAxisAttr("Z")
UsdGeom.Xformable(_pc.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.85))
_pc.MakeInvisible()
UsdPhysics.CollisionAPI.Apply(_pc.GetPrim())
for _ in range(20):
    app.update()
while is_stage_loading():
    app.update()

# ---- Start physics ----
sim = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, stage_units_in_meters=1.0, set_defaults=False)
timeline = omni.timeline.get_timeline_interface()
timeline.set_looping(False)
timeline.set_end_time(1.0e6)
sim.initialize_physics()
sim.play()
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

art = SingleArticulation(FORK)
art.initialize()
DOF = list(art.dof_names)
I_LIFT, I_STEER, I_DRIVE = DOF.index("lift_joint"), DOF.index("back_wheel_swivel"), DOF.index("back_wheel_drive")
body_rp = SingleRigidPrim(BODY)
body_rp.initialize()
pal_rp = SingleRigidPrim(args.pallet)
pal_rp.initialize()
PAL_POSE0 = pal_rp.get_world_pose()
OTHERS = []
for pr in stage.GetPrimAtPath("/World").GetChildren():
    if "pallet" in pr.GetName() and pr.HasAPI(UsdPhysics.RigidBodyAPI) and pr.GetPath() != pal.GetPath():
        r_ = SingleRigidPrim(str(pr.GetPath()))
        r_.initialize()
        OTHERS.append((r_, r_.get_world_pose()))
log(f"[setup] joints {DOF}")
box_rps = []
for _k in range(N_BOX):
    box_rps.append(SingleRigidPrim(f"/World/_obs_box_{_k}"))
    box_rps[-1].initialize()
person_rp = SingleRigidPrim("/World/_obs_person")
person_rp.initialize()


def set_obs_pose(which, pos, q):
    """Set an obstacle pose in both physics (API) and rendering (USD): for kinematic bodies the API alone does not move the rendered prim."""
    rp_, ops = (box_rps[which], BOX_OPS[which]) if isinstance(which, int) else (person_rp, PERSON_OPS)
    ops[0].Set(Gf.Vec3d(*map(float, pos)))
    ops[1].Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    rp_.set_world_pose(np.asarray(pos, dtype=float), np.asarray(q, dtype=float))


for _k in range(N_BOX):
    set_obs_pose(_k, (OBS_AWAY[0] + 2 * _k, OBS_AWAY[1], BOX_SIZE[2] / 2), (1.0, 0, 0, 0))
set_obs_pose("person", (OBS_AWAY[0], OBS_AWAY[1] + 3, 0.0), (1.0, 0, 0, 0))

cmd = {"v": 0.0, "steer": 0.0, "lift": 0.0}
PERT = {"steer": 0.0, "until": -1.0, "n": 0, "time": 0.0, "dock_t": None, "dock_left": 0, "rng": None}   # executed-steering perturbation (--perturb)
sim_t = {"t": 0.0, "n": 0}


def apply_cmd():
    pos = np.full(len(DOF), np.nan)
    vel = np.full(len(DOF), np.nan)
    pos[I_LIFT] = cmd["lift"]
    pos[I_STEER] = cmd["steer"] + PERT["steer"] if PERT["steer"] else cmd["steer"]   # perturbation affects execution only (recorded action = cmd)
    vel[I_DRIVE] = DRIVE_SIGN * SAFE["v"] / WHEEL_RADIUS        # speed after the safety monitor
    art.apply_action(ArticulationAction(joint_positions=pos, joint_velocities=vel))


# ---- Lidar + safety monitor ----
import carb  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
SQ = get_physx_scene_query_interface()
# Two safety scanners (diagonal corners, 0.15 m high, 270° each at 0.5°, 10 m range). Body-local (x: +rear, y: +right)
#   chassis cross-section at 0.15 m: x -0.95~+0.9, y -0.55~+0.55 -> the scanners sit 2-3 cm outside the corners
#   each ray reports only the closest hit (like a real scanner): rays that hit the chassis or the carried pallet first are occluded (no measurement)
SCANNERS = [("front", (-0.98, 0.58), math.radians(135.0)),       # front (fork side)-right corner; covers front, right and rear-right
            ("rear", (0.93, -0.58), math.radians(-45.0))]        # rear-left corner
SCAN_H, SCAN_FOV, SCAN_RES, SCAN_R = 0.15, math.radians(270.0), math.radians(0.5), 10.0
_SANG = np.arange(-SCAN_FOV / 2, SCAN_FOV / 2 + 1e-6, SCAN_RES)     # relative to the scanner's center direction


def _is_self(path):
    """Own chassis or the carried pallet (occluders)."""
    if path.startswith(FORK):
        return True
    return path.startswith(args.pallet) and float(art.get_joint_positions()[I_LIFT]) > 0.1


SCANRAYS = {"rays": np.zeros((0, 6))}


def scanner_points(b, fwd):
    """Points from both scanners -> (a forward +, l left +, is target pallet) relative to the body origin, plus the occluded fraction per scanner."""
    xa_w = -fwd                                            # world direction of body +x (rear)
    ya_w = np.array([-xa_w[1], xa_w[0]])                   # body +y
    left = np.array([-fwd[1], fwd[0]])
    pts, occl = [], {}
    rays = []                                              # for the map: (origin x, y, direction x, y, distance, hit)
    for name, (sx, sy), yaw in SCANNERS:
        o = b + sx * xa_w + sy * ya_w
        h0 = math.atan2(xa_w[1], xa_w[0]) + yaw
        n_occ = 0
        for a in _SANG:
            dx_, dy_ = math.cos(h0 + a), math.sin(h0 + a)
            r = SQ.raycast_closest(carb.Float3(float(o[0]), float(o[1]), SCAN_H), carb.Float3(dx_, dy_, 0.0), SCAN_R)
            if not r["hit"]:
                rays.append((o[0], o[1], dx_, dy_, SCAN_R, 0.0))
                continue
            pth = r["collision"] or r["rigidBody"]
            if _is_self(pth):
                n_occ += 1
                continue
            q = np.array([r["position"][0], r["position"][1]]) - b
            pts.append((float(q @ fwd), float(q @ left), pth.startswith(args.pallet)))
            rays.append((o[0], o[1], dx_, dy_, float(r["distance"]), 1.0))
        occl[name] = n_occ / len(_SANG)
    SCANRAYS["rays"] = np.array(rays) if rays else np.zeros((0, 6))
    return (np.array(pts) if pts else np.zeros((0, 3))), occl


LIDAR_RAYS = {"rays": np.zeros((0, 6))}
LIDAR_Z = (0.08, 3.0)                                      # only hits between these heights (above the floor) are obstacles; lower hits are floor (free)
LIDAR_STAT = {"pts": 0, "self": 0.0}


def mid_scan(M_body, n):
    """One Mid-360 frame (same computation as MID.Mid360.scan). Stacking a list of small arrays with np.array intermittently
    raised ValueError "could not broadcast (3,) into (3,)" or segfaulted, so floats are written directly into a preallocated array.
    Returns world points (m,3) and the self-occlusion fraction."""
    o, Rw = LIDAR.world(M_body)
    dirs = np.ascontiguousarray(LIDAR.frame_dirs(n) @ Rw.T, dtype=np.float64)
    ox, oy, oz = float(o[0]), float(o[1]), float(o[2])
    org = carb.Float3(ox, oy, oz)
    out = np.empty((len(dirs), 3), np.float64)
    noise = LIDAR.rng.normal(0.0, LIDAR.noise_std, len(dirs)) if LIDAR.noise_std else np.zeros(len(dirs))
    m = n_self = 0
    for k in range(len(dirs)):
        dx, dy, dz = float(dirs[k, 0]), float(dirs[k, 1]), float(dirs[k, 2])
        r = SQ.raycast_closest(org, carb.Float3(dx, dy, dz), MID.RANGE_MAX)
        if not r["hit"]:
            continue
        dist = float(r["distance"])
        if dist < MID.RANGE_MIN:
            continue
        if _is_self(r["collision"] or r["rigidBody"]):
            n_self += 1
            continue
        dist += float(noise[k])
        out[m, 0], out[m, 1], out[m, 2] = ox + dx * dist, oy + dy * dist, oz + dz * dist
        m += 1
    return out[:m].copy(), n_self / max(1, len(dirs))


def lidar_points(b, fwd):
    """One Mid-360 frame (0.1 s) -> obstacle points (a forward +, l left +, is target pallet) relative to the body origin.
    For the map only a short segment at the end of each 3D ray is used: an obstacle hit marks its cell blocked, a floor hit marks it free
    (clearing along the full ray would erase a box whenever a ray passes over it and hits the floor)."""
    pos, q = body_rp.get_world_pose()
    w, x, y, z = map(float, q)
    A = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    tw = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
    tgt = []

    def rc(o, d, rmax):
        r = SQ.raycast_closest(carb.Float3(*o.tolist()), carb.Float3(*d.tolist()), rmax)
        if not r["hit"]:
            return False, 0.0, ""
        pth = r["collision"] or r["rigidBody"]
        return True, float(r["distance"]), pth

    o, _ = LIDAR.world((A, tw))
    P, occ = mid_scan((A, tw), args.lidar_pts)
    LIDAR_STAT.update(pts=len(P), self=occ,
                      Pb=((P - tw) @ A).astype(np.float16) if len(P) else np.zeros((0, 3), np.float16))
    if not len(P):
        LIDAR_RAYS["rays"] = np.zeros((0, 6))
        return np.zeros((0, 3))
    # Target-pallet points are identified by position (scan returns no prim paths): within 0.8 m of the pallet center and below +1.3 m
    pc, _ = pallet_pose()
    is_t = (np.linalg.norm(P[:, :2] - pc[:2], axis=1) < 0.8) & (P[:, 2] < pc[2] + 1.3)
    obst = (P[:, 2] > LIDAR_Z[0]) & (P[:, 2] < LIDAR_Z[1])
    hor = P[:, :2] - o[:2]
    hn = np.linalg.norm(hor, axis=1, keepdims=True)
    dirs = np.where(hn > 1e-3, hor / np.maximum(hn, 1e-3), np.array([1.0, 0.0]))
    seg = 0.1
    LIDAR_RAYS["rays"] = np.concatenate([P[:, :2] - dirs * seg, dirs, np.full((len(P), 1), seg), obst[:, None].astype(float)], axis=1)
    left = np.array([-fwd[1], fwd[0]])
    Q = P[obst, :2] - b
    return np.stack([Q @ fwd, Q @ left, is_t[obst].astype(float)], axis=1) if len(Q) else np.zeros((0, 3))


FOOT_FRONT, FOOT_REAR, FOOT_HALF_W = 2.25, 0.9, 0.62     # chassis + load footprint rectangle (forward +, relative to the body origin)

# ---- Occupancy grid + detour planning ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planner as PL  # noqa: E402
import cv2  # noqa: E402
# Chassis + load around reference point f (fixed axle), 8 circles. Clearance 0.3 m > safety-field side margin 0.15 m, so that
# passing beside an obstacle during a detour does not trigger a safety stop that leaves the vehicle stuck
FOOT_F = PL.Footprint(-FOOT_REAR + FIX_X, FOOT_FRONT + FIX_X, FOOT_HALF_W, 0.3, n=8)
KAPPA_PLAN = 0.8 * math.tan(STEER_MAX) / STEER_L            # max planning curvature (80 % of the steering limit)
V_DETOUR = 0.3
STATIC_AGE = 1.5                                            # only detour around obstacles unchanged for this long (wait for people)
MAP_CROP = (-7.0, -10.5, 9.6, 15.6)               # staging area to drop branch
NAV = {}


def new_map():
    g = PL.OccGrid(-10.5, -12.4, 9.6, 18.2)
    g.set_outside_static(-10.33, -12.2, 9.41, 18.0)        # prior map: warehouse walls
    g.add_static_box(-10.33, -3.9, -8.6, 12.6)              # west-wall racks
    g.add_static_box(-9.8, 8.6, -5.1, 14.5)                 # boxes and carts in the north-west
    for x0_, y0_, x1_, y1_ in V4_STATIC:                    # aisle and branch boundaries (gaps included as blocked)
        g.add_static_box(x0_, y0_, x1_, y1_)
    return g


def f_pose(b, fwd):
    f = b + (-FIX_X) * fwd
    return float(f[0]), float(f[1]), math.atan2(fwd[1], fwd[0])


def map_update(b, fwd):
    g = NAV.get("grid")
    if g is None:
        return
    r = SCANRAYS["rays"]
    if len(r):
        g.update(r[:, 0:2], r[:, 2:4], r[:, 4], r[:, 5] > 0.5, sim_t["t"], SCAN_R)
    r = LIDAR_RAYS["rays"]
    if len(r):
        g.update(r[:, 0:2], r[:, 2:4], r[:, 4], r[:, 5] > 0.5, sim_t["t"], 0.15)
    g.clear_poly(FOOT_F.corners(*f_pose(b, fwd)))           # clear the forklift's own footprint (e.g. pallet traces from before lifting)


def nominal_rollout(b, fwd, dist=4.0, ds=0.1):
    """Predict the nominal carry controller (carry_reverse_steer) kinematically for dist m -> list of reference-point f poses."""
    x, y, th = f_pose(b, fwd)
    out = []
    for _ in range(int(dist / ds)):
        fw = np.array([math.cos(th), math.sin(th)])
        f = np.array([x, y])
        bb = f - (-FIX_X) * fw
        along = float((bb - GOAL) @ GOAL_D)
        if along <= 0.0:
            break
        psi = math.atan2(GOAL_D[0] * fw[1] - GOAL_D[1] * fw[0], GOAL_D @ fw)
        e_p = float((f - GOAL) @ GOAL_N) - CARRY_A * math.sin(psi)
        k = (math.sin(psi) - CARRY_K * e_p) / (CARRY_A * max(0.3, math.cos(psi)))
        k = max(-math.tan(STEER_MAX) / STEER_L, min(math.tan(STEER_MAX) / STEER_L, k))
        x, y, th = (float(v) for v in PL.arc(x, y, th, k, -ds))
        out.append((x, y, th, -1))
    return out


def blocking_age(poses, dist):
    """(blocked?, age in s of the oldest blocked cell) for the cells blocking the predicted path."""
    g = NAV["grid"]
    if not poses:
        return False, 0.0
    p = np.array([q[:3] for q in poses])
    cen = FOOT_F.circles(p[:, 0], p[:, 1], p[:, 2]).reshape(-1, 2)
    i, j = g.cell(cen)
    bad = dist[np.clip(i, 0, g.ny - 1), np.clip(j, 0, g.nx - 1)] < FOOT_F.r
    if not bad.any():
        return False, 0.0
    occ = (g.lo > PL.L_OCC)
    ages = []
    rr = int(math.ceil(FOOT_F.r / g.res))
    for ci, cj in zip(i[bad], j[bad]):
        win = (slice(max(ci - rr, 0), ci + rr + 1), slice(max(cj - rr, 0), cj + rr + 1))
        m = occ[win]
        if m.any():
            ages.append(float(np.max(sim_t["t"] - g.since[win][m])))
        elif g.static[win].any():
            ages.append(99.0)
    return True, (max(ages) if ages else 0.0)


def goal_region(x, y, th):
    """Detour goal = near the end pose of the current route, NAV['goal'] = (gx, gy, gth), reached in reverse: lateral ±0.15 m,
    -0.4~+0.6 m along the travel direction, fork heading ±8°. Going up: the spin point at the branch entrance; returning: the spin point in front of the next pallet."""
    gx, gy, gth = NAV["goal"]
    t = -np.array([math.cos(gth), math.sin(gth)])         # reverse travel direction
    d = np.array([x - gx, y - gy])
    return (abs(float(d[0] * t[1] - d[1] * t[0])) < 0.15 and -0.4 <= float(d @ t) <= 0.6
            and abs(PL.wrap(th - gth)) < math.radians(8.0))


def goal_region_v3(x, y, th):
    """Legacy detour goal: on the drop target line, 1.0~3.5 m in front of the target pose, heading ±6°, lateral ±0.12 m (reference point f).
    (Tighter windows made planning fail often; looser ones ended the detour too close and required realignment.)"""
    fG = GOAL + (-FIX_X) * GOAL_D
    q = np.array([x, y]) - fG
    return (abs(float(q @ GOAL_N)) < 0.12 and 1.0 <= float(q @ GOAL_D) <= 3.5
            and abs(PL.wrap(th - math.atan2(GOAL_D[1], GOAL_D[0]))) < math.radians(6.0))


def map_image(s, scale=1, text=""):
    g = NAV["grid"]
    fG = GOAL + (-FIX_X) * GOAL_D
    path = NAV.get("path") if NAV.get("mode") == "detour" else None
    sc = SAFE.get("scan")
    scan_xy = None
    if sc is not None and len(sc):
        left = np.array([-s["fwd"][1], s["fwd"][0]])
        scan_xy = s["b"][None, :] + sc[:, 0:1] * s["fwd"][None, :] + sc[:, 1:2] * left[None, :]
    return PL.render(g, MAP_CROP, path, f_pose(s["b"], s["fwd"]), FOOT_F, scan_xy,
                     goal_line=(tuple(fG), tuple(fG + 6.0 * GOAL_D)), zone=(ZONE, ZONE_XY), text=text, scale=scale)


def save_map(name, s, text):
    d = os.path.join(CUR["ed"], "map")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, name), map_image(s, 2, text))


def map_frame(s):
    """Map frame for the GIF, once per second."""
    if NAV.get("grid") is None or sim_t["t"] - NAV.get("last_frame", -9.0) < 1.0:
        return
    NAV["last_frame"] = sim_t["t"]
    txt = f"t {sim_t['t'] - NAV['t0']:5.1f}s  {NAV.get('mode', '-')}  {SAFE['state']}"
    NAV.setdefault("frames", []).append(cv2.cvtColor(map_image(s, 1, txt), cv2.COLOR_BGR2RGB))


def map_gif():
    fr = NAV.get("frames")
    if fr and CUR["ed"]:
        ims = [Image.fromarray(f) for f in fr]
        ims[0].save(os.path.join(CUR["ed"], "map.gif"), save_all=True, append_images=ims[1:], duration=150, loop=0)


def nav_plan(s, why, first_dir=0):
    """Plan a detour from the current pose to the current route goal."""
    t0 = time.time()
    planner = PL.HybridAStar(NAV["grid"], FOOT_F, KAPPA_PLAN)
    fG = GOAL + (-FIX_X) * GOAL_D
    gx, gy, gth = NAV["goal"]
    path, it = planner.plan(f_pose(s["b"], s["fwd"]), goal_region, (gx - 0.1 * math.cos(gth), gy - 0.1 * math.sin(gth)), first_dir)
    NAV["plans"] += 1
    ok = path is not None
    n_sw = sum(1 for a, b_ in zip(path[:-1], path[1:]) if a[3] != b_[3]) if ok else 0
    log(f"[nav] detour plan #{NAV['plans']} ({why}): {'ok' if ok else 'failed'}, {it} expansions, {time.time() - t0:.2f} s"
        + (f", length {0.15 * len(path):.1f} m, {n_sw} direction switches" if ok else ""))
    if ok:
        NAV.update(mode="detour", path=path, idx=0, detours=NAV["detours"] + 1, vmax=V_DETOUR)
    else:
        # On failure, do not replan for 3 s and wait in place (otherwise a slow plan can be retried in a tight loop)
        NAV["fails"] += 1
        NAV["no_plan_until"] = sim_t["t"] + 3.0
        NAV["mode"] = "wait"
    save_map(f"plan_{NAV['plans']}.png", s, f"plan #{NAV['plans']} {'OK' if ok else 'FAIL'}  t {sim_t['t'] - NAV['t0']:.1f}s\n"
             f"green = reverse, blue = forward")             # cv2 text supports ASCII only
    return ok


def nav_track(s):
    """Follow the detour path: pure pursuit on reference point f, one same-direction segment at a time. Returns True when done."""
    path = NAV["path"]
    P = np.array([q[:2] for q in path])
    x, y, th = f_pose(s["b"], s["fwd"])
    i = NAV["idx"]
    d = path[i][3]
    e = i
    while e + 1 < len(path) and path[e + 1][3] == d:
        e += 1
    seg = P[i:e + 1]
    k = int(np.argmin(np.linalg.norm(seg - [x, y], axis=1)))
    i = i + k
    NAV["idx"] = i
    rem = float(np.sum(np.linalg.norm(np.diff(P[i:e + 1], axis=0), axis=1))) if e > i else 0.0
    if rem < 0.12:
        if e + 1 >= len(path):
            return True
        NAV["idx"] = e + 1                                  # next segment (direction change)
        cmd.update(v=0.0)
        return False
    j = i
    while j < e and np.linalg.norm(P[j] - [x, y]) < 0.8:
        j += 1
    q = P[j] - [x, y]
    fw = np.array([math.cos(th), math.sin(th)])
    a, l = float(q @ fw), float(q @ np.array([-fw[1], fw[0]]))
    cmd["steer"] = kappa_to_steer(2.0 * l / max(a * a + l * l, 1e-6))
    cmd["v"] = d * max(0.08, min(NAV.get("vmax", V_DETOUR), 0.8 * rem))
    return False


def carry_control(s):
    """Carry: nominal reverse control <-> detour path tracking."""
    map_frame(s)
    can_plan = not args.no_detour and sim_t["t"] >= NAV["no_plan_until"]
    if NAV["mode"] == "wait":                              # planning failed -> wait, then replan (if the obstacle is cleared meanwhile, resume nominal)
        cmd.update(v=0.0)
        if can_plan:
            if goal_region(*f_pose(s["b"], s["fwd"])):              # near the spin point -> carry done; otherwise replan
                NAV["mode"] = "nominal"
                log("[nav] near the spin point -> carry done")
            else:
                nav_plan(s, "replanning while waiting", -SAFE.get("stop_dir", -1) if SAFE["state"] == "stop" else 0)
        if NAV["mode"] == "wait":
            return
    # If a safety stop lasts over 3 s and the obstacle remains (a person would have passed by then), replan from here
    if can_plan and SAFE["stop_run"] >= 3.0 and sim_t["t"] - NAV["last_stuck"] >= 3.0:
        NAV["last_stuck"] = sim_t["t"]
        # Replanning toward the blocked direction just triggers the stop again (looping) -> the first move goes the opposite way
        nav_plan(s, f"safety stop for {SAFE['stop_run']:.0f} s, backing away", -SAFE.get("stop_dir", -1))
    if NAV["mode"] == "detour":
        if sim_t["t"] - NAV["chk"] >= 0.5:
            NAV["chk"] = sim_t["t"]
            dist = NAV["grid"].dist_map()
            blk, age = blocking_age(NAV["path"][NAV["idx"]:], dist)
            if blk and age >= STATIC_AGE and sim_t["t"] >= NAV["no_plan_until"]:
                nav_plan(s, "detour blocked by a new obstacle", -SAFE.get("stop_dir", -1) if SAFE["state"] == "stop" else 0)
                if NAV["mode"] == "wait":
                    cmd.update(v=0.0)
                    return
        if NAV["mode"] == "detour":
            if nav_track(s):
                log(f"[nav] detour done -> nominal carry control ({goal_err(s)[0]:+.2f} m to the goal along the line)")
                NAV["mode"] = "nominal"
            else:
                return
    cmd.update(v=0.0)                                      # end of route (spin point) -> carry phase done
    return
    along, _, _ = goal_err(s)
    cmd["steer"] = carry_reverse_steer(s)
    cmd["v"] = -max(0.08, min(V_CARRY, 0.8 * along))
    if not args.no_detour and sim_t["t"] - NAV["chk"] >= 0.5:
        NAV["chk"] = sim_t["t"]
        dist = NAV["grid"].dist_map()
        blk, age = blocking_age(nominal_rollout(s["b"], s["fwd"]), dist)
        if blk and age >= STATIC_AGE and sim_t["t"] >= NAV["no_plan_until"]:
            if not nav_plan(s, f"nominal path blocked within 4 m, obstacle static for {age:.1f} s"):
                cmd.update(v=0.0)
FIELD_MARGIN = 0.15
V_SLOW = 0.15
CLEAR_HOLD = 1.0                                         # the field must stay clear this long before restarting
SAFE = {"v": 0.0, "state": "go", "clear_t": 0.0, "stops": 0, "stop_time": 0.0, "stop_run": 0.0, "min_clear": 99.0,
        "scan": None, "occl": {}}


def safety_update():
    """Every 0.1 s: scan -> check the fields in the travel direction -> SAFE['v'] (speed actually sent to the wheel)."""
    dt = CTRL_EVERY * PHYSICS_DT
    v = cmd["v"]
    if args.no_safety:
        SAFE["v"] = v
        return
    b, fwd, _ = body_pose()
    if args.no_scanners:
        pts, occl = np.zeros((0, 3)), {}
        SCANRAYS["rays"] = np.zeros((0, 6))
    else:
        pts, occl = scanner_points(b, fwd)
    if LIDAR is not None:
        lp = lidar_points(b, fwd)
        pts = np.concatenate([pts, lp]) if len(lp) else pts
        occl = {**occl, "mid360": LIDAR_STAT["self"]}
    SAFE["scan"], SAFE["occl"] = pts, occl
    map_update(b, fwd)
    xa, ya = pts[:, 0], pts[:, 1]                          # forward +, left +
    hit = np.ones(len(pts), dtype=bool)
    if SAFE.get("spin") and abs(v) > 1e-3:
        # In-place spin: stop if any object lies inside the chassis + load rectangle (+0.1 m) after a further 0.05~0.25 rad of rotation about the front rollers (-FIX_X forward)
        rx = xa - (-FIX_X)
        near = np.zeros(len(pts), bool)
        for a_ in (0.05, 0.1, 0.15, 0.2, 0.25):
            c_, s_ = math.cos(-SAFE["spin"] * a_), math.sin(-SAFE["spin"] * a_)
            qa, ql = c_ * rx - s_ * ya + (-FIX_X), s_ * rx + c_ * ya
            near |= (qa <= FOOT_FRONT + 0.1) & (qa >= -FOOT_REAR - 0.1) & (np.abs(ql) <= FOOT_HALF_W + 0.1)
        near &= pts[:, 2] < 0.5
        d_sp = np.hypot(rx, ya)
        mode = "stop" if near.any() else "go"
        if SAFE["state"] == "stop" and mode != "stop":
            SAFE["clear_t"] += dt
            if SAFE["clear_t"] < CLEAR_HOLD:
                mode = "stop"
        else:
            SAFE["clear_t"] = 0.0
        if mode == "stop" and SAFE["state"] != "stop":
            SAFE["stops"] += 1
            log(f"[safety] stop: object in the in-place spin sweep ({float(np.min(d_sp[near])):.2f} m from the front rollers)")
            SAFE["snap"] = f"stop_{SAFE['stops']}"
        if mode == "stop":
            SAFE["stop_time"] += dt
            SAFE["stop_run"] += dt
        else:
            SAFE["stop_run"] = 0.0
        SAFE["state"] = mode
        SAFE["v"] = 0.0 if mode == "stop" else v
        return
    # Minimum obstacle-to-footprint distance (for logging; target pallet excluded)
    dx = np.maximum(0.0, np.maximum(xa - FOOT_FRONT, -FOOT_REAR - xa))
    dy = np.maximum(0.0, np.abs(ya) - FOOT_HALF_W)
    other = pts[:, 2] < 0.5
    if other.any():
        SAFE["min_clear"] = min(SAFE["min_clear"], float(np.min(np.hypot(dx, dy)[other])))
    mode = "go"
    if abs(v) > 1e-3:
        tip_along = float((b + (-TIP_X) * fwd - np.array([PC0[0], PC0[1]])) @ D)
        docking = v > 0 and float(art.get_joint_positions()[I_LIFT]) < 0.1 and tip_along >= -PALLET_HALF - 1.5
        vs = min(abs(v), V_SLOW) if SAFE["state"] != "go" else abs(v)
        stop_l = -1.0 if docking else 0.5 + 0.6 * vs          # docking: front field off (forks entering the pallet; field switching as on real scanners)
        slow_l = 0.0 if docking else stop_l + 1.5
        if SAFE.get("drop") and v > 0:                         # unloading (driving the load up to the end wall): front field shortened to 0.1 m
            stop_l, slow_l = 0.1, 0.0
        a = xa - FOOT_FRONT if v > 0 else -FOOT_REAR - xa     # distance ahead in the travel direction (from the chassis end)
        in_w = np.abs(ya) <= FOOT_HALF_W + FIELD_MARGIN
        d_min = float(np.min(a[hit & in_w & (a >= -0.05)], initial=99.0))
        if d_min <= stop_l:
            mode = "stop"
        elif d_min <= slow_l:
            mode = "slow"
    # Stop -> go only after the field has been clear for CLEAR_HOLD
    if SAFE["state"] == "stop" and mode != "stop":
        SAFE["clear_t"] += dt
        if SAFE["clear_t"] < CLEAR_HOLD:
            mode = "stop"
    else:
        SAFE["clear_t"] = 0.0
    if mode == "stop" and SAFE["state"] != "stop":
        SAFE["stops"] += 1
        log(f"[safety] stop: obstacle {'ahead' if v > 0 else 'behind'} in the travel direction (cmd v {v:+.2f})")
        SAFE["stop_dir"] = 1 if v > 0 else -1
        SAFE["snap"] = f"stop_{SAFE['stops']}"
    if mode != "stop" and SAFE["state"] == "stop":
        log(f"[safety] restart (stopped {SAFE['stop_run']:.1f} s)")
    if mode == "stop":
        SAFE["stop_time"] += dt
        SAFE["stop_run"] += dt
    else:
        SAFE["stop_run"] = 0.0
    SAFE["state"] = mode
    SAFE["v"] = 0.0 if mode == "stop" else (math.copysign(min(abs(v), V_SLOW), v) if mode == "slow" else v)


# ---- Obstacle placement and motion ----
OBS = {"kind": "none"}


EDGE_MIX = [(int(k_), float(w_)) for k_, w_ in (t_.split(":") for t_ in args.edge_mix.split(","))]
V4_OBJ_Y = (-1.0, 9.5)                                     # y range for aisle objects (straight part of the aisle)
V4_PROTRUDE = (0.2, 1.3)                                   # how far objects protrude from the aisle edge [m] (at 1.3 the remaining 2.2 m > 1.9 m needed to pass)


def obstacle_reset(ep, s):
    """0-2 aisle-edge objects (boxes/drums) plus, with some probability, a person crossing the aisle. Uses an RNG separate from pallet placement."""
    SAFE.update(v=0.0, state="go", clear_t=0.0, stops=0, stop_time=0.0, stop_run=0.0, min_clear=99.0, spin=0.0, drop=False)
    RET["on"] = False
    NAV.clear()
    NAV.update(grid=new_map(), mode="nominal", plans=0, detours=0, fails=0, chk=-9.0, no_plan_until=-9.0, last_stuck=-9.0, t0=sim_t["t"],
               frames=[], last_frame=-9.0)
    for k in range(N_BOX):
        set_obs_pose(k, (OBS_AWAY[0] + 2 * k, OBS_AWAY[1], BOX_SIZE[2] / 2), (1.0, 0, 0, 0))
    set_obs_pose("person", (OBS_AWAY[0], OBS_AWAY[1] + 3, 0.0), (1.0, 0, 0, 0))
    OBS.clear()
    r = random.Random(args.seed * 7919 + ep)
    x_, acc_, n = r.random() * sum(w_ for _, w_ in EDGE_MIX), 0.0, 0
    for n, w_ in EDGE_MIX:
        acc_ += w_
        if x_ < acc_:
            break
    person = r.random() < args.person_prob
    kind = ("mixed" if person else "box") if n else ("person" if person else "none")
    OBS.update(kind=kind, boxes=[])
    out = dict(obstacle=kind, n_boxes=n)
    free = [0, 1, 2, 3]
    for k in range(n):
        idx = free.pop(r.randrange(len(free)))
        drum = idx in V4_DRUM
        for _ in range(100):
            y = r.uniform(*V4_OBJ_Y)
            if all(abs(y - q["pos"][1]) >= 3.5 for q in OBS["boxes"]):
                break
        side = r.choice([-1.0, 1.0])                       # -1 = west edge, +1 = east edge
        prot = r.uniform(*V4_PROTRUDE)
        yaw = 0.0 if drum else r.uniform(-0.2, 0.2)
        hx = 0.3 if drum else BOX_SIZES[idx][0] / 2
        x = (V4_X0 + prot - hx) if side < 0 else (V4_X1 - prot + hx)
        set_obs_pose(idx, (x, y, BOX_SIZES[idx][2] / 2), yaw_quat(yaw))
        OBS["boxes"].append(dict(pos=np.array([x, y]), size=BOX_SIZES[idx], yaw=yaw, idx=idx))
        log(f"[obstacle] aisle object {k}: {'drum' if drum else 'box'} y {y:.2f}, {'west' if side < 0 else 'east'} edge, "
            f"protrudes {prot:.2f} m (center x {x:.2f})")
        if k == 0:
            out.update(obs_u=y, obs_lat=side * prot)
    if OBS["boxes"]:
        OBS.update(pos=OBS["boxes"][0]["pos"], cleared=False)
    if person:
        yc = r.choice(V4_GAPS)
        side = r.choice([-1.0, 1.0])
        speed = r.uniform(0.8, 1.2)
        trig = r.uniform(1.5, 3.5)                         # start walking when the forklift rear is this far before the crossing line
        p0 = np.array([V4_CX + side * 3.2, yc])
        trip = r.choice(["up", "down"])                    # while carrying up / while returning empty
        OBS.update(p0=p0, side=side, speed=speed, trig=trig, t_start=None, walked=0.0, yc=yc, n_dir=np.array([1.0, 0.0]), trip=trip)
        _place_person(p0, np.array([-side, 0.0]))
        log(f"[obstacle] person ({'going up' if trip == 'up' else 'returning'}): crosses at gap y {yc:.1f} {'east->west' if side > 0 else 'west->east'}, speed {speed:.2f} m/s, "
            f"starts when the forklift rear is {trig:.1f} m before the line")
    look = OBS["boxes"][0]["pos"] if OBS["boxes"] else (np.array([V4_CX, OBS["yc"]]) if person else np.array([V4_CX, 4.0]))
    eye = look + np.array([6.5, -4.0])
    obs_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(eye[0], eye[1], 5.0), Gf.Vec3d(look[0], look[1], 0.5),
                                       Gf.Vec3d(0, 0, 1)).GetInverse())
    if kind == "none":
        log("[obstacle] none")
    return out


def feasible_check():
    """For evaluation only: on a map drawn from the true box positions, check whether a path exists from the docking pose to the drop target line (not used for driving)."""
    g = new_map()
    g.lo[:] = PL.L_MIN
    g.seen[:] = True
    import cv2 as _cv
    for bx in OBS["boxes"]:
        c_, s_ = math.cos(bx["yaw"]), math.sin(bx["yaw"])
        hx, hy = bx["size"][0] / 2, bx["size"][1] / 2
        cs = [bx["pos"] + np.array([c_ * a - s_ * b, s_ * a + c_ * b]) for a, b in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))]
        pts = np.stack(g.cell(np.array(cs)), axis=-1)[:, ::-1].astype(np.int32)
        m = np.zeros(g.lo.shape, np.uint8)
        _cv.fillPoly(m, [pts], 1)
        g.lo[m.astype(bool)] = PL.L_MAX
    c = np.array([PC0[0], PC0[1]])
    b_dock = c - PAL_FROM_BODY * D
    f0 = b_dock + (-FIX_X) * D
    fG = GOAL + (-FIX_X) * GOAL_D
    pl = PL.HybridAStar(g, FOOT_F, KAPPA_PLAN, max_iter=100000)
    path, it = pl.plan((f0[0], f0[1], math.atan2(D[1], D[0])), goal_region, tuple(fG + 2.3 * GOAL_D))
    # 1 = path exists, 0 = search exhausted without a path, -1 = unknown (hit the 100k-iteration search limit)
    return 1 if path is not None else (-1 if it >= pl.max_iter else 0)


def _place_person(p, heading):
    yaw = math.atan2(heading[1], heading[0])
    set_obs_pose("person", (p[0], p[1], 0.0), yaw_quat(yaw))
    OBS["pos"] = p


def obstacle_step():
    """Every 0.1 s: walk the person (starts when the forklift rear is trig m before the crossing line) / remove boxes."""
    dt = CTRL_EVERY * PHYSICS_DT
    k = OBS.get("kind")
    if k in ("box", "mixed") and not OBS["cleared"] and args.box_clear > 0 and SAFE["stop_run"] >= args.box_clear:
        for j in range(N_BOX):
            set_obs_pose(j, (OBS_AWAY[0] + 2 * j, OBS_AWAY[1], BOX_SIZE[2] / 2), (1.0, 0, 0, 0))
        OBS["cleared"] = True
        log(f"[obstacle] boxes removed (after a {SAFE['stop_run']:.1f} s stop)")
    if "p0" in OBS:
        if OBS["t_start"] is None:
            b, fwd, _ = body_pose()
            lifted = float(art.get_joint_positions()[I_LIFT]) > 0.1
            rear = b + REAR_X * (-fwd)
            in_corr = V4_X0 - 0.5 <= rear[0] <= V4_X1 + 0.5
            ahead = (OBS["yc"] - rear[1]) if OBS["trip"] == "up" else (rear[1] - OBS["yc"])
            go = (lifted and not RET["on"]) if OBS["trip"] == "up" else RET["on"]
            if go and in_corr and -0.5 <= ahead <= OBS["trig"]:
                OBS["t_start"] = sim_t["t"]
                log(f"[obstacle] person starts walking (forklift rear to crossing line {ahead:.2f} m)")
        elif OBS["walked"] < 6.4:
            OBS["walked"] = min(6.4, OBS["walked"] + OBS["speed"] * dt)
            _place_person(OBS["p0"] - OBS["side"] * OBS["walked"] * OBS["n_dir"], -OBS["side"] * OBS["n_dir"])


def physics(n):
    if not timeline.is_playing():                  # the timeline is paused after a camera capture (rep.orchestrator.step)
        timeline.play()
        timeline.commit()
    for _ in range(n):
        if sim_t["n"] % CTRL_EVERY == 0:
            obstacle_step()
            safety_update()
        apply_cmd()
        sim_t["n"] += 1
        sim.step(render=(not args.record) and sim_t["n"] % RENDER_EVERY == 0)   # when recording, the capture does the rendering
        sim_t["t"] += PHYSICS_DT


def body_pose():
    pos, q = body_rp.get_world_pose()          # q = (w, x, y, z)
    w, x, y, z = map(float, q)
    # world direction of the body-local x axis; the forward (fork) direction is its opposite
    xa = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
    fwd = -xa / np.linalg.norm(xa)
    return np.array([float(pos[0]), float(pos[1])]), fwd, float(pos[2])


def pallet_pose():
    pos, q = pal_rp.get_world_pose()
    w, x, y, z = map(float, q)
    zz = 1 - 2 * (x * x + y * y)                   # world z component of the local z axis -> tilt
    return np.array([float(pos[0]), float(pos[1]), float(pos[2])]), math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def state():
    """Errors relative to the pallet axis. along: coordinate along the approach direction (pallet center 0, face -0.5); lat: left +."""
    b, fwd, _ = body_pose()
    pp, tilt = pallet_pose()
    c = pp[:2]
    f_pt = b + (-FIX_X) * fwd                     # fixed-axle point (-FIX_X forward of the body origin)
    tip = b + (-TIP_X) * fwd
    yaw_err = math.atan2(fwd[0] * D[1] - fwd[1] * D[0], fwd @ D) * -1   # relative to the approach direction, left +
    return dict(b=b, fwd=fwd, f=f_pt, tip=tip,
                f_along=float((f_pt - c) @ D), f_lat=float((f_pt - c) @ N),
                tip_along=float((tip - c) @ D), tip_lat=float((tip - c) @ N),
                leg_along=float((b + (-LEG_X) * fwd - c) @ D),
                yaw_err=yaw_err, pallet=pp, tilt=tilt, lift=float(art.get_joint_positions()[I_LIFT]))


_DIAG = {}


def diag_log(ep, tag):
    """Diagnostics: log lift (fork) link, chassis and pallet poses relative to the chassis, to see what drifts across episodes."""
    if "lift" not in _DIAG:
        _DIAG["lift"] = SingleRigidPrim(f"{FORK}/lift")
        _DIAG["lift"].initialize()
    from scipy.spatial.transform import Rotation as _R
    bp, bq = body_rp.get_world_pose()
    lp, lq = _DIAG["lift"].get_world_pose()
    pp, pq = pal_rp.get_world_pose()
    Rb = _R.from_quat([bq[1], bq[2], bq[3], bq[0]])
    rel = Rb.inv().apply(np.asarray(lp) - np.asarray(bp))
    rrel = (Rb.inv() * _R.from_quat([lq[1], lq[2], lq[3], lq[0]])).as_euler("xyz", degrees=True)
    beul = Rb.as_euler("xyz", degrees=True)
    peul = _R.from_quat([pq[1], pq[2], pq[3], pq[0]]).as_euler("xyz", degrees=True)
    jp = art.get_joint_positions()
    log(f"[diag] [ep {ep}] {tag}: lift link (chassis frame) xyz=({rel[0]:+.4f},{rel[1]:+.4f},{rel[2]:+.4f}) rpy=({rrel[0]:+.2f},{rrel[1]:+.2f},{rrel[2]:+.2f}) "
        f"| chassis z={float(bp[2]):.4f} roll/pitch=({beul[0]:+.2f},{beul[1]:+.2f}) | pallet z={float(pp[2]):.4f} roll/pitch=({peul[0]:+.2f},{peul[1]:+.2f}) "
        f"| joints {dict(zip(DOF, [round(float(v), 4) for v in jp]))}")


def snap(dirname):
    rep.orchestrator.step(rt_subframes=2, delta_time=0.0)
    os.makedirs(dirname, exist_ok=True)
    for name, a in annots.items():
        im = np.asarray(a.get_data())
        if im.ndim == 3 and im.size:
            Image.fromarray(np.ascontiguousarray(im[..., :3])).save(os.path.join(dirname, f"{name}.png"))


def teleport(tip_dist, tip_lat, yaw_deg):
    """Place the forklift so the fork-tip center is tip_dist in front of the face, offset tip_lat sideways, with heading error yaw_deg."""
    c = np.array([PC0[0], PC0[1]])
    h = math.radians(yaw_deg)
    fwd = math.cos(h) * D + math.sin(h) * N
    tip = c + (-PALLET_HALF - tip_dist) * D + tip_lat * N
    b = tip - (-TIP_X) * fwd
    byaw = math.atan2(-fwd[1], -fwd[0])            # body +x direction
    _, _, bz = body_pose()
    art.set_world_pose(np.array([b[0], b[1], bz]), np.array([math.cos(byaw / 2), 0, 0, math.sin(byaw / 2)]))
    art.set_joint_positions(np.zeros(len(DOF)))
    art.set_joint_velocities(np.zeros(len(DOF)))
    art.set_linear_velocity(np.zeros(3))
    art.set_angular_velocity(np.zeros(3))
    cmd.update(v=0.0, steer=0.0, lift=0.0)


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def place_pallets(rng):
    """Place 1-3 pallets in the three staging-row slots; the target (args.pallet) takes a random one. Unused pallets go outside the warehouse. Returns the target (x, y, yaw)."""
    z0 = float(np.asarray(PAL_POSE0[0])[2])
    if args.rand:
        n = rng.choices([1, 2, 3], weights=[0.2, 0.3, 0.5])[0]
        slots = sorted(rng.sample(range(3), n))
        tgt = rng.choice(slots)
    else:
        slots, tgt = [0, 1, 2], 1
    others = [k for k in slots if k != tgt]

    def jit(a):
        return rng.uniform(-a, a) if args.rand else 0.0
    x, y = V4_SLOTS_X[tgt] + jit(0.15), V4_ROW_Y + jit(0.15)
    yaw = pyaw + math.radians(jit(args.pallet_yaw))
    poses = [(pal_rp, (x, y, yaw))]
    for j_, (r_, _) in enumerate(OTHERS):
        if j_ < len(others):
            k = others[j_]
            poses.append((r_, (V4_SLOTS_X[k] + jit(0.15), V4_ROW_Y + jit(0.15), pyaw + math.radians(jit(5.0)))))
        else:
            poses.append((r_, (40.0 + 3.0 * j_, 40.0, 0.0)))
    for rp_, (px, py, pw) in poses:
        rp_.set_world_pose(np.array([px, py, z0]), yaw_quat(pw))
        rp_.set_linear_velocity(np.zeros(3))
        rp_.set_angular_velocity(np.zeros(3))
    set_frame(x, y, yaw)
    place_observer()
    LAYOUT.update(n_pallets=len(slots), slot=tgt, others_x=[q[1][0] for q in poses[1:1 + len(others)]])
    return x, y, yaw


LAYOUT = {}
RET = {"on": False}                                      # returning to the staging area after unloading


def set_lights(scale):
    for attr, base in LIGHTS:
        attr.Set(float(base) * scale)


def start_ok(tip_dist, tip_lat, yaw_deg):
    """Whether the forklift footprint (fork tip to 1.0 m behind it, width ±0.65 m) lies inside the warehouse."""
    c = np.array([PC0[0], PC0[1]])
    h = math.radians(yaw_deg)
    fwd = math.cos(h) * D + math.sin(h) * N
    left = np.array([-fwd[1], fwd[0]])
    tip = c + (-PALLET_HALF - tip_dist) * D + tip_lat * N
    for a_ in (0.0, -(-TIP_X) - 1.0):
        for l_ in (-0.65, 0.65):
            q = tip + a_ * fwd + l_ * left
            if not (BOUNDS[0][0] <= q[0] <= BOUNDS[0][1] and BOUNDS[1][0] <= q[1] <= BOUNDS[1][1]):
                return False
    return True


# ---- Recording ----
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
POOL = ThreadPoolExecutor(max_workers=4)
REC = {"on": False}
class LocNoise:
    """SLAM-like localization error, one step per 0.1 s: a slowly varying bias (AR(1), steady-state std 3 cm position, 0.8° heading)
    + per-step jitter (1 cm, 0.3°) + occasional jumps (0.3 %/step, 8 cm, 1.5°, decaying slowly). Magnitudes are typical of Mid-360 + FAST-LIO2-class SLAM (not measured; recalibrate against real SLAM error)"""
    A_, B_POS, B_YAW = 0.995, 0.03, math.radians(0.8)
    W_POS, W_YAW = 0.01, math.radians(0.3)
    P_JUMP, J_POS, J_YAW = 0.003, 0.08, math.radians(1.5)

    def reset(self, rng):
        self.rng = rng
        self.bias = np.array([rng.gauss(0, self.B_POS), rng.gauss(0, self.B_POS), rng.gauss(0, self.B_YAW)])
        self.t = -1.0
        self.cur = np.zeros(3)

    def get(self):
        if sim_t["t"] - self.t >= 0.099:                  # advance only once per 0.1 s
            self.t = sim_t["t"]
            k = math.sqrt(1 - self.A_ ** 2)
            self.bias = self.A_ * self.bias + k * np.array([self.rng.gauss(0, self.B_POS), self.rng.gauss(0, self.B_POS),
                                                             self.rng.gauss(0, self.B_YAW)])
            if self.rng.random() < self.P_JUMP:
                self.bias += np.array([self.rng.gauss(0, self.J_POS), self.rng.gauss(0, self.J_POS), self.rng.gauss(0, self.J_YAW)])
            self.cur = self.bias + np.array([self.rng.gauss(0, self.W_POS), self.rng.gauss(0, self.W_POS), self.rng.gauss(0, self.W_YAW)])
        return self.cur


LOCN = LocNoise()
LOCN.reset(random.Random(0))


def goal_rel_in(s):
    """Goal state for the policy input: with --loc-noise, relative to the 'estimated pose' (true pose + error)."""
    if not args.loc_noise:
        return goal_rel(s)
    e = LOCN.get()
    c, sn = math.cos(e[2]), math.sin(e[2])
    fw = np.array([c * s["fwd"][0] - sn * s["fwd"][1], sn * s["fwd"][0] + c * s["fwd"][1]])
    return goal_rel({"b": s["b"] + e[:2], "fwd": fw})


DATA_COLS = ["frame_index", "timestamp", "phase", "state_lift", "state_steer", "state_speed",
             "state_goal_fwd", "state_goal_left", "state_goal_yaw",
             "action_speed", "action_steer", "action_lift", "priv_tip_along", "priv_tip_lat", "priv_yaw_err",
             "action_speed_exec", "safety", "nav_mode",
             "priv_goal_fwd", "priv_goal_left", "priv_goal_yaw", "pose_x", "pose_y", "pose_yaw", "pert_steer"]


def _save_jpg(im, path):
    Image.fromarray(im).save(path, quality=92)


def rec_start(ed):
    if not args.record:
        return
    for k in REC_CAMS:
        os.makedirs(os.path.join(ed, "frames", k), exist_ok=True)
    if args.save_lidar:
        os.makedirs(os.path.join(ed, "lidar"), exist_ok=True)
    REC.update(on=True, dir=ed, i=0, prev=None, rows=[])


def rec_frame(s, phase):
    """Save one frame: observations (cameras, state) and the action (cmd) applied in this control period."""
    if not REC.get("on"):
        return
    rep.orchestrator.step(rt_subframes=2, delta_time=0.0)
    i = REC["i"]
    for k, an in REC_CAMS.items():
        im = np.ascontiguousarray(np.asarray(annots[an].get_data())[..., :3])
        POOL.submit(_save_jpg, im, os.path.join(REC["dir"], "frames", k, f"{i:05d}.jpg"))
    dt = CTRL_EVERY * PHYSICS_DT
    v_meas = 0.0 if REC["prev"] is None else float((s["b"] - REC["prev"]) @ s["fwd"]) / dt
    steer_meas = float(art.get_joint_positions()[I_STEER])
    g = goal_rel_in(s)
    g0 = goal_rel(s)
    if args.save_lidar:
        POOL.submit(np.save, os.path.join(REC["dir"], "lidar", f"{i:05d}.npy"), LIDAR_STAT.get("Pb", np.zeros((0, 3), np.float16)))
    REC["rows"].append([i, f"{i * dt:.2f}", phase, f"{s['lift']:.5f}", f"{steer_meas:.5f}", f"{v_meas:.5f}",
                        f"{g[0]:.4f}", f"{g[1]:.4f}", f"{g[2]:.5f}",
                        f"{cmd['v']:.5f}", f"{cmd['steer']:.5f}", f"{cmd['lift']:.5f}",
                        f"{s['tip_along']:.4f}", f"{s['tip_lat']:.4f}", f"{s['yaw_err']:.5f}",
                        f"{SAFE['v']:.5f}", SAFE["state"], NAV.get("mode", "") if phase.startswith(("carry", "return")) else "",
                        f"{g0[0]:.4f}", f"{g0[1]:.4f}", f"{g0[2]:.5f}", f"{s['b'][0]:.4f}", f"{s['b'][1]:.4f}",
                        f"{math.atan2(s['fwd'][1], s['fwd'][0]):.5f}", f"{PERT['steer']:.5f}"])
    REC["prev"] = s["b"].copy()
    REC["i"] += 1


def rec_end():
    if not REC.get("on"):
        return 0
    with open(os.path.join(REC["dir"], "data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DATA_COLS)
        w.writerows(REC["rows"])
    REC["on"] = False
    return REC["i"]


TRACE_COLS = ["ep", "t", "phase", "x", "y", "yaw_deg", "f_along", "f_lat", "tip_along", "tip_lat", "yaw_err_deg",
              "v_cmd", "steer_cmd_deg", "lift_cmd", "lift", "pallet_x", "pallet_y", "pallet_z", "pallet_tilt",
              "v_safe", "safety", "obs_x", "obs_y"]
trace_f = open(os.path.join(RUN_DIR, "trace.csv"), "w", newline="")
trace = csv.writer(trace_f)
trace.writerow(TRACE_COLS)


def record(ep, phase, s):
    trace.writerow([ep, f"{sim_t['t']:.2f}", phase, f"{s['b'][0]:.4f}", f"{s['b'][1]:.4f}",
                    f"{math.degrees(math.atan2(s['fwd'][1], s['fwd'][0])):.2f}", f"{s['f_along']:.4f}",
                    f"{s['f_lat']:.4f}", f"{s['tip_along']:.4f}", f"{s['tip_lat']:.4f}",
                    f"{math.degrees(s['yaw_err']):.3f}", f"{cmd['v']:.3f}", f"{math.degrees(cmd['steer']):.2f}",
                    f"{cmd['lift']:.3f}", f"{s['lift']:.4f}", f"{s['pallet'][0]:.4f}", f"{s['pallet'][1]:.4f}",
                    f"{s['pallet'][2]:.4f}", f"{s['tilt']:.2f}", f"{SAFE['v']:.3f}", SAFE["state"],
                    *((f"{OBS['pos'][0]:.3f}", f"{OBS['pos'][1]:.3f}") if "pos" in OBS else ("", ""))])


def kappa_to_steer(kappa, limit=STEER_MAX):
    return max(-limit, min(limit, math.atan(kappa * STEER_L)))


def line_steer(s, limit=STEER_MAX, c=None):
    """Steering [rad] that drives the fixed-axle point onto the pallet axis (approach line), for forward driving.
    c: fixed reference pallet center; if None, the current pallet position."""
    f_lat = s["f_lat"] if c is None else float((s["f"] - c) @ N)
    return kappa_to_steer(-(KP * f_lat + KD * s["yaw_err"]), limit)


def reverse_steer(s, v):
    """Reverse steering that reduces heading error: dψ/dt = v·κ = -1.0·ψ."""
    return kappa_to_steer(-1.0 * s["yaw_err"] / min(v, -0.05))


# ---- Carry: relative to target pose G ----
CARRY_A = STEER_L                  # fixed axle to steered wheel distance (the steered wheel leads when reversing)
CARRY_K = 1.0                      # convergence rate of the steered-wheel point's lateral error when reversing [1/m]
V_CARRY = 0.4
CARRY_TOL_LAT, CARRY_TOL_YAW = 0.10, math.radians(2.0)
CARRY_RETRY = 2
PULLOUT = 1.3                      # reverse distance to pull out the forks (forks are ~1.04 m inside the pallet)


def goal_err(s):
    """Relative to the target line (through G along GOAL_D): along (+ = past G in the forward direction), fixed-axle lateral error, heading error (left +)."""
    psi = math.atan2(GOAL_D[0] * s["fwd"][1] - GOAL_D[1] * s["fwd"][0], GOAL_D @ s["fwd"])
    return float((s["b"] - GOAL) @ GOAL_D), float((s["f"] - GOAL) @ GOAL_N), psi


def goal_rel(s):
    """For the state: target pose G in the forklift frame (forward +, left +, heading difference left +)."""
    d = GOAL - s["b"]
    left = np.array([-s["fwd"][1], s["fwd"][0]])
    psi = math.atan2(s["fwd"][0] * GOAL_D[1] - s["fwd"][1] * GOAL_D[0], s["fwd"] @ GOAL_D)
    return float(d @ s["fwd"]), float(d @ left), psi


def carry_reverse_steer(s):
    """Reversing: drive the lateral error e_P of the steered-wheel point P (CARRY_A behind the fixed axle) to zero via de_P/dσ = -k·e_P (σ = reverse distance).
    de_P/dσ = -sinψ + a·cosψ·κ  ->  κ = (sinψ - k·e_P) / (a·cosψ)"""
    _, e_f, psi = goal_err(s)
    e_p = e_f - CARRY_A * math.sin(psi)
    return kappa_to_steer((math.sin(psi) - CARRY_K * e_p) / (CARRY_A * max(0.3, math.cos(psi))))


def carry_forward_steer(s):
    """Forward driving (unloading approach, realignment): fixed-axle point onto the target line (same law as the pallet approach)."""
    _, e_f, psi = goal_err(s)
    return kappa_to_steer(-(KP * e_f + KD * psi))


# ---- Steering calibration mode ----
if args.calib:
    out = []
    for deg in (15.0, -15.0, 30.0):
        teleport(4.0, 0.0, 0.0)
        cmd.update(lift=0.0)
        physics(60)
        b0 = body_pose()
        cmd.update(v=0.4, steer=math.radians(deg))
        physics(120)                          # until steady state
        samples = []
        for _ in range(20):
            b1, f1, _ = body_pose()
            physics(CTRL_EVERY)
            b2, f2, _ = body_pose()
            dt = CTRL_EVERY * PHYSICS_DT
            vel = (b2 - b1) / dt
            w = math.atan2(f1[0] * f2[1] - f1[1] * f2[0], f1 @ f2) / dt      # left turn +
            fwd = (f1 + f2) / np.linalg.norm(f1 + f2)
            left = np.array([-fwd[1], fwd[0]])
            v_f, v_l = vel @ fwd, vel @ left
            # lateral velocity of a point s ahead of the body origin on the forward axis = v_l + w*s -> find s where it is 0
            samples.append((v_f, v_l, w, -v_l / w if abs(w) > 1e-4 else float("nan")))
        cmd.update(v=0.0, steer=0.0)
        physics(30)
        a = np.array(samples)
        v_f, v_l, w, s0 = a.mean(axis=0)
        kappa = w / v_f
        L = math.tan(math.radians(deg)) / kappa
        log(f"[calib] steer {deg:+.0f}°: forward speed {v_f:.3f} m/s (cmd 0.4), yaw rate {math.degrees(w):+.2f}°/s, "
            f"curvature {kappa:+.3f} 1/m, zero-lateral-slip point = {s0:+.3f} m forward of the body origin (FIX_X={-s0:+.3f}), "
            f"STEER_L = {L:+.3f}")
        out.append((deg, v_f, w, kappa, s0, L))
    with open(os.path.join(RUN_DIR, "calib.csv"), "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["steer_deg", "v_fwd", "yaw_rate", "kappa", "zero_slip_s", "steer_L"])
        w_.writerows(out)
    app.close()
    sys.exit(0)


# ---- Episodes ----
CUR = {"ed": None}


def _stop_snap():
    """Snapshot right after a safety stop (ep_NNN/stop_K/)."""
    k = SAFE.pop("snap", None)
    if k and CUR["ed"]:
        snap(os.path.join(CUR["ed"], k))


TOPGIF = {"frames": [], "t": -9.0}


def top_frame(phase):
    """--top-gif: top-camera frame with phase name and time overlay."""
    if args.top_gif <= 0 or sim_t["t"] - TOPGIF["t"] < args.top_gif:
        return
    TOPGIF["t"] = sim_t["t"]
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
    im = np.ascontiguousarray(np.asarray(annots["top"].get_data())[..., :3])
    if im.ndim != 3 or not im.size:
        return
    im = cv2.resize(im, (im.shape[1] * 360 // im.shape[0], 360))
    txt = f"t {sim_t['t'] - TOPGIF.get('t0', 0.0):5.1f}s  {phase}  {SAFE['state']}"
    cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    TOPGIF["frames"].append(im)


def top_gif_save(ed):
    fr = TOPGIF["frames"]
    if fr:
        ims = [Image.fromarray(f) for f in fr]
        ims[0].save(os.path.join(ed, "top.gif"), save_all=True, append_images=ims[1:], duration=120, loop=0)
    TOPGIF.update(frames=[], t=-9.0, t0=sim_t["t"])


def unload_unrecorded(ep):
    """If an episode ends with the pallet still lifted (dock-only recording, carry_blocked), set it down and pull the forks out after recording.
    Teleporting for the next episode while holding the pallet made later insertions push the pallet ~3 cm sideways,
    persisting for the rest of the process."""
    cmd.update(v=0.0, steer=0.0)
    physics(60)
    cmd["lift"] = LIFT_LOW
    for _ in range(50):
        physics(CTRL_EVERY)
        if state()["lift"] <= LIFT_LOW + 0.01:
            break
    cmd.update(v=-V_REVERSE, steer=0.0)
    for _ in range(150):
        physics(CTRL_EVERY)
        if state()["tip_along"] <= -PALLET_HALF - 0.3:
            break
    cmd.update(v=0.0)
    physics(30)
    s = state()
    log(f"[ep {ep}] (unrecorded) pallet set down and forks pulled out: lift {s['lift']:+.3f}, fork tip to face {-PALLET_HALF - s['tip_along']:.2f} m")


PERT_PHASES = ("approach", "carry", "return")
PERT_STEER_LIM = math.radians(60.0)


def pert_update(phase, s):
    """Every 0.1 s: advance or start a perturbation. Carry/return: probability --pert-rate each second; docking approach: at times drawn per episode.
    No perturbation during a safety stop, an in-place spin, or while waiting for a plan (steering while stopped has no effect)."""
    t = sim_t["t"]
    dt = CTRL_EVERY * PHYSICS_DT
    ok = phase in PERT_PHASES and SAFE["state"] != "stop" and not SAFE.get("spin") and NAV.get("mode") != "wait"
    if PERT["steer"]:
        if t >= PERT["until"] or not ok:
            PERT["steer"] = 0.0
        else:
            PERT["time"] += dt
        return
    if not ok:
        return
    rng = PERT["rng"]
    if phase == "approach":
        if PERT["dock_t"] is not None and t >= PERT["dock_t"]:
            PERT["dock_t"] = None
            mag, dur = math.radians(rng.uniform(*args.pert_dock_steer)), rng.uniform(*args.pert_dock_dur)
            if PERT["dock_left"] > 0:                      # next perturbation 2-5 s after this one ends
                PERT["dock_left"] -= 1
                PERT["dock_t"] = t + dur + rng.uniform(2.0, 5.0)
        else:
            return
    else:
        if sim_t["n"] % (10 * CTRL_EVERY) != 0 or rng.random() >= args.pert_rate:
            return
        mag, dur = math.radians(rng.uniform(*args.pert_steer)), rng.uniform(*args.pert_dur)
    d = rng.choice((-1.0, 1.0)) * mag
    d = max(-PERT_STEER_LIM, min(PERT_STEER_LIM, cmd["steer"] + d)) - cmd["steer"]
    PERT.update(steer=d, until=t + dur, n=PERT["n"] + 1)
    log(f"[perturb] {phase}: adding {math.degrees(d):+.0f}° steering for {dur:.1f} s (#{PERT['n']})")


def run_phase(ep, phase, until, timeout, control=None):
    """Update commands with control(s); finish when until(s) is truthy. Returns (result, state)."""
    t0 = sim_t["t"]
    while sim_t["t"] - t0 < timeout:
        top_frame(phase)
        s = state()
        record(ep, phase, s)
        r = until(s)
        if r:
            return r, s
        if control:
            control(s)
        if args.perturb:
            pert_update(phase, s)
        rec_frame(s, phase)
        physics(CTRL_EVERY)
        _stop_snap()
        if not app.is_running():
            return "closed", s
    return "timeout", state()


V4_ROUTE_R = 2.0                                         # route corner radius (reference point f; curvature 0.5 < planner limit 0.54)
SPIN_BACK = 2.0                                          # before spinning, back the front rollers this far from the original pallet face


def spin_out(ep, c0):
    """After lifting: back out of the pallet row, then spin in place (steer ±90°) so the forks face west (rear toward the aisle entrance).
    c0 = pallet center before lifting (fixed reference)."""
    r, s = run_phase(ep, "backout", lambda s: "ok" if float((s["f"] - c0) @ D) <= -PALLET_HALF - SPIN_BACK else None, 25.0,
                     lambda s: cmd.update(v=-V_REVERSE, steer=reverse_steer(s, -V_REVERSE)))
    cmd.update(v=0.0)
    if r != "ok":
        return f"backout_{r}"
    return spin_to(ep, math.pi)                          # forks west (rear east, toward the aisle entrance)


def spin_to(ep, th_t, sgn=None, tag="spin"):
    """Spin in place about the front rollers (steer ±90°) until the fork heading is th_t. sgn: +1 CCW / -1 CW / None = shorter way.
    +90° steering + forward drive = counter-clockwise rotation."""
    def th(s):
        return math.atan2(s["fwd"][1], s["fwd"][0])
    if sgn is None:
        sgn = 1.0 if PL.wrap(th_t - th(state())) > 0 else -1.0

    def rem(s):                                          # remaining rotation [rad], 0~2π (jumps near 2π on overshoot)
        return (sgn * (th_t - th(s))) % (2 * math.pi)
    cmd.update(v=0.0, steer=sgn * SPIN_STEER)
    run_phase(ep, f"{tag}_steer", lambda s: "ok" if abs(float(art.get_joint_positions()[I_STEER]) - sgn * SPIN_STEER) < math.radians(3) else None, 3.0)
    SAFE["spin"] = sgn
    r, s = run_phase(ep, tag, lambda s: "ok" if rem(s) <= math.radians(0.8) or rem(s) >= 2 * math.pi - 0.3 else None, 90.0,
                     lambda s: cmd.update(v=max(0.06, min(0.3, 0.6 * rem(s))), steer=sgn * SPIN_STEER))
    SAFE["spin"] = 0.0
    cmd.update(v=0.0, steer=0.0)
    run_phase(ep, f"{tag}_unsteer", lambda s: "ok" if abs(float(art.get_joint_positions()[I_STEER])) < math.radians(3) else None, 3.0)
    log(f"[ep {ep}] in-place spin ({tag}) {r}: heading error {math.degrees(PL.wrap(th_t - th(state()))):+.2f}°")
    return "ok" if r == "ok" else f"{tag}_{r}"


def _round_polyline(P, R_):
    """Round polyline corners with radius-R_ arcs -> points (x, y) at 0.15 m spacing."""
    P = [np.asarray(q, float) for q in P]
    pts = [P[0]]
    for k in range(1, len(P) - 1):
        a, b, c = P[k - 1], P[k], P[k + 1]
        u1, u2 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        turn = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2)
        T = R_ * math.tan(abs(turn) / 2)
        p1 = b - T * u1
        pts += list(np.linspace(pts[-1], p1, max(2, int(np.linalg.norm(p1 - pts[-1]) / 0.15) + 1))[1:])
        nrm = np.array([-u1[1], u1[0]]) * (1 if turn > 0 else -1)
        cen = p1 + R_ * nrm
        a0 = math.atan2(p1[1] - cen[1], p1[0] - cen[0])
        n_ = max(2, int(abs(turn) * R_ / 0.15))
        for t in np.linspace(0, turn, n_ + 1)[1:]:
            pts.append(cen + R_ * np.array([math.cos(a0 + t), math.sin(a0 + t)]))
    pts += list(np.linspace(pts[-1], P[-1], max(2, int(np.linalg.norm(P[-1] - pts[-1]) / 0.15) + 1))[1:])
    return np.array(pts)


def make_route(P):
    """Polyline P (reference point f) -> reverse route [(x, y, th, -1)] (th = fork heading = opposite of travel); logs clearance on the current map. Also sets NAV['goal']."""
    xy = _round_polyline(P, V4_ROUTE_R)
    d = np.diff(xy, axis=0)
    th = np.arctan2(-d[:, 1], -d[:, 0])
    th = np.append(th, th[-1])
    path = [(float(px), float(py), float(t), -1) for (px, py), t in zip(xy, th)]
    NAV["goal"] = path[-1][:3]
    dist = NAV["grid"].dist_map()
    P_ = np.array([q[:3] for q in path])
    cen = FOOT_F.circles(P_[:, 0], P_[:, 1], P_[:, 2]).reshape(-1, 2)
    i_, j_ = NAV["grid"].cell(cen)
    dd = dist[np.clip(i_, 0, NAV["grid"].ny - 1), np.clip(j_, 0, NAV["grid"].nx - 1)]
    k_ = int(np.argmin(dd)) // len(FOOT_F.centers)
    log(f"[route] route {0.15 * len(path):.1f} m, min clearance on the current map (walls + objects seen so far) {float(dd.min()) - FOOT_F.r:+.2f} m "
        f"(at ({path[k_][0]:.2f}, {path[k_][1]:.2f}))")
    return path


V4_RET_Y = -6.0                                          # y of reference point f in front of the staging row when returning (fork tip to pallet face ~1.8 m after the spin)


def build_return_route(s, x_col):
    """Reverse back after unloading: branch -> (left turn) south along the aisle -> (right turn) to x_col in front of the staging row."""
    x, y, _ = f_pose(s["b"], s["fwd"])
    return make_route([(x, y), (V4_CX, y), (V4_CX, V4_RET_Y), (x_col, V4_RET_Y)])


def build_route(s):
    """Reverse carry with the pallet: current position -> east to the aisle centerline -> (left turn) north along the aisle -> (right turn) to the spin point at the east branch entrance (V4_SPIN_X)."""
    x, y, _ = f_pose(s["b"], s["fwd"])
    return make_route([(x, y), (V4_CX, y), (V4_CX, V4_BR_CY), (V4_SPIN_X, V4_BR_CY)])


def episode(ep, rng):
    ed = os.path.join(RUN_DIR, f"ep_{ep:03d}")
    px, py, pyw = place_pallets(rng)
    light = rng.uniform(*args.light) if args.rand else 1.0
    set_lights(light)
    for _ in range(50):
        d0 = rng.uniform(*args.dist)
        l0 = rng.uniform(-args.lat, args.lat)
        y0 = rng.uniform(-args.yaw, args.yaw)
        if start_ok(d0, l0, y0):
            break
    teleport(d0, l0, y0)
    physics(60)
    s = state()
    info = dict(ep=ep, start_dist=d0, start_lat=l0, start_yaw=y0, retries=0,
                pallet_x=px, pallet_y=py, pallet_yaw=math.degrees(pyw), light=light)
    if args.rand:
        log(f"[ep {ep}] pallet ({px:.2f}, {py:.2f}) yaw {math.degrees(pyw):+.1f}°, light ×{light:.2f}")
    log(f"[ep {ep}] start: fork tip to face {d0:.2f} m, lateral {l0:+.2f} m, heading {y0:+.1f}° "
        f"(measured lateral {s['tip_lat']:+.3f}, heading {math.degrees(s['yaw_err']):+.2f}°)")
    CUR["ed"] = ed
    LOCN.reset(random.Random(args.seed * 15485863 + ep))
    if ER is not None:
        info["env"] = ER.episode(random.Random(args.seed * 104729 + ep))
        log(f"[ep {ep}] appearance {info['env']}")
    info.update(obstacle_reset(ep, s))
    if ep == 0 and not args.no_scanners:
        _p, _o = scanner_points(s["b"], s["fwd"])
        log(f"[scanner] fraction of rays occluded by the chassis at the start pose: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in _o.items())
            + f" | {len(_p)} points")
    snap(os.path.join(ed, "0_start"))
    if args.policy_url:
        return vla_episode(ep, ed, info)
    rec_start(ed)
    t_ep = sim_t["t"]
    TOPGIF.update(frames=[], t=-9.0, t0=t_ep)
    if args.perturb:                                       # RNG separate from the layout and obstacle RNGs
        PERT.update(steer=0.0, until=-1.0, n=0, time=0.0, rng=random.Random(args.seed * 7907 + ep))
        PERT["dock_t"] = t_ep + PERT["rng"].uniform(1.0, 6.0) if PERT["rng"].random() < args.pert_dock else None
        PERT["dock_left"] = args.pert_dock_n - 1
    pre_along = -PALLET_HALF - PRE_GAP

    def approach(s):
        cmd["lift"] = 0.0
        cmd["steer"] = line_steer(s)
        remain = pre_along - s["tip_along"]
        cmd["v"] = max(0.08, min(V_APPROACH, 0.8 * remain))   # slow down near the check point

    result = None
    while True:
        r, s = run_phase(ep, "approach", lambda s: "ok" if s["tip_along"] >= pre_along else None, 40.0, approach)
        cmd.update(v=0.0)
        r2, s = run_phase(ep, "stop", lambda s: None, 0.5)
        lat_ok = abs(s["tip_lat"]) <= TOL_LAT
        yaw_ok = abs(s["yaw_err"]) <= TOL_YAW
        log(f"[ep {ep}] check #{info['retries']}: fork tip to face {-PALLET_HALF - s['tip_along']:.3f} m, "
            f"lateral {s['tip_lat']:+.3f} m (tol ±{TOL_LAT}), heading {math.degrees(s['yaw_err']):+.2f}° "
            f"(tol ±{math.degrees(TOL_YAW):.1f}) -> {'pass' if lat_ok and yaw_ok else 'retry'}")
        if r != "ok":
            result = f"approach_{r}"
            break
        if lat_ok and yaw_ok:
            break
        if info["retries"] >= MAX_RETRY:
            result = "misaligned"
            break
        info["retries"] += 1
        cmd.update(steer=0.0, v=-V_REVERSE)
        run_phase(ep, "reverse_retry", lambda s: "ok" if s["tip_along"] <= -PALLET_HALF - REVERSE_TO else None, 20.0,
                  lambda s: cmd.update(steer=reverse_steer(s, -V_REVERSE)))
        cmd.update(v=0.0)
    info.update(check_lat=s["tip_lat"], check_yaw=math.degrees(s["yaw_err"]))
    snap(os.path.join(ed, "1_aligned"))

    if result is None:
        cmd["lift"] = LIFT_LOW
        run_phase(ep, "lower", lambda s: "ok" if s["lift"] <= LIFT_LOW + 0.01 else None, 4.0)
        p_before = state()["pallet"]
        if os.environ.get("CARRY_DIAG"):            # diagnostics: pallet pose drift during insertion
            diag_log(ep, "before insertion")
        target = -PALLET_HALF - 0.02                       # leg front end 2 cm short of the face

        def insert(s):
            # During insertion, steer relative to the pallet position at the start: if friction drags the pallet slightly,
            # chasing it would steer further and push it more
            cmd["steer"] = line_steer(s, limit=math.radians(2.0), c=p_before[:2])
            cmd["v"] = max(args.insert_vmin, min(V_INSERT, 1.0 * (target - s["leg_along"])))

        last = {"a": None, "stall": 0.0}

        def ins_until(s):
            if np.linalg.norm(s["pallet"][:2] - p_before[:2]) > PUSH_TOL:
                return "pushed"
            if s["leg_along"] >= target - 0.005:
                return "ok"
            if last["a"] is not None and abs(s["tip_along"] - last["a"]) < 0.0005:
                last["stall"] += 0.1
                if last["stall"] >= 2.0:
                    return "blocked"
            else:
                last["stall"] = 0.0
            last["a"] = s["tip_along"]
            return None

        r, s = run_phase(ep, "insert", ins_until, 30.0, insert)
        cmd.update(v=0.0, steer=0.0)
        pushed = float(np.linalg.norm(s["pallet"][:2] - p_before[:2]))
        info.update(insert=r, pushed=pushed, insert_lat=s["tip_lat"])
        log(f"[ep {ep}] insert {r}: leg to face {-PALLET_HALF - s['leg_along']:.3f} m, lateral {s['tip_lat']:+.3f}, "
            f"heading {math.degrees(s['yaw_err']):+.2f}°, pallet pushed {pushed:.3f} m")
        snap(os.path.join(ed, "2_inserted"))
        if r != "ok":
            result = f"insert_{r}"

    if result is None:
        z0 = p_before[2]
        cmd["lift"] = LIFT_HIGH
        r, s = run_phase(ep, "lift", lambda s: "ok" if s["lift"] >= LIFT_HIGH - 0.01 else None, 5.0)
        run_phase(ep, "lift_hold", lambda s: None, 0.5)
        s = state()
        rise = s["pallet"][2] - z0
        info.update(rise=rise, tilt=s["tilt"])
        log(f"[ep {ep}] lift: lift {s['lift']:+.3f}, pallet rise {rise:+.3f} m, tilt {s['tilt']:.1f}°")
        snap(os.path.join(ed, "3_lifted"))
        if rise < 0.1:
            result = "lift_fail"

    if result is None:
        # Back out -> spin in place (rear east) -> reverse along the fixed aisle route -> unloading control near the drop target
        r = spin_out(ep, p_before[:2])
        if r != "ok":
            result = f"spin_{r}"
    if result is None:
        info["carry_retries"] = 0
        s = state()
        route = build_route(s)
        NAV.update(mode="detour", path=route, idx=0, vmax=V_CARRY)
        save_map("route.png", s, "route (green = reverse)")
        if args.dock_only > 0:
            # Dock-only: reverse along the aisle for args.dock_only s and end while still moving
            r, s = run_phase(ep, "carry", lambda s: "blocked" if NAV["fails"] >= 8 else None, args.dock_only, carry_control)
            result = "success" if r == "timeout" else f"carry_{r}"
        else:
            # Treat 8 accumulated planning failures as blocked and end (failed episodes are not used for training)
            r, s = run_phase(ep, "carry", lambda s: "ok" if NAV["mode"] == "nominal" else ("blocked" if NAV["fails"] >= 8 else None),
                             400.0, carry_control)
            cmd.update(v=0.0)
            if r != "ok":
                result = f"carry_{r}"
    if result is None:
        # Spin 180° counter-clockwise at the branch entrance (rear corner sweeps north, load sweeps south) -> forks east (end wall)
        r = spin_to(ep, 0.0, sgn=1.0, tag="spin2")
        if r != "ok":
            result = r
    if result is None:
        # Drive forks-first to drop pose G -> check (if off, back up 2 m and retry)
        SAFE["drop"] = True
        while True:
            r, s = run_phase(ep, "drop_approach", lambda s: "ok" if goal_err(s)[0] >= -0.005 else None, 40.0,
                             lambda s: cmd.update(v=max(0.08, min(0.25, 0.8 * -goal_err(s)[0])), steer=carry_forward_steer(s)))
            cmd.update(v=0.0)
            run_phase(ep, "carry_stop", lambda s: None, 0.5)
            s = state()
            along, e_f, psi = goal_err(s)
            lat_b = float((s["b"] - GOAL) @ GOAL_N)
            ok = abs(lat_b) <= CARRY_TOL_LAT and abs(psi) <= CARRY_TOL_YAW
            log(f"[ep {ep}] drop pose check #{info['carry_retries']}: along {along:+.3f} m, lateral {lat_b:+.3f} m "
                f"(tol ±{CARRY_TOL_LAT}), heading {math.degrees(psi):+.2f}° (tol ±{math.degrees(CARRY_TOL_YAW):.0f}) "
                f"-> {'pass' if ok else 'realign'}")
            if r != "ok":
                result = f"drop_approach_{r}"
                break
            if ok:
                break
            if info["carry_retries"] >= CARRY_RETRY:
                result = "carry_misaligned"
                break
            info["carry_retries"] += 1
            run_phase(ep, "carry_fix", lambda s: "ok" if goal_err(s)[0] <= -2.0 else None, 20.0,
                      lambda s: cmd.update(v=-V_REVERSE, steer=kappa_to_steer(-1.0 * goal_err(s)[2] / -V_REVERSE)))
            cmd.update(v=0.0)
        SAFE["drop"] = False
        info.update(goal_lat=lat_b, goal_along=along, goal_yaw=math.degrees(psi))
        snap(os.path.join(ed, "4_carried"))

    if result is None:
        # Lower the pallet -> reverse to pull the forks out
        cmd.update(v=0.0, steer=0.0, lift=LIFT_LOW)
        run_phase(ep, "drop", lambda s: "ok" if s["lift"] <= LIFT_LOW + 0.01 else None, 5.0)
        run_phase(ep, "drop_hold", lambda s: None, 0.5)
        s = state()
        p_drop = s["pallet"].copy()
        b0 = s["b"].copy()
        cmd.update(v=-V_REVERSE, steer=0.0)
        r, s = run_phase(ep, "pullout", lambda s: "ok" if np.linalg.norm(s["b"] - b0) >= PULLOUT else None, 20.0)   # slows to 0.15 m/s near the wall
        cmd.update(v=0.0)
        run_phase(ep, "end", lambda s: None, 0.5)
        s = state()
        dragged = float(np.linalg.norm(s["pallet"][:2] - p_drop[:2]))
        pz = s["pallet"][:2] - np.array(ZONE)
        pal_yaw_err = _pallet_yaw_err()
        h = s["pallet"][2] - z0
        info.update(drop_dx=float(pz[0]), drop_dy=float(pz[1]), drop_yaw=pal_yaw_err, dragged=dragged, drop_h=h)
        log(f"[ep {ep}] drop: pallet to zone center x {pz[0]:+.3f} y {pz[1]:+.3f} m, yaw {pal_yaw_err:+.1f}°, "
            f"height {h:+.3f} m, drag while pulling out {dragged:.3f} m")
        snap(os.path.join(ed, "5_dropped"))
        if r != "ok":
            result = f"pullout_{r}"
        elif abs(h) > 0.03:
            result = "not_on_floor"
        elif dragged > 0.05:
            result = "dragged"
        elif max(abs(pz[0]), abs(pz[1])) > 0.3 or abs(pal_yaw_err) > 5.0:
            result = "off_zone"
        else:
            result = "success"

    if result == "success" and not args.dock_only:
        # Return empty to the staging area: reverse out of the branch, (left turn) reverse down the aisle -> (right turn) in front of the next pallet -> spin 90° in place (forks north)
        RET["on"] = True
        xs = LAYOUT.get("others_x") or [V4_SLOTS_X[rng.randrange(3)]]
        x_col = xs[rng.randrange(len(xs))]
        s = state()
        NAV.update(mode="detour", path=build_return_route(s, x_col), idx=0, vmax=V_CARRY, fails=0, no_plan_until=-9.0)
        save_map("return_route.png", s, "return route (green = reverse)")
        r, s = run_phase(ep, "return", lambda s: "ok" if NAV["mode"] == "nominal" else ("blocked" if NAV["fails"] >= 8 else None),
                         400.0, carry_control)
        cmd.update(v=0.0)
        if r == "ok":
            r = spin_to(ep, math.pi / 2, tag="return_spin")
        RET["on"] = False
        s = state()
        snap(os.path.join(ed, "6_returned"))
        tip_gap = float(V4_ROW_Y - PALLET_HALF - s["tip"][1])
        info.update(return_x=x_col, return_tip_gap=tip_gap)
        log(f"[ep {ep}] return {r}: in front of staging row x {x_col:.2f}, fork tip to pallet face {tip_gap:.2f} m, heading {math.degrees(math.atan2(s['fwd'][1], s['fwd'][0])):+.1f}°")
        if r != "ok":
            result = f"return_{r}"

    top_gif_save(ed)
    PERT["steer"] = 0.0
    if args.perturb:
        info.update(perturbs=PERT["n"], perturb_time=PERT["time"])
        log(f"[ep {ep}] {PERT['n']} perturbations (total {PERT['time']:.1f} s)")
    info.update(result=result, sim_time=sim_t["t"] - t_ep, frames=rec_end(),
                safety_stops=SAFE["stops"], stop_time=SAFE["stop_time"], min_clear=SAFE["min_clear"],
                plans=NAV.get("plans", 0), detours=NAV.get("detours", 0), plan_fails=NAV.get("fails", 0))
    if NAV.get("grid") is not None:
        save_map("end.png", s, f"end: {result}")
        map_gif()
    if state()["lift"] > LIFT_LOW + 0.05:
        unload_unrecorded(ep)
    log(f"[ep {ep}] {SAFE['stops']} safety stops (total {SAFE['stop_time']:.1f} s), min obstacle-to-chassis distance {SAFE['min_clear']:.2f} m")
    log(f"[ep {ep}] result {result} ({info['retries']} retries, {info['sim_time']:.1f} s)")
    cmd.update(v=0.0, steer=0.0, lift=0.0)
    return info


# ---- Driving with a learned policy ----
import json  # noqa: E402
import urllib.request  # noqa: E402


def post_act(paths, st):
    body = json.dumps({"images": paths, "state": st, "task": args.task}).encode()
    req = urllib.request.Request(args.policy_url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    while True:                                   # retry for up to 3 min if the server is restarting (e.g. after a driver segfault)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            d = json.loads(e.read())
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if time.time() - t0 > 180:
                raise
            log(f"[policy] server connection failed {e!r} -> retrying")
            time.sleep(3)
    if "error" in d:
        raise RuntimeError(f"policy server error {d['error']}")
    return d["actions"], d["latency"], time.time() - t0


def _pallet_yaw_err():
    """Angle [deg] between the pallet's wide face (local x) and the drop target direction (GOAL_N), folded to ±90 (target 0)."""
    w_, x_, y_, z_ = map(float, pal_rp.get_world_pose()[1])
    a = math.degrees(math.atan2(2 * (x_ * y_ + w_ * z_), 1 - 2 * (y_ * y_ + z_ * z_)))
    a -= math.degrees(math.atan2(GOAL_N[1], GOAL_N[0]))    # drop target: wide face along GOAL_N
    return (a + 90.0) % 180.0 - 90.0


def inside(b):
    return BOUNDS[0][0] <= b[0] <= BOUNDS[0][1] and BOUNDS[1][0] <= b[1] <= BOUNDS[1][1]


class ExpertShadow:
    """Shadow expert for DAgger (legacy): follows the phase flow of episode() (approach -> check -> (retreat and retry) -> lower -> insert -> lift -> carry -> drop check -> (realign) -> lower -> pull out)
    from the current state only and returns the command (v, steer, lift) the expert would issue every 0.1 s. The wheels receive the VLA command.
    During carry it calls carry_control directly (the NAV state advances as the shadow expert's)."""

    def __init__(self):
        self.ph, self.lift, self.mem = "approach", 0.0, {}

    def label(self, s, rise=0.0, carried=0.0):
        """rise = pallet rise [m], carried = distance carried while lifted [m]. Keeps the phase in sync with the actual state when the VLA deviates from the shadow
        (e.g. the VLA has already lifted while the shadow is approaching, or has set the pallet down while the shadow is carrying)."""
        pre_along = -PALLET_HALF - PRE_GAP
        ph = self.ph
        if rise >= 0.1 and ph in ("approach", "retreat", "lower", "insert"):
            ph = "lift"
        if ph in ("lift", "carry", "carry_fix", "drop") and carried >= 2.0 and rise < 0.03 and s["lift"] < 0.0:
            ph = "pullout"                                  # already set down -> pull out the forks
            self.mem.setdefault("b0", s["b"].copy())
        v, st = 0.0, 0.0
        if ph == "approach":
            if s["tip_along"] >= pre_along:                     # reached (or the VLA passed) the check point -> decide
                ok = abs(s["tip_lat"]) <= TOL_LAT and abs(s["yaw_err"]) <= TOL_YAW
                ph = "lower" if ok else "retreat"
            else:
                self.lift = 0.0
                st = line_steer(s)
                v = max(0.08, min(V_APPROACH, 0.8 * (pre_along - s["tip_along"])))
        if ph == "retreat":
            if s["tip_along"] <= -PALLET_HALF - REVERSE_TO:
                ph = "approach"
                st = line_steer(s)
                v = V_APPROACH
            else:
                v = -V_REVERSE
                st = reverse_steer(s, -V_REVERSE)
        if ph == "lower":
            self.lift = LIFT_LOW
            if s["lift"] <= LIFT_LOW + 0.03:
                ph = "insert"
                self.mem["c"] = s["pallet"][:2].copy()
        if ph == "insert":
            self.lift = LIFT_LOW
            target = -PALLET_HALF - 0.02
            if s["leg_along"] >= target - 0.005:
                ph = "lift"
            else:
                st = line_steer(s, limit=math.radians(2.0), c=self.mem.get("c"))
                v = max(args.insert_vmin, min(V_INSERT, 1.0 * (target - s["leg_along"])))
        if ph == "lift":
            self.lift = LIFT_HIGH
            if s["lift"] >= LIFT_HIGH - 0.05 and rise >= 0.1:      # move on to carry even if the VLA lifts slightly below 0.15
                ph = "carry"
                NAV.update(mode="nominal")
        if ph == "carry":
            self.lift = LIFT_HIGH
            if NAV.get("mode") == "nominal" and goal_err(s)[0] <= 0.01:
                along, e_f, psi = goal_err(s)
                lat_b = float((s["b"] - GOAL) @ GOAL_N)
                ph = "drop" if (abs(lat_b) <= CARRY_TOL_LAT and abs(psi) <= CARRY_TOL_YAW) else "carry_fix"
            else:
                c0 = dict(cmd)
                carry_control(s)                           # shadow expert's carry command (including detours)
                v, st = cmd["v"], cmd["steer"]
                cmd.update(c0)
        if ph == "carry_fix":
            self.lift = LIFT_HIGH
            if goal_err(s)[0] >= 2.0:
                ph = "carry"
                NAV.update(mode="nominal")
            else:
                v, st = V_CARRY, carry_forward_steer(s)
        if ph == "drop":
            self.lift = LIFT_LOW
            if s["lift"] <= LIFT_LOW + 0.03:
                ph = "pullout"
                self.mem["b0"] = s["b"].copy()
        if ph == "pullout":
            self.lift = LIFT_LOW
            if np.linalg.norm(s["b"] - self.mem["b0"]) < PULLOUT:
                v = -V_REVERSE
        self.ph = ph
        return dict(v=float(v), steer=float(st), lift=float(self.lift))


HO = {"n": 0, "t": 0.0}                                  # --handover: number of handovers, time driven by the planner


def vla_episode(ep, ed, info):
    for k_ in REC_CAMS:
        os.makedirs(os.path.join(ed, "frames", k_), exist_ok=True)
    p0, _ = pallet_pose()
    t0 = sim_t["t"]
    chunk, k, i, prev, rows, lats = [], 0, 0, None, [], []
    best = dict(max_leg=-9.0, max_rise=0.0, carried=0.0)
    p_lift = None                                    # pallet position when first lifted 0.1 m or more
    dt = CTRL_EVERY * PHYSICS_DT
    result = None
    HO.update(n=0, t=0.0)
    shadow = ExpertShadow() if args.dagger else None
    drng = random.Random(args.seed * 7777 + ep)
    exec_expert = False
    if args.dagger and args.save_lidar:
        os.makedirs(os.path.join(ed, "lidar"), exist_ok=True)
    TOPGIF.update(frames=[], t=-9.0, t0=t0)               # --top-gif also applies to policy evaluation
    while app.is_running():
        top_frame("vla" if best.get("t_drop") is not None or p_lift is None else "vla lifted")
        s = state()
        record(ep, "vla", s)
        disp = float(np.linalg.norm(s["pallet"][:2] - p0[:2]))
        rise = float(s["pallet"][2] - p0[2])
        best["max_leg"] = max(best["max_leg"], s["leg_along"])
        best["max_rise"] = max(best["max_rise"], rise)
        if rise >= 0.1 and p_lift is None:
            p_lift = s["pallet"][:2].copy()
        if p_lift is not None:
            best["carried"] = max(best["carried"], float(np.linalg.norm(s["pallet"][:2] - p_lift)))
        pz = s["pallet"][:2] - np.array(ZONE)
        fork_out = float((s["tip"] - s["pallet"][:2]) @ s["fwd"]) < -PALLET_HALF - 0.05   # fork tip outside the pallet face
        in_zone = (max(abs(pz[0]), abs(pz[1])) <= 0.3 and abs(_pallet_yaw_err()) <= 5.0) if args.strict_drop \
            else (abs(pz[0]) <= ZONE_L / 2 and abs(pz[1]) <= ZONE_W / 2)   # default: success if inside the zone (GOAL_D = +x -> L is along x)
        dropped_ok = best["carried"] >= 2.0 and abs(rise) < 0.03 and fork_out and in_zone
        if dropped_ok and best.get("t_drop") is None:
            best["t_drop"] = sim_t["t"] - t0
            log(f"[ep {ep}] drop succeeded ({best['t_drop']:.1f} s) -> checking return")
            RET["on"] = True                         # for the person crossing on the way back
        # Success = pallet set down in the drop zone, then back at the staging area (rectangle in front of the row, forks north ±15°)
        back = (best.get("t_drop") is not None and V4_SLOTS_X[0] - 1.3 <= s["f"][0] <= V4_SLOTS_X[-1] + 1.3
                and V4_RET_Y - 1.0 <= s["f"][1] <= V4_ROW_Y - PALLET_HALF - 1.4
                and abs(PL.wrap(math.atan2(s["fwd"][1], s["fwd"][0]) - math.pi / 2)) <= math.radians(15.0))
        if back:
            result = "success"
        elif best.get("t_drop") is None and p_lift is None and rise < 0.05 and disp > 0.15:
            result = "pushed"
        elif best.get("t_drop") is None and best["carried"] >= 2.0 and abs(rise) < 0.03 and fork_out:
            result = "off_zone"                      # set down and forks pulled out, but outside the zone
        elif s["tilt"] > 15.0:
            result = "tipped"
        elif not inside(s["b"]):
            result = "out_of_bounds"
        elif sim_t["t"] - t0 > args.max_time:
            result = "no_return" if best.get("t_drop") is not None else "timeout"
        if result:
            break
        # Same safety-field switch as the expert's drop_approach: while carrying inside the drop branch with the forks facing east (end wall), front field 0.1 m
        #   (without it the safety stop triggers 0.3 m before the wall and the pallet ends up short of the zone center)
        SAFE["drop"] = (rise >= 0.1 and best.get("t_drop") is None and s["f"][0] >= V4_X1
                        and V4_BR_Y0 <= s["f"][1] <= V4_TOP and s["fwd"][0] >= math.cos(math.radians(30.0)))
        ctl = "vla"
        if args.handover and p_lift is not None and rise >= 0.1:
            # --handover, while carrying: the planner takes over using the expert's criteria (4 m nominal prediction blocked for 1.5 s+, or a safety stop of 3 s+)
            prev_mode = NAV["mode"]
            v_keep, st_keep = cmd["v"], cmd["steer"]
            carry_control(s)                            # if nominal, only the blockage check is used and the VLA command is restored
            if NAV["mode"] != "nominal":
                ctl = "plan"
                if prev_mode == "nominal":
                    HO["n"] += 1
                    log(f"[handover] #{HO['n']} VLA -> planner ({NAV['mode']})")
            else:
                cmd.update(v=v_keep, steer=st_keep)
                if prev_mode != "nominal":
                    chunk = []                          # on return, query again with the current view
                    log(f"[handover] planner -> VLA ({goal_err(s)[0]:+.2f} m to the goal along the line)")
            if NAV["fails"] >= 8:
                result = "carry_blocked"
                break
        if ctl == "plan":
            HO["t"] += dt
        need = ctl == "vla" and ((not chunk) or k >= min(args.exec_steps, len(chunk)))
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0)
        paths = {}
        for k_, an in REC_CAMS.items():
            im = np.ascontiguousarray(np.asarray(annots[an].get_data())[..., :3])
            paths[k_] = os.path.join(ed, "frames", k_, f"{i:05d}.jpg")
            if need:
                _save_jpg(im, paths[k_])                     # synchronous save: the server reads it immediately
            else:
                POOL.submit(_save_jpg, im, paths[k_])
        v_meas = 0.0 if prev is None else float((s["b"] - prev) @ s["fwd"]) / dt
        st = [s["lift"], float(art.get_joint_positions()[I_STEER]), v_meas, *goal_rel_in(s)]
        if need:
            chunk, lat, rtt = post_act(paths, st)
            k = 0
            lats.append(lat)
        lab = shadow.label(s, rise, best["carried"]) if shadow is not None else None
        if ctl == "vla":
            if need and shadow is not None:
                exec_expert = drng.random() < args.dagger_beta     # per 1 s chunk: whether the expert executes
            a = chunk[k]
            k += 1
            if lab is not None and exec_expert:
                cmd.update(lab)
                ctl = "expert"
            else:
                cmd.update(v=max(-0.5, min(0.6, a[0])), steer=max(-SPIN_STEER, min(SPIN_STEER, a[1])), lift=max(-0.15, min(0.3, a[2])))
        if lab is not None and args.save_lidar:
            POOL.submit(np.save, os.path.join(ed, "lidar", f"{i:05d}.npy"), LIDAR_STAT.get("Pb", np.zeros((0, 3), np.float16)))
        act = lab if lab is not None else cmd                       # DAgger: action columns = shadow expert labels
        g0 = goal_rel(s)
        rows.append([i, f"{i * dt:.2f}", ctl if lab is None else f"{ctl}:{shadow.ph}", f"{st[0]:.5f}", f"{st[1]:.5f}", f"{st[2]:.5f}",
                     f"{st[3]:.4f}", f"{st[4]:.4f}", f"{st[5]:.5f}", f"{act['v']:.5f}",
                     f"{act['steer']:.5f}", f"{act['lift']:.5f}", f"{s['tip_along']:.4f}", f"{s['tip_lat']:.4f}",
                     f"{s['yaw_err']:.5f}", f"{SAFE['v']:.5f}", SAFE["state"], NAV.get("mode", "") if (args.handover or args.dagger) else "",
                     f"{g0[0]:.4f}", f"{g0[1]:.4f}", f"{g0[2]:.5f}", f"{s['b'][0]:.4f}", f"{s['b'][1]:.4f}",
                     f"{math.atan2(s['fwd'][1], s['fwd'][0]):.5f}"])
        if i % 20 == 0:
            log(f"[ep {ep}] t {i * dt:5.1f}s fork tip to face {-PALLET_HALF - s['tip_along']:+.2f} lateral {s['tip_lat']:+.3f} "
                f"heading {math.degrees(s['yaw_err']):+.1f}° | cmd v {cmd['v']:+.2f} steer {math.degrees(cmd['steer']):+.1f}° "
                f"lift {cmd['lift']:+.3f} (actual {s['lift']:+.3f}) | pallet moved {disp:.2f} rise {rise:+.3f} | "
                f"goal fwd {st[3]:+.2f} left {st[4]:+.2f} yaw {math.degrees(st[5]):+.1f}°")
        prev = s["b"].copy()
        i += 1
        physics(CTRL_EVERY)
        _stop_snap()
    with open(os.path.join(ed, "data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DATA_COLS)
        w.writerows(rows)
    snap(os.path.join(ed, "9_end"))
    top_gif_save(ed)
    pz = s["pallet"][:2] - np.array(ZONE)
    SAFE["drop"] = False
    info.update(safety_stops=SAFE["stops"], stop_time=SAFE["stop_time"], min_clear=SAFE["min_clear"])
    if args.handover:
        info.update(plans=NAV.get("plans", 0), detours=NAV.get("detours", 0), plan_fails=NAV.get("fails", 0),
                    handovers=HO["n"], plan_time=HO["t"])
        log(f"[ep {ep}] {HO['n']} handovers, planner drove {HO['t']:.1f} s, {NAV.get('plans', 0)} plans ({NAV.get('fails', 0)} failed)")
        save_map("end.png", s, f"end: {result}")
        map_gif()
    info.update(result=result or "closed", sim_time=sim_t["t"] - t0, frames=i, queries=len(lats),
                latency=float(np.mean(lats)) if lats else 0.0, rise=best["max_rise"], carried=best["carried"],
                max_insert=-PALLET_HALF - best["max_leg"], drop_dx=float(pz[0]), drop_dy=float(pz[1]), drop_time=best.get("t_drop"),
                drop_yaw=_pallet_yaw_err(), drop_h=rise)
    log(f"[ep {ep}] result {info['result']} ({info['sim_time']:.1f} s, {len(lats)} queries, mean {info['latency']:.2f} s) | "
        f"leg to face at max insertion {info['max_insert']:+.3f} m, max rise {best['max_rise']:+.3f}, carried {best['carried']:.2f} m | "
        f"pallet to zone center x {pz[0]:+.2f} y {pz[1]:+.2f} m, yaw {info['drop_yaw']:+.1f}°, height {rise:+.3f}")
    cmd.update(v=0.0, steer=0.0, lift=0.0)
    return info


if args.layout_only:
    # Layout snapshots: top view, from the south entrance along the aisle, inside the aisle, the drop branch, and the forklift cameras
    _rng = random.Random(args.seed)
    place_pallets(_rng)
    teleport(2.5, 0.0, 0.0)
    if ER is not None:
        ER.episode(random.Random(args.seed))
    physics(30)
    LD = os.path.join(RUN_DIR, "layout")
    shots = [("1_top", None, None),
             ("2_south_entry", (6.0, -9.5, 4.5), (1.0, 2.0, 0.0)),
             ("3_corridor", (2.0, -1.0, 2.6), (2.0, 10.0, 0.5)),
             ("4_drop_branch", (1.0, 8.0, 5.0), (6.0, 12.5, 0.0)),
             ("5_loading", (2.0, -9.0, 4.0), (-3.4, -2.5, 0.3))]
    for nm, eye, tgt in shots:
        if eye is not None:
            obs_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1)).GetInverse())
        snap(os.path.join(LD, nm))
        src = os.path.join(LD, nm, "top.png" if eye is None else "observer.png")
        shutil.copy2(src, os.path.join(LD, f"{nm}.png"))
        if nm == "1_top":
            for k_ in ("front_camera", "fork_camera", "rear_camera"):
                shutil.copy2(os.path.join(LD, nm, f"{k_}.png"), os.path.join(LD, f"6_robot_{k_}.png"))
    log(f"[layout] layout snapshots -> {LD}")
    app.close()
    sys.exit(0)

if args.start_delay > 0:
    log(f"[main] starting in {args.start_delay:.0f}s")
    w0 = time.time()
    while time.time() - w0 < args.start_delay and app.is_running():
        app.update()

rng = random.Random(args.seed)
EP_COLS = ["ep", "result", "retries", "start_dist", "start_lat", "start_yaw", "pallet_x", "pallet_y", "pallet_yaw",
           "light", "check_lat", "check_yaw", "insert", "insert_lat", "pushed", "rise", "tilt", "carried",
           "carry_retries", "goal_along", "goal_lat", "goal_yaw", "drop_dx", "drop_dy", "drop_yaw", "dragged", "drop_h",
           "sim_time",
           "frames", "queries", "latency", "max_insert",
           "obstacle", "obs_u", "obs_lat", "n_boxes", "feasible", "safety_stops", "stop_time", "min_clear", "plans", "detours", "plan_fails",
           "handovers", "plan_time", "env", "return_x", "return_tip_gap", "drop_time", "perturbs", "perturb_time"]
results = []
with open(os.path.join(RUN_DIR, "episodes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(EP_COLS)
    for ep in range(args.episodes):
        if not app.is_running():
            break
        try:
            info = episode(ep, rng)
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"[ep {ep}] error {e!r}\n{traceback.format_exc()}")
            info = dict(ep=ep, result="error")
        results.append(info)
        w.writerow([(f"{info[k]:.4f}" if isinstance(info.get(k), float) else info.get(k, "")) for k in EP_COLS])
        f.flush()
        trace_f.flush()
trace_f.close()
POOL.shutdown(wait=True)

n_ok = sum(1 for r in results if r.get("result") == "success")
from collections import Counter  # noqa: E402
log(f"[result] success {n_ok}/{len(results)} | " + ", ".join(f"{k} {v}" for k, v in Counter(r.get('result') for r in results).items())
    + ("" if args.policy_url else f" | retries {np.mean([r.get('retries', 0) for r in results]):.2f} on average")
    + (f" | policy {args.policy_url}" if args.policy_url else ""))

if args.close:
    app.close()
else:
    log("[main] window kept open; close it to exit")
    while app.is_running():
        app.update()
    app.close()
