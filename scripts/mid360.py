"""Simulated Livox Mid-360 lidar (PhysX raycasts). Imported by the expert in forge_sim.py.

Published Mid-360 specifications (verify against the device manual):
  360° horizontal, -7° to +52° vertical (sensor frame), about 20k points per frame at 10 Hz (200k points/s),
  0.1 m minimum and about 40 m maximum range, non-repetitive scan pattern (each frame samples different
  directions, so coverage fills in over time), built-in IMU.
Only the ray directions are modeled here; range is the geometric distance to the simulator colliders.
  - Non-repetitive pattern: azimuth from a golden-ratio sequence, elevation from a second low-discrepancy
    sequence, so consecutive frames sample different directions
  - Elevation distribution: uniform on the sphere (uniform in sin(elevation)); may differ from the real device
  - Optional range noise noise_std (default 0.02 m) and ray dropout drop (default 0)
Mounting: body-local position + orientation (inverted = 180° flip about the sensor x axis,
tilt = rotation about the body y axis, + = tilted down toward the rear (+x)).
"""
import math

import numpy as np

EL_MIN, EL_MAX = -7.0, 52.0          # elevation in sensor frame [deg]
RANGE_MIN, RANGE_MAX = 0.1, 40.0
POINTS_PER_FRAME = 20000             # 10 Hz
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def pattern(n, start=0):
    """n unit directions in sensor frame (x forward, y left, z up). start = cumulative point index (continue it across frames for a non-repetitive pattern)."""
    k = np.arange(start, start + n, dtype=np.float64)
    az = 2.0 * math.pi * ((k * GOLDEN) % 1.0)
    u = (k * 0.7548776662466927) % 1.0            # second low-discrepancy sequence (plastic number)
    s0, s1 = math.sin(math.radians(EL_MIN)), math.sin(math.radians(EL_MAX))
    el = np.arcsin(s0 + u * (s1 - s0))
    c = np.cos(el)
    return np.stack([c * np.cos(az), c * np.sin(az), np.sin(el)], axis=1)


def in_fov(d_sensor):
    """Whether sensor-frame directions are inside the field of view."""
    el = np.degrees(np.arcsin(np.clip(d_sensor[..., 2] / np.linalg.norm(d_sensor, axis=-1), -1, 1)))
    return (el >= EL_MIN) & (el <= EL_MAX)


def mount_R(inverted, tilt_deg):
    """Sensor -> body rotation matrix. inverted: 180° about the sensor x axis (upside down),
    tilt_deg: rotation about the body y axis; positive tilts the center of the sensor's downward view toward body +x (rear, handle side)."""
    R = np.eye(3)
    if inverted:
        R = np.diag([1.0, -1.0, -1.0])
    t = math.radians(tilt_deg)
    # Rotation about body y: positive t tilts the +x vector downward (-z)
    Ry = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
    return Ry @ R


class Mid360:
    """One mounted sensor. pos = body-local position (x positive toward the rear, negative toward the forks), R = mount_R(...)"""

    def __init__(self, name, pos, inverted=True, tilt_deg=0.0, noise_std=0.02, drop=0.0, seed=0):
        self.name, self.pos = name, np.asarray(pos, dtype=np.float64)
        self.inverted, self.tilt = inverted, tilt_deg
        self.R = mount_R(inverted, tilt_deg)
        self.noise_std, self.drop = noise_std, drop
        self.k = 0
        self.rng = np.random.default_rng(seed)

    def world(self, M_body):
        """Body world transform (A, t) (column-vector convention: p_w = A @ p + t) -> sensor origin, sensor-to-world rotation"""
        A, t = M_body
        return A @ self.pos + t, A @ self.R

    def frame_dirs(self, n=POINTS_PER_FRAME):
        d = pattern(n, self.k)
        self.k += n
        return d

    def scan(self, raycast, M_body, is_self, n=POINTS_PER_FRAME):
        """One frame. raycast(o, d, r) -> (hit, dist, path). Returns world points (m,3) and the self-occlusion ratio."""
        o, Rw = self.world(M_body)
        dirs = self.frame_dirs(n) @ Rw.T
        pts, n_self = [], 0
        for d in dirs:
            if self.drop and self.rng.random() < self.drop:
                continue
            hit, dist, path = raycast(o, d, RANGE_MAX)
            if not hit or dist < RANGE_MIN:
                continue
            if is_self(path):
                n_self += 1
                continue
            dist += self.rng.normal(0.0, self.noise_std) if self.noise_std else 0.0
            pts.append(o + d * dist)
        return (np.array(pts) if pts else np.zeros((0, 3))), n_self / max(1, len(dirs))


# ---- Mount: on the rear post (handle side), sensor about 2.2 m above the floor, mounted upside down, tilted 20° backward ----
# Position is body-local (the body origin is about 0.06 m above the floor)
MOUNT = dict(name="rear_inv_2.2_t20rear", pos=(0.80, 0.0, 2.15), inverted=True, tilt_deg=20.0)
POLE_BASE_Z = 1.05                   # top surface of the rear body (body-local)


def add_visual(stage, body_path, mount=MOUNT, prefix="_mid360"):
    """Visual only (no collision). Since the inverted sensor's base faces up, a gray post rises from the rear body top
    (offset +0.1 m in y), a horizontal arm at its tip covers the sensor, and the sensor hangs below it.
    Mid-360 housing (65 mm diameter, 60 mm tall, black) + window band (gray).
    No colliders are created so raycasts ignore it (on the real robot the post would block a very narrow sector)."""
    from pxr import Gf, UsdGeom
    x, y, z = mount["pos"]
    top = z + 0.07
    for nm, c, sz in ((f"{prefix}_pole", (x, y + 0.10, (POLE_BASE_Z + top) / 2), (0.035, 0.035, top - POLE_BASE_Z)),
                      (f"{prefix}_arm", (x, y + 0.05, top), (0.08, 0.13, 0.025))):
        b = UsdGeom.Cube.Define(stage, f"{body_path}/{nm}")
        b.CreateSizeAttr(1.0)
        b.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.58)])
        bx = UsdGeom.Xformable(b.GetPrim())
        bx.AddTranslateOp().Set(Gf.Vec3d(*c))
        bx.AddScaleOp().Set(Gf.Vec3f(*sz))
    R = mount_R(mount["inverted"], mount["tilt_deg"])
    m = Gf.Matrix4d(1.0)
    for i in range(3):
        for j in range(3):
            m[i][j] = float(R[j, i])              # USD uses the row-vector convention -> transpose
    m.SetTranslateOnly(Gf.Vec3d(x, y, z))
    xf = UsdGeom.Xform.Define(stage, f"{body_path}/{prefix}")
    xf.AddTransformOp().Set(m)
    body = UsdGeom.Cylinder.Define(stage, f"{body_path}/{prefix}/housing")
    body.CreateRadiusAttr(0.0325)
    body.CreateHeightAttr(0.06)
    body.CreateAxisAttr("Z")
    body.CreateDisplayColorAttr([Gf.Vec3f(0.04, 0.04, 0.05)])
    win = UsdGeom.Cylinder.Define(stage, f"{body_path}/{prefix}/window")      # laser window (band around the upper half of the sensor)
    win.CreateRadiusAttr(0.0335)
    win.CreateHeightAttr(0.022)
    win.CreateAxisAttr("Z")
    win.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.38, 0.45)])
    UsdGeom.Xformable(win.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.012))
    return f"{body_path}/{prefix}"
