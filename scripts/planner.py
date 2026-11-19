# Detour path planner: occupancy grid + Hybrid A* + map rendering (numpy/scipy/cv2 only, no Isaac Sim needed).
#   Imported by the expert in forge_sim.py. Standalone test: python scripts/planner.py (plans around a fake box -> PNG)
#
# Coordinates: world (x, y) [m]. Forklift reference point = fixed-axle (front roller) point f; it never slips sideways
#   and only moves along the heading (= body direction).
#   Heading θ = fork (forward) direction. Travel s > 0 = forward (fork side), s < 0 = reverse. Curvature κ: dθ/ds = κ (+ = turning left)
# Map: 0.05 m grid, log-odds. Cells crossed by scanner rays are freed (-), hit cells are occupied (+).
#   Fixed structures such as walls come from a prior map (static).
#   Obstacle positions are not known in advance; only what the scanners have seen enters the map.
import heapq
import math

import numpy as np
from scipy import ndimage

L_HIT, L_FREE, L_MIN, L_MAX, L_OCC = 0.9, -0.4, -2.0, 3.5, 0.5


class OccGrid:
    def __init__(self, x0, y0, x1, y1, res=0.05):
        self.x0, self.y0, self.res = x0, y0, res
        self.nx, self.ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        self.lo = np.zeros((self.ny, self.nx), np.float32)          # log-odds (0 = unknown)
        self.seen = np.zeros((self.ny, self.nx), bool)               # observed at least once
        self.static = np.zeros((self.ny, self.nx), bool)             # prior map (walls, shelves)
        self.since = np.full((self.ny, self.nx), np.inf)             # time the cell became occupied (to detect static obstacles)

    def cell(self, xy):
        xy = np.asarray(xy, float)
        j = np.floor((xy[..., 0] - self.x0) / self.res).astype(int)
        i = np.floor((xy[..., 1] - self.y0) / self.res).astype(int)
        return i, j

    def _inside(self, i, j):
        return (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)

    def add_static_box(self, x0, y0, x1, y1):
        i0, j0 = self.cell((x0, y0))
        i1, j1 = self.cell((x1, y1))
        self.static[max(i0, 0):max(i1 + 1, 0), max(j0, 0):max(j1 + 1, 0)] = True

    def set_outside_static(self, x0, y0, x1, y1):
        """Mark everything outside this rectangle as occupied (warehouse walls)."""
        m = np.ones_like(self.static)
        i0, j0 = self.cell((x0, y0))
        i1, j1 = self.cell((x1, y1))
        m[max(i0, 0):i1 + 1, max(j0, 0):j1 + 1] = False
        self.static |= m

    def update(self, origins, dirs, dists, hit, t, max_r):
        """Integrate a batch of rays. origins (N,2), dirs (N,2) unit vectors, dists (N,), hit (N,) bool (False = free up to max range)."""
        if len(dists) == 0:
            return
        step = self.res
        ts = np.arange(0.0, max_r, step)                              # (S,)
        free_len = np.where(hit, dists - step, dists)
        pts = origins[:, None, :] + dirs[:, None, :] * ts[None, :, None]   # (N,S,2)
        mask = ts[None, :] < free_len[:, None]
        i, j = self.cell(pts[mask])
        ok = self._inside(i, j)
        i, j = i[ok], j[ok]
        lin = np.unique(i * self.nx + j)
        fi, fj = lin // self.nx, lin % self.nx
        self.lo[fi, fj] = np.maximum(L_MIN, self.lo[fi, fj] + L_FREE)
        self.seen[fi, fj] = True
        if hit.any():
            hp = origins[hit] + dirs[hit] * dists[hit][:, None]
            i, j = self.cell(hp)
            ok = self._inside(i, j)
            lin = np.unique(i[ok] * self.nx + j[ok])
            hi, hj = lin // self.nx, lin % self.nx
            self.lo[hi, hj] = np.minimum(L_MAX, self.lo[hi, hj] + L_HIT - L_FREE * 0)
            self.seen[hi, hj] = True
        occ = self.lo > L_OCC
        newly = occ & ~np.isfinite(self.since)
        self.since[newly] = t
        self.since[~occ] = np.inf

    def clear_poly(self, corners):
        """Clear the area occupied by the forklift itself (e.g. traces of the pallet seen before lifting it)."""
        import cv2
        pts = np.stack(self.cell(np.asarray(corners)), axis=-1)[:, ::-1].astype(np.int32)   # (j, i)
        m = np.zeros(self.lo.shape, np.uint8)
        cv2.fillPoly(m, [pts], 1)
        m = m.astype(bool)
        self.lo[m] = np.minimum(self.lo[m], 0.0)
        self.since[m] = np.inf

    def occupied(self):
        return (self.lo > L_OCC) | self.static

    def dist_map(self):
        """Distance to the nearest occupied cell [m]."""
        return ndimage.distance_transform_edt(~self.occupied()) * self.res


class Footprint:
    """Cover the forklift + load footprint rectangle with a few circles. a: forward (+) distance from reference point f."""

    def __init__(self, a_min, a_max, half_w, margin, n=4):
        sp = (a_max - a_min) / n
        self.centers = a_min + sp / 2 + sp * np.arange(n)
        self.r = math.hypot(half_w, sp / 2) + margin
        self.a_min, self.a_max, self.half_w = a_min, a_max, half_w

    def circles(self, x, y, th):
        """x, y, th arrays (K,) -> circle centers (K, n, 2)"""
        c, s = np.cos(th), np.sin(th)
        return np.stack([x[:, None] + c[:, None] * self.centers, y[:, None] + s[:, None] * self.centers], -1)

    def corners(self, x, y, th):
        c, s = math.cos(th), math.sin(th)
        out = []
        for a, l in ((self.a_max, self.half_w), (self.a_max, -self.half_w), (self.a_min, -self.half_w), (self.a_min, self.half_w)):
            out.append((x + c * a - s * l, y + s * a + c * l))
        return out


def arc(x, y, th, kappa, s):
    """Pose after moving signed distance s from the reference point with curvature kappa (s may be an array)."""
    s = np.asarray(s, float)
    if abs(kappa) < 1e-6:
        return x + s * math.cos(th), y + s * math.sin(th), th + 0.0 * s
    th2 = th + kappa * s
    return x + (np.sin(th2) - math.sin(th)) / kappa, y - (np.cos(th2) - math.cos(th)) / kappa, th2


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class HybridAStar:
    def __init__(self, grid, foot, kappa_max, step=0.6, xy_res=0.2, th_bins=72, rev_cost=1.0, fwd_cost=1.3,
                 switch_cost=3.0, steer_cost=0.3, max_iter=25000):
        self.g, self.f = grid, foot
        self.ks = [-kappa_max, -kappa_max / 2, 0.0, kappa_max / 2, kappa_max]
        self.step, self.xy_res, self.th_bins = step, xy_res, th_bins
        self.rev_cost, self.fwd_cost, self.switch_cost, self.steer_cost = rev_cost, fwd_cost, switch_cost, steer_cost
        self.max_iter = max_iter
        self.n_sub = max(2, int(math.ceil(step / 0.15)))

    def _clear(self, xs, ys, ths):
        cen = self.f.circles(np.asarray(xs), np.asarray(ys), np.asarray(ths)).reshape(-1, 2)
        i, j = self.g.cell(cen)
        ok = (i >= 0) & (i < self.g.ny) & (j >= 0) & (j < self.g.nx)
        if not ok.all():
            return -1.0
        return float(self.dist[i, j].min())

    def _free(self, xs, ys, ths, travelled=99.0):
        c = self._clear(xs, ys, ths)
        # If the start pose is already closer than the margin r (e.g. after a safety stop in front of an obstacle),
        # allow the first 1.2 m as long as the clearance does not get smaller than it is now
        need = self.f.r if travelled > 1.2 else min(self.f.r, self.c0 - 0.01)
        return c >= need

    def _heuristic_map(self, goal_xy):
        """2D shortest distance around obstacles (0.2 m grid, 8-connected Dijkstra), computed backward from the goal."""
        r = self.xy_res
        g = self.g
        ny, nx = int(g.ny * g.res / r), int(g.nx * g.res / r)
        # Clearance at coarse-cell centers; a cell is passable if it has at least half the forklift width
        ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        ci = np.clip(((ii + 0.5) * r / g.res).astype(int), 0, g.ny - 1)
        cj = np.clip(((jj + 0.5) * r / g.res).astype(int), 0, g.nx - 1)
        passable = self.dist[ci, cj] >= self.f.half_w
        H = np.full((ny, nx), np.inf)
        gi, gj = int((goal_xy[1] - g.y0) / r), int((goal_xy[0] - g.x0) / r)
        gi, gj = min(max(gi, 0), ny - 1), min(max(gj, 0), nx - 1)
        H[gi, gj] = 0.0
        pq = [(0.0, gi, gj)]
        nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        while pq:
            d, i, j = heapq.heappop(pq)
            if d > H[i, j]:
                continue
            for di, dj, w in nb:
                a, b = i + di, j + dj
                if 0 <= a < ny and 0 <= b < nx and passable[a, b]:
                    nd = d + w * r
                    if nd < H[a, b]:
                        H[a, b] = nd
                        heapq.heappush(pq, (nd, a, b))
        self.H, self.Hny, self.Hnx = H, ny, nx

    def _h(self, x, y):
        i, j = int((y - self.g.y0) / self.xy_res), int((x - self.g.x0) / self.xy_res)
        if 0 <= i < self.Hny and 0 <= j < self.Hnx and np.isfinite(self.H[i, j]):
            return self.H[i, j]
        return 1e3

    def plan(self, start, goal_fn, goal_xy, first_dir=0):
        """start (x, y, th) = pose of reference point f. goal_fn(x, y, th) -> True when the goal is reached.
        Returns [(x, y, th, dir)] spaced 0.15 m apart, or None.
        first_dir: if nonzero, restrict the first motion to this direction (to back out opposite to a direction blocked by a safety stop)"""
        self.dist = self.g.dist_map()
        self._heuristic_map(goal_xy)
        sx, sy, sth = start
        self.c0 = self._clear([sx], [sy], [sth])
        if self.c0 <= 0.0:
            return None, 0
        key = lambda x, y, th: (int(math.floor(x / self.xy_res)), int(math.floor(y / self.xy_res)),
                                int(round(wrap(th) / (2 * math.pi / self.th_bins))) % self.th_bins)
        nodes = [(sx, sy, sth, 0, 0.0, -1, 0.0, 0.0)]      # x, y, th, dir, kappa, parent, g, distance travelled
        openq = [(self._h(sx, sy), 0)]
        closed = {}
        it = 0
        subs = np.linspace(self.step / self.n_sub, self.step, self.n_sub)
        while openq and it < self.max_iter:
            _, ni = heapq.heappop(openq)
            x, y, th, d0, k0, _, g0, tr = nodes[ni]
            kk = key(x, y, th)
            if kk in closed:
                continue
            closed[kk] = ni
            it += 1
            if ni > 0 and goal_fn(x, y, th):
                return self._extract(nodes, ni), it
            for d in (-1, 1):
                if ni == 0 and first_dir and d != first_dir:
                    continue
                for k in self.ks:
                    xs, ys, ths = arc(x, y, th, k, d * subs)
                    if not self._free(xs, ys, ths, tr + self.step):
                        continue
                    x2, y2, th2 = float(xs[-1]), float(ys[-1]), float(wrap(ths[-1]))
                    if key(x2, y2, th2) in closed:
                        continue
                    c = self.step * (self.rev_cost if d < 0 else self.fwd_cost) + self.steer_cost * abs(k) * self.step
                    if d0 != 0 and d != d0:
                        c += self.switch_cost
                    if abs(k - k0) > 1e-6:
                        c += 0.1
                    g2 = g0 + c
                    nodes.append((x2, y2, th2, d, k, ni, g2, tr + self.step))
                    heapq.heappush(openq, (g2 + 1.2 * self._h(x2, y2), len(nodes) - 1))
        return None, it

    def _extract(self, nodes, ni):
        chain = []
        while ni >= 0:
            chain.append(nodes[ni])
            ni = nodes[ni][5]
        chain.reverse()
        out = [(chain[0][0], chain[0][1], chain[0][2], chain[1][3] if len(chain) > 1 else -1)]
        for a, b in zip(chain[:-1], chain[1:]):
            d, k = b[3], b[4]
            n = max(2, int(math.ceil(self.step / 0.15)))
            xs, ys, ths = arc(a[0], a[1], a[2], k, d * np.linspace(self.step / n, self.step, n))
            out += [(float(px), float(py), float(pt), d) for px, py, pt in zip(xs, ys, ths)]
        return out


def path_blocked(grid, foot, path, from_idx=0, dist=None):
    """Whether the path (from from_idx on) is blocked in the current map."""
    if not path or from_idx >= len(path):
        return False
    if dist is None:
        dist = grid.dist_map()
    p = np.array([q[:3] for q in path[from_idx:]])
    cen = foot.circles(p[:, 0], p[:, 1], p[:, 2]).reshape(-1, 2)
    i, j = grid.cell(cen)
    ok = (i >= 0) & (i < grid.ny) & (j >= 0) & (j < grid.nx)
    return bool((~ok).any() or (dist[i[ok], j[ok]] < foot.r).any())


# ---- map rendering ----
def render(grid, crop, path=None, pose=None, foot=None, scan_xy=None, goal_line=None, zone=None, text=None, scale=2):
    """crop = (x0, y0, x1, y1). Up = +y (north). Returns a BGR image (cv2)."""
    import cv2
    i0, j0 = grid.cell((crop[0], crop[1]))
    i1, j1 = grid.cell((crop[2], crop[3]))
    i0, j0 = max(i0, 0), max(j0, 0)
    lo = grid.lo[i0:i1, j0:j1]
    img = np.full(lo.shape + (3,), 150, np.uint8)                   # unknown = gray
    seen = grid.seen[i0:i1, j0:j1]
    img[seen & (lo <= L_OCC)] = (245, 245, 245)                     # free = white
    img[lo > L_OCC] = (20, 20, 20)                                  # occupied (seen by scanners) = black
    img[grid.static[i0:i1, j0:j1] & ~(lo > L_OCC)] = (95, 80, 70)   # prior map (walls) = dark gray-brown
    img = cv2.resize(img[::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    H = img.shape[0]

    def px(x, y):
        return int((x - grid.x0 - j0 * grid.res) / grid.res * scale), int(H - (y - grid.y0 - i0 * grid.res) / grid.res * scale)

    if zone is not None:
        (zx, zy), (zw, zl) = zone
        cv2.rectangle(img, px(zx - zw / 2, zy + zl / 2), px(zx + zw / 2, zy - zl / 2), (0, 200, 230), 2)
    if goal_line is not None:
        cv2.line(img, px(*goal_line[0]), px(*goal_line[1]), (0, 200, 230), 1)
    if path:
        for a, b in zip(path[:-1], path[1:]):
            cv2.line(img, px(a[0], a[1]), px(b[0], b[1]), (40, 170, 40) if b[3] < 0 else (220, 120, 30), 2)
    if scan_xy is not None and len(scan_xy):
        for p in scan_xy:
            cv2.circle(img, px(p[0], p[1]), 1, (40, 40, 230), -1)
    if pose is not None and foot is not None:
        cs = np.array([px(*c) for c in foot.corners(*pose)], np.int32)
        cv2.polylines(img, [cs], True, (0, 140, 255), 2)
        tip = (pose[0] + math.cos(pose[2]) * foot.a_max, pose[1] + math.sin(pose[2]) * foot.a_max)
        cv2.circle(img, px(*tip), 3, (0, 140, 255), -1)
    if text:
        for k, line in enumerate(text.split("\n")):
            cv2.putText(img, line, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, line, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


if __name__ == "__main__":
    # Standalone test: plan a carry path in an empty warehouse map where one box has been 'seen'
    import os
    import sys
    import time
    import cv2
    out = sys.argv[1] if len(sys.argv) > 1 else "planner_test.png"
    g = OccGrid(-10.5, -12.4, 9.6, 18.2)
    g.set_outside_static(-10.33, -12.2, 9.41, 18.0)
    g.seen[:] = True
    g.lo[:] = L_MIN
    box = (4.0, -4.5)
    i, j = g.cell(np.array([[box[0] - 0.3, box[1] - 0.3], [box[0] + 0.3, box[1] + 0.3]]))
    g.lo[i[0]:i[1] + 1, j[0]:j[1] + 1] = L_MAX
    FIX = 0.906
    foot = Footprint(-0.9 - FIX, 2.25 - FIX, 0.62, 0.1)
    kmax = 0.8 * math.tan(math.radians(45)) / 1.48
    G = np.array([4.0, -9.315]) + FIX * np.array([0, 1])            # reference point f at drop-off
    th_g = math.pi / 2

    def goal_fn(x, y, th):
        return abs(x - G[0]) < 0.15 and G[1] - 0.1 <= y <= G[1] + 3.0 and abs(wrap(th - th_g)) < math.radians(6)
    pl = HybridAStar(g, foot, kmax)
    t0 = time.time()
    path, it = pl.plan((4.4, 3.2, math.pi / 2 + 0.05), goal_fn, (G[0], G[1] + 1.5))
    print(f"plan {'succeeded' if path else 'failed'}: {it} expansions, {time.time() - t0:.2f} s, points {len(path) if path else 0}")
    img = render(g, (-4, -12.4, 9.4, 6), path, (path[-1][0], path[-1][1], path[-1][2]) if path else None, foot,
                 text="planner test")
    cv2.imwrite(out, img)
    print(os.path.abspath(out))
