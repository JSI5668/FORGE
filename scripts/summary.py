"""Stage-wise summary of policy evaluation runs.

    python scripts/summary.py captures/eval_forge_s100*

Each argument is a run folder written by forge_sim.py with --policy-url
(it contains episodes.csv). The stages are cumulative:
  Lift      pallet raised by at least 0.1 m
  Carry     pallet carried at least 2 m while lifted
  Drop-off  pallet put down with its center inside the drop zone (x within 0.8 m, y within 0.9 m)
  Return    forklift came back to the staging area (= mission success)
  Strict    returned episodes whose pallet is within 0.3 m and 5 deg of the zone center
"""
import csv
import os
import sys
from collections import Counter


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


runs = [d for d in sys.argv[1:] if os.path.isfile(os.path.join(d, "episodes.csv"))]
if not runs:
    sys.exit(__doc__)
n = lift = carry = drop = ok = strict = 0
why = Counter()
for d in runs:
    for r in csv.DictReader(open(os.path.join(d, "episodes.csv"))):
        n += 1
        dx, dy, yaw = f(r.get("drop_dx")) or 9, f(r.get("drop_dy")) or 9, f(r.get("drop_yaw")) or 0
        is_lift = (f(r.get("rise")) or 0) >= 0.1
        is_carry = is_lift and (f(r.get("carried")) or 0) >= 2.0
        is_drop = is_carry and r.get("result") in ("success", "off_zone", "no_return") and abs(dx) <= 0.8 and abs(dy) <= 0.9
        is_ok = is_drop and r.get("result") == "success"
        lift += is_lift
        carry += is_carry
        drop += is_drop
        ok += is_ok
        strict += is_ok and max(abs(dx), abs(dy)) <= 0.3 and abs(yaw) <= 5.0
        if r.get("result") != "success":
            why[r.get("result")] += 1
print(f"runs: {len(runs)}  episodes: {n}")
print("| Lift | Carry | Drop-off | Return | Strict |")
print("|---|---|---|---|---|")
print(f"| {lift}/{n} | {carry}/{n} | {drop}/{n} | {ok}/{n} | {strict}/{n} |")
print("failures:", ", ".join(f"{k} {v}" for k, v in why.most_common()) or "none")
