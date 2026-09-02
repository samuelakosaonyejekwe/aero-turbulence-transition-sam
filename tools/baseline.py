"""tools/baseline.py - numeric diff of every generated CSV against a snapshot.

The point of this project is that the report says what the CSVs say and the
CSVs say what the solver computes.  When the solver changes, the honest
question is not "does it still run" but "exactly which numbers moved, and by
how much".  Answering that by eye across 60-odd CSVs is how a change to the
compressibility correction gets mistaken for a change to the transition kernel.

    python3 tools/baseline.py save   <dir>          snapshot the tracked CSVs
    python3 tools/baseline.py diff   <dir> [--rtol 1e-9] [--quiet-same]

Numeric columns are compared with a relative tolerance and an absolute floor;
text columns exactly.  Exit status is 1 if anything moved, so it can gate a
commit.  A column that moved is reported with the worst row, its old and new
value, and the relative size of the move - which is what tells a rounding
change apart from a physics change.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utss_paths  # noqa: F401  (anchors ROOT and the working directory)
from utss_paths import ROOT


def tracked_csvs():
    out = subprocess.run(["git", "ls-files", "*.csv"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return sorted(out)


def save(dest):
    os.makedirs(dest, exist_ok=True)
    n = 0
    for rel in tracked_csvs():
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    print("saved %d CSVs to %s" % (n, dest))


def _cmp_frame(a, b, rtol, atol):
    """(list of (column, worst_rel, i, old, new), note) for two frames."""
    if list(a.columns) != list(b.columns):
        return [], "columns differ: %s -> %s" % (list(a.columns), list(b.columns))
    if len(a) != len(b):
        return [], "row count %d -> %d" % (len(a), len(b))
    moved = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv = x.to_numpy(dtype=float); yv = y.to_numpy(dtype=float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.abs(yv - xv)
            scale = np.maximum(np.abs(xv), np.abs(yv))
            rel = np.where(scale > atol, d/np.maximum(scale, 1e-300), d)
            rel = np.where(both_nan, 0.0, rel)
            rel = np.where(np.isnan(rel), np.inf, rel)   # NaN appeared/vanished
            bad = (d > atol) & (rel > rtol)
            bad = bad | (np.isnan(xv) != np.isnan(yv))
            if bad.any():
                i = int(np.nanargmax(np.where(bad, rel, -1.0)))
                moved.append((c, float(rel[i]), i, xv[i], yv[i]))
        else:
            xs = x.astype(str).to_numpy(); ys = y.astype(str).to_numpy()
            bad = xs != ys
            if bad.any():
                i = int(np.argmax(bad))
                moved.append((c, float("inf"), i, xs[i], ys[i]))
    return moved, None


def diff(snap, rtol=1e-9, atol=1e-12, quiet_same=False):
    files = tracked_csvs()
    changed = 0
    same = 0
    for rel in files:
        cur = os.path.join(ROOT, rel)
        old = os.path.join(snap, rel)
        if not os.path.exists(old):
            print("  NEW      %s" % rel); changed += 1; continue
        if not os.path.exists(cur):
            print("  DELETED  %s" % rel); changed += 1; continue
        try:
            a = pd.read_csv(old); b = pd.read_csv(cur)
        except Exception as e:                      # noqa: BLE001
            print("  UNREAD   %s (%s)" % (rel, e)); changed += 1; continue
        moved, note = _cmp_frame(a, b, rtol, atol)
        if note:
            print("  SHAPE    %-52s %s" % (rel, note)); changed += 1; continue
        if not moved:
            same += 1
            if not quiet_same:
                print("  same     %s" % rel)
            continue
        changed += 1
        print("  CHANGED  %s" % rel)
        for c, r, i, o, n in sorted(moved, key=lambda t: -t[1])[:8]:
            rs = "new/lost" if r == float("inf") else "%.3g" % r
            print("      %-28s worst rel %-9s row %-5d %s -> %s"
                  % (c, rs, i, o, n))
        if len(moved) > 8:
            print("      ... and %d more columns" % (len(moved) - 8))
    print("\n%d CSV(s) unchanged, %d changed (rtol=%g)" % (same, changed, rtol))
    return changed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, d = sys.argv[1], sys.argv[2]
    rtol = 1e-9
    if "--rtol" in sys.argv:
        rtol = float(sys.argv[sys.argv.index("--rtol") + 1])
    if cmd == "save":
        save(d)
    elif cmd == "diff":
        sys.exit(1 if diff(d, rtol=rtol,
                           quiet_same="--quiet-same" in sys.argv) else 0)
    else:
        sys.exit(__doc__)
