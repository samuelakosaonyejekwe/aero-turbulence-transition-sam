"""tools/pipeline.py - run the regeneration as declared stages, not as a list.

The README's reproduce block is seven commands run by hand in order.  Run it
that way and three things go wrong: a stage that fails leaves the tree half
regenerated with no record of where it stopped; stages that do not depend on
each other are run one after another on a machine with two physical cores; and
a mistake in stage one is not discovered until stage seven, twenty minutes
later.

This driver declares what each stage needs and what it produces, runs the
independent ones concurrently up to a worker cap, times every stage, and stops
at the first failure with the log of the stage that failed.  It runs
tools/smoke.py first by default, because seven seconds spent there is cheaper
than any of the failures it catches.

    python3 tools/pipeline.py                    # everything, smoke-gated
    python3 tools/pipeline.py --stages solution,validation
    python3 tools/pipeline.py --jobs 1           # strictly sequential
    python3 tools/pipeline.py --no-smoke --dry-run
    python3 tools/pipeline.py --list

Worker processes are pinned to one BLAS thread each: the matrices here are a
few hundred square, so threaded BLAS buys nothing and oversubscribing two
physical cores with four threaded workers makes the whole run slower.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utss_paths  # noqa: F401
from utss_paths import ROOT

LOGDIR = os.path.join(ROOT, ".pipeline")
TIMINGS = os.path.join(LOGDIR, "timings.json")

# name -> (script, [stage dependencies], "what it writes")
STAGES = {
    "geometry":   ("gen_geometry.py",       [],  "01_geometry/"),
    "mesh":       ("gen_mesh_setup.py",     [],  "02_mesh/, 03_model_setup/"),
    "solution":   ("run_solution.py",       [],  "04_solution/"),
    "validation": ("gen_validation.py",     [],  "06_validation/"),
    "equations":  ("gen_equations.py",      [],  "07_equations/, model.equations.docx"),
    "post":       ("gen_postprocessing.py", ["solution"], "05_postprocessing/"),
    "docx":       ("build_docx.py",         ["geometry", "mesh", "solution",
                                             "validation", "equations", "post"],
                   "case.docx"),
    "verify":     ("verify_outputs.py",     ["docx"], "(checks only)"),
}

# Heaviest first, so a two-worker pool is never left holding one long stage at
# the end while the other sits idle.  Measured on the reference machine.
_WEIGHT = {"validation": 100, "solution": 40, "post": 25, "mesh": 12,
           "geometry": 8, "docx": 8, "equations": 3, "verify": 3}


def _physical_cores():
    n = os.cpu_count() or 1
    try:
        with open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") as fh:
            sibs = len([s for s in fh.read().strip().replace("-", ",").split(",") if s])
        return max(1, n // max(1, sibs))
    except OSError:
        return max(1, n // 2)


def _env(jobs):
    e = dict(os.environ)
    # one BLAS thread per worker; see the module docstring
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        e[k] = "1"
    e["MPLBACKEND"] = "Agg"
    # A stage may fan out internally - gen_validation runs its four ablation
    # configurations concurrently - and that pool must not be sized as though
    # it had the machine to itself, or two stages each spawning two workers
    # oversubscribe two physical cores twice over.  The budget is divided here
    # and gen_validation reads it.
    e["UTSS_JOBS"] = str(max(1, _physical_cores() // max(1, jobs)))
    return e


def run_stage(name, args, env):
    script, _, _ = STAGES[name]
    log = os.path.join(LOGDIR, name + ".log")
    extra = []
    if name == "validation" and args.no_ablations:
        extra = ["--no-ablations"]
    t0 = time.time()
    with open(log, "w") as fh:
        # -u so the log fills as the stage runs.  Python buffers stdout when
        # it is a file rather than a terminal, so without this a stage's log is
        # empty until it exits - which is exactly when you want to look at it,
        # and exactly when looking is no longer useful.
        p = subprocess.run([sys.executable, "-u", os.path.join(ROOT, script)] + extra,
                           cwd=ROOT, env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return name, p.returncode, dt, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="", help="comma-separated subset")
    ap.add_argument("--jobs", type=int, default=2,
                    help="concurrent stages (default 2: the physical core count)")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--no-ablations", action="store_true",
                    help="pass --no-ablations to gen_validation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, (s, d, o) in STAGES.items():
            print("  %-11s %-24s needs %-38s -> %s"
                  % (n, s, ",".join(d) or "-", o))
        return 0

    want = [s.strip() for s in args.stages.split(",") if s.strip()] or list(STAGES)
    for s in want:
        if s not in STAGES:
            sys.exit("unknown stage %r; --list to see them" % s)
    # pull in dependencies of what was asked for
    pending = set(want)
    while True:
        add = {d for s in pending for d in STAGES[s][1]} - pending
        if not add:
            break
        pending |= add

    os.makedirs(LOGDIR, exist_ok=True)
    if not args.no_smoke and not args.dry_run:
        t = time.time()
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "tools/smoke.py")],
                            cwd=ROOT, env=_env(1)).returncode
        print("smoke: %s in %.1f s\n" % ("PASS" if rc == 0 else "FAIL", time.time()-t))
        if rc:
            print("refusing to start a regeneration on a failing smoke test")
            return rc

    done, times, failed = set(), {}, None
    env = _env(args.jobs)
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        while pending and failed is None:
            ready = sorted([s for s in pending
                            if all(d in done for d in STAGES[s][1])],
                           key=lambda s: -_WEIGHT.get(s, 1))
            if not ready:
                failed = "dependency cycle among %s" % sorted(pending)
                break
            if args.dry_run:
                print("wave: %s" % ", ".join(ready))
                done |= set(ready); pending -= set(ready)
                continue
            futs = [pool.submit(run_stage, s, args, env) for s in ready[:args.jobs]]
            for f in futs:
                name, rc, dt, log = f.result()
                times[name] = dt
                print("  %-11s %s  %6.1f s   (%s)"
                      % (name, "ok  " if rc == 0 else "FAIL", dt, log))
                if rc:
                    failed = name
                else:
                    done.add(name)
                pending.discard(name)

    total = time.time() - t_all
    if times:
        os.makedirs(LOGDIR, exist_ok=True)
        prev = {}
        if os.path.exists(TIMINGS):
            try:
                prev = json.load(open(TIMINGS))
            except Exception:                        # noqa: BLE001
                prev = {}
        prev.update(times)
        json.dump(prev, open(TIMINGS, "w"), indent=1, sort_keys=True)
    print("\ntotal %.1f s (%.1f min) over %d stage(s), %d worker(s)"
          % (total, total/60.0, len(times), args.jobs))
    if failed:
        print("FAILED at stage %r - see %s/%s.log" % (failed, LOGDIR, failed))
        try:
            tail = open(os.path.join(LOGDIR, failed + ".log")).read()[-2500:]
            print("\n--- tail of %s.log ---\n%s" % (failed, tail))
        except Exception:                            # noqa: BLE001
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
