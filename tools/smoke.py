"""tools/smoke.py - exercise every code path in seconds, not in a quarter hour.

The expensive stages of this project (gen_validation.py, the amplification
database, build_docx.py) take minutes to tens of minutes, and the failures that
matter - a renamed dict key, a changed signature, a NaN escaping a closure, an
array whose length no longer matches its stations - are all detectable in the
first second of the first solve.  Discovering them at minute fourteen has cost
this project whole runs.

This module calls the real functions, at the smallest resolution that still
visits every branch, and asserts the properties that must hold rather than the
numbers, which are the regeneration's job.  It is the gate to run after every
edit and before any full regeneration.

    python3 tools/smoke.py            # all checks
    python3 tools/smoke.py -k kernel  # only checks whose name contains "kernel"

Exit status is the number of failures, so it can gate a commit or a pipeline.
"""
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utss_paths  # noqa: F401  (anchors ROOT and the working directory)

import matplotlib
matplotlib.use("Agg")

_CHECKS = []
_FAILURES = []


def check(fn):
    _CHECKS.append(fn)
    return fn


def _finite(name, a):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a[~np.isnan(a)])):
        raise AssertionError("%s contains inf" % name)


# ----------------------------------------------------------------------
# 1.  imports - the cheapest way to catch a syntax or signature error
# ----------------------------------------------------------------------
@check
def imports_solver():
    """every solver module imports and exposes what its callers use"""
    import case_config as C
    import stability as st
    import utss_solver as U
    for mod, names in ((U, ("march_bl", "solve_airfoil", "solve_flat_plate",
                            "panel_solve", "velocity_field", "CAL")),
                       (st, ("sigma_curve", "twoeq_HL", "H_from_Hstar",
                             "crossflow_factor", "load_database")),
                       (C, ("CRUISE", "CLIMB", "WING", "VALIDATION",
                            "NLF0416", "SWEPT", "SWEPT2"))):
        for n in names:
            assert hasattr(mod, n), "%s lacks %s" % (mod.__name__, n)


@check
def imports_generators():
    """every generator imports without executing its pipeline"""
    for m in ("gen_geometry", "gen_mesh_setup", "gen_postprocessing",
              "gen_equations", "gen_validation", "run_solution",
              "verify_outputs"):
        __import__(m)


# ----------------------------------------------------------------------
# 2.  the stability tables the march reads
# ----------------------------------------------------------------------
@check
def stability_tables():
    """closures are monotone, finite and span the range the march clips to"""
    import stability as st
    Hg, Hs, L, D = st.twoeq_closure()
    assert np.all(np.diff(Hg) > 0), "H grid not increasing"
    assert np.all(np.diff(Hs) < 0), "H* must fall monotonically with H"
    for a, n in ((Hg, "H"), (Hs, "H*"), (L, "l"), (D, "d")):
        _finite(n, a)
    # the inversion the two-equation march depends on
    for h in (2.2, 2.59129, 3.0, 3.5, 3.9):
        assert abs(st.H_from_Hstar(st.twoeq_HL(h)[0]) - h) < 5e-3, \
            "H*(H) inversion off at H=%.3f" % h
    lam, K = st.crossflow_table()
    assert np.all(np.diff(lam) > 0) and np.all(K >= 0.0)


@check
def stability_reverse_branch():
    """the reverse-flow branch is present - the bubble closure depends on it"""
    import stability as st
    eta, prof = st._combined_family()
    Hmax = max(p[0] for p in prof)
    assert Hmax >= st.H_REVERSE - 1e-6, (
        "combined family reaches only H=%.3f but H_REVERSE=%.3f; the reverse "
        "branch failed to build and every bubble length is silently wrong"
        % (Hmax, st.H_REVERSE))
    s = st.sigma_curve(st.H_REVERSE, 400.0)
    assert s.max() > 0.02, "reverse-flow amplification rate collapsed to %.4g" % s.max()


@check
def amplification_database():
    """the tabulated rates are loadable, finite and amplify where they must"""
    import stability as st
    H, R, O, S = st.load_database()
    assert S.shape == (H.size, R.size, O.size)
    _finite("sigma", S)
    assert st.sigma_curve(2.59129, 150.0).max() <= 0.0, \
        "Blasius amplifies below its neutral point"
    assert st.sigma_curve(2.59129, 600.0).max() > 0.0, \
        "Blasius does not amplify well above its neutral point"


# ----------------------------------------------------------------------
# 3.  the inviscid solution
# ----------------------------------------------------------------------
@check
def panel_solution():
    """panel method: closed loop, stagnation pressure, compressibility"""
    import case_config as C
    from utss_solver import panel_solve
    X, Y = C.nlf16_panel_points(60)
    assert abs(X[0] - X[-1]) < 1e-12 and abs(Y[0] - Y[-1]) < 1e-12, "loop not closed"
    xc, yc, Cp, V, th, S = panel_solve(X, Y, 2.0)
    _finite("Cp", Cp); _finite("V", V)
    assert 0.9 < Cp.max() <= 1.0001, "stagnation Cp = %.4f" % Cp.max()
    # incompressible identity, and Karman-Tsien must raise the suction peak
    assert abs(abs(V).max() - np.sqrt(1.0 - Cp.min())) < 1e-9
    Cp_c = panel_solve(X, Y, 2.0, mach=0.42)[2]
    assert Cp_c.min() < Cp.min(), "compressibility did not deepen the suction peak"


@check
def offbody_field_matches_surface():
    """the off-body field reproduces the surface solution just off the wall"""
    import case_config as C
    from utss_solver import panel_solve, velocity_field
    X, Y = C.nlf16_panel_points(60)
    xc, yc, Cp, V, th, S = panel_solve(X, Y, 1.5)
    d = 0.004
    Px = xc - np.sin(th)*d; Py = yc + np.cos(th)*d
    Cf = velocity_field(X, Y, 1.5, Px, Py, U=1.0)[2]
    e = float(np.mean(np.abs(Cf[8:-8] - Cp[8:-8])))
    assert e < 0.06, "off-body vs surface mean |dCp| = %.4f" % e


# ----------------------------------------------------------------------
# 4.  the boundary-layer march and every transition mechanism
# ----------------------------------------------------------------------
def _plate(**kw):
    from utss_solver import solve_flat_plate
    return solve_flat_plate(kw.pop("L", 1.7), kw.pop("U", 5.4),
                            kw.pop("nu", 1.5e-5), kw.pop("Tu", 3.043),
                            npts=kw.pop("npts", 220), **kw)


@check
def march_output_contract():
    """every consumer key is present, right length, and free of sentinels"""
    r = _plate()
    n = len(r["s"])
    need = ("s Ue theta H Cf Re_theta lam gamma state Re_theta_t n_factor "
            "n_crit mechanism i_tr i_sep x_tr onset_mech H_lam theta_lam "
            "H_turb theta_turb bubble_burst").split()
    for k in need:
        assert k in r, "march_bl output lost key %r" % k
    for k in ("s Ue theta H Cf Re_theta lam gamma Re_theta_t n_factor n_crit "
              "state mechanism H_lam theta_lam H_turb theta_turb").split():
        assert len(r[k]) == n, "%s has length %d, stations %d" % (k, len(r[k]), n)
    for k in ("theta", "H", "Cf", "Re_theta", "gamma", "n_factor", "n_crit"):
        _finite(k, r[k])
    rt = np.asarray(r["Re_theta_t"], float)
    assert not np.any(rt[~np.isnan(rt)] >= 1e8), \
        "the 1e9 'branch inactive' sentinel escaped into Re_theta_t"
    assert np.all(np.asarray(r["gamma"]) >= 0.0) and np.all(np.asarray(r["gamma"]) <= 1.0)
    assert np.all(np.asarray(r["H"])[:r["i_tr"] or n] > 1.0), "shape factor below unity"


@check
def kernel_every_mechanism_fires():
    """each of the four mechanisms is reachable and labels itself"""
    import case_config as C
    from utss_solver import solve_airfoil
    got = {}
    # bypass: high free-stream turbulence on a plate
    got["bypass"] = _plate(Tu=3.043, L_turb=1.53e-3)["onset_mech"]
    # natural / TS: a quiet plate
    got["TS-natural"] = _plate(L=4.0, U=27.0, Tu=0.03, npts=320)["onset_mech"]
    # separation: the T3C4 rig geometry, adverse gradient at low speed
    got["separation"] = _plate(L=1.5, U=1.51, Tu=2.11, npts=320,
                               dUe=-0.35)["onset_mech"]
    # cross-flow: a swept section
    X, Y = C.nlf16_panel_points(80)
    r = solve_airfoil(X, Y, -2.0, 40.0, 1.5e-5, 1.83, 0.02, sweep_deg=45.0)
    got["crossflow"] = r["surfaces"]["upper"]["onset_mech"]
    for want, seen in got.items():
        assert seen == want, "expected %s, kernel selected %r" % (want, seen)


@check
def kernel_continuous_in_turbulence():
    """the answer is continuous in Tu across the natural/bypass handover"""
    import case_config as C
    from utss_solver import solve_airfoil
    cr, W = C.CRUISE, C.WING
    npan = 80
    X, Y = C.nlf16_panel_points(npan)

    def sweep(n, lo=0.08, hi=0.30):
        tus = np.linspace(lo, hi, n)
        out = []
        for tu in tus:
            r = solve_airfoil(X, Y, cr["alpha_deg"], cr["U_inf"], cr["nu_inf"],
                              W["MAC"], float(tu), sweep_deg=W["le_sweep_deg"],
                              mach=cr["mach"])
            u = r["surfaces"]["upper"]["x_tr_chord"]
            out.append(1.0 if u != u else float(u))
        return tus, np.array(out)

    # transition may only move forward as the stream gets noisier
    tus, xtr = sweep(23)
    assert np.all(np.diff(xtr) <= 1e-9), \
        "transition moves aft with rising Tu: %s" % np.array2string(xtr, precision=3)

    # Continuity is tested by REFINEMENT, not by an absolute threshold.  The
    # predicted transition location can only land on a panel, so the sampled
    # jump can never fall below the panel spacing however smooth the model is;
    # an absolute bound therefore measures the grid, not the physics.  Halving
    # the sampling halves the jump for a continuous function and leaves it
    # unchanged for a step.  The old gate produced a 0.17c step here that did
    # not refine away at all.
    j1 = float(np.max(np.abs(np.diff(sweep(12)[1]))))
    j2 = float(np.max(np.abs(np.diff(xtr))))
    xs = np.sort(np.abs(np.asarray(C.nlf16_coords(npan)["xu"])))
    spacing = float(np.max(np.diff(xs)[(xs[:-1] > 0.2) & (xs[:-1] < 0.6)]))
    assert j2 <= max(0.55*j1, 1.6*spacing) + 1e-9, (
        "the jump in transition location did not refine away: %.4fc at "
        "dTu = %.4f, %.4fc at half that, against a panel spacing of %.4fc. "
        "A step that survives refinement is a discontinuity in the model."
        % (j1, (tus[1]-tus[0])*2, j2, spacing))


@check
def bubble_closure():
    """a separating plate forms a bubble, marches it, and closes it"""
    r = _plate(L=1.5, U=1.51, Tu=2.11, npts=320, dUe=-0.35)
    assert r["i_sep"] is not None, "no separation on a strongly decelerated plate"
    assert r["onset_mech"] == "separation"
    i0, i1 = r["i_sep"], r["i_tr"]
    assert i1 > i0, "bubble closed before it opened"
    th = np.asarray(r["theta"], float)
    assert th[i1] >= th[i0]*0.99, "momentum thickness fell across the dead air"
    assert np.all(np.asarray(r["Cf"])[i0:i1] <= 1e-12), "wall shear inside the bubble"


@check
def bubble_can_reattach_laminar():
    """a bubble in a gradient that recovers reattaches instead of absorbing"""
    from utss_solver import solve_flat_plate
    # decelerate to separate, then accelerate: the classic short bubble in a
    # gradient that relaxes.  Before laminar reattachment existed the layer
    # could only transition or be declared burst at the trailing edge.
    n = 600
    s = np.linspace(1e-4, 1.5, n)
    Ue = 1.9 - 0.9*np.exp(-((s - 0.75)/0.16)**2)     # dip and recovery
    from utss_solver import march_bl
    r = march_bl(s, Ue, 1.5e-5, Tu_pct=0.05)
    # i_sep is the sticky record of the first separation and must survive the
    # reattachment that clears the live bubble state
    assert r["i_sep"] is not None, "no separation in the decelerating region"
    assert not r["bubble_burst"], "reattached, yet still reported as burst"
    assert r.get("n_reattach", 0) >= 1, (
        "the bubble never reattached even though the gradient recovers; the "
        "separated state is still absorbing")
    # the amplification accumulated across the dead air must not be thrown away
    nf = np.asarray(r["n_factor"], float)
    i0 = r["i_sep"]
    assert np.nanmax(nf[i0:]) > 0.0, "amplification vanished at reattachment"
    tail = nf[min(i0 + 40, n - 1):]
    assert tail.size and np.nanmax(tail) >= 0.9*np.nanmax(nf[:i0 + 40]), (
        "the amplification factor collapsed after reattachment: the bubble's "
        "own N was discarded instead of being carried into the attached "
        "integral")


@check
def compressible_closures():
    """raising the Mach number delays transition and thins nothing to zero"""
    import case_config as C
    from utss_solver import solve_airfoil
    cr, W = C.CRUISE, C.WING
    X, Y = C.nlf16_panel_points(80)
    out = {}
    for M in (0.0, 0.42):
        r = solve_airfoil(X, Y, cr["alpha_deg"], cr["U_inf"], cr["nu_inf"],
                          W["MAC"], cr["Tu_pct"], sweep_deg=W["le_sweep_deg"],
                          mach=M)
        out[M] = r
        _finite("Cd", [r["Cd"]]); _finite("Cl", [r["Cl"]])
        assert r["Cd"] > 0.0
    assert out[0.42]["Cl"] > out[0.0]["Cl"], "compressibility did not raise the lift"


@check
def airfoil_output_contract():
    """solve_airfoil returns what run_solution and gen_validation read"""
    import case_config as C
    from utss_solver import solve_airfoil
    X, Y = C.nlf16_panel_points(80)
    r = solve_airfoil(X, Y, 1.5, 120.0, 4.0e-5, 1.9, 0.07,
                      sweep_deg=12.0, mach=0.42)
    for k in ("panel", "surfaces", "Cl", "Cd", "alpha", "theta_te_c"):
        assert k in r
    for surf in ("upper", "lower"):
        s = r["surfaces"][surf]
        for k in ("x", "y", "Cp", "Re_x", "x_tr_chord", "x_sep_chord",
                  "bubble_burst", "theta_te_c"):
            assert k in s, "surface dict lost %r" % k
        assert len(s["x"]) == len(s["s"]) == len(s["Cf"])


# ----------------------------------------------------------------------
# 5.  the drivers, at the smallest sweep that still writes every column
# ----------------------------------------------------------------------
@check
def lifting_line_closed_form():
    """the elliptic planform check the pipeline asserts before writing"""
    import run_solution as R
    assert R._lifting_line_check() is True
    ll = R.lifting_line()
    for k in ("CL", "CDi", "e", "a0", "alpha_L0", "sweep_c4"):
        assert k in ll and np.isfinite(ll[k])
    assert 0.0 < ll["e"] <= 1.0


@check
def verify_outputs_contract():
    """verify_outputs can still find every CSV and column it reads"""
    import verify_outputs as V
    want, present, absent = V.checks()
    assert len(want) > 15
    for label, value, anchor in want:
        assert isinstance(value, str) and value, "%s has no value" % label
        assert anchor is None or isinstance(anchor, str)
    # the matcher itself: a number must not pass on a longer one containing it
    txt = V.flatten("Table 14. NLF vs fully-turbulent drag. 1683 and 0.1685")
    assert not V.find_value(txt, "168", "NLF vs fully-turbulent drag")[0], \
        "168 matched inside 1683 - the substring bug is back"
    assert V.find_value(V.flatten("NLF vs fully-turbulent drag. 168.0"),
                        "168", "NLF vs fully-turbulent drag")[0], \
        "168 did not match 168.0"


def main():
    only = None
    if "-k" in sys.argv:
        only = sys.argv[sys.argv.index("-k") + 1]
    t0 = time.time()
    run = 0
    for fn in _CHECKS:
        if only and only not in fn.__name__:
            continue
        run += 1
        t = time.time()
        try:
            fn()
            print("  ok    %-34s %6.2f s   %s"
                  % (fn.__name__, time.time() - t, fn.__doc__ or ""))
        except Exception as e:                      # noqa: BLE001
            _FAILURES.append((fn.__name__, e, traceback.format_exc()))
            print("  FAIL  %-34s %6.2f s   %s"
                  % (fn.__name__, time.time() - t, e))
    print("\n%d check(s), %d failed, %.1f s total"
          % (run, len(_FAILURES), time.time() - t0))
    for name, _, tb in _FAILURES:
        print("\n--- %s ---\n%s" % (name, tb))
    return len(_FAILURES)


if __name__ == "__main__":
    sys.exit(main())
