"""
stability.py - linear stability of the laminar boundary layer.

The transition kernel originally advanced the amplification factor with the
Drela-Giles envelope, a closure in which the growth rate dn/dRe_theta is a
function of the shape factor alone.  The envelope is a fit to the maximum over
frequency of the true amplification rate, and it is accurate where the boundary
layer it was fitted to is representative - mild gradients, Falkner-Skan-like
profiles.  It is least accurate in strong adverse gradients, where it keeps
integrating the peak growth rate of a frequency that the real boundary layer
has already left behind, and transition is then predicted too early.

This module replaces that fit with the quantity it approximates.  Falkner-Skan
profiles are computed directly, the Orr-Sommerfeld eigenvalue problem is solved
on each of them by Chebyshev collocation, and the spatial amplification rate is
recovered from the temporal one through Gaster's transformation.  The result is
tabulated as

    sigma(H, Re_theta, omega*) = -alpha_i * theta,

the dimensionless spatial growth rate, with omega* = omega*theta/U_e the
dimensionless frequency.  The boundary-layer march then integrates one
amplification factor per physical frequency and transitions on the envelope of
the integrated curves, which is the e^N method as originally defined rather
than a correlation of it.

References
  Orr (1907); Sommerfeld (1908).
  Gaster M. (1962), "A note on the relation between temporally-increasing and
    spatially-increasing disturbances in hydrodynamic stability", JFM 14, 222.
  Jordinson R. (1970), "The flat plate boundary layer.  Part 1", JFM 43, 801.
  Mack L.M. (1977), "Transition prediction and linear stability theory",
    AGARD CP-224.
  Drela M. & Giles M.B. (1987), AIAA J. 25(10), 1347  - the envelope this
    module supersedes.
"""
import os
import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import eig

# numpy renamed trapz to trapezoid in 2.0 and deprecated the old spelling, which
# is slated for removal.  Binding the name once here keeps every quadrature in
# this module running on both, rather than pinning the project to a numpy that
# still carries the deprecated alias.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ----------------------------------------------------------------------
# Falkner-Skan similarity profiles
# ----------------------------------------------------------------------
# The similarity equation f_eta_eta_eta + f f_eta_eta + beta(1 - f_eta^2) = 0,
# with f = f_eta = 0 at the wall and f_eta -> 1 in the free stream.
# The family is generated once by continuation, sweeping the Hartree parameter
# from a strong favourable gradient down to separation and seeding each solve
# with the previous solution.  Solved cold, the boundary-value problem fails to
# converge at scattered values of beta; seeded, every member converges in a few
# iterations, and the sweep also yields the monotone map beta -> H needed to
# obtain a profile at a prescribed shape factor.
_ETA_MAX = 16.0
_ETA_N = 641
_FS_FAMILY = None


def _fs_solve(beta, eta, guess):
    def rhs(t, y):
        return np.vstack([y[1], y[2], -y[0]*y[2] - beta*(1.0 - y[1]**2)])

    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[1] - 1.0])

    return solve_bvp(rhs, bc, eta, guess, tol=1e-9, max_nodes=200000)


def _fs_third(beta, f, fp, fpp):
    """f'''(eta) from the similarity equation itself, not by differencing.

    The Orr-Sommerfeld operator needs U'' - it is the term that carries the
    inflectional instability, and it is what decides the amplification rate on
    exactly the adverse-gradient profiles this database exists to tabulate.

    It used to be formed as `Upp = D1 @ dU`, a Chebyshev derivative matrix
    applied to the similarity f'' after that had been LINEARLY interpolated
    onto the collocation grid.  Spectral differentiation is only spectrally
    accurate on a smooth function; applied to a piecewise-linear interpolant it
    differentiates the interpolation kinks, and the Chebyshev matrix amplifies
    them by O(N^2).  The correct value is available in closed form, because
    f''' is what the Falkner-Skan equation gives:

        f''' + f f'' + beta (1 - f'^2) = 0.

    So f''' is stored with the family, interpolated like f' and f'', and U'' on
    the collocation grid follows by the chain rule with no differentiation at
    all.  The cached families carry it; a cache written before this change
    lacks the key and is rebuilt rather than silently reused.
    """
    return -f*fpp - beta*(1.0 - fp*fp)


def _fs_family():
    """All Falkner-Skan profiles, favourable through separation, by continuation."""
    global _FS_FAMILY
    if _FS_FAMILY is not None:
        return _FS_FAMILY
    # beta is clustered towards the separation value: H varies slowly in a
    # favourable gradient but diverges as beta approaches -0.198838, so a grid
    # uniform in beta leaves gaps of 0.12 in H near separation while wasting
    # points where the profile barely changes.  Spacing the adverse branch
    # geometrically in the distance from separation holds dH below 0.02
    # throughout.
    betas = np.concatenate([np.linspace(3.0, -0.05, 110),
                            -0.198838 + np.geomspace(0.148838, 2.0e-5, 260)[1:]])
    eta = np.linspace(0.0, _ETA_MAX, _ETA_N)
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "falkner_skan_family.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if "upp" in d.files:            # see _fs_third: caches without it are stale
            fam = [(float(b), float(h), float(t), uu, pp, qq)
                   for b, h, t, uu, pp, qq
                   in zip(d["beta"], d["H"], d["theta"], d["u"], d["up"],
                          d["upp"])]
            _FS_FAMILY = (d["eta"], fam)
            return _FS_FAMILY
    g = np.zeros((3, eta.size))
    g[0] = eta - np.tanh(eta); g[1] = np.tanh(eta); g[2] = 1.0/np.cosh(eta)**2
    fam = []
    for b in betas:
        sol = _fs_solve(b, eta, g)
        if not sol.success:
            continue
        g = sol.sol(eta)
        f, u, up = g
        ds = _trapz(1.0 - u, eta); th = _trapz(u*(1.0 - u), eta)
        fam.append((float(b), float(ds/th), float(th), u.copy(), up.copy(),
                    _fs_third(b, f, u, up)))
    H = np.array([m[1] for m in fam])
    keep = np.concatenate([[True], np.diff(H) > 1e-9])
    fam = [m for m, k in zip(fam, keep) if k]
    np.savez_compressed(cache, eta=eta,
                        beta=np.array([m[0] for m in fam]),
                        H=np.array([m[1] for m in fam]),
                        theta=np.array([m[2] for m in fam]),
                        u=np.array([m[3] for m in fam]),
                        up=np.array([m[4] for m in fam]),
                        upp=np.array([m[5] for m in fam]))
    _FS_FAMILY = (eta, fam)
    return _FS_FAMILY


def falkner_skan(beta, eta_max=_ETA_MAX, n=_ETA_N):
    """Single Falkner-Skan solution, seeded from the nearest family member."""
    eta, fam = _fs_family()
    k = int(np.argmin([abs(m[0] - beta) for m in fam]))
    g = np.vstack([np.zeros_like(eta), fam[k][3], fam[k][4]])
    g[0] = np.concatenate([[0.0],
                           np.cumsum(0.5*(g[1][1:] + g[1][:-1])*np.diff(eta))])
    sol = _fs_solve(beta, eta, g)
    if not sol.success:
        raise RuntimeError("Falkner-Skan failed at beta=%.5f: %s"
                           % (beta, sol.message))
    f, u, up = sol.sol(eta)
    ds = _trapz(1.0 - u, eta); th = _trapz(u*(1.0 - u), eta)
    return eta, u, up, float(ds/th), float(th)


_H_CACHE = {}


def _combined_family():
    """Attached and reverse-flow branches joined into one monotone H family."""
    eta, fam = _fs_family()
    prof = [(m[1], m[2], m[3], m[4], m[5]) for m in fam]   # H, theta, and the
    #                                            three similarity derivatives
    # The reverse-flow branch is not optional.  The separation-bubble closure
    # reads its amplification rate at H = H_REVERSE = 4.90, and only this
    # branch reaches that far.  Swallowing a failure here, as an earlier
    # version did with a bare `except Exception: pass`, left the family capped
    # at the attached separation profile H = 4.00, roughly halved sigma, and
    # changed every bubble length in the study - with no warning, and with
    # nothing in the self-tests that would have noticed.  If the branch cannot
    # be built, that is a fault to be fixed, not a fallback to be taken.
    rev = fs_reverse_family()
    prof += [(m[2], m[3], m[4], m[5], m[6]) for m in rev if m[2] > prof[-1][0]]
    prof.sort(key=lambda t: t[0])
    keep = [prof[0]]
    for p in prof[1:]:
        if p[0] - keep[-1][0] > 1e-9:
            keep.append(p)
    return eta, keep


_COMB = None


def fs_profile_for_H(H_target):
    """Falkner-Skan profile at a prescribed shape factor, SOLVED not blended.

    H rises monotonically along the combined family, from about 2.13 in a
    strong favourable gradient, through 4.00 at separation, to about 6.4 on the
    reverse-flow branch.  Continuing past separation is what allows the
    amplification rate inside a separation bubble to be read from the same
    table as everywhere else rather than supplied as a separate constant.

    WHY THIS SOLVES RATHER THAN INTERPOLATES.  This used to take the two family
    members bracketing H_target and blend them linearly.  A linear combination
    of two Falkner-Skan profiles is not a Falkner-Skan profile: it has the
    requested H by construction, and it satisfies no similarity equation.  The
    blend is close - its velocity is within 1.3e-4 of the true profile - but
    the Orr-Sommerfeld operator is driven by U'', and U'' comes from f''',
    which the blend gets wrong by 4e-4.

    Measured against the standard Blasius benchmark (Jordinson 1970,
    Re_delta* = 998, alpha delta* = 0.308, c = 0.36412 + 0.00796i), the blended
    profile returns a growth rate 1.20 per cent LOW, and it stays 1.20 per cent
    low however many collocation points are used, however far the outer
    boundary is put, and whether the tabulation is interpolated linearly or by
    cubic - because none of those is the cause.  Solved at the beta that gives
    the requested H, the same code returns the benchmark to 0.02 per cent.

    beta = 0 is not even a node of the family: the grid runs
    ... 0.03394, 0.00596, -0.02202 ... so Blasius itself, the one profile whose
    stability is most heavily published, was never actually in the table.

    The family is still what brackets the root and seeds the solve, so this
    costs one boundary-value solve per DISTINCT shape factor and is memoised.
    Past the fold at H = 4.03 beta is no longer a usable parameter - the two
    branches meet there - so the reverse branch is solved in the wall shear
    f''(0) instead, which is the continuation parameter it was built with.
    """
    key = round(float(H_target), 5)
    if key in _H_CACHE:
        return _H_CACHE[key]
    eta, fam = _fs_family()
    H_att = np.array([m[1] for m in fam])
    b_att = np.array([m[0] for m in fam])
    t = float(H_target)

    def _seeded(guess_u, guess_up):
        f0 = np.concatenate([[0.0],
                             np.cumsum(0.5*(guess_u[1:] + guess_u[:-1])*np.diff(eta))])
        return np.vstack([f0, guess_u, guess_up])

    if t <= H_att[-1]:
        t = float(np.clip(t, H_att[0], H_att[-1]))
        j = int(np.clip(np.searchsorted(H_att, t) - 1, 0, H_att.size - 2))
        lo, hi = float(b_att[j]), float(b_att[j+1])
        guess = _seeded(fam[j][3], fam[j][4])

        def resid(b):
            sol = _fs_solve(b, eta, guess)
            if not sol.success:
                raise RuntimeError("Falkner-Skan failed at beta=%.6f" % b)
            f, u, up = sol.sol(eta)
            return _trapz(1.0 - u, eta)/_trapz(u*(1.0 - u), eta) - t

        b = _bisect(resid, min(lo, hi), max(lo, hi))
        sol = _fs_solve(b, eta, guess)
        f, u, up = sol.sol(eta)
        upp = _fs_third(b, f, u, up)
    else:
        rev = fs_reverse_family()
        H_rev = np.array([m[2] for m in rev])
        fw_rev = np.array([m[0] for m in rev])
        t = float(np.clip(t, H_rev.min(), H_rev.max()))
        k = int(np.argmin(np.abs(H_rev - t)))
        k = int(np.clip(k, 1, len(rev) - 2))
        guess = _seeded(rev[k][4], rev[k][5])
        beta0 = rev[k][1]

        def resid_fw(fw):
            sol = _fs_solve_shear(fw, eta, guess, beta0)
            if not sol.success:
                raise RuntimeError("reverse-branch solve failed at f''(0)=%.6f" % fw)
            f, u, up = sol.sol(eta)
            return _trapz(1.0 - u, eta)/_trapz(u*(1.0 - u), eta) - t

        j = int(np.clip(np.searchsorted(H_rev, t) - 1, 0, H_rev.size - 2))
        fw = _bisect(resid_fw, min(fw_rev[j], fw_rev[j+1]),
                     max(fw_rev[j], fw_rev[j+1]))
        sol = _fs_solve_shear(fw, eta, guess, beta0)
        f, u, up = sol.sol(eta)
        upp = _fs_third(float(sol.p[0]), f, u, up)

    th = _trapz(u*(1.0 - u), eta)
    H_got = _trapz(1.0 - u, eta)/th
    pr = (eta, u, up, float(H_got), float(th), upp)
    _H_CACHE[key] = pr
    return pr


def _bisect(f, lo, hi, tol=1e-9, itmax=60):
    """Bisection on a monotone residual, with the bracket widened if need be."""
    flo, fhi = f(lo), f(hi)
    grow = 0
    while flo*fhi > 0.0 and grow < 6:
        span = hi - lo
        lo -= 0.25*span; hi += 0.25*span
        flo, fhi = f(lo), f(hi)
        grow += 1
    if flo*fhi > 0.0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(itmax):
        mid = 0.5*(lo + hi)
        fm = f(mid)
        if abs(fm) < tol or hi - lo < tol:
            return mid
        if flo*fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5*(lo + hi)


def fs_H_range():
    """The shape-factor interval spanned by the Falkner-Skan family."""
    _, fam = _fs_family()
    return fam[0][1], fam[-1][1]


# ----------------------------------------------------------------------
# Chebyshev differentiation on a stretched half-line
# ----------------------------------------------------------------------
def _cheb(N):
    """Chebyshev-Gauss-Lobatto nodes on [-1,1] and the differentiation matrix."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    j = np.arange(N + 1)
    x = np.cos(np.pi*j/N)
    c = np.ones(N + 1); c[0] = c[N] = 2.0
    c = c*(-1.0)**j
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0/c)/(dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def _grid(N, y_max, y_half):
    """Algebraic map xi in [-1,1] -> y in [0, y_max], half the nodes below y_half."""
    D, xi = _cheb(N)
    a = y_half*y_max/(y_max - 2.0*y_half)
    b = 1.0 + 2.0*a/y_max
    y = a*(1.0 + xi)/(b - xi)                       # y(1)=y_max, y(-1)=0
    dydxi = a*(b + 1.0)/(b - xi)**2
    D1 = D/dydxi[:, None]
    return y, D1


# ----------------------------------------------------------------------
# Orr-Sommerfeld, temporal formulation
# ----------------------------------------------------------------------
def os_temporal(y, D1, U, Upp, alpha, Re):
    """Most unstable temporal eigenvalue c of the Orr-Sommerfeld operator.

        [U(D^2-a^2) - U'' - (1/(i a Re))(D^2-a^2)^2] v = c (D^2-a^2) v

    with v = v' = 0 on both boundaries.  y, U, Upp are non-dimensionalised on
    the momentum thickness and the edge velocity, so Re is Re_theta and alpha
    is alpha*theta.

    Chebyshev collocation of the fourth-order operator produces, alongside the
    physical Tollmien-Schlichting mode, a discretisation of the continuous
    spectrum: a dense family of eigenvalues crowding towards c_r = 1, some of
    which acquire a spurious positive imaginary part and would otherwise be
    returned as the most unstable mode.  A Tollmien-Schlichting wave has a
    phase speed well below the edge velocity - c_r is 0.40 at the Blasius
    critical point and falls with Reynolds number - so restricting the search
    to c_r < 0.95 separates the two cleanly.  A loose check that the
    eigenfunction has decayed by the outer boundary is retained as a guard.
    """
    n = len(y)
    I = np.eye(n)
    D2 = D1 @ D1
    L = D2 - alpha**2*I
    A = np.diag(U) @ L - np.diag(Upp) - (L @ L)/(1j*alpha*Re)
    B = L.astype(complex)
    for row, con in ((0, I[0]), (1, D1[0]), (n - 2, D1[-1]), (n - 1, I[-1])):
        A[row, :] = con
        B[row, :] = 0.0
    w, V = eig(A, B, right=True)
    ok = np.isfinite(w) & (w.real > 0.02) & (w.real < 0.95) & (np.abs(w.imag) < 0.5)
    if not ok.any():
        return None
    idx = np.where(ok)[0]
    # y is ordered from the free stream (y_max) down to the wall
    outer = y > 0.80*y.max()
    keep = []
    for k in idx:
        v = np.abs(V[:, k])
        vm = v.max()
        if vm <= 0.0:
            continue
        if v[outer].max()/vm < 0.25:          # decayed by the outer boundary
            keep.append(k)
    if not keep:
        return None
    keep = np.array(keep)
    return w[keep[np.argmax(w[keep].imag)]]


def _os_profile(profile, N, y_max=60.0, y_half=6.0):
    """(U, U', U'', y, D1) on the collocation grid, in momentum-thickness units.

    One place builds the mean flow the Orr-Sommerfeld operator sees, so the
    tabulated database, the direct growth curve and the spatial-eigenvalue
    check cannot be built on three subtly different profiles - which they were:
    growth_curve() flattened U to 1 above the similarity grid and _sigma_slab()
    did not, and both formed U'' by differentiating an interpolant.

    eta = y*theta_eta, so d/dy = theta_eta d/deta and U'' = f''' theta_eta^2.
    """
    eta, u, up, H, th_eta, upp = profile
    y, D1 = _grid(N, y_max, y_half)
    eta_y = np.clip(y*th_eta, 0.0, eta[-1])
    U = np.interp(eta_y, eta, u)
    dU = np.interp(eta_y, eta, up)*th_eta
    Upp = np.interp(eta_y, eta, upp)*th_eta*th_eta
    # above the top of the similarity grid the profile is free stream: U -> 1
    # with no shear and no curvature.  Clipping eta_y already holds U, U' and
    # U'' at their edge values, which are 1, 0 and 0 to quadrature accuracy;
    # setting them explicitly keeps a truncated grid from leaking a residual
    # curvature into the operator.
    out = y > eta[-1]/th_eta
    U[out] = 1.0; dU[out] = 0.0; Upp[out] = 0.0
    return U, dU, Upp, y, D1


def growth_curve(H_target, Re_theta, alphas, N=110, y_max=60.0, y_half=6.0,
                 profile=None, with_alpha=False):
    """Spatial growth rate against frequency at one (H, Re_theta).

    Returns (omega_r, sigma) with omega_r = alpha*c_r the dimensionless
    frequency and sigma = -alpha_i*theta the spatial amplification rate,
    obtained from the temporal rate by Gaster's transformation
    alpha_i = -omega_i / c_g,  c_g = d(omega_r)/d(alpha).
    """
    if profile is None:
        profile = fs_profile_for_H(H_target)
    U, dU, Upp, y, D1 = _os_profile(profile, N, y_max, y_half)
    eta, u, up, H, th_eta = profile[:5]
    om_r, om_i = [], []
    for a in alphas:
        c = os_temporal(y, D1, U, Upp, a, Re_theta)
        if c is None:
            om_r.append(np.nan); om_i.append(np.nan)
        else:
            om_r.append(a*c.real); om_i.append(a*c.imag)
    om_r = np.array(om_r); om_i = np.array(om_i)
    good = np.isfinite(om_r) & np.isfinite(om_i)
    if good.sum() < 3:
        return ((np.array([]),)*4 if with_alpha else (np.array([]), np.array([])))
    # Sorted by frequency before the group velocity is differenced, and
    # duplicates dropped.  np.gradient assumes its coordinate is monotone and
    # returns nonsense where it is not; omega_r(alpha) is monotone in the
    # amplified band but need not be once a mode is lost and the sweep picks up
    # a neighbouring one, and this function is the reference the tabulated
    # database is checked against.  _sigma_slab has always sorted here; this
    # did not, so the two could disagree - which is exactly the drift the
    # sigma_lookup/sigma_curve comment warns about, in the other direction.
    wr = om_r[good]; wi = om_i[good]; aa = np.asarray(alphas, float)[good]
    k = np.argsort(wr); wr, wi, aa = wr[k], wi[k], aa[k]
    keep = np.concatenate([[True], np.diff(wr) > 1e-12])
    wr, wi, aa = wr[keep], wi[keep], aa[keep]
    if wr.size < 3:
        return ((np.array([]),)*4 if with_alpha else (np.array([]), np.array([])))
    cg = np.gradient(wr, aa)
    cg = np.where(np.abs(cg) < 1e-6, np.nan, cg)
    sigma = wi/cg                      # Gaster: -alpha_i = +omega_i/c_g
    if with_alpha:
        return wr, np.nan_to_num(sigma, nan=0.0), aa, np.nan_to_num(cg, nan=0.0)
    return wr, np.nan_to_num(sigma, nan=0.0)


# ----------------------------------------------------------------------
# Amplification-rate database
# ----------------------------------------------------------------------
# The table is indexed by shape factor, momentum-thickness Reynolds number and
# dimensionless frequency, and holds the spatial amplification rate
# sigma = -alpha_i*theta.  The shape-factor grid runs from H = 2.15 - strongly
# accelerated and effectively stable - PAST the Falkner-Skan separation profile
# (H = 4.00; fs_H_range returns 3.9974) and on along the reverse-flow branch to
# its last node, H = 4.90, which is H_REVERSE, so that a separated shear layer
# reads its amplification rate from the same table as an attached one.  It is
# dense below H = 3.0, where the neutral boundary of the Orr-Sommerfeld problem
# moves quickly and a coarse grid interpolates across it.
#
# This paragraph used to OPEN by saying the range "stops short of the
# Falkner-Skan separation profile ... the eigenvalue problem is stiff there and
# the separation-induced branch of the transition kernel, not the amplification
# integral, is what decides transition in a separating layer", and then CLOSE
# by saying it "now runs past separation onto the reverse-flow branch".  Both
# cannot be true; the second is, because arange(3.00, 4.96, 0.095) ends at
# 4.90.  The first sentence described the table as it stood before the bubble
# closure was written and should have gone with it - that closure reads sigma
# at H_REVERSE out of THIS table, so a range stopping at separation would have
# left it with nothing to read.
H_GRID   = np.concatenate([np.arange(2.15, 3.00, 0.025),
                           np.arange(3.00, 4.96, 0.095)])
RET_GRID = np.geomspace(40.0, 8000.0, 40)
OM_GRID  = np.geomspace(2.0e-3, 1.5e-1, 32)
ALPHAS   = np.geomspace(8.0e-3, 0.45, 28)
NCHEB    = 60
DB_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "amplification_db.npz")


def _sigma_slab(H):
    """sigma(Re_theta, omega) at one shape factor."""
    pr = fs_profile_for_H(H)
    U, dU, Upp, y, D1 = _os_profile(pr, NCHEB, 60.0, 6.0)
    out = np.zeros((RET_GRID.size, OM_GRID.size))
    for i, Re in enumerate(RET_GRID):
        om_r, om_i = [], []
        for a in ALPHAS:
            c = os_temporal(y, D1, U, Upp, a, Re)
            om_r.append(np.nan if c is None else a*c.real)
            om_i.append(np.nan if c is None else a*c.imag)
        om_r = np.array(om_r); om_i = np.array(om_i)
        g = np.isfinite(om_r) & np.isfinite(om_i)
        if g.sum() < 4:
            continue
        wr = om_r[g]; wi = om_i[g]; aa = ALPHAS[g]
        k = np.argsort(wr); wr, wi, aa = wr[k], wi[k], aa[k]
        keep = np.concatenate([[True], np.diff(wr) > 1e-9])
        wr, wi, aa = wr[keep], wi[keep], aa[keep]
        if wr.size < 4:
            continue
        cg = np.gradient(wr, aa)                       # group velocity
        cg = np.where(np.abs(cg) < 1e-4, np.nan, cg)
        sig = wi/cg                                    # Gaster: -alpha_i
        ok = np.isfinite(sig)
        if ok.sum() < 4:
            continue
        out[i] = np.interp(OM_GRID, wr[ok], sig[ok], left=0.0, right=0.0)
    return out


def build_database(path=DB_PATH, nproc=4, verbose=True, resume=True):
    """Generate and store the amplification database.

    Each shape-factor slab is written to a checkpoint directory as it
    completes, and a run that is interrupted resumes from what is already
    there.  The whole table takes a few minutes; losing it to a failure in the
    last slab, which a single blocking map would do, is not worth the tidier
    code.
    """
    from multiprocessing import Pool
    ck = os.path.join(os.path.dirname(os.path.abspath(path)), "_amp_slabs")
    os.makedirs(ck, exist_ok=True)
    todo = []
    for i, H in enumerate(H_GRID):
        f = os.path.join(ck, "slab_%03d.npy" % i)
        if resume and os.path.exists(f):
            continue
        todo.append((i, float(H)))
    if verbose:
        print("amplification database: %d H x %d Re_theta x %d alpha = %d "
              "Orr-Sommerfeld solves; %d of %d slabs still to do"
              % (H_GRID.size, RET_GRID.size, ALPHAS.size,
                 H_GRID.size*RET_GRID.size*ALPHAS.size, len(todo), H_GRID.size))
    if todo:
        with Pool(nproc) as p:
            for i, slab in p.imap_unordered(_slab_job, todo):
                np.save(os.path.join(ck, "slab_%03d.npy" % i), slab)
                if verbose:
                    done = len([f for f in os.listdir(ck) if f.endswith(".npy")])
                    print("  slab %3d done (%d/%d)" % (i, done, H_GRID.size),
                          flush=True)
    sigma = np.array([np.load(os.path.join(ck, "slab_%03d.npy" % i))
                      for i in range(H_GRID.size)])
    np.savez_compressed(path, H=H_GRID, Re_theta=RET_GRID, omega=OM_GRID,
                        sigma=sigma)
    if verbose:
        print("wrote %s   sigma range %.4g .. %.4g"
              % (path, float(sigma.min()), float(sigma.max())))
    return sigma


def _slab_job(arg):
    i, H = arg
    return i, _sigma_slab(H)


_DB = None


def load_database(path=DB_PATH):
    """Load the tabulated amplification rates, building them if absent."""
    global _DB
    if _DB is None:
        if not os.path.exists(path):
            build_database(path)
        d = np.load(path)
        _DB = (d["H"], d["Re_theta"], d["omega"], d["sigma"])
    return _DB


H_REVERSE = 4.90      # developed reverse-flow profile: the aft end of the
                      # tabulated Falkner-Skan branch, where the separated
                      # shear layer's amplification rate is evaluated

# WHERE THE RATE AT H_REVERSE IS CONVERGED, AND WHERE IT IS NOT.
#
# The separation closure and the cross-flow branch both read
# sigma_curve(H_REVERSE, Re_theta), and both rest on the same property: that
# this rate is nearly flat in Reynolds number.  Above Re_theta = 200 it is, and
# it saturates as an inflectional instability must - 0.0417 at 200, 0.0435 at
# 400, 0.0448 at 1000, 0.0461 at 8000.
#
# BELOW about Re_theta = 200 the Orr-Sommerfeld solve on this profile is NOT
# converged.  Across neighbouring RET_GRID nodes from 40 to 136 the tabulated
# rate runs 0.0312, 0.0329, 0.0343, 0.0355, 0.0365, 0.0661, 0.0793, 0.0718,
# 0.0856, 0.0558 - it jumps by a factor of two between adjacent nodes and the
# peak-amplified frequency hops bands.  This is NOT the tabulation: a direct
# eigenvalue sweep at the same shape factor is equally erratic (0.1035 at
# Re_theta = 40, 0.0356 at 61, 0.0919 at 94, 0.0751 at 144), so it is the
# eigenvalue problem itself.  A deep reverse-flow profile at low Reynolds
# number is a stiff operator and the mode filter of os_temporal is not
# separating the physical mode from the discretised continuous spectrum there.
#
# IT REACHES NO PUBLISHED RESULT, and that is measured rather than assumed.
# Instrumenting every sigma_curve call at this shape factor: the aerofoil
# bubbles read Re_theta 426 to 1020, the T3C4 plate 256 to 959, the Boltz
# swept sections 934 upwards.  The Dagenhart sections are the only case that
# reads lower - down to 122, in the leading-edge bubble that forms before the
# cross-flow integral starts - and clamping the rate at Re_theta = 200 there
# leaves all six transition locations, all six selected mechanisms and the
# 21.8 per cent mean error bit-identical.  The reads happen upstream of
# anything that decides an answer.
#
# So the floor is documented rather than imposed: imposing one would be a
# constant with no effect, and tools/smoke.py holds the property the model
# actually leans on - that the rate is monotone and saturating over the range
# it is read in.  If a rebuilt database ever moved a bubble down into this
# region, that check is where it would show.


def sigma_curve(H, Re_theta):
    """Amplification rate against dimensionless frequency at one (H, Re_theta).

    Bilinear in the shape factor and the momentum-thickness Reynolds number,
    returning the whole frequency curve at once so that a boundary-layer march
    can advance every tracked frequency with a single table access.
    """
    Hs, Rs, Os, S3 = load_database()
    H = float(np.clip(H, Hs[0], Hs[-1]))
    R = float(np.clip(Re_theta, Rs[0], Rs[-1]))
    i = int(np.clip(np.searchsorted(Hs, H) - 1, 0, Hs.size - 2))
    j = int(np.clip(np.searchsorted(Rs, R) - 1, 0, Rs.size - 2))
    th = (H - Hs[i])/(Hs[i+1] - Hs[i])
    tr = (R - Rs[j])/(Rs[j+1] - Rs[j])
    return ((1-th)*(1-tr)*S3[i, j] + th*(1-tr)*S3[i+1, j] +
            (1-th)*tr*S3[i, j+1] + th*tr*S3[i+1, j+1])


def sigma_lookup(H, Re_theta, omega):
    """Trilinear interpolation of sigma = -alpha_i*theta.

    H and Re_theta are clamped to the tabulated range, which runs from H = 2.15
    - strongly accelerated and effectively stable - past the separation profile
    and onto the reverse-flow branch, whose last node is H_REVERSE = 4.90, so
    that a detached shear layer reads its rate from the same table as an
    attached one.  (H_GRID's adverse leg is arange(3.00, 4.96, 0.095), whose
    last member is 4.90; this said 4.96, which is the open end of that range
    and not a tabulated shape factor.)  omega outside the
    tabulated band returns zero, which is correct - those frequencies are not
    amplified.

    This is the single-frequency form, for interrogating the table directly.
    It is a linear interpolation along the frequency axis of sigma_curve(),
    which is what the boundary-layer march reads, rather than a second
    implementation of the same trilinear interpolation - two copies of it were
    free to drift apart, and the march would have kept the one that was wrong.
    """
    _, _, Os, _ = load_database()
    if omega < Os[0] or omega > Os[-1]:
        return 0.0
    return float(np.interp(omega, Os, sigma_curve(H, Re_theta)))


def tabulated_neutral_Re_theta(H=2.59129, hi=400.0, step=0.25):
    """Where the INTERPOLATED table first amplifies, and the nodes bracketing it.

    Returns (Re_theta, node_below, node_above).  neutral_Re_theta() measures the
    eigenvalue solver by a direct sweep and returns 200.46 against the accepted
    200.5 - not the 201 this docstring used to give.  THIS function measures
    something else: what the boundary-layer march
    actually reads, which is the bilinear interpolation of RET_GRID.  The two
    differ because the last node below the crossing holds an exact zero, so the
    interpolant cannot turn positive until it has climbed away from that node.

    The value is NOT a node: RET_GRID has nodes at 204.21 and 233.92 and
    nothing between them.  Three places in this project described it as "the
    nearest node of the Reynolds-number grid, Re_theta = 210", which is wrong
    in both the number and the description; they now read this instead, and it
    is deliberately not restated in prose anywhere.
    """
    Rs = load_database()[1]
    for x in np.arange(float(Rs[0]), float(hi), float(step)):
        if sigma_curve(H, float(x)).max() > 0.0:
            j = int(np.searchsorted(Rs, x))
            return (float(x), float(Rs[max(j - 1, 0)]),
                    float(Rs[min(j, Rs.size - 1)]))
    return (float("nan"), float("nan"), float("nan"))


def omega_grid_bounds():
    """Tabulated range of the dimensionless frequency omega*theta/U_e."""
    _, _, Os, _ = load_database()
    return float(Os[0]), float(Os[-1])


# ----------------------------------------------------------------------
# Exact laminar closure from the Falkner-Skan family
# ----------------------------------------------------------------------
# Thwaites' method needs two closure functions, the shape factor H(lambda) and
# the shear function l(lambda) = Re_theta*C_f/2.  Both are usually taken from
# the algebraic fits Thwaites published in 1949.  They are fits to the
# Falkner-Skan family, which is computed here anyway, so the fits can be
# dispensed with and the family used directly.  In similarity variables
#
#     lambda = beta * theta_eta^2 ,      l = theta_eta * f''(0) ,
#
# with theta_eta the momentum thickness in similarity units.  The relations
# reproduce Thwaites' own values where his fit is good - l = 0.2205 against
# 0.220 at lambda = 0 - and depart from it where it is not.  The important
# departure is near separation: the fit returns H = 3.10 at its separation
# value lambda = -0.090, whereas the exact family reaches H = 4.00 there, and
# separation itself occurs at lambda = -0.0681.  That matters because the
# amplification rate sigma_lookup returns rises by AT LEAST a factor of eight
# between the Blasius profile and H = 3.9 - eighteen at Re_theta = 300, eight
# to nine from 1000 up, and never the "factor of seven" this comment used to
# give, which is below the ratio at every Reynolds number in the table - so a
# shape factor short by 0.9 near separation starves the amplification integral
# in exactly the adverse gradients where transition is decided.
_CLOSURE = None


def thwaites_closure():
    """Exact (lambda, H, l) closure table, ordered by increasing lambda."""
    global _CLOSURE
    if _CLOSURE is None:
        eta, fam = _fs_family()
        lam = np.array([b*th*th for b, H, th, u, up, _q in fam])
        H = np.array([m[1] for m in fam])
        l = np.array([m[2]*m[4][0] for m in fam])
        k = np.argsort(lam)
        lam, H, l = lam[k], H[k], l[k]
        keep = np.concatenate([[True], np.diff(lam) > 1e-12])
        _CLOSURE = (lam[keep], H[keep], l[keep])
    return _CLOSURE


def lambda_sep():
    """Value of the Thwaites parameter at which the exact family separates."""
    return float(thwaites_closure()[0][0])


def closure_HL(lam):
    """Shape factor and shear function at a given Thwaites parameter."""
    L, H, l = thwaites_closure()
    x = float(np.clip(lam, L[0], L[-1]))
    return float(np.interp(x, L, H)), float(np.interp(x, L, l))


# ----------------------------------------------------------------------
# Reverse-flow (lower) branch of the Falkner-Skan family
# ----------------------------------------------------------------------
# For -0.198838 < beta < 0 the similarity equation has two solutions: the
# attached upper branch, with positive wall shear, and a lower branch carrying
# a region of reverse flow next to the wall.  The lower branch is the profile
# family of the interior of a laminar separation bubble, and it is what the
# amplification rate inside a bubble should be read from.  Parametrising by
# beta cannot reach it, because the two branches meet at a fold; the family is
# therefore continued in the wall shear f''(0) instead, with beta carried as an
# unknown eigenvalue of the boundary-value problem.  Driving f''(0) from
# positive values through zero and negative continues smoothly around the fold
# and onto the reverse-flow branch.
def _fs_solve_shear(fw, eta, guess, beta_guess):
    """Falkner-Skan solution with the wall shear prescribed and beta unknown."""
    def rhs(t, y, p):
        b = p[0]
        return np.vstack([y[1], y[2], -y[0]*y[2] - b*(1.0 - y[1]**2)])

    def bc(ya, yb, p):
        return np.array([ya[0], ya[1], yb[1] - 1.0, ya[2] - fw])

    return solve_bvp(rhs, bc, eta, guess, p=[beta_guess], tol=1e-9,
                     max_nodes=200000)


_FS_REVERSE = None


def fs_reverse_family(fw_min=-0.120, n=150):
    """Profiles from separation into reverse flow, ordered by decreasing shear.

    Returns a list of (f''(0), beta, H, theta_eta, f', f'', f''') with f' the
    streamwise velocity in similarity units; on this branch f' is negative near
    the wall.  f''' comes from the similarity equation itself (see _fs_third)
    so that U'' never has to be formed by differentiating an interpolant.

    fw_min = -0.120 reaches H = 6.23.  It was -0.075, which stops at H = 4.99,
    and that limit is the reason the T3C4 bubble could not be modelled: the
    measurement there reaches a shape factor of 5.17, so the family the march
    reads from did not contain the profile the experiment was in.  The
    continuation converges smoothly the whole way - the reverse flow is still
    only four per cent of the edge velocity at H = 6.2 - so the old stopping
    point was a choice and not a limit of the branch.
    """
    global _FS_REVERSE
    if _FS_REVERSE is not None:
        return _FS_REVERSE
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "falkner_skan_reverse.npz")
    eta = np.linspace(0.0, _ETA_MAX, _ETA_N)
    if os.path.exists(cache):
        d = np.load(cache)
        if "upp" in d.files:            # a cache written before f''' is stale
            _FS_REVERSE = [(float(a), float(b), float(h), float(t), uu, pp, qq)
                           for a, b, h, t, uu, pp, qq
                           in zip(d["fw"], d["beta"], d["H"], d["theta"],
                                  d["u"], d["up"], d["upp"])]
            return _FS_REVERSE
    _, fam = _fs_family()
    k = int(np.argmin([m[4][0] for m in fam]))       # nearest to separation
    g = np.vstack([np.zeros_like(eta), fam[k][3], fam[k][4]])
    g[0] = np.concatenate([[0.0],
                           np.cumsum(0.5*(g[1][1:] + g[1][:-1])*np.diff(eta))])
    beta = fam[k][0]
    out = []
    for fw in np.linspace(fam[k][4][0], fw_min, n):
        sol = _fs_solve_shear(fw, eta, g, beta)
        if not sol.success:
            continue
        beta = float(sol.p[0])
        g = sol.sol(eta)
        f, u, up = g
        ds = _trapz(1.0 - u, eta); th = _trapz(u*(1.0 - u), eta)
        out.append((float(fw), beta, float(ds/th), float(th),
                    u.copy(), up.copy(), _fs_third(beta, f, u, up)))
    np.savez_compressed(cache, fw=np.array([m[0] for m in out]),
                        beta=np.array([m[1] for m in out]),
                        H=np.array([m[2] for m in out]),
                        theta=np.array([m[3] for m in out]),
                        u=np.array([m[4] for m in out]),
                        up=np.array([m[5] for m in out]),
                        upp=np.array([m[6] for m in out]), eta=eta)
    _FS_REVERSE = out
    return _FS_REVERSE


# ----------------------------------------------------------------------
# Falkner-Skan-Cooke: the swept-wing similarity family and its cross-flow
# ----------------------------------------------------------------------
# On a swept surface the external flow has a chordwise component U_e = C x^m
# and a span-wise component W_e that is constant along the chord.  The
# similarity solution splits: the chordwise momentum equation is the ordinary
# Falkner-Skan equation, and the span-wise component satisfies a linear
# equation driven by the same f,
#
#     f''' + f f'' + beta (1 - f'^2) = 0 ,      g'' + f g' = 0 ,
#     f(0) = f'(0) = 0 , f'(inf) = 1 , g(0) = 0 , g(inf) = 1 .
#
# Resolving the boundary-layer velocity (U_e f', W_e g) into components along
# and across the external streamline gives the cross-flow profile
#
#     w_cf = U_e W_e (f' - g) / sqrt(U_e^2 + W_e^2),
#
# which vanishes at the wall and in the free stream and is therefore
# inflectional - the profile whose instability drives cross-flow transition.
# Its shape depends on the pressure gradient through beta and on the local
# sweep through W_e/U_e.  The algebraic surrogate this replaces,
# Re_theta2 = k_cf Re_theta sin(L) cos(L), carries the sweep dependence but no
# dependence on beta at all, which is why it cannot transfer between two wings
# whose pressure distributions differ.
def fsc_profile(beta, sweep_deg, eta_max=_ETA_MAX, n=_ETA_N):
    """Falkner-Skan-Cooke solution and its cross-flow profile."""
    eta, fam = _fs_family()
    k = int(np.argmin([abs(m[0] - beta) for m in fam]))
    u0, up0 = fam[k][3], fam[k][4]
    f0 = np.concatenate([[0.0], np.cumsum(0.5*(u0[1:] + u0[:-1])*np.diff(eta))])
    sol = _fs_solve(beta, eta, np.vstack([f0, u0, up0]))
    if not sol.success:
        raise RuntimeError("FSC chordwise solve failed at beta=%.5f" % beta)
    f, fp, fpp = sol.sol(eta)

    # span-wise equation g'' + f g' = 0 integrates in closed form:
    #   g'(eta) = g'(0) exp(-int_0^eta f),  g normalised to 1 in the free stream
    F = np.concatenate([[0.0], np.cumsum(0.5*(f[1:] + f[:-1])*np.diff(eta))])
    gp = np.exp(-F)
    G = np.concatenate([[0.0], np.cumsum(0.5*(gp[1:] + gp[:-1])*np.diff(eta))])
    g = G/G[-1]

    L = np.radians(sweep_deg)
    w = np.abs(fp - g)*np.sin(L)*np.cos(L)      # cross-flow, in units of Q_e
    return eta, fp, g, w


def crossflow_reynolds(beta, sweep_deg, Re_theta):
    """Cross-flow Reynolds number of the Falkner-Skan-Cooke profile.

    Returns Re_cf = w_max * delta_10 / nu formed on the streamwise momentum
    thickness that the boundary-layer march already carries, where delta_10 is
    the distance from the wall to the point at which the cross-flow velocity
    has fallen back to a tenth of its peak.  That is Arnal's cross-flow
    Reynolds number, computed from the similarity solution rather than
    estimated from an algebraic surrogate.

    This is the per-case form, one boundary-value solve per beta.  The march
    reaches the same quantity through the tabulated crossflow_factor(lambda)
    below, since the sweep factors out exactly; this form is kept for checking
    that table against a direct solve.
    """
    eta, fp, g, w = fsc_profile(beta, sweep_deg)
    if w.max() <= 1e-12:
        return 0.0
    i = int(np.argmax(w))
    wmax = w[i]
    outer = np.where(w[i:] <= 0.1*wmax)[0]
    eta10 = eta[i + outer[0]] if len(outer) else eta[-1]
    th_eta = _trapz(fp*(1.0 - fp), eta)        # streamwise theta, similarity
    # Re_theta = U_e theta / nu and theta = th_eta * scale, so the scale cancels
    return float(wmax*eta10/th_eta*Re_theta)


# ----------------------------------------------------------------------
# True spatial eigenvalue problem
# ----------------------------------------------------------------------
# The tabulated database above is built with Gaster's transformation, which
# converts a temporal growth rate into a spatial one to first order in the
# growth rate.  That is what makes the table affordable: the temporal problem
# is linear in the phase speed and solves as a generalised eigenvalue problem,
# whereas the spatial problem is nonlinear in alpha and needs a Newton solve
# per frequency.
#
# The spatial problem is recovered exactly by allowing alpha to be complex and
# requiring the resulting frequency to be real and equal to the target.  The
# Orr-Sommerfeld operator is analytic in alpha, so Newton's method on the
# complex plane converges quadratically from the Gaster estimate; the
# derivative needed is the group velocity, which the temporal sweep already
# provides.  Two or three iterations suffice.
#
# spatial_alpha is not used to build the table; it is used to measure what
# using Gaster costs, which gaster_residual() below does and the module
# self-test reports.  Measured over the tabulated family at the peak-amplified
# frequency, Gaster under-predicts the spatial rate by 0.2 per cent on the
# Blasius profile at Re_theta = 300, rising to about 4 per cent in the adverse
# gradients near separation - not the "several per cent everywhere" an earlier
# version of this comment asserted, and not an eight per cent error in the
# transition Reynolds number.  It also does not propagate as one: the
# amplification factor is compared against N_crit through the fixed offset
# `anchor` of utss_solver._n_crit, which is set on measurement against these
# same rates, so a systematic scale error in sigma is absorbed there.  What is
# NOT absorbed is the variation of the deficit across the family, which is the
# 0.2-to-4 per cent spread above.
def spatial_alpha(y, D1, U, Upp, Re, omega, alpha0, cg, itmax=6, tol=1e-10):
    """Complex alpha whose Orr-Sommerfeld frequency equals the real omega.

    Returns (alpha, converged).  alpha.imag < 0 is an amplified wave, so the
    spatial amplification rate is sigma = -alpha.imag (in units of 1/theta).
    """
    a = complex(alpha0, 0.0) if np.isreal(alpha0) else complex(alpha0)
    if abs(cg) < 1e-6:
        return a, False
    a = a + 0.0j
    for _ in range(itmax):
        c = os_temporal(y, D1, U, Upp, a, Re)
        if c is None:
            return a, False
        w = a*c
        r = w - omega
        if abs(r) < tol:
            return a, True
        # group velocity re-estimated locally by a small complex step
        da = 1e-5*max(abs(a), 1e-3)
        c2 = os_temporal(y, D1, U, Upp, a + da, Re)
        if c2 is None:
            return a, False
        dwda = ((a + da)*c2 - w)/da
        if abs(dwda) < 1e-9:
            return a, False
        a = a - r/dwda
        if not np.isfinite(a):
            return a, False
    c = os_temporal(y, D1, U, Upp, a, Re)
    return a, (c is not None and abs(a*c - omega) < 1e-6)


def neutral_Re_theta(H=2.59129, lo=120.0, hi=400.0, tol=0.25, N=110,
                     alphas=None):
    """Momentum-thickness Reynolds number at which the profile first amplifies.

    Bisected on max_omega sigma(Re_theta) from a direct temporal sweep, so it
    measures the eigenvalue solver rather than the resolution of the tabulated
    Reynolds-number grid.  On the Blasius profile it returns 200 against the
    accepted 200.5 - 201 before fs_profile_for_H was made to solve the
    similarity profile rather than blend two neighbours.  What the
    boundary-layer march sees is the interpolated table, which first turns
    positive a few units above that because the node below the crossing holds
    an exact zero; that offset is the resolution of RET_GRID, not an error in
    the eigenvalue solver.  tabulated_neutral_Re_theta computes it, and the
    value is not restated here.
    """
    if alphas is None:
        alphas = np.geomspace(0.03, 0.35, 26)
    pr = fs_profile_for_H(H)

    def peak(Re):
        wr, sg = growth_curve(H, Re, alphas, N=N, profile=pr)
        return float(sg.max()) if sg.size else -1.0

    if peak(lo) > 0.0 or peak(hi) < 0.0:
        return float("nan")
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        if peak(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo + hi)


def gaster_residual(cases=((2.5913, 300.0), (2.5913, 800.0),
                            (3.0, 300.0), (3.5, 300.0), (3.5, 800.0)),
                    N=110, y_max=60.0, y_half=6.0):
    """What Gaster's transformation costs, measured rather than asserted.

    At each (H, Re_theta) the temporal sweep is run, the peak-amplified
    frequency identified, and the exact spatial eigenvalue found by Newton
    continuation from the Gaster estimate.  Returns a list of
    (H, Re_theta, sigma_gaster, sigma_exact, percentage deficit); rows whose
    Newton solve leaves the mode it started on are dropped rather than
    reported, since they measure the continuation and not the transformation.
    """
    out = []
    alphas = np.geomspace(8.0e-3, 0.35, 24)
    for H, Re in cases:
        pr = fs_profile_for_H(H)
        U, dU, Upp, y, D1 = _os_profile(pr, N, y_max, y_half)
        wr, sig, aa, cgs = growth_curve(H, Re, alphas, N=N, y_max=y_max,
                                        y_half=y_half, profile=pr,
                                        with_alpha=True)
        if not sig.size or sig.max() <= 0.0:
            continue
        # alpha and the group velocity come back from the SAME sorted sweep
        # that produced the peak.  Reading them out of the input `alphas` by
        # index, as this did, assumes growth_curve returns one frequency per
        # input wavenumber in the order given - which it does not: it drops the
        # wavenumbers whose eigenvalue solve failed and sorts what is left by
        # frequency, so the index of the peak addressed the wrong alpha.
        k = int(np.argmax(sig))
        om = float(wr[k]); sg = float(sig[k]); a0 = float(aa[k])
        cg = float(cgs[k])
        a, ok = spatial_alpha(y, D1, U, Upp, Re, om, a0, cg)
        se = -float(a.imag)
        # a Newton step that has landed on a different mode shows up as a
        # sign change or an order-of-magnitude jump; that is not a measure
        # of Gaster's error
        if not ok or se <= 0.0 or not (0.5 < sg/se < 2.0):
            continue
        out.append((H, Re, sg, se, 100.0*(sg - se)/se))
    return out


def closure_F(lam):
    """Right-hand side of the laminar momentum integral, F = 2l - 2(2+H)lambda.

    Writing the momentum-integral equation in terms of Z = theta^2 gives
    dZ/dx = (nu/U_e) F(lambda) with lambda = (Z/nu) dU_e/dx.  Thwaites replaced
    F by the straight line 0.45 - 6 lambda, which is what makes his method
    integrable in closed form and is the approximation his separation value
    lambda = -0.090 was chosen to compensate; the exact family separates at
    -0.0681.

    This is the exact right-hand side, for comparison rather than for the
    march: the default laminar branch is the two-equation march, which uses
    neither form, and the one-equation fallback behind cal["two_eq"] = False
    keeps Thwaites' own fit so that the ablation measures his method and not a
    hybrid.  Measured against the line in the module self-test, the two agree
    to 2 per cent at zero pressure gradient and 5 per cent through the adverse
    range, and part company by 17 per cent in a favourable one.
    """
    L, H, l = thwaites_closure()
    x = float(np.clip(lam, L[0], L[-1]))
    Hh = float(np.interp(x, L, H)); ll = float(np.interp(x, L, l))
    return 2.0*ll - 2.0*(2.0 + Hh)*x


# ----------------------------------------------------------------------
# Cross-flow factor from the Falkner-Skan-Cooke family
# ----------------------------------------------------------------------
# The cross-flow velocity is w = U_e W_e (f' - g)/Q, so the sweep enters only
# through sin(L)cos(L) and factors out exactly.  What remains,
#
#     K = max|f' - g| * eta_10 / theta_eta ,
#
# depends on the pressure gradient alone, and the cross-flow Reynolds number is
#
#     Re_cf = Re_theta * sin(L) cos(L) * K(lambda) .
#
# The algebraic surrogate this replaces used a constant in place of K.  K is
# not remotely constant, and it is not even monotone: it VANISHES at zero
# pressure gradient, where the chordwise and span-wise similarity equations
# give f' = g identically and there is no cross-flow at all, and it rises on
# both sides of that zero - to 4.60 at beta = 3, the strongest favourable
# member of the family, and to 1.87 at separation.  Across the family K spans
# 0.023 to 4.60, a factor of two hundred, not the factor of ten an earlier
# version of this comment claimed by quoting one adverse point (K = 0.46 at
# beta = -0.10) as though it were the minimum.  A favourable gradient thins the
# streamwise profile without thinning the cross-flow one, and an adverse one
# inflects it; either way the two profiles separate.  Two swept wings with
# different pressure distributions therefore cannot share a single constant,
# which is why the criterion did not transfer between the two experiments used
# here.  crossflow_table() is verified against a direct Falkner-Skan-Cooke
# solve in the module self-test.
#
# The span-wise equation g'' + f g' = 0 integrates in closed form given f, so
# the whole table follows from the Falkner-Skan family already computed and
# costs nothing beyond quadrature.
_CF_TABLE = None


def crossflow_table():
    """(lambda, K) table for the cross-flow factor, ordered by lambda."""
    global _CF_TABLE
    if _CF_TABLE is None:
        eta, fam = _fs_family()
        lam, K = [], []
        for b, H, th, u, upp, _q in fam:
            f = np.concatenate([[0.0],
                                np.cumsum(0.5*(u[1:] + u[:-1])*np.diff(eta))])
            F = np.concatenate([[0.0],
                                np.cumsum(0.5*(f[1:] + f[:-1])*np.diff(eta))])
            gp = np.exp(-F)
            G = np.concatenate([[0.0],
                                np.cumsum(0.5*(gp[1:] + gp[:-1])*np.diff(eta))])
            g = G/G[-1]
            w = np.abs(u - g)
            i = int(np.argmax(w)); wmax = w[i]
            if wmax <= 1e-9:
                continue
            o = np.where(w[i:] <= 0.1*wmax)[0]
            e10 = eta[i + o[0]] if len(o) else eta[-1]
            lam.append(b*th*th); K.append(wmax*e10/th)
        lam = np.array(lam); K = np.array(K)
        k = np.argsort(lam); lam, K = lam[k], K[k]
        keep = np.concatenate([[True], np.diff(lam) > 1e-12])
        _CF_TABLE = (lam[keep], K[keep])
    return _CF_TABLE


def crossflow_factor(lam):
    """K(lambda): cross-flow Reynolds number per unit Re_theta sin(L)cos(L)."""
    L, K = crossflow_table()
    return float(np.interp(float(np.clip(lam, L[0], L[-1])), L, K))


# ----------------------------------------------------------------------
# Stationary cross-flow: the Orr-Sommerfeld problem on the resolved profile
# ----------------------------------------------------------------------
# Everything above treats the two-dimensional problem, where the disturbance
# travels along the only direction the mean flow has.  On a swept wing the
# mean velocity turns through the layer, and a disturbance whose wave vector
# lies at an angle psi to the chord sees the component of that velocity
# resolved along its own direction:
#
#     U_psi(eta) = cos(L) f'(eta) cos(psi) + sin(L) g(eta) sin(psi)
#
# in units of the TOTAL edge speed Q_e, where f' and g are the chordwise and
# span-wise Falkner-Skan-Cooke similarity functions.  Writing psi = L + 90 + d
# turns that into an exact rotation between two profiles,
#
#     U_psi = cos(d) W(eta) - sin(d) S(eta)
#     W = sin(L)cos(L)(g - f')        the cross-flow profile
#     S = cos^2(L) f' + sin^2(L) g    the profile along the external streamline
#
# so at d = 0 the wave sees the cross-flow profile alone.  W vanishes at the
# wall and in the free stream and is therefore inflectional; that is the
# instability this branch exists to model.
#
# A STATIONARY cross-flow wave is one with omega = 0.  Since omega = k c, that
# is c = 0: the wave is fixed in the wing frame and the disturbance grows as it
# is convected past.  Stationary waves are the ones that matter in a quiet
# stream, because they are forced by surface roughness rather than by
# free-stream unsteadiness; Dagenhart & Saric measure them by naphthalene
# visualisation, which can only see something that does not move.
#
# Two things make this problem harder than the two-dimensional one and both
# have to be handled or the answer is nonsense:
#
#   The mode cannot be found by asking for the eigenvalue nearest zero.  In the
#   two-dimensional problem the discretised continuous spectrum crowds towards
#   c_r = 1 and a phase-speed window separates it from the physical mode.  Here
#   the resolved profile's edge value is near zero at exactly the wave angles
#   of interest, so the continuous spectrum crowds onto c = 0 - onto the
#   physical mode itself.  A Newton solve for c = 0 lands on it and returns
#   growth rates of order 10 in units of 1/theta, three orders above anything
#   physical.  What separates them is not the eigenvalue but the eigenfunction:
#   the physical mode decays away from the wall and the spurious ones do not.
#
#   The outer boundary has to be far enough out for that test to mean anything.
#   cf_modes measures the eigenfunction over the outer fifth of the domain and
#   requires it to be under 2 per cent of the peak.  A wave of wavenumber k
#   decays as exp(-k y), so at k = 0.1 the longest wave of interest is at
#   exp(-0.1*0.8*40) = 4 per cent where that is measured on a y_max = 40 theta
#   grid: the filter throws the PHYSICAL mode away, leaving only short waves and
#   putting the envelope maximum on the edge of the surviving band.  At
#   y_max = 100 theta the same wave is at exp(-8) = 0.03 per cent and passes,
#   while the discretised continuous spectrum stays above 25 per cent of its
#   peak out there (measured) and does not.  The two then separate cleanly.
#   (This gave the two decay figures as five per cent and 0.3 per cent; neither
#   is what exp(-k y) returns at the station the filter looks at, and the first
#   of them did not even fail the 2 per cent test the sentence turns on.)
#
# The stationary condition is then solved as a real equation in the wave angle
# rather than as a complex Newton step.  omega_r passes once through zero as
# psi sweeps across the direction normal to the external streamline, so
# bisection on that sign change stays on the branch by construction.
#
# The amplification is unambiguous once the group velocity is known.  A
# stationary wave packet is convected downstream at c_g and grows at omega_i
# per unit time, so over a chordwise step dx it gains
#
#     dN = omega_i dx / c_gx ,     c_gx = d omega_r / d alpha
#
# and the group velocity comes from the two derivatives the (k, psi)
# parameterisation already provides:
#
#     d omega/d alpha = cos(psi) d omega/dk - (sin(psi)/k) d omega/d psi
#
# References
#   Cooke J.C. (1950), "The boundary layer of a class of infinite yawed
#     cylinders", Proc. Camb. Phil. Soc. 46, 645.
#   Mack L.M. (1984), "Boundary-layer linear stability theory", AGARD R-709.
#   Dagenhart J.R. & Saric W.S. (1999), NASA/TP-1999-209344 - the measured
#     N-factors this implementation is checked against.
_CF_N, _CF_YMAX, _CF_YHALF = 60, 100.0, 5.0


def fsc_parts(beta):
    """f', f''', g, g'' and theta_eta of the Falkner-Skan-Cooke solution.

    Every derivative is exact rather than differentiated from an interpolant:
    f''' comes from the similarity equation through _fs_third, and g'' from the
    span-wise equation g'' = -f g' directly.
    """
    eta, fam = _fs_family()
    k = int(np.argmin([abs(m[0] - beta) for m in fam]))
    u0, up0 = fam[k][3], fam[k][4]
    f0 = np.concatenate([[0.0], np.cumsum(0.5*(u0[1:] + u0[:-1])*np.diff(eta))])
    sol = _fs_solve(beta, eta, np.vstack([f0, u0, up0]))
    if not sol.success:
        raise RuntimeError("FSC chordwise solve failed at beta=%.5f" % beta)
    f, fp, fpp = sol.sol(eta)
    fppp = _fs_third(beta, f, fp, fpp)
    F = np.concatenate([[0.0], np.cumsum(0.5*(f[1:] + f[:-1])*np.diff(eta))])
    gp = np.exp(-F)
    G = np.concatenate([[0.0], np.cumsum(0.5*(gp[1:] + gp[:-1])*np.diff(eta))])
    g = G/G[-1]; gp = gp/G[-1]
    return eta, fp, fppp, g, -f*gp, _trapz(fp*(1.0 - fp), eta)


def fsc_wave_profile(parts, sweep_deg, psi_deg):
    """U and U'' along the wave-vector direction, in units of Q_e and eta.

    The normalisation does not depend on psi, which is what lets growth rates
    at different wave angles be compared and what keeps the wave angle normal
    to the external streamline - where the resolved edge velocity is zero -
    an ordinary point rather than a singular one.
    """
    eta, fp, fppp, g, gpp, th_eta = parts
    L = np.radians(float(sweep_deg)); p = np.radians(float(psi_deg))
    cl, sl = np.cos(L), np.sin(L)
    cp, sp = np.cos(p), np.sin(p)
    return cl*fp*cp + sl*g*sp, cl*fppp*cp + sl*gpp*sp, np.cos(p - L)


def _cf_grid(parts, sweep_deg, psi_deg, N, y_max, y_half):
    eta, fp, fppp, g, gpp, th_eta = parts
    Ue, Uppe, ue = fsc_wave_profile(parts, sweep_deg, psi_deg)
    y, D1 = _grid(N, y_max, y_half)
    ey = np.clip(y*th_eta, 0.0, eta[-1])
    U = np.interp(ey, eta, Ue)
    Upp = np.interp(ey, eta, Uppe)*th_eta*th_eta
    out = y > eta[-1]/th_eta
    U[out] = ue; Upp[out] = 0.0
    return y, D1, U, Upp


def cf_modes(y, D1, U, Upp, alpha, Re, decay=0.02):
    """Boundary-layer-confined eigenvalues of the resolved-profile operator.

    Two filters, and both are needed.  An unstable mode of an inflectional
    instability has c_r between the minimum and maximum of the mean profile,
    which the discretised continuous spectrum does not respect; and the
    physical mode decays away from the wall, which the continuous spectrum
    does not do either.  Phase speed alone is not enough here, unlike in
    os_temporal: the resolved profile's edge value is near zero at the wave
    angles of interest, so both spectra occupy the same part of the plane.
    """
    n = len(y); I = np.eye(n); D2 = D1 @ D1
    L = D2 - alpha**2*I
    A = np.diag(U) @ L - np.diag(Upp) - (L @ L)/(1j*alpha*Re)
    B = L.astype(complex)
    for row, con in ((0, I[0]), (1, D1[0]), (n - 2, D1[-1]), (n - 1, I[-1])):
        A[row, :] = con; B[row, :] = 0.0
    w, V = eig(A, B, right=True)
    lo, hi = U.min(), U.max()
    pad = 0.05*max(hi - lo, 1e-12)
    ok = np.isfinite(w) & (w.real > lo - pad) & (w.real < hi + pad)
    outer = y > 0.80*y.max()
    out = []
    for k in np.where(ok)[0]:
        v = np.abs(V[:, k]); vm = v.max()
        if vm > 0 and v[outer].max()/vm < decay:
            out.append((w[k], V[:, k]))
    return out


def _cf_omega(parts, L, psi, k, Re, N, ym, yh, prev=None):
    y, D1, U, Upp = _cf_grid(parts, L, psi, N, ym, yh)
    m = cf_modes(y, D1, U, Upp, k, Re)
    if not m:
        return None
    c = (max(m, key=lambda t: t[0].imag)[0] if prev is None
         else min(m, key=lambda t: abs(t[0] - prev))[0])
    return k*c, c


def _cf_bracket(parts, L, k, Re, lo, hi, n, N, ym, yh):
    prev_p, prev_f = None, None
    for p in np.linspace(lo, hi, n):
        r = _cf_omega(parts, L, float(p), k, Re, N, ym, yh)
        if r is None:
            continue
        f = r[0].real
        if prev_p is not None and prev_f*f <= 0.0:
            return prev_p, prev_f, float(p)
        prev_p, prev_f = float(p), f
    return None


def cf_stationary_at_k(parts, L, k, Re, window, N=_CF_N, ym=_CF_YMAX,
                       yh=_CF_YHALF, itmax=24):
    """(psi, omega, c_gx) of the stationary cross-flow wave at wavenumber k.

    Returns None where no stationary wave exists on the branch, which is the
    normal answer at zero sweep and at zero pressure gradient.
    """
    br = _cf_bracket(parts, L, k, Re, window[0], window[1], window[2],
                     N, ym, yh)
    if br is None:
        return None
    a, fa, b = br
    p, r = None, None
    for _ in range(itmax):
        p = 0.5*(a + b)
        r = _cf_omega(parts, L, p, k, Re, N, ym, yh)
        if r is None:
            return None
        f = r[0].real
        if abs(f) < 1e-12 or (b - a) < 1e-3:
            break
        if fa*f <= 0.0:
            b = p
        else:
            a, fa = p, f
    om, c = r
    # a bracket produced by the branch jumping rather than by a root leaves
    # omega_r finite; a converged one sits at 1e-6 or below
    if abs(om.real) > 1e-5:
        return None
    dk, dp = 1e-4*max(k, 1e-3), 1e-3
    rk = _cf_omega(parts, L, p, k + dk, Re, N, ym, yh, prev=c)
    rp = _cf_omega(parts, L, p + dp, k, Re, N, ym, yh, prev=c)
    if rk is None or rp is None:
        return None
    dwdk = (rk[0] - om)/dk
    dwdp = (rp[0] - om)/np.radians(dp)
    cp, sp = np.cos(np.radians(p)), np.sin(np.radians(p))
    return p, om, float((cp*dwdk - sp/k*dwdp).real)


def stationary_crossflow(beta, sweep_deg, Re_theta, parts=None, ks=None,
                         half=14.0, N=_CF_N, ym=_CF_YMAX, yh=_CF_YHALF):
    """Envelope amplification rate of the stationary cross-flow wave.

    Returns (sigma, k, psi).  sigma is the growth of disturbance amplitude per
    unit CHORDWISE distance in units of 1/theta, formed on the total edge
    speed, maximised over wavenumber; the march integrates sigma/theta ds.

    It is exactly zero where there is no cross-flow to be unstable: at zero
    sweep, and at zero pressure gradient, where the chordwise and span-wise
    similarity equations give f' = g identically.
    """
    L = float(sweep_deg)
    if L <= 1e-9 or abs(float(beta)) < 1e-9:
        return 0.0, 0.0, 0.0
    parts = fsc_parts(float(beta)) if parts is None else parts
    Re = float(Re_theta)/np.cos(np.radians(L))
    ks = np.geomspace(0.03, 1.0, 22) if ks is None else np.asarray(ks, float)
    mid = int(len(ks)//2)
    wide = (L + 90.0 - half, L + 90.0 + half, 57)
    seed = cf_stationary_at_k(parts, L, float(ks[mid]), Re, wide, N, ym, yh)
    if seed is None:
        return 0.0, 0.0, 0.0
    best = (0.0, 0.0, 0.0)
    # The stationary angle moves by well under a degree per wavenumber step, so
    # each solve is seeded from its neighbour and only falls back to the wide
    # window when that fails.  Two consecutive failures mean the branch has run
    # off the end of the unstable band, and marching on out along it costs a
    # wide scan per wavenumber for nothing.
    for order in (range(mid, len(ks)), range(mid - 1, -1, -1)):
        psi = seed[0]
        miss = 0
        for j in order:
            r = cf_stationary_at_k(parts, L, float(ks[j]), Re,
                                   (psi - 4.0, psi + 4.0, 9), N, ym, yh)
            if r is None:
                r = cf_stationary_at_k(parts, L, float(ks[j]), Re, wide,
                                       N, ym, yh)
            if r is None:
                miss += 1
                if miss >= 2:
                    break
                continue
            miss = 0
            psi, om, cgx = r
            if cgx > 1e-9 and om.imag > 0.0:
                s = om.imag/cgx
                if s > best[0]:
                    best = (float(s), float(ks[j]), float(psi))
    return best


# ----------------------------------------------------------------------
# Tabulated stationary cross-flow amplification
# ----------------------------------------------------------------------
# The eigenvalue sweep above costs a few hundred Orr-Sommerfeld solves per
# station, which a boundary-layer march cannot afford, so it is tabulated once
# on the three parameters it depends on and interpolated thereafter.
#
# The three are not two.  For the cross-flow REYNOLDS NUMBER the sweep factors
# out exactly - Re_cf = Re_theta sin(L)cos(L) K(lambda) - and a table in lambda
# alone suffices.  For the growth RATE it does not: sigma/(sin L cos L) rises
# monotonically with sweep, by a factor of nearly three between 20 and 55
# degrees at fixed lambda and Re_theta, while sin(L)cos(L) itself turns over at
# 45 degrees.  The stability problem sees the sweep twice, once in the
# amplitude of the cross-flow profile and again in the mixture of chordwise and
# span-wise flow the streamwise profile carries, and only the first of those is
# what the algebraic surrogate models.
CF_LAM_BETA = np.array([-0.10, -0.05, -0.02, 0.0, 0.03, 0.06, 0.10, 0.15,
                        0.20, 0.30, 0.40, 0.55, 0.70, 0.90, 1.20, 1.60,
                        2.00, 2.50, 3.00])
CF_SWEEP = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
# The lowest node is chosen so that clamping below it is exact rather than a
# guess: the stationary mode is damped at every point of the table at
# Re_theta = 20, including the strongly-accelerated, highly-swept corner the
# leading edge of a swept wing runs through, where growth first appears between
# Re_theta = 20 and 35.  An earlier grid stopped at 50 and held the rate at its
# value there for everything below, which invents amplification in exactly the
# region where the momentum thickness is smallest and sigma/theta is largest.
CF_RETH = np.geomspace(20.0, 2000.0, 12)
CF_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "crossflow_db.npz")


# The wavenumber at which the envelope peaks is a property of the profile
# shape and is very nearly independent of Reynolds number - 0.18 at every
# Re_theta from 100 to 800 on the sections here - because the instability is
# inflectional and its Reynolds-number dependence is a saturation, not a shift.
# So the full wavenumber sweep is done once per (beta, sweep) and the other
# Reynolds numbers search a band around the wavenumber it found.  A blind
# sweep at all ten costs three and a half hours for this table; this costs one.
_CF_SEED_RETH = 400.0
_CF_BAND = np.geomspace(0.45, 2.2, 7)


def _cf_slab(arg):
    i, beta = arg
    parts = fsc_parts(float(beta))
    out = np.zeros((CF_SWEEP.size, CF_RETH.size))
    for j, L in enumerate(CF_SWEEP):
        _, kpk, _ = stationary_crossflow(float(beta), float(L), _CF_SEED_RETH,
                                         parts=parts)
        ks = None if kpk <= 0.0 else kpk*_CF_BAND
        for m, R in enumerate(CF_RETH):
            out[j, m] = stationary_crossflow(float(beta), float(L), float(R),
                                             parts=parts, ks=ks)[0]
    return i, out, float(parts[5])


def build_crossflow_database(path=CF_DB_PATH, nproc=4, verbose=True,
                             resume=True):
    """Generate and store the stationary cross-flow amplification table.

    Checkpointed per Hartree parameter, like build_database: this table is
    hours rather than minutes, and an interrupted run resumes from the slabs
    already on disk.
    """
    from multiprocessing import Pool
    ck = os.path.join(os.path.dirname(os.path.abspath(path)), "_cf_slabs")
    os.makedirs(ck, exist_ok=True)
    todo = [(i, float(b)) for i, b in enumerate(CF_LAM_BETA)
            if not (resume and os.path.exists(
                os.path.join(ck, "cf_%03d.npz" % i)))]
    if verbose:
        print("cross-flow database: %d beta x %d sweep x %d Re_theta = %d "
              "wave-angle sweeps; %d of %d slabs still to do"
              % (CF_LAM_BETA.size, CF_SWEEP.size, CF_RETH.size,
                 CF_LAM_BETA.size*CF_SWEEP.size*CF_RETH.size,
                 len(todo), CF_LAM_BETA.size), flush=True)
    if todo:
        with Pool(nproc) as p:
            for i, slab, th in p.imap_unordered(_cf_slab, todo):
                np.savez(os.path.join(ck, "cf_%03d.npz" % i), sigma=slab,
                         th_eta=th)
                if verbose:
                    done = len([f for f in os.listdir(ck)
                                if f.endswith(".npz")])
                    print("  beta = %+.2f done (%d/%d)"
                          % (CF_LAM_BETA[i], done, CF_LAM_BETA.size),
                          flush=True)
    sig, th = [], []
    for i in range(CF_LAM_BETA.size):
        d = np.load(os.path.join(ck, "cf_%03d.npz" % i))
        sig.append(d["sigma"]); th.append(float(d["th_eta"]))
    sigma = np.array(sig); th = np.array(th)
    lam = CF_LAM_BETA*th*th
    k = np.argsort(lam)
    np.savez_compressed(path, beta=CF_LAM_BETA[k], lam=lam[k], th_eta=th[k],
                        sweep=CF_SWEEP, Re_theta=CF_RETH, sigma=sigma[k])
    if verbose:
        print("wrote %s   sigma range %.4g .. %.4g"
              % (path, float(sigma.min()), float(sigma.max())))
    return sigma


_CF_DB = None


def load_crossflow_database(path=CF_DB_PATH):
    """Load the tabulated cross-flow rates, building them if absent."""
    global _CF_DB
    if _CF_DB is None:
        if not os.path.exists(path):
            build_crossflow_database(path)
        d = np.load(path)
        _CF_DB = (d["lam"], d["sweep"], d["Re_theta"], d["sigma"])
    return _CF_DB


def crossflow_sigma(lam, sweep_deg, Re_theta):
    """Tabulated stationary cross-flow amplification rate, sigma*theta.

    Trilinear in (lambda, sweep, log Re_theta).  Clamped at every edge: the
    rate saturates with Reynolds number, as an inflectional instability must,
    so holding it at the last node beyond the table is the right extrapolation
    and not a fallback.
    """
    LAM, SW, RE, S = load_crossflow_database()
    if float(sweep_deg) <= 1e-9:
        return 0.0
    # Below the lowest tabulated Reynolds number the rate is taken as zero, not
    # held at the floor value.  The mode is damped there over all but six of
    # the table's 152 (lambda, sweep) cells, and those six are the
    # plane-stagnation corner - beta >= 2 at 60 degrees of local sweep or more
    # - which a march passes through only in its first few stations, at
    # the attachment line, where a local-similarity description is not the
    # right one in any case.  Holding the rate at the floor instead invents
    # amplification exactly where the momentum thickness is smallest and
    # sigma/theta is therefore largest: on the ten swept conditions of
    # 06_validation/crossflow_amplification.csv the two treatments differ by at
    # most 6 per cent in N, against 40 per cent when the floor stood at
    # Re_theta = 50.
    if float(Re_theta) < RE[0]:
        return 0.0
    lam = float(np.clip(lam, LAM[0], LAM[-1]))
    sw = float(np.clip(sweep_deg, SW[0], SW[-1]))
    re = float(np.clip(Re_theta, RE[0], RE[-1]))

    def _w(grid, v):
        i = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
        t = (v - grid[i])/(grid[i + 1] - grid[i])
        return i, float(np.clip(t, 0.0, 1.0))

    i, a = _w(LAM, lam)
    j, b = _w(SW, sw)
    m, c = _w(np.log(RE), np.log(re))
    v = 0.0
    for di, wa in ((0, 1.0 - a), (1, a)):
        for dj, wb in ((0, 1.0 - b), (1, b)):
            for dm, wc in ((0, 1.0 - c), (1, c)):
                v += wa*wb*wc*S[i + di, j + dj, m + dm]
    return float(max(v, 0.0))


# ----------------------------------------------------------------------
# Two-equation laminar closure from the Falkner-Skan family
# ----------------------------------------------------------------------
# A one-parameter method slaves the shape factor to the local pressure
# gradient, H = H(lambda), so the layer carries no memory of how it arrived.
# That is the deepest approximation in the present formulation: two layers with
# the same local gradient but different histories are assigned the same profile
# and therefore the same amplification rate, and the error shows up wherever a
# favourable run is followed by a mild adverse one, which is the whole forward
# half of a natural-laminar-flow aerofoil.
#
# Carrying a second equation removes it.  The momentum and kinetic-energy
# integrals are
#
#   dtheta/dx = C_f/2 - (2+H) (theta/U_e) dU_e/dx
#   theta dH*/dx = 2 C_D - H* C_f/2 - H*(1-H)(theta/U_e) dU_e/dx
#
# and they close on three functions of the shape factor, all of them properties
# of the Falkner-Skan family and none of them fitted:
#
#   H*(H)          = theta*/theta , the kinetic-energy shape parameter
#   l(H)           = Re_theta C_f/2
#   d(H)           = Re_theta C_D
#
# with theta* = int u(1-u^2) dy and C_D the dissipation integral.  In
# similarity variables l = theta_eta f''(0) and d = theta_eta int f''^2 deta.
_TWOEQ = None


def twoeq_closure():
    """(H, Hstar, l, d) closure table for the two-equation laminar method."""
    global _TWOEQ
    if _TWOEQ is None:
        eta, fam = _fs_family()
        H, Hs, L, D = [], [], [], []
        for b, h, th, u, up, _q in fam:
            ths = _trapz(u*(1.0 - u*u), eta)          # energy thickness
            H.append(h); Hs.append(ths/th)
            L.append(th*up[0])
            D.append(th*_trapz(up*up, eta))
        H = np.array(H); Hs = np.array(Hs)
        L = np.array(L); D = np.array(D)
        k = np.argsort(H); H, Hs, L, D = H[k], Hs[k], L[k], D[k]
        keep = np.concatenate([[True], np.diff(H) > 1e-9])
        _TWOEQ = (H[keep], Hs[keep], L[keep], D[keep])
    return _TWOEQ


def twoeq_HL(H):
    """H*, l and d at a given shape factor."""
    Hg, Hs, L, D = twoeq_closure()
    x = float(np.clip(H, Hg[0], Hg[-1]))
    return (float(np.interp(x, Hg, Hs)), float(np.interp(x, Hg, L)),
            float(np.interp(x, Hg, D)))


_COMB_HSTAR = None


def _combined_Hstar():
    """(H, H*, l, d) along the combined attached + reverse family, ordered by H."""
    global _COMB_HSTAR
    if _COMB_HSTAR is None:
        eta, prof = _combined_family()
        H = np.array([p[0] for p in prof])
        Hs, L, D = [], [], []
        for h, th, u, up, upp in prof:
            Hs.append(_trapz(u*(1.0 - u*u), eta)/th)
            L.append(th*up[0])
            D.append(th*_trapz(up*up, eta))
        k = np.argsort(H)
        _COMB_HSTAR = (H[k], np.array(Hs)[k], np.array(L)[k], np.array(D)[k])
    return _COMB_HSTAR


def twoeq_HL_combined(H):
    """H*, l and d at a shape factor that may lie past separation.

    twoeq_HL clips to the ATTACHED family, which is right for an attached layer
    and wrong for the interior of a separation bubble.  This reads the same
    three closures off the combined family, so the dead-air march can be closed
    at the shape factor it has actually reached rather than at the largest one
    the attached branch happens to contain.
    """
    Hg, Hs, L, D = _combined_Hstar()
    x = float(np.clip(H, Hg[0], Hg[-1]))
    return (float(np.interp(x, Hg, Hs)), float(np.interp(x, Hg, L)),
            float(np.interp(x, Hg, D)))


def H_from_Hstar(Hstar, H_prev=None):
    """Invert H*(H).

    On the ATTACHED branch H* falls monotonically with H and the inversion is
    unique.  Across the whole family it is not: H* has a minimum at H = 4.03 -
    a flat one, agreeing to six decimals over 4.025 to 4.040 - the fold where
    the attached and reverse-flow branches meet, and rises again beyond it, so
    one H* names two profiles, one attached and one separated.

    With nothing else to go on the attached root is the right one, and a bare
    call returns it; the dead-air march used to be capped at the fold for
    exactly this reason.  But a march KNOWS which branch it is on, because it
    got there continuously, and passing the previous H resolves the ambiguity:
    the root nearest H_prev is the one the layer is on.  That is what lets the
    bubble march continue past H = 3.997 to the shape factors a real separation
    bubble reaches.

    The inversion is ill-conditioned on the reverse branch - H* moves by only
    0.8 per cent between H = 4.03 and H = 4.99 - so it is used there to CARRY a
    march that is already close, never to establish a shape factor from nothing.
    The caller limits how far H may move in one step for the same reason.
    """
    Hg, Hs, L, D = twoeq_closure()
    if H_prev is None:
        o = np.argsort(Hs)
        return float(np.interp(float(np.clip(Hstar, Hs[o][0], Hs[o][-1])),
                               Hs[o], Hg[o]))
    Hc, Hsc, _l, _d = _combined_Hstar()
    x = float(np.clip(Hstar, float(Hsc.min()), float(Hsc.max())))
    roots = []
    for i in range(Hsc.size - 1):
        a, b = float(Hsc[i]), float(Hsc[i+1])
        if (a - x)*(b - x) <= 0.0 and a != b:
            w = (x - a)/(b - a)
            roots.append(float(Hc[i] + w*(Hc[i+1] - Hc[i])))
    if not roots:
        return float(Hc[int(np.argmin(np.abs(Hsc - x)))])
    return float(min(roots, key=lambda r: abs(r - H_prev)))


if __name__ == "__main__":
    # ---- self-tests -------------------------------------------------------
    # 1. the family reproduces the Blasius profile
    _e, _u, _up, _H, _th = falkner_skan(0.0)
    print("Blasius from the Falkner-Skan family: H = %.5f (2.59129), "
          "f''(0) = %.5f (0.46960)  %s"
          % (_H, _up[0],
             "OK" if abs(_H - 2.59129) < 2e-4 and abs(_up[0] - 0.4696) < 2e-4
             else "FAIL"))

    # 2. the exact closure agrees with Thwaites where his fit is good, and
    #    separates at the exact family value rather than at his fitted -0.090
    print("exact Thwaites closure: l(0) = %.4f (Thwaites 0.220), "
          "lambda_sep = %.4f" % (closure_HL(0.0)[1], lambda_sep()))
    print("shape-factor range of the attached family: %.3f .. %.3f"
          % fs_H_range())

    # 3. the Blasius neutral point, from the solver and from the table
    _n = neutral_Re_theta()
    # what the march actually sees: the bilinearly interpolated table
    _tab, _lo_node, _hi_node = tabulated_neutral_Re_theta()
    print("Blasius neutral point: solver Re_theta = %.0f (accepted 200.5), "
          "interpolated table Re_theta = %.0f (between grid nodes %.0f and "
          "%.0f; not itself a node)  %s"
          % (_n, _tab, _lo_node, _hi_node,
             "OK" if abs(_n - 200.5) < 5.0 else "FAIL"))

    # 3b. the single-frequency lookup must agree with the curve the march reads
    _c = sigma_curve(3.0, 500.0)
    _, _, _Os, _ = load_database()
    _d = max(abs(sigma_lookup(3.0, 500.0, float(o)) - v) for o, v in zip(_Os, _c))
    print("sigma_lookup vs sigma_curve at (H=3.0, Re_theta=500): max |diff| = "
          "%.1e, out of band -> %.1f  %s"
          % (_d, sigma_lookup(3.0, 500.0, 1.0), "OK" if _d == 0.0 else "FAIL"))

    # 3c. the exact momentum-integral right-hand side against Thwaites' line.
    #     Thwaites replaced F(lambda) = 2l - 2(2+H)lambda by 0.45 - 6 lambda,
    #     which is what makes his method integrable in closed form.  The two
    #     agree where he fitted them and part company near separation, which is
    #     why his separation value (-0.090) is not the exact one (-0.0681).
    print("momentum-integral F(lambda): exact vs Thwaites' 0.45 - 6 lambda")
    for _l in (0.05, 0.0, -0.03, -0.06, lambda_sep()):
        _F = closure_F(_l); _T = 0.45 - 6.0*_l
        print("   lambda = %+7.4f   exact %+7.4f   Thwaites %+7.4f   %+6.1f %%"
              % (_l, _F, _T, 100.0*(_T - _F)/abs(_F) if _F else float("nan")))

    # 4. the tabulated cross-flow factor against a direct similarity solve.
    #    beta = 0 is excluded from the relative comparison and checked on its
    #    own: the cross-flow vanishes identically there, so a percentage
    #    against it divides by zero and says nothing about the table.
    _sc = np.sin(np.radians(45.0))*np.cos(np.radians(45.0))
    _rows = []
    for _b in (-0.19, -0.10, -0.05, 0.30, 1.00, 3.00):
        _lam = _b*falkner_skan(_b)[4]**2
        _direct = crossflow_reynolds(_b, 45.0, 1000.0)
        _table = 1000.0*_sc*crossflow_factor(_lam)
        _rows.append((_b, _lam, _direct, _table,
                      100.0*(_table - _direct)/_direct))
    print("cross-flow Reynolds number, tabulated K(lambda) vs a direct "
          "Falkner-Skan-Cooke solve (45 deg, Re_theta = 1000):")
    for _b, _lam, _d, _t, _e in _rows:
        print("   beta = %+5.2f  lambda = %+7.4f   direct %7.1f   table %7.1f"
              "   %+.2f %%" % (_b, _lam, _d, _t, _e))
    _worst = max(abs(r[4]) for r in _rows)
    _zero = 1000.0*_sc*crossflow_factor(0.0)
    print("   worst discrepancy %.2f %%; at beta = 0 the direct solve gives "
          "exactly 0 and the table %.1f, against %.0f at beta = 3  %s"
          % (_worst, _zero, _rows[-1][3],
             "OK" if _worst < 2.0 and _zero < 0.02*_rows[-1][3] else "FAIL"))


    # 5. the stationary cross-flow branch, against its two exact limits and
    #    against its own tabulation.
    print("stationary cross-flow: the two cases where the answer is exactly zero")
    _z1 = stationary_crossflow(0.5, 0.0, 400.0)[0]
    _z2 = stationary_crossflow(0.0, 45.0, 400.0)[0]
    _w0 = float(np.abs(fsc_profile(0.0, 45.0)[3]).max())
    print("   zero sweep, beta = 0.5      sigma = %.6f" % _z1)
    print("   zero gradient, 45 deg       sigma = %.6f   (max |w| in the "
          "profile itself %.2e: beta = 0 gives g = f' identically)  %s"
          % (_z2, _w0, "OK" if _z1 == 0.0 and _z2 == 0.0 and _w0 < 1e-5
             else "FAIL"))
    _pa = fsc_parts(0.6)
    _rt = [(R, stationary_crossflow(0.6, 45.0, R, parts=_pa,
                                    ks=np.geomspace(0.11, 0.30, 5))[0])
           for R in (100.0, 400.0, 3000.0)]
    print("   an inflectional instability saturates with Reynolds number:")
    for _R, _s in _rt:
        print("      Re_theta = %6.0f   sigma*theta = %.6f" % (_R, _s))
    _rise = all(b > a for (_, a), (_, b) in zip(_rt, _rt[1:]))
    print("      rising %s, last decade %.0f %% of the first  %s"
          % (_rise, 100.0*(_rt[2][1]-_rt[1][1])/max(_rt[1][1]-_rt[0][1], 1e-12),
             "OK" if _rise and (_rt[2][1]-_rt[1][1]) < 0.60*(_rt[1][1]-_rt[0][1])
             else "FAIL"))
    if os.path.exists(CF_DB_PATH):
        _b, _sw, _re = 0.45, 47.0, 420.0
        _lam = _b*falkner_skan(_b)[4]**2
        _tab = crossflow_sigma(_lam, _sw, _re)
        _dir = stationary_crossflow(_b, _sw, _re)[0]
        print("   tabulated rate vs a direct sweep at lambda = %+.4f, %g deg, "
              "Re_theta = %g:\n      table %.6f   direct %.6f   %+.1f %%  %s"
              % (_lam, _sw, _re, _tab, _dir, 100.0*(_tab-_dir)/_dir,
                 "OK" if abs(_tab-_dir) < 0.22*_dir else "FAIL"))
    else:
        print("   (no cross-flow table built; crossflow_sigma would build it)")

    # 6. what building the table with Gaster's transformation costs
    _g = gaster_residual()
    if _g:
        print("Gaster vs exact spatial rate at the peak-amplified frequency:")
        for _H, _Re, _sg, _se, _d in _g:
            print("   H = %.3f  Re_theta = %6.0f   gaster %.5f   exact %.5f"
                  "   %+.1f %%" % (_H, _Re, _sg, _se, _d))
        print("   worst deficit over the sample: %.1f %%"
              % max(abs(d) for *_, d in _g))
