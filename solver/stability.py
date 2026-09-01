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
        fam = [(float(b), float(h), float(t), uu, pp) for b, h, t, uu, pp
               in zip(d["beta"], d["H"], d["theta"], d["u"], d["up"])]
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
        ds = np.trapz(1.0 - u, eta); th = np.trapz(u*(1.0 - u), eta)
        fam.append((float(b), float(ds/th), float(th), u.copy(), up.copy()))
    H = np.array([m[1] for m in fam])
    keep = np.concatenate([[True], np.diff(H) > 1e-9])
    fam = [m for m, k in zip(fam, keep) if k]
    np.savez_compressed(cache, eta=eta,
                        beta=np.array([m[0] for m in fam]),
                        H=np.array([m[1] for m in fam]),
                        theta=np.array([m[2] for m in fam]),
                        u=np.array([m[3] for m in fam]),
                        up=np.array([m[4] for m in fam]))
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
    ds = np.trapz(1.0 - u, eta); th = np.trapz(u*(1.0 - u), eta)
    return eta, u, up, float(ds/th), float(th)


_H_CACHE = {}


def _combined_family():
    """Attached and reverse-flow branches joined into one monotone H family."""
    eta, fam = _fs_family()
    prof = [(m[1], m[2], m[3], m[4]) for m in fam]          # (H, theta, u, u')
    try:
        rev = fs_reverse_family()
        prof += [(m[2], m[3], m[4], m[5]) for m in rev if m[2] > prof[-1][0]]
    except Exception:
        pass
    prof.sort(key=lambda t: t[0])
    keep = [prof[0]]
    for p in prof[1:]:
        if p[0] - keep[-1][0] > 1e-9:
            keep.append(p)
    return eta, keep


_COMB = None


def fs_profile_for_H(H_target):
    """Falkner-Skan profile at a prescribed shape factor.

    H rises monotonically along the combined family, from about 2.13 in a
    strong favourable gradient, through 4.00 at separation, to about 4.99 on
    the reverse-flow branch, so the profile at a given H is obtained by
    interpolating between the two members that bracket it.  Continuing past
    separation is what allows the amplification rate inside a separation
    bubble to be read from the same table as everywhere else, rather than
    supplied as a separate constant.
    """
    global _COMB
    key = round(float(H_target), 5)
    if key in _H_CACHE:
        return _H_CACHE[key]
    if _COMB is None:
        _COMB = _combined_family()
    eta, prof = _COMB
    H = np.array([p[0] for p in prof])
    t = float(np.clip(H_target, H[0], H[-1]))
    j = int(np.clip(np.searchsorted(H, t) - 1, 0, H.size - 2))
    w = (t - H[j])/(H[j+1] - H[j])
    u = (1.0 - w)*prof[j][2] + w*prof[j+1][2]
    up = (1.0 - w)*prof[j][3] + w*prof[j+1][3]
    th = (1.0 - w)*prof[j][1] + w*prof[j+1][1]
    pr = (eta, u, up, t, th)
    _H_CACHE[key] = pr
    return pr


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


def growth_curve(H_target, Re_theta, alphas, N=110, y_max=60.0, y_half=6.0,
                 profile=None):
    """Spatial growth rate against frequency at one (H, Re_theta).

    Returns (omega_r, sigma) with omega_r = alpha*c_r the dimensionless
    frequency and sigma = -alpha_i*theta the spatial amplification rate,
    obtained from the temporal rate by Gaster's transformation
    alpha_i = -omega_i / c_g,  c_g = d(omega_r)/d(alpha).
    """
    if profile is None:
        profile = fs_profile_for_H(H_target)
    eta, u, up, H, th_eta = profile
    y, D1 = _grid(N, y_max, y_half)
    # interpolate the similarity profile onto the collocation grid, in units
    # of the momentum thickness
    eta_y = np.clip(y*th_eta, 0.0, eta[-1])
    U = np.interp(eta_y, eta, u)
    dU = np.interp(eta_y, eta, up)*th_eta
    Upp = D1 @ dU
    U[y > eta[-1]/th_eta] = 1.0
    om_r, om_i = [], []
    for a in alphas:
        c = os_temporal(y, D1, U, Upp, a, Re_theta)
        if c is None:
            om_r.append(np.nan); om_i.append(np.nan)
        else:
            om_r.append(a*c.real); om_i.append(a*c.imag)
    om_r = np.array(om_r); om_i = np.array(om_i)
    good = np.isfinite(om_r)
    if good.sum() < 3:
        return np.array([]), np.array([])
    cg = np.gradient(om_r[good], np.asarray(alphas)[good])
    cg = np.where(np.abs(cg) < 1e-6, np.nan, cg)
    sigma = -(-om_i[good]/cg)          # -alpha_i = +omega_i/c_g
    return om_r[good], np.nan_to_num(sigma, nan=0.0)


# ----------------------------------------------------------------------
# Amplification-rate database
# ----------------------------------------------------------------------
# The table is indexed by shape factor, momentum-thickness Reynolds number and
# dimensionless frequency, and holds the spatial amplification rate
# sigma = -alpha_i*theta.  The shape-factor range stops short of the
# Falkner-Skan separation profile (H = 4.04): the eigenvalue problem is stiff
# there and the separation-induced branch of the transition kernel, not the
# amplification integral, is what decides transition in a separating layer.
# The shape-factor grid is dense below H = 3.0, where the neutral boundary of
# the Orr-Sommerfeld problem moves quickly and a coarse grid interpolates
# across it, and it now runs past separation onto the reverse-flow branch so
# that a separated shear layer reads its amplification rate from the same
# table as an attached one.
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
    eta, u, up, Hh, th = pr
    y, D1 = _grid(NCHEB, 60.0, 6.0)
    eta_y = np.clip(y*th, 0.0, eta[-1])
    U = np.interp(eta_y, eta, u)
    dU = np.interp(eta_y, eta, up)*th
    Upp = D1 @ dU
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
    and onto the reverse-flow branch at H = 4.96, so that a detached shear layer
    reads its rate from the same table as an attached one.  omega outside the
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
# amplification rate of Sec. sigma_lookup rises by a factor of seven between
# the Blasius profile and H = 3.9, so a shape factor short by 0.9 near
# separation starves the amplification integral in exactly the adverse
# gradients where transition is decided.
_CLOSURE = None


def thwaites_closure():
    """Exact (lambda, H, l) closure table, ordered by increasing lambda."""
    global _CLOSURE
    if _CLOSURE is None:
        eta, fam = _fs_family()
        lam = np.array([b*th*th for b, H, th, u, up in fam])
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


def fs_reverse_family(fw_min=-0.075, n=90):
    """Profiles from separation into reverse flow, ordered by decreasing shear.

    Returns a list of (f''(0), beta, H, theta_eta, u, u') with u the streamwise
    velocity in similarity units; on this branch u is negative near the wall.
    """
    global _FS_REVERSE
    if _FS_REVERSE is not None:
        return _FS_REVERSE
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "falkner_skan_reverse.npz")
    eta = np.linspace(0.0, _ETA_MAX, _ETA_N)
    if os.path.exists(cache):
        d = np.load(cache)
        _FS_REVERSE = [(float(a), float(b), float(h), float(t), uu, pp)
                       for a, b, h, t, uu, pp in zip(d["fw"], d["beta"], d["H"],
                                                     d["theta"], d["u"], d["up"])]
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
        ds = np.trapz(1.0 - u, eta); th = np.trapz(u*(1.0 - u), eta)
        out.append((float(fw), beta, float(ds/th), float(th),
                    u.copy(), up.copy()))
    np.savez_compressed(cache, fw=np.array([m[0] for m in out]),
                        beta=np.array([m[1] for m in out]),
                        H=np.array([m[2] for m in out]),
                        theta=np.array([m[3] for m in out]),
                        u=np.array([m[4] for m in out]),
                        up=np.array([m[5] for m in out]), eta=eta)
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
    th_eta = np.trapz(fp*(1.0 - fp), eta)        # streamwise theta, similarity
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
    Reynolds-number grid.  On the Blasius profile it returns 201 against the
    accepted 200.5.  What the boundary-layer march sees is the interpolated
    table, which first turns positive at Re_theta = 210 because the node below
    the crossing holds an exact zero; that offset is the resolution of
    RET_GRID, not an error in the eigenvalue solver.
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
        eta, u, up, Hh, th = pr
        y, D1 = _grid(N, y_max, y_half)
        eta_y = np.clip(y*th, 0.0, eta[-1])
        U = np.interp(eta_y, eta, u)
        dU = np.interp(eta_y, eta, up)*th
        Upp = D1 @ dU
        U[y > eta[-1]/th] = 1.0
        wr, sig = growth_curve(H, Re, alphas, N=N, y_max=y_max,
                               y_half=y_half, profile=pr)
        if not sig.size or sig.max() <= 0.0:
            continue
        k = int(np.argmax(sig))
        om = float(wr[k]); sg = float(sig[k])
        a0 = float(np.interp(om, wr, alphas[:wr.size]))
        j = int(np.clip(k, 1, wr.size - 2))
        cg = (wr[j+1] - wr[j-1])/(alphas[j+1] - alphas[j-1])
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
    integrable in closed form and is the approximation that his separation
    value lambda = -0.090 was chosen to compensate.  Using the exact F removes
    both: the march is then consistent with the exact closure functions, and
    separation occurs at the exact Falkner-Skan value.
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
        for b, H, th, u, upp in fam:
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
        for b, h, th, u, up in fam:
            ths = np.trapz(u*(1.0 - u*u), eta)          # energy thickness
            H.append(h); Hs.append(ths/th)
            L.append(th*up[0])
            D.append(th*np.trapz(up*up, eta))
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


def H_from_Hstar(Hstar):
    """Invert H*(H).  H* falls monotonically with H across the family."""
    Hg, Hs, L, D = twoeq_closure()
    o = np.argsort(Hs)
    return float(np.interp(float(np.clip(Hstar, Hs[o][0], Hs[o][-1])),
                           Hs[o], Hg[o]))


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
    _tab = next((float(R) for R in np.arange(RET_GRID[0], 400.0, 1.0)
                 if sigma_curve(2.59129, float(R)).max() > 0.0), float("nan"))
    print("Blasius neutral point: solver Re_theta = %.0f (accepted 200.5), "
          "tabulated grid Re_theta = %.0f  %s"
          % (_n, _tab, "OK" if abs(_n - 200.5) < 5.0 else "FAIL"))

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

    # 5. what building the table with Gaster's transformation costs
    _g = gaster_residual()
    if _g:
        print("Gaster vs exact spatial rate at the peak-amplified frequency:")
        for _H, _Re, _sg, _se, _d in _g:
            print("   H = %.3f  Re_theta = %6.0f   gaster %.5f   exact %.5f"
                  "   %+.1f %%" % (_H, _Re, _sg, _se, _d))
        print("   worst deficit over the sample: %.1f %%"
              % max(abs(d) for *_, d in _g))
