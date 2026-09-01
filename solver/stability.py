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


def fs_profile_for_H(H_target):
    """Falkner-Skan profile at a prescribed shape factor.

    H rises monotonically along the family, from about 2.13 in a strong
    favourable gradient to 4.04 at separation, so the profile at a given H is
    obtained by interpolating between the two family members that bracket it.
    The family is resolved finely enough in beta that consecutive members
    differ by well under 0.01 in H, and linear interpolation of the velocity
    profile between them is far inside the accuracy of the eigenvalue solution.
    """
    key = round(float(H_target), 5)
    if key in _H_CACHE:
        return _H_CACHE[key]
    eta, fam = _fs_family()
    H = np.array([m[1] for m in fam])
    t = float(np.clip(H_target, H[0], H[-1]))
    j = int(np.clip(np.searchsorted(H, t) - 1, 0, H.size - 2))
    w = (t - H[j])/(H[j+1] - H[j])
    u = (1.0 - w)*fam[j][3] + w*fam[j+1][3]
    up = (1.0 - w)*fam[j][4] + w*fam[j+1][4]
    th = (1.0 - w)*fam[j][2] + w*fam[j+1][2]
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
H_GRID   = np.linspace(2.20, 3.90, 22)
RET_GRID = np.geomspace(60.0, 6000.0, 32)
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


def build_database(path=DB_PATH, nproc=4, verbose=True):
    """Generate and store the amplification database (runs in a few minutes)."""
    from multiprocessing import Pool
    if verbose:
        print("building amplification database: %d H x %d Re_theta x %d alpha "
              "= %d Orr-Sommerfeld solves"
              % (H_GRID.size, RET_GRID.size, ALPHAS.size,
                 H_GRID.size*RET_GRID.size*ALPHAS.size))
    with Pool(nproc) as p:
        slabs = p.map(_sigma_slab, list(H_GRID))
    sigma = np.array(slabs)
    np.savez_compressed(path, H=H_GRID, Re_theta=RET_GRID, omega=OM_GRID,
                        sigma=sigma)
    if verbose:
        print("wrote %s   sigma range %.4g .. %.4g"
              % (path, float(sigma.min()), float(sigma.max())))
    return sigma


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


def sigma_lookup(H, Re_theta, omega):
    """Trilinear interpolation of sigma = -alpha_i*theta.

    H and Re_theta are clamped to the tabulated range: below H = 2.2 the layer
    is strongly accelerated and effectively stable, above H = 3.9 it is close
    to separating and the separation branch governs.  omega outside the
    tabulated band returns zero, which is correct - those frequencies are not
    amplified.
    """
    Hs, Rs, Os, S3 = load_database()
    H = float(np.clip(H, Hs[0], Hs[-1]))
    R = float(np.clip(Re_theta, Rs[0], Rs[-1]))
    if omega < Os[0] or omega > Os[-1]:
        return 0.0
    i = int(np.clip(np.searchsorted(Hs, H) - 1, 0, Hs.size - 2))
    j = int(np.clip(np.searchsorted(Rs, R) - 1, 0, Rs.size - 2))
    k = int(np.clip(np.searchsorted(Os, omega) - 1, 0, Os.size - 2))
    th = (H - Hs[i])/(Hs[i+1] - Hs[i])
    tr = (R - Rs[j])/(Rs[j+1] - Rs[j])
    to = (omega - Os[k])/(Os[k+1] - Os[k])
    c = 0.0
    for di, wi in ((0, 1-th), (1, th)):
        for dj, wj in ((0, 1-tr), (1, tr)):
            for dk, wk in ((0, 1-to), (1, to)):
                c += wi*wj*wk*S3[i+di, j+dj, k+dk]
    return float(c)


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


def omega_grid_bounds():
    """Tabulated range of the dimensionless frequency omega*theta/U_e."""
    _, _, Os, _ = load_database()
    return float(Os[0]), float(Os[-1])
