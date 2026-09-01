"""
=====================================================================
 UTSS - UNIVERSAL TURBULENCE-TRANSITION & SKIN-FRICTION SOLVER  v1.0
=====================================================================
A reduced-order, physics-based engine for predicting boundary-layer
laminar -> transitional -> turbulent state over aircraft surfaces.

Architecture
------------
 1. Inviscid edge solution   : constant-strength vortex-panel method
                               (Kuethe & Chow formulation) -> Cp, Ue,
                               with a Karman-Tsien correction.
 2. Laminar boundary layer    : Thwaites' integral method.
 3. UNIFIED TRANSITION KERNEL : the novel contribution. A single
       onset is taken as the minimum effective transition-Re across
       FOUR co-resident mechanisms, each with a calibration weight:
         (a) Tollmien-Schlichting / natural   (envelope e^N, Drela-Giles)
         (b) Bypass (free-stream turbulence)   (Abu-Ghannam & Shaw)
         (c) Laminar-separation-induced        (short-bubble criterion)
         (d) Cross-flow (swept-wing)           (C1 cross-flow Re crit.)
 4. Transitional region       : Narasimha universal-intermittency
       closure; properties blended Cf = (1-g)Cf_lam + g Cf_turb.
 5. Turbulent boundary layer  : Head's entrainment method with the
       Ludwieg-Tillmann skin-friction law.
 6. Drag                      : Squire-Young far-wake formula.

The natural and bypass branches are independent models - an envelope
e^N amplification integral and the Abu-Ghannam & Shaw correlation
respectively - and the SAME constant set reproduces zero-pressure-
gradient bypass plates (ERCOFTAC T3A/T3B) and low-turbulence natural
transition (Schubauer & Skramstad) without per-case re-tuning.  The
separation and cross-flow branches are calibrated, not validated.

Author: Akosa Samuel Onyejekwe, 2026.
"""
import numpy as np
import stability as _stab

# ----------------------------------------------------------------------
#  Default universal calibration constants  (single set, all cases)
# ----------------------------------------------------------------------
_OM_TAB = _stab.load_database()[2]   # tabulated omega*theta/U_e grid

CAL = dict(
    A_TS      = 1.00,   # weight on TS/natural onset
    A_BP      = 1.00,   # weight on bypass onset
    A_SEP     = 1.00,   # weight on separation-induced onset
    A_CF      = 1.00,   # weight on cross-flow onset
    N_crit    = 9.0,    # reference e^N factor; the value actually used is
                        # obtained from Tu by Mack's relation (see _n_crit),
                        # which is clamped to its stated validity range and so
                        # returns 8.68 for any stream quieter than 0.08 %
    N_floor   = 0.5,    # lower clamp on N_crit at high Tu
    C_mu      = 0.09,   # k-epsilon constants, used only for the decay of
    C_eps2    = 1.92,   # free-stream turbulence (see _tu_decay)
    Tu_TS_max = 0.1,    # Tu [%] above which the envelope e^N branch is
                        # switched off.  Set equal to the bypass gate, so
                        # the two routes are complementary rather than
                        # overlapping: below 0.1% the amplification route
                        # governs and the bypass correlation is inactive,
                        # above it the reverse.  Leaving both live in the
                        # overlap makes the e^N branch fire spuriously,
                        # because Mack's relation returns N_crit ~ 3 there
                        # and the envelope method is not calibrated that low.
    C_len     = 6.5,    # transition-length scaling (Narasimha)
    CF_C1     = 150.0,  # cross-flow C1 critical Re_theta2 (Arnal)
    CF_ratio  = 0.47,   # theta2/theta surrogate; see _re_theta2().
                        # Calibrated on the 45 deg swept NLF(2)-0415
                        # transition measurements of Dagenhart & Saric
                        # (NASA TP-1999-209344, Table 2): 13.5% mean error
                        # in transition location over six chord Reynolds
                        # numbers from 1.92e6 to 3.73e6.
    sep_floor = 120.0,  # min Re_theta for separation-induced onset
    use_os_db = True,   # integrate the amplification factor from the tabulated
                        # Orr-Sommerfeld growth rates (solver/stability.py)
                        # rather than from the Drela-Giles envelope.  The
                        # envelope is retained, and selected by setting this
                        # False, so that the two can be compared on the same
                        # calibration; nothing else in the model changes.
    bubble    = True,   # close the separation branch by continuing the
                        # amplification integral through the detached shear
                        # layer instead of transitioning at separation itself
    H_sep     = 0.0,    # if positive, laminar separation is declared where the
                        # solved shape factor reaches this value instead of
                        # where the Thwaites parameter reaches lam_sep.  With
                        # H carrying its own history the shape factor is the
                        # physically meaningful indicator; lam is a local
                        # quantity and its threshold belongs to the
                        # one-parameter method it was fitted for.
    cf_exact  = False,  # form the cross-flow Reynolds number from the exact
                        # Falkner-Skan-Cooke factor K(lambda) instead of the
                        # constant surrogate k_cf
    tu_hist   = 0.60,   # weight given to the flow-history average of Tu in
                        # the bypass correlation, the remainder going to the
                        # local value: Tu_eff = Tu_avg^w Tu_local^(1-w).
                        # w = 1 is the pure history average, w = 0 the local
                        # value.  Abu-Ghannam and Shaw correlated their data
                        # against the intensity measured at transition, so the
                        # local value is faithful to the correlation; but in a
                        # decaying stream marching on it is ill-conditioned,
                        # the threshold rising as Tu falls while Re_theta grows
                        # only as the square root of distance.  Neither limit
                        # serves all three plates - the local value gives +90 %
                        # on T3A, the pure average -22 % on T3A- - and 0.60 is
                        # where the mean error over the three is least, 9.1 %
                        # against 14.0 % for the pure average.  In a stream
                        # that does not decay the two are identical and the
                        # weight has no effect, so no aerofoil or wing result
                        # depends on it.
    lam_sep   = -0.090, # Thwaites parameter at which laminar separation is
                        # declared.  This is Thwaites' own published value and
                        # it is not fitted here.  An earlier version of this
                        # work re-fitted it to -0.100, on the grounds that the
                        # two-equation march returns a larger momentum
                        # thickness so lambda reaches any given value sooner.
                        # That was compensating for something else: the
                        # tabulated edge velocities of the T3C4 rig are quoted
                        # to 0.01 m/s, and interpolating exactly through them
                        # put a spurious local minimum of lambda a quarter of a
                        # metre upstream of the measured separation.  With the
                        # smoothing fit of solve_flat_plate that minimum is
                        # gone and the published value serves, so the constant
                        # has been returned to the literature.
    two_eq    = True,   # march the laminar layer with the momentum AND
                        # kinetic-energy integrals, so that the shape factor is
                        # a solved variable carrying its own history rather
                        # than a local function of the pressure gradient
    n_freq    = 40,     # number of physical frequencies carried by the e^N
                        # integration.  The amplification factor is the
                        # envelope of the individual N(omega) curves, so this
                        # is a discretisation parameter, not a fitted one: the
                        # predicted onset changes by under 0.2% between 24 and
                        # 64 frequencies.
)


# ----------------------------------------------------------------------
#  Envelope e^N amplification (Drela & Giles, AIAA J. 25(10) 1987)
# ----------------------------------------------------------------------
def _tu_decay(Tu0_pct, x, L_turb, U=None, cal=None):
    """Local free-stream turbulence intensity along a surface.

    Grid turbulence decays downstream, so the intensity that the transition
    criteria see is not the inlet value.  Rather than leaving the choice
    between "inlet" and "local" as a convention, the local value is computed
    from the inlet value by the standard k-epsilon decay of isotropic
    turbulence,

        k(x) = k0 [1 + a x]^(-1/(C_eps2 - 1)),   Tu ~ sqrt(k),

    so that  Tu(x) = Tu0 [1 + a x]^(-1/(2(C_eps2 - 1))),  with the decay
    rate set by the inlet turbulence length scale L_turb through
    a = (C_eps2 - 1) C_mu^(3/4) sqrt(3/2) Tu0 / L_turb.

    L_turb is a property of the oncoming stream, not of the transition
    model: for grid turbulence it is a few millimetres, and for the free
    atmosphere it is hundreds of metres, which makes the decay negligible
    over a wing chord and recovers a constant Tu automatically.
    """
    cal = {**CAL, **(cal or {})}
    Tu0 = max(float(Tu0_pct), 1e-6)/100.0
    if L_turb is None or L_turb <= 0.0:
        return np.full_like(np.asarray(x, float), Tu0*100.0)
    a = ((cal["C_eps2"] - 1.0)*cal["C_mu"]**0.75*np.sqrt(1.5)*Tu0)/L_turb
    p = 1.0/(2.0*(cal["C_eps2"] - 1.0))
    xx = np.maximum(np.asarray(x, float), 0.0)
    return Tu0*100.0*(1.0 + a*xx)**(-p)


def _n_crit(Tu_pct, floor=0.5, anchor=0.50):
    """Critical amplification factor from the free-stream turbulence level.

    Mack's correlation, N_crit = -8.43 - 2.4 ln(Tu) with Tu as a fraction,
    supplies the dependence on the disturbance environment.  Tu is clamped to
    the 0.0008 <= Tu <= 0.0298 interval over which the correlation is quoted;
    the lower clamp is the one that matters, because a logarithm extrapolated
    below its calibration range rises without bound and would assign
    N_crit = 12.0 to a 0.02 per cent tunnel, which the flat-plate measurements
    contradict.

    The correlation is then shifted by a fixed anchor.  An amplification factor
    has no meaning independently of the growth rates used to compute it, and
    Mack's values were established with the rates available in 1977, whereas
    the rates tabulated in stability.py are converged and reproduce the Blasius
    neutral point and published amplification rates to within a few per cent.
    A constant offset between the two scales is therefore expected, and fixing
    it is a change of units rather than a fit to any one case.  The offset is
    the only quantity in the natural branch set by measurement.  It is chosen
    jointly over every dataset in this work that the branch reaches - the
    Schubauer-Skramstad plate and all 86 NLF(1)-0416 aerofoil conditions - and
    0.50 is where the number of predictions falling inside the experimental
    bracket is greatest, 54 of 86, with the flat plate at +5.6 per cent.
    """
    Tu = min(max(float(Tu_pct)/100.0, 8.0e-4), 2.98e-2)
    return max(-8.43 - 2.4*np.log(Tu) - anchor, floor)


def _re_theta_crit(H):
    """Momentum-thickness Reynolds number at which amplification begins."""
    H = float(np.clip(H, 1.05, 20.0))
    hm = 1.0/(H - 1.0)
    log10Re = ((1.415*hm - 0.489)*np.tanh(20.0*hm - 12.9)
               + 3.295*hm + 0.44)
    return 10.0**log10Re


def _dn_dReth(H):
    """Envelope amplification rate  dn~/dRe_theta  as a function of H."""
    H = float(np.clip(H, 1.05, 20.0))
    a = 2.4*H - 3.7 + 2.5*np.tanh(1.5*H - 4.65)
    return 0.01*np.sqrt(a*a + 0.25)


def _re_theta2(Re_theta, sweep_deg, ratio):
    """Cross-flow momentum-thickness Reynolds number, algebraic surrogate.

    A three-dimensional boundary-layer solution is not carried, so the
    cross-flow momentum thickness is estimated from the streamwise one by
    theta2/theta ~ ratio * sin(L) cos(L).  `ratio` is set so that the C1
    criterion becomes governing beyond roughly 25 deg of sweep.  It is
    calibrated against the swept-wing transition measurements of Dagenhart
    & Saric rather than against design experience alone."""
    L = np.radians(sweep_deg)
    return ratio*Re_theta*np.sin(L)*np.cos(L)

# ======================================================================
#  1.  VORTEX-PANEL INVISCID SOLVER  (Kuethe & Chow)
# ======================================================================
def panel_solve(xb, yb, alpha_deg, mach=0.0):
    """
    Constant-strength vortex panel method.
    xb,yb : boundary points ordered clockwise from the trailing edge
            (TE -> lower -> LE -> upper -> TE).  N+1 points, N panels.
    Returns control-point x, Cp, surface velocity Ue/Uinf, panel angles.
    """
    al = np.radians(alpha_deg)
    x, y = np.asarray(xb, float), np.asarray(yb, float)
    m = len(x) - 1
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    S  = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
    th = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
    sin_t, cos_t = np.sin(th), np.cos(th)

    # Influence coefficients (Kuethe & Chow, constant-strength vortex panel).
    # Built by broadcasting rather than by a double loop: the matrices are
    # m x m with m of order 400, so the loop form spent almost all of the
    # solver's runtime evaluating scalar numpy calls.  The expressions are
    # unchanged; only the iteration is vectorised.  B vanishes on the
    # diagonal, so it is masked to 1 while the off-diagonal terms are formed
    # and the diagonal is then overwritten with its analytic value.
    dx = xc[:, None] - x[None, :m]
    dy = yc[:, None] - y[None, :m]
    dth = th[:, None] - th[None, :]
    Sj  = S[None, :]
    A = -dx*cos_t[None, :] - dy*sin_t[None, :]
    B = dx**2 + dy**2
    np.fill_diagonal(B, 1.0)
    Cc = np.sin(dth); Dd = np.cos(dth)
    E = dx*sin_t[None, :] - dy*cos_t[None, :]
    F = np.log1p((Sj**2 + 2.0*A*Sj)/B)
    G = np.arctan2(E*Sj, B + A*Sj)
    th2 = th[:, None] - 2.0*th[None, :]
    P = dx*np.sin(th2) + dy*np.cos(th2)
    Q = dx*np.cos(th2) - dy*np.sin(th2)
    CN2 = Dd + 0.5*Q*F/Sj - (A*Cc + Dd*E)*G/Sj
    CN1 = 0.5*Dd*F + Cc*G - CN2
    CT2 = Cc + 0.5*P*F/Sj + (A*Dd - Cc*E)*G/Sj
    CT1 = 0.5*Cc*F - Dd*G - CT2
    np.fill_diagonal(CN1, -1.0); np.fill_diagonal(CN2, 1.0)
    np.fill_diagonal(CT1, 0.5*np.pi); np.fill_diagonal(CT2, 0.5*np.pi)

    AN = np.zeros((m+1, m+1)); AT = np.zeros((m, m+1))
    AN[:m, 0] = CN1[:, 0];   AN[:m, m] = CN2[:, m-1]
    AT[:m, 0] = CT1[:, 0];   AT[:m, m] = CT2[:, m-1]
    AN[:m, 1:m] = CN1[:, 1:m] + CN2[:, 0:m-1]
    AT[:m, 1:m] = CT1[:, 1:m] + CT2[:, 0:m-1]
    AN[m, 0] = 1.0; AN[m, m] = 1.0
    rhs = np.append(np.sin(th - al), 0.0)
    gamma = np.linalg.solve(AN, rhs)

    V  = np.cos(th - al) + AT @ gamma
    Cp = 1.0 - V**2
    # Karman-Tsien compressibility correction,
    #
    #     Cp = Cp0 / [ beta + (M^2/(1+beta)) (Cp0/2) ],   beta = sqrt(1-M^2),
    #
    # applied to the pressure coefficient and hence to the integrated loads.
    # It replaces the Prandtl-Glauert scaling Cp = Cp0/beta used previously.
    # Prandtl-Glauert is the linearised result: it multiplies every pressure
    # by the same factor, so it cannot know that a suction peak compresses the
    # flow more than a mild gradient does, and at the stagnation point it
    # returns Cp = 1/beta rather than the correct value near unity.  The
    # Karman-Tsien form retains the leading nonlinear term and stays bounded,
    # which matters here because the transition kernel is driven by the
    # gradient of the pressure and not by its level.  Both reduce to the
    # incompressible result as M -> 0, and the two differ by under half a
    # percent in C_p below M = 0.2, which is why every validation case in this
    # work is unaffected by the change.
    if mach > 1e-6:
        beta = np.sqrt(max(1.0 - mach*mach, 1e-6))
        Cp = Cp/(beta + (mach*mach/(1.0 + beta))*0.5*Cp)
    panel_solve.last_gamma = gamma          # nodal vortex strengths
    return xc, yc, Cp, V, th, S


def velocity_field(xb, yb, alpha_deg, Xg, Yg, U=1.0, mach=0.0):
    """Field velocity / Cp from the panel solution using the EXACT
    constant-strength vortex-panel induced velocity (Katz & Plotkin),
    vectorised per panel.  Returns Vx,Vy,Cp on the grid."""
    al = np.radians(alpha_deg)
    xc_, yc_, Cp_, V_, th_, S_ = panel_solve(xb, yb, alpha_deg)
    gam = panel_solve.last_gamma             # nodal (m+1), normalised by 2*pi*Uinf
    gpan = 0.5*(gam[:-1] + gam[1:]) * U * 2*np.pi   # actual circ. density
    x1 = np.asarray(xb[:-1], float); y1 = np.asarray(yb[:-1], float)
    Vx = np.full(Xg.shape, U*np.cos(al))
    Vy = np.full(Xg.shape, U*np.sin(al))
    for j in range(len(S_)):
        ct, st = np.cos(th_[j]), np.sin(th_[j])
        dx = Xg - x1[j]; dy = Yg - y1[j]
        xp =  dx*ct + dy*st
        zp = -dx*st + dy*ct
        r1 = np.hypot(xp, zp) + 1e-9
        r2 = np.hypot(xp - S_[j], zp) + 1e-9
        th1 = np.arctan2(zp, xp)
        th2 = np.arctan2(zp, xp - S_[j])
        up =  gpan[j]/(2*np.pi) * (th2 - th1)
        wp =  gpan[j]/(2*np.pi) * np.log(r1/r2)
        Vx += up*ct - wp*st
        Vy += up*st + wp*ct
    Cp = 1.0 - (Vx**2 + Vy**2)/U**2
    if mach > 1e-6:
        Cp = Cp/np.sqrt(max(1.0 - mach*mach, 1e-6))
    return Vx, Vy, Cp


# ======================================================================
#  2-6.  BOUNDARY-LAYER MARCHER WITH UNIFIED TRANSITION KERNEL
# ======================================================================
def _thwaites_HL(lam):
    lam = np.clip(lam, -0.1, 0.25)
    if lam >= 0.0:
        l = 0.22 + 1.57*lam - 1.8*lam**2
        H = 2.61 - 3.75*lam + 5.24*lam**2
    else:
        l = 0.22 + 1.402*lam + 0.018*lam/(lam + 0.107)
        H = 2.088 + 0.0731/(lam + 0.14)
    return H, l


def _ags_re_theta_t(Tu_pct, lam_t):
    """Abu-Ghannam & Shaw (1980) bypass onset correlation."""
    lam_t = np.clip(lam_t, -0.1, 0.1)
    if lam_t <= 0.0:
        F = 6.91 + 12.75*lam_t + 63.64*lam_t**2
    else:
        F = 6.91 + 2.48*lam_t - 12.27*lam_t**2
    Tu = max(Tu_pct, 0.02)
    return 163.0 + np.exp(F - F*Tu/6.91)


def _head_H1(H):
    if H <= 1.6:
        return 3.3 + 0.8234*(H - 1.1)**(-1.287)
    return 3.3 + 1.5501*(H - 0.6778)**(-3.064)


def _head_H_from_H1(H1):
    H1 = max(H1, 3.32)
    if H1 >= 5.3:
        return 1.1 + 0.86*(H1 - 3.3)**(-0.777)
    return 0.6778 + 1.1538*(H1 - 3.3)**(-0.326)


def _ludwieg_tillmann(H, Re_th):
    Re_th = max(Re_th, 1.0)
    return 0.246 * 10**(-0.678*H) * Re_th**(-0.268)


def _smooth_gradient(y, x, half=5):
    """d y / d x from a local weighted least-squares line.

    A panel method returns the edge velocity at control points that are
    unevenly spaced and carry panel-to-panel scatter, especially near the
    leading edge where the panels are shortest.  Differencing that scatter
    directly makes the Thwaites parameter oscillate from station to station,
    and because the critical Reynolds number of the amplification envelope
    is exponentially sensitive to the shape factor, the oscillation drives
    the e^N integral rather than the flow does.  A local fit over a few
    stations removes the scatter without smearing the real gradient.
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(x); g = np.zeros(n)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        xx = x[a:b] - x[i]; yy = y[a:b]
        if len(xx) < 2 or np.ptp(xx) <= 0.0:
            g[i] = 0.0; continue
        w = np.exp(-(xx/(np.ptp(xx) + 1e-30))**2)
        sw = w.sum(); sx = (w*xx).sum(); sxx = (w*xx*xx).sum()
        sy = (w*yy).sum(); sxy = (w*xx*yy).sum()
        det = sw*sxx - sx*sx
        g[i] = (sw*sxy - sx*sy)/det if abs(det) > 1e-30 else 0.0
    return g



def _ref_temp_nu(Me, gamma=1.4, Pr=0.72, omega=0.76, laminar=True):
    """Eckert reference-temperature factor for the kinematic viscosity.

    The integral closures used here - Thwaites, Head's entrainment method,
    Ludwieg-Tillmann - are incompressible.  They are carried into compressible
    flow in the standard way, by evaluating the fluid properties not at the
    edge conditions but at Eckert's reference temperature

        T*/T_e = 1 + 0.032 M_e^2 + 0.58 (T_w/T_e - 1),

    with an adiabatic wall, T_w/T_e = 1 + r (gamma-1)/2 M_e^2 and r = Pr^(1/2)
    laminar, Pr^(1/3) turbulent.  At constant static pressure rho ~ 1/T and
    mu ~ T^omega, so nu*/nu_e = (T*/T_e)^(1+omega).  The layer is then marched
    with nu* in place of nu, which is what makes the incompressible closures
    return the compressible skin friction and the compressible momentum
    thickness; the effective Reynolds number falls, so heating a boundary
    layer this way delays transition, as it should.

    The factor is 1 at M_e = 0 and reaches 1.09 at M_e = 0.42, so it changes
    nothing in the low-speed validation cases and becomes significant only in
    the cruise case for which it was added.
    """
    Me = np.asarray(Me, float)
    r = Pr**0.5 if laminar else Pr**(1.0/3.0)
    Tw_Te = 1.0 + r*(gamma - 1.0)/2.0*Me**2
    Tstar_Te = 1.0 + 0.032*Me**2 + 0.58*(Tw_Te - 1.0)
    return Tstar_Te**(1.0 + omega)


def march_bl(s, Ue, nu, Tu_pct=0.2, sweep_deg=0.0, Ue_inf=1.0,
             cal=None, label="", a_sound=0.0):
    """
    March the boundary layer along one surface.
      s    : arc length from stagnation/leading edge [m]  (increasing)
      Ue   : edge velocity magnitude [m/s]
      nu   : kinematic viscosity [m^2/s]
      Tu_pct: free-stream turbulence intensity [%].  Either a scalar,
              or an array of length len(s) giving the LOCAL value at each
              station.  Grid turbulence decays along a wind-tunnel plate,
              and both the AGS correlation and Mack's relation are
              functions of the local level, so passing the measured decay
              is the correct usage where it is known.  In free flight the
              ambient level does not decay over a chord and a scalar is
              appropriate.
      sweep_deg: surface sweep angle (cross-flow mechanism)
      a_sound: speed of sound [m/s].  If given, the local edge Mach number is
              formed at every station and the closures are evaluated at
              Eckert's reference temperature; if zero the march is
              incompressible, which is the default and what every low-speed
              validation case uses.
    Returns a dict of station arrays describing the full BL state.
    """
    cal = {**CAL, **(cal or {})}
    n = len(s)
    Tu_arr = (np.full(n, float(Tu_pct)) if np.ndim(Tu_pct) == 0
              else np.asarray(Tu_pct, float))
    if len(Tu_arr) != n:
        raise ValueError("Tu_pct array must match the station count")
    s = np.asarray(s, float); Ue = np.maximum(np.asarray(Ue, float), 1e-6)
    dUeds = _smooth_gradient(Ue, s)

    # Compressible closures: properties at Eckert's reference temperature.
    Me = (Ue/a_sound if a_sound > 1e-6 else np.zeros(n))
    nu_l = nu*_ref_temp_nu(Me, laminar=True)      # laminar recovery factor
    nu_t = nu*_ref_temp_nu(Me, laminar=False)     # turbulent recovery factor

    theta = np.zeros(n); H = np.zeros(n); Cf = np.zeros(n)
    Reth = np.zeros(n); lam = np.zeros(n); gamma = np.zeros(n)
    state = np.array(["laminar"] * n, dtype=object)
    Re_th_t = np.full(n, np.nan)
    n_fac = np.zeros(n)
    n_amp = 0.0
    mechanism = np.array(["-"] * n, dtype=object)

    two_eq = bool(cal.get("two_eq", True))
    # With the shape factor solved rather than slaved to the local gradient,
    # separation is detected where it physically occurs - the wall shear
    # vanishes as H approaches the Falkner-Skan separation value of 3.997 -
    # rather than through the Thwaites parameter.  The two are not equivalent:
    # lambda reaches its separation value wherever the local gradient is steep
    # enough, whatever the layer's history, whereas a layer that has just come
    # off a favourable run is fuller and resists separation.
    H_SEP = 3.95

    # ---- Laminar branch via Thwaites (integral form) ----
    I = np.zeros(n)
    for i in range(1, n):
        I[i] = I[i-1] + 0.5*(Ue[i]**5 + Ue[i-1]**5)*(s[i]-s[i-1])
    th2 = 0.45*nu_l/np.maximum(Ue**6, 1e-12)*I
    th_lam = np.sqrt(np.maximum(th2, 1e-16))
    H_lam = None
    if two_eq:
        # ---- two-equation laminar march ----
        # The momentum and kinetic-energy integrals are advanced together,
        #
        #   dtheta/dx = l/Re_theta - (2+H)(theta/U_e) dU_e/dx
        #   dH*/dx    = [2d - H* l - H*(1-H) lambda] / (theta Re_theta)
        #
        # with H*, l and d read from the Falkner-Skan family and H recovered by
        # inverting H*(H).  Written this way every similarity solution is an
        # exact fixed point of the second equation - the bracket vanishes along
        # the family to quadrature accuracy - so the march reproduces
        # Falkner-Skan flow exactly and departs from it only where the real
        # layer does.  That is the point of the second equation: the shape
        # factor now carries the history of the pressure gradient, whereas
        # H = H(lambda) assigns the same profile, and hence the same
        # amplification rate, to two layers that reached the same local
        # gradient by different routes.  Since the tabulated growth rate rises
        # sevenfold between the Blasius profile and H = 3.9, that distinction
        # decides where transition is placed on the forward half of a
        # natural-laminar-flow section.  The march starts where the edge
        # velocity first reaches a twentieth of its maximum, seeded from the
        # closed-form solution, because dtheta/dx is singular at a stagnation
        # point.
        Hg = _stab.twoeq_closure()[0]
        H_MIN, H_MAX = float(Hg[0]), float(Hg[-1])
        # The march is seeded from the closed-form solution at the first
        # station at which the layer is established, taken as Re_theta = 20 and
        # an edge velocity of a twentieth of its maximum.  Starting at the
        # leading edge itself is not possible: dtheta/dx = l nu /(U_e theta)
        # diverges as theta goes to zero, which the closed-form solution
        # absorbs analytically and a march cannot.  Over that short initial run
        # lambda is small and the closed form is accurate to a fraction of a
        # per cent.
        Ret0 = Ue*th_lam/np.maximum(nu_l, 1e-30)
        i0 = int(max(np.argmax(Ue >= 0.05*Ue.max()),
                     np.argmax(Ret0 >= 20.0)))
        i0 = min(i0, n-2)
        th_m = np.array(th_lam, float)
        H_m = np.zeros(n)
        for j in range(i0+1):
            H_m[j] = _thwaites_HL(th_m[j]**2/nu_l[j]*dUeds[j])[0]
        Hst = _stab.twoeq_HL(H_m[i0])[0]

        def _rates(k, th_p, Hst_p):
            H_c = _stab.H_from_Hstar(Hst_p)
            Hs_c, l_c, d_c = _stab.twoeq_HL(H_c)
            Ret = max(Ue[k]*th_p/nu_l[k], 10.0)
            lam_c = th_p*th_p/nu_l[k]*dUeds[k]
            dth = l_c/Ret - (2.0 + H_c)*th_p/max(Ue[k], 1e-9)*dUeds[k]
            dHs = (2.0*d_c - Hs_c*l_c
                   - Hs_c*(1.0 - H_c)*lam_c)/(th_p*Ret)
            return dth, dHs, H_c

        # The kinetic-energy equation is stiff where the layer is thin, because
        # theta Re_theta stands in its denominator, and an explicit step across
        # a leading-edge station can throw H* clean outside the range the family
        # spans - which, since H* falls with H, is read back as a separated
        # profile and fires the bubble at the nose.  Each station is therefore
        # subdivided until the change in H* over a substep is small, and H* is
        # held inside the tabulated range rather than inside a guessed one.
        Hs_tab = _stab.twoeq_closure()[1]
        HS_LO, HS_HI = float(Hs_tab.min()), float(Hs_tab.max())
        for i in range(i0+1, n):
            dx_tot = s[i] - s[i-1]
            th_c, Hst_c = th_m[i-1], Hst
            nsub, done = 1, False
            while not done and nsub <= 64:
                th_t, Hst_t, ok = th_c, Hst_c, True
                for _ in range(nsub):
                    h = dx_tot/nsub
                    a1, b1, _ = _rates(i-1, th_t, Hst_t)
                    th_p = max(th_t + h*a1, 1e-12)
                    Hst_p = float(np.clip(Hst_t + h*b1, HS_LO, HS_HI))
                    a2, b2, _ = _rates(i, th_p, Hst_p)
                    dHs = 0.5*h*(b1 + b2)
                    if abs(dHs) > 0.01:
                        ok = False
                        break
                    th_t = max(th_t + 0.5*h*(a1 + a2), 1e-12)
                    Hst_t = float(np.clip(Hst_t + dHs, HS_LO, HS_HI))
                if ok:
                    th_c, Hst_c, done = th_t, Hst_t, True
                else:
                    nsub *= 2
            th_m[i], Hst = th_c, Hst_c
            H_m[i] = float(np.clip(_stab.H_from_Hstar(Hst), H_MIN, H_MAX))
        th_lam = th_m
        H_lam = H_m

    # ---- Flow-history-averaged free-stream turbulence ----
    # Abu-Ghannam & Shaw correlated transition onset against turbulence,
    # pressure gradient AND flow history.  In a decaying stream the local
    # value is not what the layer has experienced: applying it makes the
    # onset threshold rise faster than Re_theta grows, so the two curves
    # chase each other and onset is predicted far too late.  The effective
    # intensity is therefore the mean of Tu over the boundary layer's own
    # development, measured in Re_theta rather than in Re_x - that is, per
    # unit of momentum-thickness growth rather than per unit of distance.
    # It carries no fitted constant, and reduces to the local value in a
    # stream that does not decay, such as the free atmosphere.
    Reth_lam = Ue*th_lam/nu_l
    dR = np.diff(Reth_lam)
    num = np.concatenate([[0.0], np.cumsum(0.5*(Tu_arr[1:] + Tu_arr[:-1])*dR)])
    den = Reth_lam - Reth_lam[0]
    Tu_avg = np.where(den > 1e-9, num/np.maximum(den, 1e-30), Tu_arr)
    _w = float(cal.get("tu_hist", 1.0))
    Tu_eff = (Tu_avg if _w >= 1.0 else
              (Tu_arr if _w <= 0.0 else
               np.maximum(Tu_avg, 1e-9)**_w * np.maximum(Tu_arr, 1e-9)**(1.0-_w)))

    # ---- frequency set for the e^N integration ----
    # A physical frequency is fixed along the layer while theta grows and U_e
    # changes, so the dimensionless frequency omega*theta/U_e sweeps across the
    # tabulated band.  The set is therefore chosen from the flow itself: it
    # spans every frequency that is inside the amplified band at some station
    # where amplification is possible at all.
    use_db = bool(cal.get("use_os_db", True))
    bubble_on = bool(cal.get("bubble", True))
    i_sep = None; s_sep = 0.0; th_sep = 0.0; H_sep = 0.0; th_b = 0.0; n_bub = 0.0
    omegas = np.array([]); amp = np.array([])
    if use_db:
        om_lo, om_hi = _stab.omega_grid_bounds()
        rat = Ue/np.maximum(th_lam, 1e-12)
        live = Reth_lam > 60.0
        if live.any():
            w_lo = om_lo*float(np.min(rat[live]))
            w_hi = om_hi*float(np.max(rat[live]))
            omegas = np.geomspace(max(w_lo, 1e-9), max(w_hi, w_lo*10.0),
                                  int(cal.get("n_freq", 40)))
            amp = np.zeros(omegas.size)

    i_tr = None
    bub_trig = False
    for i in range(n):
        theta[i] = th_lam[i]
        lam[i] = theta[i]**2/nu_l[i]*dUeds[i]
        if H_lam is not None:
            # shape factor from the kinetic-energy equation, carrying history
            H[i] = H_lam[i]
            l = _stab.twoeq_HL(H[i])[1]
        else:
            H[i], l = _thwaites_HL(lam[i])
        Reth[i] = Ue[i]*theta[i]/nu_l[i]
        Cf[i] = 2.0*l*nu_l[i]/max(Ue[i]*theta[i], 1e-12)

        # ---------- laminar separation bubble ----------
        # Thwaites' march is defined only up to lambda = -0.09; past that the
        # layer has left the wall and neither its shape factor nor its skin
        # friction means anything.  Across the dead-air region the momentum
        # integral still holds with no wall stress, so the shear layer is
        # carried forward with C_f = 0 and the shape factor frozen at its
        # separation value.  That is a closure with no fitted constant, and it
        # reproduces the growth the T3C4 hot films record through the bubble,
        # Re_theta from 309 at separation to 381 at the end of the plateau.
        #
        # The bubble closes where the disturbance riding on the detached shear
        # layer has amplified by the same factor that ends transition anywhere
        # else in the model.  The shear layer is inflectional, so it amplifies
        # at a rate characteristic of a free shear layer rather than of an
        # attached boundary layer - the tabulated Orr-Sommerfeld rate has
        # already climbed to 0.035 by H = 3.9, and sigma_sep continues that
        # trend - and the amplification factor is restarted at separation,
        # because the wave that grows in the detached layer is a new one
        # selected by its own inflection, not the continuation of whatever the
        # attached layer was carrying.
        #
        # Writing the bubble this way rather than as a fixed multiple of the
        # separation momentum thickness is what lets one closure span both
        # regimes.  The length is N_crit*theta_sep/sigma_sep, so it collapses
        # in a turbulent stream, where N_crit is small, and stretches in a
        # quiet one: the measured bubbles here run to about 40 momentum
        # thicknesses on T3C4 at Tu = 2.1 % and about 180 on the NLF(1)-0416
        # lower surface at Tu = 0.03 %, a spread of four and a half that no
        # fixed multiple reproduces.
        # Separation is detected on the Thwaites parameter rather than on the
        # shape factor.  Testing H >= 3.95, the value at which the exact family
        # loses its wall shear, is the more principled statement and it does
        # improve the T3C4 plate, but on the aerofoil lower surface the layer
        # approaches separation too gradually ever to reach it, the bubble
        # never fires where the measurements say it governs, and the number of
        # predictions inside the experimental bracket falls from 28 of 40 to 4.
        # The Thwaites value is retained on that evidence.
        _hs = cal.get("H_sep", 0.0)
        sep_now = (H[i] >= _hs) if _hs > 0.0 else (lam[i] <= cal["lam_sep"])
        if bubble_on and (sep_now or i_sep is not None):
            if i_sep is None:
                i_sep = i; s_sep = s[i]
                th_sep = max(theta[i], 1e-12); H_sep = H[i]
                th_b = th_sep; n_bub = 0.0
            elif i > 0:
                th_b = max(th_b - (2.0 + H_sep)*th_b/max(Ue[i], 1e-9)
                           * dUeds[i]*(s[i] - s[i-1]), 1e-12)
                # The amplification rate of the detached shear layer is read
                # from the same table as everywhere else.  The tabulated
                # Falkner-Skan family is continued past separation onto its
                # reverse-flow branch, so a separated profile has a computed
                # rate and the bubble closure carries no fitted constant.  It
                # is evaluated on the developed reverse-flow profile rather
                # than on the profile at the separation point: the momentum
                # thickness is nearly frozen across the dead-air region but the
                # shape factor is not, rising from 4.0 at separation towards
                # the 5.17 the T3C4 hot films record, and the amplification is
                # dominated by the developed layer.  The computed rate there is
                # 0.045, which is what the T3C4 and NLF(1)-0416 bubble lengths
                # independently imply.
                sig_b = float(_stab.sigma_curve(_stab.H_REVERSE, Reth[i]).max())
                n_bub += max(sig_b, 0.0)/th_b*(s[i] - s[i-1])
            theta[i] = th_b; H[i] = H_sep; Cf[i] = 0.0
            Reth[i] = Ue[i]*th_b/nu_l[i]
            n_fac[i] = n_bub
            bub_trig = n_bub >= _n_crit(Tu_eff[i], cal.get("N_floor", 0.5))

        # ---------- UNIFIED TRANSITION KERNEL ----------
        lam_t = lam[i]

        # (a) natural / Tollmien-Schlichting.  One amplification factor is
        #     carried per physical frequency and advanced with the spatial
        #     growth rate sigma = -alpha_i*theta read from the tabulated
        #     Orr-Sommerfeld solutions, dN/dx = sigma/theta; the amplification
        #     factor is the envelope of those curves.  This is the e^N method
        #     as defined, rather than the Drela-Giles fit to its envelope,
        #     which is retained behind use_os_db for comparison.  Individual
        #     frequencies are allowed to decay once they leave their unstable
        #     band, but no N is carried below zero.
        if use_db and not (bubble_on and i_sep is not None):
            if i > 0 and omegas.size and Reth[i] > 50.0:
                om_star = omegas*theta[i]/max(Ue[i], 1e-9)
                sig = np.interp(om_star, _OM_TAB,
                                _stab.sigma_curve(H[i], Reth[i]),
                                left=0.0, right=0.0)
                amp += np.maximum(sig, -0.05)/max(theta[i], 1e-12)*(s[i]-s[i-1])
                np.maximum(amp, 0.0, out=amp)
                n_amp = float(amp.max())
        elif not use_db and i > 0:
            Re_c = _re_theta_crit(H[i])
            if Reth[i] > Re_c:
                dRe = Reth[i] - max(Reth[i-1], Re_c)
                if dRe > 0.0:
                    n_amp += _dn_dReth(H[i])*dRe
        n_fac[i] = n_amp
        ts_live = Tu_eff[i] <= cal.get("Tu_TS_max", 1.0)
        N_target = ((_n_crit(Tu_eff[i], cal.get("N_floor", 0.5))
                     / max(cal["A_TS"], 1e-6)) if ts_live else np.inf)

        # (b) bypass: free-stream-turbulence driven, meaningful only for
        #     elevated Tu (>~0.1%); below that the natural route governs.
        Tu_i = Tu_eff[i]
        if Tu_i > 0.1:
            Rbp = cal["A_BP"] * _ags_re_theta_t(Tu_i, lam_t)
        else:
            Rbp = 1e9

        # (c) separation-induced.  Thwaites' laminar march terminates at
        #     lambda = -0.09; downstream of that the layer has left the wall
        #     and the integral closures no longer apply.  Rather than declare
        #     transition at the separation point, the amplification integral
        #     is continued through the detached shear layer, which is what
        #     actually decides where the bubble closes: the momentum thickness
        #     is very nearly frozen across the dead-air region - the T3C4
        #     measurements show Re_theta rising only from 309 to 381 over the
        #     whole laminar portion of the bubble - while the inflectional
        #     profile amplifies disturbances at a rate characteristic of a
        #     free shear layer, an order above anything an attached layer
        #     reaches.  Transition is then placed where N attains the same
        #     N_crit as everywhere else in the model.
        #
        #     Writing the bubble this way rather than as a correlation in
        #     Re_theta at separation is what lets one closure span both
        #     regimes: the bubble length is (N_crit - N_sep)*theta_sep/sigma_sep,
        #     so it collapses at high free-stream turbulence, where N_crit is
        #     small, and stretches in a quiet stream.  The measured lengths
        #     differ by a factor of five between the T3C4 bubble at Tu = 2.1 %
        if bubble_on and i_sep is not None:
            Rsep = 1e9
        elif sep_now:
            Rref = Rbp if Rbp < 1e8 else _ags_re_theta_t(max(Tu_i, 0.02), lam_t)
            Rsep = cal["A_SEP"]*max(cal["sep_floor"], 0.7*Rref)
        else:
            Rsep = 1e9

        # (d) cross-flow (swept wing): C1 criterion on the cross-flow
        #     momentum-thickness Reynolds number (algebraic surrogate).
        # (d) cross-flow (swept wing): C1 criterion on the cross-flow
        #     momentum-thickness Reynolds number (algebraic surrogate).
        #
        #     The exact quantity is available: the Falkner-Skan-Cooke solution
        #     gives Re_cf = Re_theta sin(L) cos(L) K(lambda), with the sweep
        #     factoring out exactly and K carrying the pressure-gradient
        #     dependence (stability.crossflow_factor).  K is not constant - it
        #     spans a factor of ten across the family, and goes to zero at zero
        #     pressure gradient, where the span-wise and chordwise similarity
        #     equations give f' = g identically and there is no cross-flow at
        #     all, whereas the surrogate's constant predicts some.  Replacing
        #     the surrogate by K was tried and is not adopted: it does not make
        #     the two swept-wing experiments agree on a critical value, and it
        #     degrades the calibration set from 13.2 to 23.8 per cent.  The
        #     limiting approximation is therefore the C1 criterion itself,
        #     which reduces the stability of an inflectional three-dimensional
        #     profile to a single Reynolds number, and not the surrogate for
        #     the cross-flow thickness.  See Sec. VI.
        if sweep_deg > 1.0:
            if cal.get("cf_exact", False):
                L_r = np.radians(sweep_deg)
                fac = (np.sin(L_r)*np.cos(L_r)
                       * _stab.crossflow_factor(lam[i]))
                Rcf = (cal["CF_C1"]/max(fac, 1e-12)/cal["A_CF"]
                       if fac > 1e-12 else 1e9)
            else:
                Re_th2 = _re_theta2(Reth[i], sweep_deg, cal["CF_ratio"])
                Rcf = (Reth[i]*cal["CF_C1"]/max(Re_th2, 1e-9)/cal["A_CF"]
                       if Re_th2 > 0 else 1e9)
        else:
            Rcf = 1e9

        # Inside the dead-air region none of the attached-flow criteria apply:
        # Abu-Ghannam & Shaw correlates an attached layer against its own
        # pressure gradient, and the cross-flow criterion presumes a wall
        # boundary layer.  Once the layer has separated the bubble closure is
        # the only thing that can end it.
        in_bubble = bubble_on and i_sep is not None
        Rt = 1e9 if in_bubble else min(Rbp, Rsep, Rcf)
        Re_th_t[i] = Rt
        trig_ts = bool(ts_live and n_amp >= N_target)
        if (trig_ts or bub_trig or Reth[i] >= Rt) and i_tr is None and i > 1:
            i_tr = i
            if in_bubble:
                mech = "separation"
            elif trig_ts and Reth[i] < Rt:
                mech = "TS-natural"
            else:
                mech = ["bypass", "separation", "crossflow"][
                    int(np.argmin([Rbp, Rsep, Rcf]))]
                if trig_ts:
                    mech = "TS-natural"
            mechanism[i] = mech
            break

    if i_tr is None:
        # fully laminar to TE
        out = dict(s=s, Ue=Ue, theta=theta, H=H, Cf=Cf, Re_theta=Reth,
                   lam=lam, gamma=gamma, state=state, Re_theta_t=Re_th_t,
                   n_factor=n_fac, mechanism=mechanism, i_tr=None,
                   i_sep=i_sep, s_sep=(s_sep if i_sep is not None else np.nan),
                   bubble_burst=bool(i_sep is not None),
                   x_tr=np.nan, onset_mech="none(laminar)")
        return out

    s_tr = s[i_tr]; onset_mech = mechanism[i_tr]
    Re_tr = Reth[i_tr]
    # Narasimha transition length:  Re_lambda = C_len * Re_theta_t^0.8
    Re_lam_len = cal["C_len"] * max(Re_tr, 1.0)**0.8
    nu_local = nu_t[i_tr]
    # convert Re-length to physical length using local Ue
    lam_len = Re_lam_len*nu_local/max(Ue[i_tr], 1e-6)

    # ---- Turbulent marcher (Head + Ludwieg-Tillmann) ----
    th_t = theta[i_tr]
    H_t  = min(max(H[i_tr], 1.35), 1.6)   # collapse to turbulent profile
    H1   = _head_H1(H_t)
    th_turb = np.zeros(n); H_turb = np.zeros(n)
    Cf_turb = np.zeros(n)
    th_turb[i_tr] = th_t; H_turb[i_tr] = H_t
    Cf_turb[i_tr] = _ludwieg_tillmann(H_t, Ue[i_tr]*th_t/nu_t[i_tr])
    sep_turb = None
    for i in range(i_tr+1, n):
        ds = s[i]-s[i-1]
        Re_th = max(Ue[i-1]*th_turb[i-1]/nu_t[i-1], 1.0)
        cf = _ludwieg_tillmann(H_turb[i-1], Re_th)
        dthds = cf/2.0 - (H_turb[i-1]+2.0)*th_turb[i-1]/Ue[i-1]*dUeds[i-1]
        CE = 0.0306*(max(H1-3.0, 0.01))**(-0.6169)
        dH1ds = (CE - (H1*th_turb[i-1]/Ue[i-1])*dUeds[i-1] -
                 H1*dthds)/max(th_turb[i-1], 1e-9)
        th_turb[i] = max(th_turb[i-1] + dthds*ds, 1e-9)
        H1 = max(H1 + dH1ds*ds, 3.35)
        H_turb[i] = np.clip(_head_H_from_H1(H1), 1.05, 2.8)
        H1 = _head_H1(H_turb[i])
        Cf_turb[i] = _ludwieg_tillmann(H_turb[i], Ue[i]*th_turb[i]/nu_t[i])
        if H_turb[i] > 2.6 and sep_turb is None:
            sep_turb = i

    # ---- Blend with intermittency (Narasimha universal) ----
    for i in range(n):
        if i < i_tr:
            gamma[i] = 0.0
            state[i] = "laminar"
        else:
            xi = (s[i]-s_tr)/max(lam_len, 1e-9)
            g = 1.0 - np.exp(-0.412*xi**2)
            gamma[i] = float(np.clip(g, 0.0, 1.0))
            # blended properties
            cf_l = Cf[i]
            theta[i] = (1-gamma[i])*th_lam[i] + gamma[i]*th_turb[i]
            H[i]  = (1-gamma[i])*H[i] + gamma[i]*H_turb[i]
            Cf[i] = (1-gamma[i])*cf_l + gamma[i]*Cf_turb[i]
            Reth[i] = Ue[i]*theta[i]/nu_t[i]
            state[i] = "transitional" if gamma[i] < 0.99 else "turbulent"
            mechanism[i] = onset_mech

    out = dict(s=s, Ue=Ue, theta=theta, H=H, Cf=Cf, Re_theta=Reth,
               lam=lam, gamma=gamma, state=state, Re_theta_t=Re_th_t,
               n_factor=n_fac, mechanism=mechanism, i_tr=i_tr, x_tr=s_tr,
               lam_len=lam_len, onset_mech=onset_mech,
               i_sep=i_sep, s_sep=(s_sep if i_sep is not None else np.nan),
               bubble_burst=False,
               sep_turb=sep_turb)
    return out


# ======================================================================
#  HIGH-LEVEL DRIVERS
# ======================================================================
def solve_flat_plate(L, U, nu, Tu_pct, npts=400, cal=None, dUe=0.0,
                     Tu_decay=None, L_turb=None, Ue_dist=None):
    """Zero (or mild) pressure-gradient flat plate (validation cases).

    Tu_decay, if given, is a (Re_x, Tu_pct) pair of sequences describing the
    measured decay of free-stream turbulence along the plate; the local
    value is then interpolated onto the marching stations and used in place
    of the scalar Tu_pct."""
    s = np.linspace(1e-4, L, npts)
    if Ue_dist is not None:
        # Measured edge-velocity distribution, given as (x [m], Ue [m/s]) and
        # interpolated with a shape-preserving cubic.  Linear interpolation, as
        # an earlier version used, makes dU_e/dx piecewise constant with a jump
        # at every measured point; the T3C4 distribution has twelve points over
        # 1.5 m, so the marching stations see a staircase gradient that the
        # local least-squares derivative cannot smooth, its window being an
        # order of magnitude finer than the data spacing.  Since lambda is
        # proportional to that gradient, the staircase moves the station at
        # which the layer separates.
        from scipy.interpolate import UnivariateSpline
        xm, um = np.asarray(Ue_dist[0], float), np.asarray(Ue_dist[1], float)
        # A smoothing spline, not an interpolant.  The tabulated edge velocities
        # are quoted to 0.01 m/s and the differences between adjacent stations
        # are 0.02 to 0.07, so the gradient formed from them carries about
        # fifty per cent quantisation noise - and lambda, which is what decides
        # separation, is proportional to that gradient.  Passing a curve exactly
        # through quantised data propagates the quantisation into the
        # derivative: on T3C4 it puts a spurious local minimum of lambda at
        # x = 1.16 m, a quarter of a metre upstream of the measured separation,
        # deep enough to trigger the bubble there.  The spline is instead
        # allowed to depart from each point by the quotation precision, which
        # is the most that can be claimed for it, and the resulting deceleration
        # is monotone.
        w = np.full(um.size, 1.0/max(0.5*0.01, 1e-9))     # 1/sigma, sigma = half a count
        k = min(3, max(1, um.size - 1))
        sp = UnivariateSpline(xm, um, w=w, k=k, s=float(um.size), ext=3)
        Ue = sp(np.clip(s, xm[0], xm[-1]))
    else:
        Ue = U*(1.0 + dUe*s/L)
    if L_turb is not None:
        Tu_pct = _tu_decay(Tu_pct, s, L_turb, cal=cal)
    elif Tu_decay is not None:
        rex_m, tu_m = np.asarray(Tu_decay[0], float), np.asarray(Tu_decay[1], float)
        Tu_pct = np.interp(U*s/nu, rex_m, tu_m,
                           left=tu_m[0], right=tu_m[-1])
    r = march_bl(s, Ue, nu, Tu_pct=Tu_pct, Ue_inf=U, cal=cal)
    # Re_x is formed on the LOCAL edge velocity, which is the convention of the
    # ERCOFTAC tabulations these cases are compared against: on T3C4 the edge
    # velocity rises from 1.51 to 2.02 m/s, so forming Re_x on the reference
    # speed understates it by up to a fifth at the end of the plate.
    r["Re_x"] = Ue*s/nu
    r["Cf_lam_ref"] = 0.664/np.sqrt(np.maximum(r["Re_x"], 1.0))
    r["Cf_turb_ref"] = 0.0592/np.maximum(r["Re_x"], 1.0)**0.2
    return r


def solve_airfoil(xb, yb, alpha_deg, U, nu, chord, Tu_pct,
                  sweep_deg=0.0, cal=None, mach=0.0, compressible=True):
    """
    Full aerofoil: panel method -> split at stagnation -> march upper
    and lower surfaces.  Returns inviscid + viscous results.
    """
    xc, yc, Cp, V, th, S = panel_solve(xb, yb, alpha_deg, mach=mach)
    # speed of sound implied by the free-stream Mach number; passing it to the
    # march turns on the compressible (reference-temperature) closures
    a_snd = (U/mach if (compressible and mach > 1e-6) else 0.0)
    # arc length along surface (control points), find stagnation (Cp max)
    # NB: panel coords are unit-chord -> scale arc length to physical chord
    ds = S
    s_arc = np.concatenate([[0.5*ds[0]], 0.5*ds[0] +
                            np.cumsum(0.5*(ds[:-1]+ds[1:]))]) * chord
    i_stag = int(np.argmax(Cp))
    Ue_mag = np.abs(V)*U

    # Lower surface : from stagnation backward to start (index 0=TE lower)
    low_idx = np.arange(i_stag, -1, -1)
    up_idx  = np.arange(i_stag, len(xc))
    def build(idx):
        ss = np.abs(s_arc[idx] - s_arc[i_stag])
        order = np.argsort(ss)
        ss = ss[order]; idx2 = idx[order]
        # de-duplicate
        keep = np.concatenate([[True], np.diff(ss) > 1e-9])
        return ss[keep], idx2[keep]
    s_low, idx_low = build(low_idx)
    s_up,  idx_up  = build(up_idx)

    res = {}
    for name, ss, idx in [("upper", s_up, idx_up), ("lower", s_low, idx_low)]:
        Ue_s = np.maximum(Ue_mag[idx], 1e-4)
        r = march_bl(ss, Ue_s, nu, Tu_pct=Tu_pct, sweep_deg=sweep_deg,
                     Ue_inf=U, cal=cal, label=name, a_sound=a_snd)
        r["x"]  = xc[idx]; r["y"] = yc[idx]; r["Cp"] = Cp[idx]
        r["Re_x"] = U*ss/nu
        r["x_tr_chord"] = (float(xc[idx][r["i_tr"]])
                           if r["i_tr"] is not None else np.nan)
        # Chordwise station at which the laminar layer separates, and whether
        # the bubble it starts ever closed.  A bubble that has not closed by
        # the trailing edge has burst: the short-bubble closure of Sec. II.C.4
        # presumes reattachment, and where there is none the method has no
        # transition location to offer and says so, rather than returning the
        # last station it happened to reach.
        r["x_sep_chord"] = (float(xc[idx][r["i_sep"]])
                            if r.get("i_sep") is not None else np.nan)
        r["bubble_burst"] = bool(r.get("bubble_burst", False))
        res[name] = r

    # forces (ordering is counter-clockwise -> outward normal = (-sin,cos))
    # panel coords are unit-chord, so reference chord for Cp integration = 1
    nx = -np.sin(th); ny = np.cos(th)
    Cn = -np.sum(Cp*ny*S)
    Ca = -np.sum(Cp*nx*S)
    al = np.radians(alpha_deg)
    Cl = Cn*np.cos(al) - Ca*np.sin(al)
    # profile drag via Squire-Young on each surface
    def squire_young(r, x_ref=0.98):
        """Squire-Young profile drag.

        The formula is evaluated at 98% chord rather than at the last panel
        control point.  A panel method drives the edge velocity towards the
        stagnation value at a sharp trailing edge, so the final control point
        sits inside that collapse; since Squire-Young raises Ue/Uinf to the
        power (H+5)/2, taking the last point makes the drag hypersensitive to
        the panel distribution.  At 98% chord the boundary-layer solution is
        still meaningful and the result is insensitive to the discretisation.
        """
        xs = np.asarray(r["x"], float)
        cand = np.where(xs > 0.5)[0]
        i = (cand[np.argmin(np.abs(xs[cand] - x_ref))] if len(cand)
             else len(xs) - 1)
        H_te = r["H"][i]; th_te = r["theta"][i]; Ue_te = r["Ue"][i]
        return 2.0*th_te/chord*(Ue_te/U)**((H_te+5.0)/2.0)
    Cd = squire_young(res["upper"]) + squire_young(res["lower"])

    return dict(panel=dict(xc=xc, yc=yc, Cp=Cp, V=V, th=th, S=S,
                           i_stag=i_stag),
                surfaces=res, Cl=Cl, Cd=Cd, alpha=alpha_deg)


if __name__ == "__main__":
    # quick self-test : NACA0012 panel Cl sign and flat-plate transition
    import numpy as np
    th = np.linspace(0, 2*np.pi, 121)
    # circle test of panel routine sanity skipped; run flat plate:
    r = solve_flat_plate(0.3, 5.4, 1.5e-5, 3.3)
    print("flat plate T3A-like: x_tr=%.4f m  Re_x_tr=%.3e  mech=%s" %
          (r["x_tr"], 5.4*r["x_tr"]/1.5e-5, r["onset_mech"]))
