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
 2. Laminar boundary layer    : two-equation integral march (momentum
                               and kinetic energy) closed on the
                               Falkner-Skan family, so the shape factor
                               carries its own history.  Thwaites'
                               one-equation method is retained behind
                               cal["two_eq"] = False for comparison.
 3. UNIFIED TRANSITION KERNEL : the novel contribution. A single
       onset is taken as the minimum effective transition-Re across
       FOUR co-resident mechanisms, each with a calibration weight:
         (a) Tollmien-Schlichting / natural   (e^N per frequency, from
             the tabulated Orr-Sommerfeld rates of stability.py; the
             Drela-Giles envelope fit is retained behind
             cal["use_os_db"] = False)
         (b) Bypass (free-stream turbulence)   (Abu-Ghannam & Shaw)
         (c) Laminar-separation-induced        (bubble closed by the
             amplification integral across the dead-air region)
         (d) Cross-flow (swept-wing)           (C1 threshold followed by
             a cross-flow amplification integral)
 4. Transitional region       : Narasimha universal-intermittency
       closure; properties blended Cf = (1-g)Cf_lam + g Cf_turb.
 5. Turbulent boundary layer  : Head's entrainment method with the
       Ludwieg-Tillmann skin-friction law.
 6. Drag                      : Squire-Young far-wake formula.

The natural and bypass branches are independent models - an
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
    N_crit    = 9.0,    # nominal reference e^N factor, carried for reference
                        # only: no branch reads it.  The value actually used at
                        # every station is obtained from the local Tu by Mack's
                        # relation (see _n_crit), which is clamped to its stated
                        # validity range and so returns 8.18 for any stream
                        # quieter than 0.08 %.
    N_floor   = 0.5,    # lower clamp on N_crit at high Tu
    C_mu      = 0.09,   # k-epsilon constants, used only for the decay of
    C_eps2    = 1.92,   # free-stream turbulence (see _tu_decay)
    Tu_BP_lo  = 0.10,   # Tu [%] below which the bypass correlation carries no
    Tu_BP_hi  = 0.25,   # weight, and above which it carries all of it; between
                        # them the two closures are blended (see
                        # _branch_weight and the kernel).  These replace the
                        # single Tu_TS_max = 0.1 gate, which switched the e^N
                        # branch off and the correlation on at the same value
                        # and so made the predicted transition location a STEP
                        # function of the free-stream turbulence: 0.1000 % and
                        # 0.1001 % differed by 0.17c and by a third of the
                        # profile drag.  The window is deliberately wider than
                        # the spread of any case in this study - the noisiest
                        # natural case is 0.07 % and the quietest bypass case
                        # 0.87 % - so every result here is computed at one end
                        # or the other and none is blended.  The window is what
                        # makes the model a function of Tu rather than a
                        # switch, not a constant fitted to anything.
    C_len     = 9.0,    # Dhawan & Narasimha's transition-length correlation,
                        # Re_lambda = C_len * Re_x_t^0.75, with Re_x_t formed on
                        # the streamwise distance from the leading edge (or the
                        # stagnation point) and the local edge velocity, and
                        # lambda the distance over which the intermittency goes
                        # from 0.25 to 0.75.  9.0 is their published value and
                        # is not fitted here.  An earlier version wrote the
                        # correlation as 6.5*Re_theta_t^0.8, which returns
                        # Re_lambda ~ 600 where the ERCOFTAC plates need ~4e4:
                        # the layer went from fully laminar to fully turbulent
                        # inside one marching station, and the predicted C_f
                        # jumped vertically where the measurements climb over
                        # half a decade of Re_x.  The transition LOCATION was
                        # unaffected - onset is where the kernel fires, not
                        # where the blend ends - but everything downstream of
                        # onset was.
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
    bub_rev_H = False,  # let the shape factor in the dead-air momentum equation
                        # follow the reverse-flow profile the amplification rate
                        # is already read from, instead of being capped at the
                        # attached Falkner-Skan separation value (see march_bl)
    H_sep     = 0.0,    # if positive, laminar separation is declared where the
                        # solved shape factor reaches this value instead of
                        # where the Thwaites parameter reaches lam_sep.  With
                        # H carrying its own history the shape factor is the
                        # physically meaningful indicator; lam is a local
                        # quantity and its threshold belongs to the
                        # one-parameter method it was fitted for.
    cf_amp    = True,   # close the cross-flow branch with an amplification
                        # integral rather than a local threshold.  A stationary
                        # cross-flow vortex has to grow before it breaks down;
                        # a criterion that fires the instant a local Reynolds
                        # number is exceeded places transition at the first
                        # station that is unstable, not at the first that has
                        # amplified enough, and the two differ by a factor of
                        # two on the wing of Sec. IV.C.  The rate is the
                        # computed shear-layer value of Sec. II.C.4 - a
                        # cross-flow profile is inflectional, like a separated
                        # one - and the threshold is the same N_crit every
                        # other branch uses, so the integral form adds no
                        # constant to the one the criterion already had.
    CF_N      = 0.0,    # if positive, the amplification factor at which the
                        # cross-flow branch fires, in place of the N_crit the
                        # other branches take from Mack's relation.  Mack
                        # correlated TS waves against free-stream turbulence,
                        # and a stationary cross-flow vortex is seeded by
                        # leading-edge roughness instead, so there is no reason
                        # the two thresholds should coincide; this exposes the
                        # cross-flow one so that the question can be settled by
                        # measurement rather than by assuming they do.
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
    d_omega   = 1.03,   # MAXIMUM ratio between adjacent physical frequencies
                        # carried by the e^N integration.  The amplification
                        # factor is the envelope of the individual N(omega)
                        # curves, so the frequency set is a discretisation and
                        # not a fitted constant - but it has to be stated as a
                        # resolution rather than as a count.  A fixed count was
                        # spread geometrically over the whole surface's range
                        # of U_e/theta, which on a long plate spans three
                        # decades, so 40 frequencies left only a handful inside
                        # the amplified band at any one station: onset on the
                        # Schubauer-Skramstad plate moved 5 per cent between 24
                        # and 64 frequencies and did not settle, against the
                        # "under 0.2 %" this comment used to claim.  That claim
                        # had been checked on the aerofoil, where the answer is
                        # quantised to the panel it lands on and a 0.2 % change
                        # is invisible.  Fixing the SPACING instead makes the
                        # count follow the flow, and the envelope converges.
    n_freq_min= 40,     # floor and cap on the resulting count, so a degenerate
    n_freq_max= 400,    # range cannot produce a set of two or of a million
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


def _branch_weight(Tu_pct, cal=None):
    """How much of the bypass correlation applies at this turbulence level.

    0 below Tu_BP_lo, 1 above Tu_BP_hi, and a smoothstep between.  Below the
    lower edge the layer is in the range the amplification integral was built
    for and Abu-Ghannam & Shaw is extrapolated outside its data; above the
    upper edge the reverse, because Mack's relation returns a critical
    amplification factor of about 2 there and the envelope method is not
    calibrated that low.  Between them both closures have some claim, and the
    weight interpolates rather than choosing.

    A single threshold - the old cal["Tu_TS_max"] - made this a step, and the
    step was visible in the answer: see the note in the kernel.  The smoothstep
    3t^2 - 2t^3 is used because it is continuous in value AND in slope at both
    edges, so no case sitting near an edge picks up a kink.
    """
    cal = {**CAL, **(cal or {})}
    lo = float(cal.get("Tu_BP_lo", 0.10))
    hi = float(cal.get("Tu_BP_hi", 0.25))
    if hi <= lo:
        return 0.0 if Tu_pct <= lo else 1.0
    t = (float(Tu_pct) - lo)/(hi - lo)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t*t*(3.0 - 2.0*t)


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


def _re_theta2(Re_theta, sweep_deg, ratio, normal_frame=False):
    """Cross-flow momentum-thickness Reynolds number, algebraic surrogate.

    A three-dimensional boundary-layer solution is not carried, so the
    cross-flow momentum thickness is estimated from the chordwise one.  The
    Falkner-Skan-Cooke solution gives

        Re_cf = Re_theta,n sin(L) K(lambda),

    with Re_theta,n formed on the CHORDWISE edge velocity U_e,n = Q_e cos(L).
    The surrogate replaces K by a constant `ratio`, set on the swept-wing
    transition measurements of Dagenhart & Saric.

    Which sweep factor is right depends on which frame Re_theta arrives in, and
    getting that wrong is a factor of cos(L) - 0.64 at 50 degrees:

      * normal_frame=False: Re_theta is formed on the TOTAL edge velocity, so
        Re_theta,n = Re_theta cos(L) and the factor is sin(L) cos(L).
      * normal_frame=True: the march is already running in the plane normal to
        the leading edge, Re_theta IS Re_theta,n, and the factor is sin(L).

    Applying the normal-plane transformation while keeping the sin(L)cos(L) of
    the first case counts cos(L) twice.  Measured, that mistake alone moved the
    critical value required by the independent 20-50 degree set from a spread
    of 4.6 per cent to 19.4 per cent - which is what a spurious cos(L) does to a
    quantity swept across 30 degrees of sweep."""
    L = np.radians(sweep_deg)
    f = np.sin(L) if normal_frame else np.sin(L)*np.cos(L)
    return ratio*Re_theta*f

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
    # (the vortex strengths are the incompressible ones; the compressibility
    #  correction is applied to C_p below, exactly as in panel_solve)
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
        # Katz & Plotkin (2001), Eq. 10.39-10.40: the normal component carries a
        # minus sign.  It was positive here, so the induced field had the wrong
        # sign in one of its two components; evaluated just off the surface the
        # result did not reproduce the panel solution it came from - C_p at the
        # leading-edge stagnation point came back as -0.50 against the +0.70 of
        # the surface distribution three millimetres away.  With the sign
        # corrected the two agree to 0.012 in C_p at that offset, which is the
        # offset itself.
        up =  gpan[j]/(2*np.pi) * (th2 - th1)
        wp = -gpan[j]/(2*np.pi) * np.log(r1/r2)
        Vx += up*ct - wp*st
        Vy += up*st + wp*ct
    Cp = 1.0 - (Vx**2 + Vy**2)/U**2
    # Same Karman-Tsien correction as panel_solve.  An earlier version applied
    # the linearised Prandtl-Glauert scaling here, so the off-body field and
    # the surface distribution obeyed different compressibility corrections and
    # disagreed by six per cent in C_p at the suction peak at M = 0.42.
    if mach > 1e-6:
        beta = np.sqrt(max(1.0 - mach*mach, 1e-6))
        Cp = Cp/(beta + (mach*mach/(1.0 + beta))*0.5*Cp)
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



def _edge_from_cp(Cp, mach, gamma=1.4):
    """Edge velocity ratio and edge Mach number from the corrected C_p.

    The panel solution returns an INCOMPRESSIBLE surface velocity V and then
    corrects the pressure with Karman-Tsien.  Taking the edge velocity as that
    uncorrected V, as this did, leaves the boundary layer running on a
    different flow from the one the loads are computed from: at M = 0.42 the
    corrected C_p at the suction peak implies U_e/U_inf = 1.3475 while the
    march was handed 1.3090, an error of 2.9 per cent in the edge velocity and
    in its GRADIENT, which is what drives lambda and hence the whole transition
    kernel.  The pressure was compressible and the boundary layer it fed was
    not.

    Both follow from the corrected pressure by the isentropic relations, with
    no new constant.  From C_p,

        p/p_inf = 1 + (gamma/2) M_inf^2 C_p,
        M_e^2   = (2/(gamma-1)) [ (1 + (gamma-1)/2 M_inf^2)
                                  (p/p_inf)^(-(gamma-1)/gamma) - 1 ],
        T_e/T_inf = (1 + (gamma-1)/2 M_inf^2)/(1 + (gamma-1)/2 M_e^2),
        U_e/U_inf = (M_e/M_inf) sqrt(T_e/T_inf).

    M_e formed this way also fixes a second error: the reference-temperature
    closures used to be given M_e = U_e/a_inf, the FREE-STREAM speed of sound,
    when the edge gas is at T_e and the correct divisor is a_e.  Here M_e comes
    out of the pressure directly and never needs a speed of sound at all.

    As M_inf -> 0 this returns U_e/U_inf = sqrt(1 - C_p) and M_e = 0, so the
    incompressible cases are untouched.
    """
    Cp = np.asarray(Cp, float)
    if mach <= 1e-6:
        return np.sqrt(np.maximum(1.0 - Cp, 1e-12)), np.zeros_like(Cp)
    g = gamma
    p_ratio = np.maximum(1.0 + 0.5*g*mach*mach*Cp, 1e-6)
    t0 = 1.0 + 0.5*(g - 1.0)*mach*mach
    Me2 = np.maximum((2.0/(g - 1.0))*(t0*p_ratio**(-(g - 1.0)/g) - 1.0), 0.0)
    Me = np.sqrt(Me2)
    T_ratio = t0/(1.0 + 0.5*(g - 1.0)*Me2)
    return (Me/mach)*np.sqrt(np.maximum(T_ratio, 1e-12)), Me


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


def march_bl(s, Ue, nu, Tu_pct=0.2, sweep_deg=0.0, cal=None, a_sound=0.0,
             Me=None, normal_frame=False):
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
    # Edge Mach number.  Supplied by the caller where the inviscid solution
    # knows it - solve_airfoil gets it from the corrected pressure, see
    # _edge_from_cp - and otherwise formed from a speed of sound.  Forming it
    # as U_e/a_inf, which is what the fallback does, uses the FREE-STREAM speed
    # of sound for gas that is at T_e; it is retained only for callers that
    # have no pressure field, and is exact at M_inf = 0, which is every
    # flat-plate case here.
    if Me is None:
        Me = (Ue/a_sound if a_sound > 1e-6 else np.zeros(n))
    else:
        Me = np.asarray(Me, float)
        if Me.shape != (n,):
            raise ValueError("Me must be one value per station")
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

    # The attached two-equation right-hand side and its station stepper, defined
    # once.  The initial march below uses them, and so does the marching loop
    # when a separation bubble REATTACHES laminar and the attached march has to
    # be resumed from the momentum thickness the dead-air region left it with.
    _Hs_tab0 = _stab.twoeq_closure()[1]
    _HSLO, _HSHI = float(_Hs_tab0.min()), float(_Hs_tab0.max())

    def _rates(k, th_p, Hst_p):
        H_c = _stab.H_from_Hstar(Hst_p)
        Hs_c, l_c, d_c = _stab.twoeq_HL(H_c)
        Ret = max(Ue[k]*th_p/nu_l[k], 10.0)
        lam_c = th_p*th_p/nu_l[k]*dUeds[k]
        dth = l_c/Ret - (2.0 + H_c)*th_p/max(Ue[k], 1e-9)*dUeds[k]
        dHs = (2.0*d_c - Hs_c*l_c
               - Hs_c*(1.0 - H_c)*lam_c)/(th_p*Ret)
        return dth, dHs, H_c

    def _twoeq_step(k, th_c, Hst_c, dx_tot):
        """Advance one station, subdividing until the change in H* is small.

        The kinetic-energy equation is stiff where the layer is thin, because
        theta Re_theta stands in its denominator, and an explicit step across a
        leading-edge station can throw H* clean outside the range the family
        spans - which, since H* falls with H, is read back as a separated
        profile and fires the bubble at the nose.
        """
        nsub, done = 1, False
        while not done and nsub <= 64:
            th_t, Hst_t, ok = th_c, Hst_c, True
            for _ in range(nsub):
                h = dx_tot/nsub
                a1, b1, _ = _rates(k-1, th_t, Hst_t)
                th_p = max(th_t + h*a1, 1e-12)
                Hst_p = float(np.clip(Hst_t + h*b1, _HSLO, _HSHI))
                a2, b2, _ = _rates(k, th_p, Hst_p)
                dHs = 0.5*h*(b1 + b2)
                if abs(dHs) > 0.01:
                    ok = False
                    break
                th_t = max(th_t + 0.5*h*(a1 + a2), 1e-12)
                Hst_t = float(np.clip(Hst_t + dHs, _HSLO, _HSHI))
            if ok:
                th_c, Hst_c, done = th_t, Hst_t, True
            else:
                nsub *= 2
        return th_c, Hst_c

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

        for i in range(i0+1, n):
            th_c, Hst_c = _twoeq_step(i, th_m[i-1], Hst, s[i] - s[i-1])
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
    # The critical amplification factor at every station.  It depends only on
    # the local effective turbulence intensity, so it is formed for the whole
    # surface here rather than only up to the station the march stops at: a
    # consumer plotting N against N_crit needs both defined everywhere.
    n_crit_all = np.array([_n_crit(t, cal.get("N_floor", 0.5)) for t in Tu_eff])

    # ---- frequency set for the e^N integration ----
    # A physical frequency is fixed along the layer while theta grows and U_e
    # changes, so the dimensionless frequency omega*theta/U_e sweeps across the
    # tabulated band.  The set is therefore chosen from the flow itself: it
    # spans every frequency that is inside the amplified band at some station
    # where amplification is possible at all.
    use_db = bool(cal.get("use_os_db", True))
    bubble_on = bool(cal.get("bubble", True))
    # i_bub is the LIVE bubble - None whenever the layer is attached - and
    # i_sep is the sticky record of the first station at which it ever
    # separated.  They used to be one variable, which was harmless only while
    # the bubble state was absorbing; now that a bubble can reattach, clearing
    # the live state would erase the fact that there was ever a bubble, and
    # x_sep_chord and the leading-edge-bubble test downstream both read it.
    i_bub = None; i_sep = None
    s_sep = 0.0; th_sep = 0.0; H_sep = 0.0; th_b = 0.0; n_bub = 0.0
    # State carried by the layer AFTER a bubble has reattached laminar.  While
    # it is None the march reads the precomputed attached solution, exactly as
    # before; once a bubble has formed and reattached, the attached solution
    # computed in ignorance of that bubble no longer describes this layer, so
    # the march carries its own state from the momentum thickness the dead-air
    # region left it with.  See the reattachment test in the bubble block.
    resumed = None          # (theta, Hstar) once a bubble has reattached
    n_reatt = 0
    n_cf = 0.0
    H_b = 0.0; Hst_b = 0.0
    _Hs_tab = _stab.twoeq_closure()[1]
    _HS_LO, _HS_HI = float(_Hs_tab.min()), float(_Hs_tab.max())
    _H_MAXTAB = float(_stab.twoeq_closure()[0].max())
    sig_cf = float(_stab.sigma_curve(_stab.H_REVERSE, 400.0).max())
    omegas = np.array([]); amp = np.array([])
    if use_db:
        om_lo, om_hi = _stab.omega_grid_bounds()
        rat = Ue/np.maximum(th_lam, 1e-12)
        live = Reth_lam > 60.0
        if live.any():
            w_lo = om_lo*float(np.min(rat[live]))
            w_hi = om_hi*float(np.max(rat[live]))
            w_lo = max(w_lo, 1e-9); w_hi = max(w_hi, w_lo*10.0)
            # the count follows from the span and the required spacing, so a
            # long plate gets more frequencies than a short one automatically
            _dw = max(float(cal.get("d_omega", 1.03)), 1.0 + 1e-6)
            _n = int(np.ceil(np.log(w_hi/w_lo)/np.log(_dw))) + 1
            _n = int(np.clip(_n, int(cal.get("n_freq_min", 40)),
                             int(cal.get("n_freq_max", 400))))
            omegas = np.geomspace(w_lo, w_hi, _n)
            amp = np.zeros(omegas.size)

    # ---- laminar-branch state at EVERY station ----
    # The marching loop below stops at onset, so the loop variables are only
    # defined up to it.  The intermittency blend downstream of onset needs the
    # laminar leg at stations the march never reached; taking it from the loop
    # arrays there, as an earlier version did, blended against the zeros they
    # were initialised with and returned a shape factor below unity and a
    # vanishing skin friction immediately behind transition.  The laminar
    # branch is therefore evaluated for the whole surface here, once.
    lam_all = th_lam*th_lam/nu_l*dUeds
    if H_lam is not None:
        H_all = np.asarray(H_lam, float)
        l_all = np.array([_stab.twoeq_HL(h)[1] for h in H_all])
    else:
        _hl = [_thwaites_HL(v) for v in lam_all]
        H_all = np.array([h for h, _ in _hl])
        l_all = np.array([l for _, l in _hl])
    Cf_all = 2.0*l_all*nu_l/np.maximum(Ue*th_lam, 1e-12)

    i_tr = None
    bub_trig = False
    for i in range(n):
        if resumed is None:
            theta[i] = th_lam[i]
            lam[i] = lam_all[i]
            H[i] = H_all[i]
            Cf[i] = Cf_all[i]
        else:
            # attached march resumed after a laminar reattachment
            th_r, Hst_r = _twoeq_step(i, resumed[0], resumed[1], s[i] - s[i-1])
            resumed = (th_r, Hst_r)
            H_r = float(np.clip(_stab.H_from_Hstar(Hst_r), H_all.min(), _H_MAXTAB))
            theta[i] = th_r
            H[i] = H_r
            lam[i] = th_r*th_r/nu_l[i]*dUeds[i]
            Cf[i] = 2.0*_stab.twoeq_HL(H_r)[1]*nu_l[i]/max(Ue[i]*th_r, 1e-12)
        Reth[i] = Ue[i]*theta[i]/nu_l[i]

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
        if bubble_on and (sep_now or i_bub is not None):
            if i_bub is None:
                i_bub = i
                if i_sep is None:
                    i_sep = i; s_sep = s[i]
                th_sep = max(theta[i], 1e-12); H_sep = H[i]
                th_b = th_sep; n_bub = 0.0
                H_b = H_sep; Hst_b = _stab.twoeq_HL(H_sep)[0]
            elif i > 0:
                # The dead-air region is marched with the same two equations as
                # the attached layer, with the wall shear set to zero.  Freezing
                # the shape factor at its separation value, as a first version
                # did, holds the profile fixed while the T3C4 hot films record
                # it thickening through the plateau; the momentum thickness is
                # then too small at reattachment and the onset Reynolds number
                # too low.  Letting the kinetic-energy equation run recovers
                # that growth: on T3C4 the shape factor rises from 3.34 at
                # separation to 3.44 at reattachment while Re_theta goes from
                # 255 to 274.
                #
                # The march is bounded by the ATTACHED Falkner-Skan branch,
                # H <= 3.997.  It cannot be continued onto the reverse-flow
                # branch, because H*(H) turns there - H* rises again with H past
                # the fold - so the inversion H = H(H*) the march depends on
                # ceases to exist.  Only the amplification rate below is read
                # from the reverse-flow branch, where no inversion is needed.
                dx_b = s[i] - s[i-1]
                Ret_b = max(Ue[i]*th_b/nu_l[i], 10.0)
                lam_b = th_b*th_b/nu_l[i]*dUeds[i]
                Hs_b, _l_b, d_b = _stab.twoeq_HL(H_b)
                th_b = max(th_b - (2.0 + H_b)*th_b/max(Ue[i], 1e-9)*dUeds[i]*dx_b,
                           1e-12)
                Hst_b = float(np.clip(
                    Hst_b + dx_b*(2.0*d_b - Hs_b*(1.0 - H_b)*lam_b)/(th_b*Ret_b),
                    _HS_LO, _HS_HI))
                H_b = float(np.clip(_stab.H_from_Hstar(Hst_b), H_sep, _H_MAXTAB))
                if cal.get("bub_rev_H", False):
                    # The momentum equation across the dead-air region needs H,
                    # not H*, so it is not bound by the fold that stops the
                    # kinetic-energy march: only the INVERSION H = H(H*) ceases
                    # to exist there.  Capping it at the attached separation
                    # value while reading the amplification rate off the
                    # reverse-flow profile at H_REVERSE is the model
                    # contradicting itself about which profile the detached
                    # layer has.  With this on, H relaxes across the bubble
                    # from its separation value towards that same reverse-flow
                    # profile, in step with the amplification that measures how
                    # far through the bubble the layer is.  No constant is
                    # added: both ends are quantities the model already uses.
                    _f = min(max(n_bub/max(_n_crit(Tu_eff[i],
                                                   cal.get("N_floor", 0.5)),
                                           1e-9), 0.0), 1.0)
                    H_b = H_sep + _f*(_stab.H_REVERSE - H_sep)
                # The amplification rate of the detached shear layer is read
                # from the same table as everywhere else.  The tabulated
                # Falkner-Skan family is continued past separation onto its
                # reverse-flow branch, so a separated profile has a computed
                # rate and the bubble closure carries no fitted constant.  It
                # is evaluated on the developed reverse-flow profile rather
                # than on the profile at the separation point: the momentum
                # thickness is nearly frozen across the dead-air region but the
                # shape factor is not, and the amplification is dominated by
                # the developed layer.  The computed rate there is 0.042 to
                # 0.045 over the Re_theta range these bubbles span, 0.0435 at
                # Re_theta = 400, which is what the T3C4 and NLF(1)-0416 bubble
                # lengths independently imply.
                sig_b = float(_stab.sigma_curve(_stab.H_REVERSE, Reth[i]).max())
                n_bub += max(sig_b, 0.0)/th_b*(s[i] - s[i-1])
            theta[i] = th_b; H[i] = H_b; Cf[i] = 0.0
            Reth[i] = Ue[i]*th_b/nu_l[i]
            n_fac[i] = n_bub
            bub_trig = n_bub >= _n_crit(Tu_eff[i], cal.get("N_floor", 0.5))

            # ---- laminar reattachment ----
            # A bubble does not have to end in transition.  If the pressure
            # gradient recovers before the detached shear layer has amplified
            # to N_crit, the layer reattaches laminar and carries on.  The
            # bubble state used to be ABSORBING: once i_sep was set nothing
            # ever cleared it, so a layer that separated could only transition
            # or be declared burst at the trailing edge, and a short bubble in
            # a gradient that relaxes had no representation at all.
            #
            # The test is on the dead-air layer's own Thwaites parameter, not
            # on the attached solution's - the attached solution was computed
            # in ignorance of the bubble and is not this layer.  Reattachment
            # needs the gradient to have stopped being adverse enough to hold
            # the layer off, a bubble at least two stations long so that a
            # single noisy station cannot open and close one, and the
            # amplification still short of the threshold that would have ended
            # it in transition instead.
            #
            # Measured on the 86 NLF(1)-0416 conditions, 52 of which form a
            # bubble, exactly one reaches this branch and it is a leading-edge
            # bubble at x/c = 0.002 that the method already declares.  So this
            # closes a real gap in the formulation without moving any result
            # here, which is how it is reported.
            # The gradient must be FAVOURABLE, not merely less adverse, and it
            # must have been so for two consecutive stations: the edge velocity
            # comes from a panel method and carries station-to-station scatter,
            # so a single positive dU_e/dx inside an overall adverse run is
            # noise and not a recovery.  Testing lambda against lam_sep is not
            # enough on its own - across the dead-air region theta is nearly
            # frozen while the attached solution keeps thickening, so lambda_b
            # is small and clears that threshold in any mildly adverse
            # gradient, which would reattach almost every bubble immediately.
            if (not bub_trig) and i_bub is not None and i - i_bub >= 2:
                fav = dUeds[i] > 0.0 and dUeds[i-1] > 0.0
                if fav and th_b*th_b/nu_l[i]*dUeds[i] > 0.0:
                    # The disturbance does not vanish because the layer touched
                    # down again.  Whatever amplified across the dead-air
                    # region is carried into the attached integral, which then
                    # continues from it - discarding it, as a first version of
                    # this did, restarts a wave that is already most of the way
                    # to breakdown and puts transition far too late.
                    if amp.size:
                        np.maximum(amp, n_bub, out=amp)
                        n_amp = float(amp.max())
                    else:
                        n_amp = max(n_amp, n_bub)
                    resumed = (th_b, Hst_b)
                    i_bub = None
                    n_bub = 0.0
                    n_reatt += 1

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
        if use_db and not (bubble_on and i_bub is not None):
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
        if not (bubble_on and i_bub is not None):
            # inside the dead-air region n_fac already holds the bubble's own
            # amplification factor, which is what closes it; the attached-layer
            # integral is frozen there and must not overwrite it
            n_fac[i] = n_amp
        N_crit_i = _n_crit(Tu_eff[i], cal.get("N_floor", 0.5))
        N_target = N_crit_i / max(cal["A_TS"], 1e-6)

        # (b) bypass: free-stream-turbulence driven.  Abu-Ghannam & Shaw
        #     correlated an attached layer in an ELEVATED-Tu stream; below
        #     about a tenth of a per cent the correlation is outside the data
        #     it was fitted to and the amplification route governs instead.
        #     _branch_weight decides how much of each applies; the threshold is
        #     always formed, and the weight, not a gate, is what turns the
        #     branch off.
        Tu_i = Tu_eff[i]
        Rbp = cal["A_BP"] * _ags_re_theta_t(max(Tu_i, 0.02), lam_t)
        w_bp = _branch_weight(Tu_i, cal)

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
        if bubble_on and i_bub is not None:
            Rsep = 1e9
        elif sep_now:
            Rref = _ags_re_theta_t(max(Tu_i, 0.02), lam_t)
            Rsep = cal["A_SEP"]*max(cal["sep_floor"], 0.7*Rref)
        else:
            Rsep = 1e9

        # (d) cross-flow (swept wing): C1 criterion on the cross-flow
        #     momentum-thickness Reynolds number (algebraic surrogate).
        #
        #     The exact quantity is available: the Falkner-Skan-Cooke solution
        #     gives Re_cf = Re_theta sin(L) cos(L) K(lambda), with the sweep
        #     factoring out exactly and K carrying the pressure-gradient
        #     dependence (stability.crossflow_factor).  K is not constant - it
        #     spans 0.023 to 4.60 across the family, a factor of two hundred,
        #     and vanishes at zero pressure gradient, where the span-wise and
        #     chordwise similarity equations give f' = g identically and there
        #     is no cross-flow at all, whereas the surrogate's constant
        #     predicts some.  Replacing
        #     the surrogate by K was tried and is not adopted: it does not make
        #     the two swept-wing experiments agree on a critical value, and it
        #     degrades the calibration set from 13.2 to 23.8 per cent.  The
        #     limiting approximation is therefore the C1 criterion itself,
        #     which reduces the stability of an inflectional three-dimensional
        #     profile to a single Reynolds number, and not the surrogate for
        #     the cross-flow thickness.  See Sec. VI.
        if sweep_deg > 1.0 and cal.get("cf_amp", True):
            L_r = np.radians(sweep_deg)
            if cal.get("cf_exact", False):
                _sf = (np.sin(L_r) if normal_frame
                       else np.sin(L_r)*np.cos(L_r))
                fac = _sf*_stab.crossflow_factor(lam[i])
                Re_th2 = Reth[i]*fac
            else:
                Re_th2 = _re_theta2(Reth[i], sweep_deg, cal["CF_ratio"],
                                    normal_frame)
            if Re_th2 >= cal["CF_C1"]*cal["A_CF"] and i > 0:
                n_cf += (sig_cf/max(theta[i], 1e-12))*(s[i] - s[i-1])
            Rcf = 1e9
        elif sweep_deg > 1.0:
            if cal.get("cf_exact", False):
                L_r = np.radians(sweep_deg)
                _sf = (np.sin(L_r) if normal_frame
                       else np.sin(L_r)*np.cos(L_r))
                fac = _sf*_stab.crossflow_factor(lam[i])
                Rcf = (cal["CF_C1"]/max(fac, 1e-12)/cal["A_CF"]
                       if fac > 1e-12 else 1e9)
            else:
                Re_th2 = _re_theta2(Reth[i], sweep_deg, cal["CF_ratio"],
                                    normal_frame)
                Rcf = (Reth[i]*cal["CF_C1"]/max(Re_th2, 1e-9)/cal["A_CF"]
                       if Re_th2 > 0 else 1e9)
        else:
            Rcf = 1e9

        # Inside the dead-air region none of the attached-flow criteria apply:
        # Abu-Ghannam & Shaw correlates an attached layer against its own
        # pressure gradient, and the cross-flow criterion presumes a wall
        # boundary layer.  Once the layer has separated the bubble closure is
        # the only thing that can end it.
        # ---------- selection: one progress variable per mechanism ----------
        # Every branch reports the same thing - how far through its own onset
        # criterion the layer has got, as a number that reaches 1 at onset -
        # and the kernel fires on the first station at which any of them does.
        #
        #   p_TS  = N / N_crit                    amplification, e^N
        #   p_BP  = Re_theta / Re_theta_t(AGS)    correlation
        #   p_SEP = N_bub / N_crit  (bubble on)   amplification across dead air
        #           Re_theta / Re_theta_t (off)   correlation, ablation path
        #   p_CF  = N_cf / N_cf,crit (cf_amp on)  amplification
        #           Re_theta / Re_theta_t (off)   C1 threshold
        #
        # This is the form Eq. (E13) states.  What it replaces was written as a
        # minimum over four onset REYNOLDS NUMBERS, which only two of the four
        # branches actually produce: in the shipped configuration the TS, the
        # separation and the cross-flow branches all close on amplification
        # integrals and were carried past the minimum as a 1e9 "cannot fire"
        # sentinel, so the minimum ranged over one live term and the equation
        # in the report described a calculation the solver was not doing.
        # Expressed as progress the four are commensurable, the weights a_TS,
        # a_BP, a_SEP, a_CF all act the same way on all four, and the kernel is
        # one line.
        in_bubble = bubble_on and i_bub is not None

        p_ts = (n_amp/N_target) if N_target > 0.0 else 0.0
        p_bp = 0.0 if in_bubble else Reth[i]/max(Rbp, 1e-9)
        if bubble_on:
            p_sep = n_bub/max(N_crit_i/max(cal["A_SEP"], 1e-6), 1e-9)
        else:
            p_sep = 0.0 if in_bubble else Reth[i]/max(Rsep, 1e-9)
        if sweep_deg > 1.0 and cal.get("cf_amp", True):
            _Ncf = float(cal.get("CF_N", 0.0)) or N_crit_i
            p_cf = n_cf/max(_Ncf/max(cal["A_CF"], 1e-6), 1e-9)
        else:
            p_cf = 0.0 if in_bubble else Reth[i]/max(Rcf, 1e-9)

        # The natural and bypass routes are the same transition seen through
        # two closures with different ranges of validity, so they are combined
        # rather than switched between.  A hard gate at Tu = 0.1 % made the
        # answer DISCONTINUOUS in the free-stream turbulence: on the cruise
        # section, 0.1000 % gave x_tr/c = 0.542 and 0.1001 % gave 0.373, a
        # step of 0.17c and 33 % in profile drag across a change of one part in
        # a thousand in an input the study quotes to two figures.  A geometric
        # blend of the two progresses over the declared window removes the step
        # without adding a fitted constant to either branch: at the lower edge
        # it is exactly the amplification integral, at the upper edge exactly
        # the correlation, and the crossing of unity moves monotonically
        # between the two onsets in between.  The window is set wider than any
        # case in this study, so no result here is blended - it is there so
        # that the model is a function rather than a switch.
        #
        # The blend is arithmetic in the progress, not geometric.  A geometric
        # weighting was tried first, to match the one the bypass branch uses
        # for the flow history of Tu, and it is the wrong tool here: upstream
        # of the neutral point the amplification integral is legitimately zero,
        # and a geometric mean with a zero factor is zero at ANY weight, so a
        # stream at Tu = 0.22 % - almost entirely bypass - was held laminar by
        # a three per cent share of a branch that had not started amplifying.
        # It reintroduced a 0.075c step at the upper edge of the window.  With
        # the arithmetic form each closure contributes in proportion to its
        # weight, and the onset moves monotonically from the bypass station to
        # the amplification station as the weight falls.
        if in_bubble:
            p_nat = 0.0
        elif w_bp <= 0.0:
            p_nat = p_ts
        elif w_bp >= 1.0:
            p_nat = p_bp
        else:
            p_nat = (1.0 - w_bp)*p_ts + w_bp*p_bp

        # The onset Reynolds number is reported only where a branch that
        # actually produces one is the governing branch; 1e9 was the internal
        # "cannot fire here" sentinel and used to be written straight into the
        # output, so every natural-transition case reported an onset Reynolds
        # number of a billion.  Absent is absent: it leaves as NaN.
        Rt = 1e9 if in_bubble else min(Rbp if w_bp > 0.0 else 1e9, Rsep, Rcf)
        Re_th_t[i] = np.nan if Rt >= 1e8 else Rt

        p_all = (p_nat, p_sep, p_cf)
        if max(p_all) >= 1.0 and i_tr is None and i > 1:
            i_tr = i
            if in_bubble or (bubble_on and p_sep >= 1.0):
                mech = "separation"
            else:
                k = int(np.argmax(p_all))
                if k == 0:
                    # inside the blend the label follows the closure that is
                    # carrying more of the weight, so it never claims a branch
                    # the answer did not come from
                    mech = "bypass" if (w_bp >= 0.5 or p_ts < 1.0) else "TS-natural"
                    if w_bp <= 0.0:
                        mech = "TS-natural"
                    elif w_bp >= 1.0:
                        mech = "bypass"
                else:
                    mech = ("separation", "crossflow")[k - 1]
            mechanism[i] = mech
            break

    if i_tr is None:
        # fully laminar to TE
        out = dict(s=s, Ue=Ue, theta=theta, H=H, Cf=Cf, Re_theta=Reth,
                   lam=lam, gamma=gamma, state=state, Re_theta_t=Re_th_t,
                   n_factor=n_fac, n_crit=n_crit_all,
                   mechanism=mechanism, i_tr=None,
                   i_sep=i_sep, s_sep=(s_sep if i_sep is not None else np.nan),
                   bubble_burst=bool(i_bub is not None),
                   n_reattach=n_reatt,
                   x_tr=np.nan, onset_mech="none(laminar)",
                   # the same keys the transitioning branch returns.  A surface
                   # that stays laminar used to omit lam_len and sep_turb
                   # entirely, so any consumer that read them had to know which
                   # branch it was on or take a KeyError.
                   lam_len=np.nan, sep_turb=None,
                   H_lam=H.copy(), theta_lam=theta.copy(),
                   H_turb=np.zeros(n), theta_turb=np.zeros(n))
        return out

    s_tr = s[i_tr]; onset_mech = mechanism[i_tr]
    Re_tr = Reth[i_tr]
    # Dhawan & Narasimha transition length:  Re_lambda = C_len * Re_x_t^0.75.
    # Re_x_t is formed on the distance the layer has run - arc length from the
    # stagnation point, which on a flat plate is x - and on the local edge
    # velocity, the same convention Re_x carries everywhere else in this work.
    nu_local = nu_t[i_tr]
    Re_x_tr = max(Ue[i_tr]*s_tr/nu_local, 1.0)
    Re_lam_len = cal["C_len"] * Re_x_tr**0.75
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
    # The laminar leg of the blend is the marched state where the march
    # reached (so a station inside a bubble keeps its dead-air values) and the
    # laminar branch beyond it.
    H_leg = H.copy(); Cf_leg = Cf.copy(); th_leg = theta.copy()
    if i_tr + 1 < n:
        H_leg[i_tr+1:] = H_all[i_tr+1:]
        Cf_leg[i_tr+1:] = Cf_all[i_tr+1:]
        th_leg[i_tr+1:] = th_lam[i_tr+1:]
        lam[i_tr+1:] = lam_all[i_tr+1:]
    for i in range(n):
        if i < i_tr:
            gamma[i] = 0.0
            state[i] = "laminar"
        else:
            xi = (s[i]-s_tr)/max(lam_len, 1e-9)
            g = 1.0 - np.exp(-0.412*xi**2)
            gamma[i] = float(np.clip(g, 0.0, 1.0))
            # blended properties
            theta[i] = (1-gamma[i])*th_leg[i] + gamma[i]*th_turb[i]
            H[i]  = (1-gamma[i])*H_leg[i] + gamma[i]*H_turb[i]
            Cf[i] = (1-gamma[i])*Cf_leg[i] + gamma[i]*Cf_turb[i]
            Reth[i] = Ue[i]*theta[i]/nu_t[i]
            state[i] = "transitional" if gamma[i] < 0.99 else "turbulent"
            mechanism[i] = onset_mech

    out = dict(s=s, Ue=Ue, theta=theta, H=H, Cf=Cf, Re_theta=Reth,
               lam=lam, gamma=gamma, state=state, Re_theta_t=Re_th_t,
               n_factor=n_fac, n_crit=n_crit_all,
               mechanism=mechanism, i_tr=i_tr, x_tr=s_tr,
               lam_len=lam_len, onset_mech=onset_mech,
               i_sep=i_sep, s_sep=(s_sep if i_sep is not None else np.nan),
               bubble_burst=False,
               sep_turb=sep_turb,
               n_reattach=n_reatt,
               # The two legs the intermittency blends, kept separately.  Every
               # blended quantity above is (1-g) leg_lam + g leg_turb, and a
               # wall-normal profile reconstruction has to blend the same way -
               # from the blended shape factor alone it cannot tell a
               # half-transitional station from a laminar one at the same H.
               H_lam=H_leg, theta_lam=th_leg,
               H_turb=H_turb, theta_turb=th_turb)
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
    r = march_bl(s, Ue, nu, Tu_pct=Tu_pct, cal=cal)
    # Re_x is formed on the LOCAL edge velocity, which is the convention of the
    # ERCOFTAC tabulations these cases are compared against: on T3C4 the edge
    # velocity rises from 1.51 to 2.02 m/s, so forming Re_x on the reference
    # speed understates it by up to a fifth at the end of the plate.
    r["Re_x"] = Ue*s/nu
    r["Cf_lam_ref"] = 0.664/np.sqrt(np.maximum(r["Re_x"], 1.0))
    r["Cf_turb_ref"] = 0.0592/np.maximum(r["Re_x"], 1.0)**0.2
    return r


def solve_airfoil(xb, yb, alpha_deg, U, nu, chord, Tu_pct,
                  sweep_deg=0.0, cal=None, mach=0.0, compressible=True,
                  sweep_transform=True):
    """
    Full aerofoil: panel method -> split at stagnation -> march upper
    and lower surfaces.  Returns inviscid + viscous results.

    SWEEP.  On a swept surface the boundary layer is driven by the component of
    the flow NORMAL to the leading edge, not by the streamwise flow, and simple
    sweep theory says so exactly for an infinite swept wing: the chordwise
    problem is the two-dimensional one at

        U_n = Q cos(L),   c_n = c cos(L),   alpha_n = atan(tan(alpha)/cos(L)),

    with the span-wise component Q sin(L) riding on top of it and entering only
    through the cross-flow.  The section lift in the streamwise frame is then
    c_l = c_l,n cos^2(L).

    This was not done.  The panel solve and the march were run on the
    STREAMWISE flow and the streamwise chord, and sweep_deg reached only the
    cross-flow branch.  At the 12 deg of the case-study wing that is a 2 per
    cent error in the edge velocity, but the cross-flow constant CF_ratio was
    set on a 45 deg experiment, where cos(L) = 0.707: the chordwise edge
    velocity, the momentum thickness and the pressure gradient driving them
    were all wrong by tens of per cent, and the constant absorbed it.  Worse,
    the independent check spans 20 to 50 deg of sweep, over which cos(L) runs
    from 0.94 to 0.64, so the error is not even constant across the set the
    criterion is asked to transfer to.

    Set sweep_transform=False to recover the untransformed behaviour, so that
    what the transformation is worth can be measured rather than asserted.
    """
    L_rad = np.radians(sweep_deg)
    cosL = float(np.cos(L_rad))
    swept = bool(sweep_transform and sweep_deg > 1.0 and cosL > 1e-6)
    if swept:
        # into the plane normal to the leading edge
        alpha_solve = float(np.degrees(np.arctan(
            np.tan(np.radians(alpha_deg))/cosL)))
        U_solve = U*cosL
        chord_solve = chord*cosL
        mach_solve = mach*cosL
    else:
        alpha_solve, U_solve, chord_solve, mach_solve = (
            alpha_deg, U, chord, mach)
    alpha_deg, U, chord, mach = alpha_solve, U_solve, chord_solve, mach_solve
    xc, yc, Cp, V, th, S = panel_solve(xb, yb, alpha_deg, mach=mach)
    # Edge velocity and edge Mach number from the CORRECTED pressure, so the
    # boundary layer runs on the same flow the loads are computed from; see
    # _edge_from_cp for what taking the uncorrected panel velocity cost.
    m_eff = (mach if compressible else 0.0)
    Ue_ratio, Me_all = _edge_from_cp(Cp, m_eff)
    # arc length along surface (control points), find stagnation (Cp max)
    # NB: panel coords are unit-chord -> scale arc length to physical chord
    ds = S
    s_arc = np.concatenate([[0.5*ds[0]], 0.5*ds[0] +
                            np.cumsum(0.5*(ds[:-1]+ds[1:]))]) * chord
    i_stag = int(np.argmax(Cp))
    Ue_mag = Ue_ratio*U

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
                     cal=cal, Me=Me_all[idx], normal_frame=swept)
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
        # Turbulent separation, and the margin to it.  march_bl has always
        # computed sep_turb - the first station at which the turbulent shape
        # factor passes 2.6 - and nothing has ever read it, while the README
        # and the report abstract both list "trailing-edge separation margin"
        # among the quantities this study delivers.  It is an output now.
        _st = r.get("sep_turb")
        r["x_sep_turb_chord"] = (float(xc[idx][_st]) if _st is not None else np.nan)
        r["H_te"] = float(r["H"][-1])
        r["sep_margin_H"] = float(2.6 - r["H"][-1])
        r["bubble_burst"] = bool(r.get("bubble_burst", False))
        res[name] = r

    # Forces.  The boundary points run CLOCKWISE (TE -> lower -> LE -> upper ->
    # TE), which the signed area confirms: -0.110 for the case-study section.
    # This comment used to say counter-clockwise.  The normal below is outward
    # either way - on the upper surface, traversed towards +x, theta ~ 0 gives
    # (-sin, cos) = (0, 1) - so nothing was wrong with the arithmetic, but
    # run_solution's body mask says clockwise five lines from where it uses the
    # winding to decide a sign, and the two files contradicting each other is
    # how the next sign error gets made.
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
        r["theta_te_c"] = float(th_te/chord)
        return 2.0*th_te/chord*(Ue_te/U)**((H_te+5.0)/2.0)
    Cd = squire_young(res["upper"]) + squire_young(res["lower"])
    # Squire-Young presumes a thin trailing-edge layer, and outside the
    # attached-flow envelope this formulation stops providing one: at 16 deg
    # and Re_c = 2e5 the march returns a momentum thickness of 1.34 chords and
    # the formula duly returns C_d = 1.22, which is bluff-body drag from an
    # aerofoil method.  A layer thicker than the body is long is not a marginal
    # case to be reported with a caveat, it is arithmetic that has stopped
    # meaning anything, so no number is offered.  This is not a tuned
    # threshold: within the envelope and Reynolds range of this study the
    # largest value reached is 0.028 chords, so the test never fires on any
    # result reported here.
    if max(res["upper"]["theta_te_c"], res["lower"]["theta_te_c"]) > 1.0:
        Cd = float("nan")

    # Back to the streamwise frame.  In the normal plane the section carries
    # c_l,n on a dynamic pressure formed with U_n = Q cos(L) and a chord
    # c cos(L); referred to the streamwise dynamic pressure and the streamwise
    # chord that is c_l = c_l,n cos^2(L).  The profile drag scales the same
    # way, being 2 theta/c times a velocity ratio that is frame-independent.
    # Transition LOCATIONS need no conversion: x/c is the same fraction of the
    # same chord line in either frame.
    if swept:
        Cl = Cl*cosL*cosL
        Cd = Cd*cosL*cosL
    return dict(panel=dict(xc=xc, yc=yc, Cp=Cp, V=V, th=th, S=S,
                           i_stag=i_stag),
                surfaces=res, Cl=Cl, Cd=Cd, alpha=alpha_deg,
                sweep_deg=sweep_deg, sweep_transform=swept,
                alpha_normal_deg=alpha_deg, cos_sweep=cosL,
                theta_te_c=max(res["upper"]["theta_te_c"],
                               res["lower"]["theta_te_c"]))


if __name__ == "__main__":
    # ---- self-tests -------------------------------------------------------
    # 1. the off-body field must reproduce the surface solution it comes from
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import case_config as _C
    Xb, Yb = _C.nlf16_panel_points(130)
    _xc, _yc, _Cp, _V, _th, _S = panel_solve(Xb, Yb, 1.5)
    _d = 0.003
    _Px = _xc - np.sin(_th)*_d; _Py = _yc + np.cos(_th)*_d
    _Cf = velocity_field(Xb, Yb, 1.5, _Px, _Py, U=1.0)[2]
    _e = float(np.mean(np.abs(_Cf[10:-10] - _Cp[10:-10])))
    print("off-body field vs surface C_p at %.3fc offset: mean |dCp| = %.4f  %s"
          % (_d, _e, "OK" if _e < 0.05 else "FAIL"))

    # 2. flat-plate transition
    r = solve_flat_plate(1.7, 5.4, 1.5e-5, 3.043, npts=900, L_turb=1.53e-3)
    print("ERCOFTAC T3A: x_tr = %.4f m  Re_x_tr = %.3e  Re_theta_t = %.1f  "
          "mech = %s" % (r["x_tr"], 5.4*r["x_tr"]/1.5e-5,
                         r["Re_theta"][r["i_tr"]], r["onset_mech"]))

    # 3. the intermittency blend must not leave the laminar leg at zero
    _i = r["i_tr"]
    print("H through transition: %s  %s"
          % (np.array2string(r["H"][_i:_i+4], precision=3),
             "OK" if r["H"][_i:_i+4].min() > 1.0 else "FAIL"))
