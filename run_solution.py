"""
run_solution.py
Execute the UTSS solver for the AETHER-NLF 25 case study and write every
engineering output as CSV (04_solution/).
"""
import os, sys
import numpy as np, pandas as pd
import utss_paths  # noqa: F401  - anchors the repo root and solver/ on
                   # sys.path, so this script works from any directory
import case_config as C
from utss_solver import solve_airfoil, velocity_field, panel_solve, CAL

SOL="04_solution"; os.makedirs(SOL,exist_ok=True)
W=C.WING; cr=C.CRUISE; cl=C.CLIMB

def run_case(cond, name):
    X,Y=C.nlf16_panel_points(130)
    r=solve_airfoil(X,Y,cond["alpha_deg"],cond["U_inf"],cond["nu_inf"],
                    W["MAC"],cond["Tu_pct"],sweep_deg=W["le_sweep_deg"],
                    mach=cond["mach"],T_inf_K=cond["T_inf_K"])
    for surf in ["upper","lower"]:
        s=r["surfaces"][surf]
        # THE STAGNATION STATION HAS NO BOUNDARY LAYER, and the columns that
        # describe one are absent there rather than filled in.
        #
        # solve_airfoil clamps the edge velocity to 1e-4 m/s at the stagnation
        # point so the march does not divide by zero, and every integral
        # quantity formed on that clamp IS the clamp: theta comes back at its
        # own 1e-8 m floor and C_f = 2 l nu/(U_e theta) at 1.7e7.  That number
        # was published in this CSV and carried out of it into the report's
        # sampled state table as 16804859.1563012 - sixteen significant figures
        # of a skin-friction coefficient - and the document's "no unrounded
        # float64" check missed it because that check counts DECIMAL places and
        # .round(7) has nothing to round on a quantity of order 1e7.
        # verify_outputs now also counts significant figures.
        #
        # A momentum-thickness Reynolds number below one is not a boundary
        # layer.  Those stations leave as NaN, exactly as Re_theta_t and the
        # amplification factor already do where they are undefined.  C_p, the
        # edge velocity and the arc length are properties of the inviscid
        # solution and stay: C_p = 1.046 at the stagnation point is the answer.
        _bl=np.where(np.asarray(s["Re_theta"],float) >= 1.0, 1.0, np.nan)
        # Rounded to the precision these quantities are meaningful to, as
        # every other CSV in this project is.  Unrounded, a Reynolds number
        # went into the report's sampled state tables as 1891858.3232883876.
        df=pd.DataFrame({
            "x_c":s["x"].round(6), "arc_s_m":s["s"].round(6),
            "Re_x":s["Re_x"].round(0),
            "Cp":s["Cp"].round(5), "Ue_ms":s["Ue"].round(4),
            "Ue_Uinf":(s["Ue"]/cond["U_inf"]).round(5),
            "theta_mm":(s["theta"]*1e3*_bl).round(5),
            "H_shape":(s["H"]*_bl).round(4),
            "Cf":(s["Cf"]*_bl).round(7),
            "Re_theta":(s["Re_theta"]*_bl).round(2),
            "Re_theta_trans":np.round(s["Re_theta_t"],1),
            # The amplification factor is not defined downstream of onset:
            # the laminar march stops there, so the array holds the zeros it
            # was initialised with.  Writing those published a hard 0 across
            # half the chord in a tabulated column, and drew the N curve on the
            # criterion figure collapsing to nothing just past the station it
            # had just fired at.  Absent is absent, as it is for Re_theta_t.
            "n_factor":_undefined_past_onset(s["n_factor"], s["i_tr"]),
            "n_crit":np.round(s["n_crit"],3),
            "intermittency_gamma":s["gamma"].round(5),
            "state":s["state"]})
        df.to_csv(f"{SOL}/surface_{name}_{surf}.csv",index=False)
    return r

def _undefined_past_onset(v, i_tr, nd=4):
    """v rounded, with everything strictly downstream of onset set to NaN."""
    out=np.round(np.asarray(v,float),nd)
    if i_tr is not None and i_tr+1 < out.size:
        out[i_tr+1:]=np.nan
    return out


def transition_summary(rc, rl, write=True):
    """The per-surface transition summary.

    write=False returns the frame without touching the repository, so a check
    can exercise the frame conversion below on its own solves - the same guard
    bl_profiles and run_nlf0416 already carry, and for the same reason: a check
    that overwrites a tracked output is not a check.
    """
    rows=[]
    for nm,rr,cond in [("CRUISE",rc,cr),("CLIMB",rl,cl)]:
        for surf in ["upper","lower"]:
            s=rr["surfaces"][surf]
            # THE ARC LENGTH AND Re_x BELONG TO THE PLANE THE MARCH RUNS IN.
            # solve_airfoil solves a swept section in the plane normal to the
            # leading edge: on a chord c_n = c cos(L), at a speed U_n = U cos(L),
            # and s["x_tr"] is an arc length measured on THAT section.  Dividing
            # it by the STREAMWISE chord, and forming Re_x on the STREAMWISE
            # speed, mixes the two frames and is wrong by cos(L) - 2.2 % at the
            # 12 deg of this wing, and 41 % at the 45 deg of the calibration
            # wing, had this routine ever been pointed at one.
            #
            # It showed as an impossibility: s_tr_c came out at 0.554 on the
            # cruise upper surface while the arc from the LEADING EDGE alone to
            # x/c = 0.542 is 0.5623 c, and the march starts at the stagnation
            # point, which at positive incidence is on the LOWER surface and so
            # further upstream still.  An arc length cannot be shorter than a
            # part of itself.  Referred to the chord it is measured on it is
            # 0.5665, which is longer, as it must be.
            #
            # Re_x had the same fault in the other direction, and it contradicted
            # the file beside it: solve_airfoil writes Re_x = U_n s_n / nu into
            # every surface CSV, so the same station was published here as
            # 3.554e6 and there as 3.476e6 - one Reynolds number, two values,
            # differing by exactly 1/cos(L).
            _c = rr["cos_sweep"] if rr.get("sweep_transform") else 1.0
            chord_n = W["MAC"]*_c            # the chord the march ran on
            U_n = cond["U_inf"]*_c           # the speed it ran at
            xtr=s["x_tr"]/chord_n if not np.isnan(s["x_tr"]) else 1.0
            rex=U_n*s["x_tr"]/cond["nu_inf"] if not np.isnan(s["x_tr"]) else np.nan
            it=s["i_tr"]
            has_tr = it is not None
            reth=float(s["Re_theta"][it]) if has_tr else np.nan
            xch=s.get("x_tr_chord",np.nan)
            # a surface that stays laminar to the trailing edge is reported as
            # x_tr/c = 1.0, the same convention the polar and span-wise sweeps
            # use, rather than as a blank
            xch = float(xch) if xch==xch else 1.0
            rows.append(dict(case=nm,surface=surf,s_tr_c=round(xtr,3),
                x_tr_c=round(xch,3),
                Re_x_tr=f"{rex:.3e}", Re_theta_at_onset=round(reth,1) if has_tr else None,
                mechanism=s["onset_mech"],
                laminar_run_pct=round(xch*100,1),
                Cf_te=round(float(s["Cf"][-1]),5),
                # Trailing-edge separation margin.  Both the README and the
                # report's executive summary list this among the quantities
                # the study delivers; until now nothing computed it into an
                # output, though march_bl has always found the station.
                H_te=round(float(s["H_te"]),3),
                H_te_at_clip=bool(s["H_te_at_clip"]),
                sep_margin_H=round(float(s["sep_margin_H"]),3),
                # Whether the SQUIRE-YOUNG station's shape factor is solved or
                # is Head's H = 2.8 clamp.  H_te_at_clip above describes the
                # last station; the drag is formed at 0.98c, and on the climb
                # case that station is on the clamp too.
                x_sy_c=round(float(s["x_squire_young"]),4),
                H_sy=round(float(s["H_te_squire_young"]),3),
                H_sy_at_clip=bool(s["H_sy_at_clip"]),
                x_sep_turb_c=(round(float(s["x_sep_turb_chord"]),3)
                              if s["x_sep_turb_chord"]==s["x_sep_turb_chord"]
                              else None),
                # ...and whether the evaluation station is still in attached
                # flow.  The clamp flag above says the shape factor there is a
                # bound; this says which side of the march's OWN separation
                # prediction the station sits on, which the two columns beside
                # it have always allowed a reader to work out and no column
                # stated.  On CLIMB upper it is negative.
                sy_margin_to_sep_c=(round(float(s["sy_margin_to_sep_c"]),4)
                                    if s["sy_margin_to_sep_c"]==s["sy_margin_to_sep_c"]
                                    else None),
                sy_past_sep=bool(s["sy_past_sep"])))
    df=pd.DataFrame(rows)
    if write:
        df.to_csv(f"{SOL}/transition_summary.csv",index=False)
    return df

def aero_polar():
    X,Y=C.nlf16_panel_points(130); rows=[]
    for a in np.arange(-3,8.01,1.0):
        r=solve_airfoil(X,Y,a,cr["U_inf"],cr["nu_inf"],W["MAC"],cr["Tu_pct"],
                        sweep_deg=W["le_sweep_deg"],mach=cr["mach"],
                        T_inf_K=cr["T_inf_K"])
        u=r["surfaces"]["upper"]; l=r["surfaces"]["lower"]
        rows.append(dict(alpha_deg=a, Cl=round(r["Cl"],4), Cd=round(r["Cd"],5),
            L_over_D=round(r["Cl"]/max(r["Cd"],1e-9),1),
            # the thin-layer assumption Squire-Young rests on, made visible
            theta_te_c=round(r["theta_te_c"],5),
            # half this sweep evaluates Squire-Young on the H = 2.8 clamp
            H_sy_at_clip=bool(u["H_sy_at_clip"] or l["H_sy_at_clip"]),
            # and on part of it the station is past the march's own separation
            # point, which is a stronger statement than the clamp and was not
            # published anywhere the reader could see it vary with incidence
            sy_past_sep=bool(u["sy_past_sep"] or l["sy_past_sep"]),
            sy_margin_to_sep_c=round(float(min(
                [v for v in (u["sy_margin_to_sep_c"], l["sy_margin_to_sep_c"])
                 if v == v] or [float("nan")])), 4),
            xtr_upper_c=round(u["x_tr_chord"],3) if u["x_tr_chord"]==u["x_tr_chord"] else 1.0,
            xtr_lower_c=round(l["x_tr_chord"],3) if l["x_tr_chord"]==l["x_tr_chord"] else 1.0))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/aero_polar.csv",index=False)
    return df

def laminar_bucket_edge():
    """Resolve the incidence at which the transition point jumps forward.

    The polar is tabulated at one degree and straddles a discontinuity: between
    2 and 3 degrees the upper-surface transition collapses from 0.518 chord to
    0.128 and the section drag rises 62 per cent, and the table said so without
    saying why.  Refining the incidence does not smooth it - at 0.02 degrees the
    jump is still a single step - because it is not a resolution failure.  It is
    a bifurcation, and this sweep is the evidence.

    The e^N integral has two competing amplification maxima on this section: one
    under the leading-edge suction peak and one in the mid-chord pressure
    recovery.  Onset is wherever N/N_crit first reaches one, so the answer is
    decided by WHICH maximum crosses first, and a maximum that is a fraction of
    a per cent short leaves transition to the one behind it.  Across the edge
    the forward peak goes from 0.9909 to 1.0015 - a crossing by fifteen parts in
    ten thousand - and the transition point moves nineteen per cent of the
    chord.  x_tr is therefore genuinely discontinuous in incidence while the N
    field underneath it is smooth, so no refinement removes the jump; it only
    locates it.  That is the edge of the laminar bucket, and it is a property of
    the aerofoil, not of the discretisation.

    N_over_Ncrit_forward is the peak of N/N_crit over the leading-edge band,
    x/c < 0.20, which is the quantity that decides the branch; where the layer
    has already tripped inside that band the peak IS the onset value.
    """
    X, Y = C.nlf16_panel_points(130)
    rows = []
    for a in np.arange(2.80, 3.101, 0.02):
        r = solve_airfoil(X, Y, float(a), cr["U_inf"], cr["nu_inf"], W["MAC"],
                          cr["Tu_pct"], sweep_deg=W["le_sweep_deg"],
                          mach=cr["mach"], T_inf_K=cr["T_inf_K"])
        u = r["surfaces"]["upper"]
        x = np.asarray(u["x"], float)
        N = np.asarray(u["n_factor"], float)
        nc = np.asarray(u["n_crit"], float)
        xt = u["x_tr_chord"]; xt = float(xt) if xt == xt else 1.0
        band = (x > 2e-3) & (x < 0.20) & np.isfinite(N)
        if band.any():
            ratio = N[band]/np.maximum(nc[band], 1e-9)
            j = int(np.argmax(ratio))
            pk, xpk = float(ratio[j]), float(x[band][j])
        else:
            pk, xpk = float("nan"), float("nan")
        rows.append(dict(alpha_deg=round(float(a), 3),
                         Cd_counts=round(float(r["Cd"])*1e4, 2),
                         x_tr_upper_c=round(xt, 4),
                         N_over_Ncrit_forward=round(pk, 4),
                         x_forward_peak_c=round(xpk, 4),
                         forward_peak_has_crossed=bool(pk >= 1.0),
                         mechanism=u["onset_mech"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{SOL}/laminar_bucket_edge.csv", index=False)
    return df


def spanwise():
    X,Y=C.nlf16_panel_points(130)
    eta=np.linspace(0.0,0.98,12); rows=[]
    ai=induced_angle_deg(eta)
    # THE STRIPS AND THE WING LIFT ARE TWO SECTION MODELS, and the file now
    # carries both rather than leaving a reader to integrate one and get the
    # other.  c_l_section is the PANEL section's swept-strip lift, solved on
    # the LEADING-EDGE sweep because that is the angle the boundary layer and
    # therefore the transition prediction live on.  The wing C_L two tables
    # earlier comes from the lifting line, whose section slope is reduced by
    # cos of the QUARTER-CHORD sweep, which is the convention for a lift-curve
    # slope.  cos^2(12 deg) = 0.9568 against cos(9.89 deg) = 0.9851, so the two
    # differ by about three per cent before the panel section's own departure
    # from a linear a0(alpha - alpha_L0) is counted; measured, the strips run
    # from 0.961 of the lifting-line loading at the root to 0.916 at the tip.
    # Integrating the strip column therefore returns a wing C_L a few per cent
    # below the tabulated one, and c_l_lifting_line is here so that the gap is
    # visible at the station where it arises instead of only in the total.
    _ll = _LL_CACHE.get(round(float(cr["alpha_deg"]),6)) or lifting_line(cr["alpha_deg"])
    _a0e = float(_ll["a0_eff"]); _al0 = float(_ll["alpha_L0"])
    for e,a_ind in zip(eta,ai):
        chord=W["root_chord"]+e*(W["tip_chord"]-W["root_chord"])
        Re=cr["U_inf"]*chord/cr["nu_inf"]
        twist=e*W["twist_tip_deg"]
        # effective, not geometric: the downwash of the wing these strips
        # belong to is subtracted, from the same lifting-line solve that
        # reports the wing C_L two tables earlier.  See induced_angle_deg.
        ageo=cr["alpha_deg"]+twist
        aeff=ageo-float(a_ind)
        r=solve_airfoil(X,Y,aeff,cr["U_inf"],cr["nu_inf"],chord,cr["Tu_pct"],
                        sweep_deg=W["le_sweep_deg"],mach=cr["mach"],
                        T_inf_K=cr["T_inf_K"])
        u=r["surfaces"]["upper"]; l=r["surfaces"]["lower"]
        # a fully laminar surface counts as x_tr/c = 1.0 everywhere, so the
        # mean laminar fraction is defined at every station
        xu=u["x_tr_chord"]; xu=1.0 if xu!=xu else float(xu)
        xl=l["x_tr_chord"]; xl=1.0 if xl!=xl else float(xl)
        # WHERE THE INTERMITTENCY CONTOUR ACTUALLY CHANGES COLOUR.  Onset is
        # where gamma leaves zero; the boundary an eye reads off the 3-D
        # intermittency figure is the gamma = 0.5 contour, and transition has a
        # LENGTH, so the two are not the same station.  Measured over this
        # span the second lies 0.14 to 0.20 chord aft of the first, which is
        # more than the whole difference between the cruise upper and lower
        # onsets - so a reader invited to read "the transition front" off that
        # figure was being invited to read the wrong number.  Published so the
        # report can quote it instead of asserting it.
        def _x_at_gamma(sf, q):
            xs=np.asarray(sf["x"],float); g=np.asarray(sf["gamma"],float)
            m=np.isfinite(g) & np.isfinite(xs)
            if not m.any() or not (g[m] >= q).any(): return None
            return round(float(xs[m][int(np.argmax(g[m] >= q))]), 3)
        rows.append(dict(eta=round(e,3), y_m=round(e*W["span_b"]/2,3),
            chord_m=round(chord,3), Re_local=round(Re,-2),
            alpha_geom_deg=round(ageo,2),
            alpha_induced_deg=round(float(a_ind),2),
            alpha_eff_deg=round(aeff,2),
            c_l_section=round(r["Cl"],4),
            c_l_lifting_line=round(_a0e*np.radians(aeff-_al0),4),
            xtr_upper_c=round(xu,3), xtr_lower_c=round(xl,3),
            x_gamma50_upper_c=_x_at_gamma(u,0.5),
            x_gamma50_lower_c=_x_at_gamma(l,0.5),
            Cd_section=round(r["Cd"],5),
            laminar_fraction=round(0.5*(xu+xl),3)))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/spanwise_distribution.csv",index=False)
    return df

def pressure_field(cond,name):
    X,Y=C.nlf16_panel_points(130)
    gx=np.linspace(-0.4,1.4,241); gy=np.linspace(-0.6,0.6,161)
    Xg,Yg=np.meshgrid(gx,gy)
    Vx,Vy,Cp=velocity_field(X,Y,cond["alpha_deg"],Xg,Yg,U=cond["U_inf"],
                            mach=cond["mach"])
    # Mask the body, using the SAME closed polygon the panels were built on.
    # Masking against a separately sampled contour left a ragged band of
    # unmasked cells straddling the surface, and those cells sit on the panel
    # singularity, so they carried velocities of order ten times free stream.
    from matplotlib.path import Path
    poly=np.column_stack([X,Y])
    # Path.contains_points grows or shrinks the test region by `radius`
    # according to the WINDING of the polygon, and these points run clockwise
    # (TE -> lower -> LE -> upper -> TE), so the positive radius used here was
    # shrinking the mask: it removed fewer cells than radius 0 and left a ring
    # of them within a panel length of the surface, sitting on the panel
    # singularity, which showed as speckle hugging the aerofoil on the C_p
    # contour.  The sign is taken from the signed area so the region always
    # grows, whichever way the points are ordered.
    area=0.5*np.sum(X[:-1]*Y[1:]-X[1:]*Y[:-1])
    grow=0.010 if area > 0 else -0.010
    inside=Path(poly).contains_points(np.column_stack([Xg.ravel(),Yg.ravel()]),
                                      radius=grow).reshape(Xg.shape)
    # The mask has to reach the VELOCITY COMPONENTS too, and it did not.  C_p
    # and the speed were blanked and Vx, Vy were published raw, so this file
    # asserted two contradictory things about the same 2161 of its 38,801
    # cells: speed_ms empty, and beside it a Vx, Vy pair whose magnitude is
    # speed_ms by definition - and which agrees with it to 1e-4 everywhere it
    # is published.  Recomputed on the blanked rows those components run to
    # 162.1 m/s against a 131.0 m/s free stream, because that is the panel
    # singularity, which is the whole reason the cells are masked.
    #
    # It reached the figures as well as the data.  gen_postprocessing's quiver
    # is coloured by the speed, so 98 arrows inside the section were drawn
    # with a NaN colour - invisible only because a NaN maps to the colormap's
    # transparent "bad" entry and the white body is stroked over the top at a
    # higher zorder.  Two accidents, either of which could stop being true.
    Cp=np.where(inside,np.nan,Cp)
    spd=np.sqrt(Vx**2+Vy**2); spd=np.where(inside,np.nan,spd)
    Vx=np.where(inside,np.nan,Vx); Vy=np.where(inside,np.nan,Vy)
    # Rounded to the precision these quantities are meaningful to, as every
    # other CSV in this project is.  At full float64 this file was four
    # megabytes of seventeen-significant-figure numbers whose last bits move
    # with the BLAS thread count, so a regeneration that changed nothing
    # physical still rewrote both field files and both .npz - which is a large
    # part of why the history is four times the size of the working tree.
    # `+ 0.0` collapses NEGATIVE zero, which rounding a small negative produces
    # and which these two files carried 249 times between them.  It is a
    # published data file; "-0.0" in it is a sign that is not there.
    df=pd.DataFrame({"x_c":Xg.ravel().round(6)+0.0,"y_c":Yg.ravel().round(6)+0.0,
                     "Cp":Cp.ravel().round(5)+0.0,
                     "Vx_ms":Vx.ravel().round(4)+0.0,
                     "Vy_ms":Vy.ravel().round(4)+0.0,
                     "speed_ms":spd.ravel().round(4)+0.0})
    df.to_csv(f"{SOL}/field_pressure_{name}.csv",index=False)
    # No .npz beside it.  One was written here for years, 1.8 MB per case, and
    # NOTHING read it: gen_postprocessing._solution_field reads the CSV, which
    # is the tracked, human-readable form.  It was not even the same numbers -
    # the CSV is rounded to the precision these quantities are meaningful to
    # and the .npz was not - so the project carried two copies of one field at
    # two precisions, with no check that they agreed and no reader for the
    # second.  The comment above it called it "a convenience for re-loading".
    return df

def bl_profiles(rc, write=True):
    """Wall-normal velocity and temperature profiles at four chordwise stations.

    The march carries theta and the shape factor at every station, so the
    profile is a solver output and not an assumed shape:

      * the laminar leg is the Falkner-Skan profile at the SOLVED laminar shape
        factor, read from the same family the closure functions come from, so
        its displacement-to-momentum ratio is the H the march computed;
      * the turbulent leg is the power law whose exponent that shape factor
        implies, H = (n+2)/n, i.e. n = 2/(H-1);
      * the two are blended by the same intermittency that blends C_f, theta
        and H, each on its own thickness.

    Each leg's thickness follows from its own profile and the marched momentum
    thickness - delta_99 = eta_99 theta/theta_eta for the similarity profile,
    delta = theta (n+1)(n+2)/n for the power law - rather than being assumed.

    An earlier version blended a sine against a fixed one-seventh power and
    took the thickness as a hand multiple of theta, 8 theta laminar and
    7 theta turbulent, times 1.6.  Nothing in it depended on the shape factor,
    which is the one quantity the two-equation march exists to provide, and the
    profiles it produced were the same two curves at every station and every
    flight condition.

    THE BLENDED PROFILE DOES NOT CARRY THE MARCHED SHAPE FACTOR, and the file
    says both numbers rather than leaving them to disagree in silence.

    The march blends INTEGRALS - theta = (1-g) theta_lam + g theta_turb - and
    this reconstruction blends VELOCITIES, u = (1-g) u_lam + g u_turb.  Those
    are not the same operation.  The displacement thickness is linear in u, so
    it survives: integrating the plotted profile at x/c = 0.60 returns
    1.5369e-3 m against 1.5396e-3 from the linear blend, two parts in a
    thousand.  The momentum thickness is QUADRATIC in u, and the pointwise
    blend carries a cross term u_lam u_turb that a blend of integrals does not:
    4.626e-4 m against 4.279e-4, eight per cent.  H = delta*/theta inherits the
    whole of that, so the plotted transitional profile integrates to H = 3.32
    where the march says 3.71.  At a station that is wholly laminar or wholly
    turbulent the cross term vanishes and the two agree to within one or two
    per cent, which is the interpolation error and nothing else.

    This is not repaired by rescaling: stretching y multiplies delta* and theta
    alike and leaves H exactly where it was.  Matching H would mean drawing a
    single-family profile AT the marched H - an assumed shape, which is what
    the paragraph above records having removed, and it would hide the two-layer
    structure that is the physical content of a transitional station.  So the
    blend stays and the profile's own shape factor is published beside the
    march's, as H_profile against H_shape.
    """
    import stability as _stab
    g=cr["gamma_air"]
    # Recovery factor by state, not one value for the whole surface.  The
    # profile reconstruction used the TURBULENT r = 0.89 at every station,
    # including the laminar ones, while _ref_temp_nu inside the solver has
    # always switched between Pr^(1/2) laminar and Pr^(1/3) turbulent.  The two
    # halves of the same report disagreed about the recovery of the same layer.
    _Pr=cr["Pr"]; r_lam=_Pr**0.5; r_turb=_Pr**(1.0/3.0)
    s=rc["surfaces"]["upper"]
    rows=[]; stations={"x/c=0.10":0.10,"x/c=0.30":0.30,
                       "x/c=0.60":0.60,"x/c=0.95":0.95}
    eta=np.linspace(0,1,40)

    def laminar_leg(H_lam, th_lam):
        """(delta_99, f(y/delta)) for the Falkner-Skan profile at this H."""
        # fs_profile_for_H returns (eta, f', f'', H, theta_eta, f''');
        # the third derivative rides along for the Orr-Sommerfeld operator
        e_fs,u_fs,_,_,th_eta,_=_stab.fs_profile_for_H(float(H_lam))
        i99=int(np.argmax(u_fs>=0.99))
        e99=float(e_fs[i99]) if i99 else float(e_fs[-1])
        delta=e99*th_lam/th_eta
        return delta, (lambda t: np.interp(np.clip(t,0.0,1.0)*e99, e_fs, u_fs))

    def turbulent_leg(H_turb, th_turb):
        """(delta, f(y/delta)) for the power law implied by this H."""
        n=2.0/max(float(H_turb)-1.0, 1e-3)
        delta=th_turb*(n+1.0)*(n+2.0)/n
        return delta, (lambda t: np.clip(t,0.0,1.0)**(1.0/n))

    for lab,xq in stations.items():
        i=int(np.argmin(np.abs(s["x"]-xq)))
        gam=float(s["gamma"][i])
        d_l,f_l=laminar_leg(s["H_lam"][i], s["theta_lam"][i])
        if gam>0.0:
            d_t,f_t=turbulent_leg(s["H_turb"][i], s["theta_turb"][i])
        else:
            d_t,f_t=d_l,f_l
        delta=(1.0-gam)*d_l+gam*d_t
        y=eta*delta
        u_Ue=(1.0-gam)*f_l(y/max(d_l,1e-12))+gam*f_t(y/max(d_t,1e-12))
        u_Ue=np.clip(u_Ue,0.0,1.0)
        # The shape factor of the profile actually plotted, integrated from it.
        # See the docstring: it is NOT the marched H at a transitional station,
        # because theta is quadratic in u and this blend is pointwise.
        _ds=float(np.trapz(1.0-u_Ue, y)); _th=float(np.trapz(u_Ue*(1.0-u_Ue), y))
        _Hp=_ds/_th if _th > 0 else float("nan")
        # The edge Mach number comes off the march, where solve_airfoil put
        # the value _edge_from_cp formed from the CORRECTED pressure.  It used
        # to be U_e/a_inf here - the free-stream speed of sound divided into
        # the speed of gas that is at T_e - which is exactly the error
        # _edge_from_cp's docstring records having removed from the solver, and
        # it left the two halves of the same report running different
        # compressible closures.  At cruise it reads M_e low by up to 1.4 per
        # cent across the four stations below.
        Ue=s["Ue"][i]; Me=float(s["Me"][i])
        r_rec=(1.0-gam)*r_lam+gam*r_turb
        T_Te=1+r_rec*(g-1)/2*Me**2*(1-u_Ue**2)
        # The stagnation temperature is the free stream's, so the ratio is
        # formed on the SAME Mach number the panel solve ran at - which on a
        # swept section is the normal-plane one, M_inf cos(L), because that is
        # what solve_airfoil handed the pressure correction.
        Tinf=cr["T_inf_K"]; M_ref=float(rc["mach_solve"])
        Te=Tinf*(1+(g-1)/2*M_ref**2)/(1+(g-1)/2*Me**2)
        Tabs=T_Te*Te
        for et,uu,tt,Ta in zip(eta,u_Ue,T_Te,Tabs):
            rows.append(dict(station=lab,x_c=round(xq,2),y_delta=round(et,3),
                y_mm=round(et*delta*1e3,4),u_Ue=round(uu,4),
                T_Te=round(tt,4),T_K=round(Ta,2),
                Me_edge=round(Me,3),H_shape=round(float(s["H"][i]),3),
                H_profile=round(_Hp,3),
                recovery_r=round(float(r_rec),4),
                delta_mm=round(delta*1e3,4),
                intermittency_gamma=round(gam,3),state=s["state"][i]))
    df=pd.DataFrame(rows)
    # Only a run at the shipped settings may overwrite the committed result.
    # tools/smoke.py exercises this routine on a reduced panel count, and a
    # check that mutates the repository is not a check - the first full gate
    # after it was added showed this file dirty for no reason anyone had asked
    # for.  The same guard run_nlf0416 already carries.
    if write:
        df.to_csv(f"{SOL}/bl_profiles_cruise.csv",index=False)
    return df

def nlf_vs_turbulent(rc):
    """Drag benefit: NLF (predicted transition) vs forced-fully-turbulent."""
    X,Y=C.nlf16_panel_points(130)
    # forced turbulent: tiny Re_theta_t -> trip at LE (A_BP huge)
    # A_BP scales the bypass onset threshold, so a small value trips the layer
    # at the first station; Tu = 5 % also switches the e^N branch off, which is
    # what makes the reference fully turbulent from the leading edge.
    cal_trip=dict(CAL); cal_trip.update(A_BP=0.02)
    # The reference must differ from the NLF case ONLY in where it transitions,
    # so it is run at the same Mach number and therefore with the same
    # compressible closures; solving it incompressibly, as an earlier version
    # did, put part of the quoted drag saving down to the change of flow model.
    rt=solve_airfoil(X,Y,cr["alpha_deg"],cr["U_inf"],cr["nu_inf"],W["MAC"],
                     5.0,sweep_deg=W["le_sweep_deg"],cal=cal_trip,
                     mach=cr["mach"],T_inf_K=cr["T_inf_K"])
    Cd_nlf=rc["Cd"]; Cd_turb=rt["Cd"]
    u=rc["surfaces"]["upper"]; l=rc["surfaces"]["lower"]
    # Laminar extent is a CHORDWISE fraction, the same quantity the transition
    # summary, the polar and the span-wise sweep report.  An earlier version
    # formed it from x_tr, which is the arc length from the stagnation point,
    # and divided that by the chord, and Table 1 of the report then quoted the
    # two next to each other.  NEITHER figure is restated here.  This comment
    # gave the arc-length form as 58.3 per cent and the chordwise one as 56.6,
    # and both had gone stale - the sentence explaining a stale-number bug was
    # itself one.  They are the s_tr_c and x_tr_c columns of
    # 04_solution/transition_summary.csv, and the mean_laminar_pct column of
    # 04_solution/nlf_vs_turbulent.csv.  A surface that stays laminar to
    # the trailing edge counts as 1.0, as it does everywhere else.
    def _xtr(sf):
        x=sf["x_tr_chord"]
        return 1.0 if x!=x else float(x)
    lam=0.5*(_xtr(u)+_xtr(l))
    saving=(Cd_turb-Cd_nlf)/Cd_turb*100.0
    # The drag saving is a percentage and gets its own column.  It used to be
    # written into the third row of Cd_counts, so every consumer read a
    # percentage out of a column headed "counts".
    rows=[("NLF (UTSS predicted transition)",round(Cd_nlf*1e4,1),
           round(lam*100,1),round(saving,1)),
          ("Fully turbulent (LE trip)",round(Cd_turb*1e4,1),0.0,0.0)]
    df=pd.DataFrame(rows,columns=["configuration","Cd_counts",
                                  "mean_laminar_pct",
                                  "viscous_drag_reduction_pct"])
    df.to_csv(f"{SOL}/nlf_vs_turbulent.csv",index=False)
    return df,Cd_nlf,Cd_turb

def _section_slope(mach, alphas=(-2.0, 2.0, 6.0)):
    """Section lift-curve slope [1/rad] and zero-lift incidence [deg].

    From the same panel method that produces every other force in this file,
    at the same Mach number, so the wing lift below is not built on a different
    aerodynamic model from the section lift beside it.
    """
    X,Y=C.nlf16_panel_points(130)
    cls=[]
    for a in alphas:
        xc,yc,Cp,V,th,S=panel_solve(X,Y,a,mach=mach)
        nx=-np.sin(th); ny=np.cos(th)
        Cn=-np.sum(Cp*ny*S); Ca=-np.sum(Cp*nx*S)
        al=np.radians(a)
        cls.append(Cn*np.cos(al)-Ca*np.sin(al))
    m,c0=np.polyfit(np.array(alphas,float),np.array(cls),1)   # per degree
    return float(m)*180.0/np.pi, float(-c0/m)


def lifting_line(alpha_deg=None, n_terms=40):
    """Wing C_L, span efficiency and induced drag by Prandtl's lifting line.

    integrated_forces.csv used to report the wing lift as C_L = 0.90 c_l, a
    ratio typed in rather than computed, and 0.90 is well above what this
    planform actually returns.  Glauert's monoplane equation costs one linear
    solve and uses only quantities this project already has.  With
    y = -(b/2) cos(theta),

        sum_n A_n sin(n theta) [ 4b/(a0 c(theta)) + n/sin(theta) ]
              = alpha(theta) - alpha_L0 ,

    summed over odd n for a symmetric wing, with a0 and alpha_L0 the section
    lift-curve slope and zero-lift incidence from the panel method, c(theta)
    the planform chord and alpha(theta) the geometric incidence carrying the
    wing's washout.  The section slope is reduced by cos of the quarter-chord
    sweep, which is the standard first-order swept-wing correction and is what
    makes this consistent with the 12 deg leading-edge sweep the strip sweep
    already applies to the cross-flow branch.  Then C_L = pi AR A_1 and
    C_Di = pi AR sum n A_n^2, so the span efficiency e = A_1^2/(sum n A_n^2)
    comes out of the same solve rather than being assumed.
    """
    b=W["span_b"]; AR=W["AR"]
    al_root=cr["alpha_deg"] if alpha_deg is None else float(alpha_deg)
    a0,al0=_section_slope(cr["mach"])
    # quarter-chord sweep of the trapezoidal planform
    tan_c4=(np.tan(np.radians(W["le_sweep_deg"]))
            + 0.25*(W["tip_chord"]-W["root_chord"])/(b/2.0))
    sweep_c4=np.degrees(np.arctan(tan_c4))
    a0_eff=a0*np.cos(np.radians(sweep_c4))

    N=n_terms
    ns=np.arange(1,2*N,2)                       # odd terms only (symmetric)
    thk=np.arange(1,N+1)*np.pi/(2.0*N)          # collocation, excludes the tip
    eta=np.abs(np.cos(thk))                     # |y|/(b/2)
    chord=W["root_chord"]+eta*(W["tip_chord"]-W["root_chord"])
    alpha=np.radians(al_root+eta*W["twist_tip_deg"]-al0)
    M=(np.sin(np.outer(thk,ns))
       *(4.0*b/(a0_eff*chord)[:,None] + ns[None,:]/np.sin(thk)[:,None]))
    A=np.linalg.solve(M,alpha)
    CL=np.pi*AR*A[0]
    sumn=float(np.sum(ns*A**2))
    CDi=np.pi*AR*sumn
    e=A[0]**2/sumn
    # Induced incidence at each collocation station, alpha_i = sum n A_n
    # sin(n theta)/sin(theta).  The same solve that gives the wing lift gives
    # the downwash, and the span-wise strip sweep needs it: a strip run at the
    # GEOMETRIC incidence is a two-dimensional aerofoil, not a station on a
    # finite wing.  Returned sorted in eta so a consumer can interpolate.
    ai=(np.sin(np.outer(thk,ns))@(ns*A))/np.sin(thk)
    k=np.argsort(eta)
    return dict(CL=float(CL), CDi=float(CDi), e=float(e), a0=a0,
                alpha_L0=al0, sweep_c4=float(sweep_c4), a0_eff=a0_eff,
                eta_c=eta[k], alpha_i_deg=np.degrees(ai[k]))


_LL_CACHE={}

def induced_angle_deg(eta, alpha_deg=None):
    """Downwash angle [deg] at span fraction eta, from the lifting-line solve.

    The span-wise sweep and the 3-D field both run a two-dimensional section at
    each station.  Doing that at the geometric incidence - root incidence plus
    washout, which is what alpha_eff_deg used to report - leaves out the
    downwash of the wing the stations belong to, and the report computes that
    downwash two sections earlier: a strip run at the geometric incidence is a
    two-dimensional aerofoil, not a station on a finite wing.

    No figures for it here.  This docstring gave the downwash as 0.61 deg and
    the section lift it costs as 0.517 against 0.432 at the root and 0.107
    against 0.022 at the tip, "a factor of five"; none of the five reproduces.
    04_solution/spanwise_distribution.csv carries alpha_induced_deg,
    alpha_eff_deg and c_l_section at every station, which is where the effect
    should be read: it is a large correction across the whole span, not a tip
    effect, and it is not a factor of five anywhere.

    Cached per incidence: the monoplane solve is one linear system but the
    section lift-curve slope it needs costs three panel solves.
    """
    key=round(float(cr["alpha_deg"] if alpha_deg is None else alpha_deg),6)
    if key not in _LL_CACHE:
        _LL_CACHE[key]=lifting_line(key)
    ll=_LL_CACHE[key]
    return np.interp(np.asarray(eta,float), ll["eta_c"], ll["alpha_i_deg"])


def _lifting_line_check():
    """The monoplane solve, against the one case with a closed-form answer.

    An elliptic planform must return e = 1 exactly and
    C_L = a0/(1 + a0/(pi AR)) (alpha - alpha_L0); a linear twist must shift
    C_L by a constant without changing the slope.  Both are asserted here and
    the check runs with the pipeline, so the wing lift in Table 13 is not
    taking anyone's word for the implementation.
    """
    def solve(chord, alpha_rad, AR, b=1.0, a0=2.0*np.pi, N=60):
        ns=np.arange(1,2*N,2); thk=np.arange(1,N+1)*np.pi/(2.0*N)
        M=(np.sin(np.outer(thk,ns))
           *(4.0*b/(a0*chord)[:,None] + ns[None,:]/np.sin(thk)[:,None]))
        A=np.linalg.solve(M,alpha_rad)
        ai=(np.sin(np.outer(thk,ns))@(ns*A))/np.sin(thk)
        return np.pi*AR*A[0], A[0]**2/float(np.sum(ns*A**2)), ai

    N=60; AR=8.0; a0=2.0*np.pi
    thk=np.arange(1,N+1)*np.pi/(2.0*N); eta=np.abs(np.cos(thk))
    ell=np.sqrt(np.maximum(1.0-eta**2,1e-12))
    ell=ell*(1.0/AR)/(np.pi/4.0)                      # scale to the given AR
    CL,e,ai=solve(ell,np.full(N,np.radians(1.0)),AR)
    exact=a0/(1.0+a0/(np.pi*AR))*np.radians(1.0)
    assert abs(CL-exact)/exact < 1e-6, "elliptic C_L off: %g vs %g"%(CL,exact)
    assert abs(e-1.0) < 1e-6, "elliptic span efficiency off: %g" % e
    # The downwash the span-wise strips are run at, against its closed form: on
    # an elliptic planform the induced angle is CONSTANT across the span and
    # equal to C_L/(pi AR).  This is the check that the strips are flown at the
    # right incidence, and it is asserted here because the span-wise table has
    # no other independent reference.
    ai_exact=CL/(np.pi*AR)
    assert abs(ai-ai_exact).max() < 1e-9, (
        "elliptic induced angle is not constant at C_L/(pi AR): spread %.2e, "
        "worst error %.2e" % (ai.max()-ai.min(), abs(ai-ai_exact).max()))

    tap=0.5; ch=(1.0+eta*(tap-1.0)); ch=ch*(1.0/AR)/(0.5*(1.0+tap))
    d=[solve(ch,np.radians(a+eta*(-3.0)),AR)[0] for a in (1.0,2.0,4.0)]
    assert abs((d[1]-d[0])-(d[2]-d[1])/2.0) < 1e-9, "twist is not an offset"
    return True


def transition_length_sensitivity():
    """What the case-study drag owes to the transition-length closure.

    The length is Dhawan & Narasimha's published correlation, stated in
    Re_theta - the variable in which the constant spot-formation rate their
    correlation assumes is exact - and equal to their published
    Re_lambda = 9 Re_x_t^0.75 wherever Re_theta = 0.664 sqrt(Re_x).  It is not
    an extrapolation on the wing; 06_validation/transition_length_forms.csv is
    that equivalence, measured on every plate and on this section.

    NO FIGURES HERE.  This docstring said the correlation "reproduces the
    measured extent of the skin-friction rise to within a factor of two" on
    four flat plates - a claim no artefact in this project supported until
    06_validation/transition_length_measured.csv was written, and which that
    file corrects in both halves: two of the four plates resolve a length at
    all, and on those two the model is within a third.  It also gave the wing's
    transition Reynolds number as 3.7e6 (transition_summary.csv says 3.554e6),
    the drag movement over the closing range as "a tenth of a count" (the sweep
    below returns 0.04), and the point at which the layer stops completing
    transition as "beyond twice the published value" (it is AT twice).  All
    four were typed, and none reproduced.

    Rather than damp the correlation, which would add an undeclared constant to
    a method whose claim is that it has none, the consequence is measured: the
    constant is swept over a factor of four and the section drag recorded,
    together with the transitional extent and whether it still closes on the
    section, into 04_solution/transition_length_sensitivity.csv.  That file is
    the answer; the README and the report read it.
    """
    X,Y=C.nlf16_panel_points(130); rows=[]
    for c in (2.25, 4.5, 9.0, 18.0, 36.0):
        r=solve_airfoil(X,Y,cr["alpha_deg"],cr["U_inf"],cr["nu_inf"],W["MAC"],
                        cr["Tu_pct"],sweep_deg=W["le_sweep_deg"],
                        mach=cr["mach"],cal=dict(C_len=c),
                        T_inf_K=cr["T_inf_K"])
        u=r["surfaces"]["upper"]; g=u["gamma"]; x=u["x"]
        i0=int(np.argmax(g>1e-6))
        done=bool((g>=0.99).any())
        i1=int(np.argmax(g>=0.99)) if done else None
        rows.append(dict(C_len=c,
            multiple_of_published=round(c/CAL["C_len"],2),
            Cd_counts=round(r["Cd"]*1e4,2),
            x_tr_c_upper=round(float(u["x_tr_chord"]),3),
            transitional_extent_pct_chord=(round(100*(x[i1]-x[i0]),1) if done else None),
            completes_before_TE=done))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/transition_length_sensitivity.csv",index=False)
    return df


def omitted_friction(r, xr, cond):
    """Streamwise drag still ahead of the station, integrated directly.

    IN THE SAME FRAME AS THE DRAG IT IS ADDED TO.  This used to return the
    chordwise integral alone, referred to U_n and c_n - a NORMAL-PLANE
    coefficient - and add it to Cd_counts, which squire_young() has already
    converted to the streamwise frame.  Two frames in one sum, worth 6.4
    per cent of the smaller column at the 12 degrees of this wing.

    The conversion has two terms, exactly as the drag does (E20c), and
    neither needs a trailing-edge evaluation.

    CHORDWISE.  The wall shear along the chord acts along e_n, whose
    streamwise component is cos(L); referring 2 theta_n to the streamwise
    chord c = c_n/cos(L) gives another, and the dynamic pressure a third:

        cos^3(L) * integral C_f (U_e,n/U_n)^2 d(s_n/c_n)

    SPAN-WISE.  Squire-Young's span-wise term carries the span-wise wall
    shear from the leading edge only as far as the evaluation station, so
    the friction still ahead has a span-wise part too.  It does NOT need
    theta_12 at the trailing edge, which is what this docstring previously
    claimed and is why the term was left out.  Under the same
    small-cross-flow closure the drag formula already uses, w/W = u/U_e,n,
    so the span-wise wall shear is the chordwise one scaled by W/U_e,n:

        tau_wz = (W/U_e,n) tau_wx = (1/2) rho Q sin(L) U_e,n C_f

    which integrates directly, over the same stations, to a streamwise
    share of

        cos(L) sin^2(L) * integral C_f (U_e,n/U_n) d(s_n/c_n)

    - the FIRST power of the velocity ratio against the second above, the
    same asymmetry _swept_drag_factor carries for the same reason.

    WHY BOTH TERMS.  Squire-Young's span-wise term is the span-wise friction
    from the leading edge to x_ref; adding x_ref to the trailing edge makes the
    span-wise total independent of x_ref, so the sum this sweep is about
    becomes

        C_d + omitted = cos^3(L) [ chordwise wake + chordwise friction ]
                        + a constant,

    which is a single frame throughout and is what the invariance claim can
    honestly be made about.

    IT IS NOT CHOSEN FOR GIVING THE SMALLEST SPREAD, and it does not.  Measured
    from 0.90c up: the unconverted form gave 1.19 counts, the chordwise
    half-correction gives 1.31, and this gives 1.23.  The mixed form's 1.19 is
    the smallest of the three and means nothing, because it is two frames added
    together; the argument for this one is the yawed flat plate below, where
    the answer is known independently and only this form returns it.  The
    0.04 counts between 1.19 and 1.23 changes no statement anywhere.

    LIMITS.  At zero sweep this is exactly the integral it replaces.  On a
    yawed flat plate U_e,n = U_n, so it returns
    cos(L)(cos^2 L + sin^2 L) integral C_f = cos(L) times the unswept
    friction, which is the independence principle's answer and the same
    check _swept_drag_factor is held to.
    """
    _tz=getattr(np,"trapezoid",None) or np.trapz
    cosL=r["cos_sweep"]; chord=W["MAC"]*cosL; Un=cond["U_inf"]*cosL
    # zero unless the solve actually ran in the normal plane, so
    # sweep_transform=False recovers the untransformed integral exactly
    L=np.radians(float(r["sweep_deg"])) if r.get("sweep_transform") else 0.0
    c3=np.cos(L)**3; cs2=np.cos(L)*np.sin(L)**2
    chordwise=0.0; spanwise=0.0
    for _s in (r["surfaces"]["upper"], r["surfaces"]["lower"]):
        _x=np.asarray(_s["x"],float); _cf=np.asarray(_s["Cf"],float)
        _ue=np.asarray(_s["Ue"],float)/Un; _a=np.asarray(_s["s"],float)/chord
        _m=_x>=xr
        if _m.sum()>1:
            chordwise+=_tz(_cf[_m]*_ue[_m]**2, _a[_m])
            spanwise +=_tz(_cf[_m]*_ue[_m],    _a[_m])
    return float(c3*chordwise + cs2*spanwise)


def squire_young_station_sensitivity():
    """Where Squire-Young is evaluated, and how nearly converged 0.98c is.

    The formula wants the trailing edge, and this section has a 26.8 degree
    wedge one, so the trailing edge is an inviscid stagnation point: the edge
    velocity goes to zero there physically and (U_e/U_inf)^((H+5)/2)
    degenerates.  At the last control point the formula returns 18 counts
    against 47.  The evaluation is therefore pulled forward to cal["sy_x_ref"].

    THE SPREAD ACROSS STATIONS IS NOT AN UNCERTAINTY BAND, which is how this
    project first reported it.  It is friction being correctly included.  A
    forward station is not a different estimate of the same drag; it is the
    drag of a shorter aerofoil.

    No figure for that is quoted here.  This docstring, the README and the
    report all carried "3.94 counts of friction against 3.65 of drag, the same
    quantity to a third of a count", and all three stayed put when the
    swept-drag formulation moved the drag they are differences of; the pair is
    3.05 and 4.84 now.  The six numbers the claim rests on are computed into
    04_solution/squire_young_station_summary.csv instead, so the next such
    change moves them.

    What settles the point is the SUM: the drag counted so far plus the
    friction still ahead, which varies by about a count from 0.90c up while the
    drag alone moves nearly five.  The friction 0.98c still omits is measured
    directly by integrating C_f over the remaining surface, for the climb
    condition as well as cruise - it was quoted for both and computed for
    neither.  Past 0.98c the formula turns over and falls, which is the
    inviscid singularity taking hold rather than drag being lost.

    The sweep also reports whether the shape factor at each station is still
    solved or has reached Head's H = 2.8 clamp, and carries an INDEPENDENT
    route to the drag - the integrated skin friction - so the two can be
    compared without going through Squire-Young at all.
    """
    X,Y=C.nlf16_panel_points(130); rows=[]

    for xr in (0.88,0.90,0.92,0.94,0.96,0.98,0.99,1.00):
        r=solve_airfoil(X,Y,cr["alpha_deg"],cr["U_inf"],cr["nu_inf"],W["MAC"],
                        cr["Tu_pct"],sweep_deg=W["le_sweep_deg"],
                        mach=cr["mach"],T_inf_K=cr["T_inf_K"],
                        cal=dict(sy_x_ref=xr))
        u=r["surfaces"]["upper"]; l=r["surfaces"]["lower"]
        omit=omitted_friction(r, xr, cr)
        rows.append(dict(x_ref=xr,
            x_evaluated_upper=round(float(u["x_squire_young"]),4),
            Cd_counts=round(r["Cd"]*1e4,2),
            friction_omitted_counts=round(omit*1e4,3),
            H_upper=round(float(u["H_te_squire_young"]),3),
            H_lower=round(float(l["H_te_squire_young"]),3),
            theta_te_c_upper=round(float(u["theta_te_c"]),6),
            H_on_Head_clamp=bool(u["H_sy_at_clip"] or l["H_sy_at_clip"])))
    df=pd.DataFrame(rows)
    # Against the SHIPPED station, whatever CAL says it is.  Comparing x_ref to
    # a hard-coded 0.98 ties this column to a constant it does not read: move
    # sy_x_ref and the reference row is either the wrong one or absent, and the
    # .iloc[0] then raises inside a generator rather than at the change.
    _ship = float(CAL["sy_x_ref"])
    _i = int(np.argmin(np.abs(df.x_ref.to_numpy(float) - _ship)))
    if abs(float(df.x_ref.iloc[_i]) - _ship) > 1e-9:
        raise ValueError("the shipped Squire-Young station, sy_x_ref = %g, is "
                         "not one of the stations this sweep visits: %s"
                         % (_ship, df.x_ref.tolist()))
    df["delta_from_shipped_counts"]=(df.Cd_counts
                                     - float(df.Cd_counts.iloc[_i])).round(2)
    # The column that settles it.  If the spread across stations were an
    # uncertainty, this would wander; if it is friction being included, the sum
    # of the drag counted so far and the friction still ahead is the same
    # number wherever it is evaluated.  It is, to about a count, from 0.90c up
    # to the station where the inviscid singularity takes the formula over.
    df["Cd_plus_omitted_counts"]=(df.Cd_counts+df.friction_omitted_counts).round(2)
    df.to_csv(f"{SOL}/squire_young_station_sensitivity.csv",index=False)

    # The three statements this sweep is quoted for, computed rather than read
    # off the table by eye.  All three were typed into the README, the report
    # and two docstrings as 3.94, 3.65 and "the same quantity to a third of a
    # count", and all three had gone stale by more than a count the moment the
    # swept-drag formulation changed the drag they are differences of.  The
    # climb figure had no generating source at all - the sweep runs the cruise
    # condition - so it is computed here too.
    x_lo = float(df.x_ref.min()); x_hi = _ship
    lo = df.iloc[0]; hi = df.iloc[_i]
    # The station the INVARIANT is measured from.  It is not x_lo: the drag has
    # not stopped moving by 0.88c, so the sum is quoted from 0.90c up.  Naming
    # it once is what keeps the two halves of the claim on one range - the
    # report and the README compared a spread measured over 0.90-0.98 against
    # "the drag alone moves 4.84", which is the 0.88-0.98 figure, and over the
    # range the spread is actually taken on the drag moves 3.40.  Both ends of
    # the comparison are generated here now, so they cannot be taken from
    # different rows again.
    X_INV = 0.90
    _j = int(np.argmin(np.abs(df.x_ref.to_numpy(float) - X_INV)))
    inv = df[(df.x_ref >= X_INV) & (df.x_ref <= x_hi)].Cd_plus_omitted_counts
    r_cl = solve_airfoil(X, Y, cl["alpha_deg"], cl["U_inf"], cl["nu_inf"],
                         W["MAC"], cl["Tu_pct"], sweep_deg=W["le_sweep_deg"],
                         mach=cl["mach"], T_inf_K=cl["T_inf_K"],
                         cal=dict(sy_x_ref=x_hi))
    sm = pd.DataFrame([dict(
        x_lo=x_lo, x_shipped=x_hi,
        friction_accumulated_counts=round(float(lo.friction_omitted_counts
                                                - hi.friction_omitted_counts), 3),
        squire_young_moves_counts=round(float(hi.Cd_counts - lo.Cd_counts), 3),
        difference_counts=round(float((hi.Cd_counts - lo.Cd_counts)
                                      - (lo.friction_omitted_counts
                                         - hi.friction_omitted_counts)), 3),
        x_invariant_lo=X_INV,
        invariant_spread_from_0p90_counts=round(float(inv.max() - inv.min()), 3),
        # The drag movement over THE SAME range the spread above is taken on,
        # which is what that spread has to be compared against.
        squire_young_moves_from_invariant_lo_counts=round(
            float(hi.Cd_counts - df.Cd_counts.iloc[_j]), 3),
        friction_omitted_cruise_counts=round(float(hi.friction_omitted_counts), 3),
        friction_omitted_climb_counts=round(omitted_friction(r_cl, x_hi, cl)*1e4, 3))])
    sm.to_csv(f"{SOL}/squire_young_station_summary.csv", index=False)
    return df, sm


def integrated_forces(rc):
    q=cr["q_inf"]; S=W["area_S"]
    ll=lifting_line()
    # trim incidence: C_L is linear in the root incidence, so two solves fix it
    CL_req=C.AIRCRAFT["mtow_kg"]*9.80665/(q*S)
    _a1,_a2=0.0,5.0
    _c1,_c2=lifting_line(_a1)["CL"],lifting_line(_a2)["CL"]
    al_trim=_a1+(CL_req-_c1)*(_a2-_a1)/(_c2-_c1)
    rows=[("Section lift coefficient Cl",round(rc["Cl"],4),"-"),
          ("Section profile drag Cd",round(rc["Cd"],5),"-"),
          ("Section Cd (counts)",round(rc["Cd"]*1e4,1),"counts"),
          ("Section L/D",round(rc["Cl"]/rc["Cd"],1),"-"),
          ("Section lift-curve slope a0 (panel)",round(ll["a0"],3),"1/rad"),
          ("Zero-lift incidence a_L0 (panel)",round(ll["alpha_L0"],3),"deg"),
          ("Quarter-chord sweep",round(ll["sweep_c4"],2),"deg"),
          ("Wing C_L (lifting line, taper + washout + sweep)",
           round(ll["CL"],4),"-"),
          ("Wing C_L / section c_l",round(ll["CL"]/rc["Cl"],3),"-"),
          ("Span efficiency e (lifting line)",round(ll["e"],4),"-"),
          ("Wing induced drag C_Di (lifting line)",round(ll["CDi"],5),"-"),
          ("Wing lift (lifting line)",round(ll["CL"]*q*S,0),"N"),
          # The case is defined by its section and its flight condition, and
          # the incidence is the SECTION design incidence.  It is not the trim
          # point of the aircraft the planform belongs to, and reporting a wing
          # lift without saying so invites the reader to compare it with the
          # weight and conclude the aeroplane does not fly.  The incidence that
          # does balance the weight is therefore reported beside it, from the
          # same solve.
          ("Wing C_L for level flight at MTOW",round(CL_req,4),"-"),
          ("Incidence for that C_L (lifting line)",round(al_trim,2),"deg"),
          ("Dynamic pressure q",round(q,1),"Pa"),
          ("Reynolds number Re_MAC",f"{cr['Re_MAC']:.3e}","-"),
          ("Mach number",cr["mach"],"-")]
    df=pd.DataFrame(rows,columns=["quantity","value","unit"])
    df.to_csv(f"{SOL}/integrated_forces.csv",index=False)
    return df

if __name__=="__main__":
    _lifting_line_check()
    rc=run_case(cr,"cruise"); rl=run_case(cl,"climb")
    ts=transition_summary(rc,rl)
    pol=aero_polar(); spn=spanwise(); bke=laminar_bucket_edge()
    pressure_field(cr,"cruise"); pressure_field(cl,"climb")
    bl_profiles(rc)
    nvt,cdn,cdt=nlf_vs_turbulent(rc)
    integrated_forces(rc)
    tls=transition_length_sensitivity()
    sys_,sys_sm=squire_young_station_sensitivity()
    print("=== TRANSITION SUMMARY ==="); print(ts.to_string(index=False))
    print("\n=== NLF vs TURBULENT ==="); print(nvt.to_string(index=False))
    print("\n=== TRANSITION-LENGTH SENSITIVITY ==="); print(tls.to_string(index=False))
    print("\n=== SQUIRE-YOUNG STATION SENSITIVITY ==="); print(sys_.to_string(index=False))
    print(sys_sm.to_string(index=False))
    print(f"\nCruise: Cl={rc['Cl']:.3f} Cd={rc['Cd']*1e4:.1f}cts  "
          f"Drag saving={ (cdt-cdn)/cdt*100:.1f}%")
    print("solution files:", sorted([f for f in os.listdir(SOL) if f.endswith('.csv')]))
