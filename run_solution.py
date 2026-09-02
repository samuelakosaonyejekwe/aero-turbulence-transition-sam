"""
run_solution.py
Execute the UTSS solver for the AETHER-NLF 25 case study and write every
engineering output as CSV (04_solution/).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0,"solver")
import case_config as C
from utss_solver import solve_airfoil, velocity_field, panel_solve, CAL

SOL="04_solution"; os.makedirs(SOL,exist_ok=True)
W=C.WING; cr=C.CRUISE; cl=C.CLIMB

def run_case(cond, name):
    X,Y=C.nlf16_panel_points(130)
    r=solve_airfoil(X,Y,cond["alpha_deg"],cond["U_inf"],cond["nu_inf"],
                    W["MAC"],cond["Tu_pct"],sweep_deg=W["le_sweep_deg"],
                    mach=cond["mach"])
    for surf in ["upper","lower"]:
        s=r["surfaces"][surf]
        # Rounded to the precision these quantities are meaningful to, as
        # every other CSV in this project is.  Unrounded, a Reynolds number
        # went into the report's sampled state tables as 1891858.3232883876.
        df=pd.DataFrame({
            "x_c":s["x"].round(6), "arc_s_m":s["s"].round(6),
            "Re_x":s["Re_x"].round(0),
            "Cp":s["Cp"].round(5), "Ue_ms":s["Ue"].round(4),
            "Ue_Uinf":(s["Ue"]/cond["U_inf"]).round(5),
            "theta_mm":(s["theta"]*1e3).round(5), "H_shape":s["H"].round(4),
            "Cf":s["Cf"].round(7), "Re_theta":s["Re_theta"].round(2),
            "Re_theta_trans":np.round(s["Re_theta_t"],1),
            "n_factor":np.round(s["n_factor"],4),
            "n_crit":np.round(s["n_crit"],3),
            "intermittency_gamma":s["gamma"].round(5),
            "state":s["state"]})
        df.to_csv(f"{SOL}/surface_{name}_{surf}.csv",index=False)
    return r

def transition_summary(rc, rl):
    rows=[]
    for nm,rr,cond in [("CRUISE",rc,cr),("CLIMB",rl,cl)]:
        for surf in ["upper","lower"]:
            s=rr["surfaces"][surf]
            xtr=s["x_tr"]/W["MAC"] if not np.isnan(s["x_tr"]) else 1.0
            rex=cond["U_inf"]*s["x_tr"]/cond["nu_inf"] if not np.isnan(s["x_tr"]) else np.nan
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
                sep_margin_H=round(float(s["sep_margin_H"]),3),
                x_sep_turb_c=(round(float(s["x_sep_turb_chord"]),3)
                              if s["x_sep_turb_chord"]==s["x_sep_turb_chord"]
                              else None)))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/transition_summary.csv",index=False)
    return df

def aero_polar():
    X,Y=C.nlf16_panel_points(130); rows=[]
    for a in np.arange(-3,8.01,1.0):
        r=solve_airfoil(X,Y,a,cr["U_inf"],cr["nu_inf"],W["MAC"],cr["Tu_pct"],
                        sweep_deg=W["le_sweep_deg"],mach=cr["mach"])
        u=r["surfaces"]["upper"]; l=r["surfaces"]["lower"]
        rows.append(dict(alpha_deg=a, Cl=round(r["Cl"],4), Cd=round(r["Cd"],5),
            L_over_D=round(r["Cl"]/max(r["Cd"],1e-9),1),
            # the thin-layer assumption Squire-Young rests on, made visible
            theta_te_c=round(r["theta_te_c"],5),
            xtr_upper_c=round(u["x_tr_chord"],3) if u["x_tr_chord"]==u["x_tr_chord"] else 1.0,
            xtr_lower_c=round(l["x_tr_chord"],3) if l["x_tr_chord"]==l["x_tr_chord"] else 1.0))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/aero_polar.csv",index=False)
    return df

def spanwise():
    X,Y=C.nlf16_panel_points(130)
    eta=np.linspace(0.0,0.98,12); rows=[]
    ai=induced_angle_deg(eta)
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
                        sweep_deg=W["le_sweep_deg"],mach=cr["mach"])
        u=r["surfaces"]["upper"]; l=r["surfaces"]["lower"]
        # a fully laminar surface counts as x_tr/c = 1.0 everywhere, so the
        # mean laminar fraction is defined at every station
        xu=u["x_tr_chord"]; xu=1.0 if xu!=xu else float(xu)
        xl=l["x_tr_chord"]; xl=1.0 if xl!=xl else float(xl)
        rows.append(dict(eta=round(e,3), y_m=round(e*W["span_b"]/2,3),
            chord_m=round(chord,3), Re_local=round(Re,-2),
            alpha_geom_deg=round(ageo,2),
            alpha_induced_deg=round(float(a_ind),2),
            alpha_eff_deg=round(aeff,2),
            c_l_section=round(r["Cl"],4),
            xtr_upper_c=round(xu,3), xtr_lower_c=round(xl,3),
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
    Cp=np.where(inside,np.nan,Cp)
    spd=np.sqrt(Vx**2+Vy**2); spd=np.where(inside,np.nan,spd)
    # Rounded to the precision these quantities are meaningful to, as every
    # other CSV in this project is.  At full float64 this file was four
    # megabytes of seventeen-significant-figure numbers whose last bits move
    # with the BLAS thread count, so a regeneration that changed nothing
    # physical still rewrote both field files and both .npz - which is a large
    # part of why the history is four times the size of the working tree.
    df=pd.DataFrame({"x_c":Xg.ravel().round(6),"y_c":Yg.ravel().round(6),
                     "Cp":Cp.ravel().round(5),
                     "Vx_ms":Vx.ravel().round(4),"Vy_ms":Vy.ravel().round(4),
                     "speed_ms":spd.ravel().round(4)})
    df.to_csv(f"{SOL}/field_pressure_{name}.csv",index=False)
    np.savez(f"{SOL}/field_pressure_{name}.npz",Xg=Xg,Yg=Yg,Cp=Cp,Vx=Vx,Vy=Vy,spd=spd)
    return df

def bl_profiles(rc):
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
    """
    import stability as _stab
    g=cr["gamma_air"]; a_inf=cr["a_sound"]
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
        e_fs,u_fs,_,_,th_eta=_stab.fs_profile_for_H(float(H_lam))
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
        Ue=s["Ue"][i]; Me=Ue/a_inf
        r_rec=(1.0-gam)*r_lam+gam*r_turb
        T_Te=1+r_rec*(g-1)/2*Me**2*(1-u_Ue**2)
        Tinf=cr["T_inf_K"]; Te=Tinf*(1+(g-1)/2*cr["mach"]**2)/(1+(g-1)/2*Me**2)
        Tabs=T_Te*Te
        for et,uu,tt,Ta in zip(eta,u_Ue,T_Te,Tabs):
            rows.append(dict(station=lab,x_c=round(xq,2),y_delta=round(et,3),
                y_mm=round(et*delta*1e3,4),u_Ue=round(uu,4),
                T_Te=round(tt,4),T_K=round(Ta,2),
                Me_edge=round(Me,3),H_shape=round(float(s["H"][i]),3),
                recovery_r=round(float(r_rec),4),
                delta_mm=round(delta*1e3,4),
                intermittency_gamma=round(gam,3),state=s["state"][i]))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/bl_profiles_cruise.csv",index=False)
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
                     mach=cr["mach"])
    Cd_nlf=rc["Cd"]; Cd_turb=rt["Cd"]
    u=rc["surfaces"]["upper"]; l=rc["surfaces"]["lower"]
    # Laminar extent is a CHORDWISE fraction, the same quantity the transition
    # summary, the polar and the span-wise sweep report.  An earlier version
    # formed it from x_tr, which is the arc length from the stagnation point,
    # and divided that by the chord: on the cruise section that reads 58.3 %
    # against the 56.6 % the chordwise stations give, and Table 1 of the report
    # then quoted the two next to each other.  A surface that stays laminar to
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
    downwash two sections earlier: an AR = 8.7 wing at the cruise C_L carries
    0.61 deg of it, against a root incidence of 1.5 deg.  Left out, the root
    strip runs at c_l = 0.517 instead of 0.432 and the tip strip at 0.107
    instead of 0.022, a factor of five.

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

    The length is Dhawan & Narasimha's published correlation,
    Re_lambda = 9 Re_x_t^0.75, and it is validated here on flat plates spanning
    Re_x_t = 6e4 to 1.4e6, where it reproduces the measured extent of the
    skin-friction rise to within a factor of two.  The cruise wing transitions
    at Re_x_t = 3.7e6, a factor of three beyond that range, and the correlation
    then returns a transitional zone of a third of the chord - longer than a
    real natural-laminar-flow section shows at this Reynolds number.

    Rather than damp the correlation, which would add an undeclared constant to
    a method whose claim is that it has none, the consequence is measured: the
    constant is swept over a factor of four and the section drag recorded.  It
    moves by a tenth of a count, so the reported drag does not depend on the
    part of the closure that is extrapolated.  Beyond twice the published value
    the layer no longer completes transition before the trailing edge, and
    there the closure does matter; that bound is reported too.
    """
    X,Y=C.nlf16_panel_points(130); rows=[]
    for c in (2.25, 4.5, 9.0, 18.0, 36.0):
        r=solve_airfoil(X,Y,cr["alpha_deg"],cr["U_inf"],cr["nu_inf"],W["MAC"],
                        cr["Tu_pct"],sweep_deg=W["le_sweep_deg"],
                        mach=cr["mach"],cal=dict(C_len=c))
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
    pol=aero_polar(); spn=spanwise()
    pressure_field(cr,"cruise"); pressure_field(cl,"climb")
    bl_profiles(rc)
    nvt,cdn,cdt=nlf_vs_turbulent(rc)
    integrated_forces(rc)
    tls=transition_length_sensitivity()
    print("=== TRANSITION SUMMARY ==="); print(ts.to_string(index=False))
    print("\n=== NLF vs TURBULENT ==="); print(nvt.to_string(index=False))
    print("\n=== TRANSITION-LENGTH SENSITIVITY ==="); print(tls.to_string(index=False))
    print(f"\nCruise: Cl={rc['Cl']:.3f} Cd={rc['Cd']*1e4:.1f}cts  "
          f"Drag saving={ (cdt-cdn)/cdt*100:.1f}%")
    print("solution files:", sorted([f for f in os.listdir(SOL) if f.endswith('.csv')]))
