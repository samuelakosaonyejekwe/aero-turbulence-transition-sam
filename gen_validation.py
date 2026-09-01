"""
gen_validation.py
Validate the UTSS universal transition solver against three credible,
independent, published flat-plate transition datasets, with ONE universal
calibration set (no per-case physics re-tuning).  Records all sources.

  ERCOFTAC T3A  - bypass transition, Tu=3.3%   (Roach & Brierley 1990)
  ERCOFTAC T3B  - bypass transition, Tu=6.0%   (Roach & Brierley 1990)
  Schubauer&Skramstad - natural transition, Tu=0.03% (NACA Rep.909, 1948)
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0,"solver")
import case_config as C
from utss_solver import solve_flat_plate, solve_airfoil, panel_solve
from uplot import apply_style, INK, INK_SOFT, PALETTE, new_fig, finish
import matplotlib.pyplot as plt

apply_style()
VAL="06_validation"; VP=os.path.join(VAL,"plots"); os.makedirs(VP,exist_ok=True)

# ----------------------------------------------------------------------
# Reference experimental data.  Cf = local skin-friction coefficient vs
# Re_x, together with the transition-onset momentum-thickness Reynolds
# number for each case.
#
# PROVENANCE.
#   T3A, T3B  - the Rolls-Royce Applied Science Laboratory hot-wire data
#               distributed as the ERCOFTAC Classic Collection case 020
#               (summary tabulations t3ay.dat / t3by.dat).  Re_x, C_f,
#               Re_theta and the LOCAL free-stream turbulence intensity are
#               taken directly from that source.  Tu decays along the plate
#               (T3A 3.04% -> 1.10%, T3B 5.95% -> 2.40%), and transition
#               onset is taken as the station of minimum measured C_f.
#   SS        - onset Reynolds number only, at the value quoted throughout
#               the literature; see the note in EXP below.
# ----------------------------------------------------------------------
EXP = {
 "T3A": dict(
   Re_x=[1.52e+04, 3.24e+04, 6.7e+04, 1.006e+05, 1.348e+05, 1.692e+05, 2.035e+05, 2.384e+05, 2.735e+05, 3.093e+05, 3.447e+05, 3.822e+05, 4.189e+05, 4.548e+05, 4.908e+05, 5.273e+05],
   Cf=[0.005203, 0.003723, 0.002645, 0.002272, 0.002098, 0.002209, 0.002703, 0.003801, 0.004849, 0.004861, 0.004722, 0.004553, 0.004418, 0.004292, 0.004207, 0.004079],
   Re_theta=[79.7, 117.4, 176.5, 224.9, 272.3, 322.8, 384.5, 456.3, 538.9, 627.5, 710.5, 796.5, 897, 980.2, 1055, 1137],
   Tu_local=[3.043, 2.793, 2.434, 2.197, 2.001, 1.882, 1.76, 1.647, 1.538, 1.451, 1.361, 1.295, 1.227, 1.206, 1.141, 1.101],
   x_m=None, Ue=None,
   Re_theta_t=272.3, Re_x_t=1.348e+05),
 "T3B": dict(
   Re_x=[1.51e+04, 2.77e+04, 4.31e+04, 5.91e+04, 8.93e+04, 1.245e+05, 1.885e+05, 2.53e+05, 3.181e+05, 3.822e+05, 4.469e+05, 5.794e+05, 7.018e+05, 8.31e+05, 9.57e+05],
   Cf=[0.005574, 0.004503, 0.003573, 0.00343, 0.00433, 0.005732, 0.005414, 0.00497, 0.004625, 0.004474, 0.004297, 0.004007, 0.0039, 0.003746, 0.003639],
   Re_theta=[82.9, 114.1, 146, 181.3, 243.4, 337.4, 502.1, 659.4, 848.5, 977.2, 1096, 1432, 1623, 1912, 2073],
   Tu_local=[5.952, 5.635, 5.442, 5.244, 5.019, 4.714, 4.334, 4.031, 3.724, 3.484, 3.276, 2.965, 2.749, 2.548, 2.4],
   x_m=None, Ue=None,
   Re_theta_t=181.3, Re_x_t=5.91e+04),
 "T3AM": dict(
   Re_x=[1.225e+05, 2.541e+05, 3.855e+05, 5.078e+05, 6.422e+05, 7.72e+05, 9.003e+05, 1.038e+06, 1.173e+06, 1.306e+06, 1.443e+06, 1.561e+06, 1.698e+06, 1.828e+06, 1.959e+06, 2.022e+06],
   Cf=[0.00188, 0.00125, 0.001027, 0.000901, 0.00078, 0.000733, 0.000661, 0.000624, 0.000603, 0.000565, 0.000535, 0.000557, 0.000583, 0.000735, 0.001193, 0.001514],
   Re_theta=[219.4, 326.9, 401.5, 466.6, 539.2, 579.4, 630.9, 684.8, 734.3, 784.8, 818.8, 861.5, 930.7, 999.7, 1131, 1192],
   Tu_local=[0.874, 0.793, 0.738, 0.685, 0.651, 0.61, 0.585, 0.564, 0.545, 0.525, 0.512, 0.498, 0.486, 0.478, 0.483, 0.469],
   x_m=None, Ue=None,
   Re_theta_t=818.8, Re_x_t=1.443e+06),
 "T3C4": dict(
   Re_x=[9600, 2.12e+04, 4.62e+04, 7.56e+04, 1.065e+05, 1.209e+05, 1.345e+05, 1.468e+05, 1.572e+05, 1.671e+05, 1.785e+05, 1.838e+05],
   Cf=[0.008666, 0.005735, 0.004096, 0.003549, 0.002964, 0.002501, 0.001934, 0.001236, 0.000539, 0.000187, 0.000183, 0.002202],
   Re_theta=[56.1, 84.8, 123.8, 150.4, 172.9, 191.1, 209.5, 238.1, 273.2, 309.3, 381.3, 627.8],
   Tu_local=[2.113, 1.714, 1.365, 1.079, 0.963, 0.901, 0.898, 0.883, 0.928, 0.938, 0.985, 1.29],
   x_m=[0.095, 0.195, 0.395, 0.595, 0.795, 0.895, 0.995, 1.095, 1.195, 1.295, 1.395, 1.495],
   Ue=[1.51, 1.63, 1.75, 1.9, 2, 2.02, 2.02, 2, 1.96, 1.93, 1.91, 1.84],
   Re_theta_t=381.3, Re_x_t=1.785e+05),
 "SS": dict(
   # NACA Report 909 presents its transition results as figures in a 1948
   # scan and no reliable digitisation was available to the author, so only
   # the onset Reynolds number is carried, at the value quoted throughout
   # the literature: Re_x,tr ~ 2.8e6 at Tu = 0.03%, equivalently
   # Re_theta_t ~ 1100 via Re_theta = 0.664 sqrt(Re_x).  No measured C_f
   # distribution is claimed for this case.
   Re_x=None, Cf=None, Re_theta=None, Tu_local=None,
   x_m=None, Ue=None,
   Re_theta_t=1100.0, Re_x_t=2.8e6),
}

def run_validation():
    """One turbulence treatment, not a choice between conventions: the local
    free-stream turbulence intensity is computed from the inlet value and the
    measured integral length scale of each rig by the k-epsilon decay law."""
    summ=[]; sources=[]
    for key in ["T3A","T3AM","T3B","T3C4","SS"]:
        v=C.VALIDATION[key]; ex=EXP[key]
        ue = ((ex["x_m"], ex["Ue"]) if ex.get("Ue") is not None else None)
        dec = ((ex["Re_x"], ex["Tu_local"]) if v.get("L_turb") is None
               and ex.get("Tu_local") is not None else None)
        r=solve_flat_plate(v["L"],v["U"],v["nu"],v["Tu_pct"],npts=900,
                           dUe=v["dUe"],L_turb=v.get("L_turb"),
                           Ue_dist=ue, Tu_decay=dec)
        it=r["i_tr"]; reth=float(r["Re_theta"][it])
        # the ERCOFTAC tables define Re_x on the LOCAL free-stream velocity,
        # so form it the same way; identical to U*x/nu on the ZPG plates
        rex=float(r["Ue"][it])*r["x_tr"]/v["nu"]
        dfs=pd.DataFrame({"Re_x":r["Re_x"],"Cf_solver":r["Cf"],
                          "Cf_laminar_blasius":r["Cf_lam_ref"],
                          "Cf_turbulent_ref":r["Cf_turb_ref"],
                          "intermittency_gamma":r["gamma"],
                          "Re_theta":r["Re_theta"]})
        dfs.to_csv(f"{VAL}/solver_{key}.csv",index=False)
        err=None
        if ex["Cf"] is not None:
            dfe=pd.DataFrame({"Re_x":ex["Re_x"],"Cf_experiment":ex["Cf"],
                              "Re_theta_experiment":ex["Re_theta"],
                              "Tu_local_pct":ex["Tu_local"]})
            dfe["source"]=v["source"]
            dfe.to_csv(f"{VAL}/experiment_{key}.csv",index=False)
            cfi=np.interp(ex["Re_x"],r["Re_x"],r["Cf"])
            err=round(float(np.mean(np.abs(cfi-np.array(ex["Cf"]))
                                    /np.array(ex["Cf"]))*100),1)
            # how far the laminar march under-predicts the measured, turbulence
            # thickened pre-transitional momentum thickness at onset
            io_=int(np.argmin(ex["Cf"]))
            ratio=ex["Re_theta"][io_]/(0.664*np.sqrt(ex["Re_x"][io_]))
        else:
            ratio=None
        summ.append(dict(case=v["name"], Tu_inlet_pct=v["Tu_pct"],
            L_turb_mm=(v.get("L_turb")*1e3 if v.get("L_turb") else None),
            Re_theta_t_exp=ex["Re_theta_t"], Re_theta_t_pred=round(reth,1),
            Re_theta_t_err_pct=round((reth-ex["Re_theta_t"])/ex["Re_theta_t"]*100,1),
            Re_x_tr_exp=f"{ex['Re_x_t']:.3e}", Re_x_tr_pred=f"{rex:.3e}",
            mean_Cf_err_pct=err,
            Re_theta_meas_over_blasius=(round(ratio,3) if ratio else None),
            mechanism=r["onset_mech"]))
        sources.append(dict(dataset=v["name"], source=v["source"]))
    df_sum=pd.DataFrame(summ); df_sum.to_csv(f"{VAL}/validation_summary.csv",index=False)
    # extended source list (validation + calibration + case study)
    src=[
     ("Validation","ERCOFTAC T3A flat plate",
      "Roach P.E. & Brierley D.H. (1990), 'The influence of a turbulent free "
      "stream on zero pressure gradient transitional boundary layer development', "
      "ERCOFTAC Workshop; reproduced in Savill (1993) and Langtry & Menter, AIAA J. 47(12), 2009."),
     ("Validation","ERCOFTAC T3B flat plate",
      "Roach & Brierley (1990), ERCOFTAC T3B (Tu=6%); Langtry & Menter, AIAA J. 47(12), 2009."),
     ("Validation","Schubauer & Skramstad natural transition",
      "Schubauer G.B. & Skramstad H.K. (1948), 'Laminar boundary-layer "
      "oscillations and transition on a flat plate', NACA Report 909."),
     ("Calibration","Bypass onset correlation (AGS)",
      "Abu-Ghannam B.J. & Shaw R. (1980), 'Natural transition of boundary layers "
      "- the effects of turbulence, pressure gradient and flow history', J. Mech. Eng. Sci. 22(5)."),
     ("Calibration","Laminar integral method",
      "Thwaites B. (1949), 'Approximate calculation of the laminar boundary layer', Aero. Quarterly 1."),
     ("Calibration","Turbulent entrainment / Cf",
      "Head M.R. (1958) entrainment method; Ludwieg H. & Tillmann W. (1950) skin-friction law."),
     ("Calibration","Intermittency / transition length",
      "Narasimha R. (1957); Dhawan S. & Narasimha R. (1958), J. Fluid Mech. 3."),
     ("Calibration","Cross-flow criterion",
      "Arnal D. (1984), C1 cross-flow criterion; Mayle R.E. (1991), J. Turbomachinery 113."),
     ("Case study","NLF aerofoil design reference",
      "Somers D.M. (1981), 'Design and experimental results for a natural-laminar-flow "
      "aerofoil for general aviation applications', NASA TP-1861 (NLF(1)-0416)."),
     ("Case study","Panel method",
      "Kuethe A.M. & Chow C.Y. (1998), 'Foundations of Aerodynamics', 5th ed., Wiley."),
     ("Case study","ISA atmosphere",
      "ICAO Standard Atmosphere (ISO 2533:1975), FL360 properties."),
    ]
    pd.DataFrame(src,columns=["category","item","reference"]).to_csv(
        f"{VAL}/sources_and_references.csv",index=False)
    return df_sum


# ----------------------------------------------------------------------
#  Swept-wing cross-flow validation (Dagenhart & Saric, NASA TP-1999-209344)
# ----------------------------------------------------------------------
NLF415 = "01_geometry/nlf2_0415.dat"

def _nlf415_points():
    """NLF(2)-0415 coordinates, returned TE -> lower -> LE -> upper -> TE."""
    pts=[]
    for line in open(NLF415):
        q=line.split()
        if len(q)==2:
            try: pts.append((float(q[0]),float(q[1])))
            except ValueError: pass
    a=np.array(pts)
    return a[::-1,0], a[::-1,1]

def run_swept():
    """The only case in which the cross-flow criterion is the selected one."""
    v=C.SWEPT
    X,Y=_section_points(NLF415)
    rows=[]
    for Rec,xm in zip(v["Re_c"], v["x_tr_c"]):
        U=Rec*v["nu"]/v["chord_m"]
        r=solve_airfoil(X,Y,v["alpha_deg"],U,v["nu"],v["chord_m"],v["Tu_pct"],
                        sweep_deg=v["sweep_deg"])
        u=r["surfaces"]["upper"]
        xp=u["x_tr_chord"]; xp=1.0 if xp!=xp else float(xp)
        rows.append(dict(Re_c=f"{Rec:.3e}", x_tr_c_exp=xm,
                         x_tr_c_pred=round(xp,3),
                         err_pct=round((xp-xm)/xm*100,1),
                         Re_theta_at_onset=round(float(u["Re_theta"][u["i_tr"]]),1),
                         mechanism=u["onset_mech"]))
    df=pd.DataFrame(rows); df.to_csv(f"{VAL}/swept_wing_crossflow.csv",index=False)
    pd.DataFrame([dict(dataset=v["name"], source=v["source"])]).to_csv(
        f"{VAL}/swept_wing_source.csv", index=False)
    # plot
    fig,ax=new_fig(8.4,5.4)
    rc=[float(x) for x in v["Re_c"]]
    ax.plot(rc, v["x_tr_c"], "o", ms=9, color=PALETTE[1], mec=INK_SOFT,
            label="measured (naphthalene flow visualisation)")
    ax.plot(rc, df["x_tr_c_pred"], "-s", ms=7, lw=2.2, color=PALETTE[0],
            label="UTSS, cross-flow criterion")
    ax.set_xlabel("chord Reynolds number  Re_c")
    ax.set_ylabel("transition location  x/c")
    ax.set_ylim(0,1.0)
    ax.set_title("Cross-flow validation: 45° swept NLF(2)-0415, α = −4°")
    ax.legend(loc="upper right",fontsize=10)
    finish(fig,f"{VP}/val_swept_crossflow.png",
           caption="Source: "+v["source"][:95]+"...")
    print("swept-wing cross-flow: mean |err| = %.1f%%"
          % float(np.mean(np.abs(df["err_pct"]))))
    return df


def _section_points(fn, n=200):
    """Load a Selig-format section and re-spline it, TE -> lower -> LE -> upper.

    The surfaces are splined against sqrt(x) rather than x.  A rounded
    leading edge has y ~ sqrt(x) there, so y is very nearly linear in that
    variable and a cubic spline reproduces the nose smoothly; interpolating
    linearly in x, as a first version of this routine did, turns a sparsely
    tabulated nose into a polygon and the panel method then returns a
    leading-edge pressure distribution that oscillates by order unity in
    C_p.  That is not a small cosmetic error: the critical Reynolds number
    of the amplification envelope is exponentially sensitive to the shape
    factor, so the spurious gradients drive the e^N integral and transition
    is predicted far too early.
    """
    from scipy.interpolate import CubicSpline
    pts=[]
    for line in open(fn):
        q=line.split()
        if len(q)==2:
            try: pts.append((float(q[0]),float(q[1])))
            except ValueError: pass
    a=np.array(pts); i=int(np.argmin(a[:,0]))
    up=a[:i+1][::-1]; lo=a[i:]                       # each runs LE -> TE
    beta=np.linspace(0.0,np.pi,n+1); xc=0.5*(1.0-np.cos(beta))
    out=[]
    for surf in (up, lo):
        x=np.clip(surf[:,0],0.0,None); y=surf[:,1]
        k=np.argsort(x); x,y=x[k],y[k]
        keep=np.concatenate([[True], np.diff(x)>1e-12])
        x,y=x[keep],y[keep]
        out.append(CubicSpline(np.sqrt(x), y, bc_type="natural")(np.sqrt(xc)))
    yu,yl=out
    return (np.concatenate([xc[::-1],xc[1:]]),
            np.concatenate([yl[::-1],yu[1:]]))


def run_swept2():
    """Independent swept-wing check.  Nothing is calibrated on this set."""
    v=C.SWEPT2
    X,Y=_section_points(v["section"])
    rows=[]
    for sw,al,xm,Rec in zip(v["sweep_deg"],v["alpha_deg"],v["x_tr_c"],v["Re_c"]):
        U=Rec*v["nu"]/v["chord_m"]
        r=solve_airfoil(X,Y,al,U,v["nu"],v["chord_m"],v["Tu_pct"],sweep_deg=sw)
        u=r["surfaces"]["upper"]; xp=u["x_tr_chord"]
        xp=1.0 if xp!=xp else float(xp)
        rows.append(dict(sweep_deg=sw, alpha_deg=al, Re_c=f"{Rec:.3e}",
                         x_tr_c_exp=xm, x_tr_c_pred=round(xp,3),
                         err_pct=round((xp-xm)/xm*100,1),
                         mechanism=u["onset_mech"]))
    df=pd.DataFrame(rows); df.to_csv(f"{VAL}/swept_wing_independent.csv",index=False)
    pd.DataFrame([dict(dataset=v["name"], source=v["source"])]).to_csv(
        f"{VAL}/swept_wing_independent_source.csv", index=False)
    fig,ax=new_fig(8.4,5.4)
    ax.plot(v["sweep_deg"], v["x_tr_c"], "o", ms=9, color=PALETTE[1],
            mec=INK_SOFT, label="measured (Boltz et al.)")
    ax.plot(v["sweep_deg"], df["x_tr_c_pred"], "-s", ms=7, lw=2.2,
            color=PALETTE[0], label="UTSS (nothing calibrated here)")
    ax.set_xlabel("leading-edge sweep  Λ  [deg]")
    ax.set_ylabel("transition location  x/c")
    ax.set_ylim(0,0.6)
    ax.set_title("Independent swept-wing check: NACA 64(2)A015, Λ = 0–50°")
    ax.legend(loc="upper right",fontsize=10)
    finish(fig,f"{VP}/val_swept_independent.png",
           caption="Source: "+v["source"][:95]+"...")
    print("independent swept-wing: mean |err| = %.1f%%"
          % float(np.mean(np.abs(df["err_pct"]))))
    return df


def _cl_curve(X, Y, mach, alphas=(-4.0, 2.0, 8.0)):
    """Quadratic fit of the inviscid c_l against incidence.

    The panel influence matrix does not depend on incidence, but the lift does
    so through both the circulation (linear) and C_p = 1 - (V/U_inf)^2
    (quadratic), and the departure from a straight line reaches 0.008 in c_l
    over the range needed here.  Three inviscid solves therefore fix the curve
    to better than 1e-4 in c_l, and the incidence that matches any measured
    lift coefficient follows analytically - no iteration, and no boundary-layer
    march wasted on trial incidences.
    """
    cls = []
    for a in alphas:
        xc, yc, Cp, V, th, S = panel_solve(X, Y, a, mach=mach)
        nx = -np.sin(th); ny = np.cos(th)
        Cn = -np.sum(Cp*ny*S); Ca = -np.sum(Cp*nx*S)
        al = np.radians(a)
        cls.append(Cn*np.cos(al) - Ca*np.sin(al))
    return np.polyfit(np.array(alphas, float), np.array(cls), 2)


def _alpha_for_cl(coef, cl_target):
    """Invert the quadratic c_l(alpha) fit, taking the root nearest zero lift."""
    a, b, c = coef
    r = np.roots([a, b, c - cl_target])
    r = np.array([z.real for z in r if abs(z.imag) < 1e-9])
    if not len(r):
        return float((cl_target - c)/b)
    return float(r[np.argmin(np.abs(r))])


def _panel_cl(X, Y, alpha_deg, mach):
    xc, yc, Cp, V, th, S = panel_solve(X, Y, alpha_deg, mach=mach)
    nx = -np.sin(th); ny = np.cos(th)
    Cn = -np.sum(Cp*ny*S); Ca = -np.sum(Cp*nx*S)
    al = np.radians(alpha_deg)
    return float(Cn*np.cos(al) - Ca*np.sin(al))


def _trim(X, Y, coef, cl_target, mach, tol=1e-3):
    """Analytic inversion of the c_l(alpha) fit, refined once if necessary.

    The quadratic fit is good to ~1e-4 in c_l over the design range but drifts
    to ~0.02 near alpha = -13 deg, at the extreme negative-lift end of the
    TP-1861 data.  One secant step against the true panel solution removes that
    drift; the refinement costs a single inviscid solve and is only taken when
    the fit is outside tolerance.
    """
    a = _alpha_for_cl(coef, cl_target)
    cl = _panel_cl(X, Y, a, mach)
    if abs(cl - cl_target) > tol:
        slope = coef[1] + 2.0*coef[0]*a          # dc_l/dalpha from the fit
        a2 = a + (cl_target - cl)/slope
        cl2 = _panel_cl(X, Y, a2, mach)
        if abs(cl2 - cl_target) < abs(cl - cl_target):
            a, cl = a2, cl2
    return a, cl


def run_nlf0416(Tu_pct=None, quiet=False):
    """Aerofoil transition validation against NASA TP-1861, Fig. 9.

    86 transition locations on the NLF(1)-0416 section, both surfaces, at four
    chord Reynolds numbers.  Nothing in the model is calibrated on this set.
    Each measured point is matched by trimming the incidence to the measured
    lift coefficient; the experimental uncertainty is half the 0.05c orifice
    pitch, so a prediction is counted as within the measurement bracket when it
    falls inside +/-0.025c of the tabulated midpoint.
    """
    v = C.NLF0416
    Tu = v["Tu_pct"] if Tu_pct is None else Tu_pct
    X, Y = _section_points(v["section"], n=220)
    half = v["orifice_pitch"]/2.0
    coef = _cl_curve(X, Y, v["mach"])
    rows = []
    for Rec in sorted(v["data"]):
        U = Rec*v["nu"]/v["chord_m"]
        for surf in ("upper", "lower"):
            for cl_m, x_m in v["data"][Rec][surf]:
                al, cl_got = _trim(X, Y, coef, cl_m, v["mach"])
                r = solve_airfoil(X, Y, al, U, v["nu"], v["chord_m"], Tu,
                                  mach=v["mach"])
                s = r["surfaces"][surf]
                xp = s["x_tr_chord"]
                xp = 1.0 if xp != xp else float(xp)
                # A prediction at either end of the surface is not a transition
                # location: it means the march either separated at the nose or
                # never reached onset.  Such points are kept in the table but
                # flagged, so that they are visible rather than averaged in as
                # though they were ordinary errors.
                degenerate = bool(xp < 0.02 or xp > 0.93)
                rows.append(dict(
                    Re_c=f"{Rec:.1e}", surface=surf, c_l_exp=cl_m,
                    alpha_deg=round(al, 3), c_l_solver=round(cl_got, 4),
                    x_tr_c_exp=x_m, x_tr_c_pred=round(xp, 3),
                    err_c=round(xp-x_m, 3),
                    err_pct=round((xp-x_m)/x_m*100, 1),
                    within_bracket=bool(abs(xp-x_m) <= half),
                    degenerate=degenerate,
                    mechanism=s["onset_mech"]))
    df = pd.DataFrame(rows)
    if Tu_pct is None:
        df.to_csv(f"{VAL}/aerofoil_nlf0416.csv", index=False)
        pd.DataFrame([dict(dataset=v["name"], source=v["source"],
                           n_points=len(df), Tu_pct=Tu,
                           orifice_pitch=v["orifice_pitch"])]).to_csv(
            f"{VAL}/aerofoil_nlf0416_source.csv", index=False)
    if not quiet:
        for surf in ("upper", "lower"):
            d = df[df.surface == surf]
            print("NLF(1)-0416 %s surface: %d pts, mean |err| = %.3fc (%.1f%%), "
                  "%d/%d inside the +/-0.025c measurement bracket"
                  % (surf, len(d), float(np.mean(np.abs(d.err_c))),
                     float(np.mean(np.abs(d.err_pct))),
                     int(d.within_bracket.sum()), len(d)))
    return df



def plot_nlf0416(df=None):
    """Reproduce the layout of Fig. 9 of TP-1861 with the predictions overlaid.

    One panel per chord Reynolds number, transition location against lift
    coefficient, both surfaces.  The measurement is drawn with the +/-0.025c
    bar set by the orifice pitch, so the reader can see directly which
    predictions fall inside the experimental bracket.
    """
    v = C.NLF0416
    if df is None:
        df = run_nlf0416(quiet=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), sharex=True, sharey=True)
    half = v["orifice_pitch"]/2.0
    for ax, Rec in zip(axes.ravel(), sorted(v["data"])):
        tag = f"{Rec:.1e}"
        for surf, col, mk in (("upper", PALETTE[0], "o"), ("lower", PALETTE[3], "s")):
            d = df[(df.Re_c == tag) & (df.surface == surf)].sort_values("c_l_exp")
            ax.errorbar(d.x_tr_c_exp, d.c_l_exp, xerr=half, fmt=mk, ms=6,
                        color=col, mec=INK_SOFT, lw=0, elinewidth=1.1,
                        capsize=2.5, ecolor=col, alpha=0.95,
                        label=f"measured, {surf}")
            # The line is broken at conditions outside the incidence envelope of
            # an attached-flow formulation: beyond about 8.5 deg the laminar
            # layer separates within a few per cent of chord on this section.
            # Those points remain in every error statistic reported; they are
            # only excluded from the polyline, which would otherwise sweep
            # across the panel and misrepresent the trend.
            out = d.degenerate | (d.alpha_deg.abs() > 8.5)
            good = d[~out]
            ax.plot(good.x_tr_c_pred, good.c_l_exp, "-", lw=2.0, color=col,
                    alpha=0.85, label=f"UTSS, {surf}")
            bad = d[out]
            if len(bad):
                ax.plot(bad.x_tr_c_pred, bad.c_l_exp, "x", ms=7, mew=1.6,
                        color=PALETTE[2], lw=0,
                        label="outside ±8.5° envelope" if surf == "upper" else None)
        ax.set_title(r"$R = %.1f\times10^6$" % (Rec/1e6), fontsize=11)
        ax.set_xlim(0, 1.0); ax.set_ylim(-1.3, 1.7)
        ax.grid(alpha=0.25, lw=0.6)
    for ax in axes[-1]: ax.set_xlabel(r"transition location  $x_T/c$")
    for ax in axes[:, 0]: ax.set_ylabel(r"section lift coefficient  $c_l$")
    axes[0, 0].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.suptitle("NLF(1)-0416 transition location, M = 0.10  —  "
                 "86 points digitised from NASA TP-1861, Fig. 9", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.text(0.5, 0.012,
             "Bars show the +/-0.025c orifice-pitch bracket within which the "
             "experiment localises transition.  Nothing is calibrated on this set.",
             ha="center", fontsize=8.5, color=INK_SOFT, style="italic")
    import os as _os
    _os.makedirs(VP, exist_ok=True)
    fig.savefig(f"{VP}/val_aerofoil_nlf0416.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return f"{VP}/val_aerofoil_nlf0416.png"


def plot_case(key):
    v=C.VALIDATION[key]; ex=EXP[key]
    ue = ((ex["x_m"], ex["Ue"]) if ex.get("Ue") is not None else None)
    dec = ((ex["Re_x"], ex["Tu_local"]) if v.get("L_turb") is None
           and ex.get("Tu_local") is not None else None)
    r=solve_flat_plate(v["L"],v["U"],v["nu"],v["Tu_pct"],npts=900,
                       L_turb=v.get("L_turb"), Ue_dist=ue, Tu_decay=dec)
    fig,ax=new_fig(8.4,5.4)
    ax.loglog(r["Re_x"],r["Cf"],color=PALETTE[0],lw=2.4,label="UTSS solver")
    ax.loglog(r["Re_x"],r["Cf_lam_ref"],ls="--",color=PALETTE[2],lw=1.4,
              label="laminar (Blasius 0.664/√Re)")
    ax.loglog(r["Re_x"],r["Cf_turb_ref"],ls=":",color=PALETTE[3],lw=1.6,
              label="turbulent (0.0592/Re^0.2)")
    if ex["Cf"] is not None:
        ax.scatter(ex["Re_x"],ex["Cf"],s=55,color=PALETTE[1],zorder=5,
                   edgecolor=INK_SOFT,label="experiment (ERCOFTAC case 020)")
    else:
        ax.axvline(ex["Re_x_t"],color=PALETTE[1],ls="--",lw=1.6,
                   label="measured onset (literature value)")
    ax.axvline(v["U"]*r["x_tr"]/v["nu"],color=PALETTE[4],ls="-.",lw=1.3)
    ax.set_xlabel("Re_x"); ax.set_ylabel("skin-friction  C_f")
    ax.set_ylim(2e-4,8e-3)
    ax.set_title(f"Validation: {key}  (Tu={v['Tu_pct']}%)  —  "
                 f"Re_θt pred {r['Re_theta'][r['i_tr']]:.0f} vs exp {ex['Re_theta_t']:.0f}")
    ax.legend(loc="lower left",fontsize=10)
    finish(fig,f"{VP}/val_{key}.png",
           caption=f"Source: {v['source'][:95]}...")

def plot_combined(df_sum):
    fig,ax=new_fig(8.6,5.4)
    cases=["T3A","T3AM","T3B","T3C4","SS"]
    x=np.arange(len(cases)); w=0.36
    exp=[EXP[k]["Re_theta_t"] for k in cases]
    pred=[df_sum.iloc[i]["Re_theta_t_pred"] for i in range(len(cases))]
    ax.bar(x-w/2,exp,w,color=PALETTE[1],label="experiment Re_θt")
    ax.bar(x+w/2,pred,w,color=PALETTE[0],label="UTSS predicted Re_θt")
    for xi,(e,p) in enumerate(zip(exp,pred)):
        ax.text(xi-w/2,e+10,f"{e:.0f}",ha="center",fontsize=10,color=INK)
        ax.text(xi+w/2,p+10,f"{p:.0f}",ha="center",fontsize=10,color=INK)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(cases)
    ax.set_ylabel("transition-onset  Re_θt")
    ax.set_title("Universal validation: transition-onset Re_θt, one calibration set")
    ax.legend(fontsize=10)
    finish(fig,f"{VP}/val_combined_Re_theta_t.png")

if __name__=="__main__":
    df=run_validation()
    for k in ["T3A","T3AM","T3B","T3C4","SS"]: plot_case(k)
    plot_combined(df)
    df_sw=run_swept()
    print(df_sw.to_string(index=False))
    df_sw2=run_swept2()
    print(df_sw2.to_string(index=False))
    df_nlf=run_nlf0416()
    plot_nlf0416(df_nlf)
    print(df.to_string(index=False))
    print("validation files:",sorted([f for f in os.listdir(VAL) if f.endswith('.csv')]))
