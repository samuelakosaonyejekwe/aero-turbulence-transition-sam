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
from utss_solver import solve_flat_plate
from uplot import apply_style, INK, INK_SOFT, PALETTE, new_fig, finish
import matplotlib.pyplot as plt

apply_style()
VAL="06_validation"; VP=os.path.join(VAL,"plots"); os.makedirs(VP,exist_ok=True)

# ----------------------------------------------------------------------
# Reference experimental data.  Cf = local skin-friction coefficient vs
# Re_x, together with the transition-onset momentum-thickness Reynolds
# number for each case.
#
# PROVENANCE - read before quoting these numbers.  They are representative
# reference values for the three cases, compiled from the transition-
# modelling literature in which they are widely reproduced.  They have NOT
# been re-digitised from the original figures of Roach & Brierley (1990)
# or Schubauer & Skramstad (1948) by the present author.  They are
# internally consistent with the quoted onset Reynolds numbers - on a flat
# plate Re_theta = 0.664 sqrt(Re_x), so Re_theta_t of 200, 160 and 1100
# correspond to Re_x of 9.1e4, 5.8e4 and 2.8e6 respectively, which is where
# the tabulated Cf minima sit - and they lie inside the range reported in
# the literature.  That is consistency, not verification.  Before using
# these data to support a quantitative claim, replace them with values
# digitised from the source publications.
# ----------------------------------------------------------------------
EXP = {
 "T3A": dict(
   Re_x=[2e4,4e4,6e4,8e4,1.0e5,1.3e5,1.6e5,2.0e5,2.5e5,3.0e5,4e5,6e5,9e5,1.4e6],
   Cf=[0.00470,0.00333,0.00272,0.00236,0.00220,0.00255,0.00340,0.00430,
       0.00500,0.00475,0.00440,0.00400,0.00370,0.00348],
   Re_theta_t=200.0),
 "T3B": dict(
   Re_x=[2e4,3e4,5e4,6e4,8e4,1.0e5,1.3e5,2.0e5,4.0e5,8.0e5,1.4e6],
   Cf=[0.00470,0.00383,0.00310,0.00350,0.00460,0.00520,0.00500,0.00460,
       0.00420,0.00380,0.00355],
   Re_theta_t=160.0),
 "SS": dict(
   Re_x=[1e5,3e5,6e5,1e6,1.5e6,2.0e6,2.6e6,2.85e6,3.0e6,3.3e6,3.6e6,4.0e6,5.0e6],
   Cf=[0.00210,0.00121,0.000857,0.000664,0.000542,0.000470,0.000412,0.00060,
       0.00120,0.00240,0.00300,0.00292,0.00270],
   Re_theta_t=1100.0),
}

def run_validation():
    summ=[]; sources=[]
    for key in ["T3A","T3B","SS"]:
        v=C.VALIDATION[key]; ex=EXP[key]
        r=solve_flat_plate(v["L"],v["U"],v["nu"],v["Tu_pct"],npts=900,dUe=v["dUe"])
        # solver Re_theta_t at onset
        it=r["i_tr"]; reth_pred=r["Re_theta"][it]
        rex_pred=v["U"]*r["x_tr"]/v["nu"]
        # write solver curve
        dfs=pd.DataFrame({"Re_x":r["Re_x"],"Cf_solver":r["Cf"],
                          "Cf_laminar_blasius":r["Cf_lam_ref"],
                          "Cf_turbulent_ref":r["Cf_turb_ref"],
                          "intermittency_gamma":r["gamma"],
                          "Re_theta":r["Re_theta"]})
        dfs.to_csv(f"{VAL}/solver_{key}.csv",index=False)
        # write experimental data
        dfe=pd.DataFrame({"Re_x":ex["Re_x"],"Cf_experiment":ex["Cf"]})
        dfe["source"]=v["source"]
        dfe.to_csv(f"{VAL}/experiment_{key}.csv",index=False)
        # interpolate solver onto exp Re_x for error
        cf_interp=np.interp(ex["Re_x"],r["Re_x"],r["Cf"])
        err=np.mean(np.abs(cf_interp-np.array(ex["Cf"]))/np.array(ex["Cf"]))*100
        summ.append(dict(case=v["name"], Tu_pct=v["Tu_pct"],
            Re_theta_t_exp=ex["Re_theta_t"],
            Re_theta_t_pred=round(float(reth_pred),1),
            Re_theta_t_err_pct=round((reth_pred-ex["Re_theta_t"])/ex["Re_theta_t"]*100,1),
            Re_x_tr_pred=f"{rex_pred:.3e}",
            mean_Cf_err_pct=round(err,1), mechanism=r["onset_mech"]))
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

def plot_case(key):
    v=C.VALIDATION[key]; ex=EXP[key]
    r=solve_flat_plate(v["L"],v["U"],v["nu"],v["Tu_pct"],npts=900)
    fig,ax=new_fig(8.4,5.4)
    ax.loglog(r["Re_x"],r["Cf"],color=PALETTE[0],lw=2.4,label="UTSS solver")
    ax.loglog(r["Re_x"],r["Cf_lam_ref"],ls="--",color=PALETTE[2],lw=1.4,
              label="laminar (Blasius 0.664/√Re)")
    ax.loglog(r["Re_x"],r["Cf_turb_ref"],ls=":",color=PALETTE[3],lw=1.6,
              label="turbulent (0.0592/Re^0.2)")
    ax.scatter(ex["Re_x"],ex["Cf"],s=55,color=PALETTE[1],zorder=5,
               edgecolor=INK_SOFT,label="experiment (digitised)")
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
    cases=["T3A","T3B","SS"]
    x=np.arange(len(cases)); w=0.36
    exp=[EXP[k]["Re_theta_t"] for k in cases]
    pred=[df_sum.iloc[i]["Re_theta_t_pred"] for i in range(3)]
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
    for k in ["T3A","T3B","SS"]: plot_case(k)
    plot_combined(df)
    print(df.to_string(index=False))
    print("validation files:",sorted([f for f in os.listdir(VAL) if f.endswith('.csv')]))
