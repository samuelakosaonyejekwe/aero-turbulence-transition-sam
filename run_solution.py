"""
run_solution.py
Execute the UTSS solver for the AETHER-NLF 25 case study and write every
engineering output as CSV (04_solution/).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0,"solver")
import case_config as C
from utss_solver import solve_airfoil, velocity_field, march_bl, CAL

SOL="04_solution"; os.makedirs(SOL,exist_ok=True)
W=C.WING; cr=C.CRUISE; cl=C.CLIMB

def run_case(cond, name):
    X,Y=C.nlf16_panel_points(130)
    r=solve_airfoil(X,Y,cond["alpha_deg"],cond["U_inf"],cond["nu_inf"],
                    W["MAC"],cond["Tu_pct"],sweep_deg=W["le_sweep_deg"],
                    mach=cond["mach"])
    for surf in ["upper","lower"]:
        s=r["surfaces"][surf]
        df=pd.DataFrame({
            "x_c":s["x"], "arc_s_m":s["s"], "Re_x":s["Re_x"],
            "Cp":s["Cp"], "Ue_ms":s["Ue"], "Ue_Uinf":s["Ue"]/cond["U_inf"],
            "theta_mm":s["theta"]*1e3, "H_shape":s["H"],
            "Cf":s["Cf"], "Re_theta":s["Re_theta"],
            "Re_theta_trans":s["Re_theta_t"], "intermittency_gamma":s["gamma"],
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
            reth=s["Re_theta"][it] if it else np.nan
            xch=s.get("x_tr_chord",np.nan)
            rows.append(dict(case=nm,surface=surf,s_tr_c=round(xtr,3),
                x_tr_c=round(float(xch),3) if xch==xch else None,
                Re_x_tr=f"{rex:.3e}", Re_theta_at_onset=round(float(reth),1) if it else None,
                mechanism=s["onset_mech"],
                laminar_run_pct=round(float(xch)*100,1) if xch==xch else None,
                Cf_te=round(float(s["Cf"][-1]),5)))
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
            xtr_upper_c=round(u["x_tr_chord"],3) if u["x_tr_chord"]==u["x_tr_chord"] else 1.0,
            xtr_lower_c=round(l["x_tr_chord"],3) if l["x_tr_chord"]==l["x_tr_chord"] else 1.0))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/aero_polar.csv",index=False)
    return df

def spanwise():
    X,Y=C.nlf16_panel_points(130)
    eta=np.linspace(0.0,0.98,12); rows=[]
    for e in eta:
        chord=W["root_chord"]+e*(W["tip_chord"]-W["root_chord"])
        Re=cr["U_inf"]*chord/cr["nu_inf"]
        twist=e*W["twist_tip_deg"]
        aeff=cr["alpha_deg"]+twist
        r=solve_airfoil(X,Y,aeff,cr["U_inf"],cr["nu_inf"],chord,cr["Tu_pct"],
                        sweep_deg=W["le_sweep_deg"],mach=cr["mach"])
        u=r["surfaces"]["upper"]; l=r["surfaces"]["lower"]
        rows.append(dict(eta=round(e,3), y_m=round(e*W["span_b"]/2,3),
            chord_m=round(chord,3), Re_local=round(Re,-2), alpha_eff_deg=round(aeff,2),
            xtr_upper_c=round(u["x_tr_chord"],3) if u["x_tr_chord"]==u["x_tr_chord"] else 1.0,
            xtr_lower_c=round(l["x_tr_chord"],3) if l["x_tr_chord"]==l["x_tr_chord"] else 1.0,
            Cd_section=round(r["Cd"],5),
            laminar_fraction=round(0.5*(u["x_tr_chord"]+l["x_tr_chord"]),3)))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/spanwise_distribution.csv",index=False)
    return df

def pressure_field(cond,name):
    X,Y=C.nlf16_panel_points(130)
    gx=np.linspace(-0.4,1.4,160); gy=np.linspace(-0.6,0.6,110)
    Xg,Yg=np.meshgrid(gx,gy)
    Vx,Vy,Cp=velocity_field(X,Y,cond["alpha_deg"],Xg,Yg,U=cond["U_inf"],
                            mach=cond["mach"])
    # mask interior of airfoil
    co=C.nlf16_coords(n=200)
    from matplotlib.path import Path
    poly=np.column_stack([np.concatenate([co["xu"],co["xl"][::-1]]),
                          np.concatenate([co["yu"],co["yl"][::-1]])])
    inside=Path(poly).contains_points(np.column_stack([Xg.ravel(),Yg.ravel()])).reshape(Xg.shape)
    Cp=np.where(inside,np.nan,Cp)
    spd=np.sqrt(Vx**2+Vy**2); spd=np.where(inside,np.nan,spd)
    df=pd.DataFrame({"x_c":Xg.ravel(),"y_c":Yg.ravel(),"Cp":Cp.ravel(),
                     "Vx_ms":Vx.ravel(),"Vy_ms":Vy.ravel(),"speed_ms":spd.ravel()})
    df.to_csv(f"{SOL}/field_pressure_{name}.csv",index=False)
    np.savez(f"{SOL}/field_pressure_{name}.npz",Xg=Xg,Yg=Yg,Cp=Cp,Vx=Vx,Vy=Vy,spd=spd)
    return df

def bl_profiles(rc):
    """Velocity + temperature profiles (Crocco-Busemann) at stations."""
    g=cr["gamma_air"]; r_rec=cr["recovery_r"]; a_inf=cr["a_sound"]
    s=rc["surfaces"]["upper"]
    rows=[]; stations={"x/c=0.10":0.10,"x/c=0.30":0.30,
                       "x/c=0.60":0.60,"x/c=0.95":0.95}
    eta=np.linspace(0,1,40)
    for lab,xq in stations.items():
        i=np.argmin(np.abs(s["x"]-xq))
        gam=s["gamma"][i]; H=s["H"][i]
        delta=s["theta"][i]* (8.0 if gam<0.5 else 7.0)*1.6   # est BL thickness
        Ue=s["Ue"][i]; Me=Ue/a_inf
        # blended velocity profile: laminar(sin) -> turbulent(1/7 power)
        u_lam=np.sin(np.pi/2*eta)
        u_turb=eta**(1/7.0)
        u_Ue=(1-gam)*u_lam+gam*u_turb
        T_Te=1+r_rec*(g-1)/2*Me**2*(1-u_Ue**2)
        Tinf=cr["T_inf_K"]; Te=Tinf*(1+(g-1)/2*cr["mach"]**2)/(1+(g-1)/2*Me**2)
        Tabs=T_Te*Te
        for et,uu,tt,Ta in zip(eta,u_Ue,T_Te,Tabs):
            rows.append(dict(station=lab,x_c=round(xq,2),y_delta=round(et,3),
                y_mm=round(et*delta*1e3,4),u_Ue=round(uu,4),
                T_Te=round(tt,4),T_K=round(Ta,2),
                Me_edge=round(Me,3),state=s["state"][i]))
    df=pd.DataFrame(rows); df.to_csv(f"{SOL}/bl_profiles_cruise.csv",index=False)
    return df

def nlf_vs_turbulent(rc):
    """Drag benefit: NLF (predicted transition) vs forced-fully-turbulent."""
    X,Y=C.nlf16_panel_points(130)
    # forced turbulent: tiny Re_theta_t -> trip at LE (A_BP huge)
    cal_trip=dict(CAL); cal_trip.update(A_TS=0.02,A_BP=0.02)  # trip near LE
    rt=solve_airfoil(X,Y,cr["alpha_deg"],cr["U_inf"],cr["nu_inf"],W["MAC"],
                     5.0,sweep_deg=W["le_sweep_deg"],cal=cal_trip)
    Cd_nlf=rc["Cd"]; Cd_turb=rt["Cd"]
    u=rc["surfaces"]["upper"]; l=rc["surfaces"]["lower"]
    lam=0.5*((u["x_tr"]+l["x_tr"])/W["MAC"])
    rows=[("NLF (UTSS predicted transition)",round(Cd_nlf*1e4,1),round(lam*100,1)),
          ("Fully turbulent (LE trip)",round(Cd_turb*1e4,1),0.0),
          ("Viscous drag reduction",round((Cd_turb-Cd_nlf)/Cd_turb*100,1),None)]
    df=pd.DataFrame(rows,columns=["configuration","Cd_counts","mean_laminar_pct"])
    df.to_csv(f"{SOL}/nlf_vs_turbulent.csv",index=False)
    return df,Cd_nlf,Cd_turb

def integrated_forces(rc):
    q=cr["q_inf"]; S=W["area_S"]
    rows=[("Section lift coefficient Cl",round(rc["Cl"],4),"-"),
          ("Section profile drag Cd",round(rc["Cd"],5),"-"),
          ("Section Cd (counts)",round(rc["Cd"]*1e4,1),"counts"),
          ("Section L/D",round(rc["Cl"]/rc["Cd"],1),"-"),
          ("Wing lift (3D est, e=0.85)",round(rc["Cl"]*0.9*q*S,0),"N"),
          ("Dynamic pressure q",round(q,1),"Pa"),
          ("Reynolds number Re_MAC",f"{cr['Re_MAC']:.3e}","-"),
          ("Mach number",cr["mach"],"-")]
    df=pd.DataFrame(rows,columns=["quantity","value","unit"])
    df.to_csv(f"{SOL}/integrated_forces.csv",index=False)
    return df

if __name__=="__main__":
    rc=run_case(cr,"cruise"); rl=run_case(cl,"climb")
    ts=transition_summary(rc,rl)
    pol=aero_polar(); spn=spanwise()
    pressure_field(cr,"cruise"); pressure_field(cl,"climb")
    bl_profiles(rc)
    nvt,cdn,cdt=nlf_vs_turbulent(rc)
    integrated_forces(rc)
    print("=== TRANSITION SUMMARY ==="); print(ts.to_string(index=False))
    print("\n=== NLF vs TURBULENT ==="); print(nvt.to_string(index=False))
    print(f"\nCruise: Cl={rc['Cl']:.3f} Cd={rc['Cd']*1e4:.1f}cts  "
          f"Drag saving={ (cdt-cdn)/cdt*100:.1f}%")
    print("solution files:", sorted([f for f in os.listdir(SOL) if f.endswith('.csv')]))
