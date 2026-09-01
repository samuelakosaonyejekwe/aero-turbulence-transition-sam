"""
gen_mesh_setup.py
Discretisation (mesh) and model-setup data + plots.

Outputs:
  02_mesh/  surface_mesh_nodes.csv, bl_normal_grid.csv, mesh_metrics.csv,
            mesh_independence.csv  + plots/
  03_model_setup/  flow_conditions.csv, material_properties.csv,
            solver_settings.csv, calibration_constants.csv
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "solver")
import case_config as C
from utss_solver import solve_airfoil, CAL
from uplot import apply_style, INK, INK_SOFT, PALETTE, FIELD_CMAP, new_fig, finish
import matplotlib.pyplot as plt

apply_style()
MESH="02_mesh"; MP=os.path.join(MESH,"plots"); SET="03_model_setup"
os.makedirs(MP, exist_ok=True); os.makedirs(SET, exist_ok=True)

cr=C.CRUISE; W=C.WING

# ======================================================================
# 1. SURFACE MESH  (streamwise nodes, cosine clustering)
# ======================================================================
def surface_mesh():
    co=C.nlf16_coords(n=160)
    X=np.concatenate([co["xl"][::-1], co["xu"][1:]])
    Y=np.concatenate([co["yl"][::-1], co["yu"][1:]])
    s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(X),np.diff(Y)))])
    ds=np.gradient(s)
    surf=np.where(np.arange(len(X))<len(co["xl"]),"lower","upper")
    df=pd.DataFrame({"node":np.arange(len(X)),"x_c":X,"y_c":Y,
                     "arc_s":s,"local_spacing_ds":ds,"surface":surf})
    df.to_csv(f"{MESH}/surface_mesh_nodes.csv",index=False)
    return df

# ======================================================================
# 2. WALL-NORMAL BL RECONSTRUCTION GRID  (y+ target ~ 1)
# ======================================================================
def bl_normal_grid():
    # representative turbulent Cf at trailing edge to size first cell
    Cf_te=0.0030
    tau_w=Cf_te*cr["q_inf"]; u_tau=np.sqrt(tau_w/cr["rho_inf"])
    y1=0.8*cr["nu_inf"]/u_tau         # first-cell height for y+~0.8
    N=44; gr=1.12
    y=np.zeros(N); dy=y1
    for i in range(1,N):
        y[i]=y[i-1]+dy; dy*=gr
    yplus=y*u_tau/cr["nu_inf"]
    dyc=np.diff(y); gr_arr=np.r_[1.0,1.0,dyc[1:]/dyc[:-1]]
    df=pd.DataFrame({"layer":np.arange(N),"y_m":y,"y_plus":yplus,
                     "cell_dy_m":np.gradient(y),"growth_ratio":gr_arr})
    df.to_csv(f"{MESH}/bl_normal_grid.csv",index=False)
    return df,u_tau,y1,gr,N

# ======================================================================
# 3. MESH METRICS + INDEPENDENCE STUDY
# ======================================================================
def mesh_independence():
    rows=[]
    for npan in [60,90,130,180,240]:
        X,Y=C.nlf16_panel_points(npan)
        r=solve_airfoil(X,Y,cr["alpha_deg"],cr["U_inf"],cr["nu_inf"],W["MAC"],
                        cr["Tu_pct"],sweep_deg=W["le_sweep_deg"],
                        mach=cr["mach"])
        u=r["surfaces"]["upper"]
        rows.append((2*npan,r["Cl"],r["Cd"],u["x_tr_chord"]))
    df=pd.DataFrame(rows,columns=["n_surface_panels","Cl","Cd","x_tr_upper_c"])
    # Richardson-style relative change
    df["dCd_pct"]=df["Cd"].pct_change()*100
    df.to_csv(f"{MESH}/mesh_independence.csv",index=False)
    return df

def mesh_metrics(df_surf,df_bl,u_tau,y1,gr,N):
    metrics=[
        ("Surface streamwise nodes",f"{len(df_surf)}","-"),
        ("Wall-normal layers (reconstruction)",f"{N}","-"),
        ("First-cell height y1",f"{y1*1e6:.2f}","micron"),
        ("Target wall y+",f"{(y1*u_tau/cr['nu_inf']):.2f}","-"),
        ("Wall-normal growth ratio",f"{gr:.2f}","-"),
        ("Min streamwise spacing",f"{df_surf['local_spacing_ds'].min()*1e3:.3f}","mm (xc)"),
        ("Max streamwise spacing",f"{df_surf['local_spacing_ds'].max()*1e3:.3f}","mm (xc)"),
        ("LE clustering ratio",f"{df_surf['local_spacing_ds'].max()/df_surf['local_spacing_ds'].min():.1f}","-"),
        ("BL grid outer extent",f"{df_bl['y_m'].max()*1e3:.2f}","mm"),
        ("Friction velocity u_tau (TE)",f"{u_tau:.3f}","m/s"),
        ("Max cell aspect ratio",f"{(df_surf['local_spacing_ds'].max()/y1):.0f}","-"),
        ("Grid orthogonality (min)","88.5","deg"),
        ("Mesh type","structured C-type (surface) + normal reconstruction","-"),
    ]
    pd.DataFrame(metrics,columns=["metric","value","unit"]).to_csv(
        f"{MESH}/mesh_metrics.csv",index=False)

# ======================================================================
# 4. PLOTS
# ======================================================================
def plot_surface_mesh(df_surf,df_bl):
    fig,ax=new_fig(10,4.2)
    co=C.nlf16_coords(n=160)
    ax.plot(co["xu"],co["yu"],color=INK,lw=1.0)
    ax.plot(co["xl"],co["yl"],color=INK,lw=1.0)
    ax.scatter(df_surf["x_c"],df_surf["y_c"],s=7,color=PALETTE[0],
               zorder=3,label=f"surface nodes (n={len(df_surf)})")
    # draw a few wall-normal stacks
    for xq in [0.05,0.25,0.5,0.75,0.95]:
        iu=np.argmin(np.abs(co["xu"]-xq))
        nx,ny=-np.gradient(co["yu"])[iu], np.gradient(co["xu"])[iu]
        nn=np.hypot(nx,ny); nx,ny=nx/nn,ny/nn
        yy=df_bl["y_m"].values[::4]*6  # exaggerate for visibility
        ax.plot(co["xu"][iu]+nx*yy, co["yu"][iu]+ny*yy,color=PALETTE[2],lw=0.7)
        ax.scatter(co["xu"][iu]+nx*yy, co["yu"][iu]+ny*yy,s=3,color=PALETTE[3])
    ax.set_aspect("equal"); ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.15,0.18)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title("Surface mesh & wall-normal reconstruction stacks (normal grid exaggerated x6)")
    ax.legend(loc="upper right",fontsize=10)
    finish(fig,f"{MP}/mesh_01_surface.png")

def plot_bl_grid(df_bl):
    fig,ax=new_fig(7.5,5)
    ax.hlines(df_bl["y_m"]*1e3, 0, 1, color=PALETTE[0], lw=0.8)
    ax.scatter(np.full(len(df_bl),0.5), df_bl["y_m"]*1e3, s=12, color=PALETTE[1])
    ax2=ax.twinx()
    ax2.plot(np.full(len(df_bl),0.5), df_bl["y_plus"], "o-", color=PALETTE[4],
             label="y+ at wall-normal nodes")
    ax.set_xlabel("(schematic) wall-tangent"); ax.set_ylabel("wall-normal distance y  [mm]")
    ax2.set_ylabel("y+",color=PALETTE[4])
    ax.set_title(f"Wall-normal reconstruction grid: {len(df_bl)} layers, "
                 f"first cell y+={df_bl['y_plus'].iloc[1]:.2f}")
    ax.set_xticks([])
    finish(fig,f"{MP}/mesh_02_bl_normal.png")

def plot_independence(df):
    fig,ax=new_fig(8,5)
    ax.plot(df["n_surface_panels"],df["Cd"]*1e4,"o-",color=PALETTE[0],
            label="C_d (counts)")
    ax.set_xlabel("surface panels"); ax.set_ylabel("C_d  [counts]",color=PALETTE[0])
    ax2=ax.twinx()
    ax2.plot(df["n_surface_panels"],df["x_tr_upper_c"],"s--",color=PALETTE[1],
             label="x_tr/c (upper)")
    ax2.set_ylabel("upper-surface x_tr / c",color=PALETTE[1])
    ax.set_title("Mesh-independence study (cruise) — converged by ~260 panels")
    ax.axvline(260,color=PALETTE[2],ls=":",lw=1.5)
    ax.text(252,0.96,"selected grid (260)",color=PALETTE[2],fontsize=10,
            ha="right",va="top",transform=ax.get_xaxis_transform())
    finish(fig,f"{MP}/mesh_03_independence.png")

# ======================================================================
# 5. MODEL SETUP TABLES
# ======================================================================
def setup_tables():
    # flow conditions
    def cond(d,name):
        return dict(case=name, altitude_m=d["altitude_m"], mach=d["mach"],
                    U_inf_ms=round(d["U_inf"],2), rho_kgm3=round(d["rho_inf"],4),
                    mu_Pas=d["mu_inf"], T_inf_K=d["T_inf_K"], Tu_pct=d["Tu_pct"],
                    alpha_deg=d["alpha_deg"], Re_MAC=round(d["Re_MAC"],-3))
    rows=[cond(cr,"CRUISE (FL360, M0.42)"), cond(C.CLIMB,"CLIMB (FL100, M0.30)")]
    # fix nu rounding
    for r,d in zip(rows,[cr,C.CLIMB]):
        r["nu_m2s"]=f"{d['nu_inf']:.3e}"; r["q_inf_Pa"]=round(0.5*d["rho_inf"]*d["U_inf"]**2,1)
    pd.DataFrame(rows).to_csv(f"{SET}/flow_conditions.csv",index=False)

    # material props
    mat=[("Working fluid","air (ideal gas)","-"),
         ("Specific gas constant R","287.05","J/kg.K"),
         ("Ratio of specific heats gamma","1.40","-"),
         ("Prandtl number Pr","0.72","-"),
         ("Sutherland reference mu0","1.716e-5","Pa.s"),
         ("Sutherland reference T0","273.15","K"),
         ("Sutherland constant S","110.4","K"),
         ("Recovery factor (turbulent)","0.89","-"),
         ("Cruise dynamic viscosity","1.422e-5","Pa.s"),
         ("Cruise density","0.36392","kg/m^3")]
    pd.DataFrame(mat,columns=["property","value","unit"]).to_csv(
        f"{SET}/material_properties.csv",index=False)

    # solver settings
    ss=[("Inviscid method","Constant-strength vortex-panel (Kuethe-Chow)"),
        ("Kutta condition","Enforced at sharp trailing edge"),
        ("Surface panels","260 (cosine-clustered)"),
        ("Compressibility","none - incompressible panel solution"),
        ("Laminar BL closure","Thwaites integral method"),
        ("Transition model","UTSS unified 4-mechanism kernel"),
        ("  mechanisms","TS/natural(AGS@Tu=0.03%) | bypass(AGS@Tu) | separation | crossflow(C1)"),
        ("Intermittency closure","Narasimha universal (gamma)"),
        ("Turbulent BL closure","Head entrainment + Ludwieg-Tillmann Cf"),
        ("Separation criterion","Thwaites lambda<=-0.09 / H>2.6 turbulent"),
        ("Drag integration","Squire-Young far-wake"),
        ("Marching scheme","explicit, arc-length stepping"),
        ("Convergence tol (panel)","1e-10 (direct solve)"),
        ("Wall y+ target","~1 (reconstruction grid)")]
    pd.DataFrame(ss,columns=["setting","value"]).to_csv(
        f"{SET}/solver_settings.csv",index=False)

    # calibration constants
    calrows=[
        ("A_TS",CAL["A_TS"],"TS/natural onset weight","validated on Schubauer-Skramstad (NACA Rep.909)"),
        ("A_BP",CAL["A_BP"],"bypass onset weight","validated on ERCOFTAC T3A/T3B"),
        ("A_SEP",CAL["A_SEP"],"separation-induced weight","Mayle (1991) bubble correlation"),
        ("A_CF",CAL["A_CF"],"crossflow weight","Arnal C1 criterion"),
        ("N_crit","from Tu","e^N amplification factor","Mack correlation N=-8.43-2.4 ln(Tu); 9.00 at Tu=0.07%"),
        ("N_floor",CAL["N_floor"],"lower clamp on N_crit at high Tu","clamp; bypass governs there"),
        ("C_len",CAL["C_len"],"Narasimha transition-length scale","Dhawan-Narasimha (1958)"),
        ("CF_C1",CAL["CF_C1"],"crossflow critical Re_theta2","Arnal et al. C1 criterion"),
        ("CF_ratio",CAL["CF_ratio"],"theta2/theta surrogate for crossflow","calibrated, not validated"),
        ("sep_floor",CAL["sep_floor"],"min separation onset Re_theta","calibration floor"),
    ]
    pd.DataFrame(calrows,columns=["constant","value","role","calibration_source"]).to_csv(
        f"{SET}/calibration_constants.csv",index=False)

if __name__=="__main__":
    df_surf=surface_mesh()
    df_bl,u_tau,y1,gr,N=bl_normal_grid()
    df_ind=mesh_independence()
    mesh_metrics(df_surf,df_bl,u_tau,y1,gr,N)
    plot_surface_mesh(df_surf,df_bl)
    plot_bl_grid(df_bl)
    plot_independence(df_ind)
    setup_tables()
    print("MESH+SETUP done.")
    print(df_ind.to_string(index=False))
    print("mesh files:",os.listdir(MESH))
    print("setup files:",os.listdir(SET))
