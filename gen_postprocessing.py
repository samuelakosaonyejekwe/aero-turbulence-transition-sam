"""
gen_postprocessing.py
Post-processing: clean plots of EVERY solution CSV, pressure contours,
temperature profiles, and 3D surface contours + vectors.
No black anywhere; text never overlaps the data.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0,"solver")
import case_config as C
from utss_solver import solve_airfoil
from uplot import (apply_style, INK, INK_SOFT, PALETTE, FIELD_CMAP, CF_CMAP,
                   GAMMA_CMAP, new_fig, finish)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

def _chord_interp(xs, vals, xq):
    """Interpolate a surface quantity onto a chordwise station list.

    The march is ordered by arc length from the stagnation point, and near a
    rounded nose that ordering is not monotone in x/c - the first few control
    points step back and forth by a thousandth of a chord.  np.interp requires
    an increasing abscissa and gives no warning when it does not get one, so
    the sequence is sorted and de-duplicated first.
    """
    xs = np.asarray(xs, float); vals = np.asarray(vals, float)
    k = np.argsort(xs, kind="stable")
    xs = xs[k]; vals = vals[k]
    keep = np.concatenate([[True], np.diff(xs) > 1e-12])
    return np.interp(xq, xs[keep], vals[keep])


def _tidy3d(ax):
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=9, pad=1.0)
    for a,p in [(ax.xaxis,8),(ax.yaxis,8),(ax.zaxis,6)]:
        a.labelpad=p

apply_style()
SOL="04_solution"; PP="05_postprocessing"
CSVP=f"{PP}/csv_plots"; CONT=f"{PP}/contours"; PROF=f"{PP}/profiles"; TD=f"{PP}/three_d"
for d in [CSVP,CONT,PROF,TD]: os.makedirs(d,exist_ok=True)
W=C.WING; cr=C.CRUISE; cl=C.CLIMB

LAM_BAND="#d7e3ef"      # single soft tint for the laminar run

def _onset(df):
    it=df.index[df["intermittency_gamma"]>1e-6]
    return float(df["x_c"][it[0]]) if len(it) else None

def _state_bands(ax, up, lo, cu=PALETTE[0], cl=PALETTE[6]):
    """Clean regime cue: shade only the laminar run (one soft tint) and mark
    each surface's transition onset with a thin dashed line in its own colour."""
    ou=_onset(up); ol=_onset(lo)
    first=min([v for v in [ou,ol] if v is not None], default=None)
    if first is not None:
        ax.axvspan(0, first, color=LAM_BAND, alpha=0.7, zorder=0)
    if ou is not None: ax.axvline(ou, color=cu, ls=(0,(5,3)), lw=1.2, alpha=0.8)
    if ol is not None: ax.axvline(ol, color=cl, ls=(0,(5,3)), lw=1.2, alpha=0.8)

def _cf_mask(df):
    """Drop near-stagnation points (Ue->0) where integral-method Cf is singular
    (non-physical Cf>8e-3 only occurs at the stagnation singularity)."""
    return df[(df["x_c"]>0.006) & (df["Ue_Uinf"]>0.12) & (df["Cf"]<0.008)]

def _bl_mask(df):
    """Drop LE/TE rear-stagnation points where theta and H are singular."""
    return df[(df["Ue_Uinf"]>0.12) & (df["x_c"]>0.004) & (df["x_c"]<0.995)]

def _cf_ylim(*dfs):
    """Robust Cf y-limit ignoring the singular stagnation region."""
    m=0.0
    for df in dfs:
        sub=_cf_mask(df)
        if len(sub): m=max(m,sub["Cf"].max())
    return m*1.18*1e3

# ======================================================================
# 1.  GEOMETRY / MESH CSV PLOTS
# ======================================================================
def plot_geometry_csvs():
    af=pd.read_csv("01_geometry/airfoil_UTSS-NLF16.csv")
    fig,ax=new_fig(9,3.6)
    ax.plot(af["x_upper"],af["y_upper"],color=PALETTE[0],lw=2,label="upper")
    ax.plot(af["x_lower"],af["y_lower"],color=PALETTE[2],lw=2,label="lower")
    ax.plot(af["x_c"],af["y_camber"],color=PALETTE[4],lw=1,ls="--",label="camber")
    ax.set_aspect("equal"); ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title("UTSS-NLF16 section geometry (from CSV)"); ax.legend(fontsize=10)
    finish(fig,f"{CSVP}/geo_airfoil.png")

    pl=pd.read_csv("01_geometry/wing_planform.csv")
    fig,ax=new_fig(8,5)
    ax.plot(pl["eta"],pl["chord_m"],"o-",color=PALETTE[0],label="chord [m]")
    ax.set_xlabel("span fraction η"); ax.set_ylabel("chord [m]",color=PALETTE[0])
    ax2=ax.twinx()
    ax2.plot(pl["eta"],pl["twist_deg"],"s--",color=PALETTE[1],label="twist [deg]")
    ax2.plot(pl["eta"],pl["z_dihedral_m"],"^:",color=PALETTE[2],label="dihedral z [m]")
    ax2.set_ylabel("twist [deg] / z [m]")
    ax.set_title("Wing span-wise geometry distribution")
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,la1+la2,fontsize=10,loc="center right")
    finish(fig,f"{CSVP}/geo_planform.png")

def plot_mesh_csvs():
    sm=pd.read_csv("02_mesh/surface_mesh_nodes.csv")
    fig,ax=new_fig(8.5,4.4)
    ax.plot(sm["node"],sm["local_spacing_ds"]*1e3,color=PALETTE[0],lw=1.6)
    ax.fill_between(sm["node"],sm["local_spacing_ds"]*1e3,color=PALETTE[0],alpha=0.12)
    ax.set_xlabel("surface node index"); ax.set_ylabel("local spacing Δs  [×10⁻³ c]")
    ax.set_title("Surface-mesh streamwise spacing (cosine clustering at LE/TE)")
    finish(fig,f"{CSVP}/mesh_spacing.png")

    bl=pd.read_csv("02_mesh/bl_normal_grid.csv")
    fig,ax=new_fig(8,5)
    ax.semilogy(bl["layer"],bl["y_plus"],"o-",color=PALETTE[1])
    ax.axhline(1.0,color=PALETTE[2],ls="--",lw=1.2,label="y⁺ = 1 target")
    ax.set_xlabel("wall-normal layer"); ax.set_ylabel("y⁺")
    ax.set_title("Wall-normal grid resolution (first cell y⁺≈0.8)")
    ax.legend(fontsize=10); finish(fig,f"{CSVP}/mesh_yplus.png")

    mi=pd.read_csv("02_mesh/mesh_independence.csv")
    fig,ax=new_fig(8,5)
    ax.plot(mi["n_surface_panels"],mi["Cd"]*1e4,"o-",color=PALETTE[0])
    ax.set_xlabel("surface panels"); ax.set_ylabel("C_d [counts]",color=PALETTE[0])
    ax2=ax.twinx(); ax2.plot(mi["n_surface_panels"],mi["x_tr_upper_c"],"s--",color=PALETTE[1])
    ax2.set_ylabel("x_tr/c upper",color=PALETTE[1])
    ax.axvline(260,color=PALETTE[2],ls=":"); ax.set_title("Mesh-independence convergence")
    finish(fig,f"{CSVP}/mesh_independence.png")

# ======================================================================
# 2.  SURFACE SOLUTION CSV PLOTS (cruise + climb)
# ======================================================================
def plot_surface(case):
    up=pd.read_csv(f"{SOL}/surface_{case}_upper.csv")
    lo=pd.read_csv(f"{SOL}/surface_{case}_lower.csv")
    cond=cr if case=="cruise" else cl
    # --- Cp distribution ---
    fig,ax=new_fig(8.6,5.2)
    ax.plot(up["x_c"],up["Cp"],color=PALETTE[0],lw=2.2,label="upper")
    ax.plot(lo["x_c"],lo["Cp"],color=PALETTE[2],lw=2.2,label="lower")
    for d,c in [(up,PALETTE[0]),(lo,PALETTE[2])]:
        it=d.index[d["intermittency_gamma"]>1e-6]
        if len(it): ax.axvline(d["x_c"][it[0]],color=c,ls="-.",lw=1.0,alpha=0.7)
    ax.invert_yaxis(); ax.set_xlabel("x/c"); ax.set_ylabel("C_p")
    ax.set_title(f"Pressure coefficient — {case} (dash-dot = transition onset)")
    ax.legend(fontsize=10); finish(fig,f"{CSVP}/{case}_Cp.png")

    # --- Cf with clean laminar-run cue (mask singular stagnation point) ---
    fig,ax=new_fig(8.6,5.2)
    _state_bands(ax,up,lo)
    mu=_cf_mask(up); ml=_cf_mask(lo)
    ax.plot(mu["x_c"],mu["Cf"]*1e3,color=PALETTE[0],lw=2.4,label="upper C_f")
    ax.plot(ml["x_c"],ml["Cf"]*1e3,color=PALETTE[6],lw=2.4,label="lower C_f")
    ax.set_ylim(0,_cf_ylim(up,lo)); ax.set_xlim(-0.02,1.02)
    ax.set_xlabel("x/c"); ax.set_ylabel("C_f  ×10³")
    ax.set_title(f"Skin-friction & boundary-layer state — {case}")
    handles=[Line2D([],[],color=PALETTE[0],lw=2.4,label="upper C_f"),
             Line2D([],[],color=PALETTE[6],lw=2.4,label="lower C_f"),
             Line2D([],[],color=LAM_BAND,lw=9,label="laminar run"),
             Line2D([],[],color=INK_SOFT,lw=1.2,ls=(0,(5,3)),label="transition onset")]
    ax.legend(handles=handles,fontsize=10,loc="upper right",framealpha=0.95)
    finish(fig,f"{CSVP}/{case}_Cf.png")

    # --- theta & H (mask LE/TE stagnation singularities) ---
    mbu=_bl_mask(up); mbl=_bl_mask(lo)
    fig,ax=new_fig(8.6,5.2)
    ax.plot(mbu["x_c"],mbu["theta_mm"],color=PALETTE[0],lw=2.2,label="θ upper")
    ax.plot(mbl["x_c"],mbl["theta_mm"],color=PALETTE[2],lw=2.2,label="θ lower")
    ax.set_xlabel("x/c"); ax.set_ylabel("momentum thickness θ [mm]")
    ax.set_ylim(0, max(mbu["theta_mm"].max(),mbl["theta_mm"].max())*1.25)
    ax2=ax.twinx()
    ax2.plot(mbu["x_c"],mbu["H_shape"],ls="--",color=PALETTE[3],lw=1.8,label="H upper")
    ax2.plot(mbl["x_c"],mbl["H_shape"],ls=":",color=PALETTE[1],lw=2.0,label="H lower")
    ax2.set_ylabel("shape factor H"); ax2.set_ylim(1.3,3.0)
    ax2.axhline(2.6,color=PALETTE[5],lw=0.9,alpha=0.6)
    ax2.text(0.55,2.64,"H ≈ 2.6 (incipient separation)",fontsize=10,
             color=PALETTE[5],ha="center")
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,la1+la2,fontsize=9,loc="upper center",ncol=4,
              framealpha=0.92,columnspacing=1.2,handlelength=1.6)
    ax.set_title(f"Momentum thickness & shape factor — {case}")
    finish(fig,f"{CSVP}/{case}_theta_H.png")

    # --- Re_theta vs Re_theta_t (transition criterion) ---
    fig,ax=new_fig(8.6,5.2)
    ax.plot(up["x_c"],up["Re_theta"],color=PALETTE[0],lw=2.2,label="Re_θ (upper)")
    ax.plot(up["x_c"],up["Re_theta_trans"],color=PALETTE[3],lw=1.8,ls="--",
            label="Re_θt onset criterion")
    it=up.index[up["intermittency_gamma"]>1e-6]
    if len(it):
        xt=up["x_c"][it[0]]; ax.scatter([xt],[up["Re_theta"][it[0]]],s=80,
            color=PALETTE[1],zorder=5,label="transition onset")
    ax.set_xlabel("x/c"); ax.set_ylabel("Re_θ")
    ax.set_title(f"Transition criterion: Re_θ crosses Re_θt — {case} upper")
    ax.legend(fontsize=10); finish(fig,f"{CSVP}/{case}_Retheta.png")

    # --- intermittency ---
    fig,ax=new_fig(8.6,4.6)
    ax.plot(up["x_c"],up["intermittency_gamma"],color=PALETTE[0],lw=2.2,label="upper γ")
    ax.plot(lo["x_c"],lo["intermittency_gamma"],color=PALETTE[2],lw=2.2,label="lower γ")
    ax.fill_between(up["x_c"],up["intermittency_gamma"],color=PALETTE[0],alpha=0.10)
    ax.set_xlabel("x/c"); ax.set_ylabel("intermittency γ")
    ax.set_ylim(-0.03,1.05)
    ax.set_title(f"Intermittency distribution (0=laminar, 1=turbulent) — {case}")
    ax.legend(fontsize=10); finish(fig,f"{CSVP}/{case}_gamma.png")

def plot_cruise_climb_compare():
    cu=pd.read_csv(f"{SOL}/surface_cruise_upper.csv")
    cm=pd.read_csv(f"{SOL}/surface_climb_upper.csv")
    fig,ax=new_fig(8.6,5)
    mcu=_cf_mask(cu); mcm=_cf_mask(cm)
    ax.plot(mcu["x_c"],mcu["Cf"]*1e3,color=PALETTE[0],lw=2.4,
            label="cruise (Tu=0.07%, natural)")
    ax.plot(mcm["x_c"],mcm["Cf"]*1e3,color=PALETTE[1],lw=2.4,
            label="climb (Tu=0.9%, bypass)")
    for d,c in [(cu,PALETTE[0]),(cm,PALETTE[1])]:
        it=d.index[d["intermittency_gamma"]>1e-6]
        if len(it): ax.axvline(d["x_c"][it[0]],color=c,ls="-.",lw=1.1)
    ax.set_ylim(0,_cf_ylim(cu,cm))
    ax.set_xlabel("x/c"); ax.set_ylabel("C_f ×10³")
    ax.set_title("Upper-surface C_f: cruise vs climb — regime-dependent transition")
    ax.legend(fontsize=10); finish(fig,f"{CSVP}/compare_cruise_climb_Cf.png")

# ======================================================================
# 3.  POLAR / SPANWISE / DRAG CSV PLOTS
# ======================================================================
def plot_polar():
    p=pd.read_csv(f"{SOL}/aero_polar.csv")
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    axs[0,0].plot(p["alpha_deg"],p["Cl"],"o-",color=PALETTE[0]); axs[0,0].set_title("Lift curve C_l–α")
    axs[0,0].set_xlabel("α [deg]"); axs[0,0].set_ylabel("C_l")
    axs[0,1].plot(p["Cd"]*1e4,p["Cl"],"o-",color=PALETTE[1]); axs[0,1].set_title("Drag polar")
    axs[0,1].set_xlabel("C_d [counts]"); axs[0,1].set_ylabel("C_l")
    axs[1,0].plot(p["alpha_deg"],p["L_over_D"],"o-",color=PALETTE[2]); axs[1,0].set_title("L/D vs α")
    axs[1,0].set_xlabel("α [deg]"); axs[1,0].set_ylabel("L/D")
    axs[1,1].plot(p["alpha_deg"],p["xtr_upper_c"],"o-",color=PALETTE[0],label="upper")
    axs[1,1].plot(p["alpha_deg"],p["xtr_lower_c"],"s-",color=PALETTE[3],label="lower")
    axs[1,1].set_title("Transition location vs α"); axs[1,1].set_xlabel("α [deg]")
    axs[1,1].set_ylabel("x_tr/c"); axs[1,1].legend(fontsize=10)
    for a in axs.ravel(): a.grid(alpha=0.3)
    fig.suptitle("Aerodynamic polars (UTSS, cruise Re) — AETHER-NLF 25 section",
                 color=INK,fontweight="normal")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{CSVP}/aero_polar.png",
        dpi=170,facecolor="white"); plt.close(fig)

def plot_spanwise():
    s=pd.read_csv(f"{SOL}/spanwise_distribution.csv")
    fig,ax=new_fig(8.8,5.2)
    ax.plot(s["eta"],s["xtr_upper_c"],"o-",color=PALETTE[0],label="upper x_tr/c")
    ax.plot(s["eta"],s["xtr_lower_c"],"s-",color=PALETTE[2],label="lower x_tr/c")
    ax.fill_between(s["eta"],s["xtr_upper_c"],alpha=0.10,color=PALETTE[0])
    ax.set_xlabel("span fraction η"); ax.set_ylabel("transition x_tr/c")
    ax.set_title("Span-wise transition front (laminar-flow extent)")
    ax2=ax.twinx()
    ax2.plot(s["eta"],s["Cd_section"]*1e4,"^--",color=PALETTE[1],label="C_d section")
    ax2.set_ylabel("section C_d [counts]",color=PALETTE[1])
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,la1+la2,fontsize=10,loc="center left")
    finish(fig,f"{CSVP}/spanwise_transition.png")

def plot_nlf_vs_turb():
    d=pd.read_csv(f"{SOL}/nlf_vs_turbulent.csv")
    fig,ax=new_fig(7.6,5)
    cd=d["Cd_counts"][:2]
    labels=["NLF (predicted\ntransition)","Fully turbulent\n(LE trip)"]
    xpos=[0,1]
    bars=ax.bar(xpos,cd,color=[PALETTE[2],PALETTE[1]],width=0.55)
    ax.set_xticks(xpos)
    for b,v in zip(bars,cd):
        ax.text(b.get_x()+b.get_width()/2,v+0.5,f"{v:.1f} cts",ha="center",
                color=INK,fontsize=10,fontweight="normal")
    sav=d["Cd_counts"].iloc[2]
    ax.set_ylabel("profile drag C_d [counts]")
    ax.set_title(f"NLF benefit: UTSS-predicted transition vs fully turbulent\n"
                 f"viscous drag reduction = {sav:.1f}%")
    ax.set_xticklabels(labels)
    finish(fig,f"{CSVP}/nlf_vs_turbulent.png")

# ======================================================================
# 4.  PRESSURE / VELOCITY CONTOURS  + VECTORS
# ======================================================================
def _solution_field(case):
    """The off-body field written by run_solution.py, read back as it stands.

    An earlier version of this module built its own field here by weighting the
    surface pressure with exp(-d/L) away from the wall.  That is a picture of a
    field, not a solution: it is not the potential flow, it does not satisfy
    continuity, and it disagreed with 04_solution/field_pressure_*.npz - which
    the same project computes from the exact constant-strength vortex-panel
    induced velocity - by more than the quantity being plotted.  The stored
    field is used instead, so the contours in the report are the solution the
    report tabulates.
    """
    d=np.load(f"{SOL}/field_pressure_{case}.npz")
    return (d["Xg"],d["Yg"],d["Cp"],d["Vx"],d["Vy"],d["spd"])

def plot_contours(case):
    cond=cr if case=="cruise" else cl
    Xg,Yg,Cp,Vx,Vy,spd=_solution_field(case)
    co=C.nlf16_coords(n=200)
    polyx=np.concatenate([co["xu"],co["xl"][::-1]]); polyy=np.concatenate([co["yu"],co["yl"][::-1]])
    # Cp contour
    fig,ax=new_fig(10,5.6)
    lv=np.linspace(np.nanpercentile(Cp,2),np.nanpercentile(Cp,99),28)
    cf=ax.contourf(Xg,Yg,Cp,levels=lv,cmap=FIELD_CMAP,extend="both")
    ax.contour(Xg,Yg,Cp,levels=lv[::4],colors=[INK_SOFT],linewidths=0.4,alpha=0.6)
    ax.fill(polyx,polyy,color="white",ec=INK,lw=1.5,zorder=4)
    cb=fig.colorbar(cf,ax=ax,shrink=0.85,pad=0.02); cb.set_label("C_p")
    ax.set_aspect("equal"); ax.set_xlim(-0.35,1.4); ax.set_ylim(-0.55,0.55)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title(f"Pressure-coefficient field — {case} (α={cond['alpha_deg']}°, M={cond['mach']})")
    finish(fig,f"{CONT}/contour_Cp_{case}.png")

    # velocity magnitude + streamlines
    fig,ax=new_fig(10,5.6)
    lv=np.linspace(np.nanpercentile(spd,2),np.nanpercentile(spd,99),26)
    cf=ax.contourf(Xg,Yg,spd,levels=lv,cmap=CF_CMAP,extend="both")
    try:
        ax.streamplot(Xg,Yg,np.nan_to_num(Vx),np.nan_to_num(Vy),density=1.1,
                      color=INK_SOFT,linewidth=0.6,arrowsize=0.7)
    except Exception: pass
    ax.fill(polyx,polyy,color="white",ec=INK,lw=1.5,zorder=4)
    cb=fig.colorbar(cf,ax=ax,shrink=0.85,pad=0.02); cb.set_label("speed [m/s]")
    ax.set_aspect("equal"); ax.set_xlim(-0.35,1.4); ax.set_ylim(-0.55,0.55)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title(f"Velocity magnitude & streamlines — {case}")
    finish(fig,f"{CONT}/contour_speed_{case}.png")

    # vector field (quiver) near surface
    fig,ax=new_fig(10,5.6)
    sk=(slice(None,None,5),slice(None,None,4))
    q=ax.quiver(Xg[sk],Yg[sk],Vx[sk],Vy[sk],spd[sk],cmap=FIELD_CMAP,
                scale=2600,width=0.0026)
    ax.fill(polyx,polyy,color="white",ec=INK,lw=1.5,zorder=4)
    cb=fig.colorbar(q,ax=ax,shrink=0.85,pad=0.02); cb.set_label("speed [m/s]")
    ax.set_aspect("equal"); ax.set_xlim(-0.35,1.4); ax.set_ylim(-0.55,0.55)
    ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
    ax.set_title(f"Velocity vector field — {case}")
    finish(fig,f"{CONT}/vectors_{case}.png")

# ======================================================================
# 5.  BOUNDARY-LAYER VELOCITY + TEMPERATURE PROFILES
# ======================================================================
def plot_profiles():
    df=pd.read_csv(f"{SOL}/bl_profiles_cruise.csv")
    stations=df["station"].unique()
    # velocity profiles
    fig,ax=new_fig(7.6,5.6)
    for i,st in enumerate(stations):
        s=df[df["station"]==st]
        ax.plot(s["u_Ue"],s["y_delta"],"-",color=PALETTE[i],lw=2,
                label=f"{st} ({s['state'].iloc[0]})")
    ax.set_xlabel("u / U_e"); ax.set_ylabel("y / δ")
    ax.set_title("Boundary-layer velocity profiles (cruise, upper surface)")
    ax.legend(fontsize=10); finish(fig,f"{PROF}/bl_velocity_profiles.png")

    # temperature profiles
    fig,ax=new_fig(7.6,5.6)
    for i,st in enumerate(stations):
        s=df[df["station"]==st]
        ax.plot(s["T_K"],s["y_delta"],"-",color=PALETTE[i],lw=2,label=st)
    ax.set_xlabel("temperature T [K]"); ax.set_ylabel("y / δ")
    ax.set_title("Boundary-layer temperature profiles (Crocco–Busemann, cruise)")
    ax.legend(fontsize=10); finish(fig,f"{PROF}/bl_temperature_profiles.png")

    # combined T/Te
    fig,ax=new_fig(7.6,5.6)
    for i,st in enumerate(stations):
        s=df[df["station"]==st]
        ax.plot(s["T_Te"],s["y_delta"],"-",color=PALETTE[i],lw=2,label=st)
    ax.set_xlabel("T / T_e"); ax.set_ylabel("y / δ")
    ax.set_title("Normalised temperature profiles — recovery near wall")
    ax.legend(fontsize=10); finish(fig,f"{PROF}/bl_temperature_ratio.png")

# ======================================================================
# 6.  3D SURFACE CONTOURS + SKIN-FRICTION VECTORS
# ======================================================================
def build_3d_field():
    """Run solver at span stations, map Cp/Cf/gamma onto 3D wing surface."""
    etas=np.linspace(0.0,1.0,16)
    X,Y=C.nlf16_panel_points(130)
    data={}
    for e in etas:
        chord=W["root_chord"]+e*(W["tip_chord"]-W["root_chord"])
        twist=e*W["twist_tip_deg"]; aeff=cr["alpha_deg"]+twist
        # same Mach number as run_solution.py, so the 3-D contours show the
        # solution the surface CSVs of section 9 tabulate and not a second,
        # incompressible one
        r=solve_airfoil(X,Y,aeff,cr["U_inf"],cr["nu_inf"],chord,cr["Tu_pct"],
                        sweep_deg=W["le_sweep_deg"],mach=cr["mach"])
        data[e]=r["surfaces"]
    return etas,data

def _wing_xyz(e,xc,surf):
    chord=W["root_chord"]+e*(W["tip_chord"]-W["root_chord"])
    xle=e*W["span_b"]/2*np.tan(np.radians(W["le_sweep_deg"]))
    z_dih=e*W["span_b"]/2*np.tan(np.radians(W["dihedral_deg"]))
    twist=np.radians(-(e*W["twist_tip_deg"]))
    co=C.nlf16_coords(n=130)
    if surf=="upper": ys=np.interp(xc,co["xu"],co["yu"])
    else: ys=np.interp(xc,co["xl"][::-1],co["yl"][::-1])
    xq=xc-0.25; yq=ys
    xr=0.25+xq*np.cos(twist)-yq*np.sin(twist); yr=xq*np.sin(twist)+yq*np.cos(twist)
    Xp=xle+xr*chord; Yp=e*W["span_b"]/2; Zp=z_dih+yr*chord
    return Xp,Yp,Zp

def plot_3d(field):
    etas,data=field
    from matplotlib import cm
    from matplotlib.colors import Normalize
    nchord=60
    xc=0.5*(1-np.cos(np.linspace(0,np.pi,nchord)))
    for qty,cmap,lab,fname,vmnmx in [
        ("Cp",FIELD_CMAP,"C_p","td_Cp",(-0.7,1.0)),
        ("Cf",CF_CMAP,"C_f  ×10³","td_Cf",(0,6.0)),
        ("gamma",GAMMA_CMAP,"intermittency γ","td_gamma",(0,1))]:
        fig=plt.figure(figsize=(11,7))
        ax=fig.add_subplot(111,projection="3d")
        # gather values for normalisation
        allv=[]
        surfgrids={}
        for surf in ["upper","lower"]:
            XX=[];YY=[];ZZ=[];VV=[]
            for e in etas:
                s=data[e][surf]
                xs=s["x"]; val=s[qty if qty!="Cf" else "Cf"]
                if qty=="Cp":
                    valc=np.clip(_chord_interp(xs,s["Cp"],xc),-0.7,1.0)
                elif qty=="Cf":
                    # scale to x10^3 and clip the stagnation singularity
                    valc=np.clip(_chord_interp(xs,s["Cf"],xc)*1e3,0,6.0)
                else:
                    valc=_chord_interp(xs,s["gamma"],xc)
                xr=[];yr=[];zr=[]
                for xx in xc:
                    Xp,Yp,Zp=_wing_xyz(e,xx,surf); xr.append(Xp);yr.append(Yp);zr.append(Zp)
                XX.append(xr);YY.append(yr);ZZ.append(zr);VV.append(valc)
            surfgrids[surf]=(np.array(XX),np.array(YY),np.array(ZZ),np.array(VV))
            allv.append(np.array(VV))
        av=np.concatenate([a.ravel() for a in allv])
        vmin,vmax=(vmnmx if vmnmx else (np.nanpercentile(av,2),np.nanpercentile(av,98)))
        norm=Normalize(vmin,vmax)
        for surf in ["upper","lower"]:
            XX,YY,ZZ,VV=surfgrids[surf]
            fc=cmap(norm(VV))
            ax.plot_surface(XX,YY,ZZ,facecolors=fc,rstride=1,cstride=1,
                            linewidth=0,antialiased=True,shade=False)
        m=cm.ScalarMappable(norm=norm,cmap=cmap); m.set_array([])
        cb=fig.colorbar(m,ax=ax,shrink=0.6,pad=0.02); cb.set_label(lab)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y span [m]"); ax.set_zlabel("z [m]")
        ax.view_init(elev=34,azim=-62); _tidy3d(ax)
        try: ax.set_box_aspect((3.0,7.0,1.2))
        except Exception: pass
        ax.set_title(f"3D wing surface contour: {lab} (cruise) — AETHER-NLF 25",
                     color=INK,fontweight="normal")
        fig.text(0.5,0.02,f"Upper+lower surfaces, lofted {W['section']} sections; "
                 "cruise FL360 M0.42",ha="center",color=INK_SOFT,fontsize=10)
        ax.grid(False)
        fig.savefig(f"{TD}/{fname}.png",dpi=170,facecolor="white"); plt.close(fig)

def plot_3d_vectors(field):
    """Surface-flow direction arrows on the upper surface, coloured by C_f.

    The arrows lie along the chordwise surface direction of each strip.  A
    strip formulation carries no span-wise wall shear, so this is not a
    skin-friction line pattern and is not labelled as one; what it shows is
    where the wall shear is high and low over the wing.
    """
    etas,data=field
    fig=plt.figure(figsize=(11,7))
    ax=fig.add_subplot(111,projection="3d")
    xc=np.linspace(0.02,0.98,18)
    from matplotlib import cm
    from matplotlib.colors import Normalize
    pts=[]; cfs=[]
    for e in etas[::2]:
        s=data[e]["upper"]
        cfc=_chord_interp(s["x"],s["Cf"],xc)
        for j,xx in enumerate(xc):
            Xp,Yp,Zp=_wing_xyz(e,xx,"upper")
            Xp2,_,Zp2=_wing_xyz(e,min(xx+0.03,1.0),"upper")
            pts.append((Xp,Yp,Zp,Xp2-Xp,0.0,Zp2-Zp)); cfs.append(cfc[j])
    pts=np.array(pts); cfs=np.array(cfs)
    norm=Normalize(np.percentile(cfs,5),np.percentile(cfs,95))
    cols=CF_CMAP(norm(cfs))
    ax.quiver(pts[:,0],pts[:,1],pts[:,2],pts[:,3],pts[:,4],pts[:,5],
              length=0.5,normalize=True,colors=cols,linewidth=1.3)
    m=cm.ScalarMappable(norm=norm,cmap=CF_CMAP); m.set_array([])
    cb=fig.colorbar(m,ax=ax,shrink=0.6,pad=0.02); cb.set_label("C_f")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y span [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=40,azim=-65); _tidy3d(ax)
    try: ax.set_box_aspect((3.0,7.0,1.2))
    except Exception: pass
    ax.set_title("Upper-surface flow direction, coloured by C_f (strip "
                 "formulation: chordwise only)",
                 color=INK,fontweight="normal")
    ax.grid(False)
    fig.savefig(f"{TD}/td_skinfriction_vectors.png",dpi=170,facecolor="white")
    plt.close(fig)

# ======================================================================
# 7.  REMAINING CSVs  (every other plottable CSV gets a clean chart)
# ======================================================================
def plot_remaining_csvs():
    # --- 7a. 3-D lofted wing sections (wing_sections_3d.csv) ---
    df=pd.read_csv("01_geometry/wing_sections_3d.csv")
    fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111,projection="3d")
    etas=sorted(df["eta"].unique())
    from matplotlib import cm
    for e in etas:
        sub=df[df["eta"]==e]
        col=FIELD_CMAP(e)
        for surf in ["upper","lower"]:
            s=sub[sub["surface"]==surf]
            ax.plot(s["X_m"],s["Y_m"],s["Z_m"],color=col,lw=1.1)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y span [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=26,azim=-60); _tidy3d(ax)
    try: ax.set_box_aspect((3,7,1.2))
    except Exception: pass
    ax.set_title("Lofted wing sections (from wing_sections_3d.csv)",color=INK)
    ax.grid(False)
    fig.savefig(f"{CSVP}/geo_sections_3d.png",dpi=170,facecolor="white",
                bbox_inches="tight"); plt.close(fig)

    # --- 7b. transition summary metric bars (transition_summary.csv) ---
    ts=pd.read_csv("04_solution/transition_summary.csv")
    ts["label"]=ts["case"]+" "+ts["surface"]
    fig,ax=new_fig(8.6,5.0)
    cols=[PALETTE[2] if "CRUISE" in l else PALETTE[1] for l in ts["case"]]
    bars=ax.bar(ts["label"],ts["laminar_run_pct"],color=cols,width=0.62)
    for b,v,m in zip(bars,ts["laminar_run_pct"],ts["mechanism"]):
        ax.text(b.get_x()+b.get_width()/2,v+0.8,f"{v:.1f}%\n({m})",ha="center",
                va="bottom",fontsize=9,color=INK)
    ax.set_ylabel("laminar-flow run  [% chord]")
    ax.set_ylim(0,max(ts["laminar_run_pct"])*1.35)
    ax.set_title("Predicted laminar-flow extent by case & surface")
    ax.tick_params(axis="x",labelrotation=12)
    finish(fig,f"{CSVP}/transition_summary_bar.png")

    # --- 7c. universal calibration constants (calibration_constants.csv) ---
    cc=pd.read_csv("03_model_setup/calibration_constants.csv")
    # N_crit is not a fixed constant any more - it is computed from the
    # local turbulence intensity - so its "value" is text.  Plot only the
    # numeric constants and note the rest on the axis label.
    cc=cc.copy()
    cc["num"]=pd.to_numeric(cc["value"],errors="coerce")
    nonnum=cc.loc[cc["num"].isna()]
    # two different kinds of non-numeric entry, and calling both "computed from
    # Tu" was wrong: cf_amp and two_eq are switches that select a closure, while
    # N_crit and sigma_sep are quantities the model computes at run time.
    flags=[r.constant for _,r in nonnum.iterrows()
           if str(r.value).strip() in ("True","False")]
    derived=[r.constant for _,r in nonnum.iterrows()
             if str(r.value).strip() not in ("True","False")]
    cc=cc.dropna(subset=["num"]).reset_index(drop=True)
    fig,ax=new_fig(8.6,5.0)
    yy=np.arange(len(cc))
    bars=ax.barh(yy,cc["num"],color=PALETTE[4],height=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(cc["constant"])
    # Symmetric log, not log: the constant set spans five decades AND contains a
    # negative member (the Thwaites separation parameter).  On a plain log axis
    # that bar has no finite extent and matplotlib fails while sizing the
    # figure, which is what stopped this script part-way through.
    ax.set_xscale("symlog",linthresh=0.01,linscale=0.6)
    ax.set_xlabel("constant value (symmetric-log scale)")
    for b,v in zip(bars,cc["num"]):
        off = 1.12 if v > 0 else 1.6
        ax.text(v*off if abs(v)>0.02 else (0.03 if v>0 else -0.05),
                b.get_y()+b.get_height()/2,f"{v:g}",
                va="center",ha="left" if v>0 else "right",
                fontsize=9,color=INK)
    ax.set_xlim(-0.6,600)
    ax.axvline(0.0,color=INK_SOFT,lw=0.9)
    ax.invert_yaxis()
    ax.set_title("UTSS calibration constant set (single set, all cases)")
    note=[]
    if derived: note.append("computed at run time, not fixed: "+", ".join(derived))
    if flags:   note.append("closure switches: "+", ".join(flags))
    finish(fig,f"{CSVP}/calibration_constants.png",
           caption=("Not plotted — "+";  ".join(note)) if note else None)

    # --- 7d. flight conditions comparison (flow_conditions.csv) ---
    fc=pd.read_csv("03_model_setup/flow_conditions.csv")
    keys=[("mach","Mach"),("U_inf_ms","U∞ [m/s]"),("Tu_pct","Tu [%]"),
          ("alpha_deg","α [deg]"),("Re_MAC","Re_MAC")]
    fig,axs=plt.subplots(1,5,figsize=(13.5,3.8))
    names=[c.split(" (")[0] for c in fc["case"]]
    for ax,(k,lab) in zip(axs,keys):
        ax.bar(names,fc[k],color=[PALETTE[2],PALETTE[1]],width=0.6)
        ax.set_title(lab,fontsize=11); ax.tick_params(axis="x",labelrotation=15,labelsize=9)
        if k=="Re_MAC": ax.set_yscale("log")
        for i,v in enumerate(fc[k]):
            ax.text(i,v,f"{v:g}",ha="center",va="bottom",fontsize=8.5,color=INK)
        ax.margins(y=0.18)
    fig.suptitle("Flight-condition comparison: cruise vs climb (flow_conditions.csv)",
                 color=INK)
    fig.tight_layout(rect=[0,0,1,0.93])
    fig.savefig(f"{CSVP}/conditions_compare.png",dpi=170,facecolor="white")
    plt.close(fig)

    # --- 7e. parameter/metric CSVs rendered as clean table-figures ---
    import textwrap
    def render_table_figure(csv,title,outpath,wrapcol=34,fontsize=9.5):
        df=pd.read_csv(csv).fillna("")
        def wrap(x):
            x=str(x)
            return "\n".join(textwrap.wrap(x,wrapcol)) if len(x)>wrapcol else x
        data=[[wrap(c) for c in row] for row in df.values]
        cols=[str(c) for c in df.columns]; ncol=len(cols)
        rowlines=[max(1,max(str(c).count("\n")+1 for c in row)) for row in data]
        total=sum(rowlines)+1
        fig_h=min(0.7+0.30*total, 13.5); fig_w=min(3.4+3.2*ncol,12.5)
        fig,ax=new_fig(fig_w,fig_h); ax.axis("off")
        if ncol==2: colW=[0.34,0.66]
        elif ncol==3: colW=[0.52,0.24,0.24]
        else: colW=[1/ncol]*ncol
        tbl=ax.table(cellText=data,colLabels=cols,cellLoc="left",loc="center",
                     colWidths=colW)
        tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize)
        denom=sum(rowlines)*1.3+1.5          # keep table <1 axes-height
        h_unit=0.84/denom
        for (r,c),cell in tbl.get_celld().items():
            cell.set_edgecolor(INK_SOFT); cell.set_linewidth(0.6)
            cell.set_text_props(ha="left",va="center")
            if r==0:
                cell.set_height(h_unit*1.5)
                cell.set_facecolor(PALETTE[0]); cell.get_text().set_color("white")
            else:
                cell.set_height(h_unit*rowlines[r-1]*1.3)
                cell.set_facecolor("#eef4fa" if r%2 else "white")
                cell.get_text().set_color(INK)
        ax.set_title(title,color=INK,fontsize=13,pad=10)
        fig.savefig(outpath,dpi=170,facecolor="white",bbox_inches="tight")
        plt.close(fig)

    tabfigs=[
     ("01_geometry/geometry_definition.csv","Geometry definition (geometry_definition.csv)","table_geometry_definition"),
     ("02_mesh/mesh_metrics.csv","Mesh metrics (mesh_metrics.csv)","table_mesh_metrics"),
     ("03_model_setup/material_properties.csv","Air / material properties (material_properties.csv)","table_material_properties"),
     ("03_model_setup/solver_settings.csv","Solver configuration (solver_settings.csv)","table_solver_settings"),
     ("04_solution/integrated_forces.csv","Integrated forces (integrated_forces.csv)","table_integrated_forces"),
    ]
    for csv,title,name in tabfigs:
        render_table_figure(csv,title,f"{CSVP}/{name}.png")
    print("remaining-CSV plots done")

if __name__=="__main__":
    plot_geometry_csvs(); plot_mesh_csvs()
    for case in ["cruise","climb"]: plot_surface(case)
    plot_cruise_climb_compare(); plot_polar(); plot_spanwise(); plot_nlf_vs_turb()
    for case in ["cruise","climb"]: plot_contours(case)
    plot_profiles()
    plot_remaining_csvs()
    field=build_3d_field(); plot_3d(field); plot_3d_vectors(field)
    print("postprocessing done.")
    for d in [CSVP,CONT,PROF,TD]:
        print(d,"->",sorted(os.listdir(d)))
