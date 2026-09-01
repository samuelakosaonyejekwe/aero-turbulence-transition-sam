"""
gen_geometry.py
Geometry definition + dimensioned engineering drawings for the
AETHER-NLF 25 NLF wing case study.

Outputs (01_geometry/):
  airfoil_UTSS-NLF16.csv, wing_planform.csv, wing_sections_3d.csv,
  geometry_definition.csv
  drawings/  dimensioned drawings (orthographic / iso / sectional)
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "solver")
import case_config as C
from uplot import apply_style, INK, INK_SOFT, PALETTE, finish
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

apply_style()
# Engineering drawings: clean WHITE sheet, NO data grid (drafting standard)
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = False
GEO = "01_geometry"; DWG = os.path.join(GEO, "drawings")
os.makedirs(DWG, exist_ok=True)

DIM   = "#b5651d"   # dimension lines (sienna) - clearly not black
OUT   = "#1b4965"   # object outline (deep teal-blue)
CTR   = "#7b4ea3"   # centre lines (violet)
HID   = "#9aa7b4"   # hidden lines (grey-blue)

# ----------------------------------------------------------------------
# Drafting primitives  (ISO/ASME dimensioning style)
# ----------------------------------------------------------------------
def dim_linear(ax, p1, p2, offset, text, side=1, fs=10, color=DIM,
               horiz=None):
    """Dimension with extension lines, arrowheads and centred text."""
    p1 = np.array(p1, float); p2 = np.array(p2, float)
    d = p2 - p1; L = np.hypot(*d)
    if horiz is None:
        horiz = abs(d[0]) >= abs(d[1])
    if horiz:
        yo = max(p1[1], p2[1]) + offset if side > 0 else min(p1[1], p2[1]) + offset
        a1 = np.array([p1[0], yo]); a2 = np.array([p2[0], yo])
        ax.plot([p1[0], p1[0]], [p1[1], yo], color=color, lw=0.7)
        ax.plot([p2[0], p2[0]], [p2[1], yo], color=color, lw=0.7)
    else:
        xo = max(p1[0], p2[0]) + offset if side > 0 else min(p1[0], p2[0]) + offset
        a1 = np.array([xo, p1[1]]); a2 = np.array([xo, p2[1]])
        ax.plot([p1[0], xo], [p1[1], p1[1]], color=color, lw=0.7)
        ax.plot([p2[0], xo], [p2[1], p2[1]], color=color, lw=0.7)
    ax.annotate("", xy=a2, xytext=a1,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
    mid = 0.5*(a1+a2)
    rot = 0 if horiz else 90
    dx = 0 if horiz else (0.012*(1 if side>0 else -1))
    dy = (0.012*(1 if side>0 else -1)) if horiz else 0
    ax.text(mid[0]+dx, mid[1]+dy, text, color=color, fontsize=fs,
            ha="center", va="center", rotation=rot,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                      alpha=0.9))


def angle_dim(ax, vertex, p_a, p_b, text, r=0.4, color=DIM, fs=10):
    v = np.array(vertex, float)
    a0 = np.arctan2(p_a[1]-v[1], p_a[0]-v[0])
    a1 = np.arctan2(p_b[1]-v[1], p_b[0]-v[0])
    th = np.linspace(a0, a1, 40)
    ax.plot(v[0]+r*np.cos(th), v[1]+r*np.sin(th), color=color, lw=1.0)
    am = 0.5*(a0+a1)
    ax.text(v[0]+(r+0.12)*np.cos(am), v[1]+(r+0.12)*np.sin(am), text,
            color=color, fontsize=fs, ha="center", va="center")


def title_block(ax, title, dwg_no, scale="NTS", view=""):
    fig = ax.figure
    ax.set_title(title, fontsize=14, fontweight="normal", color=INK, pad=12)
    fig.subplots_adjust(bottom=0.22, top=0.90)
    fig.text(0.012, 0.065,
             f"PROJECT: AETHER-NLF 25  |  NLF WING  |  SECTION {C.WING['section']}",
             fontsize=10, color=INK_SOFT)
    fig.text(0.012, 0.030,
             "DRAWN BY: AKOSA SAMUEL ONYEJEKWE",
             fontsize=10, color=INK_SOFT)
    fig.text(0.988, 0.065,
             f"DWG {dwg_no}   SCALE {scale}   {view}   3rd-ANGLE   "
             f"UNITS m (noted)   UTSS-CASE-2026",
             fontsize=10, color=INK_SOFT, ha="right")
    # thin border frame
    fig.patches.append(plt.Rectangle((0.006,0.012),0.988,0.974,
                       transform=fig.transFigure, fill=False,
                       edgecolor=INK_SOFT, lw=1.0))


def finish_dwg(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return path


# ======================================================================
# 1.  AIRFOIL + WING GEOMETRY DATA
# ======================================================================
def build_geometry():
    co = C.nlf16_coords(n=160)
    df_af = pd.DataFrame({
        "x_c": co["x"], "y_camber": co["yc"], "half_thickness": co["yt"],
        "x_upper": co["xu"], "y_upper": co["yu"],
        "x_lower": co["xl"], "y_lower": co["yl"]})
    df_af.to_csv(f"{GEO}/airfoil_UTSS-NLF16.csv", index=False)

    W = C.WING
    eta = np.linspace(0, 1, 21)
    y = eta*W["span_b"]/2
    chord = W["root_chord"] + eta*(W["tip_chord"]-W["root_chord"])
    x_le = y*np.tan(np.radians(W["le_sweep_deg"]))
    x_te = x_le + chord
    twist = eta*W["twist_tip_deg"]
    z_dih = y*np.tan(np.radians(W["dihedral_deg"]))
    df_pl = pd.DataFrame({"eta": eta, "y_m": y, "chord_m": chord,
                          "x_le_m": x_le, "x_te_m": x_te,
                          "twist_deg": twist, "z_dihedral_m": z_dih,
                          "Re_local": C.CRUISE["U_inf"]*chord/C.CRUISE["nu_inf"]})
    df_pl.to_csv(f"{GEO}/wing_planform.csv", index=False)

    # 3D lofted surface (sampled)
    rows = []
    for e, yy, cc, xle, tw, zz in zip(eta, y, chord, x_le, twist, z_dih):
        c2 = C.nlf16_coords(n=60)
        a = np.radians(-tw)  # washout
        for surf, xs, ys in [("upper", c2["xu"], c2["yu"]),
                             ("lower", c2["xl"], c2["yl"])]:
            # scale by chord, twist about quarter chord, place at x_le, z dihedral
            xq = (xs-0.25); yq = ys
            xr = 0.25 + xq*np.cos(a) - yq*np.sin(a)
            yr = xq*np.sin(a) + yq*np.cos(a)
            X = xle + xr*cc
            Z = zz + yr*cc
            for Xi, Zi, xc_i in zip(X, Z, xs):
                rows.append((round(e,4), surf, round(Xi,5), round(yy,5),
                             round(Zi,5), round(xc_i,5)))
    df3 = pd.DataFrame(rows, columns=["eta","surface","X_m","Y_m","Z_m","x_c"])
    df3.to_csv(f"{GEO}/wing_sections_3d.csv", index=False)

    # geometry definition table
    tmax = co["yt"].max()*2; xtmax = co["x"][np.argmax(co["yt"])]
    defs = [
        ("Aircraft", C.AIRCRAFT["name"], "-"),
        ("Wing reference area S", f"{W['area_S']:.3f}", "m^2"),
        ("Wing span b", f"{W['span_b']:.2f}", "m"),
        ("Aspect ratio AR", f"{W['AR']:.2f}", "-"),
        ("Root chord c_root", f"{W['root_chord']:.3f}", "m"),
        ("Tip chord c_tip", f"{W['tip_chord']:.3f}", "m"),
        ("Taper ratio", f"{W['taper']:.3f}", "-"),
        ("Mean aerodynamic chord MAC", f"{W['MAC']:.3f}", "m"),
        ("Leading-edge sweep", f"{W['le_sweep_deg']:.1f}", "deg"),
        ("Dihedral", f"{W['dihedral_deg']:.1f}", "deg"),
        ("Tip washout (twist)", f"{W['twist_tip_deg']:.1f}", "deg"),
        ("Section", W["section"], "-"),
        ("Section max thickness", f"{tmax*100:.1f}", "% chord"),
        ("Max-thickness location", f"{xtmax*100:.1f}", "% chord"),
        ("Section design Cl", "0.45", "-"),
    ]
    pd.DataFrame(defs, columns=["parameter","value","unit"]).to_csv(
        f"{GEO}/geometry_definition.csv", index=False)
    return df_af, df_pl, df3, co


# ======================================================================
# 2.  DRAWINGS
# ======================================================================
def draw_airfoil_section(co):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = co["x"]
    yuI = lambda xq: np.interp(xq, co["xu"], co["yu"])
    ylI = lambda xq: np.interp(xq, co["xl"], co["yl"])
    ax.plot(co["xu"], co["yu"], color=OUT, lw=2.2)
    ax.plot(co["xl"], co["yl"], color=OUT, lw=2.2)
    ax.plot(x, co["yc"], color=CTR, lw=1.1, ls=(0,(7,3,1,3)), label="mean camber line")
    ax.fill_between(co["xu"], co["yu"], co["yc"], color=PALETTE[0], alpha=0.05)
    ax.fill_between(co["xl"], co["yl"], co["yc"], color=PALETTE[1], alpha=0.05)
    ax.plot([0,1],[0,0], color=HID, lw=0.9, ls=(0,(8,4)), label="chord line")

    it = np.argmax(co["yt"]); xt = x[it]; tmax = co["yt"][it]*2
    yu_t, yl_t = yuI(xt), ylI(xt)

    # --- horizontal chord dimension (clear, below everything) ---
    dim_linear(ax, (0,-0.175),(1,-0.175), -0.045,
               "CHORD  c (reference)   |   MAC = 2.000 m", side=-1, fs=10)
    # --- x(t_max) horizontal dimension (clear, above) ---
    dim_linear(ax, (0,0.150),(xt,0.150), 0.028, f"x(t_max) = {xt:.2f} c", side=1, fs=10)
    # --- t_max vertical thickness arrow + leadered label in clear space ---
    ax.plot([xt,xt],[yl_t,yu_t], color=HID, lw=0.8, ls=(0,(4,3)))
    ax.annotate("", xy=(xt,yu_t), xytext=(xt,yl_t),
                arrowprops=dict(arrowstyle="<->", color=DIM, lw=1.2))
    ax.annotate(f"t_max = {tmax:.3f} c\n(16.0 %)", xy=(xt,yl_t),
                xytext=(xt+0.085,-0.135), color=DIM, fontsize=10, ha="left",
                va="center", arrowprops=dict(arrowstyle="->", color=DIM, lw=0.9))
    # --- leading & trailing edge callouts (clear of geometry) ---
    ax.annotate("rounded NLF leading edge\nr_LE ≈ 0.015 c  ·  favourable\ngradient to ~0.45 c",
                xy=(0.012,0.004), xytext=(0.15,0.125), color=INK_SOFT, fontsize=10,
                ha="left", arrowprops=dict(arrowstyle="->", color=INK_SOFT))
    ax.annotate("cusped, aft-loaded\ntrailing edge", xy=(0.985, co["yc"][-3]),
                xytext=(0.80,0.135), color=INK_SOFT, fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK_SOFT))
    # --- specification box (clear lower-left corner) ---
    spec=("UTSS-NLF16  natural-laminar-flow section\n"
          "t/c = 0.160 @ 0.42 c   ·   design C_l = 0.45\n"
          "aft-loaded camber   ·   sharp T.E.")
    ax.text(0.015, -0.205, spec, fontsize=10, color=INK,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef4fa", ec=INK_SOFT, lw=0.9))
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    ax.set_xlim(-0.06, 1.10); ax.set_ylim(-0.24, 0.21)
    ax.set_aspect("equal", adjustable="box"); ax.grid(False)
    ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
    title_block(ax, f"AEROFOIL SECTION  {C.WING['section']}  (16% NLF)",
                "GEO-001", "1:1 (norm.)", "SECTION VIEW")
    finish_dwg(fig, f"{DWG}/dwg_01_airfoil_section.png")


def draw_planview(df_pl):
    W = C.WING
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    y = df_pl["y_m"].values; xle = df_pl["x_le_m"].values
    xte = df_pl["x_te_m"].values
    # full span mirror
    Y = np.concatenate([-y[::-1], y]); XLE = np.concatenate([xle[::-1], xle])
    XTE = np.concatenate([xte[::-1], xte])
    ax.plot(Y, XLE, color=OUT, lw=2.0); ax.plot(Y, XTE, color=OUT, lw=2.0)
    ax.plot([Y[0],Y[0]],[XLE[0],XTE[0]], color=OUT, lw=2.0)
    ax.plot([Y[-1],Y[-1]],[XLE[-1],XTE[-1]], color=OUT, lw=2.0)
    ax.plot([0,0],[ -0.3, xte.max()+0.3], color=CTR, lw=1.0, ls=(0,(8,4)))
    # quarter chord line
    xc4 = xle + 0.25*(xte-xle)
    ax.plot(np.concatenate([-y[::-1],y]),
            np.concatenate([xc4[::-1],xc4]), color=CTR, lw=1.0, ls=(0,(6,3,1,3)))
    # dimensions
    bt = W["span_b"]
    dim_linear(ax, (-bt/2, -0.55),(bt/2,-0.55), -0.45,
               f"SPAN  b = {bt:.2f} m", side=-1)
    dim_linear(ax, (bt/2+0.1, xle[-1]),(bt/2+0.1, xte[-1]), 0.5,
               f"c_tip = {W['tip_chord']:.2f} m", horiz=False, side=1, fs=10)
    dim_linear(ax, (-bt/2-0.1, xle[0]),(-bt/2-0.1, xte[0]), -0.5,
               f"c_root = {W['root_chord']:.2f} m", horiz=False, side=-1, fs=10)
    # sweep angle
    angle_dim(ax, (0, xle[0]), (0, xte[0]), (y[-1], xle[-1]),
              f"Λ_LE = {W['le_sweep_deg']:.0f}°", r=1.4)
    ax.annotate("c/4 sweep line", xy=(y[10], xc4[10]), xytext=(2.5, 0.2),
                color=CTR, fontsize=10, arrowprops=dict(arrowstyle="->", color=CTR))
    # section cut marker B-B
    ax.plot([3.0,3.0],[xle[0]-0.2, xte.max()+0.2], color=PALETTE[1], lw=1.2, ls=(0,(2,2)))
    ax.text(3.0, xte.max()+0.35, "B", color=PALETTE[1], ha="center", fontsize=11, fontweight="normal")
    ax.text(3.0, xle[0]-0.4, "B", color=PALETTE[1], ha="center", fontsize=11, fontweight="normal")
    ax.set_xlabel("span-wise  y  [m]"); ax.set_ylabel("stream-wise  x  [m]")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(False)
    ax.set_title("")
    txt = (f"S = {W['area_S']:.2f} m²   AR = {W['AR']:.2f}   "
           f"λ = {W['taper']:.2f}   MAC = {W['MAC']:.2f} m")
    ax.text(0.5, 0.05, txt, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=INK, bbox=dict(boxstyle="round", fc="#eef4fa", ec=INK_SOFT))
    title_block(ax, "WING PLANFORM  -  PLAN VIEW (TOP)", "GEO-002", "1:120", "PLAN")
    finish_dwg(fig, f"{DWG}/dwg_02_planview.png")


def draw_front_side(df_pl):
    W = C.WING
    y = df_pl["y_m"].values; z = df_pl["z_dihedral_m"].values
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    # FRONT VIEW (dihedral)
    ax = axes[0]
    Y = np.concatenate([-y[::-1], y]); Z = np.concatenate([z[::-1], z])
    tcurve = df_pl["chord_m"].values*0.16
    Zup = Z + np.concatenate([tcurve[::-1],tcurve])*0.5
    Zlo = Z - np.concatenate([tcurve[::-1],tcurve])*0.5
    ax.fill_between(Y, Zlo, Zup, color=PALETTE[0], alpha=0.18)
    ax.plot(Y, Zup, color=OUT, lw=1.8); ax.plot(Y, Zlo, color=OUT, lw=1.8)
    ax.plot([0,0],[ -0.2, z.max()+0.4], color=CTR, lw=1.0, ls=(0,(8,4)))
    # dihedral note in clear head-room (no overlap with the wing)
    ax.annotate(f"Γ = {W['dihedral_deg']:.0f}° dihedral",
                xy=(y[-1]*0.55, z[-1]*0.55), xytext=(1.0, z.max()+1.05),
                color=DIM, fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color=DIM, lw=0.9))
    dim_linear(ax, (-W['span_b']/2,-0.35),(W['span_b']/2,-0.35), -0.25,
               f"b = {W['span_b']:.2f} m", side=-1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(z.min()-0.55, z.max()+1.5)
    ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]")
    ax.set_title("FRONT VIEW (looking aft)  -  dihedral", color=INK, fontsize=11)
    ax.grid(False)
    # SIDE VIEW (root + tip profiles)
    ax = axes[1]
    for e,col,lab in [(0,PALETTE[0],"root"),(1,PALETTE[1],"tip")]:
        sub = df_pl.iloc[0 if e==0 else -1]
        co = C.nlf16_coords(n=80)
        cc = sub["chord_m"]; xle = sub["x_le_m"]
        ax.plot(xle+co["xu"]*cc, co["yu"]*cc, color=col, lw=1.8, label=f"{lab} c={cc:.2f}m")
        ax.plot(xle+co["xl"]*cc, co["yl"]*cc, color=col, lw=1.8)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)
    ax.set_aspect("equal"); ax.invert_xaxis()
    ax.set_ylim(top=ax.get_ylim()[1]+0.06)
    ax.set_xlabel("x [m] (LE right)"); ax.set_ylabel("z [m]")
    ax.set_title("SIDE VIEW  -  root & tip sections", color=INK, fontsize=11)
    ax.grid(False)
    fig.suptitle("WING ORTHOGRAPHIC VIEWS  (GEO-003)", color=INK, fontweight="normal")
    fig.text(0.5, 0.005, "DRAWN BY: AKOSA SAMUEL ONYEJEKWE  |  PROJECT AETHER-NLF 25  |  "
             "UTSS-CASE-2026", ha="center", color=INK_SOFT, fontsize=9)
    finish(fig, f"{DWG}/dwg_03_front_side.png")


def draw_orthographic(df_pl):
    """Single 3rd-angle sheet: plan + front + side + iso inset."""
    W = C.WING
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25,1], width_ratios=[1.5,1])
    y = df_pl["y_m"].values; xle=df_pl["x_le_m"].values; xte=df_pl["x_te_m"].values
    z = df_pl["z_dihedral_m"].values; chord=df_pl["chord_m"].values
    # PLAN (top-left)
    axp = fig.add_subplot(gs[0,0])
    Y=np.concatenate([-y[::-1],y]); XLE=np.concatenate([xle[::-1],xle]); XTE=np.concatenate([xte[::-1],xte])
    axp.plot(Y,XLE,color=OUT,lw=1.8); axp.plot(Y,XTE,color=OUT,lw=1.8)
    axp.plot([Y[0],Y[0]],[XLE[0],XTE[0]],color=OUT,lw=1.8)
    axp.plot([Y[-1],Y[-1]],[XLE[-1],XTE[-1]],color=OUT,lw=1.8)
    axp.plot([0,0],[-0.3,xte.max()+0.3],color=CTR,lw=0.9,ls=(0,(8,4)))
    dim_linear(axp,(-W['span_b']/2,-0.5),(W['span_b']/2,-0.5),-0.4,f"b={W['span_b']:.1f} m",side=-1,fs=10)
    axp.set_aspect("equal"); axp.invert_yaxis(); axp.grid(False)
    axp.set_title("PLAN", fontsize=10, color=INK); axp.set_xlabel("y [m]"); axp.set_ylabel("x [m]")
    # FRONT (bottom-left)
    axf = fig.add_subplot(gs[1,0])
    Z=np.concatenate([z[::-1],z]); tt=chord*0.16
    Zu=Z+np.concatenate([tt[::-1],tt])*0.5; Zl=Z-np.concatenate([tt[::-1],tt])*0.5
    axf.fill_between(Y,Zl,Zu,color=PALETTE[0],alpha=0.18)
    axf.plot(Y,Zu,color=OUT,lw=1.6); axf.plot(Y,Zl,color=OUT,lw=1.6)
    axf.annotate(f"Γ = {W['dihedral_deg']:.0f}° dihedral",
                 xy=(y[-1]*0.55, z[-1]*0.55), xytext=(0.0, z.max()+1.15),
                 color=DIM, fontsize=10, ha="center",
                 arrowprops=dict(arrowstyle="->", color=DIM, lw=0.9))
    axf.set_aspect("equal", adjustable="box")
    axf.set_ylim(z.min()-0.55, z.max()+1.6); axf.grid(False)
    axf.set_title("FRONT", fontsize=10, color=INK); axf.set_xlabel("y [m]"); axf.set_ylabel("z [m]")
    # SIDE (top-right)
    axs = fig.add_subplot(gs[0,1])
    sub=df_pl.iloc[0]; co=C.nlf16_coords(n=80); cc=sub["chord_m"]
    axs.plot(co["xu"]*cc,co["yu"]*cc,color=OUT,lw=1.8); axs.plot(co["xl"]*cc,co["yl"]*cc,color=OUT,lw=1.8)
    dim_linear(axs,(0,-0.45),(cc,-0.45),-0.12,f"c_root={cc:.2f} m",side=-1,fs=10)
    axs.set_aspect("equal"); axs.invert_xaxis(); axs.grid(False)
    axs.set_title("SIDE (root section)", fontsize=10, color=INK)
    axs.set_xlabel("x [m]"); axs.set_ylabel("z [m]")
    # ISO inset (bottom-right)
    axi = fig.add_subplot(gs[1,1], projection="3d")
    _iso_wing(axi, df_pl)
    # pictorial inset: drop tick numbers (they collide with axis labels)
    axi.set_xticklabels([]); axi.set_yticklabels([]); axi.set_zticklabels([])
    axi.set_title("ISOMETRIC", fontsize=10, color=INK)
    fig.suptitle("AETHER-NLF 25  WING  -  ORTHOGRAPHIC PROJECTION (3rd ANGLE)   DWG GEO-004",
                 fontsize=13, fontweight="normal", color=INK)
    fig.text(0.5, 0.01, "All dimensions in metres unless noted  |  Scale 1:120  |  "
             "UTSS-CASE-2026  |  DRAWN BY: AKOSA SAMUEL ONYEJEKWE",
             ha="center", color=INK_SOFT, fontsize=10)
    fig.tight_layout(rect=[0,0.02,1,0.96])
    fig.savefig(f"{DWG}/dwg_04_orthographic.png", dpi=170, facecolor="white")
    plt.close(fig)


def _iso_wing(ax, df_pl):
    e=df_pl["eta"].values; y=df_pl["y_m"].values; xle=df_pl["x_le_m"].values
    z=df_pl["z_dihedral_m"].values; chord=df_pl["chord_m"].values
    twist=df_pl["twist_deg"].values
    Us=[]; Ls=[]
    co=C.nlf16_coords(n=40)
    for yy,xl,zz,cc,tw in zip(y,xle,z,chord,twist):
        a=np.radians(-tw)
        def place(xs,ys):
            xq=xs-0.25; yq=ys
            xr=0.25+xq*np.cos(a)-yq*np.sin(a); yr=xq*np.sin(a)+yq*np.cos(a)
            return xl+xr*cc, np.full_like(xs,yy), zz+yr*cc
        Us.append(place(co["xu"],co["yu"])); Ls.append(place(co["xl"],co["yl"]))
    for surf,col in [(Us,PALETTE[0]),(Ls,PALETTE[2])]:
        X=np.array([s[0] for s in surf]); Y=np.array([s[1] for s in surf]); Z=np.array([s[2] for s in surf])
        ax.plot_surface(X,Y,Z,color=col,alpha=0.55,linewidth=0,antialiased=True,shade=True)
    ax.set_xlabel("x",fontsize=10); ax.set_ylabel("y",fontsize=10); ax.set_zlabel("z",fontsize=10)
    ax.view_init(elev=22, azim=-58)
    try: ax.set_box_aspect((3,6,1))
    except Exception: pass
    ax.grid(False)


def draw_isometric(df_pl):
    fig = plt.figure(figsize=(10,7.5))
    ax = fig.add_subplot(111, projection="3d")
    _iso_wing(ax, df_pl)
    ax.set_title("AETHER-NLF 25 WING  -  ISOMETRIC VIEW  (DWG GEO-005)",
                 color=INK, fontweight="normal")
    fig.text(0.5,0.02,"Upper surface (blue)  /  Lower surface (teal)  -  lofted from "
             f"{C.WING['section']} sections  |  half-span shown  |  "
             "DRAWN BY: AKOSA SAMUEL ONYEJEKWE",
             ha="center", color=INK_SOFT, fontsize=10)
    fig.savefig(f"{DWG}/dwg_05_isometric.png", dpi=170, facecolor="white")
    plt.close(fig)


def draw_section_BB(df_pl):
    """Fully-detailed structural sectional view B-B at y = 3.0 m:
    skin, front & rear spar webs + caps, stringers, dimensioned spar
    stations, material hatching and callouts."""
    yc=3.0
    chord=np.interp(yc, df_pl["y_m"], df_pl["chord_m"])
    fig, ax = plt.subplots(figsize=(12.5,5.4))
    co=C.nlf16_coords(n=240)
    xu,yu = co["xu"]*chord, co["yu"]*chord
    xl,yl = co["xl"]*chord, co["yl"]*chord
    def yU(xq): return np.interp(xq, xu, yu)
    def yL(xq): return np.interp(xq, xl[::-1], yl[::-1])

    # ---- outer skin (OML) ----
    ax.plot(xu,yu,color=OUT,lw=2.4); ax.plot(xl,yl,color=OUT,lw=2.4)
    # ---- inner skin line (constant skin thickness) -> skin band w/ hatch ----
    tsk=0.012*chord
    xs_=np.linspace(0.005,0.995,200)*chord
    yUi=np.array([yU(x) for x in xs_])-tsk
    yLi=np.array([yL(x) for x in xs_])+tsk
    ax.plot(xs_,yUi,color=HID,lw=0.8); ax.plot(xs_,yLi,color=HID,lw=0.8)
    ax.fill_between(xs_,np.array([yU(x) for x in xs_]),yUi,color=PALETTE[0],alpha=0.35)
    ax.fill_between(xs_,yLi,np.array([yL(x) for x in xs_]),color=PALETTE[0],alpha=0.35)
    # interior fill (wing box / fuel) very light
    ax.fill_between(xs_,yUi,yLi,color=PALETTE[2],alpha=0.05)

    # ---- spars : web + top/bottom caps ----
    spars=[(0.20,"FRONT SPAR"),(0.65,"REAR SPAR")]
    capw=0.045*chord
    for xs,lab in spars:
        xp=xs*chord; yui=yU(xp)-tsk; yli=yL(xp)+tsk
        ax.plot([xp,xp],[yli,yui],color=PALETTE[1],lw=3.2)            # web
        for yy in (yui,yli):                                          # caps
            ax.plot([xp-capw,xp+capw],[yy,yy],color=PALETTE[1],lw=4.0,
                    solid_capstyle="butt")
    # ---- stringers : small markers along inner skin ----
    for xq in np.linspace(0.12,0.92,9)*chord:
        ax.plot(xq, yU(xq)-tsk, marker="s", ms=4, color=PALETTE[4])
        ax.plot(xq, yL(xq)+tsk, marker="s", ms=4, color=PALETTE[4])

    # ---- dimensions : spar stations from LE + chord ----
    ax.plot([0,0],[ -0.05*chord, 0.30*chord], color=CTR, lw=0.8, ls=(0,(6,3)))
    dim_linear(ax,(0,0.34*chord),(0.20*chord,0.34*chord),0.03*chord,
               "0.20 c", side=1, fs=10)
    dim_linear(ax,(0,0.46*chord),(0.65*chord,0.46*chord),0.03*chord,
               "0.65 c", side=1, fs=10)
    dim_linear(ax,(0,-0.24*chord),(chord,-0.24*chord),-0.05*chord,
               f"CHORD  c (y = 3.0 m) = {chord:.3f} m", side=-1, fs=10)
    dim_linear(ax,(0.20*chord,-0.16*chord),(0.65*chord,-0.16*chord),-0.04*chord,
               f"integral wing box = {0.45*chord:.3f} m", side=-1, fs=10)

    # ---- material callouts placed at three DISTINCT, non-overlapping spots ----
    ax.annotate("FRONT SPAR\n(Al-Li web + caps)", xy=(0.20*chord, yU(0.20*chord)-tsk),
                xytext=(0.06*chord, 0.60*chord), color=PALETTE[1], fontsize=10,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.9))
    ax.annotate("CFRP skin\n(t ≈ 12 mm)", xy=(0.42*chord, yU(0.42*chord)),
                xytext=(0.42*chord, 0.62*chord), color=PALETTE[0], fontsize=10,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.9))
    ax.annotate("REAR SPAR\n(Al-Li web + caps)", xy=(0.65*chord, yU(0.65*chord)-tsk),
                xytext=(0.86*chord, 0.60*chord), color=PALETTE[1], fontsize=10,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.9))
    ax.annotate("stringers (Z-section)", xy=(0.82*chord, yL(0.82*chord)+tsk),
                xytext=(0.74*chord,-0.34*chord), color=PALETTE[4], fontsize=10,
                ha="center", arrowprops=dict(arrowstyle="->", color=PALETTE[4], lw=0.9))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.16, chord+0.16); ax.set_ylim(-0.42*chord, 0.78*chord)
    ax.set_xlabel("x [m]  (chord-wise)"); ax.set_ylabel("z [m]")
    title_block(ax,"WING STRUCTURAL SECTION  B-B  (y = 3.0 m)",
                "GEO-006","1:15","SECTION")
    finish_dwg(fig, f"{DWG}/dwg_06_section_BB.png")


if __name__ == "__main__":
    df_af, df_pl, df3, co = build_geometry()
    draw_airfoil_section(co)
    draw_planview(df_pl)
    draw_front_side(df_pl)
    draw_orthographic(df_pl)
    draw_isometric(df_pl)
    draw_section_BB(df_pl)
    print("geometry + drawings done:")
    for f in sorted(os.listdir(DWG)): print("  ", f)
    print("CSV:", [f for f in os.listdir(GEO) if f.endswith('.csv')])
