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
import utss_paths  # noqa: F401  - anchors the repo root and solver/ on
                   # sys.path, so this script works from any directory
import case_config as C
from utss_solver import panel_solve
from uplot import apply_style, INK, INK_SOFT, PALETTE, finish, box_aspect
import matplotlib.pyplot as plt

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


def section_cl():
    """Inviscid section lift of the UNSWEPT 2-D section at cruise conditions.

    The section c_l was typed into the geometry table and onto the drawing as
    a design target, and the two disagreed with each other and with the
    solution the rest of the report tabulates.  It is computed here from the
    same panel method, at the same incidence and Mach number as the cruise
    case.

    IT IS NOT THE CASE-STUDY SECTION LIFT, and calling both "the section c_l"
    is how the README came to quote 0.517 in a headline table whose drag was
    0.496's.  This is the two-dimensional section on its own: alpha = 1.5 deg,
    M = 0.42, no sweep.  The case study is a strip of a 12-degree swept wing,
    which solve_airfoil solves in the plane normal to the leading edge at
    alpha_n = atan(tan(alpha)/cos L) and M cos L and then refers back to the
    streamwise frame as c_l = c_l,n cos^2 L; that is the 0.4962 of
    04_solution/integrated_forces.csv, four per cent below this, and it is the
    one that belongs beside the 47.3 counts of streamwise profile drag.  Both
    are real and they are different quantities, so the labels say which.
    """
    X, Y = C.nlf16_panel_points(130)
    a = C.CRUISE["alpha_deg"]
    xc, yc, Cp, V, th, S = panel_solve(X, Y, a, mach=C.CRUISE["mach"])
    nx = -np.sin(th); ny = np.cos(th)
    Cn = -np.sum(Cp*ny*S); Ca = -np.sum(Cp*nx*S)
    al = np.radians(a)
    return float(Cn*np.cos(al) - Ca*np.sin(al))

def te_wedge_deg(co):
    """Included angle between the two surfaces at the trailing edge, degrees.

    Taken from the last panel of each surface, which is the angle the panel
    method itself sees and therefore the one that makes the trailing edge a
    stagnation point of the inviscid solution.
    """
    xu = np.asarray(co["xu"], float); yu = np.asarray(co["yu"], float)
    xl = np.asarray(co["xl"], float); yl = np.asarray(co["yl"], float)
    su = (yu[-1] - yu[-2])/(xu[-1] - xu[-2])
    sl = (yl[-1] - yl[-2])/(xl[-1] - xl[-2])
    return float(abs(np.degrees(np.arctan(su) - np.arctan(sl))))


# ----------------------------------------------------------------------
# Drafting primitives  (ISO/ASME dimensioning style)
# ----------------------------------------------------------------------
def dim_linear(ax, p1, p2, offset, text, side=1, fs=10, color=DIM,
               horiz=None):
    """Dimension with extension lines, arrowheads and centred text."""
    p1 = np.array(p1, float); p2 = np.array(p2, float)
    d = p2 - p1
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
    # The label sits ON the dimension line and masks it with its own white
    # box, which is the drafting convention and, unlike the fixed 0.012 offset
    # this used to add, means the same thing on a unit-chord section and on a
    # seventeen-metre span.
    ax.text(mid[0], mid[1], text, color=color, fontsize=fs,
            ha="center", va="center", rotation=rot,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                      alpha=0.9))


def angle_dim(ax, vertex, p_a, p_b, fmt, r=0.4, color=DIM, fs=10,
              expect=None):
    """Angular dimension whose LABEL is the angle the arc subtends.

    `fmt` is formatted with the subtended angle in degrees, so the two cannot
    disagree - which they did: the planform's sweep arc was struck between the
    root CHORD and the leading edge, subtending 78 degrees, and labelled with
    the 12 the wing actually has.  `expect`, when given, is the value the
    caller believes it is dimensioning, and a mismatch stops the build rather
    than drawing a wrong angle.
    """
    v = np.array(vertex, float)
    a0 = np.arctan2(p_a[1]-v[1], p_a[0]-v[0])
    a1 = np.arctan2(p_b[1]-v[1], p_b[0]-v[0])
    ang = abs(np.degrees(a1 - a0))
    ang = min(ang, 360.0 - ang)
    if expect is not None and abs(ang - expect) > 0.5:
        raise ValueError("angular dimension subtends %.2f deg but is being "
                         "used to dimension %.2f: the arc is struck between "
                         "the wrong two rays" % (ang, expect))
    th = np.linspace(a0, a1, 40)
    ax.plot(v[0]+r*np.cos(th), v[1]+r*np.sin(th), color=color, lw=1.0)
    # The label sits at a radius PROPORTIONAL to the arc, not a fixed 0.12
    # further out: the same absolute offset that clears a unit-chord section
    # leaves the text on top of the arc on a seventeen-metre span.
    am = 0.5*(a0+a1)
    rl = r*1.32
    ax.text(v[0]+rl*np.cos(am), v[1]+rl*np.sin(am),
            fmt.format(a=ang), color=color, fontsize=fs, ha="center",
            va="center")


def title_block(ax, title, dwg_no, scale="NTS", view=""):
    """Drafting title block, laid out so the two halves cannot collide.

    The left block and the right block were both written on one row at a fixed
    font size, and on the narrower sheets they overlapped: GEO-002 rendered
    "SECTION UTSS-NLF16" and "DWG GEO-002" on top of each other as
    "UTSS-NLFDWG GEO-002".  The right block is now measured against the left
    and, if it does not fit beside it, split onto its own second row; if it
    still does not fit the whole block is shrunk until it does.
    """
    fig = ax.figure
    ax.set_title(title, fontsize=14, fontweight="normal", color=INK, pad=12)
    fig.subplots_adjust(bottom=0.22, top=0.90)
    left1 = fig.text(0.012, 0.065,
                     f"PROJECT: AETHER-NLF 25  |  NLF WING  |  "
                     f"SECTION {C.WING['section']}",
                     fontsize=10, color=INK_SOFT)
    left2 = fig.text(0.012, 0.030, "DRAWN BY: AKOSA SAMUEL ONYEJEKWE",
                     fontsize=10, color=INK_SOFT)
    rparts = [f"DWG {dwg_no}   SCALE {scale}   {view}".strip(),
              "3rd-ANGLE   UNITS m (noted)   UTSS-CASE-2026"]
    right1 = fig.text(0.988, 0.065, "   ".join(rparts),
                      fontsize=10, color=INK_SOFT, ha="right")
    right2 = None

    def clash(a, b, gap=10.0):
        fig.canvas.draw()
        return a.get_window_extent().x1 + gap > b.get_window_extent().x0

    if clash(left1, right1):
        right1.set_text(rparts[0])
        right2 = fig.text(0.988, 0.030, rparts[1], fontsize=10,
                          color=INK_SOFT, ha="right")
        fs = 10.0
        while fs > 6.5 and (clash(left1, right1) or clash(left2, right2)):
            fs -= 0.5
            for t in (left1, left2, right1, right2):
                t.set_fontsize(fs)
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
    # Rounded to the precision these quantities are meaningful to, the same
    # convention run_solution.py states and every table in the report relies
    # on.  Written raw, this file put seventeen-significant-figure numbers
    # straight into the document - the report carried 202 of them - and its
    # last bits moved with the BLAS thread count, so a regeneration that
    # changed nothing physical still produced a different file.
    df_af = pd.DataFrame({
        "x_c": co["x"].round(6), "y_camber": co["yc"].round(6),
        "half_thickness": co["yt"].round(6),
        "x_upper": co["xu"].round(6), "y_upper": co["yu"].round(6),
        "x_lower": co["xl"].round(6), "y_lower": co["yl"].round(6)})
    df_af.to_csv(f"{GEO}/airfoil_UTSS-NLF16.csv", index=False)

    W = C.WING
    eta = np.linspace(0, 1, 21)
    y = eta*W["span_b"]/2
    chord = W["root_chord"] + eta*(W["tip_chord"]-W["root_chord"])
    x_le = y*np.tan(np.radians(W["le_sweep_deg"]))
    x_te = x_le + chord
    twist = eta*W["twist_tip_deg"]
    z_dih = y*np.tan(np.radians(W["dihedral_deg"]))
    # `+ 0.0` after every round is not decoration: IEEE rounding of a small
    # negative gives NEGATIVE zero, and the root station's twist is exactly
    # -(0 * twist_tip), so wing_planform.csv published "-0.0" for it and the
    # report's planform table printed a negative twist at the root.  Adding
    # positive zero is the defined way to collapse the sign (-0.0 + 0.0 = +0.0)
    # and leaves every other value untouched.
    df_pl = pd.DataFrame({"eta": eta.round(4) + 0.0, "y_m": y.round(4) + 0.0,
                          "chord_m": chord.round(4) + 0.0,
                          "x_le_m": x_le.round(4) + 0.0,
                          "x_te_m": x_te.round(4) + 0.0,
                          "twist_deg": twist.round(3) + 0.0,
                          "z_dihedral_m": z_dih.round(4) + 0.0,
                          "Re_local": (C.CRUISE["U_inf"]*chord
                                       / C.CRUISE["nu_inf"]).round(-2) + 0.0})
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
        # The trailing-edge wedge angle.  It is quoted in the README, the
        # report, the solver and run_solution as "26.8 degrees included,
        # measured off the section" and was measured off the section exactly
        # once, by hand; it is a property of the shape function and belongs
        # with the rest of them.  Squire-Young's premise is what it decides:
        # a wedge trailing edge is an inviscid stagnation point.
        ("Trailing-edge included angle", f"{te_wedge_deg(co):.1f}", "deg"),
        # "2-D unswept" is not decoration.  This is the section on its own; the
        # case study runs it as a strip of a 12-degree swept wing and gets
        # c_l = c_l,n cos^2 L, which is the "Section lift coefficient Cl" row
        # of 04_solution/integrated_forces.csv and is four per cent lower.  The
        # two carried the same name, and the README's headline table then
        # quoted this one beside the swept wing's drag.
        (f"Section c_l, 2-D unswept "
         f"(alpha = {C.CRUISE['alpha_deg']:.1f} deg, M = {C.CRUISE['mach']:.2f})",
         f"{section_cl():.3f}", "-"),
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
    # Leading-edge radius, measured from the section rather than asserted.  The
    # thickness form is y ~ A sqrt(x) at the nose, whose radius of curvature at
    # x = 0 is A^2/2.  The drawing used to call this 0.015 c; it is 0.010 c.
    _m = (x > 1e-6) & (x < 2.0e-3)
    r_le = float(np.mean(co["yt"][_m]/np.sqrt(x[_m])))**2/2.0

    # --- horizontal chord dimension (clear, below everything) ---
    # bottom-most row of the sheet, below the specification box and clear of
    # the legend, which the dimension line used to run straight through
    dim_linear(ax, (0,-0.225),(1,-0.225), -0.045,
               "CHORD  c (reference)   |   MAC = %.3f m" % C.WING["MAC"],
               side=-1, fs=10)
    # --- x(t_max) horizontal dimension (clear, above) ---
    dim_linear(ax, (0,0.205),(xt,0.205), 0.028, f"x(t_max) = {xt:.2f} c", side=1, fs=10)
    # --- t_max vertical thickness arrow + leadered label in clear space ---
    ax.plot([xt,xt],[yl_t,yu_t], color=HID, lw=0.8, ls=(0,(4,3)))
    ax.annotate("", xy=(xt,yu_t), xytext=(xt,yl_t),
                arrowprops=dict(arrowstyle="<->", color=DIM, lw=1.2))
    # leadered clear of the specification box below it, which used to strike
    # the "(16.0 %)" line through
    ax.annotate(f"t_max = {tmax:.3f} c  ({tmax*100:.1f} %)", xy=(xt,yl_t),
                xytext=(xt+0.16,-0.115), color=DIM, fontsize=10, ha="left",
                va="center", arrowprops=dict(arrowstyle="->", color=DIM, lw=0.9))
    # --- leading & trailing edge callouts (clear of geometry) ---
    ax.annotate(f"rounded NLF leading edge\nr_LE = {r_le:.3f} c  ·  favourable"
                "\ngradient to ~0.40 c",
                xy=(0.012,0.004), xytext=(0.15,0.125), color=INK_SOFT, fontsize=10,
                ha="left", arrowprops=dict(arrowstyle="->", color=INK_SOFT))
    # NOT "cusped".  A cusped trailing edge has zero included angle, both
    # surfaces meeting tangentially; this one closes at te_wedge_deg(co) - 26.8
    # degrees - which is a WEDGE, and that fact is the whole reason
    # Squire-Young is evaluated at 0.98c rather than at the trailing edge and
    # the reason H sits on Head's clamp there.  The drawing said the opposite
    # of what the report is about.
    ax.annotate(f"aft-loaded, {te_wedge_deg(co):.1f}° wedge\ntrailing edge",
                xy=(0.985, co["yc"][-3]),
                xytext=(0.80,0.135), color=INK_SOFT, fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK_SOFT))
    # --- specification box (clear lower-left corner) ---
    # Three lines, re-balanced.  Saying "(2-D, unswept)" beside the c_l is
    # necessary - the case study's section lift is the swept strip's, four
    # per cent lower - but adding it to the line that already carried t/c
    # pushed that line under the legend and hid "M = 0.42".  No line here is
    # now longer than the longest one this box had before.
    spec=("UTSS-NLF16  natural-laminar-flow section, aft-loaded camber\n"
          f"t/c = {tmax:.3f} @ {xt:.2f} c   ·   {te_wedge_deg(co):.1f}° wedge T.E.\n"
          f"c_l = {section_cl():.2f} (2-D, unswept) "
          f"at α = {C.CRUISE['alpha_deg']:.1f}°, M = {C.CRUISE['mach']:.2f}")
    ax.text(0.015, -0.222, spec, fontsize=10, color=INK,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef4fa", ec=INK_SOFT, lw=0.9))
    # pinned above the chord dimension rather than in the axes corner, where it
    # sat on top of it
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.14),
              fontsize=10, framealpha=0.9)

    ax.set_xlim(-0.06, 1.10); ax.set_ylim(-0.30, 0.27)
    ax.set_aspect("equal", adjustable="box"); ax.grid(False)
    ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
    title_block(ax, f"AEROFOIL SECTION  {C.WING['section']}  (16% NLF)",
                "GEO-001", "1:1 (norm.)", "SECTION VIEW")
    finish_dwg(fig, f"{DWG}/dwg_01_airfoil_section.png")


def draw_planview(df_pl):
    W = C.WING
    # proportioned to the planform: with an equal aspect on a 10.5 x 8.2 sheet
    # the wing filled a wide band and left the bottom half of the sheet empty
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
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
    # Between the SPAN-WISE direction and the leading edge, which is what
    # leading-edge sweep means.  The first ray used to be the root chord, so
    # the arc subtended the complement, 78 degrees.
    angle_dim(ax, (0, xle[0]), (y[-1], xle[0]), (y[-1], xle[-1]),
              "Λ_LE = {a:.0f}°", r=2.4, expect=W["le_sweep_deg"])
    # moved clear of the sweep-angle label, which the arc's own label used to
    # be written on top of
    ax.annotate("c/4 sweep line", xy=(y[13], xc4[13]), xytext=(5.6, -0.55),
                color=CTR, fontsize=10, arrowprops=dict(arrowstyle="->", color=CTR))
    # section cut marker B-B
    ax.plot([3.0,3.0],[xle[0]-0.2, xte.max()+0.2], color=PALETTE[1], lw=1.2, ls=(0,(2,2)))
    ax.text(3.0, xte.max()+0.35, "B", color=PALETTE[1], ha="center", fontsize=11, fontweight="normal")
    ax.text(3.0, xle[0]-0.4, "B", color=PALETTE[1], ha="center", fontsize=11, fontweight="normal")
    ax.set_xlabel("span-wise  y  [m]"); ax.set_ylabel("stream-wise  x  [m]")
    ax.set_aspect("equal"); ax.grid(False)
    # room below the trailing edge for the data box, which used to sit on the
    # planform outline (y inverted: stream-wise x increases downward)
    ax.set_ylim(xte.max()+1.6, xle[0]-1.7)
    ax.set_title("")
    txt = (f"S = {W['area_S']:.2f} m²   AR = {W['AR']:.2f}   "
           f"λ = {W['taper']:.2f}   MAC = {W['MAC']:.2f} m")
    ax.text(0.5, 0.04, txt, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=INK, bbox=dict(boxstyle="round", fc="#eef4fa", ec=INK_SOFT))
    title_block(ax, "WING PLANFORM  -  PLAN VIEW (TOP)", "GEO-002", "1:120", "PLAN")
    finish_dwg(fig, f"{DWG}/dwg_02_planview.png")


def draw_front_side(df_pl):
    W = C.WING
    y = df_pl["y_m"].values; z = df_pl["z_dihedral_m"].values
    # sized to the two views: at 13.5 x 4.6 with an equal aspect the panels
    # were short bands with empty thirds above and below them
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.9))
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
    # the span dimension sits at z = -0.60; the lower limit has to clear it,
    # or its arrows and label are cut off by the axes frame
    ax.set_ylim(min(z.min(), -0.35) - 0.55, z.max()+1.5)
    ax.set_xlabel("y [m]"); ax.set_ylabel("z [m]")
    ax.set_title("FRONT VIEW (looking aft)  -  dihedral", color=INK, fontsize=11)
    ax.grid(False)
    # SIDE VIEW (root + tip profiles)
    ax = axes[1]
    # The tip is drawn WHERE IT IS: swept back by x_le, raised by the dihedral
    # and rotated by the washout, exactly as the loft in wing_sections_3d.csv
    # places it.  It used to be drawn at z = 0 and unrotated - so a view whose
    # own axis is z, on a sheet whose other panel exists to show the dihedral,
    # put the tip half a metre from where the wing carries it and hid a -3
    # degree twist entirely.
    for e,col,lab in [(0,PALETTE[0],"root"),(1,PALETTE[1],"tip")]:
        sub = df_pl.iloc[0 if e==0 else -1]
        co = C.nlf16_coords(n=80)
        cc = sub["chord_m"]; xle = sub["x_le_m"]
        zz = sub["z_dihedral_m"]; a = np.radians(-sub["twist_deg"])
        def _place(xs, ys):
            xq = xs - 0.25
            xr = 0.25 + xq*np.cos(a) - ys*np.sin(a)
            yr = xq*np.sin(a) + ys*np.cos(a)
            return xle + xr*cc, zz + yr*cc
        _xu, _zu = _place(co["xu"], co["yu"])
        _xl, _zl = _place(co["xl"], co["yl"])
        ax.plot(_xu, _zu, color=col, lw=1.8,
                label=f"{lab} c={cc:.2f}m, z={zz:.2f}m, twist {sub['twist_deg']:.1f}°")
        ax.plot(_xl, _zl, color=col, lw=1.8)
    ax.set_aspect("equal"); ax.invert_xaxis()
    # Head-room for the legend, and it is now much less than it was: with the
    # tip drawn at its own z the two sections already fill three quarters of a
    # metre, and the old 55 per cent of the range added an empty half-metre on
    # top.  The legend goes top-right, which on this inverted axis is small x,
    # high z: the root reaches z = 0.24 and the tip starts at x = 1.86, so that
    # is the one corner both sections leave clear.
    _lo, _hi = ax.get_ylim(); ax.set_ylim(_lo, _hi + 0.12*(_hi-_lo))
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    ax.set_xlabel("x [m] (LE right)"); ax.set_ylabel("z [m]")
    ax.set_title("SIDE VIEW  -  root & tip sections", color=INK, fontsize=11)
    ax.grid(False)
    fig.suptitle("WING ORTHOGRAPHIC VIEWS  (GEO-003)", color=INK, fontweight="normal")
    # Through finish()'s caption slot rather than as a bare fig.text at y=0.005:
    # tight_layout does not know about a free-floating text, so when the side
    # panel grew to hold the tip at its true height the x-axis label came down
    # on top of the title block.  The caption path reserves the strip first.
    finish(fig, f"{DWG}/dwg_03_front_side.png",
           caption="DRAWN BY: AKOSA SAMUEL ONYEJEKWE  |  PROJECT AETHER-NLF 25"
                   "  |  UTSS-CASE-2026")


def draw_orthographic(df_pl):
    """Single 3rd-angle sheet: plan + front + side + iso inset."""
    W = C.WING
    # Sized to the panels, not to a square.  Every view here is set to an
    # EQUAL data aspect, and the plan view is 17 m by 4 - a 4.3:1 box - so on a
    # 13.5 x 9.5 sheet each axes shrank to a fifth of the height of the cell it
    # was given and roughly half the drawing was white.  The row heights now
    # follow the two rows' data aspects.
    # 13.5 x 7.0 with hspace 0.42 still left a band of white between the rows
    # as wide as the panels themselves, because every panel here is set to an
    # EQUAL data aspect and so does not fill the cell it is given vertically.
    # Shortening the sheet takes the band out without touching the panels.
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1,1], width_ratios=[1.5,1],
                          hspace=0.30, wspace=0.22)
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
    # zoomed to fill its cell: a 3-D axes leaves a wide margin round its own
    # box, and the inset was drawn at about a third of the cell it was given
    _iso_wing(axi, df_pl, zoom=1.45)
    # pictorial inset: drop tick numbers (they collide with axis labels)
    axi.set_xticklabels([]); axi.set_yticklabels([]); axi.set_zticklabels([])
    axi.set_title("ISOMETRIC", fontsize=10, color=INK)
    fig.suptitle("AETHER-NLF 25  WING  -  ORTHOGRAPHIC PROJECTION (3rd ANGLE)   DWG GEO-004",
                 fontsize=13, fontweight="normal", color=INK)
    fig.text(0.5, 0.01, "All dimensions in metres unless noted  |  Scale 1:120  |  "
             "UTSS-CASE-2026  |  DRAWN BY: AKOSA SAMUEL ONYEJEKWE",
             ha="center", color=INK_SOFT, fontsize=10)
    # Explicit margins, not tight_layout.  A 3-D axes is not compatible with it,
    # so the call warned on every build and then did nothing - the sheet was
    # laid out by the figure's default subplot parameters all along.  Those
    # defaults are written down here instead, which produces the identical
    # sheet without the warning and without pretending the layout is automatic.
    fig.subplots_adjust(left=0.125, right=0.90, bottom=0.11, top=0.88)
    fig.savefig(f"{DWG}/dwg_04_orthographic.png", dpi=170, facecolor="white")
    plt.close(fig)


def _iso_wing(ax, df_pl, zoom=1.0, nticks=None):
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
    # lower surface first: plot_surface paints in call order rather than by
    # depth, so drawing the lower one last covered the upper one and the whole
    # wing read as the teal of the underside from a viewpoint above it
    for surf,col in [(Ls,PALETTE[2]),(Us,PALETTE[0])]:
        X=np.array([s[0] for s in surf]); Y=np.array([s[1] for s in surf]); Z=np.array([s[2] for s in surf])
        ax.plot_surface(X,Y,Z,color=col,alpha=0.55,linewidth=0,antialiased=True,shade=True)
    ax.set_xlabel("x",fontsize=10); ax.set_ylabel("y",fontsize=10); ax.set_zlabel("z",fontsize=10)
    ax.view_init(elev=22, azim=-58)
    # `zoom` fills the sheet.  A 3-D axes leaves a wide margin round its own
    # bounding box on top of whatever the subplot leaves, so the isometric
    # sheet had the wing occupying about a third of it and the rest white.
    box_aspect(ax,(3,6,1),zoom=zoom)
    if nticks:
        from matplotlib.ticker import MaxNLocator
        # the x labels run along a steeply foreshortened axis and ran into one
        # another at the default count
        ax.xaxis.set_major_locator(MaxNLocator(nticks))
        ax.yaxis.set_major_locator(MaxNLocator(nticks))
    ax.grid(False)


def draw_isometric(df_pl):
    # 10 x 7.5 left the wing in a band across the middle with the top and
    # bottom thirds empty, and the single-line caption ran off both edges of
    # the sheet; the caption is now two lines and the view is raised so the
    # UPPER surface it names is the one facing the reader
    fig = plt.figure(figsize=(10,6.2))
    # add_axes, not add_subplot: the default subplot box leaves a further
    # margin inside the sheet on top of the 3-D axes' own, and between the two
    # the wing was drawn at about a third of the area available to it.
    ax = fig.add_axes([0.01, 0.09, 0.98, 0.85], projection="3d")
    _iso_wing(ax, df_pl, zoom=1.30, nticks=4)
    ax.view_init(elev=26, azim=-58)
    ax.set_title("AETHER-NLF 25 WING  -  ISOMETRIC VIEW  (DWG GEO-005)",
                 color=INK, fontweight="normal")
    fig.text(0.5,0.03,"Upper surface (blue)  /  Lower surface (teal)  -  lofted from "
             f"{C.WING['section']} sections  |  half-span shown\n"
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
    def _surface(xs, ys):
        """Interpolator for one surface, on a sorted, de-duplicated abscissa.

        np.interp requires an increasing xp and gives no warning when it does
        not get one.  The lower surface was passed xl[::-1], which is
        DECREASING, so yL returned 0.0 at every station: the inner skin line,
        both lower spar caps and the whole lower stringer row were drawn flat
        along the chord line instead of on the surface.  The upper surface is
        not monotone either - near the nose the thickness term moves points
        back past their neighbours - so both are sorted here.
        """
        k = np.argsort(xs); a, b = np.asarray(xs)[k], np.asarray(ys)[k]
        keep = np.concatenate([[True], np.diff(a) > 0])
        a, b = a[keep], b[keep]
        return lambda xq: np.interp(xq, a, b)
    yU = _surface(xu, yu)
    yL = _surface(xl, yl)

    # ---- outer skin (OML) ----
    ax.plot(xu,yu,color=OUT,lw=2.4); ax.plot(xl,yl,color=OUT,lw=2.4)
    # ---- inner skin line (constant skin thickness) -> skin band w/ hatch ----
    # A FRACTION OF CHORD, which is what the drawing draws.  Its annotation
    # read "t ~ 12 mm", as though the 0.012 were metres: at this station the
    # band drawn is 0.012 x 2.141 m = 25.7 mm, so the label and the geometry
    # disagreed by a factor of two.  The label is derived from this line now
    # and gives both forms, so the two cannot part company again.
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
    x_fs, x_rs = spars[0][0], spars[1][0]      # box width follows the spars
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
    ax.plot([0,0],[ -0.05*chord, 0.38*chord], color=CTR, lw=0.8, ls=(0,(6,3)))
    dim_linear(ax,(0,0.24*chord),(x_fs*chord,0.24*chord),0.03*chord,
               f"{x_fs:.2f} c", side=1, fs=10)
    dim_linear(ax,(0,0.34*chord),(x_rs*chord,0.34*chord),0.03*chord,
               f"{x_rs:.2f} c", side=1, fs=10)
    dim_linear(ax,(0,-0.28*chord),(chord,-0.28*chord),-0.05*chord,
               f"CHORD  c (y = {yc:.1f} m) = {chord:.3f} m", side=-1, fs=10)
    dim_linear(ax,(x_fs*chord,-0.11*chord),(x_rs*chord,-0.11*chord),-0.04*chord,
               f"integral wing box = {(x_rs-x_fs)*chord:.3f} m", side=-1, fs=10)

    # ---- material callouts, each in its own band of clear head-room --------
    # These sit in DATA coordinates, so they have to stay inside the limits set
    # below; at 0.60c they were pushed outside the frame and printed across the
    # drawing title.
    ax.annotate("FRONT SPAR\n(Al-Li web + caps)", xy=(0.20*chord, yU(0.20*chord)-tsk),
                xytext=(0.10*chord, 0.44*chord), color=PALETTE[1], fontsize=10,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.9))
    ax.annotate(f"CFRP skin\n(t = {tsk/chord:.3f} c = {tsk*1e3:.0f} mm)",
                xy=(0.42*chord, yU(0.42*chord)),
                xytext=(0.46*chord, 0.30*chord), color=PALETTE[0], fontsize=10,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=0.9))
    ax.annotate("REAR SPAR\n(Al-Li web + caps)", xy=(0.65*chord, yU(0.65*chord)-tsk),
                xytext=(0.84*chord, 0.44*chord), color=PALETTE[1], fontsize=10,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=0.9))
    # leadered below the section: above it there is no band left, and at
    # -0.34c it used to sit on the CHORD dimension text
    ax.annotate("stringers (Z-section)", xy=(0.82*chord, yL(0.82*chord)+tsk),
                xytext=(0.62*chord,-0.20*chord), color=PALETTE[4], fontsize=10,
                ha="center", arrowprops=dict(arrowstyle="->", color=PALETTE[4], lw=0.9))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.16, chord+0.16); ax.set_ylim(-0.40*chord, 0.52*chord)
    ax.set_xlabel("x [m]  (chord-wise)"); ax.set_ylabel("z [m]")
    title_block(ax,f"WING STRUCTURAL SECTION  B-B  (y = {yc:.1f} m)",
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
