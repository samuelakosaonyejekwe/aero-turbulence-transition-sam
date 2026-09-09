"""
build_docx.py
Assemble the full industrial case-study report -> case.docx
Embeds: narrative, all governing equations, every engineering drawing,
all CSV data tables, all plots/curves, contours, profiles, 3D contours
and vectors, validation comparisons, calibration record and sources.

No black: body text and headings use navy ink; tables use accent borders.
"""
import math
import os
import re

import utss_paths  # noqa: F401  - anchors the repo root and solver/ on
                   # sys.path, so this script works from any directory.  It has
                   # to be imported BEFORE anything reads a relative path: this
                   # sat forty lines below the equations_index.csv read, so the
                   # script still died with FileNotFoundError when run from
                   # anywhere but the repository root - which is the one thing
                   # the import is there to prevent.
import case_config as _C
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
import latex2mathml.converter as _L
import mathml2omml as _M

_MATHNS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def _omml_element(latex):
    mml = _L.convert(latex)
    omml = _M.convert(mml)
    if "xmlns:m" not in omml:
        omml = omml.replace("<m:oMath",
                            '<m:oMath xmlns:m="%s"' % _MATHNS, 1)
    return parse_xml(omml)

INK   = RGBColor(0x1d,0x2f,0x45)
BLUE  = RGBColor(0x1b,0x6c,0xa8)
ROSE  = RGBColor(0xd1,0x49,0x5b)
GREEN = RGBColor(0x2a,0x9d,0x8f)

doc = Document()

# ---- base styles (navy text, never black) ----
st = doc.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(11); st.font.color.rgb = INK
for h,sz in [("Title",26),("Heading 1",17),("Heading 2",13.5),("Heading 3",11.5)]:
    s=doc.styles[h]; s.font.color.rgb = BLUE if h!="Title" else INK
    s.font.size=Pt(sz); s.font.name="Calibri"

EQD="07_equations"; eq_index=pd.read_csv(f"{EQD}/equations_index.csv").set_index("key")

C_CLIMB_TU = _C.CLIMB["Tu_pct"]

# ---- page footer with author on every page (no black) ----
foot=doc.sections[0].footer.paragraphs[0]
foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
fr=foot.add_run("Akosa Samuel Onyejekwe  ·  UTSS Universal Transition Solver  ·  "
                "Case Study UTSS-CASE-2026")
fr.font.size=Pt(8.5); fr.font.color.rgb=BLUE

# ----------------------------------------------------------------------
def h1(t): doc.add_heading(t, level=1)
def h2(t): doc.add_heading(t, level=2)

def para(t, italic=False, bold=False, size=11):
    p=doc.add_paragraph(); r=p.add_run(t); r.italic=italic; r.bold=bold
    r.font.size=Pt(size); r.font.color.rgb=INK; return p

def bullet(t):
    p=doc.add_paragraph(style="List Bullet"); r=p.add_run(t); r.font.color.rgb=INK
    return p

# ----------------------------------------------------------------------
#  Figure and table numbering
#
#  These were written by hand, and hand numbering drifts as sections are
#  inserted: the report reached the point of presenting "Fig. 8a" and "Fig. 8b"
#  in section 6 before "Fig. 1" in section 7, and "Table 20a" after "Table 24".
#  Numbers are now allocated in the order the captions are emitted, so they
#  cannot fall out of order, and cross-references in the body are written as
#  tokens that are substituted once every number is known.  A reference to
#  Fig. 9 OF ANOTHER PAPER is ordinary text and is left alone, which hand
#  renumbering would not have been safe with.
_FIG = [0]; _TAB = [0]; _REF = {}

def _fignum(key=None):
    _FIG[0] += 1
    if key: _REF["FIG:"+key] = "Fig. %d" % _FIG[0]
    return _FIG[0]

def _tabnum(key=None):
    _TAB[0] += 1
    if key: _REF["TAB:"+key] = "Table %d" % _TAB[0]
    return _TAB[0]

_UNRESOLVED = set()


def resolve_refs(document):
    """Substitute @@KIND:key@@ tokens once all numbers are allocated.

    An unknown key used to be substituted by ITSELF, so a mistyped reference
    left a bare "TAB:cf_forms" in the rendered report rather than a number, and
    nothing anywhere failed: the token had gone, so a search for "@@" found
    nothing and the sentence read as though a table had been named.  Unknown
    keys are collected and the build refuses to save.
    """
    def _sub(m):
        k = m.group(1)
        if k not in _REF:
            _UNRESOLVED.add(k)
        return _REF.get(k, k)

    def fix(par):
        if "@@" not in par.text: return
        for r in par.runs:
            if "@@" in r.text:
                r.text = re.sub(r"@@([A-Z]+:[a-z0-9_]+)@@", _sub, r.text)
        if "@@" in par.text:            # split across runs: rebuild in run 0
            t = re.sub(r"@@([A-Z]+:[a-z0-9_]+)@@", _sub, par.text)
            par.runs[0].text = t
            for e in par.runs[1:]: e.text = ""
    for par in document.paragraphs: fix(par)
    for t in document.tables:
        for row in t.rows:
            for c in row.cells:
                for par in c.paragraphs: fix(par)


def caption(t):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(t); r.italic=True; r.font.size=Pt(9); r.font.color.rgb=BLUE
    return p

def image(path, width=6.3, cap=None, key=None):
    if not os.path.exists(path):
        para(f"[missing figure: {path}]", italic=True); return
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    if cap: caption("Fig. %d. %s" % (_fignum(key), cap))

_EQ_PLACED = set()
_EQ_HEAD_USED = set()


def _eq_section(prefix):
    """The chapter-4 heading, taken from the equations index that names it.

    build_docx and gen_equations each carried their own wording for these five
    sections - "4.1 Inviscid panel flow" here against "4.1 Inviscid edge
    solution" there - so model.equations.docx and this report gave the same
    five sections different names, and 4.2 was called "(Thwaites)" while the
    shipped closure is the two-equation march that solver_settings.csv
    describes.  One source now, and a missing one stops the build.
    """
    hits = sorted({str(x) for x in eq_index["section"]
                   if str(x).startswith(prefix)})
    if len(hits) != 1:
        raise SystemExit("equations_index.csv has %d sections beginning %r: %s"
                         % (len(hits), prefix, hits))
    _EQ_HEAD_USED.add(hits[0])
    return hits[0]


def equation(key, show_title=True):
    if key not in eq_index.index:
        para(f"[missing eq {key}]"); return
    _EQ_PLACED.add(key)
    row=eq_index.loc[key]
    # caption line first: (key) descriptive title
    if show_title:
        c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
        r=c.add_run(f"({key})  {row['equation']}"); r.italic=True
        r.font.size=Pt(9); r.font.color.rgb=BLUE
        c.paragraph_format.space_after=Pt(2)
    # native, editable Word equation (LaTeX -> OMML)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(8)
    try:
        p._p.append(_omml_element(row["latex"]))
    except Exception as e:
        fr=p.add_run(row["latex"]); fr.font.color.rgb=INK
        print("  ! eq fallback", key, e)

def table_from_csv(path, max_rows=40, ncols=None, cap=None, sample=False, key=None):
    if not os.path.exists(path):
        para(f"[missing CSV: {path}]", italic=True); return
    # read as text: the CSVs carry their own formatting (Re_x written as
    # 1.785e+05, values already rounded to the precision they are claimed to),
    # and re-parsing them as floats renders 1.785e+05 as 178500.0
    df=pd.read_csv(path, dtype=str)
    n_full=len(df)
    if ncols: df=df.iloc[:,:ncols]
    if len(df)>max_rows:
        if sample:
            idx=list(range(0,len(df),max(1,len(df)//max_rows)))[:max_rows]
            df=df.iloc[idx]
        else:
            df=df.head(max_rows)
        truncated=True
    else: truncated=False
    add_table(df, cap, key=key)
    if truncated:
        # "sampled" and "truncated" are not the same thing and the note used to
        # call both of them sampled: sample=True takes an even spread across
        # the whole file, sample=False keeps the FIRST max_rows and drops the
        # tail, which a reader needs to know before concluding anything from
        # the last row shown.
        how = ("sampled evenly to %d of %d rows" % (len(df), n_full) if sample
               else "truncated to the first %d of %d rows" % (len(df), n_full))
        para(f"(table {how}; full data in {path})", italic=True, size=8.5)

MAX_TABLE_COLS = 7      # what stays legible across a portrait text column

def _one_table(df, size):
    t=doc.add_table(rows=1, cols=len(df.columns))
    try: t.style="Light Grid Accent 1"
    except Exception: t.style="Table Grid"
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    hdr=t.rows[0].cells
    for j,c in enumerate(df.columns):
        # a zero-width space after each underscore gives Word somewhere to
        # break a header like Re_theta_t_err_pct; without one it breaks
        # mid-token and the header reads as rubble
        hdr[j].text=str(c).replace("_", "_\u200b")
        for pp in hdr[j].paragraphs:
            for r in pp.runs: r.font.bold=True; r.font.size=Pt(size); r.font.color.rgb=INK
    for _,rowv in df.iterrows():
        cells=t.add_row().cells
        for j,v in enumerate(rowv):
            cells[j].text=str(v)
            for pp in cells[j].paragraphs:
                for r in pp.runs: r.font.size=Pt(size); r.font.color.rgb=INK
    return t


def add_table(df, cap=None, max_cols=MAX_TABLE_COLS, key=None):
    """Render a table, splitting a wide one into legible column blocks.

    Word autofits a table to the text width, so a fifteen-column CSV dropped
    straight in gets under a centimetre per column and breaks every cell into
    two- and three-character fragments.  Table 20 rendered "ERCOFTAC" as
    "ER/CO/FT/AC" down four lines and the predicted Re_theta_t of 271.4 as
    "271./4" - the validation summary, the aerofoil point-by-point comparison
    and the two surface-state tables were all unreadable in the compiled
    report.  Beyond max_cols the table is therefore split into blocks that each
    repeat the first column, the one that identifies the row.
    """
    df=df.fillna("")
    n = _tabnum(key) if cap else None
    cap = ("Table %d. %s" % (n, cap)) if cap else None
    cols=list(df.columns)
    if len(cols) <= max_cols:
        _one_table(df, 8.5)
        if cap: caption(cap)
        return
    first, rest = cols[0], cols[1:]
    per = max_cols - 1
    blocks = [rest[i:i+per] for i in range(0, len(rest), per)]
    for k, blk in enumerate(blocks):
        _one_table(df[[first]+blk], 8.0)
        tag = "columns %d-%d of %d" % (2+k*per, 1+k*per+len(blk), len(cols))
        if k < len(blocks)-1:
            caption("(%s; %s repeated on each block)" % (tag, first))
            doc.add_paragraph()
        elif cap:
            caption("%s  (%s; the table is split across %d blocks so that no "
                    "cell is compressed to illegibility, with %s repeated on "
                    "each)" % (cap, tag, len(blocks), first))
        else:
            caption("(%s)" % tag)

def manual_table(headers, rows, cap=None, key=None):
    add_table(pd.DataFrame(rows, columns=headers), cap, key=key)

# ======================================================================
#  TITLE PAGE
# ======================================================================
t=doc.add_paragraph(); t.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=t.add_run("INDUSTRIAL CASE STUDY"); r.bold=True; r.font.size=Pt(15); r.font.color.rgb=GREEN
ttl=doc.add_paragraph(); ttl.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=ttl.add_run("Prediction of Boundary-Layer Turbulence Transition\nover Aircraft Surfaces")
r.bold=True; r.font.size=Pt(24); r.font.color.rgb=INK
s=doc.add_paragraph(); s.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=s.add_run("using the UTSS Universal Transition & Skin-Friction Solver")
r.font.size=Pt(15); r.font.color.rgb=BLUE
doc.add_paragraph()
image("01_geometry/drawings/dwg_05_isometric.png", width=5.6)
sub=doc.add_paragraph(); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=sub.add_run("Case vehicle: AETHER-NLF 25 regional natural-laminar-flow demonstrator\n"
              "Wing section UTSS-NLF16  ·  Cruise FL360, M0.42  ·  3-D swept tapered wing")
r.font.size=Pt(12); r.font.color.rgb=INK
doc.add_paragraph()
auth=doc.add_paragraph(); auth.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=auth.add_run("Prepared by"); r.font.size=Pt(11); r.italic=True; r.font.color.rgb=BLUE
auth2=doc.add_paragraph(); auth2.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=auth2.add_run("AKOSA SAMUEL ONYEJEKWE"); r.bold=True; r.font.size=Pt(15); r.font.color.rgb=INK
meta=doc.add_paragraph(); meta.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=meta.add_run("Document UTSS-CASE-2026  ·  Three-dimensional analysis  ·  "
               "All units SI unless noted"); r.font.size=Pt(10); r.italic=True
r.font.color.rgb=BLUE
doc.add_page_break()

# ======================================================================
h1("1.  Executive Summary")
para("This case study demonstrates the prediction of laminar-to-turbulent boundary-layer "
 "transition over the surfaces of an aircraft wing, and the engineering quantities that "
 "depend on it (skin-friction drag, laminar-flow extent, boundary-layer growth and "
 "trailing-edge separation margin). The analysis vehicle is the AETHER-NLF 25, a regional "
 "natural-laminar-flow (NLF) demonstrator whose three-dimensional swept, tapered and twisted "
 "wing uses the purpose-designed UTSS-NLF16 section. Predictions are produced by the UTSS "
 "(Universal Transition & Skin-friction Solver), a novel, fast, robust engine that couples a "
 "vortex-panel inviscid solution to an integral boundary-layer marcher driven by a single, "
 "unified four-mechanism transition kernel.")
para("Key results at the cruise design point (FL360, M=0.42, Re_MAC ≈ 6.4×10⁶):", bold=True)


def _headline():
    """Read Table 1 straight out of the generated CSVs.

    These numbers were typed in by hand in an earlier version of this script
    and had drifted from the solution they describe.  They are now derived, so
    the summary cannot disagree with section 9.
    """
    ts = pd.read_csv("04_solution/transition_summary.csv")
    nvt = pd.read_csv("04_solution/nlf_vs_turbulent.csv")
    frc = pd.read_csv("04_solution/integrated_forces.csv").set_index("quantity")

    def row(case, surf):
        r = ts[(ts.case == case) & (ts.surface == surf)].iloc[0]
        return float(r.x_tr_c), str(r.mechanism)
    xu, mu = row("CRUISE", "upper")
    xl, ml = row("CRUISE", "lower")
    xcu, mcu = row("CLIMB", "upper")
    lam_pct = float(nvt.mean_laminar_pct.iloc[0])
    cd_nlf = float(nvt.Cd_counts.iloc[0])
    saving = float(nvt.viscous_drag_reduction_pct.iloc[0])
    cl = float(frc.loc["Section lift coefficient Cl", "value"])
    return [["Upper-surface transition x_tr/c", f"{xu:.3f} ({mu})"],
            ["Lower-surface transition x_tr/c", f"{xl:.3f} ({ml})"],
            ["Mean laminar-flow extent", f"{lam_pct:.1f} % of chord"],
            ["Section lift coefficient C_l", f"{cl:.3f}"],
            ["Section profile drag C_d", f"{cd_nlf:.1f} counts"],
            ["Viscous drag reduction vs fully-turbulent", f"{saving:.1f} %"],
            [f"Climb (Tu={C_CLIMB_TU:g} %) transition x_tr/c (upper)",
             f"{xcu:.3f} ({mcu})"]]


manual_table(["Quantity","Predicted value"], _headline(),
 cap="Headline predictions, read from the generated solution CSVs.")
para("Validated against eight independent, credible published datasets with one "
 "universal calibration set — four ERCOFTAC flat plates (case 020), the Schubauer-Skramstad "
 "plate, the NLF(1)-0416 aerofoil at 86 conditions, and two swept wings — the solver "
 "reproduces the transition-onset Reynolds number Re_θt and the measured transition location "
 "without per-case re-tuning of the physics. All four criteria of the kernel are selected "
 "somewhere in this study and each is supported by measurement. The remainder of this report sets "
 "out the background, problem, governing equations, the complete input dataset, every generated "
 "engineering output (CSVs, curves, metrics, contours, temperature profiles, 3-D contours and "
 "vectors), the validation and calibration record with all sources, and the contribution to knowledge.")

# ======================================================================
h1("2.  Background")
para("Skin-friction drag is one of the largest components of total aircraft drag — typically "
 "45–50 % of cruise drag for a transport aircraft. A turbulent boundary layer produces several "
 "times the skin friction of a laminar one at the same Reynolds number. Consequently, extending "
 "the laminar run over wing, nacelle and empennage surfaces (natural-laminar-flow, NLF, and "
 "hybrid-laminar-flow control, HLFC) is among the highest-leverage technologies for fuel-burn "
 "and emissions reduction. The enabling capability is the ability to PREDICT, reliably and "
 "cheaply, WHERE the boundary layer transitions from laminar to turbulent over a 3-D surface, "
 "across the flight envelope.")
para("Transition is not a single phenomenon. Over aircraft surfaces it is driven by several, "
 "often co-resident, physical mechanisms:")
bullet("Natural / Tollmien–Schlichting (TS): amplification of viscous instability waves in "
       "low-disturbance free flight, predicted here by an e^N integral carried per physical "
       "frequency, with the spatial growth rates read from a tabulated solution of the "
       "Orr-Sommerfeld problem rather than from an envelope correlation.")
bullet("Bypass: free-stream-turbulence-induced transition that skips the TS route — dominant in "
       "high-turbulence environments (climb through cloud, turbomachinery-like inflow).")
bullet("Laminar-separation-induced: a laminar separation bubble that reattaches turbulent. "
       "The bubble is closed explicitly here — the shear layer is carried across the dead-air "
       "region by the momentum integral and reattachment placed where it has amplified by the "
       "same critical factor used elsewhere — so a length, and not merely a separation point, "
       "is predicted.")
bullet("Cross-flow: three-dimensional instability of the cross-flow velocity profile on swept "
       "wings — the principal NLF-limiter at moderate-to-high sweep.")
para("A practical solver for aircraft design must capture all of these with a SINGLE, "
 "consistent formulation and calibration, run in seconds for thousands of design iterations, "
 "and remain robust across the operating envelope. Existing tools each address part of the "
 "problem (Section 5). UTSS unifies them.")

# ======================================================================
h1("3.  Problem Statement and Solution Approach")
h2("3.1  Problem statement")
para("Given a three-dimensional wing geometry and a flight condition, predict — quickly, "
 "robustly and accurately — the chordwise and span-wise location of boundary-layer transition "
 "on both surfaces, the governing transition mechanism at each station, and the resulting "
 "boundary-layer state (skin friction C_f, momentum thickness θ, shape factor H, intermittency "
 "γ), and from these the laminar-flow extent and the viscous (profile) drag. The method must be "
 "universal: one set of physics and calibration constants valid for natural, bypass, "
 "separation-induced and cross-flow transition across the Reynolds- and turbulence-number range "
 "of real aircraft surfaces.")
h2("3.2  How we solve it")
para("UTSS solves the problem in a layered, fully-coupled manner:")
bullet("Inviscid edge flow: a constant-strength vortex-panel method (Kuethe–Chow) returns the "
       "surface pressure C_p and edge velocity U_e, with a Karman-Tsien correction applied to "
       "C_p and the integrated loads. It returns a stagnation pressure coefficient of 1.048 at "
       "M = 0.42 against the exact isentropic 1.045, where the linearised Prandtl-Glauert "
       "scaling gives 1.102, and the two agree to better than half a per cent below M = 0.2.")
bullet("Laminar boundary layer: the momentum AND kinetic-energy integral equations are advanced "
       "together from the stagnation point along each surface, so the shape factor is a solved "
       "variable carrying its own history rather than a local function of the pressure gradient. "
       "The three closure functions - H*(H), Re_theta*Cf/2 and Re_theta*C_D - are properties of "
       "the Falkner-Skan family, computed from it and not fitted; every similarity solution is an "
       "exact fixed point of the second equation, and the march returns H = 2.5914 on a flat plate "
       "against the Blasius 2.5913. Fluid properties are evaluated at Eckert's reference "
       "temperature so that the incompressible closures return the compressible skin friction.")
bullet("Unified transition kernel (the novel core): at every station each of the four "
       "mechanisms reports how far through its own onset criterion the layer has got, as a "
       "progress that reaches unity at onset and carries a calibration weight, and the kernel "
       "fires at the first station where any of them completes. This paragraph used to describe "
       "the kernel as a minimum over four onset Reynolds numbers, which is the form Section 4.3 "
       "sets out at length as the one this replaced: only the bypass branch produces such a "
       "Reynolds number.")
bullet("Transitional region: a Narasimha universal-intermittency closure blends laminar and "
       "turbulent properties through γ.")
bullet("Turbulent boundary layer: Head's entrainment method with the Ludwieg–Tillmann "
       "skin-friction law advances the turbulent BL to the trailing edge.")
bullet("Integration: the Squire–Young formula returns the chordwise profile drag, which is "
       "carried into the streamwise frame together with the span-wise wall shear the "
       "span-wise momentum integral supplies (Eq. E20b-E20c); a span-wise strip sweep with "
       "the cross-flow mechanism extends the solution to the full 3-D wing.")
# The tabulated neutral point is computed, not typed: this paragraph read
# "resolves it to the nearest node of its Reynolds-number grid, Re_theta = 210",
# and 210 is neither the value (208) nor a node (RET_GRID goes 204.21 -> 233.92
# with nothing between).  The paragraph reads it from the function; only this
# note carried a figure, and it had drifted from 208 to "209" already.
import stability as _stab_doc
_NEUT, _NEUT_LO, _NEUT_HI = _stab_doc.tabulated_neutral_Re_theta()
_NEUT_SOLVER = _stab_doc.neutral_Re_theta()
_bls = pd.read_csv("06_validation/bubble_length_scaling.csv")
para("Two elements of the formulation are not correlations. The amplification rate driving the "
 "e^N integral is tabulated in solver/amplification_db.npz from 61,600 Orr-Sommerfeld "
 "eigenvalue solutions on the Falkner-Skan family, continued past separation onto its reverse-flow branch: profiles are generated by continuation, the "
 "eigenvalue problem is solved by Chebyshev collocation, and the spatial growth rate is "
 "recovered from the temporal one through Gaster's transformation. The stability solver returns "
 "the Blasius neutral point at Re_theta = %.0f against the accepted 200.5, and reproduces "
 "the standard Blasius Orr-Sommerfeld eigenvalue - c = 0.36412 + 0.00796i at "
 "Re_delta* = 998 - to better than a tenth of a per cent in the growth rate. What the march "
 "reads is the interpolated table, which first amplifies at Re_theta = %.0f, between "
 "Reynolds-number nodes at %.0f and %.0f, so that offset is the resolution of that grid and "
 "not an error in the eigenvalue solver - a shape factor of "
 "2.59129 and a wall shear parameter f''(0) = 0.46960. At run time the cost is a table lookup, "
 % (_NEUT_SOLVER, _NEUT, _NEUT_LO, _NEUT_HI) +
 "so the sub-second solution is preserved. The second is the separation-bubble closure "
 "described above, whose length scales with the disturbance environment. That is measured "
 "rather than asserted (@@TAB:bubble_len@@): %.0f momentum thicknesses on the separating plate at "
 "%.2f %% free-stream turbulence against a median of %.0f over the %d aerofoil bubbles at "
 "%.2f %%, a spread of %.1f that no fixed multiple of θ_s reproduces. Three places in this "
 "project previously quoted that spread, at four and a half, five and a half and from two "
 "different pairs of lengths; none of them was what the solver returns."
 % (_bls.median_len_theta_s.iloc[0], _bls.Tu_pct.iloc[0],
    _bls.median_len_theta_s.iloc[1], int(_bls.n_bubbles.iloc[1]),
    _bls.Tu_pct.iloc[1], _bls.ratio_to_T3C4.iloc[1]))

# ======================================================================
h1("4.  The UTSS Universal Solver — Governing Equations")
para("All equations implemented in the solver are listed below as native, editable Word "
 "equations at standard size. They constitute the complete mathematical definition of the "
 "method, and are also collected in the companion file model.equations.docx.")
h2(_eq_section("4.1"))
for k in ["E02","E03","E01","E04"]: equation(k)
h2(_eq_section("4.2"))
for k in ["E05","E06","E06b","E06c","E07"]: equation(k)


def _narrative_probes():
    """The solves that back statements in the narrative, run rather than recalled.

    Three of them: the 16 deg / Re_c = 2e5 example outside the incidence
    envelope, at three panel counts because the point being made is that the
    quantity has stopped meaning anything and a single value would read as a
    result; the same section solved unswept, which is what says whether the
    corrected swept-drag conversion leaves a 12 deg wing where a 12 deg wing
    should be; and the natural/bypass step the blend of Eq. E14 removes, which
    was quoted from an edition of the solver that no longer exists.
    """
    import case_config as _CC
    from utss_solver import solve_airfoil as _sa, _swept_drag_factor as _sdf
    _W, _cr = _CC.WING, _CC.CRUISE
    _U = 2.0e5*_cr["nu_inf"]/_W["MAC"]
    out = {}
    for tag, npan in (("te160", 80), ("te260", 130), ("te360", 180)):
        _X, _Y = _CC.nlf16_panel_points(npan)
        _r = _sa(_X, _Y, 16.0, _U, _cr["nu_inf"], _W["MAC"], _cr["Tu_pct"],
                 sweep_deg=_W["le_sweep_deg"], mach=_cr["mach"])
        out[tag] = float(_r["theta_te_c"])
        if tag == "te260":
            # the C_d the thin-layer guard suppresses, which is the number the
            # sentence is about
            out["cd"] = sum(
                2.0*_s["theta_te_c"]*_sdf(_s["Ue_te_ratio"],
                                          _s["H_te_squire_young"],
                                          _W["le_sweep_deg"], swept=True)
                for _s in _r["surfaces"].values())
    _X, _Y = _CC.nlf16_panel_points(130)
    _kw = dict(mach=_cr["mach"], T_inf_K=_cr["T_inf_K"])
    _u = _sa(_X, _Y, _cr["alpha_deg"], _cr["U_inf"], _cr["nu_inf"], _W["MAC"],
             _cr["Tu_pct"], sweep_deg=0.0, **_kw)
    _s12 = _sa(_X, _Y, _cr["alpha_deg"], _cr["U_inf"], _cr["nu_inf"], _W["MAC"],
               _cr["Tu_pct"], sweep_deg=_W["le_sweep_deg"], **_kw)
    out["cd_unswept"] = float(_u["Cd"])*1e4
    out["cd_swept"] = float(_s12["Cd"])*1e4
    # The step the natural/bypass BLEND removes, measured on the shipped solver
    # instead of quoted from the version that still had the hard gate.  Forcing
    # each closure alone at the old gate value, Tu = 0.1 %, is what a switch
    # there did to the answer.  The pair used to be typed here, in the README
    # and in two solver comments as x_tr/c = 0.542 against 0.373 and "a third
    # of the profile drag", and none of the three reproduces on this solver -
    # the swept-drag formulation and the anchor have both moved since.
    for _tag, _c in (("nat", dict(Tu_BP_lo=1e9, Tu_BP_hi=2e9)),
                     ("byp", dict(Tu_BP_lo=0.0, Tu_BP_hi=1e-9))):
        _rr = _sa(_X, _Y, _cr["alpha_deg"], _cr["U_inf"], _cr["nu_inf"],
                  _W["MAC"], 0.1, sweep_deg=_W["le_sweep_deg"], cal=_c, **_kw)
        out["step_xtr_" + _tag] = float(_rr["surfaces"]["upper"]["x_tr_chord"])
        out["step_cd_" + _tag] = float(_rr["Cd"])*1e4
    # What share of the swept C_d the SPAN-WISE term carries.  The narrative
    # said four per cent and it is nearly seven, which matters because the
    # sentence goes on to bound what the theta_12 closure is worth: that bound
    # is this share times the closure's own error, so it moves with it.
    import numpy as _np
    _L = _np.radians(_W["le_sweep_deg"])
    _span = _tot = 0.0
    for _s in _s12["surfaces"].values():
        _r_te = _s["Ue_te_ratio"]; _p = (_s["H_te_squire_young"] + 5.0)/2.0
        _sp = _np.sin(_L)**2*_r_te
        _ch = _np.cos(_L)**2*_r_te**_p
        _span += 2.0*_s["theta_te_c"]*_np.cos(_L)*_sp
        _tot += 2.0*_s["theta_te_c"]*_np.cos(_L)*(_sp + _ch)
    out["span_share_pct"] = 100.0*_span/_tot
    return out


_OFF = _narrative_probes()


h2(_eq_section("4.3"))
para("Bypass onset uses the Abu-Ghannam & Shaw correlation evaluated at the flow-history-averaged "
 "Tu; natural/TS onset integrates one amplification factor per physical frequency using the "
 "tabulated Orr-Sommerfeld growth rates and triggers on their envelope at N_crit; "
 "separation-induced onset closes a laminar bubble across the dead-air region; and cross-flow "
 "onset closes an amplification integral on the stationary vortex (Eq. E11b), the C1 criterion "
 "of Eq. E11 serving only to mark where that integral starts.")
para("Each branch reports the same quantity — how far through its own criterion the layer has "
 "got, as a number that reaches unity at onset — and the kernel fires at the first station "
 "where any of them does. Writing the four commensurably is what makes them one kernel: only "
 "the bypass branch produces an onset REYNOLDS NUMBER, the other three closing on "
 "amplification integrals, so a minimum taken over four Reynolds numbers ranges over one live "
 "term and three placeholders. Earlier versions of this report stated the kernel that way, and "
 "the output showed it: the onset-Reynolds-number column of every cruise surface file was "
 "entirely empty, because the branch that governs there does not produce one.")
for k in ["E08","E09","E10","E10b","E10c","E11","E11b","E11c","E11d","E12","E12b","E13","E13b","E14"]: equation(k)
para("The natural and bypass routes are the same transition seen through two closures with "
 "different ranges of validity, so Eq. E14 blends them over a declared window rather than "
 "switching between them. A single threshold made the predicted transition location a STEP "
 "function of the free-stream turbulence. Measured on this solver by forcing each closure "
 "alone at the old gate value of Tu = 0.1 %%, which is what a switch there did: the "
 "amplification integral puts the cruise section's upper-surface transition at x_tr/c = %.3f "
 "and the correlation at %.3f, a step of %.3f c and of %.0f counts (%.0f %%) in profile drag "
 "across one part in a thousand of an input this study quotes to two figures, with the design "
 "point at 0.07 %%. The window, Tu = 0.10–0.25 %%, is wider than the spread of any case here "
 "(the noisiest natural case is 0.07 %%, the quietest bypass case 0.87 %%), so no result in "
 "this work is blended; it is there so that the model is a function of Tu rather than a "
 "switch. This paragraph used to quote 0.542 and 0.373 and \"a third of the profile drag\", "
 "a pair frozen from the edition that still had the gate; neither number reproduces now."
 % (_OFF["step_xtr_nat"], _OFF["step_xtr_byp"],
    _OFF["step_xtr_nat"] - _OFF["step_xtr_byp"],
    _OFF["step_cd_byp"] - _OFF["step_cd_nat"],
    100.0*(_OFF["step_cd_byp"] - _OFF["step_cd_nat"])/_OFF["step_cd_nat"]),
 italic=True, size=10)
h2(_eq_section("4.4"))
for k in ["E15","E16","E17","E18","E19"]: equation(k)
_pol = pd.read_csv("04_solution/aero_polar.csv")



h2(_eq_section("4.5"))
for k in ["E20","E20b","E20c","E21","E22","E22b","E25","E26","E23","E24"]: equation(k)
para("Eq. E20c is the conversion of the profile drag out of the plane normal to the leading "
 "edge, and it is not the cos²Λ the section lift takes. Two forces act on a swept strip and "
 "Squire-Young returns only one of them. Resolving the free stream as Q = U_n + W with "
 "U_n = Q cosΛ normal to the leading edge and W = Q sinΛ along it, the chordwise profile drag "
 "acts along the normal direction, whose streamwise component is cosΛ, while the span-wise "
 "wall shear acts along the leading edge, whose streamwise component is sinΛ. The second is "
 "not in the chordwise wake deficit at all, but the span-wise momentum integral supplies it "
 "in closed form (Eq. E20b): with no span-wise pressure gradient the deficit grows at the "
 "rate of the wall shear, so integrating from the leading edge gives the whole span-wise "
 "friction force from the trailing-edge state, and no Squire-Young extrapolation applies to "
 "it because downstream of the trailing edge there is no wall and the span-wise deficit is "
 "frozen. That is why the two terms of E20c carry the edge-velocity ratio to different "
 "powers, and why the conversion cannot be a single factor on the total.", italic=True, size=10)
para("This work previously applied cos²Λ to the drag as well as to the lift, justified as "
 "\"2θ/c times a velocity ratio that is frame-independent\" — which does not produce cos²Λ "
 "either, since referring 2θ_n to the streamwise chord c = c_n/cosΛ gives one power of cosΛ, "
 "not two. The yawed flat plate settles it, because its answer is known independently: a flat "
 "plate at yaw is a flat plate in the free stream, so its drag is the unyawed value at the "
 "streamwise run length. At zero pressure gradient U_e,n = U_n and E20c returns exactly cosΛ "
 "for every sweep and every shape factor, and 2(θ_n/c_n)cosΛ = 2θ_n/c is that answer, the "
 "independence principle giving θ_n = θ_streamwise. cos²Λ is short by a further cosΛ — 2 per "
 "cent at the 12° of this wing, 29 per cent at 45° — and keeping only the correct cos³Λ on "
 "Squire-Young, with the span-wise term dropped, would be short by cos²Λ. The check is "
 "asserted in tools/smoke.py rather than described. On the case-study wing the corrected "
 "conversion leaves the section profile drag within "
 + ("%.2f" % abs(_OFF["cd_swept"] - _OFF["cd_unswept"]))
 + " counts of the same section solved unswept ("
 + ("%.1f against %.1f" % (_OFF["cd_swept"], _OFF["cd_unswept"]))
 + "), which is what a 12° sweep should do to a viscous drag; the drag REDUCTION "
 "relative to the fully-turbulent reference is unchanged, both configurations scaling "
 "together. θ_12 is taken as θ_n, the small-cross-flow closure w/W = u/U_e,n, which is exact "
 "at zero pressure gradient and is the standard turbulent one; the span-wise term carries "
 + ("%.1f" % _OFF["span_share_pct"])
 + " per cent of the total at this sweep (this said 4), so an error of a tenth in that closure "
 "is worth "
 + ("%.2f" % (0.1*_OFF["span_share_pct"]))
 + " per cent of C_d.", italic=True, size=10)

# ======================================================================
h1("5.  Why UTSS is Better — Comparison with Existing Solvers")
para("UTSS is designed to be fast, robust and broad in regime coverage. The table "
 "contrasts its capabilities with the principal classes of existing transition tools.")
manual_table(
 ["Capability","XFOIL / e^N codes","RANS γ–Re_θ (CFD)","LES / DNS","UTSS (this work)"],
 [["Natural / TS transition","Yes","Indirect","Yes","Yes (tabulated e^N)"],
  ["Bypass (free-stream Tu)","No","Yes","Yes","Yes (AGS)"],
  ["Separation-induced","Limited","Yes","Yes","Yes (bubble closure)"],
  ["Cross-flow (3-D swept)","No","Add-on only","Yes","Yes (C1, built-in)"],
  ["Single universal calibration","n/a","Needs re-tuning","n/a","Yes (one constant set)"],
  ["3-D wing capability","2-D only","Yes","Yes","Yes (strip + cross-flow)"],
  ["Robustness / convergence","Can fail in sep.","Stiff, costly","Very costly","Robust, direct"],
  ["Typical CPU cost","seconds","hours","days–weeks","< 1 second"],
  ["Suited to design loops","Partly","No","No","Yes"]],
 cap="Capability comparison versus existing solver classes.")
para("Novelty. The distinguishing element is the unified transition kernel "
 "(Eq. E13): a single closed expression that selects the governing transition mechanism "
 "locally, by putting four co-resident mechanisms on one commensurable scale of onset "
 "progress — each carrying a calibration weight — and "
 "firing at the first station where any of them completes, then feeding a single intermittency "
 "closure. Those weights are NOT all in the same sense, and this sentence used to say they "
 "were: a_TS, and a_SEP and a_CF in the shipped configuration, multiply a PROGRESS, so a small "
 "value switches their branch off, while a_BP — and a_SEP and a_CF on their non-default paths — "
 "multiply an onset THRESHOLD, where switching off takes a large one. Eq. E13b writes each "
 "weight where it acts. That is not a nicety: a diagnostic in gen_validation.py took this "
 "paragraph at its word, set every weight small, and left the separation branch live on all ten "
 "swept-wing conditions it was meant to hold open. "
 "Unlike e^N codes (TS only) or correlation RANS models (which require case-by-case re-tuning "
 "and a full CFD solve), UTSS reproduces natural, bypass, separation and cross-flow transition "
 "with ONE calibration set at panel-method cost. Every one of the four branches is the "
 "selected mechanism somewhere in this study — natural/TS on the cruise wing, the "
 "Schubauer-Skramstad plate and the NLF(1)-0416 upper surface; bypass on the climb case and "
 "the ERCOFTAC plates; separation on T3C4 and the NLF(1)-0416 lower surface; cross-flow on "
 "both swept wings — and each is checked against measurement in Section 11.")

# ======================================================================
h1("6.  Case-Study Definition and Input Data")
para("All input data required to run the prediction are tabulated below. The case is fully "
 "three-dimensional.")
h2("6.1  Aircraft and wing geometry (input)")
table_from_csv("01_geometry/geometry_definition.csv", cap="Geometry definition (input).")
h2("6.2  Flight conditions (input)")
table_from_csv("03_model_setup/flow_conditions.csv", cap="Flight / flow conditions (input).", key="flow_conditions")
image("05_postprocessing/csv_plots/conditions_compare.png", width=6.4,
      cap="Cruise vs climb flight-condition comparison (flow_conditions.csv).")
h2("6.3  Fluid (material) properties (input)")
table_from_csv("03_model_setup/material_properties.csv", cap="Air properties (input).")
h2("6.4  Solver settings (input)")
table_from_csv("03_model_setup/solver_settings.csv", cap="Solver configuration (input).")
h2("6.5  Universal calibration constants (input)")
table_from_csv("03_model_setup/calibration_constants.csv",
               cap="Calibration constants and their sources (input).", key="calibration")
image("05_postprocessing/csv_plots/calibration_constants.png", width=6.0,
      cap="UTSS universal calibration constant set (calibration_constants.csv).")

# ======================================================================
h1("7.  Geometry and Engineering Drawings")
para("The wing is drawn to standard third-angle orthographic projection. All drawings are "
 "dimensioned and to scale; views provided: section, plan, front, side, full orthographic "
 "sheet, isometric and structural section.")
for f,c in [("dwg_01_airfoil_section","UTSS-NLF16 aerofoil section (dimensioned)."),
            ("dwg_02_planview","Wing planform — plan (top) view, dimensioned."),
            ("dwg_03_front_side","Front and side orthographic views."),
            ("dwg_04_orthographic","Full third-angle orthographic projection sheet."),
            ("dwg_05_isometric","Isometric view of the wing."),
            ("dwg_06_section_BB","Structural sectional view B–B.")]:
    image(f"01_geometry/drawings/{f}.png", width=6.4, cap=c)
h2("7.1  Geometry data (from CSV)")
image("05_postprocessing/csv_plots/geo_airfoil.png", width=5.8,
      cap="Section geometry plotted from airfoil_UTSS-NLF16.csv.")
image("05_postprocessing/csv_plots/geo_planform.png", width=5.8,
      cap="Span-wise geometry from wing_planform.csv.")
table_from_csv("01_geometry/wing_planform.csv", max_rows=21,
               cap="Wing planform stations (wing_planform.csv).")
image("05_postprocessing/csv_plots/geo_sections_3d.png", width=5.8,
      cap="Lofted 3-D wing sections (wing_sections_3d.csv).")

# ======================================================================
h1("8.  Mesh / Discretisation")
_mi = pd.read_csv("02_mesh/mesh_independence.csv")
_cd = _mi.Cd.to_numpy(float)*1e4
_mm = pd.read_csv("02_mesh/mesh_metrics.csv").set_index("metric")
# The grid every case-study number is computed on, and what this sweep returns
# on it - so the gap to the finest grid is stated rather than left to a reader
# who happens to cross-reference two tables.
_npan_prod = int(float(_mm.loc["Surface streamwise nodes", "value"])) - 1
_cd_prod = float(_mi.loc[(_mi.n_surface_panels - _npan_prod).abs().idxmin(),
                         "Cd"])*1e4
para("THERE IS NO VOLUME MESH. This method is a surface panel discretisation coupled to "
 "an integral boundary layer, and the wall-normal stack below is a reconstruction grid "
 "used to recover profiles from the marched integral quantities - it is not a grid any "
 "equation is solved on. The y⁺, growth ratio and aspect ratio in %s are reported "
 "because they characterise that reconstruction, and they are the quantities a reader "
 "coming from a finite-volume solver will look for; they should not be read as "
 "evidence of a resolved near-wall grid, because there is none to resolve. This "
 "paragraph used to say so only in the last row of the table."
 % "@@TAB:mesh_metrics@@", italic=True, size=10)
para("The surface is discretised with cosine-clustered streamwise nodes; the wall-normal "
 "reconstruction grid places its first point at y⁺ = %s. A "
 "panel-count sweep does NOT demonstrate asymptotic convergence, and is reported because "
 "it does not: from %d to %d surface panels — a factor of %.0f, and a decade on the grid "
 "every case-study result is computed on — the section drag rises at EVERY refinement, "
 "from %.1f to %.1f counts, and the successive changes do not decay (%s per cent over the "
 "last four). There is no asymptote here to quote a discretisation error against."
 % (str(_mm.loc["Target wall y+","value"]),
    _mi.n_surface_panels.min(), _mi.n_surface_panels.max(),
    _mi.n_surface_panels.max()/_mi.n_surface_panels.min(),
    _cd.min(), _cd.max(),
    ", ".join("%+.2f" % v for v in _mi.dCd_pct.to_numpy()[-4:])) +
 "  What that costs the headline number is stated rather than left to be inferred: the "
 "%d-panel production grid returns %.1f counts and the finest grid computed returns %.1f, "
 "so the section drag quoted throughout this report sits %.1f counts — %.1f %% — BELOW the "
 "finest resolution tested, and that gap is still opening."
 % (_npan_prod, _cd_prod, _cd[-1], _cd[-1]-_cd_prod,
    100.0*(_cd[-1]-_cd_prod)/_cd_prod) +
 "  This report previously called the residual a wander \"set by which panel the "
 "transition point lands on\", and then explained it by a momentum thickness at the "
 "Squire-Young station that \"rises MONOTONICALLY\". Neither is what the table says. The "
 "transition location moves by only %.3f chord across the whole sweep and with no trend, so "
 "it is not the first; and θ at that station falls at %d of the %d refinements, so it is "
 "not the second. What does climb without interruption is the SHAPE FACTOR there, %.2f to "
 "%.2f, and C_d = 2(θ/c)(U_e/U_∞)^((H+5)/2) is EXPONENTIAL in it. x/c = 0.98 sits close "
 "enough to the trailing edge that refining the panels resolves the inviscid singularity "
 "there progressively more sharply, so what is not converging is the EVALUATION STATION, "
 "not the transition station hopping between panels. The %d-panel grid is used for every "
 "case-study result; the tabulated validation sections are re-splined onto their own "
 "cosine-clustered grids of 400 and 440 panels."
 % (_mi.x_tr_upper_c.max() - _mi.x_tr_upper_c.min(),
    int((_mi.theta_at_sy_c.diff() < 0).sum()), len(_mi) - 1,
    _mi.H_at_sy.iloc[0], _mi.H_at_sy.iloc[-1],
    _npan_prod))
table_from_csv("02_mesh/mesh_metrics.csv", key="mesh_metrics",
               cap="Metrics of the surface discretisation and of the wall-normal reconstruction stack. No volume mesh is generated.")
table_from_csv("02_mesh/mesh_independence.csv",
               cap="Panel-count sensitivity study.")
for f,c in [("plots/mesh_01_surface","Surface mesh and wall-normal stacks."),
            ("plots/mesh_02_bl_normal","Wall-normal reconstruction grid / y⁺."),
            ("plots/mesh_03_independence","Panel-count sensitivity of C_d and "
             "transition location.")]:
    image(f"02_mesh/{f}.png", width=5.8, cap=c)
table_from_csv("02_mesh/bl_normal_grid.csv", max_rows=22, sample=True,
               cap="Wall-normal grid (bl_normal_grid.csv, sampled).")

# ======================================================================
h1("9.  Solution — Generated Engineering Output Data")
para("This section presents every generated output: numerical tables (CSV) and the plotted "
 "curve for each. The boundary-layer state is reported on both surfaces at the cruise and "
 "climb conditions.")
h2("9.1  Transition prediction summary")
table_from_csv("04_solution/transition_summary.csv", key="ts",
               cap="Transition prediction summary (transition_summary.csv).")
image("05_postprocessing/csv_plots/transition_summary_bar.png", width=6.0,
      cap="Predicted laminar-flow extent by case and surface (transition_summary.csv).")
h2("9.2  Integrated forces and drag breakdown")
# read, not typed: this paragraph said the planform "actually returns 0.49"
_frc_ll = pd.read_csv("04_solution/integrated_forces.csv").set_index(
    "quantity").loc["Wing C_L / section c_l", "value"]
para("The wing lift is obtained from Prandtl's lifting-line theory - Glauert's monoplane "
 "equation solved for the odd Fourier coefficients of the loading - using the section "
 "lift-curve slope and zero-lift incidence returned by the same panel method, the planform "
 "chord distribution, the wing's washout, and the section slope reduced by the cosine of "
 "the quarter-chord sweep. An earlier version of this table asserted C_L = 0.90 c_l; the "
 "planform actually returns " + str(_frc_ll) + ", because a −3° washout is large relative to the 3.7° by "
 "which the root exceeds its zero-lift incidence, and because the downwash of an AR = 8.7 "
 "wing reduces the slope by a quarter. The span efficiency and the induced drag come out of "
 "the same solve. The implementation is checked against the one case with a closed-form "
 "answer - an elliptic planform, for which it returns e = 1 and the exact lift-curve slope "
 "to machine precision - every time the solution is regenerated.")
para("The incidence in @@TAB:flow_conditions@@ is the SECTION design incidence, not the trim point of the "
 "aircraft the planform belongs to: at 1.5° the wing carries 23.6 kN against an all-up "
 "weight of 83.4 kN, and level flight at MTOW would need 7.6°. @@TAB:forces@@ reports both, so "
 "the two are not confused. Every transition result in this study is quoted at the design "
 "incidence and is unaffected by that distinction.", italic=True, size=10)
table_from_csv("04_solution/integrated_forces.csv", key="forces", cap="Integrated forces, "
               "with the wing quantities from the lifting-line solve.")
table_from_csv("04_solution/nlf_vs_turbulent.csv", cap="NLF vs fully-turbulent drag.")
image("05_postprocessing/csv_plots/nlf_vs_turbulent.png", width=5.4,
      cap="Drag benefit of predicted laminar flow vs fully-turbulent.")
# The wing's transition Reynolds number was typed here as 3.6e6 and in the
# README as 3.7e6, for the same quantity.  It is 3.554e6.
_ts_cr = pd.read_csv("04_solution/transition_summary.csv")
_ts_cr = _ts_cr[(_ts_cr.case == "CRUISE") & (_ts_cr.surface == "upper")].iloc[0]
h2("9.2a  The transition-length closure and what the result owes to it")
para("The extent of the transitional region is Dhawan and Narasimha's published correlation, "
 "Re_λ = 9 Re_x,t^0.75, with λ the distance over which the intermittency rises from 0.25 to "
 "0.75. This work previously reported it as validated on the flat plates of Section 11.1, "
 "which span Re_x,t = 6×10⁴ to 1.4×10⁶ — the four ERCOFTAC plates, which are the ones that "
 "carry C_f through transition at all, though only two of them complete it (@@TAB:len_meas@@); "
 "Schubauer & "
 "Skramstad reaches 2.8×10⁶ but gives only a station — and EXTRAPOLATED on the wing, which "
 "transitions at %.1f×10⁶. That extrapolation was of the variable, not of the physics."
 % (_ts_cr.Re_x_tr/1e6))
para("Their correlation is Narasimha's spot model with a constant dimensionless spot formation "
 "rate. Matching γ = 1 − exp(−0.412 ξ²) against γ = 1 − exp(−n σ (x−x_t)²/U), with "
 "N̂ = n σ θ_t³/ν, gives Re_λ = √(0.412/N̂) Re_θ,t^1.5; and on a Blasius plate "
 "Re_θ = 0.664 √Re_x, so Re_θ^1.5 = 0.5411 Re_x^0.75 and the two are THE SAME LAW with "
 "C = 9/0.5411 = 16.63. Checked over Re_x from 6×10⁴ to 10⁷ they agree to every figure "
 "printed. What is assumed constant is therefore the spot formation rate, and Re_θ is the "
 "variable in which that assumption is stated exactly. Nothing is fitted and nothing is "
 "extrapolated: the constant is still their published 9.0.")
_tlf = pd.read_csv("06_validation/transition_length_forms.csv")
_zpg = _tlf[~_tlf.pressure_gradient.astype(bool)]
_t3c4 = _tlf[_tlf.case.str.contains("T3C4")].iloc[0]
_wing = _tlf.iloc[-1]
para("Measured, the two forms agree to %.2f per cent on the %d zero-pressure-gradient plates "
 "— as they must, since Re_θ = 0.664 √Re_x there — and differ by %.2f per cent on T3C4, the "
 "one plate that carries a pressure gradient, where the Re_θ form is the correct one. On the "
 "cruise wing the transitional length moves by %.2f per cent and the section drag by %.2f "
 "counts. That is not the point. The point is that the wing no longer sits outside "
 "the range of a correlation; it sits inside the range of an assumption that was always that "
 "correlation's actual content. cal[\"len_re_x\"] recovers the published form so the "
 "equivalence can be checked rather than taken on trust, and "
 "@@TAB:len_forms@@ is that check."
 % (_zpg.diff_pct.abs().max(), len(_zpg), _t3c4.diff_pct,
    abs(_wing.diff_pct),
    abs(_wing.Cd_counts_Re_x_form - _wing.Cd_counts_Re_theta_form)))
table_from_csv("06_validation/transition_length_forms.csv", key="len_forms",
               cap="The two forms of the transition-length correlation, on every "
                   "plate and on the cruise section "
                   "(transition_length_forms.csv). They are the same law "
                   "wherever Re_theta = 0.664 sqrt(Re_x), and part company only "
                   "where a pressure gradient breaks that.")
# The LENGTH itself, against the plates that resolve one.  This report, the
# README and run_solution all said the correlation was "validated on the four
# ERCOFTAC plates ... the only ones that constrain a length rather than an
# onset" and reproduced the measured extent of the C_f rise "to within a factor
# of two", and nothing in this project measured a length.  Both halves needed
# correcting, so both are read from the table now.
_tlm = pd.read_csv("06_validation/transition_length_measured.csv")
_tlm_ok = _tlm[_tlm.resolves_the_length.astype(bool)]
_tlm_no = _tlm[~_tlm.resolves_the_length.astype(bool)]
para("The LENGTH the closure returns is a separate question from the onset every other table "
 "here scores, and it is measured in @@TAB:len_meas@@ rather than asserted. Taking Narasimha's "
 "own definition — the distance over which the intermittency runs from 0.25 to 0.75, with the "
 "measured intermittency formed from the measured skin friction against the laminar and "
 "turbulent flat-plate correlations — only %d of the %d plates carrying C_f data resolve a "
 "length at all: %s. On the %d that do, the model returns %s times the measured extent. This "
 "report previously called the closure \"validated on the four ERCOFTAC plates\" and quoted a "
 "factor of two, and neither figure had a generating source."
 % (len(_tlm_ok), len(_tlm),
    "; ".join("%s because %s"
              % (r.case.split(" flat plate")[0], r.not_resolved_because)
              for _, r in _tlm_no.iterrows()),
    len(_tlm_ok),
    " and ".join("%.2f" % v for v in _tlm_ok.model_over_measured)))
table_from_csv("06_validation/transition_length_measured.csv", key="len_meas",
               cap="The transition LENGTH against the plates that resolve one "
                   "(transition_length_measured.csv). Two of the four plates "
                   "with skin-friction data do not: T3A- has not completed the "
                   "rise at its last measured station, and T3C4's pressure "
                   "gradient invalidates the flat-plate correlations the "
                   "measured intermittency is formed against.")
para("What the case-study drag owes to the closure is measured all the same, by sweeping the "
 "constant over a factor of four:")
table_from_csv("04_solution/transition_length_sensitivity.csv",
               cap="Sensitivity of the case-study result to the transition-length "
                   "constant (transition_length_sensitivity.csv).")

h2("9.3  Surface distributions — cruise")
for f,c in [("cruise_Cp","Pressure coefficient C_p — cruise."),
            ("cruise_Cf","Skin-friction C_f and laminar run — cruise."),
            ("cruise_theta_H","Momentum thickness θ and shape factor H — cruise."),
            ("cruise_Retheta","The governing transition criterion — cruise. "
             "Two of the four branches are Reynolds-number thresholds and two are "
             "amplification integrals, so both pairs are drawn; at cruise it is N "
             "reaching N_crit that fires, and no Re_θt threshold is active at any "
             "station."),
            ("cruise_gamma","Intermittency γ — cruise.")]:
    image(f"05_postprocessing/csv_plots/{f}.png", width=5.8, cap=c)
table_from_csv("04_solution/surface_cruise_upper.csv", max_rows=30, sample=True,
   cap="Cruise upper-surface BL state (surface_cruise_upper.csv, sampled).")
table_from_csv("04_solution/surface_cruise_lower.csv", max_rows=30, sample=True,
   cap="Cruise lower-surface BL state (surface_cruise_lower.csv, sampled).")
h2("9.4  Surface distributions — climb (off-design, elevated Tu)")
for f,c in [("climb_Cp","Pressure coefficient C_p — climb."),
            ("climb_Cf","Skin-friction C_f and laminar run — climb."),
            # generated for both conditions and shown for one: the cruise block
            # above carries cruise_theta_H and this one dropped its counterpart,
            # so the two sections were not the same section twice
            ("climb_theta_H","Momentum thickness θ and shape factor H — climb."),
            ("climb_Retheta","The governing transition criterion — climb: "
             "here Re_θ crosses the falling Abu-Ghannam & Shaw bypass threshold "
             "while N is still far below N_crit."),
            ("climb_gamma","Intermittency γ — climb."),
            ("compare_cruise_climb_Cf","C_f cruise vs climb — regime-dependent transition.")]:
    image(f"05_postprocessing/csv_plots/{f}.png", width=5.8, cap=c)
h2("9.5  Aerodynamic polars")
para("The polar carries θ_TE/c, the trailing-edge momentum thickness as a fraction of chord, "
 "because that is the assumption Squire-Young rests on and it is the quantity that bounds "
 "where the drag can be believed. Over the sweep it stays below %.3f - read from the table "
 "below rather than typed, where it had been left at 0.009 - so the thin-layer "
 "assumption holds throughout. Outside the incidence envelope it does not merely degrade: "
 "at 16° and a chord Reynolds number of 2×10⁵ the march returns a trailing-edge momentum "
 "thickness of %.0f chords, and it is not even grid-converged — %.0f chords at 160 panels "
 "against %.0f at 360 — so the formula returns a C_d of order %.0f, which is bluff-body drag "
 "from an aerofoil method. A layer thicker than the body is long is arithmetic that has "
 "stopped meaning anything rather than a marginal case, so the solver returns no drag there "
 "instead of a number. Within the envelope and Reynolds range of this study the test never "
 "fires. (This paragraph quoted 1.34 chords and C_d = 1.22; both are now read from the solve "
 "rather than typed, and neither was what it returns.)"
 % (float(_pol.theta_te_c.max()), _OFF["te260"], _OFF["te160"], _OFF["te360"],
    _OFF["cd"]))
_bke = pd.read_csv("04_solution/laminar_bucket_edge.csv")
_bke_i = int(_bke.forward_peak_has_crossed.astype(bool).to_numpy().argmax())
_bke_a, _bke_b = _bke.iloc[_bke_i - 1], _bke.iloc[_bke_i]
para("THE POLAR IS TABULATED AT ONE DEGREE AND STEPS OVER A DISCONTINUITY. Between 2° and 3° "
 "the upper-surface transition collapses from %.3f chord to %.3f and the section drag rises "
 "%.0f per cent, and the table above shows it without saying why. It is not a resolution "
 "failure: refining the incidence does not smooth the jump, only locate it. This section has "
 "TWO competing amplification maxima — one under the leading-edge suction peak, one in the "
 "mid-chord recovery — and onset is wherever N/N_crit first reaches one, so the answer is "
 "decided by which of them crosses first. %s resolves the crossing: the forward peak rises "
 "smoothly through %.4f at %.2f° and %.4f at %.2f°, and on those two neighbouring solves the "
 "transition point moves from %.4f chord to %.4f — %.0f per cent of the chord, for a crossing "
 "by %.0f parts in ten thousand. x_tr is genuinely discontinuous in incidence while the N "
 "field beneath it is smooth. That is the edge of the laminar bucket, a property of the "
 "aerofoil and not of the discretisation, and the drag either side of it is as trustworthy as "
 "anywhere else on the polar."
 % (float(_pol.loc[_pol.alpha_deg == 2.0, "xtr_upper_c"].iloc[0]),
    float(_pol.loc[_pol.alpha_deg == 3.0, "xtr_upper_c"].iloc[0]),
    100.0*(float(_pol.loc[_pol.alpha_deg == 3.0, "Cd"].iloc[0])
           / float(_pol.loc[_pol.alpha_deg == 2.0, "Cd"].iloc[0]) - 1.0),
    "@@TAB:bucket_edge@@",
    float(_bke_a.N_over_Ncrit_forward), float(_bke_a.alpha_deg),
    float(_bke_b.N_over_Ncrit_forward), float(_bke_b.alpha_deg),
    float(_bke_a.x_tr_upper_c), float(_bke_b.x_tr_upper_c),
    100.0*(float(_bke_a.x_tr_upper_c) - float(_bke_b.x_tr_upper_c)),
    1e4*(float(_bke_b.N_over_Ncrit_forward) - 1.0)))
table_from_csv("04_solution/laminar_bucket_edge.csv", key="bucket_edge",
               cap="The edge of the laminar bucket, resolved (laminar_bucket_edge.csv). "
                   "The transition point jumps when the leading-edge amplification peak "
                   "crosses N_crit, not when the incidence passes a threshold.")
_sys = pd.read_csv("04_solution/squire_young_station_sensitivity.csv")
# The shipped station is the row the sweep marks as its own reference, not a
# literal 0.98: that equality tied the report to a constant it does not read.
_sy_ship = _sys[_sys.delta_from_shipped_counts == 0.0].iloc[0]
_sy_lo = _sys.iloc[0]
_pol_clip = int(pd.read_csv("04_solution/aero_polar.csv").H_sy_at_clip.sum())
_ts_clip = pd.read_csv("04_solution/transition_summary.csv")
# The trailing-edge wedge angle decides Squire-Young's premise, and was typed
# here, in the README, in the solver and in run_solution as "26.8 degrees,
# measured off the section".  gen_geometry measures it off the section now.
_geo = pd.read_csv("01_geometry/geometry_definition.csv").set_index("parameter")
para("Where Squire-Young is evaluated matters, and this report first said why in the wrong "
 "terms. The formula wants the trailing edge and this section has a %.1f° wedge one, so the "
 "trailing edge is a stagnation point of the inviscid flow, U_e goes to zero there physically "
 "and (U_e/U_∞)^((H+5)/2) degenerates — at the last control point it returns %.0f counts against "
 "%.1f. The evaluation is therefore pulled forward to %.2f c, and the drag over the range "
 "either side runs from %.1f counts to %.1f. That spread was reported here as a five-count "
 "uncertainty band on the headline number. It is not a band."
 % (float(_geo.loc["Trailing-edge included angle", "value"]),
    _sys.iloc[-1].Cd_counts,
    _sy_ship.Cd_counts, _sy_ship.x_ref, _sy_lo.Cd_counts, _sy_ship.Cd_counts),
 italic=True, size=10)
# Every figure in this paragraph is read from the summary the sweep writes.
# Two of them - the friction accumulated between the two stations and the
# distance the formula moves - were typed here, in the README and in two
# docstrings as 3.94 and 3.65, "the same quantity to a third of a count", and
# all four had been stale by more than a count since the swept-drag
# formulation changed the drag they are differences of.
_sy_sm = pd.read_csv("04_solution/squire_young_station_summary.csv").iloc[0]
para("It is friction being correctly INCLUDED. Between %.2f c and %.2f c the layer accumulates "
 "%.2f counts of real skin friction, measured by integrating C_f over the surface directly, "
 "while the formula moves %.2f — the same quantity to %.1f counts, not the third of a count "
 "this paragraph used to claim. A forward station is not a worse estimate of the same drag; it "
 "is the drag of a shorter aerofoil. What settles it is their SUM, the last column of "
 "@@TAB:sy_station@@: the drag counted so far plus the friction still ahead varies by only "
 "%.2f counts from %.2f c up, where over that same range the drag alone moves %.2f. (Both "
 "halves of that comparison are now measured on the SAME range. This sentence used to set the "
 "spread, which is taken from %.2f c, against the %.2f counts the drag moves from %.2f c - two "
 "different ranges, which overstated the contrast.) The friction the chosen station "
 "still omits is %.3f counts at cruise and %.3f at climb, so the station is converged to under "
 "a fifth of a count and is not uncertain by five. Past %.2f c the formula turns over and "
 "falls — that is the inviscid singularity taking hold, not drag being lost. "
 "The omitted friction is in the SAME frame as the drag it is added to, and getting it there "
 "is the argument of Eq. E20c applied to the wall shear instead of the wake. It used to be the "
 "chordwise integral alone, referred to U_n and c_n — a normal-plane coefficient added to a "
 "streamwise one. The conversion has two terms and neither needs a trailing-edge evaluation. "
 "The chordwise wall shear contributes cos³Λ ∫C_f(U_e,n/U_n)² d(s_n/c_n). The span-wise wall "
 "shear contributes as well, because Squire-Young's span-wise term carries it only as far as "
 "the evaluation station; under the same small-cross-flow closure the drag formula already "
 "uses, τ_wz = (W/U_e,n)τ_wx, so it integrates over the same stations to "
 "cosΛ sin²Λ ∫C_f(U_e,n/U_n) d(s_n/c_n) — the FIRST power of the velocity ratio against the "
 "second, the same asymmetry E20c carries and for the same reason. The two together make the "
 "span-wise contribution independent of the station, so the sum above is cos³Λ times a purely "
 "chordwise quantity plus a constant, which is what the invariance claim can be made about. "
 "It is not adopted for giving the smallest spread and does not: measured from 0.90c up, the "
 "unconverted form gave 1.19 counts, the chordwise half-correction gives 1.31 and this gives "
 "1.23. The smallest of those is the one that adds two frames together. What settles the form "
 "is the yawed flat plate, where U_e,n = U_n and the two terms collapse to cos³Λ + cosΛsin²Λ "
 "= cosΛ exactly — the independence principle's answer, and the same check E20c itself is held "
 "to in tools/smoke.py."
 % (_sy_sm.x_lo, _sy_sm.x_shipped, _sy_sm.friction_accumulated_counts,
    _sy_sm.squire_young_moves_counts, _sy_sm.difference_counts,
    _sy_sm.invariant_spread_from_0p90_counts, _sy_sm.x_invariant_lo,
    _sy_sm.squire_young_moves_from_invariant_lo_counts,
    _sy_sm.x_invariant_lo, _sy_sm.squire_young_moves_counts, _sy_sm.x_lo,
    _sy_sm.friction_omitted_cruise_counts,
    _sy_sm.friction_omitted_climb_counts, _sy_ship.x_ref), italic=True, size=10)
para("And whether the shape factor at that station is solved. Head's entrainment method has no "
 "validity past separation, so H is clamped at 2.8, and on the climb case and at every "
 "incidence above about 3° the upper surface is ON that clamp at the evaluation station — "
 "%d of the %d polar points. That is the same wedge trailing edge: a layer decelerating into a "
 "stagnation point genuinely approaches separation, so the clamp is reached for a physical "
 "reason and it is Squire-Young's thin-attached-layer premise that fails, not the march. The "
 "drag there is formed from a bound, and H_sy_at_clip says so per surface and per point — a "
 "hazard this report already flagged for the trailing-edge separation margin and had not for "
 "the headline number. Closing it properly needs viscous-inviscid coupling rather than a wake "
 "march, and it is worth the same fifth of a count."
 % (_pol_clip, len(_pol)), italic=True, size=10)
para("The stronger statement, which the clamp flag does not make, is WHICH SIDE OF SEPARATION "
 "the evaluation station is on. march_bl finds the first station at which the turbulent shape "
 "factor passes 2.6 and has always written it out; nothing compared it with the station the "
 "drag is evaluated at. On %d of the %d polar points it is UPSTREAM of that station, so the "
 "wake deficit Squire-Young integrates is being read in flow this same march calls separated, "
 "and the margin widens with incidence to %.4f chord at %.0f°. The cruise case is not among "
 "them — both surfaces are attached at the station, by %.4f and %.4f chord — but the CLIMB "
 "UPPER surface is, by %.4f chord. Every one of those points is a point at which the clamp "
 "flag was already true, so no drag quoted here changes; what changes is that the condition is "
 "now a published column, sy_past_sep and sy_margin_to_sep_c, in the polar and in the "
 "transition summary, rather than something a reader had to derive by comparing two others."
 % (int(_pol.sy_past_sep.sum()), len(_pol),
    float(_pol.sy_margin_to_sep_c.min()),
    float(_pol.loc[_pol.sy_margin_to_sep_c.idxmin(), "alpha_deg"]),
    float(_ts_clip.loc[(_ts_clip.case == "CRUISE") & (_ts_clip.surface == "upper"),
                    "sy_margin_to_sep_c"].iloc[0]),
    float(_ts_clip.loc[(_ts_clip.case == "CRUISE") & (_ts_clip.surface == "lower"),
                    "sy_margin_to_sep_c"].iloc[0]),
    float(_ts_clip.loc[(_ts_clip.case == "CLIMB") & (_ts_clip.surface == "upper"),
                    "sy_margin_to_sep_c"].iloc[0])), italic=True, size=10)
table_from_csv("04_solution/squire_young_station_sensitivity.csv", key="sy_station",
               cap="Sensitivity of the section drag to the Squire-Young "
                   "evaluation station (squire_young_station_sensitivity.csv). "
                   "The last row is on Head's H = 2.8 clamp.")
table_from_csv("04_solution/aero_polar.csv", cap="Aerodynamic polar (aero_polar.csv).")
image("05_postprocessing/csv_plots/aero_polar.png", width=6.3,
      cap="Lift curve, drag polar, L/D and transition vs angle of attack.")
h2("9.6  Span-wise distribution (3-D)")
_sp = pd.read_csv("04_solution/spanwise_distribution.csv")
_sp_S = _C.WING["area_S"]
_if = pd.read_csv("04_solution/integrated_forces.csv").set_index("quantity")
# the quarter-chord sweep as the solution PUBLISHES it, not recomputed here
_geo_sweep_c4 = float(_if.loc["Quarter-chord sweep", "value"])
_CL_pub = float(_if.loc["Wing C_L (lifting line, taper + washout + sweep)", "value"])


def _span_CL(col):
    """(2/S) * trapezoid of c_l * chord over the tabulated semi-span."""
    y = list(_sp.y_m); c = list(_sp.chord_m); v = list(_sp[col])
    tot = sum(0.5*(v[i]*c[i] + v[i+1]*c[i+1])*(y[i+1] - y[i])
              for i in range(len(y) - 1))
    return 2.0*tot/_sp_S


_CL_strip = _span_CL("c_l_section")
_CL_ll_int = _span_CL("c_l_lifting_line")
para("INTEGRATING THIS TABLE DOES NOT RETURN THE WING C_L TWO TABLES EARLIER, and the reason "
 "is that they are two section models rather than one number computed twice. c_l_section is "
 "the PANEL section's swept-strip lift, solved on the LEADING-EDGE sweep, because that is the "
 "angle the boundary layer lives on and transition is what these strips exist to predict. The "
 "wing C_L comes from the lifting line, whose section slope is reduced by cos of the "
 "QUARTER-CHORD sweep — %.2f° against %.2f° — which is the convention for a lift-curve slope. "
 "cos²Λ_LE = %.4f against cos Λ_c/4 = %.4f is three per cent before the panel section's own "
 "departure from a linear a₀(α−α_L0) is counted, and measured the strips run from %.3f of the "
 "lifting-line loading at the root to %.3f at the tip. So integrating c_l_section over the "
 "span returns C_L = %.4f where the wing table says %.4f, a %.1f per cent shortfall, and "
 "integrating c_l_lifting_line — published beside it for exactly this reason — returns %.4f, "
 "which is the tabulated value to within the truncation of a twelve-station sweep that stops "
 "at η = %.2f. Neither number is wrong; they answer different questions, and the column that "
 "closes the wing lift is now in the table rather than left to be reconstructed."
 % (float(_geo_sweep_c4), _C.WING["le_sweep_deg"],
    math.cos(math.radians(_C.WING["le_sweep_deg"]))**2,
    math.cos(math.radians(_geo_sweep_c4)),
    float((_sp.c_l_section/_sp.c_l_lifting_line).iloc[0]),
    float((_sp.c_l_section/_sp.c_l_lifting_line).iloc[-1]),
    _CL_strip, _CL_pub, 100.0*(_CL_pub-_CL_strip)/_CL_pub, _CL_ll_int,
    float(_sp.eta.max())), italic=True, size=10)
table_from_csv("04_solution/spanwise_distribution.csv", key="spanwise",
               cap="Span-wise distribution (spanwise_distribution.csv).")
image("05_postprocessing/csv_plots/spanwise_transition.png", width=5.8,
      cap="Span-wise transition front and section drag.")

# ======================================================================
h1("10.  Post-Processing — Contours, Profiles and 3-D Fields")
h2("10.1  Pressure and velocity contours")
# Both conditions, both fields.  gen_postprocessing writes the velocity
# magnitude and the vector field for climb as well as cruise, and this block
# showed the cruise pair and only the climb pressure - so the two conditions
# were not the same section twice, which is the fault already corrected in 9.4
# for the momentum thickness and shape factor.  Two figures were generated on
# every run and appeared nowhere.
for f,c in [("contour_Cp_cruise","Pressure-coefficient contour — cruise."),
            ("contour_speed_cruise","Velocity magnitude and streamlines — cruise."),
            ("vectors_cruise","Velocity vector field — cruise."),
            ("contour_Cp_climb","Pressure-coefficient contour — climb."),
            ("contour_speed_climb","Velocity magnitude and streamlines — climb."),
            ("vectors_climb","Velocity vector field — climb.")]:
    image(f"05_postprocessing/contours/{f}.png", width=6.2, cap=c)
h2("10.2  Boundary-layer velocity and temperature profiles")
para("The profiles are reconstructed from the marched state and not from an assumed shape. "
 "The laminar leg is the Falkner-Skan profile at the shape factor the march solved for, read "
 "from the same family that supplies the closure functions; the turbulent leg is the power "
 "law that shape factor implies, H = (n+2)/n; and the two are blended by the same "
 "intermittency that blends C_f, θ and H, each on its own thickness — δ99 = η99 θ/θ_η for "
 "the similarity profile and δ = θ(n+1)(n+2)/n for the power law. Temperature profiles then "
 "follow from the compressible Crocco–Busemann relation (Eq. E21), showing wall-recovery "
 "heating through the boundary layer.")
_blp = pd.read_csv("04_solution/bl_profiles_cruise.csv").groupby("station").first()
# the cross term scales as gamma(1-gamma); x/c = 0.95 is 99.7 per cent
# turbulent and carries none worth naming
_bl_g = _blp.intermittency_gamma*(1.0 - _blp.intermittency_gamma)
_bl_tr = _blp[_bl_g >= 0.01]
_bl_ok = _blp.drop(index=_bl_tr.index)
para("THE PLOTTED PROFILE DOES NOT CARRY THE MARCHED SHAPE FACTOR AT A TRANSITIONAL STATION, "
 "and the table gives both numbers rather than letting them disagree in silence. The march "
 "blends integrals, θ = (1−γ)θ_lam + γθ_turb; this reconstruction blends velocities, "
 "u = (1−γ)u_lam + γu_turb. Those are different operations. δ* is linear in u and survives "
 "it — the plotted profile returns δ* to two parts in a thousand of the linear blend — but θ "
 "is QUADRATIC in u, and the pointwise blend carries a cross term u_lam·u_turb that a blend "
 "of integrals does not, worth eight per cent of θ. H = δ*/θ inherits all of it: at "
 "%s, where γ = %.3f, the curve integrates to %.3f against the marched %.3f. Where γ is 0 or "
 "1 the cross term vanishes and the two agree to %.1f per cent or better, which is "
 "interpolation error and nothing else. Rescaling does not repair it — stretching y "
 "multiplies δ* and θ alike and leaves H untouched — and matching H would mean drawing a "
 "single-family profile AT the marched H, which is the assumed shape this reconstruction "
 "exists to avoid and would erase the two-layer structure that is the physical content of a "
 "transitional station. So H_profile is published beside H_shape."
 % (", ".join(_bl_tr.index.astype(str)),
    float(_bl_tr.intermittency_gamma.iloc[0]),
    float(_bl_tr.H_profile.iloc[0]), float(_bl_tr.H_shape.iloc[0]),
    100.0*float(((_bl_ok.H_profile - _bl_ok.H_shape).abs()
                 / _bl_ok.H_shape).max())), italic=True, size=10)
for f,c in [("bl_velocity_profiles","Boundary-layer velocity profiles."),
            ("bl_temperature_profiles","Boundary-layer temperature profiles (K)."),
            ("bl_temperature_ratio","Normalised temperature profiles T/T_e.")]:
    image(f"05_postprocessing/profiles/{f}.png", width=5.5, cap=c)
table_from_csv("04_solution/bl_profiles_cruise.csv", max_rows=28, sample=True,
   cap="BL velocity/temperature profiles (bl_profiles_cruise.csv, sampled).")
h2("10.3  Three-dimensional surface contours and vectors")
para("The full 3-D wing surface is coloured by the predicted fields. The intermittency contour "
 "shows the laminar-to-turbulent change, BUT THE COLOUR BOUNDARY IS NOT THE ONSET LINE, and "
 "this sentence used to invite exactly that misreading by calling it \"the transition front\". "
 "Transition has a length: γ leaves zero at onset and reaches one only after the Narasimha "
 "spot-growth distance, so the boundary the eye picks out — the γ = ½ contour — lies aft of "
 "onset by %.3f to %.3f chord across this span, more than the whole %.3f chord difference "
 "between the cruise upper and lower onsets that Section 9 treats as a result. On the upper "
 "surface onset runs %.3f at the root to %.3f at the tip while the half-intermittency contour "
 "runs %.3f to %.3f; both columns are in %s so the figure can be read against the numbers "
 "rather than instead of them. Outboard of about %.0f per cent semi-span the layer does not "
 "reach γ = 0.9 before the trailing edge at all, so there the contour never closes."
 % ((_sp.x_gamma50_upper_c - _sp.xtr_upper_c).min(),
    (_sp.x_gamma50_upper_c - _sp.xtr_upper_c).max(),
    abs(float(_ts_clip.loc[(_ts_clip.case == "CRUISE") & (_ts_clip.surface == "upper"), "x_tr_c"].iloc[0])
        - float(_ts_clip.loc[(_ts_clip.case == "CRUISE") & (_ts_clip.surface == "lower"), "x_tr_c"].iloc[0])),
    _sp.xtr_upper_c.min(), _sp.xtr_upper_c.max(),
    _sp.x_gamma50_upper_c.min(), _sp.x_gamma50_upper_c.max(),
    "@@TAB:spanwise@@", 80.0))
for f,c in [("td_Cp","3-D wing surface contour — pressure C_p."),
            ("td_Cf","3-D wing surface contour — skin friction C_f (×10³)."),
            ("td_gamma","3-D wing surface contour — intermittency γ (transition front)."),
            ("td_skinfriction_vectors","Upper-surface flow direction coloured by C_f "
              "(chordwise only — the strip formulation carries no span-wise wall shear).")]:
    image(f"05_postprocessing/three_d/{f}.png", width=6.2, cap=c)

# ======================================================================
h1("11.  Validation and Calibration")
para("To qualify as universal, the solver is validated against every published dataset the "
 "kernel\'s four branches reach — five flat plates spanning 0.03 to 6 per cent free-stream "
 "turbulence (bypass, natural and separation-induced transition), 86 aerofoil conditions on "
 "the NLF(1)-0416 section, and two swept wings from different facilities and eras — using ONE "
 "universal calibration set with no per-case re-tuning of the physics. The transition-onset "
 "momentum-thickness Reynolds number Re_θt is the metric on the plates and the transition "
 "location x_tr/c on the aerofoil and the swept wings. The skin-friction error is reported "
 "separately over the laminar run and over the turbulent run, on the stations where the "
 "measurement and the prediction are in the same state; pooled across transition it measures "
 "the onset error a second time, in the wrong units, because a plate whose onset is early by "
 "16 % is then charged with the whole laminar-to-turbulent step in C_f over the interval "
 "between the two onsets.")
table_from_csv("06_validation/validation_summary.csv",
               cap="Validation summary — predicted vs published Re_θt.")
h2("11.1  Flat plates")
for f,c in [("val_T3A","Validation — ERCOFTAC T3A flat plate (Tu = 3.0 %, bypass)."),
            ("val_T3AM","Validation — ERCOFTAC T3A⁻ flat plate (Tu = 0.87 %, bypass)."),
            ("val_T3B","Validation — ERCOFTAC T3B flat plate (Tu = 6.0 %, bypass)."),
            ("val_T3C4","Validation — ERCOFTAC T3C4 flat plate (laminar separation bubble)."),
            ("val_SS","Validation — Schubauer & Skramstad natural transition (Tu = 0.03 %)."),
            ("val_combined_Re_theta_t","Universal validation — Re_θt across all five plates.")]:
    image(f"06_validation/plots/{f}.png", width=5.9, cap=c)

# Placed with the flat plates, which is what it is about.  It carried the
# number 11.3a and sat between the two swept-wing sections, so a section
# on T3C4's bubble and the ERCOFTAC momentum integral interrupted the
# cross-flow argument in the middle.
h2("11.1a  Where the flat-plate residuals come from")
para("The residuals are accounted for here, and the accounting is generated by "
 "gen_validation.py rather than argued.")
_rd0 = pd.read_csv("06_validation/residual_diagnostics.csv")
_lam_r = _rd0.laminar_run_meas_over_march.dropna()
_lam_worst = float((_lam_r - 1.0).abs().max())*100.0
_lam_t3c4 = float((_rd0[_rd0.case.str.contains("T3C4")]
                   .laminar_run_meas_over_march.iloc[0] - 1.0))*100.0
para("The laminar branch is not where they are. Compared against the model's own marched "
 "momentum thickness at the same stations — not against flat-plate Blasius, which is the "
 "wrong reference for the one plate that has a pressure gradient — the measured laminar "
 "layer agrees to within %.0f %% on all %d plates that carry C_f data, and to %.0f %% on T3C4. "
 % (_lam_worst, len(_lam_r), abs(_lam_t3c4)) +
 "An earlier version of this report attributed the T3C4 residual to the pre-transitional "
 "thickening a laminar layer undergoes in a turbulent free stream, on the strength of the "
 "measured momentum thickness being 1.36 times Blasius at onset. That comparison was wrong: "
 "a laminar layer in an adverse gradient is thicker than Blasius for a reason that has "
 "nothing to do with turbulence, and against the march that carries the gradient the "
 "agreement is 2 %.")
para("The T3C4 residual is in the bubble, and @@TAB:bubble@@ localises it — but not where "
 "this study previously placed it. Three things are now established. First, the shape-factor "
 "cap is gone: H*(H) folds at H = 4.03 where the attached and reverse-flow branches meet, so "
 "the inversion H = H(H*) is not unique there, but a march knows which branch it is on because "
 "it arrived continuously, and taking the root nearest the previous H carries it across. The "
 "reverse branch itself was stopped at H = 4.99 by the continuation parameter, which was a "
 "choice and not a limit — it continues smoothly to H = 6.41, past the measured 5.17.")
para("Second, and this is the substantive point: THE MOMENTUM INTEGRAL CANNOT BE CLOSED WITH "
 "THIS EXPERIMENT'S OWN DATA. Across the measured bubble, reproducing the measured "
 "dθ/dx = 0.00591 m⁻¹ from the measured dU_e/dx = −0.200 s⁻¹ and θ = 2.70 mm requires a shape "
 "factor of 19. The measurement itself reports 5.17, which returns 0.00202 m⁻¹ — short by a "
 "factor of 2.93. No integral method closed on any physical profile family reproduces this "
 "bubble, so the shape-factor cap was never the leading term, and the earlier attribution of "
 "a factor 1.32 of the shortfall to that cap was accounting for the wrong thing.")
# Both figures were typed here as -14.2 and -30.4 against the -13.9 and -30.2
# the summary generates.  verify_outputs checks the bracket value unanchored,
# so it was satisfied by the copy in the table beside this paragraph while the
# prose said something else.
_t3c4 = pd.read_csv("06_validation/validation_summary.csv")
_t3c4 = _t3c4[_t3c4.case.str.contains("T3C4")].iloc[0]
para("Third, onset is not resolved to a station here. It is taken as the station of minimum "
 "measured C_f, as on the other plates. On this plate C_f is 1.87×10⁻⁴ at x = 1.295 m and "
 "1.83×10⁻⁴ at x = 1.395 m — two per cent apart, in a hot-film measurement of a quantity at "
 "its floor. They are not distinguishable, so onset is bracketed by them, Re_θ from 309.3 to "
 "381.3, and quoting the second alone reports the end of the plateau as though it were the "
 "beginning. Against that bracket the error is %.1f %%, not %.1f %%. The point value is "
 "retained in the tables for continuity with the literature; the bracket is what the residual "
 "should be read against."
 % (_t3c4.Re_theta_t_err_bracket_pct, _t3c4.Re_theta_t_err_pct))
# The shipped bubble length was typed as 0.052 m and is 0.055; the other three
# figures in this sentence were typed too and are read now.
_bub = pd.read_csv("06_validation/bubble_diagnostics.csv").set_index("quantity")
_nsum0 = pd.read_csv("06_validation/aerofoil_nlf0416_summary.csv").set_index("set")
para("Reading the amplification rate at the marched shape factor instead of at the developed "
 "reverse-flow profile became possible once the march crossed the fold, and looks more "
 "principled because it removes a constant. It is rejected on measurement: the T3C4 bubble "
 "lengthens from the %.3f m the shipped closure returns to 0.087, towards the measured %.2f, "
 "but the 86 NLF(1)-0416 conditions collapse from %d to 21 inside the bracket. The reason is "
 "physical — the marched H is the integral shape factor of the whole dead-air region, still "
 "near %.1f, while the "
 "detached shear layer riding on it is inflectional from the moment the flow leaves the wall. "
 "It is exposed as cal[\"bub_sigma_local\"] so the test is reproducible."
 % (_bub.loc["bubble length [m]", "model"],
    _bub.loc["bubble length [m]", "measured"],
    int(_nsum0.loc["All", "within_bracket"]),
    _bub.loc["shape factor at reattachment", "model"]),
 italic=True, size=10)
# residual_diagnostics.csv is read once, as _rd0 above.  It was read twice more
# here - the first into a name nothing ever used - and three reads of one file
# are three chances for them to become three different files.
_gain = {r.case.split(" flat plate")[0].replace("ERCOFTAC ", ""):
         r.location_gain_bypass_threshold
         for _, r in _rd0.iterrows()
         if r.location_gain_bypass_threshold == r.location_gain_bypass_threshold}
para("Conditioning. In a decaying stream the onset threshold rises while Re_θ grows only as "
 "the square root of distance, so the two curves close at a shallow angle and the crossing "
 "is sensitive. Shifting the threshold by ±10 %% moves the predicted transition location by a "
 "factor of %.1f on T3A, %.1f on T3A⁻ and %.1f on T3B. A correlation accurate to ten per cent "
 "cannot locate transition to ten per cent in such a flow."
 % (_gain["T3A"], _gain["T3A-"], _gain["T3B"]))
table_from_csv("06_validation/residual_diagnostics.csv",
               cap="The laminar branch against the measurements, and the "
                   "sensitivity of the crossing (residual_diagnostics.csv).")
table_from_csv("06_validation/bubble_diagnostics.csv", key="bubble",
               cap="The T3C4 bubble, modelled against measured "
                   "(bubble_diagnostics.csv). The momentum integral across the dead-air "
                   "region is exact given U_e and H, so these are the whole of the residual.")

h2("11.2  NLF(1)-0416 aerofoil — 86 transition locations")
_nsum_11 = pd.read_csv("06_validation/aerofoil_nlf0416_summary.csv").set_index("set")
para("The largest single body of evidence in this work, and genuinely out of sample. That "
 "was not always true: the anchor of the natural branch — the one quantity that branch takes "
 "from measurement — used to be chosen at whatever value put the most of these 86 predictions "
 "inside the experimental bracket, which made this a calibration set while this section, the "
 "figure below and the project README all called it otherwise. It is now set on the "
 "Schubauer & Skramstad plate alone, which it reproduces to 0.07 %%, and the cost of that is "
 "reported rather than absorbed: the bracket count fell from 51 to %d and the mean absolute "
 "error rose, which is what happens when a constant stops being fitted to the set it is "
 "scored on. The error today is %.4f chord (%s below); this sentence used to give it as "
 "0.0334c, which was the value at the time of that change and has since moved with "
 "corrections that have nothing to do with the anchor - a before-and-after pair frozen in the "
 "narrative stops describing the current model the moment anything else moves, so only the "
 "current figure is quoted and it is read from the table. "
 "Transition locations were digitised from Fig. 9 of the source report at four "
 "chord Reynolds numbers on both surfaces, and each condition is matched by trimming the "
 "incidence to the measured lift coefficient. The experiment brackets transition between "
 "adjacent orifices 0.05c apart, so its own uncertainty is ±0.025c and a prediction inside "
 "that band cannot be distinguished from the measurement. Both the natural (TS) and the "
 "separation-induced branches are selected on this set."
 % (int(_nsum_11.loc["All", "within_bracket"]),
    float(_nsum_11.loc["All", "mean_abs_err_c"]), "@@TAB:nlf_stats@@"))
table_from_csv("06_validation/aerofoil_nlf0416_summary.csv", key="nlf_stats",
               cap="NLF(1)-0416 error statistics by surface "
                   "(aerofoil_nlf0416_summary.csv).")
image("06_validation/plots/val_aerofoil_nlf0416.png", width=6.4,
      cap="Transition location against lift coefficient at four chord "
          "Reynolds numbers, measurement bars = the ±0.025c orifice bracket.")
table_from_csv("06_validation/aerofoil_nlf0416.csv", max_rows=30, sample=True,
               cap="NLF(1)-0416 point-by-point comparison "
                   "(aerofoil_nlf0416.csv, sampled).")
para("The separation branch claims to predict a LENGTH and not a point, and that the length "
 "scales with the disturbance environment because it is N_crit θ_s/σ_sep. Both of these "
 "datasets form bubbles, at free-stream turbulence levels seventy times apart, so together "
 "they measure that claim rather than illustrate it.")
table_from_csv("06_validation/bubble_length_scaling.csv", key="bubble_len",
               cap="Bubble length against the disturbance environment "
                   "(bubble_length_scaling.csv): the separating plate at "
                   "Tu = 2.11 % against every NLF(1)-0416 bubble at 0.03 %, "
                   "in separation momentum thicknesses.")

h2("11.3  Swept wings — the cross-flow branch")
# read out of the CSVs rather than typed: these three numbers were left at
# 14.7 / 51.7 / 18.4 after the swept sections began being solved in the plane
# normal to the leading edge, and nothing checked them
_sw1 = pd.read_csv("06_validation/swept_wing_crossflow.csv")
_sw2 = pd.read_csv("06_validation/swept_wing_independent.csv")
_e_cal  = float(_sw1.err_pct.abs().mean())
_e_ind  = float(_sw2.err_pct_C1_150.abs().mean())
_e_ind2 = float(_sw2.err_pct_C1_200.abs().mean())
para("The cross-flow coefficient is set on the first of these two experiments and nothing is "
 "calibrated on the second, which is a different facility, section and era. The branch "
 "reproduces the calibration set to " + ("%.1f" % _e_cal) + " % at the frozen constant "
 "C1 = 150. On the independent set that same constant gives " + ("%.1f" % _e_ind) +
 " %, and C1 = 200 gives " + ("%.1f" % _e_ind2) + " % — so the "
 "criterion has the right functional form on both wings but not one critical value that "
 "serves both. The independent set is therefore tabulated at BOTH ends of that band: "
 "reporting only C1 = 150 understates what the criterion does here, and reporting only "
 "C1 = 200 would be a per-case re-tune of the kind this work is claiming not to need. The "
 "spread between the two columns is the limitation, stated rather than averaged away. "
 "Nothing else in the model differs between the two columns.")
# The within-facility Reynolds-number trend, read from the CSV rather than
# typed: the slope and correlation quoted in 11.3 stood at 251 and -0.88
# against the generated values.
_trend = pd.read_csv("06_validation/crossflow_reynolds_trend.csv").set_index("dataset")
_tr_dag = _trend.loc["Dagenhart & Saric (calibration)"]
_tr_bol = _trend.loc["Boltz et al. (independent)"]
_cfs = pd.read_csv("06_validation/crossflow_criticals_summary.csv")
_cfs = _cfs[_cfs.criterion == "surrogate Re_theta2"].set_index("dataset")
_rcp = pd.read_csv("06_validation/crossflow_receptivity_summary.csv")
# The two facilities' chord Reynolds ranges.  The factor between them was typed
# as seven here and in gen_validation and as six in the README, for the same
# quantity; the ranges themselves do not need a factor and cannot disagree.
_rtr = pd.read_csv("06_validation/crossflow_reynolds_trend.csv")
_cts = pd.read_csv("06_validation/crossflow_threshold_sweep.csv")
_cfe = pd.read_csv("06_validation/crossflow_criticals_summary.csv")
_cfe = _cfe[_cfe.criterion != "surrogate Re_theta2"].set_index("dataset")
_dag = _cfs.loc["Dagenhart & Saric (calibration)"]
_bol = _cfs.loc["Boltz et al. (independent)"]
_poo = _cfs.loc["Both facilities pooled"]
_pooe = _cfe.loc["Both facilities pooled"]
para("What each experiment requires of the criterion is measured rather than asserted "
 "(@@TAB:cf_criticals@@). The march is run with every branch disabled so that it reaches the measured "
 "transition station, and the criterion is evaluated there. Dagenhart & Saric require a "
 "critical Re_θ2 of %.0f with a %.1f %% coefficient of variation over six chord Reynolds "
 "numbers; Boltz et al. require %.0f with %.1f %% over four sweep angles and a factor of three in chord "
 "Reynolds number. Each facility is therefore internally consistent — the second markedly so — "
 "and the two differ by %.0f %%. That is the shape of a receptivity difference rather than of a "
 "criterion with the wrong form: stationary cross-flow vortices are seeded by leading-edge "
 "roughness, and neither report documents the surface finish. THREE attempts to close the gap "
 "fail and are recorded rather than dropped. Solving the swept sections in the plane normal "
 "to the leading edge, which is the correct mean flow and had not been done, widens the gap "
 "rather than closing it — it was the strongest remaining physical candidate and it is now "
 "tested rather than argued about. Replacing the constant surrogate by the exact "
 "Falkner-Skan-Cooke factor K(λ) makes matters worse, taking the pooled coefficient of "
 "variation from %.0f to %.0f %% and inverting the ratio between the two sets. Giving the "
 "cross-flow branch its own "
 % (_dag.mean_critical_value, _dag.coeff_of_variation_pct,
    _bol.mean_critical_value, _bol.coeff_of_variation_pct,
    100.0*(_bol.mean_critical_value - _dag.mean_critical_value)/_dag.mean_critical_value,
    _poo.coeff_of_variation_pct, _pooe.coeff_of_variation_pct) +
 "amplification threshold, separate from the one Mack's relation supplies for "
 "Tollmien-Schlichting waves — which is defensible, since a stationary cross-flow vortex is "
 "not seeded by free-stream turbulence — does not help either (@@TAB:cf_thresh@@). Sweeping "
 "that threshold from N = %g to N = %g, with C1 refitted on the calibration set at every "
 "value so the two constants are not confounded, takes the calibration set from %.1f to "
 "%.1f %% and the INDEPENDENT set from %.1f to %.1f %% — away from agreement, not towards it. "
 "This report previously said it moved the independent set \"only from 55 to 51 %%\"; the "
 "sweep is generated now, and both the size and the sign of that were wrong. The reason it "
 "cannot help is that once Re_θ2 exceeds C1 the amplification builds so quickly that the "
 "threshold is nearly redundant with C1 itself, so the two constants cannot be separated by "
 "these data — which the refitted C1 column shows directly, barely moving across the sweep."
 % (_cts.CF_N.iloc[0], _cts.CF_N.iloc[-1],
    _cts.calibration_err_pct.iloc[0], _cts.calibration_err_pct.iloc[-1],
    _cts.independent_err_pct.iloc[0], _cts.independent_err_pct.iloc[-1]))
table_from_csv("06_validation/crossflow_threshold_sweep.csv", key="cf_thresh",
               cap="A separate amplification threshold for the cross-flow "
                   "branch, with C1 refitted on the calibration set at each "
                   "value (crossflow_threshold_sweep.csv).")
# The counts are read off the table rather than typed: it grew a row when the
# solved eigenvalue problem was added, and "the two that cost nothing" was
# already stale by one the moment it did.
_cfv = pd.read_csv("06_validation/crossflow_formulations.csv")
_n_free = int(((~_cfv.costs_calibration_vs_shipped)
               & (~_cfv.helps_independent_vs_shipped)).sum()) - 1
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten"}
para("The %s formulations of the branch that this solver can be put into are scored against both "
 "experiments in @@TAB:cf_forms@@, and not one of them reconciles the two. The pattern is the "
 "same throughout: every variant that helps the independent set costs more on the calibration "
 "set, and the %s that cost nothing there help nothing here. That was a claim about what had "
 "been tried; it is a table now."
 % (_WORDS.get(len(_cfv), str(len(_cfv))), _WORDS.get(_n_free, str(_n_free))))
table_from_csv("06_validation/crossflow_formulations.csv", key="cf_forms",
               cap="Every cross-flow formulation the solver exposes, scored on "
                   "both swept wings (crossflow_formulations.csv).")
# ---- the fourth attempt: solve the problem the criterion stands in for ----
# Every number below is read from the generated CSVs.  The paragraph this
# replaces did not exist; the limitation it describes was stated in the
# conclusions as something that "would need" the stability problem solved, and
# it has now been solved, so what it produced belongs here.
_amp = pd.read_csv("06_validation/crossflow_amplification.csv")
_ampd = _amp[_amp.dataset.str.startswith("Dagenhart")].iloc[2]
_amps = pd.read_csv("06_validation/crossflow_amplification_summary.csv").set_index("dataset")
_ad = _amps.loc["Dagenhart & Saric (calibration)"]
_ab = _amps.loc["Boltz et al. (independent)"]
_asal = _amps.loc["vs SALLY (three conditions)"]
_cff = pd.read_csv("06_validation/crossflow_formulations.csv")
_cfen = _cff[_cff.formulation.str.startswith("solved stationary")].iloc[0]
para("A fourth attempt goes after the criterion's form directly, by solving the problem C1 "
 "stands in for. The Orr-Sommerfeld equation is solved on the velocity resolved along each "
 "wave-angle direction, U_ψ = cos Λ f′ cos ψ + sin Λ g sin ψ in units of the total edge speed; "
 "the wave angle at which the mode is stationary — ω_r = 0, which is what a naphthalene "
 "visualisation can see — is found by bisection on the sign change, and the amplification a "
 "stationary packet accumulates over dx is ω_i dx/c_gx with the chordwise group velocity taken "
 "from the two derivatives the (k, ψ) parameterisation already provides. The sweep angle the "
 "similarity solution is given is the LOCAL one, between the external streamline and the chord "
 "line: the span-wise edge velocity is constant on an infinite swept wing while the chordwise "
 "one grows through the favourable run, so that angle falls from %.0f° at 3 %% chord to %.0f° at "
 "60 %% on these sections, against a leading-edge value of %.0f°."
 % (_ampd.sweep_local_deg_at_x003, _ampd.sweep_local_deg_at_x060,
    _ampd.sweep_deg))
para("Two things make this harder than the two-dimensional problem and both decide the answer. "
 "The mode cannot be found by asking for the eigenvalue nearest c = 0: the resolved profile's "
 "edge velocity is near zero at exactly the wave angles of interest, so the discretised "
 "continuous spectrum crowds onto the physical mode rather than onto c_r = 1, and a nearest-root "
 "search returns growth rates three orders above anything physical. What separates them is the "
 "eigenfunction, not the eigenvalue. And the outer boundary must be far enough out for that "
 "test to mean anything. The filter measures the eigenfunction over the outer fifth of the "
 "domain and requires it to be under 2 per cent of its peak; a wave of wavenumber k decays as "
 "exp(−k y), so at k = 0.1 — the longest wave of interest — that is exp(−0.1×0.8×40) = 4 per "
 "cent on a y_max = 40 θ grid, and the filter throws the PHYSICAL mode away, leaving only short "
 "waves and putting the envelope maximum on the edge of the surviving band. At y_max = 100 θ "
 "the same wave is at exp(−8) = 0.03 per cent and passes, while the discretised continuous "
 "spectrum stays above 25 per cent of its peak out there and does not. This paragraph gave the "
 "two decay figures as five and 0.3 per cent; neither is what exp(−k y) returns where the "
 "filter looks, and the first did not even fail the 2 per cent test the argument turns on.")
_abz = _amp[_amp.dataset.str.startswith("Boltz")].sort_values("sweep_deg")
para("The result is checked against an independent stability code before it is used for anything. "
 "Dagenhart & Saric computed stationary N-factors with SALLY for three of their six conditions "
 "and tabulated them; against those three the present solve differs by %.2f on average and "
 "%.2f root-mean-square (@@TAB:cf_amp@@), which is agreement, not calibration — nothing here is "
 "fitted to them. On the levels the two facilities require it does better than the surrogate and "
 "still not well enough. Dagenhart & Saric's transitions occur at N_cf = %.2f ± %.2f and Boltz "
 "et al.'s at %.2f ± %.2f, a ratio of %.2f where the algebraic surrogate's critical Reynolds "
 "numbers differ by %.2f. The levels move together; the scatter does not. Within Boltz's four "
 "conditions the coefficient of variation rises from %.1f %% on the surrogate to %.1f %% here, "
 "and the reason is visible rather than statistical: N_cf at the measured transition falls "
 "monotonically with sweep angle, %.2f at %.0f° down to %.2f at %.0f°, so no single threshold "
 "can pass through all four."
 % (_asal.mean_N_cf, _asal.sd_N_cf, _ad.mean_N_cf, _ad.sd_N_cf,
    _ab.mean_N_cf, _ab.sd_N_cf, _ad.mean_N_cf/_ab.mean_N_cf,
    _bol.mean_critical_value/_dag.mean_critical_value,
    _bol.coeff_of_variation_pct, _ab.coeff_of_variation_pct,
    _abz.N_cf.iloc[0], _abz.sweep_deg.iloc[0],
    _abz.N_cf.iloc[-1], _abz.sweep_deg.iloc[-1]))
table_from_csv("06_validation/crossflow_amplification.csv", key="cf_amp",
               cap="Stationary cross-flow amplification factor at the measured "
                   "transition, every branch of the kernel held off, against "
                   "Dagenhart & Saric's own SALLY N-factors where they exist "
                   "(crossflow_amplification.csv).")
table_from_csv("06_validation/crossflow_amplification_summary.csv",
               cap="What each facility requires of the amplification factor "
                   "(crossflow_amplification_summary.csv).")
image("06_validation/plots/val_crossflow_amplification.png", width=5.9,
      cap="Cross-flow N-factor at the measured transition. The two facilities "
          "sit on different levels; the stars are SALLY.")
para("Put through the kernel with its threshold set on the calibration set alone — the same way "
 "C1 was — it follows the pattern every other variant follows, from the other side. It IMPROVES "
 "the calibration set, from %.1f %% to %.1f %% in transition location, and it costs the "
 "independent set %.1f %% against %.1f %%, for a pooled %.1f %% against %.1f %% "
 "(@@TAB:cf_forms@@ carries it as a row like every other variant). It is therefore not adopted, "
 "for the reason the shipped branch is: a formulation that is better only on the set its one "
 "constant was fitted to has not been shown to be better. What fails is that a "
 "stationary cross-flow vortex is forced by surface roughness, and the amplification factor "
 "carries no information about how large the disturbance was when it started. Dagenhart & Saric "
 "say so themselves: “the receptivity portion of the transition process is equally important in "
 "the vortex development, growth, and eventual breakdown”, and they cite Radeztsky et al. for "
 "the finding that micron-sized roughness near the attachment line strongly influences "
 "crossflow-dominated transition. A polished natural-laminar-flow model of 1993 and an untapered "
 "wing tested in 1960 are not the same surface, and neither report gives a roughness height. "
 "Closing the gap needs an input the experiments do not contain, not a better stability "
 "calculation — and that is now a measurement rather than a conjecture."
 % (_cff.calibration_err_pct.iloc[0], _cfen.calibration_err_pct,
    _cfen.independent_err_pct, _cff.independent_err_pct.iloc[0],
    _cfen.pooled_err_pct, _cff.pooled_err_pct.iloc[0]))
table_from_csv("06_validation/swept_wing_crossflow.csv",
               cap="Cross-flow validation, 45° swept NLF(2)-0415 "
                   "(Dagenhart & Saric — the calibration set).")
image("06_validation/plots/val_swept_crossflow.png", width=5.9,
      cap="Transition location against chord Reynolds number, 45° swept "
          "NLF(2)-0415.")
table_from_csv("06_validation/swept_wing_independent.csv",
               cap="Independent swept-wing check, NACA 64(2)A015 (Boltz et al.) "
                   "at both ends of the reported cross-flow band — nothing calibrated here.")
image("06_validation/plots/val_swept_independent.png", width=5.9,
      cap="Transition location against sweep angle, NACA 64(2)A015, "
          "with the C1 = 150–200 band shaded.")

h2("11.4  What the two swept-wing experiments require of the criterion")
table_from_csv("06_validation/crossflow_criticals_summary.csv", key="cf_criticals",
               cap="What each swept-wing experiment requires of the cross-flow "
                   "criterion, evaluated at the measured transition station "
                   "(crossflow_criticals_summary.csv).")
table_from_csv("06_validation/crossflow_criticals.csv",
               cap="Point-by-point cross-flow criterion values at the measured "
                   "transition stations (crossflow_criticals.csv).")
para("The gap cannot be converted into a roughness ratio by this method, and saying why is "
 "firmer than attributing it to receptivity by elimination. For a roughness-seeded stationary "
 "vortex the natural currency is amplification, ΔN = ln(A₀,₁/A₀,₂), which is the ratio of "
 "initial amplitudes and so of effective leading-edge roughness. But the rate this branch "
 "integrates is read off a separated STREAMWISE profile and is nearly Reynolds-independent — "
 "0.0417 at Re_θ = 200 against 0.0461 at 8000 — so N_cf is essentially σ times run length over "
 "θ, which scales as √Re_c: a property of the chord Reynolds number, not of the cross-flow "
 "instability. The two facilities do not overlap in chord Reynolds number at all — "
 "%.2f–%.2f million against %.1f–%.1f — and the N they require "
 "at their own measured stations is %.1f–%.1f and %.1f–%.1f respectively "
 "(@@TAB:cf_recept@@), while the critical cross-flow Reynolds number is tight within each, at "
 % (_rtr.Re_c_min.iloc[0]/1e6, _rtr.Re_c_max.iloc[0]/1e6,
    _rtr.Re_c_min.iloc[1]/1e6, _rtr.Re_c_max.iloc[1]/1e6,
    _rcp.N_cf_min.iloc[0], _rcp.N_cf_max.iloc[0],
    _rcp.N_cf_min.iloc[1], _rcp.N_cf_max.iloc[1])
 + ("%.1f and %.1f" % (_rcp.Re_theta2_cov_pct.iloc[0], _rcp.Re_theta2_cov_pct.iloc[1]))
 + " per cent. The elimination behind the receptivity attribution was therefore "
 "incomplete: it had not considered that the branch's own rate carries no cross-flow physics, "
 "which is a defect of the model and not a property of the experiments.")
_pool_amp = float(pd.concat([_sw1.err_pct.abs(),
                             _sw2.err_pct_C1_150.abs()]).mean())
# The local-threshold variant, read from the table that scores it rather than
# typed beside the two figures that already are read.  Its three numbers stood
# here as 17.2, 55.1 and 32.4 with nothing regenerating them.
_cf_loc = _cff[_cff.formulation.str.startswith(
    "local C1 threshold, no")].iloc[0]
para("What would close it is a defined piece of work on the method rather than a request for "
 "measurements on two wings from 1960 and 1999: the Orr-Sommerfeld problem solved on the "
 "Falkner-Skan-Cooke CROSS-FLOW profile — which stability.fsc_profile already returns — and "
 "tabulated the way the streamwise rates of Sec. 4.3 are. Both closures were compared over "
 "both facilities while this was established: the amplification integral, which is the shipped "
 "form, gives %.1f and %.1f per cent (read from the two tables above, where they had been "
 "left at 21.8 and 51.1), the local C1 criterion %.1f and %.1f, pooled %.1f against %.1f. "
 "Neither dominates, so the shipped form is kept and the comparison recorded rather than the "
 "choice asserted."
 % (_e_cal, _e_ind, _cf_loc.calibration_err_pct, _cf_loc.independent_err_pct,
    _pool_amp, _cf_loc.pooled_err_pct))
table_from_csv("06_validation/crossflow_receptivity_summary.csv", key="cf_recept",
               cap="What each swept-wing facility requires, in the two currencies: a critical "
                   "cross-flow Reynolds number, which is consistent within each facility, and "
                   "an amplification factor, which is not comparable between them.")
para("TN D-338 reports crossflow Reynolds numbers of its own: \"the critical values ... for "
 "vortex formation ... range from about 135 to 190\", and \"the values ... for beginning "
 "transition were found to be between 190 and 260.\" It is tempting to note that the values "
 "this work's SURROGATE requires — %.0f on Dagenhart & Saric and %.0f on Boltz et al. — fall "
 "in those two ranges respectively, and to read that as the two datasets marking two "
 "different events. That reading does not survive checking, and it is recorded here as "
 "rejected rather than dropped. The surrogate is Re_theta2 = 0.47 Re_theta sin(L), a "
 "MOMENTUM-THICKNESS quantity; what a 1960 report means by a crossflow Reynolds number is "
 "w_max delta_10 / nu, the displacement-type one. This work computes that too, in the "
 "\"exact Falkner-Skan-Cooke\" column of @@TAB:cf_criticals@@, and it gives %.0f and %.0f — "
 "the first far above TN D-338's transition range, the second below its vortex-formation "
 "range, and the two INVERTED against the surrogate. Two quantities on different scales "
 "landing in the right intervals is a coincidence, not corroboration."
 % (_dag.mean_critical_value, _bol.mean_critical_value,
    _cfe.loc["Dagenhart & Saric (calibration)"].mean_critical_value,
    _cfe.loc["Boltz et al. (independent)"].mean_critical_value), italic=True, size=10)
# The second factor is the same quantity the section above gives as a
# percentage, so it is read from the same two means rather than typed as 1.5
# beside a "53 per cent more" derived four paragraphs earlier.
para("What does survive is weaker and qualitative. TN D-338 separates two events in ONE "
 "facility by roughly a factor of 1.4, and the two facilities here differ by a factor of "
 "%.2f. A gap of that size between experiments is therefore the size of the gap a single "
 "facility reports between vortex formation and the beginning of transition, so what each "
 "experiment CALLS transition remains a candidate explanation alongside receptivity — as an "
 "argument about event definition, not as a numerical match."
 % (_bol.mean_critical_value/_dag.mean_critical_value), italic=True, size=10)
para("One explanation can be ruled out rather than merely doubted. If the difference "
 "between the two facilities were a Reynolds-number effect the criterion is missing, the "
 "requirement would have to vary with chord Reynolds number in the same direction within a "
 "facility as it does between them. It does not. Within Dagenhart & Saric the required "
 "critical value FALLS steeply with chord Reynolds number, by %.0f per decade with a "
 "correlation of %+.2f over a factor of two in Re_c; Boltz et al. sit at six times that "
 "Reynolds number and require %.0f %% MORE, not less, and are themselves flat across a factor "
 "of three, at %+.0f per decade. The between-facility offset therefore has the opposite sign "
 "to the within-facility trend, and no monotone function of Re_c can carry one set into the "
 "other. That is what leaves receptivity — the leading-edge surface finish neither report "
 "documents — as the explanation, and it is now a measurement rather than an appeal to the "
 "literature."
 % (abs(_tr_dag.slope_per_decade_Re_c), _tr_dag.correlation,
    100.0*(_bol.mean_critical_value - _dag.mean_critical_value)/_dag.mean_critical_value,
    _tr_bol.slope_per_decade_Re_c))
para("Two numbers in this section have to be reconciled or they look inconsistent: the "
 "band is reported as C1 = 150–200, while @@TAB:cf_criticals@@ says the independent facility requires a "
 "critical Re_θ2 of 234. They are different quantities. C1 does not fire the branch; it "
 "starts the amplification integral, which then decides where transition is placed, so the "
 "EFFECTIVE critical value at the predicted station is higher than C1. Measured at the four "
 "Boltz conditions it is " + ("%.0f" % _sw2[f"Re_theta2_at_onset_C1_150"].mean()) +
 " for C1 = 150 and " + ("%.0f" % _sw2[f"Re_theta2_at_onset_C1_200"].mean()) +
 " for C1 = 200 — read from the table above rather than typed, where they stood at 157 and "
 "209 — so the integral adds about "
 "5 %, not the 17 % that would be needed to reach 234. That residual 12 % is precisely why "
 "the upper end of the band still leaves a " + ("%.1f" % _e_ind2) + " % error in location "
 "on the independent set, and it is the honest reason the band is offered as a bound rather "
 "than as a value.")
table_from_csv("06_validation/crossflow_reynolds_trend.csv",
               cap="The required critical value against chord Reynolds number "
                   "within each facility (crossflow_reynolds_trend.csv). The trends have "
                   "opposite signs to the offset between the facilities.")

h2("11.5  Ablations — what each element of the formulation is worth")
para("Each of the three elements that distinguish this formulation is switched off in turn, "
 "everything else held fixed, and the same two datasets the natural branch reaches are re-run. "
 "The table is regenerated by gen_validation.py and is not quoted from memory.")
table_from_csv("06_validation/ablations.csv",
               cap="Ablation study (ablations.csv): Schubauer-Skramstad onset error "
                   "and the 86 aerofoil conditions, one closure removed at a time.")
para("Calibration. The solver is calibrated through the single constant set of @@TAB:calibration@@. The "
 "critical amplification factor is not among the constants: it follows from the free-stream "
 "turbulence intensity by Mack's correlation, clamped to the 0.0008-0.0298 range over which "
 "that correlation is quoted. The SAME "
 "constants reproduce every dataset here — five flat plates, two swept wings and 86 "
 "aerofoil conditions — demonstrating that no case-specific physics tuning is required, which "
 "is the essence of the universality claim.")
para("Which sets are calibration and which are validation. The constants are frozen across "
 "every case, but three of them were SET on data in these tables, and a universality claim is "
 "only worth what its out-of-sample evidence is worth, so this is stated rather than left to "
 "be inferred. N_anchor, the units offset of the amplification scale, is set on the "
 "Schubauer & Skramstad plate. tu_hist, the weight on the flow history of Tu in the "
 "Abu-Ghannam & Shaw correlation, is set on the three ERCOFTAC T3 plates, which are the only "
 "bypass data here. CF_ratio, the cross-flow surrogate, is set on the Dagenhart & Saric wing. "
 "Everything else is out of sample: ERCOFTAC T3C4, which is the only test of the separation "
 "branch; all 86 NLF(1)-0416 conditions, which are the only test of the natural branch on a "
 "real aerofoil; and the Boltz et al. swept wing, which is the only independent test of the "
 "cross-flow branch. One constant per branch, each set on the smallest dataset that "
 "determines it, with the largest dataset for each branch held back.", italic=True, size=10)
h2("11.6  Sources and references")
para("All data sources used for validation, for calibration of the closures, and for the "
 "case-study definition are recorded below.")
table_from_csv("06_validation/sources_and_references.csv", max_rows=40,
               cap="Validation, calibration and case-study sources.")
table_from_csv("06_validation/swept_wing_source.csv",
               cap="Cross-flow calibration dataset provenance.")
table_from_csv("06_validation/swept_wing_independent_source.csv",
               cap="Independent swept-wing dataset provenance.")
table_from_csv("06_validation/aerofoil_nlf0416_source.csv",
               cap="Aerofoil dataset provenance.")

# ======================================================================
h1("12.  Contribution to Knowledge")
bullet("A single unified transition kernel (Eq. E13) that locally selects the governing "
       "mechanism among natural-TS, bypass, separation-induced and cross-flow transition by "
       "putting all four on one scale of onset progress and firing at the first to complete — "
       "reproducing all regimes with one calibration set, and continuous in the free-stream "
       "turbulence across the natural/bypass handover.")
bullet("An amplification database in place of an envelope correlation: 61,600 Orr-Sommerfeld "
       "eigenvalue solutions on the Falkner-Skan family, tabulated against shape factor, "
       "momentum-thickness Reynolds number and frequency, with the march carrying one "
       "amplification factor per physical frequency. The stability solver reproduces the "
       "Blasius neutral point at Re_theta = %.0f against the accepted 200.5, and the "
       "standard Blasius eigenvalue to better than a tenth of a per cent."
       % _NEUT_SOLVER)
bullet("A separation-bubble closure that predicts a length rather than a point: the shear layer "
       "is carried across the dead-air region by the momentum integral with no wall stress — a "
       "step with no fitted constant, which reproduces the growth the T3C4 hot films record — "
       "and reattachment placed where the disturbance has amplified by the same N_crit used "
       "elsewhere, so the length scales with the disturbance environment.")
# The figures quoted below are read from the generated CSVs, so this section
# cannot drift from the results it summarises.
_abl = pd.read_csv("06_validation/ablations.csv").set_index("configuration")
_nsum = pd.read_csv("06_validation/aerofoil_nlf0416_summary.csv").set_index("set")
_vsum = pd.read_csv("06_validation/validation_summary.csv")
_plate_err = ", ".join(
    "%+.1f %% on %s" % (r.Re_theta_t_err_pct,
                        r.case.split(" flat plate")[0].replace("ERCOFTAC ", ""))
    for _, r in _vsum.iterrows())
_all = _nsum.loc["All"]
bullet("A two-equation laminar march whose closures are computed from the Falkner-Skan family "
       "rather than fitted, giving the shape factor a history. On the 86 aerofoil conditions it "
       "raises the number of predictions inside the experimental bracket from %d to %d, improving "
       "both surfaces at once."
       % (_abl.loc["one-equation laminar march", "within_bracket"],
          _abl.loc["full model", "within_bracket"]))
bullet("Regime coverage with one constant set: transition-onset Re_theta_t predicted to "
       "%s, spanning 0.03-6 %% free-stream turbulence intensity." % _plate_err)
bullet("An aerofoil validation built for this work: 86 transition locations digitised from Fig. 9 "
       "of NASA TP-1861 for the NLF(1)-0416 section, both surfaces, four chord Reynolds numbers "
       "and lift coefficients from -1.03 to +1.62, with nothing calibrated on them. Mean error "
       "%.3f chord, and %d of the 86 predictions fall inside the +/-0.025c bracket within which "
       "the experiment itself localises transition."
       % (_all.mean_abs_err_c, _all.within_bracket))
bullet("A robust, panel-method-cost (<1 s) 3-D capability via a span-wise strip formulation with "
       "built-in cross-flow, suitable for design-loop use where RANS/LES are impractical.")
bullet("An end-to-end, auditable workflow (geometry → mesh → setup → solution → post-processing "
       "→ validation) producing a complete engineering output set (CSVs, curves, metrics, "
       "contours, temperature profiles, 3-D contours and vectors).")
_nvt = pd.read_csv("04_solution/nlf_vs_turbulent.csv")
bullet("Quantified NLF benefit for the case vehicle: %.0f %% laminar flow and %.0f %% viscous "
       "drag reduction relative to a fully-turbulent wing at the cruise design point."
       % (_nvt.mean_laminar_pct.iloc[0], _nvt.viscous_drag_reduction_pct.iloc[0]))

# ======================================================================
h1("13.  Conclusions")
para("The UTSS universal transition & skin-friction solver predicts boundary-layer transition "
 "over the three-dimensional NLF wing of the AETHER-NLF 25 and the dependent engineering "
 "quantities (skin friction, laminar extent, boundary-layer growth, separation margin, profile "
 "drag) at panel-method cost. The unified four-mechanism kernel, validated with a single "
 "calibration set against five flat plates, two independent swept-wing experiments and 86 "
 "aerofoil conditions, spans 0.03-6 %% free-stream turbulence intensity. At cruise the wing "
 "achieves %.0f %% laminar flow and a %.0f %% viscous-drag reduction versus a turbulent wing; at "
 "the higher-turbulence climb condition the solver switches to the bypass route and predicts "
 "early transition, demonstrating regime coverage across the flight envelope."
 % (_nvt.mean_laminar_pct.iloc[0], _nvt.viscous_drag_reduction_pct.iloc[0]))
# Both figures below are read from the CSVs.  The cross-flow gap had been left
# at a typed 42 %% against the generated 53 %%, and the count of declared
# conditions at three against the two the point-by-point file flags.
_gap_pct = (100.0*(_bol.mean_critical_value - _dag.mean_critical_value)
            / _dag.mean_critical_value)
_nlf_pts = pd.read_csv("06_validation/aerofoil_nlf0416.csv")
_n_declared = int(_nlf_pts.degenerate.sum())
para("Two limitations bound that claim and are stated here rather than left to be discovered. "
 "The cross-flow critical constant does not transfer between facilities: the two independent "
 "swept-wing experiments support the functional form of the criterion, and the measured data "
 "collapse on it, but reproducing the second requires a critical value %.0f %% larger than the "
 "first. That is now known not to be a defect of the criterion's form: the stationary "
 "cross-flow eigenvalue problem has been solved and checked against Dagenhart & Saric's own "
 "SALLY N-factors, and it brings the two levels closer — N_cf = %.1f against %.1f, a ratio of "
 "%.2f where the critical Reynolds numbers differ by %.2f — while scattering worse within each "
 "facility, so it is not adopted. What the branch is missing is a receptivity input — the "
 "roughness height that seeds a stationary cross-flow vortex — which neither report gives, so "
 "no stability calculation performed here can supply it. And the method is an attached-flow "
 "formulation, so it has an incidence envelope — but "
 "that envelope is asymmetric and is set by the section, not by a round number. On the "
 "NLF(1)-0416 it declares its first condition at %.1f° of incidence, where the upper-surface "
 "layer separates within two per cent of chord, and it handles the same section down to %.1f° "
 "without complaint; at positive incidence nothing in the 86 conditions, which reach +%.1f°, is "
 "declared. %d of the 86 fall outside it and the method says so rather than "
 "returning a location. An earlier version of this report quoted ±8.5°, which was a plotting "
 "threshold rather than a measured envelope, and a later one added that a leading-edge bubble "
 "does not appear on the case-study section itself until about +14°; that is not so — it "
 "appears at +6°, and above it the separation station is not even monotone in incidence, "
 "alternating between a leading-edge bubble and a trailing-edge one. A single positive-"
 "incidence bound is therefore not a meaningful thing to quote for that section, and the "
 "evidence for the asymmetry is the aerofoil set, where it is measured."
 % (_gap_pct, _ad.mean_N_cf, _ab.mean_N_cf,
    _ad.mean_N_cf/_ab.mean_N_cf,
    _bol.mean_critical_value/_dag.mean_critical_value,
    _nlf_pts[_nlf_pts.degenerate].alpha_deg.max(),
    _nlf_pts[~_nlf_pts.degenerate].alpha_deg.min(),
    _nlf_pts.alpha_deg.max(), _n_declared))

# ======================================================================
h1("Appendix A.  Complete Generated-Output Inventory")
para("Every file generated for this case study, by folder:")
def inventory():
    """Every generated data or figure file the repository tracks.

    solver/ is walked too.  The appendix is titled COMPLETE, and the two
    tabulated databases are generated outputs like any other - they are
    committed precisely because regenerating them costs minutes and hours -
    so leaving them out made the title false and, since 04_solution's field
    .npz was removed, left the NPZ count reading zero on a project that ships
    two of them.  assets/ is walked for the same reason: since gen_assets.py
    exists, the banner, the social card and their headline provenance file are
    generated outputs like any other.
    """
    rows=[]
    for root in ["01_geometry","02_mesh","03_model_setup","04_solution",
                 "05_postprocessing","06_validation","07_equations","solver",
                 "assets"]:
        for dp,_,fs in os.walk(root):
            if "__pycache__" in dp or "_slabs" in dp:
                continue
            for f in sorted(fs):
                ext=f.split(".")[-1].lower()
                # the Falkner-Skan families are build caches, rebuilt in well
                # under a minute and deliberately not tracked; the two
                # databases beside them are the generated artefacts
                if f.startswith("falkner_skan"):
                    continue
                if ext in ("csv","png","npz"):
                    rows.append([os.path.join(dp,f), ext.upper()])
    return rows
inv=inventory()
para(f"Total generated data/figure files: {len(inv)} "
     f"(CSV: {sum(1 for r in inv if r[1]=='CSV')}, "
     f"PNG: {sum(1 for r in inv if r[1]=='PNG')}, "
     f"NPZ: {sum(1 for r in inv if r[1]=='NPZ')}).")
manual_table(["file","type"], inv, cap="Generated-output inventory.")

# ======================================================================
h1("Appendix B.  Parameter / Metric CSVs Rendered as Figures")
para("For completeness, every remaining parameter and metric CSV is also "
     "rendered as a clean figure so that all generated CSV data appear in "
     "plotted form (the same data also appear as native tables earlier).")
# image() allocates and prints the figure number itself, so these captions
# carry only the caption.  They used to start "Fig. B1." and so on, and the
# rendered document duly read "Fig. 49. Fig. B1. geometry_definition.csv." -
# two numbers on one figure, in the only appendix that had them.
for f,c in [("table_geometry_definition","geometry_definition.csv, rendered as a figure."),
            ("table_mesh_metrics","mesh_metrics.csv, rendered as a figure."),
            ("table_material_properties","material_properties.csv, rendered as a figure."),
            ("table_solver_settings","solver_settings.csv, rendered as a figure."),
            ("table_integrated_forces","integrated_forces.csv, rendered as a figure.")]:
    image(f"05_postprocessing/csv_plots/{f}.png", width=5.9, cap=c)

# Every governing equation gen_equations.py writes must actually appear in the
# report.  Two were added to the index and placed nowhere, which no check
# noticed: the report is assembled from explicit key lists, so an equation can
# be defined, written to the CSV, rendered into model.equations.docx and still
# be absent from the document that claims to contain all of them.
_eq_missing_head = sorted({str(x) for x in eq_index["section"]} - _EQ_HEAD_USED)
if _eq_missing_head:
    raise SystemExit("equations_index.csv declares sections the report never "
                     "heads: %s" % ", ".join(_eq_missing_head))
_eq_unplaced = sorted(set(eq_index.index) - _EQ_PLACED)
if _eq_unplaced:
    raise SystemExit("equations defined in %s/equations_index.csv but never "
                     "placed in the report: %s" % (EQD, ", ".join(_eq_unplaced)))

resolve_refs(doc)
if _UNRESOLVED:
    raise SystemExit("cross-references in the report name keys that were never "
                     "allocated: %s" % ", ".join(sorted(_UNRESOLVED)))
doc.save("case.docx")
print("case.docx written:", os.path.getsize("case.docx")//1024, "KB")
print("figures embedded across", len(inv), "generated files")
