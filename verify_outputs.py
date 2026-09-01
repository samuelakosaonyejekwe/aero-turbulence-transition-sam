"""
verify_outputs.py
Check that the compiled report says what the generated CSVs say.

Every headline number in case.docx is read out of a CSV by build_docx.py, so
the two cannot disagree by construction - unless a number is typed into the
narrative by hand, which is how the mean laminar extent, the section design
c_l, the leading-edge radius, the Blasius neutral point and the wing lift all
came to be wrong at some point in this project's history.  This script closes
that loop: it reads the CSVs, then searches the RENDERED document for each
value.  Run it after build_docx.py.

    python3 verify_outputs.py                 # checks the PDF if present,
                                              # else case.docx
    python3 verify_outputs.py <file>          # .pdf or .docx

A .docx is checked through python-docx; a .pdf through PyMuPDF, which is the
only one of the two that sees what a reader sees.  Exits non-zero on any
failure so it can be wired into a build.
"""
import os
import sys
import pandas as pd


def document_text(path):
    if path.lower().endswith(".pdf"):
        import fitz
        with fitz.open(path) as d:
            return "\n".join(p.get_text() for p in d), d.page_count
    from docx import Document
    d = Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for r in t.rows:
            parts.append(" | ".join(c.text for c in r.cells))
    return "\n".join(parts), None


def checks():
    nvt = pd.read_csv("04_solution/nlf_vs_turbulent.csv")
    frc = pd.read_csv("04_solution/integrated_forces.csv").set_index("quantity")
    ts = pd.read_csv("04_solution/transition_summary.csv")
    geo = pd.read_csv("01_geometry/geometry_definition.csv").set_index("parameter")
    nsum = pd.read_csv("06_validation/aerofoil_nlf0416_summary.csv").set_index("set")
    vsum = pd.read_csv("06_validation/validation_summary.csv")
    abl = pd.read_csv("06_validation/ablations.csv").set_index("configuration")
    cl_row = next(i for i in geo.index if i.startswith("Section c_l"))

    def tr(case, surf, col):
        return ts[(ts.case == case) & (ts.surface == surf)][col].iloc[0]

    want = [
        ("mean laminar extent", f"{nvt.mean_laminar_pct.iloc[0]:.1f}"),
        ("viscous drag reduction", f"{nvt.viscous_drag_reduction_pct.iloc[0]:.1f}"),
        ("section C_d, counts", f"{nvt.Cd_counts.iloc[0]:.1f}"),
        ("fully turbulent C_d", f"{nvt.Cd_counts.iloc[1]:.1f}"),
        ("cruise upper x_tr/c", f"{tr('CRUISE','upper','x_tr_c'):.3f}"),
        ("cruise lower x_tr/c", f"{tr('CRUISE','lower','x_tr_c'):.3f}"),
        ("climb upper x_tr/c", f"{tr('CLIMB','upper','x_tr_c'):.3f}"),
        ("section c_l", str(geo.loc[cl_row, "value"])),
        ("wing C_L", str(frc.loc["Wing C_L (lifting line, taper + washout + sweep)", "value"])),
        ("span efficiency e", str(frc.loc["Span efficiency e (lifting line)", "value"])),
        ("trim incidence", str(frc.loc["Incidence for that C_L (lifting line)", "value"])),
        ("aerofoil mean abs err", f"{nsum.loc['All','mean_abs_err_c']:.4f}"),
        ("aerofoil within bracket", str(int(nsum.loc["All", "within_bracket"]))),
        ("ablation, full model", str(int(abl.loc["full model", "within_bracket"]))),
        ("ablation, one-equation march",
         str(int(abl.loc["one-equation laminar march", "within_bracket"]))),
    ]
    for _, r in vsum.iterrows():
        want.append(("Re_theta_t, " + r.case.split(" flat plate")[0],
                     f"{r.Re_theta_t_pred:g}"))

    # things that must appear, and things that must not
    present = [("Somers attribution", "Somers"),
               ("section 11.4", "11.4  What the two swept-wing")]
    absent = [("no stale McGhee attribution", "McGhee et al. NLF"),
              ("no missing figure", "[missing figure"),
              ("no missing CSV", "[missing CSV"),
              ("no equation fallback", "[missing eq"),
              # a cell squeezed to illegibility breaks mid-token; these are
              # fragments the wide-table split is there to prevent
              ("no cell broken mid-token", "ER\nCO\nFT\nAC")]
    return want, present, absent


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.path.exists("aero_turbulence_transition_report.pdf"):
        path = "aero_turbulence_transition_report.pdf"
    else:
        path = "case.docx"
    if not os.path.exists(path):
        sys.exit("no document to check: %s" % path)
    txt, pages = document_text(path)
    print("checking %s%s  (%d characters of text)"
          % (path, "" if pages is None else "  [%d pages]" % pages, len(txt)))

    want, present, absent = checks()
    bad = 0
    for label, value in want:
        ok = value in txt
        bad += not ok
        print("  %-8s %-30s %s" % ("OK" if ok else "MISSING", label, value))
    for label, needle in present:
        ok = needle in txt
        bad += not ok
        print("  %-8s %s" % ("OK" if ok else "MISSING", label))
    for label, needle in absent:
        ok = needle not in txt
        bad += not ok
        print("  %-8s %s" % ("OK" if ok else "FOUND", label))

    print("\n%d check(s) failed" % bad if bad else "\nall checks passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
