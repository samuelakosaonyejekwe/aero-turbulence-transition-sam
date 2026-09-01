# AETHER-NLF 25 — Turbulence-Transition Case Study (UTSS Universal Solver)

**Author: Akosa Samuel Onyejekwe**  ·  Document UTSS-CASE-2026


Industrial case study: **prediction of laminar→turbulent boundary-layer
transition over a 3-D aircraft wing** and the engineering quantities that
depend on it (skin-friction drag, laminar-flow extent, boundary-layer growth,
trailing-edge separation margin), using the novel **UTSS — Universal
Transition & Skin-friction Solver**.

The full report (background, problem, all equations, input data, every
generated output, validation, calibration, sources, contribution to knowledge)
is compiled in **`case.docx`**.

## Method (3-D)
Vortex-panel inviscid solution with a Karman–Tsien compressibility correction →
Thwaites laminar BL → **unified four-mechanism transition kernel**
(natural/TS · bypass · separation bubble · cross-flow) → Narasimha
intermittency → Head + Ludwieg–Tillmann turbulent BL evaluated at Eckert's
reference temperature → Squire–Young drag, swept across the span
(strip + cross-flow) for the full 3-D wing.

Two elements are not correlations:

* **Amplification database.** The e^N integral is driven by spatial growth
  rates read from `solver/amplification_db.npz`, built from 19,712
  Orr–Sommerfeld eigenvalue solutions on the Falkner–Skan family (Chebyshev
  collocation, Gaster transformation). The march carries one amplification
  factor per physical frequency. The stability solver returns the Blasius
  neutral point at Re_θ = 200 against the accepted 200.5. Regenerate with
  `python3 -c "import sys; sys.path.insert(0,'solver'); import stability;
  stability.build_database()"` (~4 min on 4 cores).
* **Separation-bubble closure.** The shear layer is carried across the dead-air
  region by the momentum integral with C_f = 0 and H frozen at its separation
  value; reattachment is placed where the disturbance has amplified by the same
  N_crit used elsewhere, so the bubble length scales with the disturbance
  environment (≈40 θ_s at Tu = 2.1 %, ≈180 θ_s at Tu = 0.03 %).

## Validation (one frozen constant set)

Flat plates — onset momentum-thickness Reynolds number:

| Case | Tu % | Re_θt exp | Re_θt UTSS | err | criterion |
|------|------|-----------|------------|-----|-----------|
| ERCOFTAC T3B  | 5.95 | 181.3 | 166.5 | −8.2 %  | bypass |
| ERCOFTAC T3A  | 3.04 | 272.3 | 240.0 | −11.9 % | bypass |
| ERCOFTAC T3C4 | 2.11 | 381.3 | 297.9 | −21.9 % | separation |
| ERCOFTAC T3A⁻ | 0.87 | 818.8 | 636.3 | −22.3 % | bypass |
| Schubauer & Skramstad | 0.03 | 1100 | 1115 | +1.4 % | natural |

Aerofoil — NLF(1)-0416, 86 transition locations digitised from NASA TP-1861
Fig. 9, both surfaces, four chord Reynolds numbers, c_l from −1.03 to +1.62.
Nothing is calibrated on this set; the experiment brackets transition within
the 0.05c orifice pitch, so its own uncertainty is ±0.025c:

| Set | Points | mean abs. err | bias | within ±0.025c |
|-----|--------|---------------|------|----------------|
| Upper surface | 46 | 0.049 c | +0.010 c | 18 (39 %) |
| Lower surface | 40 | 0.026 c | −0.006 c | 28 (70 %) |
| All           | 86 | 0.038 c | +0.003 c | 46 (53 %) |
| Within ±8.5° incidence | 78 | 0.029 c | −0.001 c | 44 (56 %) |

Swept wings — cross-flow: 13.2 % mean error on the 45° NLF(2)-0415 of
Dagenhart & Saric (on which the one cross-flow coefficient is set), 54.0 % on
the independent NACA 64(2)A015 of Boltz et al. digitised from NACA TN D-338.
The measured data collapse on the model's own similarity variable to 12 %, so
it is the critical constant and not the functional form that fails to transfer
between facilities.

Sources recorded in `06_validation/sources_and_references.csv`.

## Folder structure
```
01_geometry/      geometry CSVs + dimensioned engineering drawings (drawings/)
02_mesh/          surface & wall-normal grids, metrics, independence (plots/)
03_model_setup/   flow conditions, material properties, solver settings, calibration
04_solution/      solver outputs: Cp, Cf, theta, H, Re_theta, gamma, polars, fields
05_postprocessing/ csv_plots/ contours/ profiles/ three_d/  (all curves, contours, 3D)
06_validation/    experiment vs solver CSVs + plots + sources
07_equations/     equations_index.csv (LaTeX source of every governing equation)
solver/           utss_solver.py (engine), stability.py (Orr-Sommerfeld +
                  amplification database), case_config.py, uplot.py (style)
case.docx         FULL compiled report
```

## Reproduce
```bash
python3 gen_geometry.py        # geometry + drawings
python3 gen_mesh_setup.py      # mesh + model setup
python3 run_solution.py        # run solver, write all output CSVs
python3 gen_validation.py      # validation vs published data
python3 gen_postprocessing.py  # all plots, contours, profiles, 3D
python3 gen_equations.py       # build model.equations.docx (native equations)
python3 build_docx.py          # assemble case.docx
```

Plot rule enforced throughout: **no black** (navy ink + contrasting healthy
palette); text never overlaps the data.
