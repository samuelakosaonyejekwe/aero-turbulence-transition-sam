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
**two-equation laminar BL** (momentum + kinetic energy, closures computed from
the Falkner–Skan family, so the shape factor carries its own history) →
**unified four-mechanism transition kernel**
(natural/TS · bypass · separation bubble · cross-flow) → Narasimha
intermittency → Head + Ludwieg–Tillmann turbulent BL evaluated at Eckert's
reference temperature → Squire–Young drag, swept across the span
(strip + cross-flow) for the full 3-D wing.

Two elements are not correlations:

* **Amplification database.** The e^N integral is driven by spatial growth
  rates read from `solver/amplification_db.npz`, built from 61,600
  Orr–Sommerfeld eigenvalue solutions on the Falkner–Skan family (Chebyshev
  collocation, Gaster transformation), continued past separation onto the
  reverse-flow branch so that a separated profile has a computed rate too. The march carries one amplification
  factor per physical frequency. The stability solver returns the Blasius
  neutral point at Re_θ = 200 against the accepted 200.5. Regenerate with
  `python3 -c "import sys; sys.path.insert(0,'solver'); import stability;
  stability.build_database()"` (~4 min on 4 cores).
* **Separation-bubble closure.** The shear layer is carried across the dead-air
  region by the momentum integral with C_f = 0 and H frozen at its separation
  value; reattachment is placed where the disturbance has amplified by the same
  N_crit used elsewhere, so the bubble length scales with the disturbance
  environment (≈40 θ_s at Tu = 2.1 %, ≈180 θ_s at Tu = 0.03 %). The
  amplification rate is **not fitted**: it is read from the reverse-flow branch
  of the tabulated family, which returns 0.0435.  The build is checkpointed per
  shape factor, so an interrupted run resumes rather than restarting.

## Validation (one frozen constant set)

Flat plates — onset momentum-thickness Reynolds number:

| Case | Tu % | Re_θt exp | Re_θt UTSS | err | criterion |
|------|------|-----------|------------|-----|-----------|
| ERCOFTAC T3B  | 5.95 | 181.3 | 168.0 | −7.3 %  | bypass |
| ERCOFTAC T3A  | 3.04 | 272.3 | 282.1 | +3.6 %  | bypass |
| ERCOFTAC T3C4 | 2.11 | 381.3 | 273.8 | −28.2 % | separation |
| ERCOFTAC T3A⁻ | 0.87 | 818.8 | 683.6 | −16.5 % | bypass |
| Schubauer & Skramstad | 0.03 | 1100 | 1162 | +5.6 % | natural |

Aerofoil — NLF(1)-0416, 86 transition locations digitised from NASA TP-1861
Fig. 9, both surfaces, four chord Reynolds numbers, c_l from −1.03 to +1.62.
Nothing is calibrated on this set; the experiment brackets transition within
the 0.05c orifice pitch, so its own uncertainty is ±0.025c:

| Set | Points | mean abs. err | bias | within ±0.025c |
|-----|--------|---------------|------|----------------|
| Upper surface | 46 | 0.041 c | +0.002 c | 25 (54 %) |
| Lower surface | 40 | 0.032 c | −0.016 c | 28 (70 %) |
| All           | 86 | 0.037 c | −0.007 c | 53 (62 %) |
| Conditions the method accepts | 83 | 0.028 c | −0.004 c | 53 (64 %) |

The method now declares where it does not apply, rather than returning the last
station its march reached: a bubble that has not reattached by the trailing edge
has **burst**, and a layer separating within 2 % of chord has a **leading-edge
bubble**.  Three of the 86 are declared on those grounds and are kept in the
all-points row.

Ablations, everything else held fixed (all 86 aerofoil points):

| configuration | S&S | mean abs. err | within ±0.025c |
|---|---|---|---|
| no bubble closure | +5.6 % | 0.058 c | 19/86 |
| one-equation laminar march | −4.7 % | 0.041 c | 45/86 |
| Drela–Giles envelope | −6.6 % | 0.041 c | 52/86 |
| **full model** | **+5.6 %** | **0.037 c** | **53/86** |

Swept wings — cross-flow: 13.7 % mean error on the 45° NLF(2)-0415 of
Dagenhart & Saric (on which the one cross-flow coefficient is set), 55.1 % on
the independent NACA 64(2)A015 of Boltz et al. digitised from NACA TN D-338 (55.1 %).
Marching each of the ten conditions laminar to its **measured** transition
station and evaluating the cross-flow criterion there gives a coefficient of
variation of **21 %** across both facilities — six sweep angles, a 14× range of
chord Reynolds number, two sections and two eras.  The criterion therefore
transfers in the quantity it is posed on; the transition *location* is an
ill-conditioned function of it, which is why 21 % in the criterion becomes
30–55 % in location.  No single C1 improves the combined result.

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
