<p align="center">
  <img src="assets/banner.png" alt="AETHER-NLF 25 — Laminar to Turbulent Boundary-Layer Transition over a 3-D NLF Wing" width="100%">
</p>

# AETHER-NLF 25 — Laminar→Turbulent Transition Case Study

**Prediction of boundary-layer transition over a 3-D natural-laminar-flow aircraft wing, and the engineering quantities that depend on it.**

Author: **Akosa Samuel Onyejekwe** · Document UTSS-CASE-2026

---

## Overview

This repository contains a complete industrial-style aerodynamics case study on the prediction of **laminar-to-turbulent boundary-layer transition** over a three-dimensional aircraft wing, together with every downstream engineering quantity that the transition location drives:

- skin-friction drag and total profile drag,
- the extent of natural laminar flow (NLF) over the wing,
- boundary-layer growth (momentum thickness, shape factor),
- and the trailing-edge separation margin.

The study was produced with the **UTSS — Universal Transition & Skin-friction Solver**, a strip-based 3-D method that couples an inviscid panel solution to laminar and turbulent integral boundary-layer methods through a unified, multi-mechanism transition kernel.

The full written report — background, governing equations, input data, all generated results, validation, calibration, and sources — is included as **[`aero_turbulence_transition_report.pdf`](aero_turbulence_transition_report.pdf)**.

---

## The aircraft and flight envelope

The test article is the **AETHER-NLF 25**, a natural-laminar-flow wing using the custom **UTSS-NLF16** aerofoil section.

| Parameter | Value |
|---|---|
| Wing reference area | 33.15 m² |
| Span | 17.0 m |
| Aspect ratio | 8.72 |
| Root / tip chord | 2.60 m / 1.30 m (taper 0.50) |
| Mean aerodynamic chord | 2.022 m |
| Leading-edge sweep / dihedral / washout | 12° / 4° / −3° |
| Section | UTSS-NLF16, 16% thick, max-thickness at 42.2% chord, design Cl 0.45 |

Two operating points are analysed:

| Condition | Altitude | Mach | U∞ | Re(MAC) | Tu | α |
|---|---|---|---|---|---|---|
| **Cruise** | FL360 (11 km) | 0.42 | 123.9 m/s | 6.41 × 10⁶ | 0.07 % | 1.5° |
| **Climb** | FL100 (3 km) | 0.30 | 98.6 m/s | 10.7 × 10⁶ | 0.90 % | 3.5° |

---

## Method

```
Inviscid panel solution  →  Thwaites laminar boundary layer
        →  unified four-mechanism transition kernel
              (natural/Tollmien–Schlichting · bypass · separation-induced · cross-flow)
        →  Narasimha intermittency / transition length
        →  Head entrainment + Ludwieg–Tillmann turbulent boundary layer
        →  Squire–Young profile drag
        →  swept across the span (strip + cross-flow) for the full 3-D wing
```

A single, fixed calibration set (see `03_model_setup/calibration_constants.csv`) is used for every case — no per-case tuning.

---

## Headline results

**Transition location**

| Case | Surface | x_tr/c | Mechanism | Laminar run |
|---|---|---|---|---|
| Cruise | upper | 0.445 | TS / natural | 44.5% |
| Cruise | lower | 0.578 | TS / natural | 57.8% |
| Climb | upper | 0.025 | bypass | 2.5% |
| Climb | lower | 0.157 | bypass | 15.7% |

**Drag benefit of natural laminar flow (cruise)**

| Configuration | Profile drag | Mean laminar extent |
|---|---|---|
| NLF (UTSS-predicted transition) | 49.6 counts | 52.8% |
| Fully turbulent (LE trip) | 87.7 counts | 0% |
| **Viscous drag reduction** | **≈ 43%** | — |

---

## Results figures

A selection of generated outputs. The full set of curves, contours, profiles, and 3-D renders lives in [`05_postprocessing/`](05_postprocessing/) and [`06_validation/plots/`](06_validation/plots/).

<table>
  <tr>
    <td width="33%" align="center">
      <img src="05_postprocessing/csv_plots/cruise_Cp.png" width="100%"><br>
      <sub><b>Pressure distribution (cruise).</b> Upper/lower-surface C<sub>p</sub> over the UTSS-NLF16 section.</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/csv_plots/cruise_Cf.png" width="100%"><br>
      <sub><b>Skin friction &amp; BL state.</b> Laminar run shaded; the C<sub>f</sub> jump marks transition onset.</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/contours/contour_Cp_cruise.png" width="100%"><br>
      <sub><b>C<sub>p</sub> contour field (cruise).</b> Inviscid panel solution around the aerofoil.</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <img src="05_postprocessing/csv_plots/nlf_vs_turbulent.png" width="100%"><br>
      <sub><b>NLF benefit.</b> Profile drag of predicted-transition vs fully-turbulent wing (≈ 37% viscous reduction).</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/csv_plots/aero_polar.png" width="100%"><br>
      <sub><b>Aerodynamic polar.</b> Lift–drag behaviour of the wing section.</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/csv_plots/spanwise_transition.png" width="100%"><br>
      <sub><b>Spanwise transition.</b> Chordwise transition location swept across the span.</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <img src="05_postprocessing/profiles/bl_velocity_profiles.png" width="100%"><br>
      <sub><b>Boundary-layer velocity profiles.</b> Developing BL through laminar, transitional, and turbulent stations.</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/three_d/td_Cp.png" width="100%"><br>
      <sub><b>3-D surface C<sub>p</sub>.</b> Pressure mapped over the lofted 3-D wing.</sub>
    </td>
    <td width="33%" align="center">
      <img src="06_validation/plots/val_combined_Re_theta_t.png" width="100%"><br>
      <sub><b>Validation.</b> Re<sub>θt</sub> at transition: UTSS vs ERCOFTAC T3A/T3B and Schubauer–Skramstad data.</sub>
    </td>
  </tr>
</table>

---

## Validation

The solver is validated against three canonical published transition experiments, using **one calibration set**. The critical amplification factor is not among the constants: it is fixed by the free-stream turbulence intensity through Mack's correlation, which returns N_crit = 9.00 at the cruise level of 0.07%.

| Case | Tu (%) | Re_θt experiment | Re_θt UTSS | Error |
|---|---|---|---|---|
| Case | Tu (%) | criterion | Re_θt err | **x_tr err** |
|---|---|---|---|---|
| ERCOFTAC T3B | 5.95 | bypass | −8.2% | +4.4% |
| ERCOFTAC T3A | 3.04 | bypass | −11.9% | −5.0% |
| ERCOFTAC T3C4 (separation bubble) | 2.11 | **separation** | −29.1% | **−0.7%** |
| ERCOFTAC T3A⁻ | 0.87 | bypass | −22.3% | −37.6% |
| Schubauer & Skramstad | 0.03 | natural | +8.6% | +13.3% |

**T3C4** is the case that exercises the separation-induced branch: its adverse gradient drives the
shape factor to 5.17 and `Cf` to 1.8 × 10⁻⁴ before the layer reattaches turbulent. The criterion is
selected there and places the bubble to **−0.7%** in transition location. Its −29.1% in `Re_theta` is
a different statement — the measured value belongs to a *separated* shear layer while the march
carries an attached one, so for a bubble it is the location that can be compared. The method locates
a bubble; it does not resolve one.

**Cross-flow — 45° swept NLF(2)-0415** (Dagenhart & Saric, NASA/TP-1999-209344, Table 2). The
cross-flow criterion is the selected one at all six chord Reynolds numbers; mean absolute error in
transition location is **13.2%**, and the onset momentum-thickness Reynolds number is constant to
**0.3%** across the sweep — the evidence that the form of the criterion is right, not merely
well placed.

**Independent cross-flow check — NACA 64(2)A015, sweep 0–50°** (Boltz, Kenyon & Allen, NACA TN
D-338, 1960). A different facility, section and era, with the sweep varied rather than fixed, and
**nothing calibrated on it**. Mean absolute error **34.2%**, and the criterion selection is correct
in kind at every condition — amplification below 10° of sweep, cross-flow from 30° upwards. This,
not the 13.2% above, is the accuracy to expect on a new configuration.

| Λ | α | Re_c | x/c measured | x/c predicted | error | criterion |
|---|---|---|---|---|---|---|
| 0° | +4.0° | 6.27 × 10⁶ | 0.21 | 0.098 | −53.4% | natural |
| 10° | 0.0° | 1.50 × 10⁷ | 0.45 | 0.364 | −19.0% | natural |
| 30° | −3.0° | 7.13 × 10⁶ | 0.21 | 0.256 | +21.8% | cross-flow |
| 40° | −1.5° | 6.30 × 10⁶ | 0.35 | 0.197 | −43.8% | cross-flow |
| 50° | −1.0° | 7.36 × 10⁶ | 0.24 | 0.161 | −33.1% | cross-flow |

| Re_c | x/c measured | x/c predicted | error |
|---|---|---|---|
| 1.92 × 10⁶ | 0.78 | 0.680 | −12.8% |
| 2.19 × 10⁶ | 0.73 | 0.590 | −19.2% |
| 2.37 × 10⁶ | 0.58 | 0.543 | −6.4% |
| 2.73 × 10⁶ | 0.45 | 0.473 | +5.0% |
| 3.27 × 10⁶ | 0.33 | 0.395 | +19.6% |
| 3.73 × 10⁶ | 0.30 | 0.349 | +16.4% |

Full experiment-vs-solver data, plots, and bibliographic sources are in `06_validation/`.

The T3A and T3B reference values are the Rolls-Royce hot-wire measurements distributed as **ERCOFTAC Classic Collection Case 020**, with onset taken as the station of minimum measured `C_f`; no digitisation was performed here. For Schubauer & Skramstad only the onset Reynolds number is carried, at the value quoted throughout the literature.

**Why the flat-plate errors are what they are.** The Abu-Ghannam & Shaw correlation is not the
problem: evaluated at the *local* turbulence intensity at the *measured* onset it gives +9.6%,
−6.7% and −7.2% on T3A, T3A⁻ and T3B — ordinary scatter. What enlarges that is **conditioning**.
In a decaying stream the onset threshold rises as `Tu` falls while `Re_theta` grows only as
`sqrt(Re_x)`, so the two curves are near-parallel where they meet: perturbing the threshold by ±10%
moves the predicted transition location by a factor of **4 on T3A, 3 on T3A⁻, 2 on T3B**. A
correlation good to ten per cent cannot locate transition to ten per cent in such a flow. Finite
memory in the effective intensity, a cumulative crossing criterion, and carrying each rig's measured
free-stream velocity distribution were each tested; none improves the result.

**Flow history.** Abu-Ghannam & Shaw correlated onset against turbulence, pressure gradient *and flow history*. In a decaying stream the local intensity is not what the layer has experienced: applied locally the onset threshold outruns `Re_theta` and onset is predicted far downstream; applied at the inlet the decay is ignored and onset comes too early. The effective intensity is the mean of `Tu` over the boundary layer's own development in `Re_theta` — per unit of momentum-thickness growth rather than per unit of distance. It carries no fitted constant and reduces to the local value in a stream that does not decay, such as the free atmosphere.

**Scope of the validation.** Three of the four criteria are now supported by measurement — amplification by Schubauer & Skramstad, bypass by three ERCOFTAC plates, separation-induced by the T3C4 bubble, and cross-flow by two swept-wing experiments. All four criteria of the kernel are now selected somewhere and each is supported by measurement. The separation-induced and cross-flow criteria are carried in the kernel but are selected at no condition reported here and validated at none; the cross-flow branch further rests on an algebraic surrogate for the cross-flow momentum thickness whose coefficient is calibrated against design experience rather than measurement. The Prandtl-Glauert correction is applied to the pressure field and integrated loads only, the boundary-layer closures being incompressible formulations. Mean skin-friction error within the transitional region is 5.8-22.8% across the three cases.

---

## Repository structure

```
01_geometry/        Geometry definition CSVs (aerofoil, planform, 3-D sections) +
                    dimensioned engineering drawings in drawings/
02_mesh/            Surface & wall-normal grids, mesh metrics, and grid-independence
                    study, with plots/
03_model_setup/     Flow conditions, material (air) properties, solver settings,
                    and the fixed calibration constants
04_solution/        Solver outputs: surface Cp/Cf, momentum thickness, shape factor,
                    Re_theta, intermittency, aerodynamic polars, spanwise loading,
                    transition summary, and pressure fields (CSV + NPZ)
05_postprocessing/  Publication figures —
                      csv_plots/  all curves and data tables
                      contours/   Cp, speed, and vector contour maps
                      profiles/   boundary-layer velocity & temperature profiles
                      three_d/    3-D Cp, Cf, intermittency, skin-friction vectors
06_validation/      Experiment vs solver CSVs, validation plots, and sources
07_equations/       LaTeX source of every governing equation (equations_index.csv)
solver/             UTSS engine — utss_solver.py (panel method, boundary-layer
                    march, transition kernel), case_config.py (case definition
                    and validation cases), uplot.py (plot style)
run_solution.py     Runs the solver, writes every output CSV
gen_geometry.py     Geometry CSVs and dimensioned drawings
gen_mesh_setup.py   Grids, mesh metrics, grid-independence study, model setup
gen_validation.py   Validation against the three published experiments
gen_postprocessing.py  All curves, contours, profiles and 3-D renders
gen_equations.py    Native-equation document and the equations index
build_docx.py       Assembles the full report
aero_turbulence_transition_report.pdf   Full compiled report
```

---

## Reproducing the results

Every number, table and figure in this repository is regenerated by the scripts above.

```bash
pip install -r requirements.txt

python3 gen_geometry.py        # geometry + drawings
python3 run_solution.py        # run the solver, write all output CSVs
python3 gen_mesh_setup.py      # grids, metrics, grid independence, model setup
python3 gen_validation.py      # validation against the published experiments
python3 gen_postprocessing.py  # all plots, contours, profiles, 3-D renders
python3 gen_equations.py       # native-equation document + equations index
python3 build_docx.py          # assemble the full report
```

The whole pipeline runs in a few minutes on a desktop processor; a single two-dimensional operating point costs under one second.

### Method as implemented

The transition kernel evaluates four onset criteria at every marching station and takes the governing onset to be the earliest of them:

- **Natural / Tollmien–Schlichting** — envelope e^N amplification integrated in `Re_theta` using the Drela & Giles critical-Reynolds-number and growth-rate correlations. `N_crit` is *not* a free constant: it follows from the free-stream turbulence intensity through Mack's relation, returning 9.00 at the cruise level of 0.07%.
- **Bypass** — Abu-Ghannam & Shaw correlation at the local turbulence intensity, active above `Tu = 0.1%`.
- **Separation-induced** — laminar-bubble criterion on the Thwaites parameter, referred to the bypass value where live and to the AGS value at the local `Tu` otherwise.
- **Cross-flow** — C1 criterion on the cross-flow momentum-thickness Reynolds number, estimated from the streamwise value by an algebraic `theta2/theta` surrogate whose coefficient is calibrated against natural-laminar-flow design experience, not against measurement.

The inviscid solution carries a Prandtl–Glauert correction on `Cp` and the integrated loads only; the boundary-layer closures are incompressible formulations, so the correction is deliberately not applied to the edge velocity.

---

## Data formats

- **CSV** — all tabular geometry, setup, solution, and validation data; column headers are self-describing with units.
- **NPZ** — NumPy archives of the 2-D pressure fields (`04_solution/field_pressure_*.npz`).
- **PNG** — all figures, contours, profiles, drawings, and 3-D renders.

---

## Key references

- Abu-Ghannam B.J. & Shaw R. (1980), *Natural transition of boundary layers*, J. Mech. Eng. Sci. 22(5).
- Thwaites B. (1949), *Approximate calculation of the laminar boundary layer*, Aero. Quarterly 1.
- Head M.R. (1958), entrainment method; Ludwieg H. & Tillmann W. (1950), turbulent skin-friction law.
- Narasimha R. (1957); Dhawan S. & Narasimha R. (1958), J. Fluid Mech. 3.
- Arnal D. (1984), C1 cross-flow criterion; Mayle R.E. (1991), J. Turbomachinery 113.
- Roach P.E. & Brierley D.H. (1990), ERCOFTAC T3A/T3B; Langtry & Menter, AIAA J. 47(12), 2009.
- Schubauer G.B. & Skramstad H.K. (1948), NACA Report 909.

A complete, categorised bibliography is in `06_validation/sources_and_references.csv`.

---

## License & attribution

© 2026 Akosa Samuel Onyejekwe. Two licences apply, both non-commercial:

- **Source code** (`solver/`, `run_solution.py`, `gen_*.py`, `build_docx.py`) — [PolyForm Noncommercial 1.0.0](LICENSE-CODE). A software licence: it grants a patent licence and disclaims warranty, which a Creative Commons licence does not.
- **Data, figures and the report** — [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)

Under both, you may use, share and adapt this work for non-commercial purposes — including academic research, teaching and study — with appropriate credit to the author. For commercial use, please contact the author. Please credit the author when citing the method, data, or figures.
