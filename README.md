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
| Cruise | upper | 0.409 | TS / natural | 40.9% |
| Cruise | lower | 0.456 | TS / natural | 45.6% |
| Climb | upper | 0.047 | bypass | 4.7% |
| Climb | lower | 0.159 | bypass | 15.9% |

**Drag benefit of natural laminar flow (cruise)**

| Configuration | Profile drag | Mean laminar extent |
|---|---|---|
| NLF (UTSS-predicted transition) | 30.8 counts | 43.2% |
| Fully turbulent (LE trip) | 48.9 counts | 0% |
| **Viscous drag reduction** | **≈ 37%** | — |

---

## Validation

The solver is validated against three canonical published transition experiments, using the **one universal calibration set**:

| Case | Tu (%) | Re_θt experiment | Re_θt UTSS | Error |
|---|---|---|---|---|
| ERCOFTAC T3A flat plate (bypass) | 3.3 | 200 | 200.3 | 0.2% |
| ERCOFTAC T3B flat plate (high-Tu bypass) | 6.0 | 160 | 166.5 | 4.1% |
| Schubauer & Skramstad (natural) | 0.03 | 1100 | 1135.9 | 3.3% |

Full experiment-vs-solver data, plots, and bibliographic sources are in `06_validation/`.

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
aero_turbulence_transition_report.pdf   Full compiled report
```

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

© 2026 Akosa Samuel Onyejekwe. All rights reserved. Provided for review and educational reference. Please credit the author when citing the method, data, or figures.
