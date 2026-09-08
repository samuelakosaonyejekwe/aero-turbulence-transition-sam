<p align="center">
  <img src="assets/banner.png" alt="AETHER-NLF 25 — Laminar to Turbulent Boundary-Layer Transition over a 3-D NLF Wing" width="100%">
</p>

# AETHER-NLF 25 — Turbulence-Transition Case Study (UTSS Universal Solver)

**Author: Akosa Samuel Onyejekwe**  ·  Document UTSS-CASE-2026


Industrial case study: **prediction of laminar→turbulent boundary-layer
transition over a 3-D aircraft wing** and the engineering quantities that
depend on it (skin-friction drag, laminar-flow extent, boundary-layer growth,
trailing-edge separation margin), using the novel **UTSS — Universal
Transition & Skin-friction Solver**.

The full report (background, problem, all equations, input data, every
generated output, validation, calibration, sources, contribution to knowledge)
is committed as **[`aero_turbulence_transition_report.pdf`](aero_turbulence_transition_report.pdf)**.
`build_docx.py` regenerates it as `case.docx`, which is a build product and is
not tracked - this file used to point readers at that name, which a clone does
not contain.

## Method (3-D)
Vortex-panel inviscid solution with a Karman–Tsien compressibility correction →
**two-equation laminar BL** (momentum + kinetic energy, closures computed from
the Falkner–Skan family, so the shape factor carries its own history) →
**unified four-mechanism transition kernel**
(natural/TS · bypass · separation bubble · cross-flow) → Narasimha
intermittency, with the transitional extent from Dhawan & Narasimha's
Re_λ = 9 Re_x,t^0.75 → Head + Ludwieg–Tillmann turbulent BL evaluated at
Eckert's reference temperature → Squire–Young drag, swept across the span
(strip + cross-flow) for the full 3-D wing.

On a swept strip the profile drag is **not** the section lift's cos²Λ applied to
Squire–Young.  Two forces act — the chordwise wake deficit, which is what
Squire–Young returns, and the span-wise wall shear, which is not in that
deficit at all — and the span-wise momentum integral supplies the second in
closed form, so

    C_d = 2 (θ_TE,n/c_n) cosΛ [ cos²Λ (U_e,TE,n/U_n)^((H+5)/2)
                                + sin²Λ (U_e,TE,n/U_n) ]

with the two velocity-ratio powers differing because no Squire–Young
extrapolation applies to the span-wise deficit: downstream of the trailing edge
there is no wall, so it is frozen.  The **yawed flat plate** settles the form,
because its answer is known independently — a plate at yaw is a plate in the
free stream, so its drag is the unyawed value at the streamwise run length — and
at zero pressure gradient the expression returns exactly cosΛ for every sweep
and every shape factor, which is that answer.  The cos²Λ this replaced is short
by a further cosΛ: 2 % at the 12° of this wing, 29 % at 45°.  It is asserted in
`tools/smoke.py`, not described.  The correction raises the section profile drag
from 45.0 to 47.3 counts and leaves it within half a count of the same section
unswept, which is what 12° of sweep should do to a viscous drag; the drag
*reduction* is unchanged, both configurations scaling together.

**Where the drag is evaluated, and what that is actually worth.** Squire-Young
wants the trailing edge; this section has a **26.8° wedge** one, so the trailing
edge is a stagnation point of the inviscid flow, U_e goes to zero there
*physically*, and (U_e/U_∞)^((H+5)/2) degenerates — at the last control point it
returns 18 counts against 47.3. The evaluation is pulled forward to 0.98c, and
across the range either side the drag runs 42.5 → 47.3 counts.

That spread was reported here as a five-count uncertainty band. **It is not a
band — it is friction being correctly included.** Between 0.88c and 0.98c the
layer accumulates 3.94 counts of real skin friction, measured by integrating
C_f over the surface directly, and the formula moves 3.65: the same quantity to
a third of a count. A forward station is not a worse estimate of the same drag;
it is the drag of a shorter aerofoil. What matters is the friction the chosen
station still *omits*, and that is **0.157 counts at cruise, 0.128 at climb** —
under a fifth of a count. `04_solution/squire_young_station_sensitivity.csv`
carries the omitted friction beside the drag at every station and their sum,
which is the same number wherever it is evaluated. The station is converged to
a fifth of a count. Past 0.98c the formula turns over, which is the inviscid
singularity taking hold rather than drag being lost.

The shape-factor clamp is the same wedge. Head's method has no validity past
separation, so H is clamped at 2.8; on the climb case and above about 3° of
incidence the upper surface is on that clamp at the evaluation station — half
the polar — and `H_sy_at_clip` says so per surface and per point. A layer
decelerating into a stagnation point genuinely approaches separation, so the
clamp is reached for a physical reason and it is Squire-Young's
thin-attached-layer premise that fails, not the march. Closing it properly
needs **viscous–inviscid coupling** — displacement feedback that removes the
inviscid stagnation point — not the wake march this README previously named,
and it is worth that same fifth of a count. Marching this solver's own aft
state to the trailing edge and applying the wake relation returns 18 counts
against 47, because θ has by then grown to a per cent of chord in response to a
deceleration the viscous flow does not have.

The transition-length correlation is validated on the flat plates below, which
span Re_x,t = 6×10⁴ to 1.4×10⁶, and extrapolated on the wing, which transitions
at 3.7×10⁶.  The extrapolation is not damped — that would add an undeclared
constant — but what it costs is measured: sweeping the constant over a factor
of four moves the section drag by a tenth of a count
(`04_solution/transition_length_sensitivity.csv`).

Two elements are not correlations:

* **Amplification database.** The e^N integral is driven by spatial growth
  rates read from `solver/amplification_db.npz`, built from 61,600
  Orr–Sommerfeld eigenvalue solutions on the Falkner–Skan family (Chebyshev
  collocation, Gaster transformation), continued past separation onto the
  reverse-flow branch so that a separated profile has a computed rate too. The
  march carries one amplification factor per physical frequency. The stability
  solver puts the Blasius neutral point at Re_θ = 200 against the accepted
  200.5, and reproduces the standard Blasius Orr–Sommerfeld eigenvalue to
  better than a tenth of a per cent; what the march reads is the interpolated
  table, which first amplifies a few units above that — the node below the
  crossing holds an exact zero, so the offset is the resolution of that grid
  rather than an error in the eigenvalue solver.
  (`stability.tabulated_neutral_Re_theta()` computes it, and it is not restated
  here; this used to say "the nearest node of its Reynolds-number grid,
  Re_θ = 210", which is neither the value nor a node.)  Regenerate with
  `python3 -c "import sys; sys.path.insert(0,'solver'); import stability;
  stability.build_database()"` (minutes, on several cores); the build is checkpointed
  per shape factor, so an interrupted run resumes rather than restarting.
* **Separation-bubble closure.** The shear layer is carried across the dead-air
  region by the same two integral equations as the attached layer with the wall
  shear set to zero, so the shape factor keeps growing through the plateau
  (on T3C4 the shape factor rises through the plateau); reattachment is placed where the
  disturbance has amplified by the same N_crit used elsewhere, so the bubble
  length scales with the disturbance environment — measured, not asserted, into
  `06_validation/bubble_length_scaling.csv`: 28 θ_s on the separating plate at
  Tu = 2.11 % against a median of 208 over the 50 aerofoil bubbles at 0.03 %, a
  spread of 7.4.  (This line said ≈42 and ≈226, the report said "a spread of
  five and a half" and the solver's own comment said ≈40, ≈180 and "four and a
  half"; the claim is true and none of the three numbers was.)
  The amplification rate is **not fitted**: it is read
  from the reverse-flow branch of the tabulated family, which returns 0.0435 at
  Re_θ = 400 (0.042–0.045 over the range these bubbles span). The shape factor
  itself is marched on the *attached* branch and bounded by its separation value
  H = 3.997 — it cannot be continued past the fold, where H*(H) turns and the
  inversion the march needs ceases to exist. Only the amplification rate, which
  needs no inversion, is read beyond separation.

## Case-study result (AETHER-NLF 25 at cruise)

Every number here is read out of `04_solution/` by `verify_outputs.py` and
checked against the compiled report, so this table cannot drift from the solver:

| quantity | value |
|---|---|
| Section lift coefficient c_l | 0.517 |
| Section profile drag | 47.3 counts |
| Fully-turbulent reference (LE trip) | 95.4 counts |
| Viscous drag reduction | 50.4 % |
| Mean laminar extent | 55.4 % chord |
| Transition, cruise upper / lower | x/c = 0.542 / 0.566 (both natural-TS) |
| Wing C_L (lifting line, taper + washout + sweep) | 0.2548 |
| Span efficiency e | 0.8718 |
| Incidence for level flight at MTOW | 7.55° |

The section is quoted at its design incidence, which is not the aircraft trim
point — the last row is there so the two are not confused.

## Results figures

A selection of generated outputs. The full set of curves, contours, profiles
and 3-D renders lives in [`05_postprocessing/`](05_postprocessing/) and
[`06_validation/plots/`](06_validation/plots/).

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
      <sub><b>NLF benefit.</b> Profile drag of the predicted-transition wing against a fully turbulent one; the figure carries the current number.</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/csv_plots/aero_polar.png" width="100%"><br>
      <sub><b>Aerodynamic polar.</b> Lift curve, drag polar, L/D and transition location against incidence.</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/csv_plots/spanwise_transition.png" width="100%"><br>
      <sub><b>Span-wise transition.</b> Chordwise transition location swept across the span.</sub>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center">
      <img src="05_postprocessing/profiles/bl_velocity_profiles.png" width="100%"><br>
      <sub><b>Boundary-layer velocity profiles.</b> Laminar, transitional and turbulent stations on the cruise upper surface.</sub>
    </td>
    <td width="33%" align="center">
      <img src="05_postprocessing/three_d/td_Cp.png" width="100%"><br>
      <sub><b>3-D surface C<sub>p</sub>.</b> Pressure mapped over the lofted 3-D wing.</sub>
    </td>
    <td width="33%" align="center">
      <img src="06_validation/plots/val_combined_Re_theta_t.png" width="100%"><br>
      <sub><b>Validation.</b> Transition-onset Re<sub>&theta;t</sub> across all five flat plates, one calibration set.</sub>
    </td>
  </tr>
</table>

## Validation (one frozen constant set)

**Which sets are calibration and which are validation.** The constants are
frozen across every case, but three of them were *set* on data in this table,
and saying which is part of reporting the result:

| constant | set on | so these are |
|---|---|---|
| `N_anchor` — the e^N units offset | Schubauer & Skramstad plate | that plate is calibration |
| `tu_hist` — flow-history weight in Abu-Ghannam & Shaw | ERCOFTAC T3A, T3A⁻, T3B | those three are calibration |
| `CF_ratio` — cross-flow surrogate | Dagenhart & Saric 45° wing | that wing is calibration |

Everything else is out of sample: **T3C4** (the separation branch), all **86
NLF(1)-0416 aerofoil conditions**, and the **Boltz et al.** swept wing.

`N_anchor` used to be chosen at whatever value put the most of those 86
aerofoil predictions inside the experimental bracket, which made the largest
body of evidence in this work a calibration set while three places in the
project described it as one on which nothing is calibrated. It is now set on
the Schubauer & Skramstad plate alone, which it reproduces to 0.07 %. The cost
is reported rather than buried: the bracket count fell from 51 to 50 — which is
what happens when a constant stops being fitted to the set it is scored on.
The error figures are the table below, read from
`06_validation/aerofoil_nlf0416_summary.csv`; a before-and-after pair frozen in
prose stops describing the model the moment anything else moves, so only the
current values are quoted.

Flat plates — onset momentum-thickness Reynolds number:

| Case | Tu % | Re_θt exp | Re_θt UTSS | err | criterion |
|------|------|-----------|------------|-----|-----------|
| ERCOFTAC T3B  | 5.95 | 181.3 | 168.0 | −7.3 %  | bypass |
| ERCOFTAC T3A  | 3.04 | 272.3 | 282.1 | +3.6 %  | bypass |
| ERCOFTAC T3C4 | 2.11 | 309–381 † | 266.2 | -13.9 % † | separation |
| ERCOFTAC T3A⁻ | 0.87 | 818.8 | 683.6 | −16.5 % | bypass |
| Schubauer & Skramstad | 0.03 | 1100 | 1100.7 | +0.1 % | natural |

† T3C4's onset is not resolved to a station. Onset is taken as the station of
minimum C_f, and on this plate C_f is 1.87×10⁻⁴ at x = 1.295 m against
1.83×10⁻⁴ at 1.395 m — two per cent apart, at the hot-film floor. The two are
indistinguishable, so onset is bracketed by them and quoting the second alone
reports the *end* of the plateau as its beginning. Against the point value the
error is -30.2 %; against the bracket, -13.9 %.

Aerofoil — NLF(1)-0416, 86 transition locations digitised from NASA TP-1861
Fig. 9, both surfaces, four chord Reynolds numbers, c_l from −1.03 to +1.62.
Genuinely out of sample (see the calibration table above); the experiment
brackets transition within the 0.05c orifice pitch, so its own uncertainty is
±0.025c:

| Set | Points | mean abs. err | bias | within ±0.025c |
|-----|--------|---------------|------|----------------|
| Upper surface | 46 | 0.0389 c | -0.010 c | 23 (50 %) |
| Lower surface | 40 | 0.0382 c | -0.031 c | 27 (68 %) |
| All           | 86 | 0.0386 c | -0.020 c | 50 (58 %) |
| **Conditions the method accepts** | **84** | **0.0315 c** | -0.015 c | **50 (60 %)** |

The "All" row includes two conditions the method explicitly declares it cannot
handle — a burst bubble and a leading-edge bubble at x/c = 0.0016 — whose
predicted locations are meaningless either way. The accepted row is the one to
read.

Regenerated by `gen_validation.py` into `06_validation/aerofoil_nlf0416_summary.csv`.

The method now declares where it does not apply, rather than returning the last
station its march reached: a bubble that has not reattached by the trailing edge
has **burst**, and a layer separating within 2 % of chord has a **leading-edge
bubble**.  Two of the 86 are declared on those grounds and are kept in the
all-points row - the same two the paragraph above counts, which this line used
to give as three.

Ablations, everything else held fixed (all 86 aerofoil points):

| configuration | S&S | mean abs. err | within ±0.025c |
|---|---|---|---|
| no bubble closure | +0.1 % | 0.0630 c | 18/86 |
| one-equation laminar march | -10.0 % | 0.0415 c | 42/86 |
| Drela–Giles envelope | -11.2 % | 0.0435 c | 46/86 |
| **full model** | **+0.1 %** | **0.0386 c** | **50/86** |

Reproduced by `python3 gen_validation.py`, which writes `06_validation/ablations.csv`;
pass `--no-ablations` to skip the sweep.

Swept wings — cross-flow, on the 45° NLF(2)-0415 of Dagenhart & Saric (on which
the one cross-flow coefficient is set) and the independent NACA 64(2)A015 of
Boltz et al. digitised from NACA TN D-338:

| set | C1 = 150 (frozen constant) | C1 = 200 |
|---|---|---|
| Dagenhart & Saric — calibration | **21.8 %** | — |
| Boltz et al. — independent | 51.1 % | **17.6 %** |

Both columns of the independent set are computed and tabulated by
`gen_validation.py` into `06_validation/swept_wing_independent.csv`; the spread
between them is the limitation, not a choice to be made per case.

What the two experiments actually require is measured rather than asserted, in
`06_validation/crossflow_criticals.csv`: the march is run with every branch
disabled so that it reaches the measured transition station, and the criterion
is evaluated there.

| set | critical Re_θ2 | coeff. of variation | points |
|---|---|---|---|
| Dagenhart & Saric | 153 | 17.8 % | 6 |
| Boltz et al. | 234 | 4.0 % | 4 |
| pooled | 185 | 24.4 % | 10 |

Each facility is internally consistent — Boltz to 4 % across four sweep angles
and a factor of three in chord Reynolds number — and the two differ by 53 %.

**A tempting external corroboration, checked and rejected.** TN D-338 states
crossflow Reynolds numbers of its own: *"the critical values ... for vortex
formation ... range from about 135 to 190"*, and *"the values ... for beginning
transition were found to be between 190 and 260."* The values this work's
**surrogate** requires — 153 on Dagenhart and 234 on Boltz — fall in those two
ranges respectively, which looks like the two datasets marking two different
events. It does not hold. The surrogate Re_θ2 = 0.47 Re_θ sin Λ is a
*momentum-thickness* quantity; what a 1960 report means by a crossflow Reynolds
number is w_max·δ₁₀/ν, the displacement-type one. This work computes that too —
the "exact Falkner–Skan–Cooke" rows of `crossflow_criticals_summary.csv` — and
it gives **429 and 118**: the first far above their transition range, the second
below their vortex-formation range, and the two *inverted* against the
surrogate. Two quantities on different scales landing in the right intervals is
a coincidence. It is recorded as rejected rather than dropped.

What survives is weaker and qualitative: TN D-338 separates two events within
*one* facility by about a factor of 1.4, and the two facilities here differ by a
factor of 1.5. So what each experiment *calls* transition remains a candidate
explanation alongside receptivity — as an argument about event definition, not
as a numerical match.  Two attempts to close the gap fail and are recorded rather than
dropped: the exact Falkner–Skan–Cooke factor K(λ) in place of the constant
surrogate makes it worse (pooled variation 77 % against 24 %, and the ratio
between the two sets inverts), and giving the cross-flow branch its own
amplification threshold,
separate from Mack's, does not help either: swept from N = 2 to 12 with C1
refitted on the calibration set at each value, the independent set goes from
55.1 % to 61.9 % — away from agreement, not towards it
(`06_validation/crossflow_threshold_sweep.csv`).  This used to say "only from
55 % to 51 %", which had the size and the sign wrong.  The reason it cannot
help is that once Re_θ2 exceeds C1 the amplification builds fast enough that
the threshold is nearly redundant with C1 itself, which the barely-moving
refitted C1 column shows directly.  The
branch is closed by an **amplification integral** rather than a local threshold
— a stationary cross-flow vortex must grow before it breaks down — using the
computed rate 0.0435 and the same N_crit as every other branch, so it adds no
constant.
Seven formulations of the branch are scored against both experiments in
`06_validation/crossflow_formulations.csv`, and none reconciles them: every one
that helps the independent set costs more on the calibration set, and the two
that cost nothing there help nothing here.  That used to be a claim about what
the author had tried, which nothing could check; it is now a generated table.
Whether the branch needs its own amplification threshold, separate from the
N_crit Mack's relation supplies for TS waves, is settled the same way in
`06_validation/crossflow_threshold_sweep.csv`, which refits C1 on the
calibration set at each threshold so the two constants are not confounded.  The two require critical values differing by about half, and
that difference is in none of the mean-flow quantities the method computes —
which the exact Falkner–Skan–Cooke similarity solution establishes.  It is a
**receptivity** difference: stationary cross-flow vortices are seeded by
leading-edge roughness, and neither report documents the surface finish.  The
branch is therefore reported with a band, **C1 = 150–200**, and every other
result in this work uses the frozen C1 = 150.

### Where the residuals come from

**Not in the laminar branch.** Against the model's own marched momentum
thickness at the same stations — not against flat-plate Blasius, which is the
wrong reference for the one plate that has a pressure gradient — the measured
laminar layer agrees to within 6 % on all four plates with C_f data, and to 2 %
on T3C4.

**T3C4: the momentum integral cannot be closed with this experiment's own
data.** The shape-factor cap is gone — H\*(H) folds at H = 4.03 where the
attached and reverse branches meet, but a march knows which branch it is on, so
taking the root nearest the previous H carries it across; and the reverse branch
itself, stopped at H = 4.99 by a continuation parameter, continues smoothly to
H = 6.41, past the measured 5.17. That was not the leading term. Across the
measured bubble, reproducing the measured dθ/dx = 0.00591 m⁻¹ from the measured
dU_e/dx and θ requires **H = 19**; the measurement reports 5.17, which gives
0.00202 m⁻¹ — short by a factor of **2.93**. No integral method closed on any
physical profile family reproduces this bubble. Combined with the unresolved
onset station (see † above), the residual is −14.2 % against the bracket, and
what remains of it is a property of the data, not of the closure.

**Cross-flow: the gap is a critical Reynolds number, and this method cannot
convert it to a roughness ratio.** Expressing it in amplification units is the
natural move — ΔN = ln(A₀,₁/A₀,₂) is the roughness ratio for a roughness-seeded
vortex — and it fails, for a reason worth stating. The rate this branch
integrates is read off a separated *streamwise* profile and is nearly
Reynolds-independent (0.0417 at Re_θ = 200 against 0.0461 at 8000), so N_cf
scales as √Re_c. The two facilities differ 7× in Re_c, and the N they require is
0.0–11.9 and 43.1–129.2 — not comparable — while the critical Re_θ2 is tight
within each (17.8 % and 4.0 %). The receptivity attribution was reached by an
elimination that had not considered that the branch's own rate carries no
cross-flow physics. **What would close it is named**: the Orr–Sommerfeld problem
solved on the Falkner–Skan–Cooke *cross-flow* profile — which
`stability.fsc_profile` already returns — tabulated as the streamwise rates are.
That is work on the method, not a request for measurements on wings from 1960
and 1999.

**Transition length: not an extrapolation.** Dhawan & Narasimha's
Re_λ = 9 Re_x,t^0.75 *is* Narasimha's spot model at constant dimensionless spot
formation rate: Re_λ = √(0.412/N̂) Re_θ,t^1.5, and on a Blasius plate
Re_θ^1.5 = 0.5411 Re_x^0.75, so the two are the same law with C = 16.63 —
identical over Re_x from 6×10⁴ to 10⁷. What is assumed constant is the spot
rate, and Re_θ is the variable in which that is exact. The forms agree to
0.03 % on the four ZPG plates and differ by 2.85 % on T3C4, the one plate with
a pressure gradient. The wing moves 0.02 counts and no longer sits outside any
validated range.

**Conditioning.** Shifting the bypass threshold by ±10 % moves the predicted
transition location by a factor of 3.5 on T3A, 2.5 on T3A⁻ and 2.0 on T3B
(`06_validation/residual_diagnostics.csv`).

**Cross-flow.** One explanation is ruled out rather than doubted: within
Dagenhart & Saric the required critical value *falls* with chord Reynolds number
(see 06_validation/crossflow_reynolds_trend.csv), while Boltz sits at six times the Reynolds number
and requires 53 % *more*. The between-facility offset has the opposite sign to
the within-facility trend, so no monotone function of Re_c carries one set into
the other (`06_validation/crossflow_reynolds_trend.csv`). That leaves
receptivity — the leading-edge finish neither report documents — by elimination.

Sources recorded in `06_validation/sources_and_references.csv`.

## Folder structure
```
01_geometry/      geometry CSVs + dimensioned engineering drawings (drawings/)
02_mesh/          surface & wall-normal grids, metrics, independence (plots/)
03_model_setup/   flow conditions, material properties, solver settings, calibration
04_solution/      solver outputs: Cp, Cf, theta, H, Re_theta, gamma, polars, fields
05_postprocessing/ csv_plots/ contours/ profiles/ three_d/  (all curves, contours, 3D)
06_validation/    experiment vs solver CSVs + plots + sources, the aerofoil
                  error summary (aerofoil_nlf0416_summary.csv) and the
                  ablation sweep (ablations.csv)
07_equations/     equations_index.csv (LaTeX source of every governing equation)
solver/           utss_solver.py (engine), stability.py (Orr-Sommerfeld +
                  amplification database), case_config.py, uplot.py (style)
tools/            smoke.py (25 checks over the whole solver, well under a
                  minute), pipeline.py (staged regeneration, timings written to
                  .pipeline/timings.json), baseline.py (numeric diff of every
                  generated CSV against a snapshot)
utss_paths.py     anchors every entry point to the repository root, so a
                  generator run from another directory still reads and writes
                  here
verify_outputs.py checks the compiled report against the generated CSVs
aero_turbulence_transition_report.pdf   FULL compiled report (tracked)
case.docx         the same report as .docx - a build product, not tracked
```

## Reproduce
```bash
python3 tools/smoke.py         # 25 checks over the whole solver, under a minute.
                               #   Run this FIRST and after every edit: a full
                               #   regeneration is minutes and the faults
                               #   that waste it are all visible here in the
                               #   first second.
python3 tools/pipeline.py      # the whole regeneration, as dependency-ordered
                               #   stages, two at a time, smoke-gated and
                               #   timed, stopping at the first failure with
                               #   that stage's log.  Per-stage timings land
                               #   in .pipeline/timings.json; quoting them
                               #   here is how they went stale.
```

The stages can still be run one at a time, in this order, from any directory:

```bash
python3 gen_geometry.py        # geometry + drawings
python3 gen_mesh_setup.py      # mesh + model setup
python3 run_solution.py        # run solver, write all output CSVs
python3 gen_validation.py      # validation vs published data + ablation sweep
                              #   (--no-ablations skips the sweep)
python3 gen_postprocessing.py  # all plots, contours, profiles, 3D
                              #   (reads 04_solution/, so run it after
                              #    run_solution.py)
python3 gen_equations.py       # build model.equations.docx (native equations)
python3 build_docx.py          # assemble case.docx
python3 verify_outputs.py      # check the compiled report against the CSVs
```

To see exactly what a change moved, snapshot before and diff after:

```bash
python3 tools/baseline.py save /tmp/before
python3 tools/pipeline.py
python3 tools/baseline.py diff /tmp/before --quiet-same
```

Self-checks, all of which run on the numbers rather than quoting them:

```bash
python3 solver/utss_solver.py  # off-body field vs the surface solution;
                               #   T3A onset; the intermittency blend
python3 solver/stability.py    # Blasius H, f''(0) and neutral point; the
                               #   exact Thwaites closure; the tabulated
                               #   cross-flow factor against a direct
                               #   Falkner-Skan-Cooke solve; and what
                               #   Gaster's transformation costs
python3 verify_outputs.py      # every headline number in the rendered PDF
                               #   (or case.docx) read back and compared with
                               #   the CSV it came from
```

`run_solution.py` additionally asserts the lifting-line solve against the one
case with a closed-form answer - an elliptic planform, for which it must return
a span efficiency of exactly 1 and the exact lift-curve slope - before it
writes anything.

The smoke set also holds the solver to results published outside this work:
the Falkner-Skan wall shear f''(0) at five values of beta (to 5e-5); the
standard Blasius Orr-Sommerfeld benchmark, c = 0.36412 + 0.00796i at
Re_delta* = 998 and alpha delta* = 0.308, which it reproduces to 1e-5 in phase
speed and better than 0.1 % in growth rate; a symmetric section carrying exactly zero lift
at zero incidence and lift exactly odd in incidence; and Kutta-Joukowski, the
lift from the pressure integral against the lift from the circulation.

Plot rule enforced throughout: **no black** (navy ink + contrasting healthy
palette); text never overlaps the data.

## License & attribution

© 2026 Akosa Samuel Onyejekwe. Two licences apply, both non-commercial:

- **Source code** (`solver/`, `run_solution.py`, `gen_*.py`, `build_docx.py`) — [PolyForm Noncommercial 1.0.0](LICENSE-CODE). A software licence: it grants a patent licence and disclaims warranty, which a Creative Commons licence does not.
- **Data, figures and the report** — [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)

Under both, you may use, share and adapt this work for non-commercial purposes — including academic research, teaching and study — with appropriate credit to the author. For commercial use, please contact the author. Please credit the author when citing the method, data, or figures.
