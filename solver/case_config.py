"""
case_config.py  -  Single source of truth for the industrial case study.

CASE: Natural-Laminar-Flow (NLF) wing of the "AETHER-NLF 25" regional
business jet technology demonstrator.  Goal: predict boundary-layer
transition over the wing upper/lower surfaces at cruise to quantify
laminar-flow extent and viscous (skin-friction) drag.
"""
import numpy as np

# ----------------------------------------------------------------------
AIRCRAFT = dict(
    name        = "AETHER-NLF 25",
    role        = "Regional business jet / natural-laminar-flow demonstrator",
    mtow_kg     = 8500.0,
    n_pax       = 8,
)

# ---- Wing planform (trapezoidal, mild sweep to preserve NLF) ----------
WING = dict(
    span_b        = 17.00,    # m, full span
    root_chord    = 2.60,     # m
    tip_chord     = 1.30,     # m
    le_sweep_deg  = 12.0,     # deg, leading-edge sweep
    dihedral_deg  = 4.0,      # deg
    twist_tip_deg = -3.0,     # deg washout
    section       = "UTSS-NLF16",
)
WING["taper"]   = WING["tip_chord"]/WING["root_chord"]
WING["area_S"]  = 0.5*(WING["root_chord"]+WING["tip_chord"])*WING["span_b"]
WING["MAC"]     = (2/3)*WING["root_chord"]*(1+WING["taper"]+WING["taper"]**2)/(1+WING["taper"])
WING["AR"]      = WING["span_b"]**2/WING["area_S"]

# ---- Cruise flight condition (FL360, M=0.42) --------------------------
CRUISE = dict(
    altitude_m  = 11000.0,
    mach        = 0.42,
    T_inf_K     = 216.65,
    p_inf_Pa    = 22632.0,
    rho_inf     = 0.36392,    # kg/m^3
    mu_inf      = 1.422e-5,   # Pa.s (Sutherland)
    a_sound     = 295.07,     # m/s
    Tu_pct      = 0.07,       # free-atmosphere turbulence intensity
    alpha_deg   = 1.5,        # design incidence (~ design Cl 0.4)
    gamma_air   = 1.4,
    R_air       = 287.05,
    Pr          = 0.72,
    recovery_r  = 0.89,       # turbulent recovery factor ~ Pr^(1/3)
)
CRUISE["U_inf"] = CRUISE["mach"]*CRUISE["a_sound"]
CRUISE["nu_inf"] = CRUISE["mu_inf"]/CRUISE["rho_inf"]
CRUISE["Re_MAC"] = CRUISE["U_inf"]*WING["MAC"]/CRUISE["nu_inf"]
CRUISE["q_inf"]  = 0.5*CRUISE["rho_inf"]*CRUISE["U_inf"]**2

# ---- Climb / off-design condition (higher Tu -> bypass) ---------------
CLIMB = dict(
    altitude_m  = 3000.0, mach=0.30, T_inf_K=268.66, rho_inf=0.9093,
    mu_inf=1.694e-5, a_sound=328.58, Tu_pct=0.90, alpha_deg=3.5,
)
CLIMB["U_inf"]=CLIMB["mach"]*CLIMB["a_sound"]
CLIMB["nu_inf"]=CLIMB["mu_inf"]/CLIMB["rho_inf"]
CLIMB["Re_MAC"]=CLIMB["U_inf"]*WING["MAC"]/CLIMB["nu_inf"]

# ----------------------------------------------------------------------
# UTSS-NLF16 : designed natural-laminar-flow section
#   16% thick, max thickness pushed aft to ~0.42c (favourable gradient),
#   mild aft camber for design Cl ~ 0.4.
# ----------------------------------------------------------------------
def nlf16_coords(n=130, t=0.16, xt=0.42, camber=0.030):
    beta = np.linspace(0.0, np.pi, n+1)
    x = 0.5*(1.0 - np.cos(beta))               # cosine clustering
    # thickness: rounded LE (sqrt), peak shifted aft, closed TE
    base = np.sqrt(x)*(1.0 - x)
    shift = 1.0 + 1.6*x - 0.9*x**2             # bias the peak rearward
    yt = base*shift
    yt = yt/np.max(yt)*(t/2.0)                  # scale to half-thickness
    yt = yt*(1.0 - 0.0*x)                        # (closed TE already ~0)
    # aft-loaded camber line (cubic, loaded behind mid-chord)
    yc = camber*(1.6*x*(1-x) + 1.1*x**2*(1-x))
    dyc = np.gradient(yc, x)
    thb = np.arctan(dyc)
    xu = x - yt*np.sin(thb); yu = yc + yt*np.cos(thb)
    xl = x + yt*np.sin(thb); yl = yc - yt*np.cos(thb)
    return dict(x=x, yc=yc, yt=yt, xu=xu, yu=yu, xl=xl, yl=yl)


def nlf16_panel_points(n=130, **kw):
    """Closed-loop points ordered TE -> lower -> LE -> upper -> TE."""
    c = nlf16_coords(n=n, **kw)
    X = np.concatenate([c["xl"][::-1], c["xu"][1:]])
    Y = np.concatenate([c["yl"][::-1], c["yu"][1:]])
    return X, Y


# ---- Validation cases (real, published reference datasets) ------------
VALIDATION = dict(
    # L_turb is the integral length scale of the oncoming grid turbulence,
    # obtained by fitting the k-epsilon decay law to the measured Tu(x) of
    # each rig (ERCOFTAC case 020).  It is a property of the tunnel, not of
    # the transition model, and it fixes the local turbulence intensity so
    # that no choice between "inlet" and "local" values arises.
    T3A = dict(name="ERCOFTAC T3A flat plate (bypass transition)",
               U=5.4, Tu_pct=3.043, nu=1.5e-5, L=1.7, dUe=0.0, L_turb=1.53e-3,
               source="Roach & Brierley (1990), ERCOFTAC T3A; "
                      "Savill (1993); Langtry & Menter, AIAA J. 47(12) 2009."),
    T3B = dict(name="ERCOFTAC T3B flat plate (high-Tu bypass)",
               U=9.4, Tu_pct=5.952, nu=1.5e-5, L=1.7, dUe=0.0, L_turb=3.83e-3,
               source="Roach & Brierley (1990), ERCOFTAC T3B; "
                      "Langtry & Menter, AIAA J. 47(12) 2009."),
    T3AM= dict(name="ERCOFTAC T3A- flat plate (low-Tu bypass)",
               U=19.5, Tu_pct=0.874, nu=1.5e-5, L=1.7, dUe=0.0, L_turb=0.98e-3,
               source="Roach & Brierley (1990), ERCOFTAC T3A-; "
                      "ERCOFTAC Classic Collection Case 020."),
    # T3C4 is the only case in this set that separates: the adverse gradient
    # at the lowest tunnel speed drives the shape factor to 5.17 and the skin
    # friction to 1.8e-4 before the layer reattaches turbulent.  It is the
    # only measurement against which the separation-induced branch is tested.
    T3C4= dict(name="ERCOFTAC T3C4 flat plate (laminar separation bubble)",
               U=1.51, Tu_pct=2.11, nu=1.5e-5, L=1.5, dUe=0.0, L_turb=None,
               separation=True,
               source="Coupland J. (1990), ERCOFTAC T3C4; ERCOFTAC Classic "
                      "Collection Case 020, variable-pressure-gradient series."),
    SS  = dict(name="Schubauer & Skramstad flat plate (natural transition)",
               U=27.0, Tu_pct=0.03, nu=1.5e-5, L=4.0, dUe=0.0, L_turb=None,
               source="Schubauer & Skramstad (1948), NACA Report 909."),
)

# ---- Swept-wing cross-flow validation (Dagenhart & Saric) --------------
# 45 deg swept NLF(2)-0415, alpha = -4 deg, ASU Unsteady Wind Tunnel.
# Transition locations from naphthalene flow visualisation, Table 2 of
# NASA/TP-1999-209344.  This is the only case in which the cross-flow
# criterion is the selected one, and the only measurement against which
# it is calibrated.
SWEPT = dict(
    name      = "Dagenhart & Saric 45 deg swept NLF(2)-0415 (cross-flow)",
    sweep_deg = 45.0,
    alpha_deg = -4.0,
    chord_m   = 1.83,
    nu        = 1.5e-5,
    Tu_pct    = 0.02,
    Re_c      = [1.92e6, 2.19e6, 2.37e6, 2.73e6, 3.27e6, 3.73e6],
    x_tr_c    = [0.78,   0.73,   0.58,   0.45,   0.33,   0.30],
    source    = "Dagenhart J.R. & Saric W.S. (1999), 'Crossflow Stability and "
                "Transition Experiments in Swept-Wing Flow', NASA/TP-1999-209344, "
                "Table 2.",
)

# ---- Second, independent swept-wing dataset -------------------------------
# Boltz, Kenyon & Allen (NACA TN D-338, 1960), NACA 64(2)A015 untapered wing,
# Ames 12-Foot Low-Turbulence Pressure Tunnel, sweep 0-50 deg.  Different
# facility, section, era and measurement technique from the Dagenhart & Saric
# experiment on which the cross-flow coefficient was set, so this set is used
# as an independent check and NOTHING is calibrated on it.  The five fully
# specified points below are those quoted in the text of Beyak, Choudhari, Li
# & Shankara (AIAA 2024-xxxx / NASA NTRS 20230016659), which cites TN D-338;
# the original report presents the full database only in figures.
SWEPT2 = dict(
    name      = "Boltz, Kenyon & Allen NACA 64(2)A015 untapered wing",
    section   = "01_geometry/naca642a015.dat",
    chord_m   = 1.0,
    nu        = 1.5e-5,
    Tu_pct    = 0.05,
    sweep_deg = [0.0,    10.0,  30.0,  40.0,  50.0],
    alpha_deg = [4.0,     0.0,  -3.0,  -1.5,  -1.0],
    x_tr_c    = [0.21,   0.45,  0.21,  0.35,  0.24],
    Re_c      = [6.27e6, 15.0e6, 7.13e6, 6.30e6, 7.36e6],
    source    = "Boltz F.W., Kenyon G.C. & Allen C.Q. (1960), 'Effects of Sweep "
                "Angle on the Boundary-Layer Stability Characteristics of an "
                "Untapered Wing at Low Speeds', NACA TN D-338; values as quoted "
                "by Beyak, Choudhari, Li & Shankara (NASA NTRS 20230016659).",
)
