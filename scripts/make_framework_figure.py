#!/usr/bin/env python
"""Draw the framework diagram of the KAN-PPO individual pitch controller, sized for an IEEE journal.

    python scripts/make_framework_figure.py        # writes docs/framework.{pdf,eps,svg,png}

The figure is drawn at its final size: 7.16 in wide (IEEE double-column width), text 7-8 pt in a
Times-compatible face, lines 0.8 pt. Insert it at 100 % scale (\\includegraphics[width=\\textwidth])
so the type size in print is what is set here. Fonts are embedded as TrueType in the PDF.

Three panels: (a) the closed loop on OpenFAST, with a drawing of the turbine to scale, (b) PPO
training, (c) evaluation and interpretation.
All coordinates below are in inches.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parents[1] / "docs"
GROW = 0.74                      # extra height of the turbine box, which holds a drawing of the turbine
W, H = 7.16, 4.66 + GROW

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],   # Liberation Serif = Times metrics
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,       # embed TrueType, no Type 3 fonts
    "svg.fonttype": "none",                      # keep text editable in the SVG
})

BLACK, GRAY = "#000000", "#3c3c3c"
BLUE, ORANGE, GREEN, PURPLE = "#1f5fa8", "#b4560f", "#2a7a4b", "#5b3fa8"
PANEL = {BLUE: "#eef4fb", ORANGE: "#fcf3e8", GREEN: "#edf6f0"}
KAN_FILL = "#ece7f7"
WATER, STEEL, PAINT = "#cfe2f3", "#f3f3f3", "#e3b21a"
TITLE, BODY, NOTE = 8.0, 7.5, 7.0
LW = 0.8

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def panel(color, y0, y1, title):
    ax.add_patch(FancyBboxPatch((0.03, y0), W - 0.06, y1 - y0, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=PANEL[color], ec=color, lw=0.6, zorder=0))
    ax.text(0.1, y1 - 0.045, title, fontsize=TITLE, fontweight="bold", color=BLACK, va="top", ha="left")


def box(xc, yc, w, h, title, body, color):
    """Box centred at (xc, yc). KAN parts (purple) are tinted so they also stand out in grayscale."""
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=KAN_FILL if color == PURPLE else "white", ec=color, lw=LW, zorder=2))
    ax.text(xc, yc + h / 2 - 0.04, title, fontsize=TITLE, fontweight="bold", color=BLACK, ha="center", va="top", zorder=3)
    ax.text(xc, yc - 0.075, body, fontsize=BODY, color=BLACK, ha="center", va="center", linespacing=1.22, zorder=3)


def note(x, y, text, ha="center", va="bottom", italic=False):
    ax.text(x, y, text, fontsize=NOTE, color=GRAY, ha=ha, va=va, zorder=5, style="italic" if italic else "normal")


def arrow(p, q, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=LW, color=BLACK, linestyle=ls,
                                 zorder=4, shrinkA=0, shrinkB=0))


def path(points, ls="-"):
    """Polyline with an arrow head on its last segment."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=BLACK, lw=LW, ls=ls, zorder=4, solid_capstyle="butt", solid_joinstyle="miter")
    arrow(points[-2], points[-1], ls=ls)


def turbine(xc, y_bed, height, width):
    """Front view of the turbine on its monopile, to scale: 34 m of water, hub 170 m above the sea,
    blades of 142 m. `y_bed` is the sea bed, `height` reaches from there to the tip of the upper blade,
    `width` is the width of the strip of sea. Tower and blades are drawn about three times as thick as
    to scale, so that they stay visible at this size."""
    s = height / (34.0 + 170.0 + 142.0)                       # inches per metre
    sea, hub, length = y_bed + 34.0 * s, y_bed + 204.0 * s, 142.0 * s
    base = sea + 15.0 * s                                     # tower base, on the transition piece
    edge = {"ec": "#222222", "lw": 0.6, "joinstyle": "round"}
    # sea with a wavy surface, and the sea bed
    xs = np.linspace(xc - width / 2, xc + width / 2, 121)
    surface = sea + 0.007 * np.sin(2 * np.pi * 6 * (xs - xs[0]) / width)
    ax.fill_between(xs, y_bed, surface, color=WATER, lw=0, zorder=2.5)
    ax.plot(xs, surface, color=BLUE, lw=0.6, zorder=3)
    ax.plot(xs[[0, -1]], [y_bed, y_bed], color=GRAY, lw=0.7, zorder=3)
    # monopile, transition piece, tower
    for half, y0, y1, half_top, colour in ((0.022, y_bed, sea + 4 * s, 0.022, "#9a9a9a"), (0.027, sea + 4 * s, base, 0.027, PAINT),
                                           (0.021, base, hub, 0.013, STEEL)):
        ax.add_patch(Polygon([(xc - half, y0), (xc + half, y0), (xc + half_top, y1), (xc - half_top, y1)], fc=colour,
                             zorder=3.2, **edge))
    # blades, one pointing up; the chord is largest at a fifth of the span
    span = np.array([0.0, 0.04, 0.2, 0.5, 0.8, 1.0]) * length
    chord = np.array([0.026, 0.026, 0.046, 0.030, 0.016, 0.005])
    centre = np.array([xc, hub])
    for angle in np.radians([90.0, 210.0, 330.0]):
        along, across = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
        lead = [centre + r * along + 0.35 * c * across for r, c in zip(span, chord)]
        trail = [centre + r * along - 0.65 * c * across for r, c in zip(span, chord)]
        ax.add_patch(Polygon(lead + trail[::-1], fc=STEEL, zorder=3.4, **edge))
    ax.add_patch(Circle((xc, hub), 0.022, fc="#d0d0d0", ec="#222222", lw=0.6, zorder=3.6))
    return hub


# ---------------------------------------------------------------------------------------------
# (a) closed loop
# ---------------------------------------------------------------------------------------------
panel(BLUE, 2.73, 4.63 + GROW, "(a)  Closed-loop control on OpenFAST")
ROW, BOT = 3.70, 3.24            # lower edge of the top row; centre of the bottom row (boxes 0.50 high)
TOP = ROW + (0.66 + GROW) / 2    # centre of the top row: the turbine box is 0.66 + GROW high, the others 0.66

box(0.665, TOP, 1.07, 0.66, "Wind and waves", "TurbSim, NTM class B\n13, 16, 19 m/s\nirregular waves", BLUE)
# The turbine box: title, a drawing of the turbine, then the text.
box(2.145, TOP, 1.45, 0.66 + GROW, "IEA 22-MW turbine", "", BLUE)
ax.text(2.145, ROW + 0.045, "monopile foundation\nOpenFAST aero-hydro-\nservo-elastic model", fontsize=BODY, color=BLACK,
        ha="center", va="bottom", linespacing=1.22, zorder=3)
HUB = turbine(2.145, ROW + 0.47, GROW, 1.2)
box(4.5, TOP, 2.0, 0.66, "State  $\\mathbf{s}$  (6 inputs, normalized)",
    "$\\Delta M_{1,2,3}$   blade-root load differences\n"
    "$\\hat{v}$   estimated wind speed\n$\\omega_g$   generator speed,     $\\beta$   collective pitch", BLUE)
box(6.345, TOP, 1.37, 0.66, "KAN actor  [6, 3]",
    "$z_i = b_i + \\Sigma_j\\, \\varphi_{ij}(s_j)$\n$a_i = \\tanh z_i$\n330 parameters", PURPLE)
box(2.145, BOT, 1.45, 0.50, "ROSCO baseline", "collective pitch PI\nand torque control", BLUE)
box(4.0, BOT, 1.25, 0.50, "Add and limit", "$\\beta_i = \\beta_c + \\Delta\\beta_i$\nrate limit 2°/s", BLUE)
box(6.0, BOT, 1.45, 0.50, "Scale and filter", "$\\Delta\\beta_i = \\theta_{\\max} a_i$,   $\\theta_{\\max} = 4°$\nfirst-order filter, 0.1 s", BLUE)

arrow((1.2, HUB), (1.42, HUB))                   # the wind arrives at hub height
arrow((2.87, TOP), (3.5, TOP))
note(3.185, TOP + 0.03, "measurements")
note(3.185, TOP - 0.035, "ZeroMQ", va="top", italic=True)
arrow((5.5, TOP), (5.66, TOP))
arrow((6.1, TOP - 0.33), (6.1, 3.49))
note(6.05, (TOP - 0.33 + 3.49) / 2, "action  $\\mathbf{a} \\in [-1, 1]^3$", ha="right", va="center")
arrow((5.275, BOT), (4.625, BOT))
note(4.95, BOT + 0.03, "$\\Delta\\beta_{1,2,3}$")
note(4.95, BOT - 0.035, "ZeroMQ", va="top", italic=True)
arrow((2.87, BOT), (3.375, BOT))
note(3.12, BOT + 0.03, "$\\beta_c$")
arrow((1.9, 3.70), (1.9, 3.49))
note(1.95, 3.595, "$\\omega_g$, loads", ha="left", va="center")
path([(4.0, 2.99), (4.0, 2.84), (1.31, 2.84), (1.31, 3.92), (1.42, 3.92)])
note(2.65, 2.855, "blade pitch commands  $\\beta_{1,2,3}$  and generator torque")

# ---------------------------------------------------------------------------------------------
# (b) training
# ---------------------------------------------------------------------------------------------
panel(ORANGE, 1.53, 2.65, "(b)  PPO training")
MID = 1.94                       # boxes 0.70 high: 1.59 .. 2.29

box(0.69, MID, 1.12, 0.70, "Calibration", "ROSCO-only runs set\nthe normalization\nconstants", ORANGE)
box(2.1, MID, 1.38, 0.70, "Parallel rollouts", "3 episodes per update\n300 s each, in parallel\n13, 16, 19 m/s", ORANGE)
box(3.76, MID, 1.62, 0.70, "Reward  $r$", "", ORANGE)
for row, (term, meaning) in enumerate([("$-\\,\\langle |\\Delta M_i| \\rangle\\,/\\,M_s$", "load balance"),
                                       ("$-\\,5\\,|P - P_r|\\,/\\,P_r$", "power tracking"),
                                       ("$-\\,0.1\\,\\langle |a_{i,k} - a_{i,k-1}| \\rangle$", "smoothness")]):
    y = MID + 0.06 - 0.137 * row
    ax.text(3.03, y, term, fontsize=BODY, color=BLACK, ha="left", va="center", zorder=3)
    ax.text(4.5, y, meaning, fontsize=BODY, color=BLACK, ha="right", va="center", zorder=3)
box(5.37, MID, 1.28, 0.70, "KAN critic  [6, 6, 1]", "value estimate\nGAE,  $\\gamma = 0.99$,  $\\lambda = 0.95$\n385 parameters", PURPLE)
box(6.6, MID, 0.86, 0.70, "PPO update", "clip 0.2\n10 epochs\n100 updates", ORANGE)

arrow((1.25, MID), (1.41, MID))
arrow((2.79, MID), (2.95, MID))
arrow((4.57, MID), (4.73, MID))
arrow((6.01, MID), (6.17, MID))
arrow((2.1, 2.73), (2.1, 2.29))
note(2.16, 2.46, "trajectories  $(\\mathbf{s}, \\mathbf{a}, r)$", ha="left", va="center")
arrow((6.88, 2.29), (6.88, TOP - 0.33))
note(6.83, 2.46, "updated  $\\varphi_{ij}$, $b_i$", ha="right", va="center")

# ---------------------------------------------------------------------------------------------
# (c) evaluation and interpretation
# ---------------------------------------------------------------------------------------------
panel(GREEN, 0.03, 1.275, "(c)  Evaluation and interpretation")
EV = 0.62                        # boxes 0.74 high: 0.25 .. 0.99

box(0.63, EV, 1.0, 0.74, "Unseen winds", "seeds 201, 202\n13, 16, 19 m/s\n600-s runs", GREEN)
box(1.94, EV, 1.3, 0.74, "Controllers", "ROSCO PI,  ROSCO IPC\nMLP-PPO,  KAN-PPO\nexplicit four-gain law", GREEN)
box(3.44, EV, 1.38, 0.74, "Indicators", "blade and tower DELs\npower and speed variation\npitch travel", GREEN)
box(5.04, 0.83, 1.5, 0.34, "KAN edge functions", "18 curves  $\\varphi_{ij}$,  canonical form", PURPLE)
box(5.04, 0.38, 1.5, 0.34, "Explicit control law", "four gains and a constant", PURPLE)
box(6.49, EV, 1.08, 0.74, "Physical reading", "edge sizes\n1P complex gain  $G_1$\nload / speed channels", PURPLE)

arrow((1.13, EV), (1.29, EV))
arrow((2.59, EV), (2.75, EV))
arrow((5.04, 0.66), (5.04, 0.55))
arrow((5.79, 0.83), (5.95, 0.74))
arrow((5.79, 0.38), (5.95, 0.50))
path([(4.29, 0.38), (4.21, 0.38), (4.21, 0.12), (1.94, 0.12), (1.94, 0.25)], ls=(0, (4, 2)))
note(3.08, 0.135, "explicit law run in closed loop, no retraining")

# Trained policy: from the PPO update to both the evaluation and the interpretation.
ax.plot([6.6, 6.6, 1.94], [1.59, 1.37, 1.37], color=BLACK, lw=LW, zorder=4, solid_joinstyle="miter")
arrow((1.94, 1.37), (1.94, 0.99))
arrow((5.04, 1.37), (5.04, 1.0))
note(3.5, 1.385, "trained policy")

OUT.mkdir(exist_ok=True)
for ext, kw in (("pdf", {}), ("eps", {}), ("svg", {}), ("png", {"dpi": 600})):
    fig.savefig(OUT / f"framework.{ext}", facecolor="white", **kw)
print(f"written docs/framework.pdf, .eps, .svg, .png  ({W} x {H} in)")
