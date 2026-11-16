#!/usr/bin/env python
"""Draw the framework diagram of the KAN-PPO individual pitch controller.

    python scripts/make_framework_figure.py        # writes docs/framework.{svg,pdf,png}

Three lanes: (A) the closed loop on OpenFAST, (B) PPO training, (C) evaluation and interpretation.
Edit the text in this file and re-run to change the figure; the SVG can also be edited by hand.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs"

INK, MUTED = "#1f2933", "#52606d"
BLUE, ORANGE, GREEN, PURPLE = "#2b6cb0", "#c05621", "#2f855a", "#6b46c1"
FILL = {BLUE: "#eaf2fb", ORANGE: "#fdf1e3", GREEN: "#e9f6ee", PURPLE: "#f3effb"}
W, H = 15.6, 10.45

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def lane(color, y0, y1, title):
    ax.add_patch(FancyBboxPatch((0.15, y0), W - 0.3, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=FILL[color], ec=color, lw=1.2, zorder=0))
    ax.text(0.35, y1 - 0.13, title, fontsize=12.5, fontweight="bold", color=color, va="top", ha="left")


def box(x, y, w, h, title, body, color):
    """Box centred at (x, y); KAN parts (purple) get a tinted fill."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=FILL[PURPLE] if color == PURPLE else "white", ec=color, lw=1.5, zorder=2))
    ax.text(x, y + h / 2 - 0.12, title, fontsize=10.3, fontweight="bold", color=color, ha="center", va="top", zorder=3)
    ax.text(x, y - 0.17, body, fontsize=8.8, color=INK, ha="center", va="center", linespacing=1.4, zorder=3)


def label(x, y, text, color=MUTED, ha="center", va="bottom", italic=False):
    ax.text(x, y, text, fontsize=8.6, color=color, ha=ha, va=va, zorder=5, style="italic" if italic else "normal")


def arrow(p, q, color=INK, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0))


def path(points, color=INK, ls="-"):
    """Polyline with an arrow head on its last segment."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=1.4, ls=ls, zorder=4, solid_capstyle="round")
    arrow(points[-2], points[-1], color=color, ls=ls)


# ---------------------------------------------------------------------------------------------
# A: closed loop
# ---------------------------------------------------------------------------------------------
lane(BLUE, 6.3, 10.3, "A   Closed loop on OpenFAST   (policy decision every 0.1 s)")
TOP, BOT = 8.9, 7.4

box(1.4, TOP, 2.2, 1.25, "Wind and waves", "TurbSim, IEC NTM class B\n13 / 16 / 19 m/s\nirregular waves", BLUE)
box(4.6, TOP, 3.4, 1.25, "IEA 22 MW monopile turbine", "OpenFAST\naero-hydro-servo-elastic simulation", BLUE)
box(9.55, TOP, 3.9, 1.25, "State  s  (6 inputs, normalised)",
    "$\\Delta M_1,\\ \\Delta M_2,\\ \\Delta M_3$   blade-root load differences\n"
    "$\\hat v$  wind estimate  ·  $\\omega_g$  speed  ·  $\\beta$  pitch", BLUE)
box(13.6, TOP, 3.0, 1.25, "KAN actor  [6 → 3]",
    "$z_i = b_i + \\Sigma_j\\, \\varphi_{ij}(s_j)$,    $a_i = \\tanh z_i$\none layer, 330 parameters", PURPLE)
box(4.6, BOT, 3.4, 0.9, "ROSCO baseline controller", "collective pitch PI + torque", BLUE)
box(8.65, BOT, 2.3, 0.9, "Add and limit", "$\\beta_i = \\beta_c + \\Delta\\beta_i$,  rate ≤ 2°/s", BLUE)
box(12.55, BOT, 3.0, 0.9, "Scale and smooth", "$\\Delta\\beta_i = 4°\\cdot a_i$,  low-pass 0.1 s", BLUE)

arrow((2.5, 9.05), (2.9, 9.05))
arrow((6.3, TOP), (7.6, TOP))
label(6.95, TOP + 0.06, "measurements")
label(6.95, TOP - 0.08, "ZeroMQ", va="top", italic=True)
arrow((11.5, TOP), (12.1, TOP))
arrow((13.2, 8.275), (13.2, 7.85))
label(13.1, 8.06, "action  $a \\in [-1,1]^3$", ha="right", va="center")
arrow((11.05, BOT), (9.8, BOT))
label(10.42, BOT + 0.06, "$\\Delta\\beta_{1..3}$")
label(10.42, BOT - 0.08, "ZeroMQ", va="top", italic=True)
arrow((6.3, BOT), (7.5, BOT))
label(6.9, BOT + 0.06, "$\\beta_c$")
arrow((4.0, 8.275), (4.0, 7.85))
label(4.1, 8.06, "$\\omega_g$, loads", ha="left", va="center")
path([(8.65, 6.95), (8.65, 6.55), (2.68, 6.55), (2.68, 8.7), (2.9, 8.7)])
label(5.7, 6.59, "blade pitch commands  $\\beta_{1..3}$  and generator torque")

# ---------------------------------------------------------------------------------------------
# B: training
# ---------------------------------------------------------------------------------------------
lane(ORANGE, 3.6, 6.15, "B   PPO training")
MID = 4.7

box(1.75, MID, 2.7, 1.4, "Calibration", "ROSCO-only runs on the\ntraining winds give the\nstate normalisation", ORANGE)
box(5.0, MID, 3.1, 1.4, "Parallel rollouts", "3 episodes × 300 s per update\n13, 16, 19 m/s;  seeds 101 / 102\none OpenFAST process each", ORANGE)
box(8.55, MID, 3.4, 1.4, "Reward  r",
    "$-\\ \\overline{|\\Delta M_i|}\\,/\\,M_s$     load balance\n"
    "$-\\ 5\\,|P-P_{rated}|\\,/\\,P_{rated}$     power\n"
    "$-\\ 0.1\\ \\overline{|a_{i,k}-a_{i,k-1}|}$     smoothness", ORANGE)
box(11.9, MID, 2.7, 1.4, "KAN critic  [6 → 6 → 1]", "value estimate, 385 parameters\nadvantages by GAE\n$\\gamma = 0.99$,  $\\lambda = 0.95$", PURPLE)
box(14.45, MID, 1.9, 1.4, "PPO update", "clip 0.2\n10 epochs\n100 updates", ORANGE)

arrow((3.1, MID), (3.45, MID))
arrow((6.55, MID), (6.85, MID))
arrow((10.25, MID), (10.55, MID))
arrow((13.25, MID), (13.5, MID))
arrow((5.0, 6.3), (5.0, 5.4), color=BLUE)
label(5.12, 5.78, "trajectories  $(s, a, r)$  from the closed loop", color=BLUE, ha="left", va="center")
arrow((14.45, 5.4), (14.45, 8.275), color=ORANGE)
label(14.35, 5.78, "updates  $\\varphi_{ij}$, $b_i$", color=ORANGE, ha="right", va="center")

# ---------------------------------------------------------------------------------------------
# C: evaluation and interpretation
# ---------------------------------------------------------------------------------------------
lane(GREEN, 0.15, 3.1, "C   Evaluation and interpretation")
EV = 1.55

box(1.6, EV, 2.4, 1.5, "Unseen winds", "seeds 201 / 202\n13, 16, 19 m/s\n600-s runs", GREEN)
box(4.7, EV, 2.9, 1.5, "Controllers compared", "ROSCO PI  ·  ROSCO IPC\nMLP-PPO  ·  KAN-PPO\nexplicit four-gain law", GREEN)
box(8.05, EV, 3.0, 1.5, "Indicators", "blade and tower DEL (rainflow)\npower std,  speed RMSE\npitch travel", GREEN)
box(11.35, 2.17, 2.9, 0.85, "Read the KAN edges", "18 curves $\\varphi_{ij}$ in canonical form", PURPLE)
box(11.35, 0.93, 2.9, 0.95, "Explicit control law", "gains $g_s$, $g_n$, $g_p$, $k$ and a constant", PURPLE)
box(14.2, EV, 2.3, 1.5, "Physical reading", "edge sizes, attribution\n1P complex gain $G_1$\nload vs speed channel", PURPLE)

arrow((2.8, EV), (3.25, EV))
arrow((6.15, EV), (6.55, EV))
arrow((11.35, 1.745), (11.35, 1.405))
arrow((12.8, 2.1), (13.05, 1.9), color=PURPLE)
arrow((12.8, 1.0), (13.05, 1.2), color=PURPLE)
path([(9.9, 0.93), (9.72, 0.93), (9.72, 0.42), (4.7, 0.42), (4.7, 0.8)], color=PURPLE, ls="--")
label(7.2, 0.45, "explicit law run in closed loop, no retraining", color=PURPLE)

# Trained policy: from the PPO update down to both the evaluation and the interpretation.
ax.plot([14.45, 14.45, 4.7], [4.0, 3.35, 3.35], color=PURPLE, lw=1.4, zorder=4, solid_capstyle="round")
arrow((4.7, 3.35), (4.7, 2.3), color=PURPLE)
arrow((11.35, 3.35), (11.35, 2.595), color=PURPLE)
label(8.0, 3.38, "trained policy", color=PURPLE)

OUT.mkdir(exist_ok=True)
for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"framework.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
print("written:", ", ".join(str(OUT / f"framework.{e}") for e in ("svg", "pdf", "png")))
