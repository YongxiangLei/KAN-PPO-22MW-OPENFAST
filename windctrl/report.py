"""Collect results, draw the paper figures, write the LaTeX tables and build the Overleaf zip.

    python -m windctrl.report

Reads whatever exists under outputs/ and writes

    results/                 copies of the numbers worth keeping in git (CSV, JSON, checkpoints)
    paper/figures/*.pdf      figures at IEEE column widths (3.5 in / 7.16 in), embedded fonts
    paper/tables/*.tex       tables and \\newcommand values that paper/main.tex \\input's
    paper/overleaf.zip       the paper/ directory, ready for "New Project > Upload Project"

It never invents a number: a table or figure is only written when its data exist, and
paper/main.tex prints a visible placeholder for anything missing. Safe to re-run at any time.
"""
import csv
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT, RES, PAPER = ROOT / "outputs", ROOT / "results", ROOT / "paper"
FIG, TAB = PAPER / "figures", PAPER / "tables"
KAN_RUN, MLP_RUN, EVAL, EVAL_LAW, SWEEP = "kan_v1", "mlp_v1", "eval_v1", "eval_law", "ipc_sweep"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in", "savefig.facecolor": "white",
})
COLORS = {"PI": "#2a9d6f", "ROSCO IPC": "#1f5fa8", "MLP-PPO": "#d1601a", "KAN-PPO": "#6a2fa8", "Four gains": "#444444"}
ORDER = ["PI", "ROSCO IPC", "MLP-PPO", "KAN-PPO", "Four gains"]
INDICATORS = [("blade_DEL", "Blade-root DEL", "MN$\\cdot$m"), ("tower_base_DEL", "Tower-base DEL", "MN$\\cdot$m"),
              ("tower_top_DEL", "Tower-top DEL", "MN$\\cdot$m"), ("power_std", "Power std", "MW"),
              ("speed_RMSE", "Speed RMSE", "rpm"), ("pitch_travel", "Pitch travel", "$^\\circ$/s"),
              ("max_pitch", "Max. pitch", "$^\\circ$")]


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def short_name(spec):
    kind = spec.split(":")[0]
    return {"pi": "PI", "rosco_ipc": "ROSCO IPC", "kan": "KAN-PPO", "mlp": "MLP-PPO", "law": "Four gains"}[kind]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v, digits=3):
    return "--" if not np.isfinite(v) else f"{v:.{digits}g}" if abs(v) < 100 else f"{v:.0f}"


def written(path, done):
    done.append(str(path.relative_to(ROOT)))


# ---- results/ -----------------------------------------------------------------------------------

def collect_results(done):
    RES.mkdir(exist_ok=True)
    wanted = [(ROOT / "config" / "norm.json", "norm.json")]
    for run in (KAN_RUN, MLP_RUN):
        wanted += [(OUT / run / f, f"{run}/{f}") for f in ("progress.csv", "args.json", "checkpoint.pt")]
    for name in (SWEEP, EVAL, EVAL_LAW):
        wanted += [(OUT / name / f, f"{name}/{f}") for f in ("summary.md", "results.csv", "args.json")]
    wanted += [(OUT / KAN_RUN / "interpret" / f, f"{KAN_RUN}/interpret/{f}") for f in ("report.md", "law.json", "edges.png")]
    for src, rel in wanted:
        if src.is_file():
            dst = RES / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            written(dst, done)


# ---- figures ------------------------------------------------------------------------------------

def fig_training(done):
    runs = {name: read_csv(OUT / run / "progress.csv") for name, run in (("KAN-PPO", KAN_RUN), ("MLP-PPO", MLP_RUN))
            if (OUT / run / "progress.csv").is_file()}
    runs = {k: v for k, v in runs.items() if len(v) >= 2}
    if not runs:
        return
    speeds = [c for c in next(iter(runs.values()))[0] if c.startswith("reward_ws")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for name, rows in runs.items():
        u = np.array([int(r["update"]) for r in rows])
        for col, ls in zip(speeds, ["-", "--", ":"]):
            y = np.array([num(r[col]) for r in rows])
            k = min(5, len(y))
            smooth = np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
            ax.plot(u, smooth, ls, color=COLORS[name], label=f"{name}, {col[9:]} m/s")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Mean reward per decision")
    ax.legend(frameon=False, ncol=len(runs), columnspacing=1.0, handlelength=1.8)
    ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "training.pdf")
    plt.close(fig)
    written(FIG / "training.pdf", done)


def evaluation_rows():
    """Per-run evaluation rows from the main evaluation plus the four-gain law run, keyed by short name."""
    rows = []
    for name in (EVAL, EVAL_LAW):
        path = OUT / name / "results.csv"
        if path.is_file():
            for r in read_csv(path):
                r["name"] = short_name(r["spec"])
                if name == EVAL_LAW and r["name"] != "Four gains":
                    continue                      # baselines are taken from the main evaluation
                rows.append(r)
    return rows


def fig_indicators(rows, done):
    names = [n for n in ORDER if any(r["name"] == n for r in rows)]
    if not names:
        return
    speeds = sorted({num(r["wind_speed"]) for r in rows})
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.5))
    width = 0.8 / len(names)
    for ax, (key, title, unit) in zip(axs.flat, INDICATORS[:6]):
        for i, n in enumerate(names):
            vals = [[num(r[key]) for r in rows if r["name"] == n and num(r["wind_speed"]) == ws] for ws in speeds]
            mean = [np.nanmean(v) if v else np.nan for v in vals]
            top = [np.nanmax(v) if v else np.nan for v in vals]
            x = np.arange(len(speeds)) + (i - (len(names) - 1) / 2) * width
            ax.bar(x, mean, width * 0.9, color=COLORS[n], label=n, yerr=[np.zeros(len(mean)), np.subtract(top, mean)],
                   error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_xticks(range(len(speeds)), [f"{ws:g}" for ws in speeds])
        ax.set_ylabel(f"{title} ({unit})")
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    for ax in axs[1]:
        ax.set_xlabel("Mean wind speed (m/s)")
    fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="upper center", ncol=len(names), frameon=False)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "indicators.pdf")
    plt.close(fig)
    written(FIG / "indicators.pdf", done)


def fig_edges(done):
    path = OUT / KAN_RUN / "interpret" / "edges.npz"
    if not path.is_file():
        return
    d = np.load(path, allow_pickle=False)
    labels = ["Load difference $\\Delta M_1$ (MN$\\cdot$m)", "Load difference $\\Delta M_2$ (MN$\\cdot$m)",
              "Load difference $\\Delta M_3$ (MN$\\cdot$m)", "Wind speed estimate (m/s)", "Generator speed (rpm)",
              "Collective pitch ($^\\circ$)"]
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.4), sharey=True)
    for j, ax in enumerate(axs.flat):
        lo, hi = d["range_phys"][j]
        ax.axvspan(lo, hi, color="0.9", lw=0)
        for i, ls in enumerate(["-", "--", ":"]):
            ax.plot(d["x_phys"][j], d["edges_deg"][:, i, j], ls, color=COLORS["KAN-PPO"], label=f"Blade {i + 1}")
        ax.axhline(0, color="0.5", lw=0.4)
        ax.set_xlabel(labels[j])
        ax.text(0.03, 0.93, f"({chr(97 + j)})", transform=ax.transAxes, fontweight="bold", va="top")
    for ax in axs[:, 0]:
        ax.set_ylabel("Edge value ($^\\circ$)")
    axs[0, 2].legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "edges.pdf")
    plt.close(fig)
    written(FIG / "edges.pdf", done)


# ---- tables -------------------------------------------------------------------------------------

def table(path, caption, label, header, body, done, wide=False, note=""):
    env = "table*" if wide else "table"
    cols = "l" + "c" * (len(header) - 1)
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\renewcommand{\\arraystretch}{1.15}", f"\\begin{{tabular}}{{{cols}}}", "\\hline\\hline",
             " & ".join(header) + " \\\\", "\\hline", *[" & ".join(r) + " \\\\" for r in body], "\\hline\\hline",
             "\\end{tabular}"]
    if note:
        lines.append(f"\\\\[2pt]\\footnotesize {note}")
    lines.append(f"\\end{{{env}}}")
    path.write_text("\n".join(lines) + "\n")
    written(path, done)


def tab_performance(rows, done):
    names = [n for n in ORDER if any(r["name"] == n for r in rows)]
    if not names:
        return
    mean = {n: {k: np.nanmean([num(r[k]) for r in rows if r["name"] == n]) for k, _, _ in INDICATORS} for n in names}
    failed = {n: sum(r["failed"] == "True" for r in rows if r["name"] == n) for n in names}
    ref = "PI" if "PI" in names else names[0]
    body = []
    for key, title, unit in INDICATORS:
        cells = [f"{title} ({unit})"]
        for n in names:
            v, r = mean[n][key], mean[ref][key]
            pct = f" ({(v / r - 1) * 100:+.0f}\\%)" if n != ref and np.isfinite(v) and r else ""
            cells.append(fmt(v) + pct)
        body.append(cells)
    n_runs = max(sum(r["name"] == n for r in rows) for n in names)
    note = f"Mean over {n_runs} runs; change from {ref} in parentheses."
    if any(failed.values()):
        note += " Aborted runs: " + ", ".join(f"{n} {c}" for n, c in failed.items() if c) + "."
    table(TAB / "performance.tex", "Closed-loop performance on the evaluation wind fields", "tab:performance",
          ["Indicator", *names], body, done, wide=True, note=note)

    speeds = sorted({num(r["wind_speed"]) for r in rows})
    body = []
    for key, title, unit in INDICATORS[:3]:
        for ws in speeds:
            body.append([f"{title}, {ws:g} m/s"] + [fmt(np.nanmean([num(r[key]) for r in rows if r["name"] == n
                                                                    and num(r["wind_speed"]) == ws] or [np.nan])) for n in names])
    table(TAB / "performance_by_wind.tex", "Damage-equivalent loads per mean wind speed (MN$\\cdot$m)",
          "tab:bywind", ["Indicator", *names], body, done, wide=True)


def tab_sweep(done):
    path = OUT / SWEEP / "results.csv"
    if not path.is_file():
        return None
    rows = read_csv(path)
    specs = list(dict.fromkeys(r["spec"] for r in rows))
    mean = {s: {k: np.nanmean([num(r[k]) for r in rows if r["spec"] == s]) for k, _, _ in INDICATORS} for s in specs}
    failed = {s: sum(r["failed"] == "True" for r in rows if r["spec"] == s) for s in specs}
    body = []
    for s in specs:
        gain = "PI (no IPC)" if s == "pi" else f"$K_I = {float(s.split(':')[1]) * 1e9:g}\\times10^{{-3}}$"
        body.append([gain, *[fmt(mean[s][k]) for k in ("blade_DEL", "tower_base_DEL", "tower_top_DEL", "power_std",
                                                        "speed_RMSE", "pitch_travel")], str(failed[s])])
    table(TAB / "sweep.tex", "ROSCO IPC integral gain on the training winds", "tab:sweep",
          ["Controller", "Blade", "Tower base", "Tower top", "Power std", "Speed RMSE", "Pitch travel", "Aborted"],
          body, done, wide=True,
          note="$K_I$ in rad/(MN$\\cdot$m$\\cdot$s). DELs in MN$\\cdot$m, power in MW, speed in rpm, pitch travel in $^\\circ$/s.")
    ok = [s for s in specs if s != "pi" and not failed[s] and np.isfinite(mean[s]["blade_DEL"])]
    return min(ok, key=lambda s: mean[s]["blade_DEL"]) if ok else None


def values(best_ipc, done):
    """\\newcommand definitions for numbers quoted in the text."""
    lines = ["% Generated by windctrl.report; do not edit by hand."]
    norm_path = ROOT / "config" / "norm.json"
    if norm_path.is_file():
        n = json.loads(norm_path.read_text())
        rpm = 60 / (2 * np.pi)
        lines += [f"\\newcommand{{\\NormMs}}{{{n['Ms'] / 1e3:.0f}}}", f"\\newcommand{{\\NormVbar}}{{{n['v_bar']:.2f}}}",
                  f"\\newcommand{{\\NormSigmaV}}{{{n['sigma_v']:.2f}}}", f"\\newcommand{{\\NormSigmaW}}{{{n['sigma_w'] * rpm:.3f}}}",
                  f"\\newcommand{{\\NormBetaBar}}{{{n['beta_bar']:.3f}}}", f"\\newcommand{{\\NormSigmaBeta}}{{{n['sigma_beta']:.3f}}}"]
    for name, run in (("Kan", KAN_RUN), ("Mlp", MLP_RUN)):
        p = OUT / run / "progress.csv"
        if p.is_file() and len(read_csv(p)):
            rows = read_csv(p)
            lines += [f"\\newcommand{{\\{name}Updates}}{{{rows[-1]['update']}}}",
                      f"\\newcommand{{\\{name}Failed}}{{{sum(int(r['failed']) for r in rows)}}}"]
    if best_ipc:
        lines.append(f"\\newcommand{{\\BestIpcGain}}{{{float(best_ipc.split(':')[1]) * 1e9:g}\\times10^{{-3}}}}")
    law_path = OUT / KAN_RUN / "interpret" / "law.json"
    if law_path.is_file():
        p = json.loads(law_path.read_text())["physical"]
        lines += [f"\\newcommand{{\\LawOwn}}{{{p['own_load_deg_per_MNm']:+.3f}}}",
                  f"\\newcommand{{\\LawNext}}{{{p['next_blade_deg_per_MNm']:+.3f}}}",
                  f"\\newcommand{{\\LawPrev}}{{{p['previous_blade_deg_per_MNm']:+.3f}}}",
                  f"\\newcommand{{\\LawSpeed}}{{{p['speed_deg_per_rpm']:+.2f}}}",
                  f"\\newcommand{{\\LawConst}}{{{p['constant_deg']:+.2f}}}",
                  f"\\newcommand{{\\GoneAbs}}{{{p['G1_abs_deg_per_MNm']:.3f}}}",
                  f"\\newcommand{{\\GonePhase}}{{{p['G1_phase_deg']:.1f}}}"]
    (TAB / "values.tex").write_text("\n".join(lines) + "\n")
    written(TAB / "values.tex", done)


def tab_law(done):
    path = OUT / KAN_RUN / "interpret" / "law.json"
    if not path.is_file():
        return
    law = json.loads(path.read_text())
    if "edge_size_deg" in law:
        heads = ["$\\Delta M_1$", "$\\Delta M_2$", "$\\Delta M_3$", "$\\hat v$", "$\\omega_g$", "$\\beta$"]
        body = [[f"Blade {i + 1}", *[f"{v:.2f}" for v in row]] for i, row in enumerate(law["edge_size_deg"])]
        table(TAB / "edge_sizes.tex", "Size of each KAN edge on the evaluation states ($^\\circ$)", "tab:edges",
              ["Output", *heads], body, done,
              note="Standard deviation of the edge contribution $\\theta_{\\max}\\varphi_{ij}$.")
    if "r2_linear" in law:
        body = [["Linear law", *[f"{v:.3f}" for v in law["r2_linear"]]],
                ["Four gains", *[f"{v:.3f}" for v in law["r2_four_gains"]]]]
        table(TAB / "law_fit.tex", "$R^2$ of the simplified laws against the KAN mean action", "tab:lawfit",
              ["Law", "Blade 1", "Blade 2", "Blade 3"], body, done)


# ---- zip ----------------------------------------------------------------------------------------

def build_zip(done):
    target = PAPER / "overleaf.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PAPER.rglob("*")):
            if path.is_file() and path != target and path.suffix not in (".aux", ".log", ".out", ".blg", ".bbl", ".gz"):
                if path.name == "main.pdf":
                    continue
                zf.write(path, path.relative_to(PAPER))
    written(target, done)


def main():
    done = []
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    collect_results(done)
    if (ROOT / "docs" / "framework.pdf").is_file():
        shutil.copy2(ROOT / "docs" / "framework.pdf", FIG / "framework.pdf")
        written(FIG / "framework.pdf", done)
    fig_training(done)
    rows = evaluation_rows()
    fig_indicators(rows, done)
    fig_edges(done)
    tab_performance(rows, done)
    best = tab_sweep(done)
    tab_law(done)
    values(best, done)
    build_zip(done)
    print("\n".join(done))
    if best:
        print(f"best ROSCO IPC gain on the sweep: {best}")


if __name__ == "__main__":
    main()
