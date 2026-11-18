"""Where in frequency the controllers act: spectra, band statistics and time histories of an evaluation.

    python -m windctrl.spectra --tag v2

Reads the raw OpenFAST outputs kept by windctrl.evaluate (runs/eval_<tag>_c<i> and, for the
four-gain law, runs/eval_law_<tag>_c0) and writes

    outputs/spectra_<tag>/summary.md    standard deviation of the blade-root and tower-top moments and
                                        of the individual pitch action per frequency band, share of
                                        time at the pitch-rate limit, tower-top DEL with and without
                                        the content above 3.2P; mean of all runs and per wind speed
    outputs/spectra_<tag>/bands.csv     the same quantities for every run
    outputs/spectra_<tag>/spectra.pdf   (a) PSD of the out-of-plane blade-root moment, (b) PSD of the
                 and .png               tower-top fore-aft moment, (c) pitch angle and (d) root moment
                                        of blade 1 over a few rotor revolutions of one run

Bands are multiples of the rotor frequency at rated speed (1P = 7.061 rpm): a band "nP" reaches from
(n - 0.2)P to (n + 0.2)P. The first 60 s of every run are left out, as in the evaluation.
"""
import argparse
import csv
import json

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from windctrl import report            # matplotlib style and colours of the other figures
from windctrl.env import ROOT
from windctrl.evaluate import del_

plt = report.plt
P1 = 7.061 / 60                         # rotor frequency at rated speed, Hz
SKIP = 60.0                             # s
SEGMENT = 160.0                         # s per Welch segment: resolution 1/160 Hz, about 19 revolutions
RATE_LIMIT = 2.0                        # deg/s
NAMES = {"pi": "PI", "rosco_ipc": "ROSCO IPC", "mlp": "MLP-PPO", "kan": "KAN-PPO", "law": "Four-gain law"}
ORDER = ["pi", "rosco_ipc", "mlp", "kan", "law"]
COLORS = {**report.COLORS, "Four-gain law": report.COLORS["Four gains"]}
STYLE = {"pi": "-", "rosco_ipc": "-", "mlp": "-", "kan": "-", "law": "--"}
BANDS = [("below 0.5P", 0.005, 0.5 * P1), ("1P", 0.8 * P1, 1.2 * P1), ("2P", 1.8 * P1, 2.2 * P1),
         ("3P", 2.8 * P1, 3.2 * P1), ("above 3.2P", 3.2 * P1, 5.0)]


def read(path):
    from openfast_io.FAST_output_reader import FASTOutputFile
    df = FASTOutputFile(str(path)).toDataFrame()
    return {c.split("_[")[0]: df[c].to_numpy() for c in df.columns}


def band_std(f, p, lo, hi):
    """Standard deviation of the part of a signal between two frequencies, from its PSD."""
    m = (f >= lo) & (f < hi)
    return float(np.sqrt(np.trapezoid(p[m], f[m])))


def analyse(tag, window_run="ws16_s201", window=(300.0, 340.0)):
    """Returns (kinds, psd, stats, traces) of the evaluation `eval_<tag>` and `eval_law_<tag>`.

    psd[kind] = (f, blade-root PSD, tower-top PSD), means over the runs; stats[kind] = one dict per
    run; traces[kind] = (t, pitch of blade 1, root moment of blade 1) inside `window` of `window_run`.
    """
    # Which case directory belongs to which controller, from the arguments of the two evaluations.
    cases, episodes = {}, None
    for name in (f"eval_{tag}", f"eval_law_{tag}"):
        path = ROOT / "outputs" / name / "args.json"
        if not path.is_file():
            continue
        a = json.loads(path.read_text())
        seeds = a["seeds"] or {"train": [101, 102], "eval": [201, 202]}[a["winds"]]
        eps = [f"ws{ws:g}_s{sd}" for ws in a["speeds"] for sd in seeds]
        assert episodes in (None, eps), "the two evaluations used different winds"
        episodes = eps
        for i, spec in enumerate(a["controllers"]):
            kind = spec.split(":")[0]
            if kind not in cases and not (name.startswith("eval_law") and kind != "law"):
                cases[kind] = ROOT / "runs" / f"{name}_c{i}"
    if not cases:
        raise SystemExit(f"no evaluation found for tag '{tag}' under outputs/")
    kinds = [k for k in ORDER if k in cases]

    psd, stats, traces = {}, {}, {}
    for k in kinds:
        pm, pt, rows = [], [], []
        for j, ep in enumerate(episodes):
            d = read(cases[k] / f"env{j:02d}" / ep / f"{ep}.outb")
            t = d["Time"]
            sel, fs = t >= SKIP, 1.0 / (t[1] - t[0])
            moment = np.stack([d[f"RootMyc{b}"][sel] for b in (1, 2, 3)]) / 1e3          # MN m
            pitch = np.stack([d[f"BldPitch{b}"][sel] for b in (1, 2, 3)])                # deg
            top = d["YawBrMyp"][sel] / 1e3
            nper = int(SEGMENT * fs)
            f, p_m = welch(moment, fs, nperseg=nper, axis=1)
            _, p_t = welch(top, fs, nperseg=nper)
            _, p_d = welch(pitch - pitch.mean(0), fs, nperseg=nper, axis=1)             # individual part
            _, p_c = welch(moment.mean(0), fs, nperseg=nper)                            # part common to the blades
            rate = np.abs(np.diff(pitch, axis=1)) * fs
            # Tower-top DEL as in the evaluation, and again with the content above 3.2P filtered out.
            dur = float(t[sel][-1] - t[sel][0])
            slow = sosfiltfilt(butter(4, 3.2 * P1, fs=fs, output="sos"), top)
            pm.append(p_m.mean(0))
            pt.append(p_t)
            rows.append({**{f"blade {b}": band_std(f, p_m.mean(0), lo, hi) for b, lo, hi in BANDS},
                         "blade below 0.5P, common": band_std(f, p_c, *BANDS[0][1:]),
                         **{f"top {b}": band_std(f, p_t, lo, hi) for b, lo, hi in BANDS},
                         **{f"ind. pitch {b}": band_std(f, p_d.mean(0), lo, hi) for b, lo, hi in BANDS[1:]},
                         "top std": float(top.std()), "ind. pitch std": float((pitch - pitch.mean(0)).std()),
                         "top DEL": del_(top, 4, dur), "top DEL, low-pass 3.2P": del_(slow, 4, dur),
                         "at limit %": float((rate > 0.99 * RATE_LIMIT).mean() * 100), "max rate": float(rate.max()),
                         "wind": float(ep[2:].split("_")[0])})
            if ep == window_run:
                w = (t >= window[0]) & (t <= window[1])
                traces[k] = (t[w], d["BldPitch1"][w], d["RootMyc1"][w] / 1e3)
        psd[k], stats[k] = (f, np.mean(pm, 0), np.mean(pt, 0)), rows
    return kinds, psd, stats, traces


def figure(kinds, psd, traces, window, paths):
    """Draws the four-panel figure at IEEE double-column width and saves it to every path in `paths`."""
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.0))
    for k in kinds:
        f, p_m, p_t = psd[k]
        kw = {"color": COLORS[NAMES[k]], "ls": STYLE[k], "lw": 0.9, "label": NAMES[k]}
        axs[0, 0].loglog(f[1:], p_m[1:], **kw)
        axs[0, 1].loglog(f[1:], p_t[1:], **kw)
        if k in traces:
            t, pitch, moment = traces[k]
            axs[1, 0].plot(t, pitch, **kw)
            axs[1, 1].plot(t, moment, **kw)
    for ax, label in ((axs[0, 0], "Blade-root moment PSD\n((MN$\\cdot$m)$^2$/Hz)"),
                      (axs[0, 1], "Tower-top moment PSD\n((MN$\\cdot$m)$^2$/Hz)")):
        ax.set_xlim(0.02, 2.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(label)
        for n in (1, 2, 3):
            ax.axvline(n * P1, color="0.6", lw=0.5, ls=":", zorder=0)
            ax.text(n * P1, 0.97, f"{n}P", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6,
                    color="0.3", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        ax.grid(alpha=0.3, lw=0.4)
    for col, ax in ((1, axs[0, 0]), (2, axs[0, 1])):
        inside = [psd[k][col][(psd[k][0] > 0.02) & (psd[k][0] < 2)] for k in kinds]
        ax.set_ylim(min(v.min() for v in inside) / 1.5, max(v.max() for v in inside) * 4)
    axs[1, 0].set_ylabel("Pitch angle of blade 1 ($^\\circ$)")
    axs[1, 1].set_ylabel("Root moment of blade 1 (MN$\\cdot$m)")
    for ax in axs[1]:
        ax.set_xlabel("Time (s)")
        ax.set_xlim(*window)
        ax.grid(alpha=0.3, lw=0.4)
    for ax, letter in zip(axs.flat, "abcd"):
        ax.text(0.02, 0.05, f"({letter})", transform=ax.transAxes, fontweight="bold", va="bottom")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(kinds), frameon=False)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.95))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **({"dpi": 200} if path.suffix == ".png" else {}))
    plt.close(fig)


def mean_of(stats, kind, key, wind=None):
    return float(np.mean([r[key] for r in stats[kind] if wind is None or r["wind"] == wind]))


def keys_of(stats):
    return [k for k in next(iter(stats.values()))[0] if k != "wind"]


def text_lines(kinds, stats, window_run, window):
    """The band statistics as plain text, one block for all runs and one per wind speed."""
    winds = sorted({r["wind"] for r in stats[kinds[0]]})
    lines = [f"rotor frequency 1P = {P1:.4f} Hz; bands: " + ", ".join(f"{b} {lo:.3f}-{hi:.3f} Hz" for b, lo, hi in BANDS),
             f"window of panels (c), (d): {window_run}, {window[0]:g}-{window[1]:g} s"]
    for ws in (None, *winds):
        lines += ["", f"=== {'mean of all runs' if ws is None else f'{ws:g} m/s'}: standard deviation per band "
                      "(blade and tower-top moments in MN m, individual pitch in deg), change from PI in brackets ==="]
        for key in keys_of(stats):
            cells = []
            for k in kinds:
                v, ref = mean_of(stats, k, key, ws), mean_of(stats, kinds[0], key, ws)
                cells.append(f"{NAMES[k]} {v:.3g}" + (f" ({(v / ref - 1) * 100:+.0f}%)" if k != kinds[0] and ref > 1e-6 else ""))
            lines.append(f"{key:24s} " + "  ".join(cells))
    return lines


def markdown(tag, kinds, stats):
    """summary.md: one table for all runs and one per wind speed."""
    units = {"blade": "MN m", "top": "MN m", "ind.": "deg", "at": "%", "max": "deg/s"}
    winds = sorted({r["wind"] for r in stats[kinds[0]]})
    n_runs = len(stats[kinds[0]])
    lines = [f"# Spectra of the evaluation `eval_{tag}`", "",
             f"Generated by `python -m windctrl.spectra --tag {tag}` from the raw OpenFAST outputs of {n_runs} runs per "
             "controller; first 60 s left out. `blade`: out-of-plane blade-root moment, mean of the three blades; "
             "`top`: tower-top fore-aft moment; `ind. pitch`: pitch angle minus the mean of the three blades. "
             "Entries are standard deviations inside a frequency band (nP = n times the rotor frequency, 0.118 Hz), "
             "except the DELs (Woehler exponent 4), the share of time with the pitch rate at its limit of 2 deg/s, and "
             "the largest pitch rate. Brackets: change from PI.", ""]
    for ws in (None, *winds):
        lines += [f"## {'Mean of all runs' if ws is None else f'{ws:g} m/s'}", "",
                  "| Quantity | " + " | ".join(NAMES[k] for k in kinds) + " |", "|---|" + "---|" * len(kinds)]
        for key in keys_of(stats):
            unit = "MN m" if "DEL" in key else units[key.split()[0]]
            cells = []
            for k in kinds:
                v, ref = mean_of(stats, k, key, ws), mean_of(stats, kinds[0], key, ws)
                cells.append((f"{v:.3g}" if abs(v) > 5e-4 else "0") +
                             (f" ({(v / ref - 1) * 100:+.0f}%)" if k != kinds[0] and ref > 1e-6 else ""))
            lines.append(f"| {key} ({unit}) | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v2", help="evaluations eval_<tag> and eval_law_<tag>")
    ap.add_argument("--window", type=float, nargs=2, default=[300.0, 340.0], help="time window of panels (c), (d)")
    ap.add_argument("--window-run", default="ws16_s201", help="run shown in panels (c), (d)")
    args = ap.parse_args()

    kinds, psd, stats, traces = analyse(args.tag, args.window_run, args.window)
    out = ROOT / "outputs" / f"spectra_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    figure(kinds, psd, traces, args.window, [out / "spectra.pdf", out / "spectra.png"])
    (out / "summary.md").write_text("\n".join(markdown(args.tag, kinds, stats)) + "\n")
    with open(out / "bands.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["controller", "run", *keys_of(stats)])
        for k in kinds:
            for j, r in enumerate(stats[k]):
                w.writerow([NAMES[k], j, *[f"{r[key]:.6g}" for key in keys_of(stats)]])
    print("\n".join(text_lines(kinds, stats, args.window_run, args.window)))
    print(f"\nwritten: {out}/summary.md, bands.csv, spectra.pdf, spectra.png")


if __name__ == "__main__":
    main()
