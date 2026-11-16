"""Read a trained KAN policy: plot its edge functions and reduce it to an explicit control law.

The mean action of the single-layer KAN is, before the tanh, an exact sum

    z_i = b_i + sum_j phi_ij(s_j),      d_beta_i = theta_max * tanh(z_i),

so every input's contribution to every blade is known without a surrogate model. This script

  1. puts the edges in canonical form. Because dM1 + dM2 + dM3 = 0, adding the same slope to the
     three load edges of an output changes nothing, so the edges are not unique; the mean load
     slope of each output is removed, and every edge is centred on the states (constants -> b_i);
  2. plots the 18 edges in physical units, with the 1-99 % range of the states shaded;
  3. reports the size of each edge (std of its contribution over the states, in degrees);
  4. fits the best linear law, and the blade-symmetric "four gains + constant" law
         d_beta_i = theta_max tanh(g_s dM_i + g_n dM_i+1 + g_p dM_i-1 + k (w_g - w_ref) + b),
     with R^2 against the KAN on the states;
  5. reports the 1P complex gain G1 = g_s + g_n e^{+j120} + g_p e^{-j120}: the pitch offset of a
     blade equals its own once-per-revolution load scaled by |G1| and shifted by arg(G1).

    python -m windctrl.interpret --run kan_v1 --states outputs/eval_v1/states_c2.npz

`--states` is an .npz with an `obs` array written by windctrl.evaluate; without it the states are
assumed standard normal, which is only a rough stand-in. The four-gain law is saved to
outputs/<run>/interpret/law.json and can be run in closed loop with `--controllers law:<run>`.
"""
import argparse
import json

import numpy as np
import torch

from windctrl.env import ROOT
from windctrl.train import OBS_NAMES, KANPolicy

RAD2RPM = 60 / (2 * np.pi)


def physical_axes(norm):
    """Per input: (label, physical value at normalised x, input units per normalised unit)."""
    return [
        *[(f"dM{k} (MN m)", lambda x, n=norm: x * n["Ms"] / 1e6, norm["Ms"] / 1e6) for k in (1, 2, 3)],
        ("wind speed estimate (m/s)", lambda x, n=norm: n["v_bar"] + x * n["sigma_v"], norm["sigma_v"]),
        ("generator speed (rpm)", lambda x, n=norm: (7.061 / RAD2RPM + x * n["sigma_w"]) * RAD2RPM, norm["sigma_w"] * RAD2RPM),
        ("collective pitch (deg)", lambda x, n=norm: np.rad2deg(n["beta_bar"] + x * n["sigma_beta"]), np.rad2deg(norm["sigma_beta"])),
    ]


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)) if y.var() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run under outputs/")
    ap.add_argument("--states", help=".npz with an 'obs' array of visited states (from windctrl.evaluate)")
    args = ap.parse_args()

    ckpt = torch.load(ROOT / "outputs" / args.run / "checkpoint.pt", weights_only=False)
    if ckpt["policy"] != "kan":
        raise SystemExit("interpretation by edges needs a KAN policy")
    model = KANPolicy()
    model.load_state_dict(ckpt["model"])
    norm, theta_max = ckpt["norm"], ckpt["args"]["theta_max_deg"]
    out = ROOT / "outputs" / args.run / "interpret"
    out.mkdir(parents=True, exist_ok=True)

    if args.states:
        S = np.load(args.states)["obs"].astype(np.float32)
        source = args.states
    else:
        S = np.random.default_rng(0).standard_normal((20000, 6)).astype(np.float32)
        S[:, :3] -= S[:, :3].mean(1, keepdims=True)   # respect dM1 + dM2 + dM3 = 0
        source = "standard-normal stand-in (no --states given)"

    layer = model.mean
    with torch.no_grad():
        E = layer.edges(torch.as_tensor(S)).numpy()          # (N, out i, in j)
        bias = layer.b.numpy().copy()
        grid = np.linspace(-3, 3, 241, dtype=np.float32)
        G = layer.edges(torch.as_tensor(np.repeat(grid[:, None], 6, 1))).numpy()   # edges on a grid
    mean_action = np.tanh(E.sum(-1) + bias)

    # 1. Canonical form: remove the common slope of the three load edges, then centre every edge.
    slopes = np.array([[np.polyfit(S[:, j], E[:, i, j], 1)[0] for j in range(6)] for i in range(3)])
    common = slopes[:, :3].mean(1)                            # per output
    for i in range(3):
        for j in range(3):
            E[:, i, j] -= common[i] * S[:, j]
            G[:, i, j] -= common[i] * grid
    centre = E.mean(0)
    E -= centre
    G -= centre
    bias += centre.sum(1)
    drift = float(np.abs(np.tanh(E.sum(-1) + bias) - mean_action).max())

    # 3. Edge sizes in degrees.
    size = theta_max * E.std(0)

    # 4. Linear law (gain matrix), then the blade-symmetric four-gain law.
    X = np.c_[S, np.ones(len(S))]
    W = np.linalg.lstsq(X, E.sum(-1) + bias, rcond=None)[0].T          # (3, 7): gains + constant
    W[:, :3] -= W[:, :3].mean(1, keepdims=True)                         # load gains sum to zero
    r2_lin = [r2(mean_action[:, i], np.tanh(X @ W[i])) for i in range(3)]
    g_s = np.mean([W[i, i] for i in range(3)])
    g_n = np.mean([W[i, (i + 1) % 3] for i in range(3)])
    g_p = np.mean([W[i, (i - 1) % 3] for i in range(3)])
    k_w, b0 = W[:, 4].mean(), W[:, 6].mean()
    four = np.stack([np.tanh(g_s * S[:, i] + g_n * S[:, (i + 1) % 3] + g_p * S[:, (i - 1) % 3] + k_w * S[:, 4] + b0)
                     for i in range(3)], 1)
    r2_four = [r2(mean_action[:, i], four[:, i]) for i in range(3)]

    # Physical units: deg per MN m, deg per rpm.
    per_mnm, per_rpm = theta_max / (norm["Ms"] / 1e6), theta_max / (norm["sigma_w"] * RAD2RPM)
    g1 = (g_s + g_n * np.exp(2j * np.pi / 3) + g_p * np.exp(-2j * np.pi / 3)) * per_mnm
    law = {"theta_max_deg": theta_max, "g_s": float(g_s), "g_n": float(g_n), "g_p": float(g_p), "k": float(k_w),
           "b": float(b0), "norm": {k: v for k, v in norm.items() if not k.startswith("_")},
           "physical": {"own_load_deg_per_MNm": float(g_s * per_mnm), "next_blade_deg_per_MNm": float(g_n * per_mnm),
                        "previous_blade_deg_per_MNm": float(g_p * per_mnm), "speed_deg_per_rpm": float(k_w * per_rpm),
                        "constant_deg": float(b0 * theta_max), "G1_abs_deg_per_MNm": float(abs(g1)),
                        "G1_phase_deg": float(np.degrees(np.angle(g1)))}}
    (out / "law.json").write_text(json.dumps(law, indent=2))

    # 2. Plot the 18 edges.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes_info = physical_axes(norm)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    for j, ax in enumerate(axs.flat):
        label, to_phys, _ = axes_info[j]
        lo, hi = np.percentile(S[:, j], [1, 99])
        ax.axvspan(to_phys(lo), to_phys(hi), color="0.9", zorder=0)
        for i, ls in enumerate(["-", "--", ":"]):
            ax.plot(to_phys(grid), theta_max * G[:, i, j], ls, color="tab:purple", lw=1.6, label=f"blade {i + 1}")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel(label)
        ax.set_title(f"input: {OBS_NAMES[j]}", fontsize=10)
        if j % 3 == 0:
            ax.set_ylabel("edge value (deg)")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{args.run}: KAN edge functions in canonical form (shaded: 1-99 % of states)")
    fig.tight_layout()
    fig.savefig(out / "edges.png", dpi=150)

    p = law["physical"]
    lines = [f"# Interpretation of `{args.run}` (update {ckpt['update']})", "",
             f"States: {source}, {len(S)} samples. Canonical form changes the mean action by at most {drift:.1e}.", "",
             "## Edge sizes (std of each edge's contribution over the states, deg)", "",
             "| | " + " | ".join(OBS_NAMES) + " |", "|---|" + "---|" * 6,
             *[f"| blade {i + 1} | " + " | ".join(f"{size[i, j]:.3f}" for j in range(6)) + " |" for i in range(3)], "",
             "## Linear and four-gain laws", "",
             f"R^2 against the KAN mean action per blade: linear {', '.join(f'{v:.3f}' for v in r2_lin)}; "
             f"four gains {', '.join(f'{v:.3f}' for v in r2_four)}.", "",
             "Four-gain law near zero action (deg, MN m, rpm):", "",
             f"    d_beta_i = {p['own_load_deg_per_MNm']:+.3f} dM_i {p['next_blade_deg_per_MNm']:+.3f} dM_i+1 "
             f"{p['previous_blade_deg_per_MNm']:+.3f} dM_i-1 {p['speed_deg_per_rpm']:+.3f} (w_g - w_ref) "
             f"{p['constant_deg']:+.3f}", "",
             f"1P complex gain: |G1| = {p['G1_abs_deg_per_MNm']:.3f} deg/(MN m), phase {p['G1_phase_deg']:+.1f} deg.", "",
             "![edges](edges.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:-1]))
    print(f"\nwritten: {out}/report.md, edges.png, law.json")


if __name__ == "__main__":
    main()
