"""Closed-loop evaluation: fatigue DELs and regulation quality of several controllers on the same winds.

Controllers are given as specs:

    pi                 ROSCO alone (collective pitch + torque), no pitch offsets
    rosco_ipc:<KI>     ROSCO's own 1P IPC (Coleman transform + integrator), KI in rad/(N m s)
    kan:<run>          trained policy from outputs/<run>/checkpoint.pt, mean action (also mlp:<run>)
    law:<run>          explicit four-gain law extracted by windctrl.interpret from a KAN run

    python -m windctrl.evaluate --name eval_v1 --controllers pi rosco_ipc:1e-9 kan:kan_v1 mlp:mlp_v1

    # choose the ROSCO IPC gain on the *training* winds, as the reference paper does
    python -m windctrl.evaluate --name ipc_sweep --winds train --controllers pi rosco_ipc:3e-10 rosco_ipc:1e-9 rosco_ipc:3e-9

Every (controller, wind) pair is one OpenFAST run; they all run in parallel, so give the job
as many cores as there are pairs. Indicators follow the reference paper:

    DEL = ( sum_k n_k S_k^m / (f_eq T) )^(1/m),  rainflow ranges S_k with half counts for the
    residue, f_eq = 1 Hz, m = 10 for the blades and 4 for the tower; the first 60 s are dropped.
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from windctrl.env import NORM_FILE, ROOT, load_norm
from windctrl.train import KANPolicy, MLPPolicy, Workers

RATED_RPM = 7.061
WIND_SETS = {"train": (101, 102), "eval": (201, 202)}
INDICATORS = [("blade_DEL", "MN m"), ("tower_base_DEL", "MN m"), ("tower_top_DEL", "MN m"), ("power_std", "MW"),
              ("speed_RMSE", "rpm"), ("pitch_travel", "deg/s"), ("max_pitch", "deg"), ("mean_power", "MW")]


# ---- fatigue ------------------------------------------------------------------------------------

def rainflow(x):
    """ASTM E1049-85 rainflow counting. Returns (ranges, counts) with counts of 1 or 0.5."""
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    keep = np.concatenate(([True], d[1:] * d[:-1] < 0, [True])) if len(x) > 2 else np.ones(len(x), bool)
    tp = x[keep]                      # turning points (reversals), end points included
    ranges, counts, stack = [], [], []
    for p in tp:
        stack.append(p)
        while len(stack) >= 3:
            rx, ry = abs(stack[-1] - stack[-2]), abs(stack[-2] - stack[-3])
            if rx < ry:
                break
            if len(stack) == 3:       # range Y contains the starting point: half cycle
                ranges.append(ry); counts.append(0.5)
                stack.pop(0)
            else:                     # full cycle; discard its two points
                ranges.append(ry); counts.append(1.0)
                last = stack.pop()
                stack.pop(); stack.pop()
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):   # residue: half cycles
        ranges.append(abs(b - a)); counts.append(0.5)
    return np.array(ranges), np.array(counts)


def del_(x, m, duration, f_eq=1.0):
    s, n = rainflow(x)
    return float((np.sum(n * s ** m) / (f_eq * duration)) ** (1.0 / m))


def indicators(outb, skip):
    """Indicators of one run from its OpenFAST binary output, ignoring the first `skip` seconds."""
    from openfast_io.FAST_output_reader import FASTOutputFile
    df = FASTOutputFile(str(outb)).toDataFrame()
    col = lambda name: df[next(c for c in df.columns if c.split("_[")[0] == name)].to_numpy()
    t = col("Time")
    sel = t >= skip
    dur = float(t[sel][-1] - t[sel][0])
    pitch = np.stack([col(f"BldPitch{k}")[sel] for k in (1, 2, 3)])
    return {
        "blade_DEL": np.mean([del_(col(f"RootMyc{k}")[sel], 10, dur) for k in (1, 2, 3)]) / 1e3,   # kN m -> MN m
        "tower_base_DEL": del_(col("TwrBsMyt")[sel], 4, dur) / 1e3,
        "tower_top_DEL": del_(col("YawBrMyp")[sel], 4, dur) / 1e3,
        "power_std": float(col("GenPwr")[sel].std() / 1e3),                                       # kW -> MW
        "speed_RMSE": float(np.sqrt(np.mean((col("GenSpeed")[sel] - RATED_RPM) ** 2))),
        "pitch_travel": float(np.abs(np.diff(pitch, axis=1)).sum() / 3 / dur),
        "max_pitch": float(pitch.max()),
        "mean_power": float(col("GenPwr")[sel].mean() / 1e3),
        "duration": dur,
    }


# ---- controllers --------------------------------------------------------------------------------

def build_controller(spec):
    """spec -> (label, extra env kwargs, act_fn, norm or None)."""
    kind, _, arg = spec.partition(":")
    zero = lambda obs: np.zeros((len(obs), 3), dtype=np.float32)
    if kind == "pi":
        return "PI", {}, zero, None
    if kind == "rosco_ipc":
        return f"ROSCO IPC {float(arg):.0e}", {"rosco_ipc_ki": float(arg)}, zero, None
    if kind in ("kan", "mlp"):
        ckpt = torch.load(ROOT / "outputs" / arg / "checkpoint.pt", weights_only=False)
        if ckpt["policy"] != kind:
            raise SystemExit(f"{arg} holds a {ckpt['policy']} policy, not {kind}")
        model = (KANPolicy if kind == "kan" else MLPPolicy)()
        model.load_state_dict(ckpt["model"])

        def act(obs):
            with torch.no_grad():
                return model.dist(torch.as_tensor(obs)).mean.numpy()   # mean action, no exploration noise
        return f"{kind.upper()} {arg} (update {ckpt['update']})", {}, act, ckpt["norm"]
    if kind == "law":
        law = json.loads((ROOT / "outputs" / arg / "interpret" / "law.json").read_text())

        def act(obs):
            z = [law["g_s"] * obs[:, i] + law["g_n"] * obs[:, (i + 1) % 3] + law["g_p"] * obs[:, (i - 1) % 3]
                 + law["k"] * obs[:, 4] + law["b"] for i in range(3)]
            return np.tanh(np.stack(z, 1)).astype(np.float32)
        return f"Four-gain law {arg}", {"theta_max_deg": law["theta_max_deg"]}, act, law["norm"]
    raise SystemExit(f"unknown controller spec '{spec}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="eval", help="results go to outputs/<name>, runs to runs/<name>_*")
    ap.add_argument("--controllers", nargs="+", default=["pi"])
    ap.add_argument("--winds", choices=list(WIND_SETS), default="eval", help="train: seeds 101/102, eval: 201/202")
    ap.add_argument("--speeds", type=float, nargs="+", default=[13.0, 16.0, 19.0])
    ap.add_argument("--seeds", type=int, nargs="+", help="override the seeds of --winds")
    ap.add_argument("--episode-seconds", type=float, default=600.0)
    ap.add_argument("--skip-seconds", type=float, default=60.0)
    ap.add_argument("--pitch-rate-deg", type=float, default=2.0, help="same limit for every controller")
    ap.add_argument("--jobs", type=int, help="controllers run at once (default: all); 1 = one after another")
    ap.add_argument("--only", type=int, nargs="+",
                    help="positions (from 0) in --controllers to run in this call. The rest are expected from other "
                         "calls with the same --name and --controllers (e.g. on another node); the summary is "
                         "written by whichever call finds all of them done")
    args = ap.parse_args()

    seeds = args.seeds or WIND_SETS[args.winds]
    episodes = [{"wind_speed": ws, "wind_seed": sd, "name": f"ws{ws:g}_s{sd}"} for ws in args.speeds for sd in seeds]
    out = ROOT / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(args).items() if k != "only"}, indent=2))
    if args.only is None:                      # a complete run: forget rows left by an earlier one
        for old in out.glob("rows_c*.json"):
            old.unlink()
    if not NORM_FILE.is_file():
        print("note: config/norm.json missing; baselines do not need it, policies carry their own")

    def run(i_spec):
        i, spec = i_spec
        label, extra, act, norm = build_controller(spec)
        tag = f"{args.name}_c{i}"
        workers = Workers(len(episodes), {"episode_seconds": args.episode_seconds, "pitch_rate_deg": args.pitch_rate_deg,
                                          "norm": norm or load_norm(), "keep_runs": True, **extra}, tag)
        traj = workers.rollout(episodes, act)
        workers.close()
        np.savez_compressed(out / f"states_c{i}.npz", obs=np.concatenate([t["obs"] for t in traj]),
                            act=np.concatenate([t["act"] for t in traj]))
        rows = []
        for j, (ep, t) in enumerate(zip(episodes, traj)):
            row = {"controller": label, "spec": spec, "wind_speed": ep["wind_speed"], "seed": ep["wind_seed"],
                   "failed": t["failed"], "mean_reward": float(t["rew"].mean())}
            outb = ROOT / "runs" / tag / f"env{j:02d}" / ep["name"] / f"{ep['name']}.outb"
            if not t["failed"] and outb.is_file():
                row.update(indicators(outb, args.skip_seconds))
            rows.append(row)
        (out / f"rows_c{i}.json").write_text(json.dumps(rows))

    todo = [(i, spec) for i, spec in enumerate(args.controllers) if args.only is None or i in args.only]
    with ThreadPoolExecutor(args.jobs or len(todo)) as pool:
        list(pool.map(run, todo))
    parts = [out / f"rows_c{i}.json" for i in range(len(args.controllers))]
    missing = [args.controllers[i] for i, part in enumerate(parts) if not part.is_file()]
    if missing:
        print(f"not evaluated yet: {missing}; the summary is written once they are done")
        return
    rows = [r for part in parts for r in json.loads(part.read_text())]

    fields = ["controller", "spec", "wind_speed", "seed", "failed", "mean_reward", *[k for k, _ in INDICATORS], "duration"]
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Summary: mean over winds per controller, and change from the first controller (normally PI).
    labels = list(dict.fromkeys(r["controller"] for r in rows))
    mean = {lab: {k: np.mean([r[k] for r in rows if r["controller"] == lab and k in r]) for k, _ in INDICATORS}
            for lab in labels}
    failed = {lab: sum(r["failed"] for r in rows if r["controller"] == lab) for lab in labels}
    ref = labels[0]
    lines = [f"Winds: {args.winds} seeds {list(seeds)}, speeds {args.speeds} m/s, {args.episode_seconds:g} s runs, "
             f"first {args.skip_seconds:g} s dropped, pitch rate limit {args.pitch_rate_deg:g} deg/s. "
             f"Mean over {len(episodes)} runs; brackets: change from {ref}.", "",
             "| Indicator | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for k, unit in INDICATORS:
        cells = []
        for lab in labels:
            v, r = mean[lab][k], mean[ref][k]
            cells.append(f"{v:.3g}" + (f" ({(v / r - 1) * 100:+.0f}%)" if lab != ref and r else ""))
        lines.append(f"| {k} ({unit}) | " + " | ".join(cells) + " |")
    lines.append("| failed runs | " + " | ".join(str(failed[lab]) for lab in labels) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nper-run results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
