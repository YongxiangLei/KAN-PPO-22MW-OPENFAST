"""PPO for per-blade residual pitch control of the IEA 22 MW turbine, with a KAN or MLP policy.

Networks, hyper-parameters and training protocol follow the reference KAN-PPO IPC paper:

    KAN  actor: two single KAN layers [6, 3] (mean = tanh(.), std = softplus(.)), 330 parameters
         critic: KAN [6, 6, 1], output x10, 385 parameters
    MLP  actor: 6-64-64 ReLU body with mean and std heads, 4998 parameters; critic 6-64-64-1

    lr 3e-4 (actor) / 1e-3 (critic) with linear decay, gamma 0.99, lambda 0.95, clip 0.2, 10 epochs,
    minibatch 512, no entropy bonus, grad-norm clip 0.5, initial std 0.3, three 300-s episodes per
    update (13, 16, 19 m/s; turbulence seeds 101/102 alternating), 100 updates.

The same batch can be collected faster with more, shorter episodes side by side, e.g. twelve 75-s
episodes (four seeds per wind speed) after a 20-s start-up that is simulated but not learned from:

    python -m windctrl.train --run kan_v2 --episodes-per-speed 4 --episode-seconds 75 \
        --burn-in-seconds 20 --seeds 101 102 103 104 105 106 107 108

One deliberate difference: the pitch rate limit defaults to the 22 MW model's own 2 deg/s rather
than the paper's 10 deg/s (pass --pitch-rate-deg 10 to reproduce the paper's setting).

    python -m windctrl.train --policy kan --run kan_v1
    python -m windctrl.train --policy mlp --run mlp_v1

The first run calibrates the state normalisation (config/norm.json) from ROSCO-only episodes on
the training winds. One update takes about 17 minutes of wall time (300 s at ~0.3x real time),
so 100 updates take about 29 hours; the three episodes of an update run in parallel.
"""
import argparse
import csv
import json
import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from windctrl.env import NORM_FILE, P_RATED, ROOT, W_REF, IPCEnv, load_norm
from windctrl.kan import KAN, KANLayer

OBS_NAMES = ["dM1", "dM2", "dM3", "v_hat", "omega_g", "beta"]
INV_SOFTPLUS = lambda y: float(np.log(np.expm1(y)))


# ---- policies ----------------------------------------------------------------------------------

class KANPolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=3, init_std=0.3):
        super().__init__()
        self.mean = KANLayer(obs_dim, act_dim, sigma_w=0.01, sigma_c=0.01)
        self.std = KANLayer(obs_dim, act_dim, sigma_w=0.01, sigma_c=0.01, bias=INV_SOFTPLUS(init_std))
        self.critic = KAN([obs_dim, 6, 1], sigma_w=0.3, sigma_c=0.1)

    def dist(self, s):
        return torch.distributions.Normal(torch.tanh(self.mean(s)), F.softplus(self.std(s)) + 1e-4)

    def value(self, s):
        return 10.0 * self.critic(s).squeeze(-1)   # returns are of order -100

    def actor_parameters(self):
        return [*self.mean.parameters(), *self.std.parameters()]


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=3, init_std=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.mean, self.std = nn.Linear(64, act_dim), nn.Linear(64, act_dim)
        self.critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean.weight, 0.01)
        nn.init.orthogonal_(self.std.weight, 0.01)
        nn.init.constant_(self.std.bias, INV_SOFTPLUS(init_std))

    def dist(self, s):
        h = self.body(s)
        return torch.distributions.Normal(torch.tanh(self.mean(h)), F.softplus(self.std(h)) + 1e-4)

    def value(self, s):
        return self.critic(s).squeeze(-1)

    def actor_parameters(self):
        return [*self.body.parameters(), *self.mean.parameters(), *self.std.parameters()]


def count(params):
    return sum(p.numel() for p in params)


# ---- parallel episodes -------------------------------------------------------------------------

def _worker(conn, env_kwargs):
    env = IPCEnv(**env_kwargs)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(env.reset(options=data))
            elif cmd == "step":
                conn.send(env.step(data))
            else:
                break
    finally:
        env.close()
        conn.close()


class Workers:
    """One IPCEnv (one OpenFAST process) per worker, stepped in lock-step."""

    def __init__(self, n, env_kwargs, tag):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            kw = {**env_kwargs, "workdir": ROOT / "runs" / tag / f"env{i:02d}"}
            p = ctx.Process(target=_worker, args=(child, kw), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def rollout(self, episodes, act_fn):
        """Run one episode per entry of `episodes` (reset options); act_fn(obs[n_live, 6]) -> actions.

        Returns one dict per episode with arrays obs, act, rew, the final observation, whether the
        episode failed, and the raw ROSCO measurements of every decision step.
        """
        live = list(range(len(episodes)))
        for i in live:
            self.conns[i].send(("reset", episodes[i]))
        obs = {i: self.conns[i].recv()[0] for i in live}
        traj = [{"obs": [], "act": [], "rew": [], "info": [], "failed": False} for _ in episodes]
        while live:
            actions = act_fn(np.stack([obs[i] for i in live]))
            for i, a in zip(live, actions):
                self.conns[i].send(("step", a))
            still = []
            for i, a in zip(live, actions):
                o, r, term, trunc, info = self.conns[i].recv()
                t = traj[i]
                t["obs"].append(obs[i]); t["act"].append(a); t["rew"].append(r); t["info"].append(info)
                obs[i] = o
                if term or trunc:
                    t["final_obs"], t["failed"] = o, bool(term)
                else:
                    still.append(i)
            live = still
        for t in traj:
            t["obs"], t["act"], t["rew"] = (np.asarray(t[k], dtype=np.float32) for k in ("obs", "act", "rew"))
        return traj

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=30)


# ---- calibration of the state normalisation ----------------------------------------------------

def calibrate(args, env_kwargs):
    """ROSCO-only (zero offset) episodes on the training winds -> config/norm.json and the PI reward level."""
    episodes = [{"wind_speed": ws, "wind_seed": sd, "name": f"pi_ws{ws:g}_s{sd}"}
                for ws in args.speeds for sd in args.seeds]
    workers = Workers(len(episodes), {**env_kwargs, "norm": load_norm()}, f"{args.run}_calib")
    traj = workers.rollout(episodes, lambda o: np.zeros((len(o), 3), dtype=np.float32))
    workers.close()
    skip = int(min(60.0, args.episode_seconds / 2) / 0.1)   # drop the start-up transient
    info = [m for t in traj for m in t["info"][skip:]]
    moops = np.array([[m[f"rootMOOP{k}"] for k in (1, 2, 3)] for m in info])
    d_m = moops - moops.mean(1, keepdims=True)
    v, w, beta = (np.array([m[k] for m in info]) for k in ("WE_Vw", "GenSpeed", "BlPitchCMeas"))
    norm = {"Ms": float(d_m.std()), "v_bar": float(v.mean()), "sigma_v": float(v.std()),
            "sigma_w": float(np.sqrt(np.mean((w - W_REF) ** 2))), "beta_bar": float(beta.mean()),
            "sigma_beta": float(beta.std()),
            "_source": {"speeds": args.speeds, "seeds": args.seeds, "episode_seconds": args.episode_seconds,
                        "failed_episodes": int(sum(t["failed"] for t in traj)),
                        "mean_power_MW": float(np.mean([m["VS_GenPwr"] for m in info]) / 1e6)}}
    NORM_FILE.parent.mkdir(exist_ok=True)
    NORM_FILE.write_text(json.dumps(norm, indent=2))
    print("calibrated normalisation:", json.dumps(norm), flush=True)
    return norm


# ---- PPO ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["kan", "mlp"], default="kan")
    ap.add_argument("--run", default="kan_ipc", help="name for outputs/<run> and runs/<run>_*")
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--speeds", type=float, nargs="+", default=[13.0, 16.0, 19.0], help="mean wind speeds (m/s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102], help="training turbulence seeds, alternated")
    ap.add_argument("--episode-seconds", type=float, default=300.0, help="simulated time used for learning per episode")
    ap.add_argument("--episodes-per-speed", type=int, default=1,
                    help="parallel episodes per wind speed in one update, each on its own seed and CPU core")
    ap.add_argument("--burn-in-seconds", type=float, default=0.0,
                    help="start-up time simulated before each episode and left out of the PPO batch")
    ap.add_argument("--pitch-rate-deg", type=float, default=2.0,
                    help="pitch rate limit (deg/s). 2 is the 22 MW model's own; the reference paper used 10")
    ap.add_argument("--theta-max-deg", type=float, default=4.0)
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--lr-critic", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--recalibrate", action="store_true", help="recompute config/norm.json even if it exists")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    out = ROOT / "outputs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    env_kwargs = {"episode_seconds": args.episode_seconds, "pitch_rate_deg": args.pitch_rate_deg,
                  "theta_max_deg": args.theta_max_deg}
    norm = calibrate(args, env_kwargs) if args.recalibrate or not NORM_FILE.is_file() else load_norm()
    env_kwargs.update(norm=norm, episode_seconds=args.episode_seconds + args.burn_in_seconds)
    per_speed, burn = args.episodes_per_speed, int(round(args.burn_in_seconds / 0.1))

    model = (KANPolicy if args.policy == "kan" else MLPPolicy)()
    print(f"{args.policy}: actor {count(model.actor_parameters())} parameters, "
          f"critic {count(model.critic.parameters())}", flush=True)
    opt = torch.optim.Adam([{"params": model.actor_parameters(), "lr": args.lr_actor},
                            {"params": model.critic.parameters(), "lr": args.lr_critic}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: 1.0 - u / args.updates)

    workers = Workers(len(args.speeds) * per_speed, env_kwargs, f"{args.run}_train")

    def act(obs):
        with torch.no_grad():
            return model.dist(torch.as_tensor(obs)).sample().numpy()

    with open(out / "progress.csv", "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["update", "seed", *[f"reward_ws{ws:g}" for ws in args.speeds], "reward_mean",
                      "balance_term", "power_term", "smooth_term", "policy_std", "failed", "minutes"])
        start = time.time()
        for u in range(1, args.updates + 1):
            # Each update takes the next `per_speed` seeds in turn, so all training winds are used equally.
            seeds = [args.seeds[((u - 1) * per_speed + k) % len(args.seeds)] for k in range(per_speed)]
            seed = "+".join(str(sd) for sd in seeds)
            episodes = [{"wind_speed": ws, "wind_seed": sd, "name": f"ws{ws:g}_s{sd}"} for ws in args.speeds for sd in seeds]
            traj = workers.rollout(episodes, act)
            for t in traj:                      # leave the start-up period out of the batch
                t["obs"], t["act"], t["rew"], t["info"] = t["obs"][burn:], t["act"][burn:], t["rew"][burn:], t["info"][burn:]
            speed_of = [ep["wind_speed"] for ep, t in zip(episodes, traj) if len(t["rew"])]
            traj = [t for t in traj if len(t["rew"])]   # an episode that aborted during start-up has nothing left

            obs_l, act_l, logp_l, adv_l, ret_l = [], [], [], [], []
            for t in traj:
                o = torch.as_tensor(t["obs"])
                with torch.no_grad():
                    v = model.value(o).numpy()
                    # Time-limit end: bootstrap from the value of the last state. Failure: no bootstrap.
                    v_last = 0.0 if t["failed"] else model.value(torch.as_tensor(t["final_obs"])[None]).item()
                    logp = model.dist(o).log_prob(torch.as_tensor(t["act"])).sum(-1).numpy()
                adv, gae = np.zeros(len(v), np.float32), 0.0
                for k in reversed(range(len(v))):
                    nxt = v_last if k == len(v) - 1 else v[k + 1]
                    gae = t["rew"][k] + args.gamma * nxt - v[k] + args.gamma * args.gae_lambda * gae
                    adv[k] = gae
                obs_l.append(t["obs"]); act_l.append(t["act"]); logp_l.append(logp); adv_l.append(adv); ret_l.append(adv + v)
            f_obs, f_act, f_logp, f_adv, f_ret = (torch.as_tensor(np.concatenate(x)) for x in
                                                  (obs_l, act_l, logp_l, adv_l, ret_l))
            n = len(f_obs)
            for _ in range(args.epochs):
                for idx in torch.randperm(n).split(args.minibatch):
                    a = f_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    d = model.dist(f_obs[idx])
                    ratio = (d.log_prob(f_act[idx]).sum(-1) - f_logp[idx]).exp()
                    pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    v_loss = (model.value(f_obs[idx]) - f_ret[idx]).pow(2).mean()
                    opt.zero_grad()
                    (pg + 0.5 * v_loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
            sched.step()

            # Reward terms recomputed from the logged measurements, for the training curves.
            info = [m for t in traj for m in t["info"] if "VS_GenPwr" in m]
            moops = np.array([[m[f"rootMOOP{k}"] for k in (1, 2, 3)] for m in info])
            bal = np.abs(moops - moops.mean(1, keepdims=True)).sum(1).mean() / 3 / norm["Ms"]
            pw = 5.0 * np.mean([abs(m["VS_GenPwr"] - P_RATED) for m in info]) / P_RATED
            rew = [float(np.mean([t["rew"].mean() for t, w in zip(traj, speed_of) if w == ws] or [np.nan]))
                   for ws in args.speeds]
            with torch.no_grad():
                std = model.dist(f_obs).stddev.mean().item()
            row = [u, seed, *rew, float(np.mean(rew)), bal, pw, -float(np.mean(rew)) - bal - pw, std,
                   sum(t["failed"] for t in traj) + len(episodes) - len(traj), (time.time() - start) / 60]
            log.writerow([f"{x:.5g}" if isinstance(x, float) else x for x in row])
            fh.flush()
            print(f"update {u:3d} seed {seed}  reward " + " ".join(f"{r:+.3f}" for r in rew) +
                  f"  balance {bal:.3f} power {pw:.3f}  std {std:.3f}  failed {row[-2]}  {row[-1]:.1f} min", flush=True)
            torch.save({"policy": args.policy, "model": model.state_dict(), "norm": norm, "args": vars(args),
                        "obs_names": OBS_NAMES, "update": u}, out / "checkpoint.pt")
    workers.close()


if __name__ == "__main__":
    main()
