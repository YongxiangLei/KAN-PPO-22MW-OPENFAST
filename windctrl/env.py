"""Gymnasium environment: per-blade residual pitch control (IPC) on top of ROSCO, IEA 22 MW monopile.

Follows the problem formulation of "Individual Pitch Control of a Large-Scale Wind Turbine With
an Interpretable Kolmogorov-Arnold Reinforcement Learning Policy" (IEA 15 MW there), with the
turbine replaced by the IEA 22 MW reference turbine.

ROSCO does collective pitch and torque control. Every 0.1 s the policy receives

    s = [dM1/Ms, dM2/Ms, dM3/Ms, (v_hat - v_bar)/sigma_v, (w_g - w_ref)/sigma_w, (beta - beta_bar)/sigma_beta]

where dMi = M_y,i - mean_j M_y,j is the out-of-plane root-moment difference of blade i, v_hat the
wind speed estimated by ROSCO, w_g the generator speed and beta the collective pitch. The action
a in [-1, 1]^3 sets one pitch offset per blade, d_beta_i = theta_max * a_i (theta_max = 4 deg),
smoothed by a first-order filter (time constant 0.1 s) and added to ROSCO's pitch command. The
policy works in the rotating frame: no azimuth input, no Coleman transform.

    r = -(w_bal/3) sum_i |dMi|/Ms - w_pow |P - P_rated|/P_rated - (w_smooth/3) sum_i |a_i,k - a_i,k-1|

Differences from the paper that come from the tooling, not the method:
  * OpenFAST runs standalone and talks to Python over ROSCO's ZeroMQ interface (the paper uses a
    Simulink S-function). The exchange happens every 0.02 s so the offset filter runs between
    decisions; ROSCO is patched to also send v_hat and beta (patches/).
  * OpenFAST time step 0.002 s with ElastoDyn blades (see README) instead of 0.005 s.
"""
import fcntl
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import zmq

import importlib.util

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_cases", ROOT / "scripts" / "make_cases.py")
make_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_cases)

# Sent by the patched ROSCO, in this order (rosco_registry/wfc_interface.yaml + patches/).
MEASUREMENTS = ["ZMQ_ID", "iStatus", "Time", "VS_MechGenPwr", "VS_GenPwr", "GenSpeed", "RotSpeed", "GenTqMeas",
                "NacHeading", "NacVane", "HorWindV", "rootMOOP1", "rootMOOP2", "rootMOOP3", "FA_Acc_TT",
                "NacIMU_FA_RAcc", "Azimuth", "WE_Vw", "BlPitchCMeas"]
SETPOINT_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]   # offsets 0, power-reference ratios 1
PITCH_OFFSET_SLOTS = (2, 3, 4)

P_RATED = 22.0e6                         # W
W_REF = 7.061 * 2 * np.pi / 60           # rated generator (= rotor, direct drive) speed, rad/s
NORM_FILE = ROOT / "config" / "norm.json"
WIND_DIR = ROOT / "runs" / "winds"
# Placeholders until `python -m windctrl.train` has calibrated them from ROSCO-only runs.
DEFAULT_NORM = {"Ms": 1.0e7, "v_bar": 16.0, "sigma_v": 2.5, "sigma_w": 0.035, "beta_bar": 0.21, "sigma_beta": 0.07}


def load_norm():
    return json.loads(NORM_FILE.read_text()) if NORM_FILE.is_file() else dict(DEFAULT_NORM)


def set_discon(path, key, value):
    """DISCON.IN lines look like `value   ! key - description`."""
    new, n = re.subn(rf"^.*?(\s*!\s*{re.escape(key)}\s)", lambda m: f"{value}{m.group(1)}", path.read_text(), flags=re.M)
    if n != 1:
        raise ValueError(f"expected one '{key}' line in {path.name}, found {n}")
    path.write_text(new)


def turbulent_wind(ws, seed, seconds, turb_class="B", turb_type="NTM"):
    """Path of the TurbSim field for (mean speed, seed), generating it once; safe across processes."""
    WIND_DIR.mkdir(parents=True, exist_ok=True)
    # No dots in the stem: TurbSim and Path.with_suffix both treat the last dot as the extension.
    stem = f"{turb_type}{turb_class}_ws{make_cases.tag(ws)}_seed{seed}_{int(round(seconds))}s"
    bts, inp = WIND_DIR / f"{stem}.bts", WIND_DIR / f"{stem}.inp"
    with open(WIND_DIR / f"{stem}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not bts.is_file():
            inp.write_text(make_cases.TURBSIM_TEMPLATE.format(
                seed=seed, ngrid=25, dt=0.05, tmax=seconds, hub=make_cases.HUB_HT, grid=make_cases.GRID,
                uref=ws, plexp=make_cases.PL_EXP, turb_class=turb_class, turb_type=turb_type))
            with open(WIND_DIR / f"{stem}.log", "w") as log:
                subprocess.run(["turbsim", inp.name], cwd=WIND_DIR, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not bts.is_file():
                raise RuntimeError(f"TurbSim did not write {bts}")
    return bts


class IPCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_seconds=300.0, decision_dt=0.1, exchange_dt=0.02, theta_max_deg=4.0,
                 filter_tau=0.1, pitch_rate_deg=2.0, wave_hs=4.52, wave_tp=9.45,
                 w_bal=1.0, w_pow=5.0, w_smooth=0.1, norm=None, workdir=None, keep_runs=False, timeout=600.0,
                 rosco_ipc_ki=0.0, output_dt=0.02):
        self.episode_seconds, self.decision_dt, self.exchange_dt = episode_seconds, decision_dt, exchange_dt
        self.substeps = int(round(decision_dt / exchange_dt))
        self.theta_max = np.deg2rad(theta_max_deg)
        self.alpha = 1.0 - np.exp(-exchange_dt / filter_tau)
        self.pitch_rate = np.deg2rad(pitch_rate_deg)
        self.wave_hs, self.wave_tp = wave_hs, wave_tp
        self.w_bal, self.w_pow, self.w_smooth = w_bal, w_pow, w_smooth
        # rosco_ipc_ki > 0 switches on ROSCO's own 1P IPC (Coleman transform + integrator) with that
        # integral gain in rad/(N m s); used as a baseline, with the policy offsets left at zero.
        self.rosco_ipc_ki, self.output_dt = rosco_ipc_ki, output_dt
        self.norm = norm or load_norm()
        self.workdir = Path(workdir) if workdir else ROOT / "runs" / f"env_{os.getpid()}"
        self.keep_runs, self.timeout_ms = keep_runs, int(timeout * 1000)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self._ctx = zmq.Context.instance()
        self._sock = self._proc = self._log = None
        self._episode = 0

    # ---- episode set-up -------------------------------------------------------------------

    def _build_case(self, ws, seed, port, name):
        sweep = self.workdir
        sweep.mkdir(parents=True, exist_ok=True)
        link = sweep / make_cases.SHARED.name
        if not link.exists():
            link.symlink_to(make_cases.SHARED)
        case = sweep / name
        if case.exists():
            shutil.rmtree(case)
        shutil.copytree(make_cases.MONOPILE, case)
        shutil.copy(make_cases.SHARED / make_cases.INFLOW, case / make_cases.INFLOW)

        bts = turbulent_wind(ws, seed, self.episode_seconds)
        make_cases.edit(case / make_cases.INFLOW, {"WindType": 3, "FileName_BTS": f'"{bts}"'})
        make_cases.edit(case / make_cases.FST, {
            "TMax": self.episode_seconds, "TStart": 0.0, "InflowFile": f'"{make_cases.INFLOW}"',
            "CompElast": 1, "DT": 0.002, "DT_Out": self.output_dt})
        make_cases.edit(case / "IEA-22-280-RWT_SeaState.dat", {
            "WaveMod": 2, "WaveHs": self.wave_hs, "WaveTp": self.wave_tp, "WaveSeed(1)": seed})

        ed = case / "IEA-22-280-RWT_ElastoDyn.dat"
        rpm, pitch = make_cases.initial_conditions(ws)
        make_cases.edit(ed, {"RotSpeed": f"{rpm:.3f}", **{f"BlPitch({k})": f"{pitch:.3f}" for k in (1, 2, 3)}})
        # Out-of-plane blade-root moments, for fatigue (DEL) post-processing of the .outb file.
        text = ed.read_text()
        ed.write_text(text.replace("\nEND of input file", '\n"RootMyc1"\n"RootMyc2"\n"RootMyc3"\nEND of input file', 1))

        # Ptch_Cntrl = 1: ServoDyn must accept per-blade commands or the offsets are ignored.
        make_cases.edit(case / "IEA-22-280-RWT_ServoDyn.dat", {
            "DLL_FileName": f'"{make_cases.ZMQ_LIBDISCON}"', "Ptch_Cntrl": 1})
        discon = case / "IEA-22-280-RWT_DISCON.IN"
        set_discon(discon, "ZMQ_Mode", 1)
        set_discon(discon, "ZMQ_CommAddress", f'"tcp://localhost:{port}"')
        set_discon(discon, "ZMQ_UpdatePeriod", self.exchange_dt)
        set_discon(discon, "PC_MaxRat", f"{self.pitch_rate:.6f}")
        set_discon(discon, "PC_MinRat", f"{-self.pitch_rate:.6f}")
        set_discon(discon, "LoggingLevel", 0)
        if self.rosco_ipc_ki > 0:
            set_discon(discon, "IPC_ControlMode", 1)
            set_discon(discon, "IPC_KI", f"{self.rosco_ipc_ki:.4e}    0.000e+00")
        (case / make_cases.FST).rename(case / f"{name}.fst")
        return case

    def reset(self, *, seed=None, options=None):
        """options: {"wind_speed": m/s, "wind_seed": int}; both are required for reproducible winds."""
        super().reset(seed=seed)
        self._stop()
        if not make_cases.ZMQ_LIBDISCON.is_file():
            raise RuntimeError(f"ZeroMQ-enabled ROSCO not found at {make_cases.ZMQ_LIBDISCON}; see README.md")
        opts = options or {}
        ws = float(opts.get("wind_speed", self.np_random.choice([13.0, 16.0, 19.0])))
        wind_seed = int(opts.get("wind_seed", self.np_random.integers(1, 10_000)))
        name = opts.get("name", f"ep{self._episode:05d}")
        self._episode += 1

        # Bind first so the port is ours before OpenFAST is told about it.
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        port = self._sock.bind_to_random_port("tcp://*")
        self._case = self._build_case(ws, wind_seed, port, name)
        self._log = open(self._case / f"{name}.log", "w")
        self._proc = subprocess.Popen(["openfast", f"{name}.fst"], cwd=self._case, stdout=self._log,
                                      stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"})
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._offset = np.zeros(3)
        self._meas = self._receive()
        if self._meas is None:
            raise RuntimeError(f"OpenFAST exited before the first measurement; see {self._log.name}")
        if len(self._meas) != len(MEASUREMENTS):
            raise RuntimeError("ROSCO sent an unexpected number of measurements; rebuild it with patches/ applied")
        return self._observe(), {"wind_speed": ws, "wind_seed": wind_seed, "case": str(self._case), **self._meas}

    # ---- ZeroMQ exchange ------------------------------------------------------------------

    def _receive(self):
        """Next measurement dict, or None if OpenFAST died or stopped talking."""
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        waited = 0
        while waited < self.timeout_ms:
            if poller.poll(500):
                values = [float(v) for v in self._sock.recv_string().replace("\x00", "").split(",")]
                return dict(zip(MEASUREMENTS, values)) if len(values) == len(MEASUREMENTS) else dict(enumerate(values))
            waited += 500
            if self._proc.poll() is not None:
                return None
        return None

    def _send(self, offsets):
        setpoints = list(SETPOINT_DEFAULTS)
        for slot, off in zip(PITCH_OFFSET_SLOTS, offsets):
            setpoints[slot] = float(off)
        # The reply ends in a zero byte: ROSCO's ZeroMQ client parses it as a C string but does not
        # terminate it, so without the zero byte it reads on into whatever its buffer held before.
        self._sock.send(", ".join(f"{s:016.5f}" for s in setpoints).encode() + b"\x00")

    # ---- step -----------------------------------------------------------------------------

    def _delta_m(self):
        m = np.array([self._meas[f"rootMOOP{k}"] for k in (1, 2, 3)])
        return m - m.mean()

    def _observe(self):
        n, m = self.norm, self._meas
        return np.array([*(self._delta_m() / n["Ms"]), (m["WE_Vw"] - n["v_bar"]) / n["sigma_v"],
                         (m["GenSpeed"] - W_REF) / n["sigma_w"], (m["BlPitchCMeas"] - n["beta_bar"]) / n["sigma_beta"]],
                        dtype=np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self.theta_max * action
        for _ in range(self.substeps):
            self._offset += self.alpha * (target - self._offset)   # first-order smoothing of the offset
            self._send(self._offset)
            meas = self._receive()
            if meas is None:
                # OpenFAST aborted (numerical blow-up, tower strike, ...): end the episode with a penalty.
                self._stop()
                return self._observe(), -10.0, True, False, {"failed": True}
            self._meas = meas
            if meas["iStatus"] == -1:   # ROSCO's final call: the simulated time is up
                self._send(np.zeros(3))
                self._stop()
                break

        obs = self._observe()
        reward = -(self.w_bal / 3 * np.abs(self._delta_m()).sum() / self.norm["Ms"]
                   + self.w_pow * abs(self._meas["VS_GenPwr"] - P_RATED) / P_RATED
                   + self.w_smooth / 3 * np.abs(action - self._prev_action).sum())
        self._prev_action = action
        return obs, float(reward), False, self._meas["iStatus"] == -1, dict(self._meas)

    # ---- teardown -------------------------------------------------------------------------

    def _stop(self):
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def close(self):
        self._stop()
        if not self.keep_runs and self.workdir.exists():
            shutil.rmtree(self.workdir)
