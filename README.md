# WindCtrl

Control experiments on the [IEA 22 MW offshore reference wind turbine](https://github.com/IEAWindSystems/IEA-22-280-RWT)
(monopile) using OpenFAST with the ROSCO controller as the baseline.

Project 1 (this repository so far): per-blade individual pitch control (IPC) learned with PPO, with a
Kolmogorov-Arnold network (KAN) in place of the usual MLP so that the learned law can be read
directly. The policy outputs a pitch offset per blade that is added to ROSCO's own pitch command,
so ROSCO keeps doing speed and power regulation. The formulation follows "Individual Pitch Control
of a Large-Scale Wind Turbine With an Interpretable Kolmogorov-Arnold Reinforcement Learning
Policy" (IEA 15 MW there), with the turbine replaced by the IEA 22 MW.

Project 2 (planned): a multi-agent (MAPPO) controller for a wind farm of 22 MW turbines.

Everything runs on the official OpenFAST and ROSCO code; Python talks to the running simulation
through ROSCO's ZeroMQ interface.

![Framework](docs/framework.png)

| Path | What it is |
|---|---|
| `windctrl/env.py` | Gymnasium environment: OpenFAST + ROSCO with per-blade pitch offsets over ZeroMQ |
| `windctrl/kan.py` | KAN layers (linear term + cubic B-splines per edge) |
| `windctrl/train.py` | PPO with a KAN or MLP policy; calibrates the state normalisation on first use |
| `windctrl/evaluate.py` | closed-loop comparison of controllers: fatigue DELs (rainflow), power and speed regulation, pitch activity |
| `windctrl/interpret.py` | reads a trained KAN: edge plots, edge sizes, linear and four-gain laws, 1P complex gain |
| `windctrl/spectra.py` | where in frequency the controllers act: spectra, statistics per frequency band, share of time at the pitch-rate limit |
| `windctrl/report.py` | collects results, draws the paper figures, writes LaTeX tables and `paper/overleaf_<tag>.zip` |
| `scripts/train.sbatch`, `scripts/evaluate.sbatch` | Slurm jobs for training and evaluation |
| `scripts/filter_study.py`, `scripts/filter_study.sbatch` | the four-gain law with slower filters on its pitch offsets |
| `scripts/pipeline.sh` | unattended driver: wait for training, evaluate, interpret, publish |
| `tests/test_basics.py` | checks that need no simulation: rainflow counting, network sizes, the 1P gain, band statistics |
| `paper/` | manuscript skeleton (IEEEtran) with the generated figures and tables of each set of runs |
| `results/` | numbers kept in git: training curves, checkpoints, evaluation tables, interpretation, spectra |
| `scripts/make_cases.py`, `scripts/run_cases.sbatch` | open-loop batches of steady, step and turbulent cases |
| `scripts/zmq_pitch_test.py` | checks that pitch offsets sent over ZeroMQ reach the blades |
| `scripts/make_framework_figure.py` | redraws the framework diagram in `docs/` (SVG, PDF, PNG) |
| `patches/` | small ROSCO patch: also send the wind-speed estimate and collective pitch over ZeroMQ |

## Status (2026-11-18)

Two sets of runs exist, named by a tag. Both train a KAN and an MLP policy for 100 updates with the
same 9000 decisions per update; they differ in how an update is collected.

| Tag | Episodes per update | Training | Evaluation | Interpretation |
|---|---|---|---|---|
| `v2` | 12 episodes of 75 s (+20 s start-up), seeds 101-108 | done, 13 h each | done | done, four-gain law tested in closed loop |
| `v1` | 3 episodes of 300 s, seeds 101/102 | done, 37 h each | queued | after the evaluation |

Open: the `v1` evaluation, and a test of the four-gain law with slower filters on its pitch
offsets (`scripts/filter_study.sbatch`, queued).

## Results of the `v2` runs

Six turbulent wind fields that were not used for training or tuning (13, 16 and 19 m/s, seeds
201/202, 600-s runs, first 60 s dropped), pitch-rate limit 2 deg/s. Mean of the six runs, change
from PI in brackets. PI is ROSCO alone; ROSCO IPC uses the integral gain that gave the lowest blade
DEL on the training winds (2e-10 rad/(N m s)); the four-gain law is the explicit law read from the
KAN policy, run in closed loop without retraining.

| Indicator | PI | ROSCO IPC | MLP-PPO | KAN-PPO | Four-gain law |
|---|---|---|---|---|---|
| Blade-root DEL (MN m) | 28.2 | 26.5 (-6%) | 24.6 (-13%) | 24.3 (-14%) | 24.5 (-13%) |
| Tower-base DEL (MN m) | 48.1 | 40.0 (-17%) | 45.7 (-5%) | 52.0 (+8%) | 48.4 (+1%) |
| Tower-top DEL (MN m) | 15.2 | 15.3 (+1%) | 16.6 (+9%) | 16.2 (+6%) | 16.5 (+8%) |
| Power std (MW) | 0.250 | 0.250 | 0.247 | 0.270 | 0.261 |
| Speed RMSE (rpm) | 0.325 | 0.325 | 0.288 | 0.342 | 0.332 |
| Pitch travel (deg/s) | 0.14 | 0.54 | 1.17 | 1.11 | 1.08 |
| 1P blade-root moment, std (MN m) | 6.4 | 4.8 (-25%) | 3.4 (-47%) | 3.3 (-48%) | 3.2 (-49%) |
| Pitch rate at its limit (% of time) | 0 | 0 | 25 | 20 | 18 |

What these numbers support:

- The KAN policy (330 actor parameters) reduces the blade-root DEL as much as the MLP policy (4998),
  and both reduce it more than ROSCO IPC with its tuned gain. The learned controllers halve the
  once-per-revolution (1P) part of the blade-root moment; ROSCO IPC removes a quarter of it.
- The four parameters of the law read from the KAN reproduce its blade-load reduction. In physical
  units the law is close to `d_beta_i = 0.166 deg/(MN m) x dM_i`: each blade pitches toward feather in
  proportion to its own excess load, with a phase of 3 deg at 1P.
- The price is pitch activity (twice the pitch travel of ROSCO IPC, the pitch rate at its limit for
  about a fifth of the time) and a tower-top DEL 6-9 % above PI. The tower-top increase comes from
  load cycles above 3P: with that content filtered out, the tower-top DEL of the KAN policy is 27 %
  below PI.

What they do not support:

- Differences between the KAN policy, the MLP policy and the four-gain law, and any difference in
  tower-base DEL: with six runs of 540 s, the run-to-run scatter is larger (the tower-base DEL of
  single runs differs from PI by -13 to +20 MN m for the KAN policy).
- A blade-load reduction at 13 m/s: there is none, because at that wind speed the slow load common
  to the three blades is larger than the 1P part, and individual pitch control cannot reach it.
- A verdict on conventional IPC: the ROSCO module was tuned through one gain only, and the gain
  chosen on the training winds gave -20 % there but -6 % here. Larger gains fail because of the
  2 deg/s rate limit (gain 1e-9: blade DEL +48 % at 2 deg/s, -11 % at 10 deg/s, same winds).

Sources: `results/eval_v2`, `results/eval_law_v2`, `results/kan_v2/interpret`, `results/spectra_v2`,
`results/ipc_sweep`, `results/ipc_sweep_rate10`, `results/training_time.md`.

## Setup

```bash
git clone <this repo> WindCtrl && cd WindCtrl

# Turbine model (inputs only)
git clone https://github.com/IEAWindSystems/IEA-22-280-RWT.git

# OpenFAST 4.2.1 is the version the model repo tests against
mamba create -n windctrl -c conda-forge python=3.11 openfast==4.2.1 openfast-io==4.2.1 "rosco>=2.10.1" \
    numpy pandas matplotlib
conda activate windctrl
pip install gymnasium
pip install torch --index-url https://download.pytorch.org/whl/cpu   # training is CPU-bound by OpenFAST
```

The conda-forge ROSCO library is built without the ZeroMQ client, so `ZMQ_Mode > 0` aborts with
"The ZeroMQ client has not been properly installed". Build ROSCO from source against the
environment's libzmq; `make_cases.py` picks this build up automatically when it exists:

```bash
git clone --depth 1 --branch v2.10.5 https://github.com/NREL/ROSCO.git tools/ROSCO
git -C tools/ROSCO apply ../../patches/rosco-zmq-extra-measurements.patch
cd tools/ROSCO/rosco/controller && mkdir build && cd build
PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_BUILD_RPATH=$CONDA_PREFIX/lib
make -j8
```

## Training

```bash
sbatch --export=ALL,POLICY=kan,RUN=kan_v1 scripts/train.sbatch
sbatch --export=ALL,POLICY=mlp,RUN=mlp_v1 scripts/train.sbatch     # MLP comparison

# the same 9000 decisions per update from twelve 75-s episodes side by side (the `v2` runs)
sbatch -c 32 --export=ALL,POLICY=kan,RUN=kan_v2,EXTRA="--episodes-per-speed 4 --episode-seconds 75 --burn-in-seconds 20 --seeds 101 102 103 104 105 106 107 108" scripts/train.sbatch
```

Progress is written to `outputs/<run>/progress.csv` and the latest policy to
`outputs/<run>/checkpoint.pt`.

Training can be interrupted and continued. The checkpoint written after every update holds the
networks, the optimiser, the learning-rate schedule and the random state, and `scripts/train.sbatch`
passes `--resume`, so submitting the same command again carries on from the last finished update
(a finished run exits at once; to start over, use a new `RUN` name). A run interrupted after two
updates and resumed gave bit-identical weights to the same run done in one go. With
`EXTRA="--stop-after N"` a job ends after update `N`, which lets a long training be split over
several jobs. Checkpoints written before this was added (the `v1` and `v2` runs) hold the networks
only; resuming one of those starts from a fresh optimiser state.

With the default settings an update is three 300-s episodes (13, 16 and 19 m/s) and took 22 minutes
of wall time on the compute nodes, 37 hours for 100 updates. With twelve 75-s episodes it took
7.8 minutes, 13 hours for 100 updates. Nearly all of that time is simulation
(`results/training_time.md`).

## Unattended pipeline

```bash
nohup setsid scripts/pipeline.sh > logs/pipeline.out 2>&1 < /dev/null &
tail -f logs/pipeline.out
```

Waits for the training jobs, then evaluates, interprets, writes `results/` and the paper project
under `paper/` (including `paper/overleaf_<tag>.zip`), and commits and pushes after each stage.
Every stage is skipped when its output exists, so the script can be restarted at any point. A
second set of runs is handled by a second instance, e.g.
`TAG=v2 KAN_JOB=<id> MLP_JOB=<id> nohup setsid scripts/pipeline.sh > logs/pipeline_v2.out 2>&1 < /dev/null &`.

`python -m windctrl.report --tag <tag>` alone refreshes `results/`, the paper figures and tables of
that tag (`paper/<tag>/`) and the zip from whatever is under `outputs/`. To start the paper on
Overleaf, use New Project > Upload Project with `paper/overleaf_<tag>.zip`; it is an IEEEtran
journal manuscript skeleton in which missing results and unwritten text are printed in red.

## Evaluation and interpretation

```bash
# tune the ROSCO IPC integral gain (rad/(N m s)) on the training winds
sbatch --export=ALL,ARGS="--name ipc_sweep --winds train --controllers pi rosco_ipc:1e-10 rosco_ipc:2e-10 rosco_ipc:3e-10 rosco_ipc:5e-10 rosco_ipc:7e-10 rosco_ipc:1e-9" scripts/evaluate.sbatch

# same winds for every controller; evaluation seeds 201/202, 600-s runs, first 60 s dropped
sbatch -c 56 --export=ALL,ARGS="--name eval_v2 --controllers pi rosco_ipc:2e-10 kan:kan_v2 mlp:mlp_v2" scripts/evaluate.sbatch

# read the trained KAN, then run its four-gain reduction in closed loop
python -m windctrl.interpret --run kan_v2 --states outputs/eval_v2/states_c2.npz
sbatch -c 16 --export=ALL,ARGS="--name eval_law_v2 --controllers law:kan_v2" scripts/evaluate.sbatch

# spectra, statistics per frequency band and pitch-rate saturation, from the kept OpenFAST outputs
python -m windctrl.spectra --tag v2

# the four-gain law with slower filters on its pitch offsets
sbatch scripts/filter_study.sbatch

# checks that need no simulation
python tests/test_basics.py
```

`outputs/<name>/summary.md` holds the comparison table and `results.csv` the per-run values. The
evaluation keeps its case directories under `runs/<name>_c<i>`; `windctrl.spectra` reads them.

`results/training_time.md` (and `.csv`) records how long each training run took: minutes per update,
total hours, simulations run at once and CPUs allocated. It is refreshed by `python -m windctrl.report`;
the Slurm job of each run is listed in `config/training_jobs.json`.
`states_c<i>.npz` holds the states visited by controller number `i` of the list.

Settings taken from the paper: 6-input state, one offset per blade up to 4 deg with a 0.1 s
smoothing filter, the three-term reward, network sizes (KAN actor 330 parameters, KAN critic 385;
MLP 4998 and 4673), PPO hyper-parameters, class B normal turbulence, seeds 101/102, irregular
waves (Hs 4.52 m, Tp 9.45 s).

Settings that differ from the paper:

- Pitch rate limit 2 deg/s, the 22 MW model's own value (the paper used 10 deg/s; `--pitch-rate-deg 10`).
- OpenFAST runs standalone with ZeroMQ instead of as a Simulink S-function, at a 0.002 s step.
- The normalisation constants are measured for the 22 MW turbine from ROSCO-only runs on the
  training winds; `Ms` is taken as the standard deviation of the blade-load differences, which is
  an assumption, since the paper does not define it.

## Open-loop cases

```bash
python scripts/make_cases.py --type steady --wind 4:25:1 --tmax 600
python scripts/make_cases.py --type step   --wind 6:24:2 --step-size 1 --tmax 400
python scripts/make_cases.py --type turb   --wind 4:24:2 --seeds 6 --tmax 720 --tstart 120
```

Each command writes `runs/<sweep>/` and prints the matching `sbatch` line for
`scripts/run_cases.sbatch`. The Slurm account and partition in that script are specific to the
A*CRC Orion cluster; change them for other systems.

OpenFAST is essentially single-threaded, so throughput comes from running many cases at once,
one per physical core.

### Model settings that differ from the upstream monopile model

| Setting | Upstream | Here (default) | Why |
|---|---|---|---|
| Blades | BeamDyn, `DT = 0.0002` | ElastoDyn, `DT = 0.002` | About 20x faster (0.29x vs 0.015x real time at 18 m/s). Larger steps are not usable: 0.005 aborts and 0.01 diverges, and with `NumCrctn = 1` a 0.01 run reports "terminated normally" while the output is garbage, so check the time series, not just the exit status. |
| Initial rotor speed and pitch | 3 rpm, 0 deg | from the upstream steady-state table | Starting an above-rated case at 0 deg pitch ends in a tower strike. |

Pass `--beamdyn` to keep the upstream blade model when load fidelity matters more than speed.

## ZeroMQ pitch-command check

```bash
python scripts/zmq_pitch_test.py
```

Runs a 60 s case at 18 m/s while a Python ZeroMQ server sends collective and then per-blade pitch
offsets, and reports whether the blades followed. Per-blade offsets need `Ptch_Cntrl = 1` in
ServoDyn, which the script sets.

The reply that `windctrl/env.py` sends to ROSCO ends in a zero byte (since 2026-11-18). ROSCO's
ZeroMQ client (`zmq_client.c`, also in the official release) parses the reply as a C string but
does not terminate it, so without the zero byte it reads on into whatever its buffer held before.
The runs made before this change (the `v2` results and the `v1` training) used the unterminated
reply. A 15-s episode with fixed commands (150 decisions at 16 m/s) gave bit-identical
observations, rewards and measurements before and after the change; the set-point that stale
bytes reach first, the last of the eight, is not used in this project.
