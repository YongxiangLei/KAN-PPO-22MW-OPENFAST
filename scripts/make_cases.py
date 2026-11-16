#!/usr/bin/env python
"""Generate a sweep of OpenFAST + ROSCO cases for the IEA 22 MW monopile turbine.

Each case is a private copy of the (small) monopile input directory with its own
wind input, so cases are independent and can run in parallel, one per core.

Wind types:
    steady  constant wind speed
    step    constant wind speed, then a step change at --step-time
    turb    IEC turbulent wind from TurbSim (the .bts file is generated inside the
            Slurm job, just before OpenFAST starts, not on the login node)

Layout produced:
    runs/<sweep>/IEA-22-280-RWT -> symlink to the shared blade/airfoil files
    runs/<sweep>/<case>/        -> per-case inputs; run `openfast <case>.fst` here
    runs/<sweep>/cases.txt      -> one .fst path per line (consumed by run_cases.sbatch)

Examples:
    python scripts/make_cases.py --type steady --wind 4:25:1 --tmax 600
    python scripts/make_cases.py --type step   --wind 6:24:2 --step-size 1 --tmax 400
    python scripts/make_cases.py --type turb   --wind 4:24:2 --seeds 6 --tmax 720 --tstart 120
"""
import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "IEA-22-280-RWT" / "OpenFAST"
SHARED = MODEL / "IEA-22-280-RWT"
MONOPILE = MODEL / "IEA-22-280-RWT-Monopile"
FST = "IEA-22-280-RWT-Monopile.fst"
INFLOW = "IEA-22-280-RWT_InflowFile.dat"

HUB_HT = 170.0      # m
ROTOR_D = 284.0     # m
GRID = 320.0        # m, TurbSim grid height/width: rotor plus margin for deflection
PL_EXP = 0.14       # shear exponent, same as the model's steady-wind setting

# ROSCO built from source with the ZeroMQ client (the conda-forge build has it disabled,
# so ZMQ_Mode > 0 aborts). Used by default when present; see tools/ROSCO.
ZMQ_LIBDISCON = ROOT / "tools" / "ROSCO" / "rosco" / "controller" / "build" / "libdiscon.so"

TURBSIM_TEMPLATE = """---------TurbSim v2.00.* Input File------------------------
IEA 22 MW: {turb_type} class {turb_class}, URef = {uref} m/s, seed {seed}
---------Runtime Options-----------------------------------
False         Echo            - Echo input data to <RootName>.ech (flag)
{seed:<13d} RandSeed1       - First random seed  (-2147483648 to 2147483647)
"RanLux"      RandSeed2       - Second random seed (-2147483648 to 2147483647) for intrinsic pRNG, or an alternative pRNG: "RanLux" or "RNSNLW"
False         WrBHHTP         - Output hub-height turbulence parameters in binary form?  (Generates RootName.bin)
False         WrFHHTP         - Output hub-height turbulence parameters in formatted form?  (Generates RootName.dat)
False         WrADHH          - Output hub-height time-series data in AeroDyn form?  (Generates RootName.hh)
True          WrADFF          - Output full-field time-series data in TurbSim/AeroDyn form? (Generates RootName.bts)
False         WrBLFF          - Output full-field time-series data in BLADED/AeroDyn form?  (Generates RootName.wnd)
False         WrADTWR         - Output tower time-series data? (Generates RootName.twr)
False         WrHAWCFF        - Output full-field time-series data in HAWC form?  (Generates RootName-u.bin, RootName-v.bin, RootName-w.bin, RootName.hawc)
False         WrFMTFF         - Output full-field time-series data in formatted (readable) form?  (Generates RootName.u, RootName.v, RootName.w)
False         WrACT           - Output coherent turbulence time steps in AeroDyn form? (Generates RootName.cts)
0             ScaleIEC        - Scale IEC turbulence models to exact target standard deviation? [0=no additional scaling; 1=use hub scale uniformly; 2=use individual scales]

--------Turbine/Model Specifications-----------------------
{ngrid:<13d} NumGrid_Z       - Vertical grid-point matrix dimension
{ngrid:<13d} NumGrid_Y       - Horizontal grid-point matrix dimension
{dt:<13g} TimeStep        - Time step [seconds]
{tmax:<13g} AnalysisTime    - Length of analysis time series [seconds] (program will add time if necessary: AnalysisTime = MAX(AnalysisTime, UsableTime+GridWidth/MeanHHWS) )
"ALL"         UsableTime      - Usable length of output time series [seconds] (program will add GridWidth/MeanHHWS seconds unless UsableTime is "ALL")
{hub:<13g} HubHt           - Hub height [m] (should be > 0.5*GridHeight)
{grid:<13g} GridHeight      - Grid height [m]
{grid:<13g} GridWidth       - Grid width [m] (should be >= 2*(RotorRadius+ShaftLength))
0             VFlowAng        - Vertical mean flow (uptilt) angle [degrees]
0             HFlowAng        - Horizontal mean flow (skew) angle [degrees]

--------Meteorological Boundary Conditions-------------------
"IECKAI"      TurbModel       - Turbulence model ("IECKAI","IECVKM","GP_LLJ","NWTCUP","SMOOTH","WF_UPW","WF_07D","WF_14D","TIDAL","API","USRINP","TIMESR", or "NONE")
"unused"      UserFile        - Name of the file that contains inputs for user-defined spectra or time series inputs (used only for "USRINP" and "TIMESR" models)
"1-ED3"       IECstandard     - Number of IEC 61400-x standard (x=1,2, or 3 with optional 61400-1 edition number (i.e. "1-Ed2") )
"{turb_class}"           IECturbc        - IEC turbulence characteristic ("A", "B", "C" or the turbulence intensity in percent) ("KHTEST" option with NWTCUP model, not used for other models)
"{turb_type}"         IEC_WindType    - IEC turbulence type ("NTM"=normal, "xETM"=extreme turbulence, "xEWM1"=extreme 1-year wind, "xEWM50"=extreme 50-year wind, where x=wind turbine class 1, 2, or 3)
"default"     ETMc            - IEC Extreme Turbulence Model "c" parameter [m/s]
"PL"          WindProfileType - Velocity profile type ("LOG";"PL"=power law;"JET";"H2L"=Log law for TIDAL model;"API";"USR";"TS";"IEC"=PL on rotor disk, LOG elsewhere; or "default")
"unused"      ProfileFile     - Name of the file that contains input profiles for WindProfileType="USR" and/or TurbModel="USRVKM" [-]
{hub:<13g} RefHt           - Height of the reference velocity (URef) [m]
{uref:<13g} URef            - Mean (total) velocity at the reference height [m/s] (or "default" for JET velocity profile) [must be 1-hr mean for API model; otherwise is the mean over AnalysisTime seconds]
"default"     ZJetMax         - Jet height [m] (used only for JET velocity profile, valid 70-490 m)
{plexp:<13g} PLExp           - Power law exponent [-] (or "default")
"default"     Z0              - Surface roughness length [m] (or "default")

--------Non-IEC Meteorological Boundary Conditions------------
"default"     Latitude        - Site latitude [degrees] (or "default")
0.05          RICH_NO         - Gradient Richardson number [-]
"default"     UStar           - Friction or shear velocity [m/s] (or "default")
"default"     ZI              - Mixing layer depth [m] (or "default")
"default"     PC_UW           - Hub mean u'w' Reynolds stress [m^2/s^2] (or "default" or "none")
"default"     PC_UV           - Hub mean u'v' Reynolds stress [m^2/s^2] (or "default" or "none")
"default"     PC_VW           - Hub mean v'w' Reynolds stress [m^2/s^2] (or "default" or "none")

--------Spatial Coherence Parameters----------------------------
"default"     SCMod1          - u-component coherence model ("GENERAL","IEC","API","NONE", or "default")
"default"     SCMod2          - v-component coherence model ("GENERAL","IEC","NONE", or "default")
"default"     SCMod3          - w-component coherence model ("GENERAL","IEC","NONE", or "default")
"default"     InCDec1         - u-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")
"default"     InCDec2         - v-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")
"default"     InCDec3         - w-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")
"default"     CohExp          - Coherence exponent for general model [-] (or "default")

--------Coherent Turbulence Scaling Parameters-------------------
"unused"      CTEventPath     - Name of the path where event data files are located
"random"      CTEventFile     - Type of event files ("LES", "DNS", or "RANDOM")
true          Randomize       - Randomize the disturbance scale and locations? (true/false)
1             DistScl         - Disturbance scale [-] (ratio of event dataset height to rotor disk). (Ignored when Randomize = true.)
0.5           CTLy            - Fractional location of tower centerline from right [-] (looking downwind) to left side of the dataset. (Ignored when Randomize = true.)
0.5           CTLz            - Fractional location of hub height from the bottom of the dataset. [-] (Ignored when Randomize = true.)
30            CTStartTime     - Minimum start time for coherent structures in RootName.cts [seconds]

====================================================
! NOTE: Do not add or remove any lines in this file!
====================================================
"""


# Steady operating points (wind speed m/s, rotor speed rpm, blade pitch deg), from
# IEA-22-280-RWT/outputs/01_steady_states/OpenFAST. Cases start from these: starting an
# above-rated case from the model's default 0 deg pitch overloads the rotor and ends in a tower strike.
STEADY_STATES = [
    (3, 1.873, 0.570), (4, 2.498, 0.314), (5, 3.134, -0.023), (6, 3.775, -0.378), (7, 4.418, -0.700),
    (8, 5.063, -1.059), (9, 5.676, -0.866), (10, 6.157, 0.731), (11, 7.017, 2.104), (12, 7.061, 5.384),
    (13, 7.061, 7.508), (14, 7.061, 9.226), (15, 7.061, 10.724), (16, 7.061, 12.087), (17, 7.061, 13.358),
    (18, 7.061, 14.560), (19, 7.061, 15.710), (20, 7.061, 16.817), (21, 7.061, 17.891), (22, 7.061, 18.935),
    (23, 7.061, 19.953), (24, 7.061, 20.948),
]


def initial_conditions(ws):
    """Rotor speed (rpm) and blade pitch (deg) to start from at wind speed `ws`."""
    w, rpm, pitch = zip(*STEADY_STATES)
    if ws > w[-1]:  # continue the last pitch slope above the table (np.interp would hold it flat)
        return rpm[-1], pitch[-1] + (ws - w[-1]) * (pitch[-1] - pitch[-2]) / (w[-1] - w[-2])
    return float(np.interp(ws, w, rpm)), float(np.interp(ws, w, pitch))


def set_value(text, key, value):
    """Replace the value on the line whose keyword (2nd token) is `key`."""
    pattern = re.compile(rf"^(\s*)(\"[^\"]*\"|\S+)(\s+{re.escape(key)}\s)", re.M)
    new, n = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", text)
    if n != 1:
        raise ValueError(f"expected exactly one '{key}' line, found {n}")
    return new


def edit(path, changes):
    text = path.read_text()
    for key, value in changes.items():
        text = set_value(text, key, value)
    path.write_text(text)


def wind_speeds(spec):
    """'4:25:1' -> 4,5,...,25   or   '6,8,11.5' -> those values."""
    if ":" in spec:
        lo, hi, step = (float(x) for x in spec.split(":"))
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + i * step, 6) for i in range(n)]
    return [float(x) for x in spec.split(",")]


def tag(value):
    return f"{value:05.2f}".replace(".", "p")


def write_step_wind(path, ws, step_size, step_time, tmax, ramp=0.1):
    """Uniform wind file: `ws` until step_time, then ws+step_size (over `ramp` s)."""
    rows = [(0.0, ws), (step_time, ws), (step_time + ramp, ws + step_size), (tmax + 10.0, ws + step_size)]
    lines = [
        "! Uniform wind file: step change in hub-height wind speed",
        "! Time  WindSpeed  WindDir  VertSpeed  HorizShear  PwrLawVertShear  LinVertShear  GustSpeed",
        "! (s)   (m/s)      (deg)    (m/s)      (-)         (-)              (-)           (m/s)",
    ]
    lines += [f"{t:10.3f} {v:10.4f} 0.0 0.0 0.0 {PL_EXP} 0.0 0.0" for t, v in rows]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=["steady", "step", "turb"], default="steady", help="wind type")
    ap.add_argument("--sweep", help="name of the sweep directory under runs/ (default: the wind type)")
    ap.add_argument("--wind", default="4:25:1", help="(mean) wind speeds: lo:hi:step or comma list (m/s)")
    ap.add_argument("--tmax", type=float, default=600.0, help="simulated time per case (s)")
    ap.add_argument("--tstart", type=float, default=0.0, help="time at which output starts (s)")
    ap.add_argument("--beamdyn", action="store_true",
                    help="keep the BeamDyn blades (DT=0.0002 s, ~20x slower). Default: ElastoDyn blades.")
    # 0.002 s is the largest step found stable for the monopile model at 18 m/s: 0.005 aborts,
    # and 0.01 diverges (with NumCrctn=1 it even "terminates normally" with garbage output).
    ap.add_argument("--dt", type=float, default=0.002, help="time step when using ElastoDyn blades (s)")
    ap.add_argument("--libdiscon", help="ROSCO controller library (default: the ZeroMQ-enabled build "
                    "under tools/ROSCO if present, otherwise the conda one)")
    step = ap.add_argument_group("step wind")
    step.add_argument("--step-size", type=float, default=1.0, help="size of the step, may be negative (m/s)")
    step.add_argument("--step-time", type=float, help="time of the step (s) (default: tmax/2)")
    turb = ap.add_argument_group("turbulent wind")
    turb.add_argument("--seeds", type=int, default=1, help="number of random seeds per wind speed")
    turb.add_argument("--turb-class", default="B", help='IEC turbulence class A/B/C, or TI in percent')
    turb.add_argument("--turb-type", default="NTM", help='IEC wind type, e.g. NTM or 1ETM')
    turb.add_argument("--ngrid", type=int, default=25, help="TurbSim grid points per side")
    turb.add_argument("--turb-dt", type=float, default=0.05, help="TurbSim time step (s)")
    args = ap.parse_args()

    libdiscon = Path(args.libdiscon).resolve() if args.libdiscon else next(
        (p for p in (ZMQ_LIBDISCON, Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / "libdiscon.so") if p.is_file()),
        None)
    if libdiscon is None or not libdiscon.is_file():
        raise SystemExit("libdiscon.so not found - activate the 'windctrl' conda env first")

    sweep = ROOT / "runs" / (args.sweep or args.type)
    sweep.mkdir(parents=True, exist_ok=True)
    link = sweep / SHARED.name
    if not link.exists():
        link.symlink_to(SHARED)

    step_time = args.step_time if args.step_time is not None else args.tmax / 2
    seeds = range(1, args.seeds + 1) if args.type == "turb" else [None]

    fsts = []
    for ws in wind_speeds(args.wind):
        for seed in seeds:
            name = f"ws{tag(ws)}" + (f"_s{seed}" if seed else "")
            case = sweep / name
            if case.exists():
                shutil.rmtree(case)
            shutil.copytree(MONOPILE, case)
            shutil.copy(SHARED / INFLOW, case / INFLOW)

            if args.type == "steady":
                inflow = {"WindType": 1, "HWindSpeed": ws}
            elif args.type == "step":
                write_step_wind(case / "wind_step.dat", ws, args.step_size, step_time, args.tmax)
                inflow = {"WindType": 2, "Filename_Uni": '"wind_step.dat"', "RefHt_Uni": HUB_HT, "RefLength": ROTOR_D}
            else:
                # Distinct, reproducible seed per (wind speed, seed index).
                rand = 100000 * seed + int(round(ws * 100))
                (case / f"{name}_wind.inp").write_text(TURBSIM_TEMPLATE.format(
                    seed=rand, ngrid=args.ngrid, dt=args.turb_dt, tmax=args.tmax, hub=HUB_HT, grid=GRID,
                    uref=ws, plexp=PL_EXP, turb_class=args.turb_class, turb_type=args.turb_type))
                inflow = {"WindType": 3, "FileName_BTS": f'"{name}_wind.bts"'}
            edit(case / INFLOW, inflow)

            fst_changes = {"TMax": args.tmax, "TStart": args.tstart, "InflowFile": f'"{INFLOW}"'}
            if not args.beamdyn:
                fst_changes.update({"CompElast": 1, "DT": args.dt})
            edit(case / FST, fst_changes)
            edit(case / "IEA-22-280-RWT_ServoDyn.dat", {"DLL_FileName": f'"{libdiscon}"'})
            rpm, pitch = initial_conditions(ws)
            edit(case / "IEA-22-280-RWT_ElastoDyn.dat", {
                "RotSpeed": f"{rpm:.3f}", "BlPitch(1)": f"{pitch:.3f}", "BlPitch(2)": f"{pitch:.3f}", "BlPitch(3)": f"{pitch:.3f}"})

            fst = case / f"{name}.fst"
            (case / FST).rename(fst)
            fsts.append(fst)

    (sweep / "cases.txt").write_text("".join(f"{p}\n" for p in fsts))
    print(f"{len(fsts)} {args.type} cases written to {sweep}")
    print(f"submit with: sbatch -c {min(240, 2 * len(fsts))} --export=ALL,SWEEP={sweep.name} scripts/run_cases.sbatch")


if __name__ == "__main__":
    main()
