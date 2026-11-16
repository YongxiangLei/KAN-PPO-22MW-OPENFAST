#!/usr/bin/env python
"""Check that an external Python process can command blade pitch through ROSCO's ZeroMQ interface.

Runs one short OpenFAST case (steady above-rated wind) with ZMQ_Mode = 1 while a
ZeroMQ server in this process sends pitch offsets:

    t < 20 s        no offset                      (ROSCO alone)
    20 <= t < 35 s  +2 deg on all three blades     (collective offset)
    t >= 40 s       +1 / 0 / -1 deg on blades 1-3  (individual offsets)

It then reads the OpenFAST output and reports whether the blades followed.

    conda activate windctrl
    python scripts/zmq_pitch_test.py            # ~1 min, fine on a login node
"""
import multiprocessing as mp
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP, WIND, TMAX, PORT, PERIOD = "zmq_test", 18.0, 60.0, 15555, 0.1
DEG = np.pi / 180


def set_discon(path, key, value):
    """DISCON.IN lines look like `value   ! key - description`."""
    text = path.read_text()
    new, n = re.subn(rf"^.*?(\s*!\s*{re.escape(key)}\s)", lambda m: f"{value}{m.group(1)}", text, flags=re.M)
    if n != 1:
        raise ValueError(f"expected one '{key}' line in {path.name}, found {n}")
    path.write_text(new)


class Policy:
    """Stand-in for the RL agent: maps (time, measurements) -> setpoints."""

    def update_setpoints(self, id, current_time, measurements):
        if 20.0 <= current_time < 35.0:
            off = [2 * DEG, 2 * DEG, 2 * DEG]
        elif current_time >= 40.0:
            off = [1 * DEG, 0.0, -1 * DEG]
        else:
            off = [0.0, 0.0, 0.0]
        return {f"ZMQ_PitOffset({i + 1})": o for i, o in enumerate(off)}


def serve(logfile):
    from rosco.toolbox.control_interface import wfc_zmq_server
    server = wfc_zmq_server(f"tcp://*:{PORT}", timeout=120.0, verbose=False, logfile=str(logfile))
    server.wfc_controller = Policy()
    server.runserver()


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/make_cases.py"), "--type", "steady", "--sweep", SWEEP,
                    "--wind", str(WIND), "--tmax", str(TMAX)], check=True, stdout=subprocess.DEVNULL)
    case = Path((ROOT / "runs" / SWEEP / "cases.txt").read_text().split()[0]).parent
    name = case.name

    discon = case / "IEA-22-280-RWT_DISCON.IN"
    set_discon(discon, "ZMQ_Mode", 1)
    set_discon(discon, "ZMQ_CommAddress", f'"tcp://localhost:{PORT}"')
    set_discon(discon, "ZMQ_UpdatePeriod", PERIOD)
    # ServoDyn must accept per-blade commands, otherwise only the collective value is used.
    servo = case / "IEA-22-280-RWT_ServoDyn.dat"
    servo.write_text(re.sub(r"^\S+(\s+Ptch_Cntrl\s)", r"1\1", servo.read_text(), flags=re.M))

    server = mp.Process(target=serve, args=(case / "zmq_server.log",))
    server.start()
    with open(case / f"{name}.log", "w") as log:
        rc = subprocess.run(["openfast", f"{name}.fst"], cwd=case, stdout=log, stderr=subprocess.STDOUT).returncode
    server.join(timeout=30)
    if server.is_alive():
        server.terminate()
    if rc != 0:
        print(f"OpenFAST failed (exit {rc}); last lines of {case / (name + '.log')}:")
        print("".join((case / f"{name}.log").read_text().splitlines(keepends=True)[-25:]))
        sys.exit(1)

    from openfast_io.FAST_output_reader import FASTOutputFile
    df = FASTOutputFile(str(case / f"{name}.outb")).toDataFrame()
    t = df["Time_[s]"].to_numpy()
    p = [df[f"BldPitch{i}_[deg]"].to_numpy() for i in (1, 2, 3)]

    def mean(x, lo, hi):
        return float(x[(t >= lo) & (t < hi)].mean())

    print(f"case: {case}")
    print(f"{'window (s)':<14}{'pitch1':>9}{'pitch2':>9}{'pitch3':>9}{'p1-p2':>9}{'p3-p2':>9}   (deg)")
    for lo, hi in [(15, 20), (22, 24), (30, 35), (37, 40), (50, 60)]:
        m = [mean(x, lo, hi) for x in p]
        print(f"{lo:>4}-{hi:<9}{m[0]:9.3f}{m[1]:9.3f}{m[2]:9.3f}{m[0] - m[1]:9.3f}{m[2] - m[1]:9.3f}")

    jump = mean(p[0], 22, 24) - mean(p[0], 18, 20)
    d12, d32 = mean(p[0] - p[1], 50, 60), mean(p[2] - p[1], 50, 60)
    print(f"\ncollective: pitch change shortly after +2 deg command at t=20 s: {jump:+.2f} deg")
    print(f"individual: blade1-blade2 = {d12:+.3f} deg (commanded +1), blade3-blade2 = {d32:+.3f} deg (commanded -1)")
    ok = abs(d12 - 1) < 0.05 and abs(d32 + 1) < 0.05 and jump > 1.0
    print("RESULT:", "PASS - pitch commands sent over ZeroMQ reach the blades" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
