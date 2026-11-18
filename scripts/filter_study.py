#!/usr/bin/env python
"""Closed-loop test of the four-gain law with a slower filter on its pitch offsets.

    python scripts/filter_study.py --name eval_lawfilter_v2 \
        --controllers pi law:kan_v2:tau=0.1 law:kan_v2:tau=0.3 law:kan_v2:tau=0.6 law:kan_v2:tau=1.0

Same evaluation as windctrl.evaluate (same winds, indicators and output files), with one more
controller spec: `law:<run>:tau=<seconds>` runs the four-gain law of <run> with the first-order
filter on the offsets set to that time constant instead of the 0.1 s used in training.

Why: the spectra (windctrl.spectra) show that the higher tower-top DEL of the learned controllers
comes from their pitch action above 3P, while the blade-load reduction is at 1P and 2P. A slower
filter removes fast pitch action; this test measures what it does to both.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windctrl import evaluate          # noqa: E402

_build = evaluate.build_controller


def build_controller(spec):
    kind, _, rest = spec.partition(":")
    if kind == "law" and ":tau=" in rest:
        run, _, tau = rest.partition(":tau=")
        label, extra, act, norm = _build(f"law:{run}")
        return f"{label}, filter {float(tau):g} s", {**extra, "filter_tau": float(tau)}, act, norm
    return _build(spec)


if __name__ == "__main__":
    evaluate.build_controller = build_controller
    evaluate.main()
