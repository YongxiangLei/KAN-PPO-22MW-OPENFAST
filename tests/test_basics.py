"""Checks of the parts that do not need a simulation: fatigue counting, networks, the 1P gain, band statistics.

    python tests/test_basics.py          # or: python -m pytest tests

Everything here runs in a few seconds without OpenFAST.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windctrl.evaluate import del_, rainflow          # noqa: E402
from windctrl.kan import KANLayer                     # noqa: E402
from windctrl.train import KANPolicy, MLPPolicy, count   # noqa: E402


def test_rainflow_reproduces_the_astm_example():
    # ASTM E1049-85, rainflow counting example (Fig. 6): cycle counts per range.
    ranges, counts = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    found = {}
    for r, c in zip(ranges, counts):
        found[float(r)] = found.get(float(r), 0.0) + float(c)
    assert found == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}, found


def test_del_of_a_sine_is_its_range():
    # 1 Hz sine of amplitude 1 for 100 s: 100 cycles of range 2 per 100 s at f_eq = 1 Hz -> DEL = 2.
    t = np.arange(0, 100, 0.01)
    for m in (4, 10):
        assert abs(del_(np.sin(2 * np.pi * t), m, 100.0) - 2.0) < 0.02


def test_network_sizes():
    kan, mlp = KANPolicy(), MLPPolicy()
    assert count(kan.actor_parameters()) == 330 and count(kan.critic.parameters()) == 385
    assert count(mlp.actor_parameters()) == 4998 and count(mlp.critic.parameters()) == 4673


def test_policy_outputs_are_valid_actions():
    s = torch.randn(64, 6) * 2
    for policy in (KANPolicy(), MLPPolicy()):
        d = policy.dist(s)
        assert d.mean.shape == (64, 3) and bool((d.mean.abs() < 1).all()) and bool((d.stddev > 0).all())
        assert policy.value(s).shape == (64,)


def test_kan_edges_sum_to_the_layer_output_and_basis_sums_to_one():
    layer = KANLayer(6, 3, sigma_w=0.5, sigma_c=0.5)
    x = torch.linspace(-2.9, 2.9, 50).unsqueeze(1).repeat(1, 6)
    assert torch.allclose(layer.basis(x).sum(-1), torch.ones(50, 6), atol=1e-5)          # partition of unity
    assert torch.allclose(layer.edges(x).sum(-1) + layer.b, layer(x), atol=1e-6)         # additive in the inputs
    # Beyond the spline range only the linear term continues.
    far = layer.edges(torch.tensor([[5.0] * 6])) - layer.edges(torch.tensor([[4.0] * 6]))
    assert torch.allclose(far[0], layer.w, atol=1e-5)


def test_one_per_rev_gain_of_the_four_gain_law():
    # Blade i+1 leads blade i by 120 deg. For a 1P load A cos(psi_i) the load part of the law,
    # g_s dM_i + g_n dM_(i+1) + g_p dM_(i-1), equals |G1| A cos(psi_i + arg G1).
    g_s, g_n, g_p, amp = 0.25, -0.11, -0.14, 1.7
    g1 = g_s + g_n * np.exp(2j * np.pi / 3) + g_p * np.exp(-2j * np.pi / 3)
    psi = np.linspace(0, 2 * np.pi, 200)
    blades = [amp * np.cos(psi + k * 2 * np.pi / 3) for k in range(3)]            # blade 1, 2, 3
    for i in range(3):
        law = g_s * blades[i] + g_n * blades[(i + 1) % 3] + g_p * blades[(i - 1) % 3]
        expected = abs(g1) * amp * np.cos(psi + i * 2 * np.pi / 3 + np.angle(g1))
        assert np.allclose(law, expected, atol=1e-12)
    assert abs(sum(blades)).max() < 1e-12                                          # the three differences sum to zero


def test_band_statistics_recover_a_sine():
    from scipy.signal import welch
    from windctrl.spectra import P1, band_std
    fs, t = 50.0, np.arange(0, 540, 1 / 50.0)
    x = 3.0 * np.sin(2 * np.pi * P1 * t) + 0.5 * np.sin(2 * np.pi * 3 * P1 * t)
    f, p = welch(x, fs, nperseg=int(160 * fs))
    assert abs(band_std(f, p, 0.8 * P1, 1.2 * P1) - 3.0 / np.sqrt(2)) < 0.05
    assert abs(band_std(f, p, 2.8 * P1, 3.2 * P1) - 0.5 / np.sqrt(2)) < 0.02
    assert band_std(f, p, 1.8 * P1, 2.2 * P1) < 0.02


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
