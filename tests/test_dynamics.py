"""Runnable self-check, no test framework: python tests/test_dynamics.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_consensus.dynamics import median_interval, protocol_rhs, v1
from dynamic_consensus.simulate import simulate_closed, simulate_open


def test_median_odd():
    m, lo, hi = median_interval(np.array([3.0, 1.0, 2.0]))
    assert m == lo == hi == 2.0


def test_median_even():
    m, lo, hi = median_interval(np.array([1.0, 2.0, 3.0, 4.0]))
    assert lo == 2.0 and hi == 3.0 and m == 2.5


def test_consensus_drift_zero_at_median():
    # at consensus, sum(dx) = -alpha * sum(sign(x-u)); this vanishes exactly at the median
    # (individual dx need not be zero: the fidelity term still pulls each agent).
    x = np.array([2.0, 2.0, 2.0])
    u = np.array([1.0, 2.0, 3.0])
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    dx = protocol_rhs(x, u, adj, lam=10.0, alpha=1.0)
    assert np.isclose(dx.sum(), 0.0)


def test_v1_zero_at_consensus():
    assert v1(np.array([5.0, 5.0, 5.0])) == 0.0


def test_closed_network_reaches_consensus():
    res = simulate_closed(n=5, lam=90.0, alpha=8.0, t_end=0.3, dt=1e-4, seed=1)
    tail = res["x"][-200:]
    spread = tail.max(axis=1) - tail.min(axis=1)
    assert spread.max() < 0.05, f"agents did not agree, spread={spread.max()}"


def test_open_network_runs_and_bounds_error():
    res = simulate_open(n0=6, n_max=10, lam=120.0, alpha=6.0, t_end=0.5, dt=2e-4,
                         dwell_tau=0.05, band_b=2.0, join_prob=0.5, seed=2)
    assert len(res["t"]) > 0
    assert max(res["v2"][-100:]) < 10.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
