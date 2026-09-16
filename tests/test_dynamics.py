"""Runnable self-check, no test framework: python tests/test_dynamics.py"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_consensus.dynamics import closed_gain_report, median_interval, open_gain_report, protocol_rhs, v1
from dynamic_consensus.graphs import attach_new_node
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


def test_closed_gain_report_flags_violation():
    ok = closed_gain_report(lam=90.0, alpha=8.0, n=5, pi=0.5)
    assert ok["thm41_ok"] and ok["thm42_ok"]
    bad = closed_gain_report(lam=10.0, alpha=8.0, n=5, pi=0.5)  # alpha > 2*lam/n = 4
    assert not bad["thm41_ok"] and not bad["thm42_ok"]


def test_open_gain_report_flags_violation():
    ok = open_gain_report(lam=120.0, alpha=6.0, n_max=10, pi=0.2, band_b=0.01, dwell_tau=0.05)
    assert ok["thm41_ok"] and ok["thm43_gain_ok"] and ok["thm43_bound_ok"]
    bad = open_gain_report(lam=120.0, alpha=6.0, n_max=10, pi=0.2, band_b=5.0, dwell_tau=0.05)
    assert not bad["thm43_bound_ok"]


def test_attach_new_node_can_have_more_than_one_neighbor():
    # with p_edge=1 the join must connect to every existing node, not just one
    rng = np.random.default_rng(0)
    g = nx.path_graph(5)  # nodes 0-4
    attach_new_node(g, 5, p_edge=1.0, rng=rng)
    assert set(g.neighbors(5)) == {0, 1, 2, 3, 4}


def test_attach_new_node_stays_connected_even_at_p_zero():
    # p_edge=0 would draw no neighbours; must still force one edge (Ass. 3.5)
    rng = np.random.default_rng(0)
    g = nx.path_graph(5)
    attach_new_node(g, 5, p_edge=0.0, rng=rng)
    assert len(list(g.neighbors(5))) == 1
    assert nx.is_connected(g)


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
