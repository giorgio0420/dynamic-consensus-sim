"""Runnable self-check, no test framework: python tests/test_dynamics.py"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamic_consensus.data_source import load_csv_signals, make_reference_from_csv, pick_window, resample_to_grid
from dynamic_consensus.dynamics import closed_gain_report, median_interval, open_gain_report, protocol_rhs, v1
from dynamic_consensus.graphs import attach_new_node
from dynamic_consensus.simulate import simulate_closed, simulate_open

TINY_CSV = (
    "created_at,a,b\n"
    "2021-01-01 00:00:00+00:00,0.0,10.0\n"
    "2021-01-01 00:00:10+00:00,1.0,8.0\n"
    "2021-01-01 00:00:20+00:00,2.0,6.0\n"
).encode()


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


def test_load_csv_signals_sorts_and_parses():
    t, u, labels = load_csv_signals(TINY_CSV)
    assert labels == ["a", "b"]
    assert np.allclose(t, [0.0, 10.0, 20.0])
    assert np.allclose(u[:, 0], [0.0, 1.0, 2.0])
    assert np.allclose(u[:, 1], [10.0, 8.0, 6.0])


def test_load_csv_signals_auto_detects_first_column_as_time():
    # no "created_at" here -- an arbitrary CSV should still work, time_col defaults to first
    csv_bytes = b"ts,x,y\n0,1.0,2.0\n5,1.5,2.5\n10,2.0,3.0\n"
    t, u, labels = load_csv_signals(csv_bytes)
    assert labels == ["x", "y"]
    assert np.allclose(t, [0.0, 5.0, 10.0])


def test_load_csv_signals_rejects_bad_signal_column():
    csv_bytes = b"ts,x\n0,1.0\n5,not_a_number\n"
    try:
        load_csv_signals(csv_bytes)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "signal column" in str(exc)


def test_pick_window_caps_by_step_budget_and_covers_target_samples():
    t_data = np.arange(0, 1000, 10.0)  # 100 rows, 10s apart, 990s span
    # plenty of budget: should cover ~60 native samples = 600s
    assert np.isclose(pick_window(t_data, dt=0.01, max_steps=1_000_000, target_samples=60), 600.0)
    # tight budget: capped by max_steps*dt regardless of target_samples
    assert np.isclose(pick_window(t_data, dt=0.01, max_steps=100, target_samples=60), 1.0)
    # short file: never exceeds the actual span
    short = np.arange(0, 30, 10.0)
    assert np.isclose(pick_window(short, dt=0.01, max_steps=1_000_000, target_samples=60), 20.0)


def test_make_reference_from_csv_interpolates_and_bounds_pi():
    ref = make_reference_from_csv(TINY_CSV)
    assert ref["n"] == 2
    assert np.allclose(ref["u_func"](5.0), [0.5, 9.0])  # midpoint, linear interp
    assert np.isclose(ref["pi_bound"], 0.2)  # |10-8|/10 = |8-6|/10 = 0.2 uV/s


def test_resample_to_grid_matches_u_func():
    ref = make_reference_from_csv(TINY_CSV)
    t_grid = np.array([0.0, 2.5, 7.5, 20.0])
    grid = resample_to_grid(ref["t_data"], ref["u_data"], t_grid)
    expected = np.array([ref["u_func"](t) for t in t_grid])
    assert np.allclose(grid, expected)


def test_simulate_closed_with_precomputed_u_array_tracks_reference():
    ref = make_reference_from_csv(TINY_CSV)
    dt, t_end = 1e-3, 15.0
    steps = int(t_end / dt)
    t_grid = np.arange(steps) * dt
    u_array = resample_to_grid(ref["t_data"], ref["u_data"], t_grid)
    res = simulate_closed(n=2, lam=50.0, alpha=5.0, t_end=t_end, dt=dt, seed=0,
                           u_array=u_array, pi_bound=ref["pi_bound"], x0=u_array[0])
    # Thm 4.2 puts c(t) in the median *interval*, not on the point m(u) — with n=2 the
    # interval spans both signals, so any point inside it (not just the midpoint) is valid.
    assert res["lo"][-1] - 0.1 <= res["c"][-1] <= res["hi"][-1] + 0.1


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
