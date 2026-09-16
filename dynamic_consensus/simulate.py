from __future__ import annotations

import networkx as nx
import numpy as np

from .dynamics import median_interval, protocol_rhs, v1
from .graphs import random_connected_graph


def simulate_closed(
    n: int = 5,
    lam: float = 90.0,
    alpha: float = 8.0,
    t_end: float = 0.3,
    dt: float = 1e-4,
    p_edge: float = 0.5,
    seed: int = 0,
) -> dict:
    """Fixed graph, Theorem 4.1/4.2 setting. References u_i(t) = a_i*sin(2*pi*f_i*t)."""
    rng = np.random.default_rng(seed)
    g = random_connected_graph(n, p_edge, rng)
    nodes = list(g.nodes())
    adj = nx.to_numpy_array(g, nodelist=nodes)

    x = rng.uniform(0, 2, n)
    a = rng.uniform(0, 2, n)
    f = rng.uniform(0, 0.05, n)
    pi_bound = float(np.max(a * 2 * np.pi * f))

    steps = int(t_end / dt)
    ts = np.empty(steps)
    xs = np.empty((steps, n))
    us = np.empty((steps, n))
    ms = np.empty(steps)
    v1s = np.empty(steps)

    for k in range(steps):
        t = k * dt
        u = a * np.sin(2 * np.pi * f * t)
        m, _, _ = median_interval(u)
        ts[k], xs[k], us[k], ms[k], v1s[k] = t, x, u, m, v1(x)
        x = x + dt * protocol_rhs(x, u, adj, lam, alpha)

    return dict(t=ts, x=xs, u=us, m=ms, v1=v1s, pi=pi_bound, graph=g)


def simulate_open(
    n0: int = 9,
    n_max: int = 15,
    lam: float = 90.0,
    alpha: float = 8.0,
    t_end: float = 1.0,
    dt: float = 2e-4,
    dwell_tau: float = 0.02,
    band_b: float = 2.0,
    join_prob: float = 0.5,
    p_edge: float = 0.6,
    seed: int = 0,
) -> dict:
    """Join/leave network, eq.(6): a joining agent starts at the mean of its neighbours."""
    rng = np.random.default_rng(seed)
    g = random_connected_graph(n0, p_edge, rng)
    next_id = n0

    x = {i: float(rng.uniform(0, 2)) for i in g.nodes()}
    a = {i: float(rng.uniform(0, band_b)) for i in g.nodes()}
    f = {i: float(rng.uniform(0, 0.05)) for i in g.nodes()}
    pi_bound = max(a[i] * 2 * np.pi * f[i] for i in g.nodes())

    steps = int(t_end / dt)
    next_event = dwell_tau
    log: list[tuple[float, str, int]] = []
    hist_t, hist_x, hist_m, hist_v2 = [], [], [], []

    for k in range(steps):
        t = k * dt
        nodes = list(g.nodes())
        adj = nx.to_numpy_array(g, nodelist=nodes)
        xv = np.array([x[i] for i in nodes])
        uv = np.array([a[i] * np.sin(2 * np.pi * f[i] * t) for i in nodes])
        m, _, _ = median_interval(uv)
        c = float(xv.mean())

        hist_t.append(t)
        hist_x.append(dict(zip(nodes, xv)))
        hist_m.append(m)
        hist_v2.append(abs(c - m))

        dxv = protocol_rhs(xv, uv, adj, lam, alpha)
        for i, dxi in zip(nodes, dxv):
            x[i] += dt * dxi

        if t >= next_event:
            next_event += dwell_tau
            if rng.random() < join_prob and len(nodes) < n_max:
                neighbor = rng.choice(nodes)
                g.add_node(next_id)
                g.add_edge(next_id, neighbor)
                neigh = list(g.neighbors(next_id))
                x[next_id] = float(np.mean([x[j] for j in neigh]))
                a[next_id] = float(rng.uniform(0, band_b))
                f[next_id] = float(rng.uniform(0, 0.05))
                pi_bound = max(pi_bound, a[next_id] * 2 * np.pi * f[next_id])
                log.append((t, "join", next_id))
                next_id += 1
            elif len(nodes) > 2:
                leaving = rng.choice(nodes)
                gc = g.copy()
                gc.remove_node(leaving)
                if nx.is_connected(gc):
                    g = gc
                    del x[leaving], a[leaving], f[leaving]
                    log.append((t, "leave", leaving))

    return dict(
        t=hist_t, x=hist_x, m=hist_m, v2=hist_v2, log=log,
        pi=pi_bound, final_graph=g,
    )
