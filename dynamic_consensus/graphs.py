from __future__ import annotations

import networkx as nx
import numpy as np


def random_connected_graph(n: int, p_edge: float = 0.5, rng: np.random.Generator | None = None) -> nx.Graph:
    rng = rng or np.random.default_rng()
    while True:
        seed = int(rng.integers(1_000_000_000))
        g = nx.gnp_random_graph(n, p_edge, seed=seed)
        if nx.is_connected(g):
            return g
