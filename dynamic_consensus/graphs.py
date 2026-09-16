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


def attach_new_node(g: nx.Graph, node_id, p_edge: float, rng: np.random.Generator) -> None:
    """Connect a joining node to a random subset of the existing nodes, each included
    independently with probability p_edge (same density as the initial graph) — an
    Erdos-Renyi-style attachment, not "exactly one neighbour". The paper's Ass. 3.5 only
    requires the graph to stay connected, so at least one edge is forced if the random
    draw would otherwise leave the new node isolated.
    """
    existing = list(g.nodes())
    g.add_node(node_id)
    neighbors = [j for j in existing if rng.random() < p_edge]
    if not neighbors:
        neighbors = [rng.choice(existing)]
    for j in neighbors:
        g.add_edge(node_id, j)
