"""Shared, plotting-library-agnostic layout/color helpers for the network explorer
(used by both the Streamlit app and the offline GIF script) so an agent keeps the
same position and color across frames regardless of who else joins or leaves."""
from __future__ import annotations

import networkx as nx

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def fixed_layout(frames: list[dict]) -> dict:
    """One layout for the union of every node/edge that ever appears across frames."""
    g = nx.Graph()
    for fr in frames:
        g.add_nodes_from(fr["nodes"])
        g.add_edges_from(fr["edges"])
    return nx.spring_layout(g, seed=42)


def fixed_colors(frames: list[dict]) -> dict:
    """One color per agent id, stable across frames; value is shown via size instead."""
    ids = sorted({i for fr in frames for i in fr["nodes"]})
    return {i: PALETTE[k % len(PALETTE)] for k, i in enumerate(ids)}


def value_range(frames: list[dict]) -> tuple[float, float]:
    vals = [v for fr in frames for v in fr["x"].values()]
    return min(vals), max(vals)
