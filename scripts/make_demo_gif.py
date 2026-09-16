"""Render demo.gif: an open network with several join/leave events.
Run from the repo root: python scripts/make_demo_gif.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

from dynamic_consensus.simulate import simulate_open
from dynamic_consensus.viz import fixed_colors, fixed_layout, value_range

BG = "#0e1117"

res = simulate_open(
    n0=6, n_max=18, lam=140.0, alpha=6.0, t_end=1.5, dt=1e-4,
    dwell_tau=0.06, band_b=1.0, join_prob=0.6, seed=7, n_frames=70,
)
frames = res["frames"]
pos = fixed_layout(frames)
colors = fixed_colors(frames)
vmin, vmax = value_range(frames)

fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)


def draw(k: int) -> None:
    ax.clear()
    ax.set_facecolor(BG)
    frame = frames[k]
    g = nx.Graph()
    g.add_nodes_from(frame["nodes"])
    g.add_edges_from(frame["edges"])
    node_colors = [colors[i] for i in g.nodes()]
    sizes = [250 + 900 * (frame["x"][i] - vmin) / (vmax - vmin + 1e-9) for i in g.nodes()]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#4a4f5a", width=1)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=sizes,
                            edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white", font_size=8)
    ax.set_title(f"t = {frame['t']:.3f} s   |   {len(frame['nodes'])} agents",
                 color="white", fontsize=11)
    ax.axis("off")


anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=120)
out = Path(__file__).resolve().parent.parent / "demo.gif"
anim.save(out, writer="pillow", fps=10)
print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")
