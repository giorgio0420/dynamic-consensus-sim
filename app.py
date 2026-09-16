import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from dynamic_consensus.dynamics import closed_gain_report, open_gain_report
from dynamic_consensus.simulate import simulate_closed, simulate_open
from dynamic_consensus.viz import fixed_colors, fixed_layout, value_range

st.set_page_config(page_title="Dynamic Median Consensus", layout="wide")
st.title("Dynamic Consensus on the Median Value — simulator")
st.caption(
    "Own implementation of the protocol in Sanai Dashti, Seatzu, Franceschelli, "
    "\"Dynamic Consensus on the Median Value in Open Multi-Agent Systems\", IEEE TAC. "
    "No text or figures from the paper are reproduced here — only the math is implemented."
)

DT_OPTIONS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]


def gain_alert(ok: bool, label: str, detail: str) -> None:
    if ok:
        st.success(f"✅ {label}: {detail}")
    else:
        st.error(f"⚠️ DANGER — {label} violated: {detail}. This is a sufficient condition from the paper — "
                 f"violating it voids the guarantee (no bound applies), it does not by itself prove divergence. "
                 f"Watch the plots below: with the network's actual topology the trajectory may still settle, "
                 f"chatter without fully settling, or truly diverge.")


def agents_chart(t, x, n_or_ids, m) -> go.Figure:
    fig = go.Figure()
    if isinstance(x, list):  # open network: list of {node_id: value} dicts
        ids = sorted({i for d in x for i in d})
        for i in ids:
            ys = [d.get(i, None) for d in x]
            fig.add_trace(go.Scatter(x=t, y=ys, name=f"x_{i}", line=dict(width=1), connectgaps=False))
    else:  # closed network: (steps, n) array
        for i in range(n_or_ids):
            fig.add_trace(go.Scatter(x=t, y=x[:, i], name=f"x_{i}", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=t, y=m, name="m(u)", line=dict(color="black", width=3, dash="dash")))
    fig.update_layout(xaxis_title="t [s]", yaxis_title="state", height=450)
    return fig


def consensus_vs_median_chart(t, c, m, lo, hi) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(t) + list(t)[::-1], y=list(hi) + list(lo)[::-1],
                              fill="toself", fillcolor="rgba(255,165,0,0.15)",
                              line=dict(width=0), name="median interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=m, name="m(u) target median", line=dict(color="orange", width=2)))
    fig.add_trace(go.Scatter(x=t, y=c, name="mean(x) (≈ consensus value)", line=dict(color="royalblue", width=2)))
    fig.update_layout(xaxis_title="t [s]", yaxis_title="value", height=350)
    return fig


def network_frame_figure(frame: dict, pos: dict, colors: dict, vrange: tuple[float, float]) -> go.Figure:
    g = nx.Graph()
    g.add_nodes_from(frame["nodes"])
    g.add_edges_from(frame["edges"])

    edge_x, edge_y = [], []
    for u, v in g.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="rgba(150,150,150,0.6)", width=1), hoverinfo="skip")

    vmin, vmax = vrange
    node_x = [pos[i][0] for i in g.nodes()]
    node_y = [pos[i][1] for i in g.nodes()]
    node_colors = [colors[i] for i in g.nodes()]
    node_sizes = [16 + 28 * (frame["x"][i] - vmin) / (vmax - vmin + 1e-9) for i in g.nodes()]
    node_text = [f"agent {i}<br>x={frame['x'][i]:.3f}" for i in g.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[str(i) for i in g.nodes()], textposition="top center",
        hovertext=node_text, hoverinfo="text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="black")),
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def network_explorer(frames: list[dict], key: str) -> None:
    st.subheader("Network step explorer")
    st.caption("Color = fixed per agent id (identity). Size = current state x_i (bigger = larger value).")
    idx_key = f"{key}_frame_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    c1, c2, c3 = st.columns([1, 6, 1])
    if c1.button("◀ Prev", key=f"{key}_prev"):
        st.session_state[idx_key] = max(0, st.session_state[idx_key] - 1)
    if c3.button("Next ▶", key=f"{key}_next"):
        st.session_state[idx_key] = min(len(frames) - 1, st.session_state[idx_key] + 1)
    c2.slider("step", 0, len(frames) - 1, key=idx_key)

    pos = fixed_layout(frames)
    colors = fixed_colors(frames)
    vrange = value_range(frames)
    frame = frames[st.session_state[idx_key]]
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(network_frame_figure(frame, pos, colors, vrange), use_container_width=True)
    with right:
        st.metric("t", f"{frame['t']:.4g} s")
        st.metric("mean(x)  (≈ consensus)", f"{frame['c']:.4g}")
        st.metric("m(u)  (target median)", f"{frame['m']:.4g}")
        st.metric("median interval", f"[{frame['lo']:.3g}, {frame['hi']:.3g}]")
        st.metric("error |mean(x) − m(u)|", f"{abs(frame['c'] - frame['m']):.4g}")


scenario = st.sidebar.radio("Scenario", ["Closed network", "Open network"])
lam = st.sidebar.slider("λ (coupling gain)", 1.0, 200.0, 90.0)
alpha = st.sidebar.slider("α (fidelity gain)", 0.1, 50.0, 8.0)
seed = st.sidebar.number_input("seed", 0, 10_000, 0)

if scenario == "Closed network":
    n = st.sidebar.slider("agents", 2, 20, 5)
    t_end = st.sidebar.slider("sim time (s)", 0.01, 2.0, 0.3)
    dt = st.sidebar.select_slider("dt", options=DT_OPTIONS, value=1e-4)

    if st.sidebar.button("Run simulation", type="primary"):
        with st.spinner("integrating..."):
            res = simulate_closed(n=n, lam=lam, alpha=alpha, t_end=t_end, dt=dt, seed=seed)
        st.session_state["res"] = res
        st.session_state["kind"] = "closed"
        st.session_state["closed_frame_idx"] = 0

    if st.session_state.get("kind") == "closed":
        res = st.session_state["res"]
        rep = closed_gain_report(lam, alpha, n, res["pi"])

        gain_alert(rep["thm41_ok"], "Thm 4.1 (consensus)",
                   f"need 0 < α < 2λ/n = {rep['upper']:.3g}, got α={alpha:.3g}, μ2={rep['mu2']:.3g}")
        gain_alert(rep["thm42_ok"], "Thm 4.2 (median tracking)",
                   f"need n·Π={rep['lower']:.3g} < α < 2λ/n={rep['upper']:.3g}, got α={alpha:.3g}, Π={res['pi']:.3g}")

        spread0 = float(res["x"][0].max() - res["x"][0].min())
        spread_end = float(res["x"][-200:].max(axis=1).max() - res["x"][-200:].min(axis=1).min())
        if rep["thm41_ok"] and spread_end > 0.05 * max(spread0, 1e-9):
            st.error(f"⚠️ Simulation shows NO empirical convergence: spread(t_end)≈{spread_end:.3g} "
                     f"vs spread(t=0)={spread0:.3g}.")

        st.plotly_chart(agents_chart(res["t"], res["x"], n, res["m"]), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Thm 4.1 bound T1", f"{(spread0 / rep['mu2']):.4g} s" if rep["mu2"] > 0 else "n/a")
        col2.metric("μ2", f"{rep['mu2']:.3g}")
        col3.metric("Π (ref. speed)", f"{res['pi']:.3g}")

        st.subheader("Consensus value vs. target median")
        st.plotly_chart(consensus_vs_median_chart(res["t"], res["c"], res["m"], res["lo"], res["hi"]),
                         use_container_width=True)

        v1_fig = go.Figure(go.Scatter(x=res["t"], y=res["v1"], name="V1"))
        v1_fig.update_layout(xaxis_title="t [s]", yaxis_title="V1(x)", height=250)
        st.plotly_chart(v1_fig, use_container_width=True)

        network_explorer(res["frames"], key="closed")
    else:
        st.info("Set parameters in the sidebar and press Run simulation.")

else:
    n0 = st.sidebar.slider("initial agents", 2, 15, 9)
    n_max = st.sidebar.slider("n_max", n0, 30, 15)
    dwell = st.sidebar.slider("dwell time Δτ (s)", 0.001, 0.2, 0.02)
    band_b = st.sidebar.slider("reference band B", 0.1, 5.0, 2.0)
    join_prob = st.sidebar.slider("join probability per event", 0.0, 1.0, 0.5)
    t_end = st.sidebar.slider("sim time (s)", 0.05, 3.0, 1.0)
    dt = st.sidebar.select_slider("dt", options=DT_OPTIONS, value=1e-4)

    if st.sidebar.button("Run simulation", type="primary"):
        with st.spinner("integrating..."):
            res = simulate_open(
                n0=n0, n_max=n_max, lam=lam, alpha=alpha, t_end=t_end, dt=dt,
                dwell_tau=dwell, band_b=band_b, join_prob=join_prob, seed=seed,
            )
        st.session_state["res"] = res
        st.session_state["kind"] = "open"
        st.session_state["open_frame_idx"] = 0

    if st.session_state.get("kind") == "open":
        res = st.session_state["res"]
        rep = open_gain_report(lam, alpha, n_max, res["pi"], band_b, dwell)

        gain_alert(rep["thm41_ok"], "Thm 4.1 (consensus)",
                   f"need 0 < α < 2λ/n_max = {rep['upper']:.3g}, got α={alpha:.3g}, μ2={rep['mu2']:.3g}")
        gain_alert(rep["thm43_gain_ok"], "Thm 4.3 gains",
                   f"need n_max·Π={rep['lower']:.3g} < α < 2λ/n_max={rep['upper']:.3g}, got α={alpha:.3g}, Π={res['pi']:.3g}")
        gain_alert(rep["thm43_bound_ok"], "Thm 4.3 bound",
                   f"need B ≤ (α/n_max − Π)·Δτ = {rep['margin'] * dwell:.3g}, got B={band_b:.3g} "
                   f"(net decrement D={rep['net_decrement']:.3g} per period)")

        tail_err = max(res["v2"][-200:])
        if not (rep["thm41_ok"] and rep["thm43_gain_ok"] and rep["thm43_bound_ok"]) and tail_err > 2 * band_b:
            st.error(f"⚠️ Simulation shows divergent/unbounded tracking error: |c−m(u)| reaches {tail_err:.3g} "
                     f"near t_end, well above the target band B={band_b:.3g}.")

        st.plotly_chart(agents_chart(res["t"], res["x"], None, res["m"]), use_container_width=True)
        st.write(f"{len(res['log'])} join/leave events (green = join, red = leave) — "
                 "see markers on the network explorer below.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Π (ref. speed)", f"{res['pi']:.3g}")
        col2.metric("Net decrement D per period", f"{rep['net_decrement']:.3g}")
        col3.metric("Events", f"{len(res['log'])}")

        st.subheader("Consensus value vs. target median")
        st.plotly_chart(consensus_vs_median_chart(res["t"], res["c"], res["m"], res["lo"], res["hi"]),
                         use_container_width=True)

        v2_fig = go.Figure(go.Scatter(x=res["t"], y=res["v2"], name="|mean(x) - m(u)|"))
        v2_fig.add_hline(y=band_b, line=dict(color="red", dash="dot"), annotation_text="B bound")
        v2_fig.update_layout(xaxis_title="t [s]", yaxis_title="tracking error", height=250)
        st.plotly_chart(v2_fig, use_container_width=True)

        network_explorer(res["frames"], key="open")
    else:
        st.info("Set parameters in the sidebar and press Run simulation.")
