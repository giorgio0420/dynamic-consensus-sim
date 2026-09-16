import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dynamic_consensus.simulate import simulate_closed, simulate_open

st.set_page_config(page_title="Dynamic Median Consensus", layout="wide")
st.title("Dynamic Consensus on the Median Value — simulator")
st.caption(
    "Own implementation of the protocol in Sanai Dashti, Seatzu, Franceschelli, "
    "\"Dynamic Consensus on the Median Value in Open Multi-Agent Systems\", IEEE TAC. "
    "No text or figures from the paper are reproduced here — only the math is implemented."
)

scenario = st.sidebar.radio("Scenario", ["Closed network", "Open network"])
lam = st.sidebar.slider("λ (coupling gain)", 1.0, 200.0, 90.0)
alpha = st.sidebar.slider("α (fidelity gain)", 0.1, 50.0, 8.0)
seed = st.sidebar.number_input("seed", 0, 10_000, 0)

DT_OPTIONS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

if scenario == "Closed network":
    n = st.sidebar.slider("agents", 2, 20, 5)
    t_end = st.sidebar.slider("sim time (s)", 0.01, 2.0, 0.3)
    dt = st.sidebar.select_slider("dt", options=DT_OPTIONS, value=1e-4)

    if st.sidebar.button("Run simulation", type="primary"):
        with st.spinner("integrating..."):
            res = simulate_closed(n=n, lam=lam, alpha=alpha, t_end=t_end, dt=dt, seed=seed)

        fig = go.Figure()
        for i in range(n):
            fig.add_trace(go.Scatter(x=res["t"], y=res["x"][:, i], name=f"x_{i}", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=res["t"], y=res["m"], name="m(u)",
                                  line=dict(color="black", width=3, dash="dash")))
        fig.update_layout(xaxis_title="t [s]", yaxis_title="state", height=500)
        st.plotly_chart(fig, use_container_width=True)

        mu2 = 2 * (2 * lam / n - alpha)
        t1_bound = (res["x"][0].max() - res["x"][0].min()) / mu2 if mu2 > 0 else float("nan")
        col1, col2, col3 = st.columns(3)
        col1.metric("Thm 4.1 bound T1", f"{t1_bound:.4g} s")
        col2.metric("μ2", f"{mu2:.3g}")
        col3.metric("Π (ref. speed)", f"{res['pi']:.3g}")
        if not (n * res["pi"] < alpha < 2 * lam / n):
            st.warning("Gain condition n·Π < α < 2λ/n of Theorem 4.2 is NOT satisfied.")

        v1_fig = go.Figure(go.Scatter(x=res["t"], y=res["v1"], name="V1"))
        if mu2 > 0:
            v1_fig.add_vline(x=t1_bound, line=dict(color="red", dash="dot"), annotation_text="T1 bound")
        v1_fig.update_layout(xaxis_title="t [s]", yaxis_title="V1(x)", height=250)
        st.plotly_chart(v1_fig, use_container_width=True)
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

        all_ids = sorted({i for d in res["x"] for i in d})
        fig = go.Figure()
        for i in all_ids:
            ys = [d.get(i, None) for d in res["x"]]
            fig.add_trace(go.Scatter(x=res["t"], y=ys, name=f"x_{i}", line=dict(width=1), connectgaps=False))
        fig.add_trace(go.Scatter(x=res["t"], y=res["m"], name="m(u)",
                                  line=dict(color="black", width=3, dash="dash")))
        for t_ev, kind, _node in res["log"]:
            fig.add_vline(x=t_ev, line=dict(color="green" if kind == "join" else "crimson", width=1, dash="dot"))
        fig.update_layout(xaxis_title="t [s]", yaxis_title="state", height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"{len(res['log'])} join/leave events (green = join, red = leave).")

        d_margin = (alpha / n_max - res["pi"]) * dwell - band_b
        col1, col2, col3 = st.columns(3)
        col1.metric("Π (ref. speed)", f"{res['pi']:.3g}")
        col2.metric("Net decrement D per period", f"{d_margin:.3g}")
        col3.metric("Events", f"{len(res['log'])}")
        if d_margin <= 0:
            st.warning("Theorem 4.3 condition B ≤ (α/n_max − Π)·Δτ is NOT satisfied — bounded-error guarantee may not hold.")

        v2_fig = go.Figure(go.Scatter(x=res["t"], y=res["v2"], name="|c - m(u)|"))
        v2_fig.add_hline(y=band_b, line=dict(color="red", dash="dot"), annotation_text="B bound")
        v2_fig.update_layout(xaxis_title="t [s]", yaxis_title="tracking error", height=250)
        st.plotly_chart(v2_fig, use_container_width=True)
    else:
        st.info("Set parameters in the sidebar and press Run simulation.")
