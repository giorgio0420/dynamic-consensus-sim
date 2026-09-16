"""Non-smooth consensus protocol.

Own implementation of the interaction law and median statistics described in
Sanai Dashti, Seatzu, Franceschelli, "Dynamic Consensus on the Median Value
in Open Multi-Agent Systems", IEEE Transactions on Automatic Control.
No text or figures from the paper are reproduced here.
"""
from __future__ import annotations

import numpy as np


def protocol_rhs(x: np.ndarray, u: np.ndarray, adj: np.ndarray, lam: float, alpha: float) -> np.ndarray:
    """dx/dt = -lam * sum_j sign(x_i - x_j) - alpha * sign(x_i - u_i)."""
    diff = x[:, None] - x[None, :]
    coupling = -lam * (adj * np.sign(diff)).sum(axis=1)
    fidelity = -alpha * np.sign(x - u)
    return coupling + fidelity


def median_interval(u: np.ndarray) -> tuple[float, float, float]:
    """Return (m, lo, hi): unique median value and the median interval bounds."""
    s = np.sort(u)
    n = len(s)
    if n % 2:
        m = float(s[n // 2])
        return m, m, m
    lo, hi = float(s[n // 2 - 1]), float(s[n // 2])
    return (lo + hi) / 2, lo, hi


def v1(x: np.ndarray) -> float:
    """Lyapunov function of Thm 4.1: gap between the top group and bottom group mean."""
    xmax, xmin = x.max(), x.min()
    return float(x[x == xmax].mean() - x[x == xmin].mean())


def closed_gain_report(lam: float, alpha: float, n: int, pi: float) -> dict:
    """Margins for Thm 4.1 (0 < alpha < 2*lam/n) and Thm 4.2 (n*pi < alpha < 2*lam/n)."""
    upper = 2 * lam / n
    lower = n * pi
    mu2 = 2 * (upper - alpha)
    return dict(upper=upper, lower=lower, mu2=mu2,
                thm41_ok=mu2 > 0, thm42_ok=lower < alpha < upper)


def open_gain_report(lam: float, alpha: float, n_max: int, pi: float, band_b: float, dwell_tau: float) -> dict:
    """Margins for Thm 4.3: n_max*pi < alpha < 2*lam/n_max and B <= (alpha/n_max - pi)*dwell_tau."""
    upper = 2 * lam / n_max
    lower = n_max * pi
    mu2 = 2 * (upper - alpha)
    margin = alpha / n_max - pi
    net_decrement = margin * dwell_tau - band_b
    return dict(upper=upper, lower=lower, mu2=mu2, margin=margin, net_decrement=net_decrement,
                thm41_ok=mu2 > 0, thm43_gain_ok=lower < alpha < upper, thm43_bound_ok=net_decrement > 0)
