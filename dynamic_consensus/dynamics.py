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
