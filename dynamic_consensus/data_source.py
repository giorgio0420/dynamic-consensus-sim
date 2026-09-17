"""Turn a wide time-series CSV (one timestamp column + one column per agent) into a
reference signal u_i(t) the closed-network protocol can consume, in place of the
synthetic sinusoids. No scenario-specific parsing: any CSV with this shape works,
including the PM2.5 dataset bundled in data/pm25_demo.csv."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import numpy as np

DEMO_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "pm25_demo.csv"


def _read_rows(text: str, time_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    cols = [c for c in rows[0] if c != time_col]
    times = [datetime.fromisoformat(r[time_col]) for r in rows]
    t0 = times[0]
    t = np.array([(ti - t0).total_seconds() for ti in times])
    u = np.array([[float(r[c]) for c in cols] for r in rows])
    order = np.argsort(t)
    return t[order], u[order], cols


def load_csv_signals(source: str | Path | bytes, time_col: str = "created_at") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (t seconds-from-start, u (steps, n_signals), column names), sorted by time."""
    if isinstance(source, bytes):
        text = source.decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return _read_rows(text, time_col)


def make_reference_from_csv(source: str | Path | bytes, time_col: str = "created_at") -> dict:
    """Build a closed-network reference from a CSV: linear interpolation between
    samples for u_func(t), and Pi (Ass. 3.2 bound) from the largest observed slope."""
    t_data, u_data, labels = load_csv_signals(source, time_col)
    diffs = np.diff(u_data, axis=0) / np.diff(t_data)[:, None]
    pi_bound = float(np.max(np.abs(diffs)))

    def u_func(t: float) -> np.ndarray:
        return np.array([np.interp(t, t_data, u_data[:, j]) for j in range(u_data.shape[1])])

    return dict(
        u_func=u_func, pi_bound=pi_bound, labels=labels, n=len(labels),
        t_data=t_data, u_data=u_data, t_span=(float(t_data[0]), float(t_data[-1])),
    )


def resample_to_grid(t_data: np.ndarray, u_data: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Vectorized linear interpolation of every column onto a fixed time grid, once.

    u_func above calls np.interp per agent per simulation step, which is fine for
    interactive single lookups but far too slow over a fine dt x long t_end grid
    (hundreds of thousands of Python-level calls). This does the same interpolation
    with one np.interp call per agent over the whole grid, then the simulation loop
    just indexes into the result.
    """
    return np.column_stack([np.interp(t_grid, t_data, u_data[:, j]) for j in range(u_data.shape[1])])
