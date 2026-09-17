"""Turn a wide time-series CSV (one timestamp column + one column per agent) into a
reference signal u_i(t) the closed-network protocol can consume, in place of the
synthetic sinusoids. Robust to arbitrary CSVs, not just the bundled PM2.5 demo: the
timestamp column defaults to whichever is first, and its values can be either ISO
timestamps or plain numeric seconds — no fixed schema assumed."""
from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import numpy as np

DEMO_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "pm25_demo.csv"


def _parse_time(value: str) -> float:
    """ISO timestamp -> POSIX seconds; otherwise treat the column as already numeric."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return float(value)


def _read_rows(text: str, time_col: str | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows")
    fieldnames = reader.fieldnames or list(rows[0])
    time_col = time_col or fieldnames[0]
    if time_col not in fieldnames:
        raise ValueError(f"time column '{time_col}' not found; columns are {fieldnames}")
    cols = [c for c in fieldnames if c != time_col]
    if not cols:
        raise ValueError("CSV has a timestamp column but no signal columns")
    try:
        times = [_parse_time(r[time_col]) for r in rows]
    except ValueError as exc:
        raise ValueError(f"could not parse column '{time_col}' as a timestamp or a number: {exc}") from exc
    t0 = times[0]
    t = np.array([ti - t0 for ti in times])
    try:
        u = np.array([[float(r[c]) for c in cols] for r in rows])
    except ValueError as exc:
        raise ValueError(f"could not parse a signal column as a number: {exc}") from exc
    order = np.argsort(t)
    return t[order], u[order], cols


def load_csv_signals(source: str | Path | bytes, time_col: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (t seconds-from-start, u (steps, n_signals), column names), sorted by time.

    time_col defaults to the first column in the file; pass it explicitly only if the
    timestamp isn't first.
    """
    if isinstance(source, bytes):
        text = source.decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return _read_rows(text, time_col)


def make_reference_from_csv(source: str | Path | bytes, time_col: str | None = None) -> dict:
    """Build a closed-network reference from a CSV: linear interpolation between
    samples for u_func(t), and Pi (Ass. 3.2 bound) from the largest observed slope."""
    t_data, u_data, labels = load_csv_signals(source, time_col)
    if len(t_data) > 1:
        diffs = np.diff(u_data, axis=0) / np.diff(t_data)[:, None]
        pi_bound = float(np.max(np.abs(diffs)))
    else:
        pi_bound = 0.0

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


def pick_window(t_data: np.ndarray, dt: float, max_steps: int, target_samples: int = 60) -> float:
    """Automatically size the simulated window from the data itself: no user-chosen
    start/length. Aim to cover `target_samples` raw rows (enough to see the reference
    actually move), capped by how many Euler steps are affordable at this dt.
    """
    if len(t_data) < 2:
        return dt
    span = float(t_data[-1] - t_data[0])
    if span <= 0:
        return dt
    native_dt = float(np.median(np.diff(t_data)))
    desired = min(span, target_samples * native_dt)
    return max(dt, min(desired, max_steps * dt))
