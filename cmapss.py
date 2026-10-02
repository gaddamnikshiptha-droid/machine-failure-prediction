"""Data loading, preprocessing, windowing and evaluation utilities for NASA C-MAPSS.

The C-MAPSS dataset simulates turbofan engines that run until failure. Each row is
one operating cycle of one engine, with three operating settings and 21 sensors.
The task is to predict the Remaining Useful Life (RUL): how many cycles remain
before the engine fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMNS: list[str] = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Sensors that carry degradation signal in FD001. The remaining channels are
# constant or near-constant across the fleet and add only noise.
SENSORS: list[str] = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

WINDOW: int = 30
# Piecewise-linear RUL target: early in life an engine is treated as fully
# healthy, which is the standard formulation for this benchmark.
RUL_CAP: int = 125

CRITICAL_THRESHOLD: int = 30
WARNING_THRESHOLD: int = 75


def _read_table(path: Path) -> pd.DataFrame:
    """Read a whitespace-separated C-MAPSS file and assign column names."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}. Download the NASA C-MAPSS dataset and place "
            "train_FD001.txt, test_FD001.txt and RUL_FD001.txt in the data directory."
        )
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] < len(COLUMNS):
        raise ValueError(
            f"{path} has {frame.shape[1]} columns; expected at least {len(COLUMNS)}."
        )
    frame = frame.iloc[:, : len(COLUMNS)].copy()
    frame.columns = COLUMNS
    frame["unit"] = frame["unit"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


def load_train(data_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Load the run-to-failure training set and attach the capped RUL target."""
    frame = _read_table(Path(data_dir) / f"train_{subset}.txt")
    max_cycle = frame.groupby("unit")["cycle"].transform("max")
    frame["rul"] = (max_cycle - frame["cycle"]).clip(upper=RUL_CAP).astype(np.float32)
    return frame


def load_test(data_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Load the truncated test set and attach the capped RUL for every cycle."""
    data_dir = Path(data_dir)
    frame = _read_table(data_dir / f"test_{subset}.txt")

    truth_path = data_dir / f"RUL_{subset}.txt"
    if not truth_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {truth_path}.")
    truth = pd.read_csv(truth_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()

    units = np.sort(frame["unit"].unique())
    if len(truth) != len(units):
        raise ValueError(
            f"{truth_path} has {len(truth)} rows but the test set has {len(units)} engines."
        )

    final_rul = pd.Series(truth, index=units)
    max_cycle = frame.groupby("unit")["cycle"].transform("max")
    rul = frame["unit"].map(final_rul) + (max_cycle - frame["cycle"])
    frame["rul"] = rul.clip(upper=RUL_CAP).astype(np.float32)
    return frame


def split_units(
    frame: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by engine so no engine appears in both training and validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    units = frame["unit"].unique().copy()
    np.random.default_rng(seed).shuffle(units)
    n_val = max(1, int(round(len(units) * val_fraction)))
    val_mask = frame["unit"].isin(set(units[:n_val].tolist()))
    return frame[~val_mask].reset_index(drop=True), frame[val_mask].reset_index(drop=True)


def fit_scaler(frame: pd.DataFrame) -> MinMaxScaler:
    """Fit a min-max scaler on the selected sensor channels."""
    scaler = MinMaxScaler()
    scaler.fit(frame[SENSORS].to_numpy(dtype=np.float64))
    return scaler


def scale(frame: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Return a copy of the frame with sensor channels scaled to [0, 1]."""
    scaled = frame.copy()
    values = scaler.transform(frame[SENSORS].to_numpy(dtype=np.float64))
    scaled[SENSORS] = values.astype(np.float32)
    return scaled


def _front_pad(values: np.ndarray, length: int) -> np.ndarray:
    """Repeat the first row so the array has at least `length` rows."""
    if len(values) >= length:
        return values
    padding = np.repeat(values[:1], length - len(values), axis=0)
    return np.concatenate([padding, values], axis=0)


def _sliding(values: np.ndarray, window: int) -> np.ndarray:
    """Return all windows of shape (window, features) from a (time, features) array."""
    views = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    return views.transpose(0, 2, 1)


def make_windows(frame: pd.DataFrame, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Build every sliding window per engine; the target is the RUL at the window end."""
    windows, targets = [], []
    for _, group in frame.groupby("unit", sort=True):
        values = _front_pad(group[SENSORS].to_numpy(np.float32), window)
        ruls = _front_pad(group["rul"].to_numpy(np.float32).reshape(-1, 1), window).ravel()
        windows.append(_sliding(values, window))
        targets.append(ruls[window - 1:])
    return (
        np.concatenate(windows).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
    )


def last_windows(
    frame: pd.DataFrame, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the final window, unit id and final RUL for each engine."""
    windows, units, ruls = [], [], []
    for unit, group in frame.groupby("unit", sort=True):
        values = _front_pad(group[SENSORS].to_numpy(np.float32), window)
        windows.append(values[-window:])
        units.append(int(unit))
        ruls.append(float(group["rul"].iloc[-1]))
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(units, dtype=int),
        np.asarray(ruls, dtype=np.float32),
    )


def unit_trajectory_windows(unit_frame: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """Return one window ending at every cycle of a single engine (front-padded)."""
    values = unit_frame[SENSORS].to_numpy(np.float32)
    if len(values) == 0:
        raise ValueError("unit_frame is empty.")
    padded = _front_pad(values, len(values) + window - 1)
    return _sliding(padded, window).astype(np.float32)


def summary_features(windows: np.ndarray) -> np.ndarray:
    """Compress each window into mean, last value and linear slope per sensor.

    Used by the SVR baseline, which works best on a compact feature vector
    rather than a raw time sequence.
    """
    steps = np.arange(windows.shape[1], dtype=np.float32)
    centered = steps - steps.mean()
    mean = windows.mean(axis=1)
    last = windows[:, -1, :]
    slope = np.einsum("t,ntf->nf", centered, windows - mean[:, None, :]) / np.sum(centered**2)
    return np.concatenate([mean, last, slope], axis=1).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and the asymmetric NASA scoring function (late predictions cost more)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    nasa = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_score": float(np.sum(nasa)),
    }


def health_status(rul: float) -> str:
    """Map a predicted RUL to a maintenance status label."""
    if rul < CRITICAL_THRESHOLD:
        return "Critical"
    if rul < WARNING_THRESHOLD:
        return "Service Soon"
    return "Healthy"
