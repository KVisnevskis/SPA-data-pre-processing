from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd


GyroBiasMethod = Literal["mean", "median"]


@dataclass(frozen=True)
class GyroCalibration:
    bias_x_rads: float = 0.0
    bias_y_rads: float = 0.0
    bias_z_rads: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_array(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.isfinite(arr).all():
        bad_idx = int(np.flatnonzero(~np.isfinite(arr))[0])
        raise ValueError(f"{name} contains non-finite values at index {bad_idx}")
    return arr


def fit_gyro_bias_from_arrays(
    gyr_x_rads: Sequence[float] | np.ndarray,
    gyr_y_rads: Sequence[float] | np.ndarray,
    gyr_z_rads: Sequence[float] | np.ndarray,
    *,
    method: GyroBiasMethod = "mean",
) -> tuple[GyroCalibration, dict[str, float]]:
    x = _validate_array("gyr_x_rads", gyr_x_rads)
    y = _validate_array("gyr_y_rads", gyr_y_rads)
    z = _validate_array("gyr_z_rads", gyr_z_rads)
    if not (x.size == y.size == z.size):
        raise ValueError("gyr_x_rads, gyr_y_rads, gyr_z_rads must have equal lengths")
    if method not in ("mean", "median"):
        raise ValueError(f"Unsupported method: {method}. Supported: 'mean', 'median'")

    reducer = np.mean if method == "mean" else np.median
    calib = GyroCalibration(
        bias_x_rads=float(reducer(x)),
        bias_y_rads=float(reducer(y)),
        bias_z_rads=float(reducer(z)),
    )

    stats = {
        "n_samples": float(x.size),
        "method": method,
        "std_x_rads": float(np.std(x, ddof=0)),
        "std_y_rads": float(np.std(y, ddof=0)),
        "std_z_rads": float(np.std(z, ddof=0)),
    }
    return calib, stats


def fit_gyro_bias_from_dataframe(
    df: pd.DataFrame,
    *,
    gyro_columns: tuple[str, str, str] = ("gyr_x_rads", "gyr_y_rads", "gyr_z_rads"),
    method: GyroBiasMethod = "mean",
) -> tuple[GyroCalibration, dict[str, float]]:
    cx, cy, cz = gyro_columns
    missing = [c for c in (cx, cy, cz) if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required gyro columns: {missing}")
    return fit_gyro_bias_from_arrays(
        pd.to_numeric(df[cx], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[cy], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df[cz], errors="coerce").to_numpy(dtype=np.float64),
        method=method,
    )


def apply_gyro_bias_correction(
    gyr_x_rads: Sequence[float] | np.ndarray,
    gyr_y_rads: Sequence[float] | np.ndarray,
    gyr_z_rads: Sequence[float] | np.ndarray,
    *,
    calibration: GyroCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _validate_array("gyr_x_rads", gyr_x_rads)
    y = _validate_array("gyr_y_rads", gyr_y_rads)
    z = _validate_array("gyr_z_rads", gyr_z_rads)
    if not (x.size == y.size == z.size):
        raise ValueError("gyr_x_rads, gyr_y_rads, gyr_z_rads must have equal lengths")

    return (
        x - calibration.bias_x_rads,
        y - calibration.bias_y_rads,
        z - calibration.bias_z_rads,
    )


def load_gyro_calibration_json(path: str | Path) -> GyroCalibration:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Gyro calibration JSON root must be an object")

    source: dict[str, object]
    if all(k in payload for k in ("bias_x_rads", "bias_y_rads", "bias_z_rads")):
        source = payload
    elif "gyro_calibration" in payload and isinstance(payload["gyro_calibration"], dict):
        source = payload["gyro_calibration"]  # type: ignore[assignment]
    else:
        raise ValueError(
            "Gyro calibration JSON must contain either top-level bias_x_rads/bias_y_rads/bias_z_rads "
            "or nested gyro_calibration object"
        )

    try:
        bx = float(source["bias_x_rads"])  # type: ignore[index]
        by = float(source["bias_y_rads"])  # type: ignore[index]
        bz = float(source["bias_z_rads"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid gyro calibration JSON values; expected numeric bias_x/y/z_rads") from exc

    return GyroCalibration(bias_x_rads=bx, bias_y_rads=by, bias_z_rads=bz)
