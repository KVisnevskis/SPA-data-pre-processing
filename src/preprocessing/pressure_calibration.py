from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SteadySegment:
    segment_id: int
    start_sample: int
    end_sample: int
    n_samples: int
    duration_s: float
    adc_median: float
    adc_mean: float
    adc_std: float
    adc_mad: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AffinePressureCalibration:
    slope_pa_per_count: float
    intercept_pa: float

    def pressure_pa(self, adc: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(adc, dtype=np.float64)
        return self.slope_pa_per_count * arr + self.intercept_pa


def decode_pressure_adc_from_raw_csv(
    path: str | Path,
    *,
    expected_columns: int = 16,
    lsb_col: int = 12,
    msb_col: int = 13,
    adc_right_shift: int = 4,
) -> np.ndarray:
    raw = pd.read_csv(Path(path), header=None)
    if raw.shape[1] != expected_columns:
        raise ValueError(
            f"{Path(path).name}: expected {expected_columns} columns, got {raw.shape[1]}"
        )
    lo = raw[lsb_col].to_numpy(dtype=np.int64)
    hi = raw[msb_col].to_numpy(dtype=np.int64)
    return ((lo | (hi << 8)) >> adc_right_shift).astype(np.float64)


def detect_steady_pressure_segments(
    pressure_adc: Sequence[float] | np.ndarray,
    *,
    sample_rate_hz: float = 240.0,
    smooth_window: int = 31,
    std_window: int = 121,
    derivative_threshold: float = 1.2,
    std_threshold: float = 5.0,
    min_duration_s: float = 1.0,
) -> list[SteadySegment]:
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be > 0")
    if smooth_window < 1 or std_window < 1:
        raise ValueError("smooth_window and std_window must be >= 1")
    if derivative_threshold <= 0.0 or std_threshold <= 0.0:
        raise ValueError("derivative_threshold and std_threshold must be > 0")
    if min_duration_s <= 0.0:
        raise ValueError("min_duration_s must be > 0")

    adc = np.asarray(pressure_adc, dtype=np.float64)
    if adc.ndim != 1 or adc.size == 0:
        raise ValueError("pressure_adc must be a non-empty 1D array")
    if not np.isfinite(adc).all():
        bad_idx = int(np.flatnonzero(~np.isfinite(adc))[0])
        raise ValueError(f"pressure_adc contains non-finite values at index {bad_idx}")

    smoothed = (
        pd.Series(adc)
        .rolling(window=smooth_window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=np.float64)
    )
    abs_diff = np.abs(np.diff(smoothed, prepend=smoothed[0]))
    local_std = (
        pd.Series(adc)
        .rolling(window=std_window, center=True, min_periods=1)
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    steady_mask = (abs_diff <= derivative_threshold) & (local_std <= std_threshold)
    min_samples = int(np.ceil(min_duration_s * sample_rate_hz))

    segments: list[SteadySegment] = []
    in_segment = False
    start = 0
    seg_id = 0

    for i, is_steady in enumerate(steady_mask):
        if is_steady and not in_segment:
            start = i
            in_segment = True
        elif not is_steady and in_segment:
            end = i - 1
            n_samples = end - start + 1
            if n_samples >= min_samples:
                vals = adc[start : end + 1]
                median = float(np.median(vals))
                mad = float(np.median(np.abs(vals - median)))
                segments.append(
                    SteadySegment(
                        segment_id=seg_id,
                        start_sample=int(start),
                        end_sample=int(end),
                        n_samples=int(n_samples),
                        duration_s=float(n_samples / sample_rate_hz),
                        adc_median=median,
                        adc_mean=float(np.mean(vals)),
                        adc_std=float(np.std(vals, ddof=0)),
                        adc_mad=mad,
                    )
                )
                seg_id += 1
            in_segment = False

    if in_segment:
        end = int(len(steady_mask) - 1)
        n_samples = end - start + 1
        if n_samples >= min_samples:
            vals = adc[start : end + 1]
            median = float(np.median(vals))
            mad = float(np.median(np.abs(vals - median)))
            segments.append(
                SteadySegment(
                    segment_id=seg_id,
                    start_sample=int(start),
                    end_sample=end,
                    n_samples=int(n_samples),
                    duration_s=float(n_samples / sample_rate_hz),
                    adc_median=median,
                    adc_mean=float(np.mean(vals)),
                    adc_std=float(np.std(vals, ddof=0)),
                    adc_mad=mad,
                )
            )

    return segments


def segments_to_dataframe(segments: Sequence[SteadySegment]) -> pd.DataFrame:
    rows = [s.to_dict() for s in segments]
    return pd.DataFrame(rows)


def fit_affine_pressure_calibration(
    adc_values: Sequence[float] | np.ndarray,
    pressure_pa_values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[AffinePressureCalibration, dict[str, float]]:
    x = np.asarray(adc_values, dtype=np.float64)
    y = np.asarray(pressure_pa_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("adc_values and pressure_pa_values must be 1D arrays of equal length")
    if x.size < 2:
        raise ValueError("Need at least 2 calibration points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Calibration points contain non-finite values")
    if np.ptp(x) == 0.0:
        raise ValueError("adc_values must span at least two distinct values")

    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size != x.size:
            raise ValueError("weights must be a 1D array with same length as calibration points")
        if not np.isfinite(w).all() or np.any(w <= 0.0):
            raise ValueError("weights must be finite and > 0")

    sqrt_w = np.sqrt(w)
    X = np.column_stack([x, np.ones_like(x)])
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    slope = float(beta[0])
    intercept = float(beta[1])

    model = AffinePressureCalibration(slope_pa_per_count=slope, intercept_pa=intercept)
    y_pred = model.pressure_pa(x)
    residual = y - y_pred
    w_sum = float(np.sum(w))

    weighted_rmse = float(np.sqrt(np.sum(w * residual**2) / w_sum))
    weighted_mae = float(np.sum(w * np.abs(residual)) / w_sum)
    max_abs_error = float(np.max(np.abs(residual)))

    y_w_mean = float(np.sum(w * y) / w_sum)
    ss_res = float(np.sum(w * residual**2))
    ss_tot = float(np.sum(w * (y - y_w_mean) ** 2))
    r2_weighted = 1.0 if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

    metrics = {
        "n_points": float(x.size),
        "weighted_rmse_pa": weighted_rmse,
        "weighted_mae_pa": weighted_mae,
        "max_abs_error_pa": max_abs_error,
        "r2_weighted": r2_weighted,
    }
    return model, metrics


def affine_to_ratiometric_params(
    calibration: AffinePressureCalibration,
    *,
    p_max_pa: float,
    adc_max: int = 4095,
) -> dict[str, float]:
    m = float(calibration.slope_pa_per_count)
    b = float(calibration.intercept_pa)
    if m <= 0.0:
        raise ValueError(f"slope_pa_per_count must be > 0, got {m}")
    if p_max_pa <= 0.0:
        raise ValueError(f"p_max_pa must be > 0, got {p_max_pa}")
    if adc_max <= 0:
        raise ValueError(f"adc_max must be > 0, got {adc_max}")

    v_min_ratio = -b / (adc_max * m)
    span = p_max_pa / (adc_max * m)
    v_max_ratio = v_min_ratio + span

    return {
        "adc_max": float(adc_max),
        "p_max_pa": float(p_max_pa),
        "v_min_ratio": float(v_min_ratio),
        "v_max_ratio": float(v_max_ratio),
    }
