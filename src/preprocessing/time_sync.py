from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd


SyncMode = Literal["fixed", "freehand"]
RotationDirection = Literal["world_to_body", "body_to_world"]
FixedPhiTransform = Literal["none", "invert", "abs", "auto"]


def _validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _to_float_array(df: pd.DataFrame, col: str) -> np.ndarray:
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad_row = int(np.flatnonzero(~np.isfinite(values))[0])
        raise ValueError(f"Column '{col}' contains non-finite data at row {bad_row}")
    return values


def normalize_to_unit_range(signal: np.ndarray, *, invert: bool = False) -> np.ndarray:
    """
    Min-max normalize to [-1, 1].
    If input is constant, returns all zeros.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("Signal must contain at least one sample")
    if not np.isfinite(x).all():
        raise ValueError("Signal contains non-finite values")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    denom = x_max - x_min
    if denom <= 0.0:
        out = np.zeros_like(x)
    else:
        out = (x - x_min) / denom
        out = (out - 0.5) * 2.0

    if invert:
        out = -out
    return out


def _cross_correlate_full_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Equivalent to np.correlate(x, y, mode="full"), using FFT for efficiency
    on long signals.
    """
    n = x.size + y.size - 1
    n_fft = 1 << (n - 1).bit_length()
    x_fft = np.fft.rfft(x, n=n_fft)
    y_rev_fft = np.fft.rfft(y[::-1], n=n_fft)
    corr = np.fft.irfft(x_fft * y_rev_fft, n=n_fft)[:n]
    return corr


def estimate_shift_from_signals(
    x_signal: np.ndarray,
    y_signal: np.ndarray,
    *,
    max_lag: int | None = None,
) -> tuple[int, int, float]:
    """
    Estimate alignment between x (Arduino-like) and y (OptiTrack-like) via
    full cross-correlation.

    Returns:
      sample_shift_x: integer shift to apply to x via pandas.DataFrame.shift
      lag: best lag k* in thesis convention R_xy[k] = sum x[n] y[n+k]
      max_corr: maximal correlation value

    Sign convention:
      - Positive lag means y occurs later than x.
      - sample_shift_x = lag
    """
    x = np.asarray(x_signal, dtype=np.float64)
    y = np.asarray(y_signal, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_signal and y_signal must be 1D arrays")
    if x.size == 0 or y.size == 0:
        raise ValueError("Signals must not be empty")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Signals contain non-finite values")

    corr = _cross_correlate_full_fft(x, y)
    lags = np.arange(-(y.size - 1), x.size, dtype=np.int64)

    if max_lag is not None:
        if max_lag < 0:
            raise ValueError("max_lag must be >= 0 when provided")
        mask = np.abs(lags) <= int(max_lag)
        if not np.any(mask):
            raise ValueError("max_lag excludes all candidate lags")
        corr_search = corr[mask]
        lags_search = lags[mask]
    else:
        corr_search = corr
        lags_search = lags

    best_idx = int(np.argmax(corr_search))
    lag_np = int(lags_search[best_idx])
    max_corr = float(corr_search[best_idx])
    # NumPy lag is opposite to thesis convention for R_xy[k] = sum x[n] y[n+k].
    lag = -lag_np
    sample_shift_x = lag
    return sample_shift_x, lag, max_corr


def _combine_with_shift_and_trim(
    arduino_df: pd.DataFrame,
    optitrack_df: pd.DataFrame,
    *,
    sample_shift_arduino: int,
) -> tuple[pd.DataFrame, int, int]:
    arduino_shifted = arduino_df.shift(sample_shift_arduino)
    combined = pd.concat(
        [arduino_shifted.reset_index(drop=True), optitrack_df.reset_index(drop=True)],
        axis=1,
    )
    rows_before = len(combined)
    combined = combined.dropna(how="any").reset_index(drop=True)
    rows_after = len(combined)
    return combined, rows_before, rows_after


def _quaternion_to_rotation_matrices(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    Rotation matrix for quaternion (w,x,y,z), interpreting q as body->world.
    """
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return np.stack(
        [
            np.stack([r00, r01, r02], axis=1),
            np.stack([r10, r11, r12], axis=1),
            np.stack([r20, r21, r22], axis=1),
        ],
        axis=1,
    )


def compute_virtual_accelerometer_from_optitrack(
    optitrack_df: pd.DataFrame,
    *,
    gravity_vector_world: tuple[float, float, float] = (0.0, 0.0, -9.81),
    rotation_direction: RotationDirection = "world_to_body",
) -> np.ndarray:
    """
    Build virtual accelerometer signal from base quaternion.

    Returns Nx3 acceleration array aligned with optitrack_df rows.
    """
    required = ["BR_W", "BR_X", "BR_Y", "BR_Z"]
    _validate_required_columns(optitrack_df, required, label="OptiTrack table")

    w = _to_float_array(optitrack_df, "BR_W")
    x = _to_float_array(optitrack_df, "BR_X")
    y = _to_float_array(optitrack_df, "BR_Y")
    z = _to_float_array(optitrack_df, "BR_Z")

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if np.any(norm <= 0.0):
        bad_row = int(np.flatnonzero(norm <= 0.0)[0])
        raise ValueError(f"Base quaternion has zero norm at row {bad_row}")

    w = w / norm
    x = x / norm
    y = y / norm
    z = z / norm

    rot = _quaternion_to_rotation_matrices(w, x, y, z)  # body->world
    g_world = np.asarray(gravity_vector_world, dtype=np.float64)
    if g_world.shape != (3,):
        raise ValueError("gravity_vector_world must have shape (3,)")

    g_stack = np.repeat(g_world[None, :], repeats=len(optitrack_df), axis=0)

    if rotation_direction == "world_to_body":
        # world->body uses inverse rotation: R^T
        return np.einsum("nij,nj->ni", np.transpose(rot, (0, 2, 1)), g_stack)
    if rotation_direction == "body_to_world":
        return np.einsum("nij,nj->ni", rot, g_stack)
    raise ValueError(f"Unsupported rotation_direction: {rotation_direction}")


def synchronize_fixed_orientation(
    arduino_df: pd.DataFrame,
    optitrack_features_df: pd.DataFrame,
    *,
    pressure_col: str = "pressure",
    phi_col: str = "phi",
    phi_transform: FixedPhiTransform = "auto",
    max_lag: int | None = None,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Stage-3 sync for fixed-orientation runs:
    cross-correlate Arduino pressure and OptiTrack phi.
    """
    _validate_required_columns(arduino_df, [pressure_col], label="Arduino table")
    _validate_required_columns(optitrack_features_df, [phi_col], label="OptiTrack features")

    x = normalize_to_unit_range(_to_float_array(arduino_df, pressure_col))
    phi_raw = _to_float_array(optitrack_features_df, phi_col)

    def _transform_phi(values: np.ndarray, transform: FixedPhiTransform) -> np.ndarray:
        if transform == "none":
            return values
        if transform == "invert":
            return -values
        if transform == "abs":
            return np.abs(values)
        raise ValueError(f"Unsupported phi_transform: {transform}")

    if phi_transform == "auto":
        candidates: list[FixedPhiTransform] = ["none", "invert", "abs"]
    else:
        candidates = [phi_transform]

    best_transform: FixedPhiTransform = candidates[0]
    best_shift = 0
    best_lag = 0
    best_corr = -np.inf

    for transform in candidates:
        y = normalize_to_unit_range(_transform_phi(phi_raw, transform))
        shift, lag, corr = estimate_shift_from_signals(x, y, max_lag=max_lag)
        if corr > best_corr:
            best_transform = transform
            best_shift = shift
            best_lag = lag
            best_corr = corr

    synced, rows_before, rows_after = _combine_with_shift_and_trim(
        arduino_df,
        optitrack_features_df,
        sample_shift_arduino=best_shift,
    )

    if return_info:
        info = {
            "mode": "fixed",
            "sync_variable": phi_col,
            "phi_transform_used": best_transform,
            "sample_shift_arduino": best_shift,
            "lag_samples": best_lag,
            "max_cross_correlation": float(best_corr),
            "rows_before_trim": rows_before,
            "rows_after_trim": rows_after,
            "rows_trimmed": rows_before - rows_after,
        }
        return synced, info
    return synced


def synchronize_freehand(
    arduino_df: pd.DataFrame,
    optitrack_features_df: pd.DataFrame,
    *,
    measured_accel_cols: Sequence[str] = ("acc_y", "acc_x", "acc_z"),
    virtual_accel_axis_names: Sequence[str] = ("acc_x", "acc_y", "acc_z"),
    gravity_vector_world: tuple[float, float, float] = (0.0, 0.0, -9.81),
    rotation_direction: RotationDirection = "world_to_body",
    max_lag: int | None = None,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Stage-3 sync for freehand runs:
    cross-correlate measured accelerometer with virtual accelerometer derived
    from OptiTrack base orientation.
    """
    _validate_required_columns(arduino_df, list(measured_accel_cols), label="Arduino table")
    _validate_required_columns(
        optitrack_features_df,
        ["BR_W", "BR_X", "BR_Y", "BR_Z"],
        label="OptiTrack features",
    )

    measured = np.column_stack([_to_float_array(arduino_df, c) for c in measured_accel_cols])
    virtual = compute_virtual_accelerometer_from_optitrack(
        optitrack_features_df,
        gravity_vector_world=gravity_vector_world,
        rotation_direction=rotation_direction,
    )

    best_idx = 0
    best_shift = 0
    best_lag = 0
    best_corr = -np.inf

    for i in range(3):
        x = normalize_to_unit_range(measured[:, i])
        y = normalize_to_unit_range(virtual[:, i])
        shift, lag, corr = estimate_shift_from_signals(x, y, max_lag=max_lag)
        if corr > best_corr:
            best_corr = corr
            best_idx = i
            best_shift = shift
            best_lag = lag

    synced, rows_before, rows_after = _combine_with_shift_and_trim(
        arduino_df,
        optitrack_features_df,
        sample_shift_arduino=best_shift,
    )

    if return_info:
        info = {
            "mode": "freehand",
            "sync_variable": virtual_accel_axis_names[best_idx],
            "measured_accel_column": measured_accel_cols[best_idx],
            "sample_shift_arduino": best_shift,
            "lag_samples": best_lag,
            "max_cross_correlation": float(best_corr),
            "rotation_direction": rotation_direction,
            "gravity_vector_world": list(gravity_vector_world),
            "rows_before_trim": rows_before,
            "rows_after_trim": rows_after,
            "rows_trimmed": rows_before - rows_after,
        }
        return synced, info
    return synced


def synchronize_stage3(
    arduino_df: pd.DataFrame,
    optitrack_features_df: pd.DataFrame,
    *,
    mode: SyncMode,
    fixed_phi_transform: FixedPhiTransform = "auto",
    max_lag: int | None = None,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Convenience wrapper for Stage-3 synchronization.
    """
    if mode == "fixed":
        return synchronize_fixed_orientation(
            arduino_df,
            optitrack_features_df,
            phi_transform=fixed_phi_transform,
            max_lag=max_lag,
            return_info=return_info,
        )
    if mode == "freehand":
        return synchronize_freehand(
            arduino_df,
            optitrack_features_df,
            max_lag=max_lag,
            return_info=return_info,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def synchronize_time_streams(
    arduino_df: pd.DataFrame,
    optitrack_features_df: pd.DataFrame,
    *,
    mode: SyncMode,
    fixed_phi_transform: FixedPhiTransform = "auto",
    max_lag: int | None = None,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Alias for synchronize_stage3 with a stage-agnostic name.
    """
    return synchronize_stage3(
        arduino_df,
        optitrack_features_df,
        mode=mode,
        fixed_phi_transform=fixed_phi_transform,
        max_lag=max_lag,
        return_info=return_info,
    )
