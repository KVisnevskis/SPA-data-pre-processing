from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_INPUT_COLUMNS: list[str] = [
    "BR_X",
    "BR_Y",
    "BR_Z",
    "BR_W",
    "BP_X",
    "BP_Y",
    "BP_Z",
    "TR_X",
    "TR_Y",
    "TR_Z",
    "TR_W",
    "TP_X",
    "TP_Y",
    "TP_Z",
]


def _validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OptiTrack columns: {missing}")


def _to_float_array(df: pd.DataFrame, column: str, *, strict: bool) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
    if strict and not np.isfinite(values).all():
        bad_row = int(np.flatnonzero(~np.isfinite(values))[0])
        raise ValueError(f"Column '{column}' contains non-finite data at row {bad_row}")
    return values


def _normalize_quaternion_components(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    label: str,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if strict:
        bad = np.flatnonzero(norm <= 0.0)
        if bad.size:
            raise ValueError(f"{label} quaternion has zero norm at row {int(bad[0])}")

    safe_norm = np.where(norm > 0.0, norm, np.nan)
    return w / safe_norm, x / safe_norm, y / safe_norm, z / safe_norm


def _quaternion_multiply(
    a_w: np.ndarray,
    a_x: np.ndarray,
    a_y: np.ndarray,
    a_z: np.ndarray,
    b_w: np.ndarray,
    b_x: np.ndarray,
    b_y: np.ndarray,
    b_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_w = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z
    out_x = a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y
    out_y = a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x
    out_z = a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w
    return out_w, out_x, out_y, out_z


def _euler_zyx_from_quaternion(
    q_w: np.ndarray,
    q_x: np.ndarray,
    q_y: np.ndarray,
    q_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Legacy-compatible ZYX equations:
    # phi   = roll (x-axis), theta = pitch (y-axis), psi = yaw (z-axis)
    phi = np.arctan2(
        2.0 * (q_w * q_x + q_y * q_z),
        1.0 - 2.0 * (q_x * q_x + q_y * q_y),
    )
    theta_arg = 2.0 * (q_w * q_y - q_z * q_x)
    theta = np.arcsin(np.clip(theta_arg, -1.0, 1.0))
    psi = np.arctan2(
        2.0 * (q_w * q_z + q_x * q_y),
        1.0 - 2.0 * (q_y * q_y + q_z * q_z),
    )
    return phi, theta, psi


def _unwrap_radians(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite_idx = np.flatnonzero(np.isfinite(out))
    if finite_idx.size == 0:
        return out

    # Unwrap contiguous finite segments independently.
    split = np.flatnonzero(np.diff(finite_idx) > 1)
    starts = np.concatenate(([0], split + 1))
    ends = np.concatenate((split + 1, [finite_idx.size]))
    for s, e in zip(starts, ends):
        seg_idx = finite_idx[s:e]
        out[seg_idx] = np.unwrap(out[seg_idx])
    return out


def compute_optitrack_relative_features(
    df: pd.DataFrame,
    *,
    normalize_quaternions: bool = True,
    strict: bool = False,
    include_relative_quaternion: bool = False,
    unwrap_phi: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Stage-2 OptiTrack feature computation.

    Computes:
      q_rel = q_base^{-1} * q_tip
      phi/theta/psi from q_rel using ZYX equations
      optional phase-unwrapping of phi for continuity across +/-pi
      dx/dy/dz using schema convention: BP - TP

    If strict=False, non-finite input rows are preserved as NaN outputs.
    """
    _validate_required_columns(df, REQUIRED_INPUT_COLUMNS)
    out = df.copy(deep=True) if copy else df

    bw = _to_float_array(df, "BR_W", strict=strict)
    bx = _to_float_array(df, "BR_X", strict=strict)
    by = _to_float_array(df, "BR_Y", strict=strict)
    bz = _to_float_array(df, "BR_Z", strict=strict)

    tw = _to_float_array(df, "TR_W", strict=strict)
    tx = _to_float_array(df, "TR_X", strict=strict)
    ty = _to_float_array(df, "TR_Y", strict=strict)
    tz = _to_float_array(df, "TR_Z", strict=strict)

    bp_x = _to_float_array(df, "BP_X", strict=strict)
    bp_y = _to_float_array(df, "BP_Y", strict=strict)
    bp_z = _to_float_array(df, "BP_Z", strict=strict)
    tp_x = _to_float_array(df, "TP_X", strict=strict)
    tp_y = _to_float_array(df, "TP_Y", strict=strict)
    tp_z = _to_float_array(df, "TP_Z", strict=strict)

    if normalize_quaternions:
        bw, bx, by, bz = _normalize_quaternion_components(
            bw, bx, by, bz, label="Base", strict=strict
        )
        tw, tx, ty, tz = _normalize_quaternion_components(
            tw, tx, ty, tz, label="Tip", strict=strict
        )

    # For unit quaternions: q^{-1} = conjugate(q)
    bw_conj = bw
    bx_conj = -bx
    by_conj = -by
    bz_conj = -bz

    qrel_w, qrel_x, qrel_y, qrel_z = _quaternion_multiply(
        bw_conj,
        bx_conj,
        by_conj,
        bz_conj,
        tw,
        tx,
        ty,
        tz,
    )

    phi, theta, psi = _euler_zyx_from_quaternion(qrel_w, qrel_x, qrel_y, qrel_z)
    if unwrap_phi:
        phi = _unwrap_radians(phi)

    out["phi"] = phi
    out["theta"] = theta
    out["psi"] = psi
    out["dx"] = bp_x - tp_x
    out["dy"] = bp_y - tp_y
    out["dz"] = bp_z - tp_z

    if include_relative_quaternion:
        out["QREL_W"] = qrel_w
        out["QREL_X"] = qrel_x
        out["QREL_Y"] = qrel_y
        out["QREL_Z"] = qrel_z

    return out


def compute_angle(data: pd.DataFrame, name: str | None = None) -> pd.DataFrame:
    """
    Backward-compatible wrapper for legacy scripts.
    """
    _ = name  # preserved for compatibility with legacy signature
    return compute_optitrack_relative_features(data)


def insert_cols(data: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible helper retained for legacy scripts.
    """
    for c in ["phi", "theta", "psi", "dx", "dy", "dz"]:
        if c not in data.columns:
            data.insert(len(data.columns), c, np.nan)
    return data
