from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, overload

import numpy as np
import pandas as pd


# OptiTrack export column indices for:
# Time, Base quat (x,y,z,w), Base pos (x,y,z), Tip quat (x,y,z,w), Tip pos (x,y,z)
DEFAULT_USECOLS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 26, 27, 28, 29, 30, 31, 32]

DEFAULT_COLUMNS: list[str] = [
    "Time",
    "BR_X", "BR_Y", "BR_Z", "BR_W",
    "BP_X", "BP_Y", "BP_Z",
    "TR_X", "TR_Y", "TR_Z", "TR_W",
    "TP_X", "TP_Y", "TP_Z",
]

POSE_COLUMNS: list[str] = [c for c in DEFAULT_COLUMNS if c != "Time"]
QUATERNION_GROUPS: list[tuple[str, str, str, str]] = [
    ("BR_W", "BR_X", "BR_Y", "BR_Z"),
    ("TR_W", "TR_X", "TR_Y", "TR_Z"),
]


def find_optitrack_header_row(path: str | Path, marker: str = "Frame,Time (Seconds)") -> int:
    """
    Returns the line index (0-based) of the OptiTrack CSV header row that starts with:
    'Frame,Time (Seconds),...'
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith(marker):
                return i
            # Safety: OptiTrack header is typically very short
            if i > 200:
                break
    raise ValueError(f"Could not find OptiTrack header row starting with: {marker}")


def load_optitrack_raw_csv(
    path: str | Path,
    usecols: Sequence[int] = DEFAULT_USECOLS,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """
    Load OptiTrack CSV export and return a standardized table with:
      Time, BR_*, BP_*, TR_*, TP_*
    """
    path = Path(path)
    header_row = find_optitrack_header_row(path)

    df = pd.read_csv(path, skiprows=header_row, usecols=list(usecols))
    if len(columns) != df.shape[1]:
        raise ValueError(f"Expected {len(columns)} columns, got {df.shape[1]}")

    df = df.set_axis(list(columns), axis=1)

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _longest_true_run(mask: np.ndarray) -> tuple[int, int | None, int | None]:
    best_len = 0
    best_start: int | None = None
    best_end: int | None = None

    current_start: int | None = None
    current_len = 0

    for i, is_true in enumerate(mask):
        if is_true:
            if current_start is None:
                current_start = i
            current_len += 1
        else:
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
                best_end = i - 1
            current_start = None
            current_len = 0

    if current_len > best_len:
        best_len = current_len
        best_start = current_start
        best_end = len(mask) - 1

    return best_len, best_start, best_end


def _build_ffill_report(
    *,
    before_na: pd.DataFrame,
    after_na: pd.DataFrame,
) -> dict[str, object]:
    before_any = before_na.any(axis=1).to_numpy()
    filled_mask = before_na & (~after_na)
    filled_any = filled_mask.any(axis=1).to_numpy()

    longest_len, longest_start, longest_end = _longest_true_run(before_any)

    return {
        "rows_forward_filled_position": np.flatnonzero(filled_any).tolist(),
        "rows_forward_filled_count": int(filled_any.sum()),
        "cells_forward_filled_count": int(filled_mask.to_numpy().sum()),
        "rows_with_missing_pose_before_fill_position": np.flatnonzero(before_any).tolist(),
        "rows_with_missing_pose_before_fill_count": int(before_any.sum()),
        "longest_occlusion_stretch_length": int(longest_len),
        "longest_occlusion_stretch_start_position": (
            int(longest_start) if longest_start is not None else None
        ),
        "longest_occlusion_stretch_end_position": (
            int(longest_end) if longest_end is not None else None
        ),
        "filled_cells_per_column": {
            col: int(filled_mask[col].sum()) for col in before_na.columns
        },
    }


@overload
def repair_optitrack_missing_samples(
    df: pd.DataFrame,
    *,
    pose_columns: Sequence[str] = POSE_COLUMNS,
    renormalize_quaternions: bool = False,
    strict: bool = False,
    copy: bool = True,
    return_report: Literal[False] = False,
) -> pd.DataFrame:
    ...


@overload
def repair_optitrack_missing_samples(
    df: pd.DataFrame,
    *,
    pose_columns: Sequence[str] = POSE_COLUMNS,
    renormalize_quaternions: bool = False,
    strict: bool = False,
    copy: bool = True,
    return_report: Literal[True],
) -> tuple[pd.DataFrame, dict[str, object]]:
    ...


def repair_optitrack_missing_samples(
    df: pd.DataFrame,
    *,
    pose_columns: Sequence[str] = POSE_COLUMNS,
    renormalize_quaternions: bool = False,
    strict: bool = False,
    copy: bool = True,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Stage-1 OptiTrack repair:
    Forward-fill missing pose samples (zero-order hold) so downstream
    kinematic computation remains defined at each timestamp.

    If strict=True, raises when missing values remain in pose columns after
    forward-fill (e.g., a run starts with missing pose).
    """
    missing = [c for c in pose_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required pose columns: {missing}")

    out = df.copy(deep=True) if copy else df
    cols = list(pose_columns)
    before_na = out[cols].isna().copy(deep=True)
    out[cols] = out[cols].ffill()
    after_na = out[cols].isna()

    if renormalize_quaternions:
        for w_col, x_col, y_col, z_col in QUATERNION_GROUPS:
            for c in (w_col, x_col, y_col, z_col):
                if c not in out.columns:
                    raise ValueError(f"Missing quaternion column: {c}")

            w = pd.to_numeric(out[w_col], errors="coerce").to_numpy(dtype=np.float64)
            x = pd.to_numeric(out[x_col], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(out[y_col], errors="coerce").to_numpy(dtype=np.float64)
            z = pd.to_numeric(out[z_col], errors="coerce").to_numpy(dtype=np.float64)

            norm = np.sqrt(w * w + x * x + y * y + z * z)
            safe_norm = np.where(norm > 0.0, norm, np.nan)

            out[w_col] = w / safe_norm
            out[x_col] = x / safe_norm
            out[y_col] = y / safe_norm
            out[z_col] = z / safe_norm

    if strict:
        na_mask = out[cols].isna()
        if na_mask.to_numpy().any():
            bad_row = int(np.flatnonzero(na_mask.any(axis=1).to_numpy())[0])
            bad_cols = na_mask.columns[na_mask.iloc[bad_row]].tolist()
            raise ValueError(
                "Pose columns still contain missing values after forward-fill "
                f"at row {bad_row}: {bad_cols}"
            )

    if return_report:
        return out, _build_ffill_report(before_na=before_na, after_na=after_na)

    return out
