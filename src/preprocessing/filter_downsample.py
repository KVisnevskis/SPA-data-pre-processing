from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd


FilterAlignment = Literal["centered", "causal"]


def _validate_positive_int(value: int, *, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _validate_positive_rate(value: float, *, name: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return float(value)


def _validate_decimation_offset(*, decimation_factor: int, decimation_offset: int) -> int:
    if not isinstance(decimation_offset, int):
        raise ValueError(
            "decimation_offset must be an integer in [0, decimation_factor - 1], "
            f"got {type(decimation_offset).__name__}"
        )
    if decimation_offset < 0 or decimation_offset >= decimation_factor:
        raise ValueError(
            "decimation_offset must satisfy 0 <= decimation_offset < decimation_factor, "
            f"got decimation_offset={decimation_offset}, decimation_factor={decimation_factor}"
        )
    return decimation_offset


def _validate_numeric_finite_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            bad_row = int(np.flatnonzero(~np.isfinite(values))[0])
            raise ValueError(f"Column '{col}' contains non-finite values at row {bad_row}")


def _resolve_filter_columns(
    df: pd.DataFrame,
    *,
    filter_columns: Sequence[str] | None,
    excluded_columns: Sequence[str],
) -> list[str]:
    excluded = set(excluded_columns)

    if filter_columns is None:
        cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded
        ]
    else:
        cols = list(filter_columns)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"filter_columns contains missing columns: {missing}")

        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"filter_columns contains non-numeric columns: {non_numeric}")

    if not cols:
        raise ValueError("No columns selected for moving-average filtering")
    return cols


def apply_moving_average_filter(
    df: pd.DataFrame,
    *,
    window_size: int = 5,
    filter_alignment: FilterAlignment = "centered",
    filter_columns: Sequence[str] | None = None,
    excluded_columns: Sequence[str] = ("sample_index", "t_arduino_s", "Time"),
) -> pd.DataFrame:
    """
    Apply moving-average filtering to selected numeric columns.

    Conventions:
    - `window_size` is the number of input samples per averaging window.
    - `filter_alignment="centered"` uses symmetric windows around each sample.
    - `filter_alignment="causal"` uses trailing windows (current and previous samples).
    - Edge handling uses partial windows (`min_periods=1`) to avoid dropping rows.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    window_size = _validate_positive_int(window_size, name="window_size")
    if filter_alignment not in {"centered", "causal"}:
        raise ValueError(
            "filter_alignment must be one of {'centered', 'causal'}, "
            f"got {filter_alignment}"
        )

    cols = _resolve_filter_columns(
        df,
        filter_columns=filter_columns,
        excluded_columns=excluded_columns,
    )
    _validate_numeric_finite_columns(df, cols)

    out = df.copy(deep=True)
    numeric = out[cols].apply(pd.to_numeric, errors="coerce")
    center = filter_alignment == "centered"
    out[cols] = numeric.rolling(
        window=window_size,
        min_periods=1,
        center=center,
    ).mean()
    return out


def decimate_by_factor(
    df: pd.DataFrame,
    *,
    decimation_factor: int = 5,
    decimation_offset: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Downsample a dataframe by integer decimation.

    Conventions:
    - Keep row indices: `decimation_offset, decimation_offset + factor, ...`
    - Reset index in the returned dataframe.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")

    decimation_factor = _validate_positive_int(decimation_factor, name="decimation_factor")
    decimation_offset = _validate_decimation_offset(
        decimation_factor=decimation_factor,
        decimation_offset=decimation_offset,
    )

    kept_indices = np.arange(decimation_offset, len(df), decimation_factor, dtype=np.int64)
    out = df.iloc[kept_indices].reset_index(drop=True)
    return out, kept_indices


def filter_and_downsample_stage4(
    synced_df: pd.DataFrame,
    *,
    input_sample_rate_hz: float = 240.0,
    decimation_factor: int = 5,
    decimation_offset: int = 0,
    moving_average_window: int = 5,
    filter_alignment: FilterAlignment = "centered",
    filter_columns: Sequence[str] | None = None,
    excluded_filter_columns: Sequence[str] = ("sample_index", "t_arduino_s", "Time"),
    time_col: str = "Time",
    rebase_time: bool = True,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Stage-4 preprocessing: moving-average filtering + integer-factor decimation.

    Defaults target the standard SPA sample-rate conversion:
    - input rate: 240 Hz
    - decimation factor: 5
    - output rate: 48 Hz
    """
    input_sample_rate_hz = _validate_positive_rate(
        input_sample_rate_hz,
        name="input_sample_rate_hz",
    )
    decimation_factor = _validate_positive_int(decimation_factor, name="decimation_factor")
    decimation_offset = _validate_decimation_offset(
        decimation_factor=decimation_factor,
        decimation_offset=decimation_offset,
    )
    moving_average_window = _validate_positive_int(
        moving_average_window,
        name="moving_average_window",
    )

    stage4_df = apply_moving_average_filter(
        synced_df,
        window_size=moving_average_window,
        filter_alignment=filter_alignment,
        filter_columns=filter_columns,
        excluded_columns=excluded_filter_columns,
    )
    downsampled_df, kept_indices = decimate_by_factor(
        stage4_df,
        decimation_factor=decimation_factor,
        decimation_offset=decimation_offset,
    )

    output_sample_rate_hz = input_sample_rate_hz / float(decimation_factor)
    if rebase_time:
        if time_col not in downsampled_df.columns:
            raise ValueError(
                f"time_col '{time_col}' does not exist in dataframe; "
                "set rebase_time=False or provide an existing time column"
            )
        downsampled_df[time_col] = np.arange(
            len(downsampled_df), dtype=np.float64
        ) / output_sample_rate_hz

    if return_info:
        rows_before = int(len(synced_df))
        rows_after = int(len(downsampled_df))
        info = {
            "stage": "stage4",
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_dropped": rows_before - rows_after,
            "decimation_factor": int(decimation_factor),
            "decimation_offset": int(decimation_offset),
            "decimation_anchor_policy": (
                "keep index decimation_offset + n*decimation_factor"
            ),
            "kept_first_input_row": int(kept_indices[0]) if rows_after > 0 else None,
            "kept_last_input_row": int(kept_indices[-1]) if rows_after > 0 else None,
            "input_sample_rate_hz": float(input_sample_rate_hz),
            "output_sample_rate_hz": float(output_sample_rate_hz),
            "moving_average_window": int(moving_average_window),
            "filter_alignment": filter_alignment,
            "edge_handling": "partial_window_mean(min_periods=1)",
            "filtered_columns": list(
                _resolve_filter_columns(
                    synced_df,
                    filter_columns=filter_columns,
                    excluded_columns=excluded_filter_columns,
                )
            ),
            "time_col": time_col,
            "time_rebased": bool(rebase_time),
        }
        return downsampled_df, info

    return downsampled_df
