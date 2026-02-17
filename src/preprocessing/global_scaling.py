from __future__ import annotations

from typing import Literal, Mapping, Sequence, overload

import numpy as np
import pandas as pd


ConstantColumnPolicy = Literal["zero"]

DEFAULT_SCALE_COLUMNS: tuple[str, ...] = (
    "pressure",
    "acc_x",
    "acc_y",
    "acc_z",
    "phi",
)


def _validate_scale_columns(scale_columns: Sequence[str]) -> list[str]:
    cols = list(scale_columns)
    if not cols:
        raise ValueError("scale_columns must contain at least one column")
    duplicate_cols = sorted({c for c in cols if cols.count(c) > 1})
    if duplicate_cols:
        raise ValueError(f"scale_columns contains duplicates: {duplicate_cols}")
    return cols


def _validate_run_tables(run_tables: Mapping[str, pd.DataFrame]) -> list[str]:
    run_ids = list(run_tables.keys())
    if not run_ids:
        raise ValueError("run_tables must contain at least one run")
    for run_id, run_df in run_tables.items():
        if run_df.empty:
            raise ValueError(f"Run '{run_id}' is empty")
    return run_ids


def _to_finite_float_array(*, run_id: str, df: pd.DataFrame, col: str) -> np.ndarray:
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad_row = int(np.flatnonzero(~np.isfinite(values))[0])
        raise ValueError(
            f"Run '{run_id}' column '{col}' contains non-finite values at row {bad_row}"
        )
    return values


def fit_global_min_max_scaler(
    run_tables: Mapping[str, pd.DataFrame],
    *,
    scale_columns: Sequence[str] = DEFAULT_SCALE_COLUMNS,
) -> dict[str, dict[str, float | bool]]:
    """
    Fit global min/max scaling parameters across all runs for selected columns.
    """
    run_ids = _validate_run_tables(run_tables)
    cols = _validate_scale_columns(scale_columns)

    for run_id in run_ids:
        missing = [c for c in cols if c not in run_tables[run_id].columns]
        if missing:
            raise ValueError(f"Run '{run_id}' is missing columns required for scaling: {missing}")

    scaler: dict[str, dict[str, float | bool]] = {}
    for col in cols:
        global_min = np.inf
        global_max = -np.inf
        for run_id in run_ids:
            values = _to_finite_float_array(run_id=run_id, df=run_tables[run_id], col=col)
            col_min = float(np.min(values))
            col_max = float(np.max(values))
            global_min = min(global_min, col_min)
            global_max = max(global_max, col_max)

        if not np.isfinite(global_min) or not np.isfinite(global_max):
            raise ValueError(
                f"Could not compute finite global bounds for column '{col}': "
                f"min={global_min}, max={global_max}"
            )

        col_range = global_max - global_min
        if col_range < 0.0:
            raise ValueError(
                f"Invalid global bounds for column '{col}': min={global_min}, max={global_max}"
            )

        scaler[col] = {
            "min": float(global_min),
            "max": float(global_max),
            "range": float(col_range),
            "is_constant": bool(col_range == 0.0),
        }

    return scaler


@overload
def apply_global_min_max_scaler(
    df: pd.DataFrame,
    *,
    scaler_parameters: Mapping[str, Mapping[str, float | bool]],
    scale_columns: Sequence[str] | None = None,
    constant_column_policy: ConstantColumnPolicy = "zero",
    copy: bool = True,
    return_info: Literal[False] = False,
) -> pd.DataFrame:
    ...


@overload
def apply_global_min_max_scaler(
    df: pd.DataFrame,
    *,
    scaler_parameters: Mapping[str, Mapping[str, float | bool]],
    scale_columns: Sequence[str] | None = None,
    constant_column_policy: ConstantColumnPolicy = "zero",
    copy: bool = True,
    return_info: Literal[True],
) -> tuple[pd.DataFrame, dict[str, object]]:
    ...


def apply_global_min_max_scaler(
    df: pd.DataFrame,
    *,
    scaler_parameters: Mapping[str, Mapping[str, float | bool]],
    scale_columns: Sequence[str] | None = None,
    constant_column_policy: ConstantColumnPolicy = "zero",
    copy: bool = True,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Apply global min/max scaling to selected columns, mapping to [-1, 1].
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")
    if constant_column_policy != "zero":
        raise ValueError(
            "Unsupported constant_column_policy: "
            f"{constant_column_policy}. Supported: 'zero'"
        )

    cols = list(scaler_parameters.keys()) if scale_columns is None else _validate_scale_columns(scale_columns)
    missing_scaler = [c for c in cols if c not in scaler_parameters]
    if missing_scaler:
        raise ValueError(f"Scaler parameters missing columns: {missing_scaler}")

    missing_df = [c for c in cols if c not in df.columns]
    if missing_df:
        raise ValueError(f"Input dataframe missing columns required for scaling: {missing_df}")

    out = df.copy(deep=True) if copy else df
    diagnostics: dict[str, object] = {}

    for col in cols:
        values = _to_finite_float_array(run_id="input", df=out, col=col)
        params = scaler_parameters[col]
        col_min = float(params["min"])
        col_max = float(params["max"])
        col_range = float(params.get("range", col_max - col_min))

        if not np.isfinite(col_min) or not np.isfinite(col_max) or not np.isfinite(col_range):
            raise ValueError(
                f"Scaler parameters for column '{col}' contain non-finite values: {params}"
            )
        if col_range < 0.0:
            raise ValueError(f"Scaler range for column '{col}' must be >= 0, got {col_range}")

        if col_range == 0.0:
            scaled = np.zeros_like(values)
        else:
            scaled = ((values - col_min) / col_range) * 2.0 - 1.0

        if not np.isfinite(scaled).all():
            bad_row = int(np.flatnonzero(~np.isfinite(scaled))[0])
            raise ValueError(f"Scaled column '{col}' contains non-finite values at row {bad_row}")

        out[col] = scaled
        diagnostics[col] = {
            "min_before": float(np.min(values)),
            "max_before": float(np.max(values)),
            "min_after": float(np.min(scaled)),
            "max_after": float(np.max(scaled)),
            "scaler_min": col_min,
            "scaler_max": col_max,
            "scaler_range": col_range,
            "is_constant_scaler_column": bool(col_range == 0.0),
        }

    if return_info:
        info = {
            "rows": int(len(out)),
            "scaled_columns": cols,
            "constant_column_policy": constant_column_policy,
            "columns": diagnostics,
        }
        return out, info
    return out


@overload
def fit_and_apply_global_min_max_scaling(
    run_tables: Mapping[str, pd.DataFrame],
    *,
    scale_columns: Sequence[str] = DEFAULT_SCALE_COLUMNS,
    constant_column_policy: ConstantColumnPolicy = "zero",
    return_info: Literal[False] = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float | bool]]]:
    ...


@overload
def fit_and_apply_global_min_max_scaling(
    run_tables: Mapping[str, pd.DataFrame],
    *,
    scale_columns: Sequence[str] = DEFAULT_SCALE_COLUMNS,
    constant_column_policy: ConstantColumnPolicy = "zero",
    return_info: Literal[True],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, float | bool]],
    dict[str, object],
]:
    ...


def fit_and_apply_global_min_max_scaling(
    run_tables: Mapping[str, pd.DataFrame],
    *,
    scale_columns: Sequence[str] = DEFAULT_SCALE_COLUMNS,
    constant_column_policy: ConstantColumnPolicy = "zero",
    return_info: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float | bool]]] | tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, float | bool]],
    dict[str, object],
]:
    """
    Fit global min/max parameters across runs, then apply to each run.
    """
    run_ids = _validate_run_tables(run_tables)
    cols = _validate_scale_columns(scale_columns)
    scaler_parameters = fit_global_min_max_scaler(run_tables, scale_columns=cols)

    scaled_runs: dict[str, pd.DataFrame] = {}
    run_diagnostics: dict[str, object] = {}
    for run_id in run_ids:
        scaled_df, info = apply_global_min_max_scaler(
            run_tables[run_id],
            scaler_parameters=scaler_parameters,
            scale_columns=cols,
            constant_column_policy=constant_column_policy,
            copy=True,
            return_info=True,
        )
        scaled_runs[run_id] = scaled_df
        run_diagnostics[run_id] = info

    if return_info:
        info = {
            "scale_columns": cols,
            "constant_column_policy": constant_column_policy,
            "scaler_parameters": scaler_parameters,
            "runs": run_diagnostics,
        }
        return scaled_runs, scaler_parameters, info

    return scaled_runs, scaler_parameters
