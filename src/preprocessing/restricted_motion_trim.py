from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class StreamTrimBounds:
    """
    Index bounds for one stream.

    Semantics:
    - trim_up_to: zero-based index of first sample to keep
    - trim_after: zero-based index of last sample to keep
    """

    trim_up_to: int | None
    trim_after: int | None


@dataclass(frozen=True)
class RunTrimConfig:
    run_id: str
    arduino: StreamTrimBounds
    optitrack: StreamTrimBounds
    notes: str


def _coerce_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int >= 0 or null, got bool")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an int >= 0 or null, got {value!r}") from exc
    if out < 0:
        raise ValueError(f"{field_name} must be >= 0, got {out}")
    return out


def _coerce_stream_bounds(value: Any, *, field_name: str) -> StreamTrimBounds:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")

    trim_up_to = _coerce_optional_non_negative_int(
        value.get("trim_up_to"),
        field_name=f"{field_name}.trim_up_to",
    )
    trim_after = _coerce_optional_non_negative_int(
        value.get("trim_after"),
        field_name=f"{field_name}.trim_after",
    )
    if trim_up_to is not None and trim_after is not None and trim_after < trim_up_to:
        raise ValueError(
            f"{field_name}.trim_after must be >= {field_name}.trim_up_to when both are set"
        )
    return StreamTrimBounds(trim_up_to=trim_up_to, trim_after=trim_after)


def load_restricted_motion_trim_config(
    path: str | Path,
) -> dict[str, RunTrimConfig]:
    cfg_path = Path(path)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Restricted-motion trim config root must be an object")

    runs_value = payload.get("runs")
    if not isinstance(runs_value, list):
        raise ValueError("Restricted-motion trim config must contain a 'runs' array")

    out: dict[str, RunTrimConfig] = {}
    for i, item in enumerate(runs_value):
        if not isinstance(item, dict):
            raise ValueError(f"runs[{i}] must be an object")

        run_id = str(item.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"runs[{i}].run_id must be a non-empty string")
        if run_id in out:
            raise ValueError(f"Duplicate run_id in restricted-motion trim config: {run_id}")

        arduino_bounds = _coerce_stream_bounds(item.get("arduino"), field_name=f"runs[{i}].arduino")
        optitrack_bounds = _coerce_stream_bounds(
            item.get("optitrack"),
            field_name=f"runs[{i}].optitrack",
        )
        notes = str(item.get("notes", ""))

        out[run_id] = RunTrimConfig(
            run_id=run_id,
            arduino=arduino_bounds,
            optitrack=optitrack_bounds,
            notes=notes,
        )

    return out


def _trim_single_dataframe(
    df: pd.DataFrame,
    *,
    trim_up_to: int | None,
    trim_after: int | None,
    label: str,
) -> tuple[pd.DataFrame, dict[str, int | None]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")

    n_rows = int(len(df))
    if n_rows == 0:
        raise ValueError(f"{label} is empty; cannot apply restricted-motion trim")

    start = trim_up_to if trim_up_to is not None else 0
    stop_inclusive = trim_after if trim_after is not None else n_rows - 1

    if start < 0:
        raise ValueError(f"{label}.trim_up_to must be >= 0")
    if start >= n_rows:
        raise ValueError(
            f"{label}.trim_up_to={start} is out of bounds for {n_rows} rows"
        )
    if stop_inclusive < start:
        raise ValueError(
            f"{label}.trim_after must be >= trim_up_to (got {stop_inclusive} < {start})"
        )

    stop_inclusive = min(stop_inclusive, n_rows - 1)
    trimmed = df.iloc[start : stop_inclusive + 1].reset_index(drop=True)

    if len(trimmed) == 0:
        raise ValueError(f"{label} trim produced an empty dataframe")

    info: dict[str, int | None] = {
        "rows_before": n_rows,
        "rows_after": int(len(trimmed)),
        "trim_up_to_applied": int(start),
        "trim_after_applied": int(stop_inclusive),
        "rows_trimmed": int(n_rows - len(trimmed)),
    }
    return trimmed, info


def apply_restricted_motion_trim_for_run(
    *,
    run_id: str,
    arduino_df: pd.DataFrame,
    optitrack_df: pd.DataFrame,
    trim_config_by_run: Mapping[str, RunTrimConfig] | None = None,
    return_info: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    run_key = str(run_id).strip()
    if not run_key:
        raise ValueError("run_id must be non-empty")

    if not trim_config_by_run or run_key not in trim_config_by_run:
        arduino_out = arduino_df.reset_index(drop=True)
        optitrack_out = optitrack_df.reset_index(drop=True)
        if return_info:
            return arduino_out, optitrack_out, {
                "applied": False,
                "reason": "no_trim_entry_for_run",
                "run_id": run_key,
            }
        return arduino_out, optitrack_out

    cfg = trim_config_by_run[run_key]
    arduino_out, arduino_info = _trim_single_dataframe(
        arduino_df,
        trim_up_to=cfg.arduino.trim_up_to,
        trim_after=cfg.arduino.trim_after,
        label=f"{run_key}.arduino",
    )
    optitrack_out, optitrack_info = _trim_single_dataframe(
        optitrack_df,
        trim_up_to=cfg.optitrack.trim_up_to,
        trim_after=cfg.optitrack.trim_after,
        label=f"{run_key}.optitrack",
    )

    if return_info:
        return arduino_out, optitrack_out, {
            "applied": True,
            "run_id": run_key,
            "notes": cfg.notes,
            "arduino": arduino_info,
            "optitrack": optitrack_info,
        }
    return arduino_out, optitrack_out

