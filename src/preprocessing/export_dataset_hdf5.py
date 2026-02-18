from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from preprocessing.arduino_raw import load_arduino_raw_csv
from preprocessing.filter_downsample import FilterAlignment, filter_and_downsample
from preprocessing.global_scaling import (
    ConstantColumnPolicy,
    DEFAULT_SCALE_COLUMNS,
    fit_and_apply_global_min_max_scaling,
)
from preprocessing.optitrack_compute_angle import compute_optitrack_relative_features
from preprocessing.optitrack_raw import (
    load_optitrack_raw_csv,
    repair_optitrack_missing_samples,
)
from preprocessing.time_sync import (
    FixedPhiTransform,
    RotationDirection,
    SyncMode,
    synchronize_fixed_orientation,
    synchronize_freehand,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    hdf5_key: str
    requested_hdf5_key: str
    arduino_raw_csv: Path
    optitrack_raw_csv: Path
    sync_mode: SyncMode
    fixed_phi_transform: FixedPhiTransform
    freehand_rotation_direction: RotationDirection
    export_enabled: bool


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _to_str(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        out = value.strip()
        if out:
            return out
    raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")


def _coerce_sync_mode(value: Any, *, field_name: str) -> SyncMode:
    v = str(value).strip().lower()
    if v == "fixed":
        return "fixed"
    if v == "freehand":
        return "freehand"
    raise ValueError(f"{field_name} must be 'fixed' or 'freehand', got {value!r}")


def _coerce_fixed_phi_transform(value: Any, *, field_name: str) -> FixedPhiTransform:
    v = str(value).strip().lower()
    if v == "auto":
        return "auto"
    if v == "none":
        return "none"
    if v == "invert":
        return "invert"
    if v == "abs":
        return "abs"
    raise ValueError(
        f"{field_name} must be one of 'auto', 'none', 'invert', 'abs', got {value!r}"
    )


def _coerce_rotation_direction(value: Any, *, field_name: str) -> RotationDirection:
    v = str(value).strip()
    if v == "world_to_body":
        return "world_to_body"
    if v == "body_to_world":
        return "body_to_world"
    raise ValueError(
        f"{field_name} must be one of 'world_to_body', 'body_to_world', got {value!r}"
    )


def _coerce_filter_alignment(value: Any, *, field_name: str) -> FilterAlignment:
    v = str(value).strip().lower()
    if v == "centered":
        return "centered"
    if v == "causal":
        return "causal"
    raise ValueError(f"{field_name} must be 'centered' or 'causal', got {value!r}")


def _coerce_constant_column_policy(
    value: Any, *, field_name: str
) -> ConstantColumnPolicy:
    v = str(value).strip().lower()
    if v == "zero":
        return "zero"
    raise ValueError(f"{field_name} must be 'zero', got {value!r}")


def _coerce_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int or null, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an int or null, got {value!r}") from exc


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be int-like, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be int-like, got {value!r}") from exc


def _coerce_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be float-like, got bool")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be float-like, got {value!r}") from exc


def _coerce_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{i}] must be a string, got {type(item).__name__}")
        out.append(item)
    if not out:
        raise ValueError(f"{field_name} must not be empty")
    return out


def _normalize_hdf5_key(key: str) -> str:
    k = key.strip()
    if not k:
        raise ValueError("hdf5_key must not be empty")
    k = k if k.startswith("/") else f"/{k}"
    parts = [p for p in k.split("/") if p]
    if not parts:
        raise ValueError("hdf5_key must contain at least one non-empty path segment")

    safe_parts: list[str] = []
    for part in parts:
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", part)
        if not safe:
            safe = "node"
        if not (safe[0].isalpha() or safe[0] == "_"):
            safe = f"run_{safe}"
        safe_parts.append(safe)
    return "/" + "/".join(safe_parts)


def _resolve_dataset_root(manifest: dict[str, Any]) -> Path:
    dataset_root_value = manifest.get("dataset_root", ".")
    root_path = Path(dataset_root_value)
    if root_path.is_absolute():
        return root_path
    return (REPO_ROOT / root_path).resolve()


def _resolve_path(dataset_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (dataset_root / p).resolve()


def load_manifest_json(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest JSON root must be an object")
    if "runs" not in payload or not isinstance(payload["runs"], list):
        raise ValueError("Manifest JSON must contain a 'runs' array")
    return payload


def build_run_plan(
    manifest: dict[str, Any],
    *,
    include_run_ids: Sequence[str] | None = None,
) -> list[RunPlan]:
    dataset_root = _resolve_dataset_root(manifest)
    defaults = manifest.get("default_settings", {})
    sync_defaults: Mapping[str, Any] = defaults.get("sync", {}) if isinstance(defaults, dict) else {}

    include_set = set(include_run_ids) if include_run_ids is not None else None
    plans: list[RunPlan] = []
    seen_run_ids: set[str] = set()
    seen_keys: set[str] = set()

    for raw_run in manifest["runs"]:
        if not isinstance(raw_run, dict):
            raise ValueError(f"Each run entry must be an object, got {type(raw_run).__name__}")

        run_id = str(raw_run.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("Each run must define non-empty 'run_id'")
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate run_id in manifest: {run_id}")
        seen_run_ids.add(run_id)

        if include_set is not None and run_id not in include_set:
            continue

        arduino_raw_csv = str(raw_run.get("arduino_raw_csv", "")).strip()
        optitrack_raw_csv = str(raw_run.get("optitrack_raw_csv", "")).strip()
        if not arduino_raw_csv or not optitrack_raw_csv:
            raise ValueError(
                f"Run '{run_id}' must define both 'arduino_raw_csv' and 'optitrack_raw_csv'"
            )

        sync_mode = _coerce_sync_mode(raw_run.get("sync_mode", ""), field_name=f"{run_id}.sync_mode")

        fixed_phi_raw = raw_run.get(
            "fixed_phi_transform",
            sync_defaults.get("fixed_phi_transform", "auto"),
        )
        freehand_rot_raw = raw_run.get(
            "freehand_rotation_direction",
            sync_defaults.get("freehand_rotation_direction", "world_to_body"),
        )
        fixed_phi_transform = _coerce_fixed_phi_transform(
            fixed_phi_raw if fixed_phi_raw not in (None, "") else "auto",
            field_name=f"{run_id}.fixed_phi_transform",
        )
        freehand_rotation_direction = _coerce_rotation_direction(
            freehand_rot_raw if freehand_rot_raw not in (None, "") else "world_to_body",
            field_name=f"{run_id}.freehand_rotation_direction",
        )

        requested_hdf5_key = str(raw_run.get("hdf5_key", f"runs/{run_id}"))
        hdf5_key = _normalize_hdf5_key(requested_hdf5_key)
        if hdf5_key in seen_keys:
            raise ValueError(f"Duplicate hdf5_key in manifest: {hdf5_key}")
        seen_keys.add(hdf5_key)

        export_enabled = _parse_bool(raw_run.get("export_enabled"), default=True)

        plans.append(
            RunPlan(
                run_id=run_id,
                hdf5_key=hdf5_key,
                requested_hdf5_key=requested_hdf5_key,
                arduino_raw_csv=_resolve_path(dataset_root, arduino_raw_csv),
                optitrack_raw_csv=_resolve_path(dataset_root, optitrack_raw_csv),
                sync_mode=sync_mode,
                fixed_phi_transform=fixed_phi_transform,
                freehand_rotation_direction=freehand_rotation_direction,
                export_enabled=export_enabled,
            )
        )

    if include_set is not None:
        missing = sorted(include_set - {p.run_id for p in plans})
        if missing:
            raise ValueError(f"Requested run_ids not found in manifest: {missing}")

    return [p for p in plans if p.export_enabled]


def preview_run_plan(
    manifest_path: str | Path,
    *,
    include_run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest_json(manifest_path)
    plans = build_run_plan(manifest, include_run_ids=include_run_ids)
    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "n_runs": len(plans),
        "run_ids": [p.run_id for p in plans],
        "sync_mode_counts": {
            "fixed": sum(1 for p in plans if p.sync_mode == "fixed"),
            "freehand": sum(1 for p in plans if p.sync_mode == "freehand"),
        },
    }


def _read_calibration_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration JSON must be an object: {path}")
    return payload


def export_dataset_hdf5_from_manifest(
    manifest_path: str | Path,
    output_hdf5_path: str | Path,
    *,
    include_run_ids: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    output_hdf5_path = Path(output_hdf5_path).resolve()

    manifest = load_manifest_json(manifest_path)
    run_plan = build_run_plan(manifest, include_run_ids=include_run_ids)
    if not run_plan:
        raise ValueError("No runs selected for export")

    defaults: Mapping[str, Any] = manifest.get("default_settings", {}) if isinstance(manifest, dict) else {}
    decisions: Mapping[str, Any] = manifest.get("decisions", {}) if isinstance(manifest, dict) else {}
    if decisions and decisions.get("scaling_scope") not in {None, "all_trials"}:
        raise ValueError(
            "This exporter currently supports only scaling_scope='all_trials'"
        )

    sample_rate_hz_arduino = _coerce_float(
        defaults.get("sample_rate_hz_arduino", 240.0), field_name="default_settings.sample_rate_hz_arduino"
    )
    sync_defaults: Mapping[str, Any] = defaults.get("sync", {}) if isinstance(defaults, dict) else {}
    stage4_defaults: Mapping[str, Any] = (
        defaults.get("filter_downsample", {}) if isinstance(defaults, dict) else {}
    )
    scaling_defaults: Mapping[str, Any] = defaults.get("scaling", {}) if isinstance(defaults, dict) else {}
    calibration_defaults: Mapping[str, Any] = (
        defaults.get("calibration_paths", {}) if isinstance(defaults, dict) else {}
    )

    default_fixed_phi_transform = _coerce_fixed_phi_transform(
        sync_defaults.get("fixed_phi_transform", "auto"),
        field_name="default_settings.sync.fixed_phi_transform",
    )
    default_freehand_rotation_direction = _coerce_rotation_direction(
        sync_defaults.get("freehand_rotation_direction", "world_to_body"),
        field_name="default_settings.sync.freehand_rotation_direction",
    )
    max_lag = _coerce_optional_int(sync_defaults.get("max_lag"), field_name="default_settings.sync.max_lag")
    filter_alignment = _coerce_filter_alignment(
        stage4_defaults.get("filter_alignment", "centered"),
        field_name="default_settings.filter_downsample.filter_alignment",
    )
    constant_column_policy = _coerce_constant_column_policy(
        scaling_defaults.get("constant_column_policy", "zero"),
        field_name="default_settings.scaling.constant_column_policy",
    )

    pressure_calib_path = (
        _resolve_path(_resolve_dataset_root(manifest), _to_str(calibration_defaults["pressure"], field_name="default_settings.calibration_paths.pressure"))
        if "pressure" in calibration_defaults
        else None
    )
    gyro_calib_path = (
        _resolve_path(_resolve_dataset_root(manifest), _to_str(calibration_defaults["gyro"], field_name="default_settings.calibration_paths.gyro"))
        if "gyro" in calibration_defaults
        else None
    )
    pressure_calib_payload = _read_calibration_payload(pressure_calib_path)
    gyro_calib_payload = _read_calibration_payload(gyro_calib_path)

    run_stage4_tables: dict[str, pd.DataFrame] = {}
    run_meta_rows: list[dict[str, Any]] = []
    run_log_rows: list[dict[str, Any]] = []

    for plan in run_plan:
        arduino_df = load_arduino_raw_csv(
            plan.arduino_raw_csv,
            sample_rate_hz=sample_rate_hz_arduino,
            accel_to_mps2=True,
            pressure_calib_json_path=pressure_calib_path,
            gyro_calib_json_path=gyro_calib_path,
        )

        opt_raw_df = load_optitrack_raw_csv(plan.optitrack_raw_csv)
        opt_repaired_df, repair_report = repair_optitrack_missing_samples(
            opt_raw_df,
            renormalize_quaternions=False,
            strict=False,
            return_report=True,
        )
        opt_features_df = compute_optitrack_relative_features(
            opt_repaired_df,
            strict=False,
        )

        if plan.sync_mode == "fixed":
            sync_result = synchronize_fixed_orientation(
                arduino_df,
                opt_features_df,
                phi_transform=plan.fixed_phi_transform or default_fixed_phi_transform,
                max_lag=max_lag,
                return_info=True,
            )
            if not isinstance(sync_result, tuple):
                raise RuntimeError("Expected synchronize_fixed_orientation(..., return_info=True) to return tuple")
            synced_df, sync_info = sync_result
        else:
            sync_result = synchronize_freehand(
                arduino_df,
                opt_features_df,
                rotation_direction=plan.freehand_rotation_direction or default_freehand_rotation_direction,
                max_lag=max_lag,
                return_info=True,
            )
            if not isinstance(sync_result, tuple):
                raise RuntimeError("Expected synchronize_freehand(..., return_info=True) to return tuple")
            synced_df, sync_info = sync_result

        stage4_result = filter_and_downsample(
            synced_df,
            input_sample_rate_hz=_coerce_float(
                stage4_defaults.get("input_sample_rate_hz", 240.0),
                field_name="default_settings.filter_downsample.input_sample_rate_hz",
            ),
            decimation_factor=_coerce_int(
                stage4_defaults.get("decimation_factor", 5),
                field_name="default_settings.filter_downsample.decimation_factor",
            ),
            decimation_offset=_coerce_int(
                stage4_defaults.get("decimation_offset", 0),
                field_name="default_settings.filter_downsample.decimation_offset",
            ),
            moving_average_window=_coerce_int(
                stage4_defaults.get("moving_average_window", 5),
                field_name="default_settings.filter_downsample.moving_average_window",
            ),
            filter_alignment=filter_alignment,
            time_col=str(stage4_defaults.get("time_col", "Time")),
            rebase_time=_parse_bool(stage4_defaults.get("rebase_time"), default=True),
            return_info=True,
        )
        if not isinstance(stage4_result, tuple):
            raise RuntimeError("Expected filter_and_downsample(..., return_info=True) to return tuple")
        downsampled_df, stage4_info = stage4_result

        run_stage4_tables[plan.run_id] = downsampled_df

        run_meta_rows.append(
            {
                "run_id": plan.run_id,
                "hdf5_key": plan.hdf5_key,
                "requested_hdf5_key": plan.requested_hdf5_key,
                "sync_mode": plan.sync_mode,
                "arduino_raw_csv": str(plan.arduino_raw_csv),
                "optitrack_raw_csv": str(plan.optitrack_raw_csv),
                "rows_final": int(len(downsampled_df)),  # pre-scaling; scaling preserves row count
            }
        )
        run_log_rows.append(
            {
                "run_id": plan.run_id,
                "samples_shifted_arduino": _coerce_int(
                    sync_info.get("sample_shift_arduino"),
                    field_name=f"{plan.run_id}.sync_info.sample_shift_arduino",
                ),
                "sync_lag_samples": _coerce_int(
                    sync_info.get("lag_samples"),
                    field_name=f"{plan.run_id}.sync_info.lag_samples",
                ),
                "sync_variable": str(sync_info.get("sync_variable", "")),
                "sync_max_cross_correlation": _coerce_float(
                    sync_info.get("max_cross_correlation"),
                    field_name=f"{plan.run_id}.sync_info.max_cross_correlation",
                ),
                "rows_forward_filled_count": _coerce_int(
                    repair_report.get("rows_forward_filled_count"),
                    field_name=f"{plan.run_id}.repair_report.rows_forward_filled_count",
                ),
                "cells_forward_filled_count": _coerce_int(
                    repair_report.get("cells_forward_filled_count"),
                    field_name=f"{plan.run_id}.repair_report.cells_forward_filled_count",
                ),
                "longest_occlusion_stretch_length": int(
                    _coerce_int(
                        repair_report.get("longest_occlusion_stretch_length"),
                        field_name=f"{plan.run_id}.repair_report.longest_occlusion_stretch_length",
                    )
                ),
                "rows_after_sync_trim": _coerce_int(
                    sync_info.get("rows_after_trim"),
                    field_name=f"{plan.run_id}.sync_info.rows_after_trim",
                ),
                "rows_after_downsample": _coerce_int(
                    stage4_info.get("rows_after"),
                    field_name=f"{plan.run_id}.stage4_info.rows_after",
                ),
                "sync_info_json": _json_dumps(sync_info),
                "repair_report_json": _json_dumps(repair_report),
                "downsample_info_json": _json_dumps(stage4_info),
            }
        )

    scale_columns = _coerce_string_list(
        scaling_defaults.get("scale_columns", list(DEFAULT_SCALE_COLUMNS)),
        field_name="default_settings.scaling.scale_columns",
    )

    scaled_runs, scaler_parameters, scaling_info = fit_and_apply_global_min_max_scaling(
        run_stage4_tables,
        scale_columns=scale_columns,
        constant_column_policy=constant_column_policy,
        return_info=True,
    )

    scaler_rows = [
        {
            "column": col,
            "min": float(params["min"]),
            "max": float(params["max"]),
            "range": float(params["range"]),
            "is_constant": bool(params["is_constant"]),
        }
        for col, params in scaler_parameters.items()
    ]
    scaling_runs_obj = scaling_info.get("runs")
    if not isinstance(scaling_runs_obj, dict):
        raise ValueError("scaling_info['runs'] must be a dict")
    run_scaling_rows = [
        {
            "run_id": run_id,
            "scaling_info_json": _json_dumps(info),
        }
        for run_id, info in scaling_runs_obj.items()
    ]

    if output_hdf5_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_hdf5_path}. Pass overwrite=True to replace it."
        )
    output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    if output_hdf5_path.exists():
        output_hdf5_path.unlink()

    runs_df = pd.DataFrame(run_meta_rows)
    run_logs_df = pd.DataFrame(run_log_rows)
    run_scaling_df = pd.DataFrame(run_scaling_rows)
    scaler_df = pd.DataFrame(scaler_rows)
    calibration_df = pd.DataFrame(
        [
            {
                "calibration_type": "pressure",
                "path": str(pressure_calib_path) if pressure_calib_path is not None else "",
                "payload_json": _json_dumps(pressure_calib_payload)
                if pressure_calib_payload is not None
                else "",
            },
            {
                "calibration_type": "gyro",
                "path": str(gyro_calib_path) if gyro_calib_path is not None else "",
                "payload_json": _json_dumps(gyro_calib_payload)
                if gyro_calib_payload is not None
                else "",
            },
        ]
    )
    export_settings_df = pd.DataFrame(
        [
            {
                "manifest_path": str(manifest_path),
                "output_hdf5_path": str(output_hdf5_path),
                "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "decisions_json": _json_dumps(decisions),
                "default_settings_json": _json_dumps(defaults),
                "scaling_info_json": _json_dumps(scaling_info),
            }
        ]
    )

    with pd.HDFStore(output_hdf5_path, mode="w") as store:
        for plan in run_plan:
            store.put(plan.hdf5_key, scaled_runs[plan.run_id], format="fixed")
        store.put("/meta/runs", runs_df, format="fixed")
        store.put("/meta/run_logs", run_logs_df, format="fixed")
        store.put("/meta/run_scaling", run_scaling_df, format="fixed")
        store.put("/meta/scaler_parameters", scaler_df, format="fixed")
        store.put("/meta/calibration", calibration_df, format="fixed")
        store.put("/meta/export_settings", export_settings_df, format="fixed")

    return {
        "output_hdf5_path": str(output_hdf5_path),
        "runs_exported": len(run_plan),
        "run_ids": [p.run_id for p in run_plan],
        "scale_columns": list(scale_columns),
        "metadata_keys": [
            "/meta/runs",
            "/meta/run_logs",
            "/meta/run_scaling",
            "/meta/scaler_parameters",
            "/meta/calibration",
            "/meta/export_settings",
        ],
    }
