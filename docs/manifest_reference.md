# Manifest Reference

This document describes `configs/preprocessing_manifest_all_trials.json`.

Primary parser and validator: `src/preprocessing/export_dataset_hdf5.py`

## Top-Level Structure

```json
{
  "manifest_version": "v1",
  "dataset_root": ".",
  "decisions": {},
  "default_settings": {},
  "runs": []
}
```

## Top-Level Fields

### `manifest_version` (optional)

- String label for your own tracking.
- Not required by the exporter.

### `dataset_root` (optional, default `"."`)

- Base path used to resolve relative CSV, calibration, and trim-config paths.
- Relative paths are resolved against repository root plus `dataset_root`.

### `decisions` (optional object)

Supported fields:

- `scaling_scope`: currently must be `"all_trials"` when set.
- `exclude_restricted_motion_periods`: boolean flag.
- `restricted_motion_trim_config_path`: path to trim JSON, required if
  `exclude_restricted_motion_periods` is `true`.
- `store_intermediate_stages`: accepted as metadata snapshot only.
- `export_format`: accepted as metadata snapshot only.
- `hdf5_backend_tables_installed`: accepted as metadata snapshot only.

### `default_settings` (optional object)

Shared defaults for runs that do not override values.

Supported fields:

- `sample_rate_hz_arduino` (float, default `240.0`)
- `sync` object:
  - `fixed_phi_transform`: `auto | none | invert | abs`
  - `freehand_rotation_direction`: `world_to_body | body_to_world`
  - `max_lag`: integer or `null`
- `filter_downsample` object:
  - `input_sample_rate_hz`
  - `decimation_factor`
  - `decimation_offset`
  - `moving_average_window`
  - `filter_alignment`: `centered | causal`
  - `time_col`
  - `rebase_time`
- `scaling` object:
  - `scope` (metadata only, exporter expects all-trials behavior)
  - `scale_columns` (list of column names)
  - `constant_column_policy` (`zero`)
- `calibration_paths` object:
  - `pressure` path
  - `gyro` path

### `runs` (required array)

Each run entry must include:

- `run_id` (unique, non-empty)
- `arduino_raw_csv`
- `optitrack_raw_csv`
- `sync_mode` (`fixed` or `freehand`)

Optional per-run fields:

- `fixed_phi_transform`
- `freehand_rotation_direction`
- `hdf5_key` (default `runs/<run_id>`, normalized at export time)
- `export_enabled` (default `true`)

## Minimal Example

```json
{
  "dataset_root": ".",
  "default_settings": {
    "sample_rate_hz_arduino": 240.0,
    "sync": {
      "fixed_phi_transform": "auto",
      "freehand_rotation_direction": "world_to_body",
      "max_lag": null
    },
    "filter_downsample": {
      "input_sample_rate_hz": 240.0,
      "decimation_factor": 5,
      "decimation_offset": 0,
      "moving_average_window": 5,
      "filter_alignment": "centered",
      "time_col": "Time",
      "rebase_time": true
    },
    "scaling": {
      "scale_columns": ["pressure", "acc_x", "acc_y", "acc_z", "phi"],
      "constant_column_policy": "zero"
    },
    "calibration_paths": {
      "pressure": "calibration/arduino_pressure_calibration.json",
      "gyro": "calibration/arduino_gyro_calibration.json"
    }
  },
  "runs": [
    {
      "run_id": "0roll_0pitch_tt_1",
      "arduino_raw_csv": "data/arduino_raw/m0roll_0pitch_tt_1.csv",
      "optitrack_raw_csv": "data/optitrack_raw/o0roll_0pitch_tt_1.csv",
      "sync_mode": "fixed",
      "export_enabled": true,
      "hdf5_key": "runs/0roll_0pitch_tt_1"
    }
  ]
}
```

## Validation Notes

- Duplicate `run_id` or normalized `hdf5_key` values fail fast.
- Relative paths are resolved to absolute paths before processing.
- Unknown run IDs passed via CLI `--run-ids` raise an error.
- If no runs remain after filtering by `export_enabled` and `--run-ids`,
  export fails with "No runs selected for export".
