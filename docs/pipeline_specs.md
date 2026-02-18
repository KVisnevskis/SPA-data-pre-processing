# Data Pre-processing Pipeline Specification

## Purpose
Transform raw Arduino sensor logs and raw OptiTrack motion-capture logs into one HDF5 dataset containing synchronized, filtered, downsampled, and globally scaled per-run tables plus reproducibility metadata.

## Inputs (per run)
1. Arduino raw CSV
- Pressure ADC and flex ADC
- Accelerometer (3-axis)
- Gyroscope (3-axis)

2. OptiTrack raw CSV
- Base rigid-body quaternion and position
- Tip rigid-body quaternion and position

## Final Output (single HDF5)

Per-run tables:
- One table per run at `/runs/<normalized_run_key>`
- Each run table is final stage output only (post-sync, post-downsample, post-scaling)

Metadata tables:
- `/meta/runs`
- `/meta/run_logs`
- `/meta/run_scaling`
- `/meta/scaler_parameters`
- `/meta/calibration`
- `/meta/export_settings`

Not exported:
- Intermediate stage tables (`arduino_raw_table`, repaired OptiTrack table, pre-scaling stage tables)

---

## Pipeline stages

### Stage 0 - Ingest and calibrate

What it does:
- Load raw Arduino and OptiTrack CSVs.
- Decode Arduino bytes into typed sensor channels.
- Apply pressure calibration from JSON to produce `pressure` (Pa).
- Apply gyro bias calibration from JSON to produce corrected `gyr_x/y/z`.
- Standardize column names and typed run tables.

Output:
- `arduino_raw_table`
- `optitrack_raw_table`

### Stage 1 - Repair missing OptiTrack samples

What it does:
- Forward-fill missing OptiTrack samples (zero-order hold).
- Keep diagnostics on filled rows/cells and occlusion stretch lengths.

Output:
- `optitrack_repaired_table`
- `repair_report`

### Stage 2 - Compute OptiTrack features

What it does:
- Compute base-to-tip relative orientation.
- Convert to ZYX Euler (`phi`, `theta`, `psi`).
- Compute relative displacement (`dx`, `dy`, `dz`).

Output:
- `optitrack_features_table`

### Stage 3 - Synchronize and overlap-trim

What it does:
- Fixed runs: sync Arduino pressure to transformed `phi` (`none` / `invert` / `abs` / `auto`).
- Freehand runs: sync Arduino accelerometer to virtual accelerometer from OptiTrack orientation.
- Apply sample shift and trim to overlapping valid region.

Recorded diagnostics:
- Applied Arduino sample shift
- Lag at max cross-correlation
- Sync variable and max cross-correlation value
- Rows after sync trim

Output:
- `synced_table`
- `sync_info`

### Stage 4 - Filter and downsample

What it does:
- Moving-average filter.
- Integer decimation (default `240 Hz -> 48 Hz`, factor `5`, offset `0`).
- Optional time rebasing (default enabled).

Recorded diagnostics:
- Row counts before/after
- Effective sample rates and decimation settings

Output:
- `filtered_downsampled_table`
- `stage4_info`

### Stage 5 - Global scaling

What it does:
- Fit one global min-max scaler across all selected runs.
- Apply to configured columns (default: `pressure`, `acc_x`, `acc_y`, `acc_z`, `phi`).
- Overwrite the same column names with scaled values.

Recorded artifacts:
- Per-column scaler parameters (`min`, `max`, `range`, `is_constant`)
- Per-run scaling diagnostics JSON

Output:
- `scaled_table` (per run)
- `scaler_parameters`
- `scaling_info`

### Stage 6 - Export HDF5

What it does:
- Write each run's `scaled_table` to its run key (`/runs/...`).
- Write metadata tables to `/meta/*`.

Exported table contents:
- `/meta/runs`: run identity, resolved HDF5 key, input CSV paths, sync mode, final row count.
- `/meta/run_logs`: sync shift, lag, correlation, forward-fill counts, stage-4 row counts, and JSON blobs (`sync_info_json`, `repair_report_json`, `downsample_info_json`).
- `/meta/run_scaling`: per-run `scaling_info_json`.
- `/meta/scaler_parameters`: global scaler fit parameters.
- `/meta/calibration`: pressure/gyro calibration path and payload JSON snapshots.
- `/meta/export_settings`: manifest path, output path, UTC export timestamp, decisions/default settings/scaling payload JSON.

Storage format:
- All HDF5 tables are written with pandas fixed format (`format="fixed"`).
