# Data Pre-processing Pipeline Specification

## Purpose

Convert raw Arduino and OptiTrack CSV runs into one reproducible HDF5 dataset
containing final per-run tables and metadata needed to audit preprocessing
decisions.

Primary implementation: `src/preprocessing/export_dataset_hdf5.py`

## Inputs

Per run:

1. Arduino raw CSV:
   pressure ADC, flex ADC, accelerometer, gyroscope (16-byte row format).
2. OptiTrack raw CSV:
   base/tip quaternion and position streams.

Global inputs:

- Preprocessing manifest: `configs/preprocessing_manifest_all_trials.json`
- Optional trim config: `configs/restricted_motion_trim_config.json`
- Calibration JSONs in `calibration/`

## Final Output

Single HDF5 file (default `outputs/preprocessed_all_trials.h5`) with:

- Per-run tables at `/runs/<normalized_run_key>`
- Metadata tables at:
  - `/meta/runs`
  - `/meta/run_logs`
  - `/meta/run_scaling`
  - `/meta/scaler_parameters`
  - `/meta/calibration`
  - `/meta/export_settings`

Intermediate stage tables are not persisted to HDF5.

## Stage-by-Stage Specification

### Stage 0: Arduino and OptiTrack ingest

Code:

- `src/preprocessing/arduino_raw.py`
- `src/preprocessing/optitrack_raw.py`

Behavior:

- Decode Arduino byte fields into typed sensor channels.
- Apply pressure and gyro calibration when configured.
- Parse OptiTrack CSV and standardize expected columns.

Outputs:

- `arduino_raw_table`
- `optitrack_raw_table`

### Stage 1: OptiTrack missing-sample repair

Code:

- `src/preprocessing/optitrack_raw.py` (`repair_optitrack_missing_samples`)

Behavior:

- Forward-fill missing pose samples.
- Produce diagnostics: filled rows/cells, longest occlusion stretch.

Outputs:

- `optitrack_repaired_table`
- `repair_report`

### Stage 2: OptiTrack feature computation

Code:

- `src/preprocessing/optitrack_compute_angle.py`

Behavior:

- Compute base-to-tip relative orientation.
- Convert to ZYX Euler angles (`phi`, `theta`, `psi`).
- Unwrap `phi` to avoid `+/-pi` discontinuities.
- Compute relative displacement (`dx`, `dy`, `dz`).

Output:

- `optitrack_features_table`

### Stage 2.5: Restricted-motion trimming (optional)

Code:

- `src/preprocessing/restricted_motion_trim.py`

Controlled by:

- `decisions.exclude_restricted_motion_periods`
- `decisions.restricted_motion_trim_config_path`

Behavior:

- Apply per-run trim bounds independently to Arduino and OptiTrack pre-sync
  tables.
- Record applied bounds, removed rows, and skip reason when not applied.

Outputs:

- Trimmed Arduino and OptiTrack stage tables
- `trim_info`

### Stage 3: Time synchronization and overlap trim

Code:

- `src/preprocessing/time_sync.py`

Behavior:

- Fixed runs: correlate pressure with transformed `phi` using
  `none`/`invert`/`abs`/`auto`.
- Freehand runs: correlate measured IMU acceleration with virtual
  accelerometer from OptiTrack orientation.
- Apply sample shift and trim to overlapping valid interval.

Outputs:

- `synced_table`
- `sync_info`

### Stage 4: Filtering and downsampling

Code:

- `src/preprocessing/filter_downsample.py`

Behavior:

- Moving-average filter (`centered` or `causal`).
- Integer decimation (default `240 Hz -> 48 Hz` with factor `5`).
- Optional time rebasing to uniform output timeline.

Outputs:

- `filtered_downsampled_table`
- `stage4_info`

### Stage 5: Global scaling

Code:

- `src/preprocessing/global_scaling.py`

Behavior:

- Fit one global min-max scaler across selected runs.
- Default scaled columns: `pressure`, `acc_x`, `acc_y`, `acc_z`, `phi`.
- Apply scaling in place to selected columns.

Outputs:

- `scaled_table` (per run)
- `scaler_parameters`
- `scaling_info`

### Stage 6: HDF5 export

Code:

- `src/preprocessing/export_dataset_hdf5.py`

Behavior:

- Write each run table to its normalized HDF5 key.
- Write all metadata tables under `/meta/*`.
- Store JSON payload snapshots for calibrations and settings.

Storage:

- All HDF5 tables are written with pandas fixed format (`format="fixed"`).

## Runtime Entry Point

Use `scripts/export_dataset_hdf5.py`:

```powershell
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --dry-run
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --overwrite
```
