# Data Schema

Schema version: **v1.1**  
Applies to: **final HDF5 export output** produced by `scripts/export_dataset_hdf5.py` / `preprocessing.export_dataset_hdf5`.

## HDF5 File Structure

- Run tables are stored under run keys (default request is `runs/{run_id}`).
- Requested run keys are normalized to safe HDF5 segments before write (for example, `runs/0roll` becomes `/runs/run_0roll`).
- Only final per-run tables are persisted. Intermediate stage tables are not written to the output file.
- All tables are written with pandas HDF fixed format (`format="fixed"`).

### Run tables

- Key pattern: `/runs/<normalized_run_key>`
- Value: final preprocessed DataFrame for one run (post-sync, post-downsample, post-scaling)

Minimum required columns for each run table:
- `pressure`
- `acc_x`
- `acc_y`
- `acc_z`
- `phi`
- `Time`

### Metadata tables

- `/meta/runs`
- `/meta/run_logs`
- `/meta/run_scaling`
- `/meta/scaler_parameters`
- `/meta/calibration`
- `/meta/export_settings`

`/meta/runs` columns:
- `run_id`
- `hdf5_key`
- `requested_hdf5_key`
- `sync_mode`
- `arduino_raw_csv`
- `optitrack_raw_csv`
- `rows_final`

`/meta/run_logs` columns:
- `run_id`
- `samples_shifted_arduino`
- `sync_lag_samples`
- `sync_variable`
- `sync_max_cross_correlation`
- `rows_forward_filled_count`
- `cells_forward_filled_count`
- `longest_occlusion_stretch_length`
- `rows_after_sync_trim`
- `rows_after_downsample`
- `restricted_motion_trim_json`
- `sync_info_json`
- `repair_report_json`
- `downsample_info_json`

`/meta/run_scaling` columns:
- `run_id`
- `scaling_info_json`

`/meta/scaler_parameters` columns:
- `column`
- `min`
- `max`
- `range`
- `is_constant`

`/meta/calibration` columns:
- `calibration_type` (`pressure` or `gyro`)
- `path`
- `payload_json`

`/meta/export_settings` columns:
- `manifest_path`
- `output_hdf5_path`
- `export_timestamp_utc`
- `decisions_json`
- `default_settings_json`
- `scaling_info_json`

## Run-table Conventions

### Time

- `Time` is in seconds.
- Final sampling period is nominally `1/48 s` when defaults are used (`240 Hz` input, decimation factor `5`).
- By default, time is rebased after decimation to `0, 1/fs_out, 2/fs_out, ...`.

### Frames and orientation

- OptiTrack columns are in the OptiTrack world frame.
- IMU columns (`acc_*`, `gyr_*`) are in the IMU/base frame.
- Quaternions are stored by components as `*_X, *_Y, *_Z, *_W`.
- `phi`, `theta`, `psi` are ZYX Euler angles from base-to-tip relative orientation.

### Sensor calibration and scaling

- `pressure` is calibrated from ADC using the configured pressure calibration JSON.
- `gyr_x`, `gyr_y`, `gyr_z` are bias-corrected using the configured gyro calibration JSON.
- Global min-max scaling to `[-1, 1]` is applied across all selected runs for configured columns.

## Common Run Columns

| Column | Type | Units | Description |
|---|---:|---|---|
| `Time` | float | s | Final run timebase after sync/downsample/rebase. |
| `pressure` | float | Pa | Calibrated pressure. |
| `pressure_adc` | float | counts | Raw pressure ADC retained for traceability. |
| `acc_x`, `acc_y`, `acc_z` | float | m/s^2 | IMU acceleration channels. |
| `gyr_x`, `gyr_y`, `gyr_z` | float | rad/s | Bias-corrected gyroscope channels. |
| `flex` | float | counts | Flex sensor channel. |
| `BR_X..BR_W`, `TR_X..TR_W` | float | unitless | Base/tip quaternions (X,Y,Z,W component columns). |
| `BP_X..BP_Z`, `TP_X..TP_Z` | float | m | Base/tip positions. |
| `phi`, `theta`, `psi` | float | rad | Relative Euler angles (ZYX). |
| `dx`, `dy`, `dz` | float | m | Relative displacement components. |
