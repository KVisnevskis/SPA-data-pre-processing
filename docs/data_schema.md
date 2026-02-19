# Data Schema

Schema version: `v1.1`  
Applies to: HDF5 output produced by `scripts/export_dataset_hdf5.py`.

## HDF5 Layout

### Run tables

- Key pattern: `/runs/<normalized_run_key>`
- Value: final per-run DataFrame after sync, downsampling, and scaling

Run keys are normalized before write. Example:

- requested `runs/0roll` -> stored `/runs/run_0roll`

### Metadata tables

- `/meta/runs`
- `/meta/run_logs`
- `/meta/run_scaling`
- `/meta/scaler_parameters`
- `/meta/calibration`
- `/meta/export_settings`

All tables are written with pandas fixed format (`format="fixed"`).

## Required Final Run Columns

Each exported run table must contain:

- `pressure`
- `acc_x`
- `acc_y`
- `acc_z`
- `phi`
- `Time`

## Metadata Table Columns

### `/meta/runs`

- `run_id`
- `hdf5_key`
- `requested_hdf5_key`
- `sync_mode`
- `arduino_raw_csv`
- `optitrack_raw_csv`
- `rows_final`

### `/meta/run_logs`

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

### `/meta/run_scaling`

- `run_id`
- `scaling_info_json`

### `/meta/scaler_parameters`

- `column`
- `min`
- `max`
- `range`
- `is_constant`

### `/meta/calibration`

- `calibration_type` (`pressure` or `gyro`)
- `path`
- `payload_json`

### `/meta/export_settings`

- `manifest_path`
- `output_hdf5_path`
- `export_timestamp_utc`
- `decisions_json`
- `default_settings_json`
- `scaling_info_json`

## Run-Table Conventions

### Time

- `Time` is in seconds.
- Default output sample rate is `48 Hz` (`240 Hz` input, decimation factor `5`).
- By default, `Time` is rebased after decimation to `0, 1/fs, 2/fs, ...`.

### Orientation and frames

- OptiTrack pose columns are in the OptiTrack world frame.
- IMU columns (`acc_*`, `gyr_*`) are in the IMU/base frame.
- Quaternions are split into scalar components (`*_X`, `*_Y`, `*_Z`, `*_W`).
- `phi`, `theta`, `psi` are ZYX Euler angles from base-to-tip relative rotation.
- `phi` is phase-unwrapped to keep continuity across `+/-pi`.

### Calibration and scaling

- `pressure` is calibrated from ADC using pressure calibration settings.
- `gyr_x`, `gyr_y`, `gyr_z` are bias-corrected using gyro calibration settings.
- Global min-max scaling maps configured columns to `[-1, 1]`.

## Common Columns

| Column | Units | Description |
|---|---|---|
| `Time` | s | Final run timebase after sync/downsample/rebase |
| `pressure` | Pa | Calibrated pressure |
| `pressure_adc` | counts | Raw pressure ADC retained for traceability |
| `acc_x`, `acc_y`, `acc_z` | m/s^2 | IMU acceleration channels |
| `gyr_x`, `gyr_y`, `gyr_z` | rad/s | Bias-corrected gyro channels |
| `flex` | counts | Flex sensor channel |
| `BR_X..BR_W`, `TR_X..TR_W` | unitless | Base/tip quaternion components |
| `BP_X..BP_Z`, `TP_X..TP_Z` | m | Base/tip positions |
| `phi`, `theta`, `psi` | rad | Relative ZYX Euler angles |
| `dx`, `dy`, `dz` | m | Relative displacement components |
