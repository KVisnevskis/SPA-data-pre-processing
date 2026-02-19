# CLI Reference

All scripts are in `scripts/` and can be run with:

```powershell
.\.venv\Scripts\python.exe <script> [options]
```

## Main Export

### `scripts/export_dataset_hdf5.py`

Exports final per-run tables and metadata to a single HDF5.

Examples:

```powershell
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --dry-run
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --overwrite
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --run-ids 0roll_0pitch_tt_1 0roll_0pitch_tt_2 --overwrite
```

Key options:

- `--manifest-path`
- `--output-hdf5`
- `--run-ids`
- `--overwrite`
- `--dry-run`

## Calibration Utilities

### `scripts/fit_pressure_calibration.py`

Fits pressure calibration from steady plateaus in a raw Arduino file.

Typical flow:

1. Detect segments and save CSV.
2. Add `pressure_kpa` labels.
3. Re-run fit and write output JSONs.

Key options:

- `--input-raw`
- `--segments-csv`
- `--pressures-kpa`
- `--output-json`
- `--module-calibration-json`
- `--plot-path`
- `--no-plot`

### `scripts/fit_gyro_calibration.py`

Fits gyro bias from a stationary Arduino run.

Key options:

- `--input-raw`
- `--sample-rate-hz`
- `--method {mean,median}`
- `--fit-json`
- `--module-calibration-json`

## Sample Data Stage Scripts

These scripts generate staged sample artifacts in `sample_data/processed_*`.

### `scripts/generate_sample_arduino_features.py`

Raw Arduino -> decoded/calibrated sample CSVs.

### `scripts/generate_sample_optitrack_features.py`

Raw OptiTrack -> repaired + feature sample CSVs.

### `scripts/generate_sample_synced_data.py`

Processed Arduino + OptiTrack -> Stage-3 synchronized sample CSVs.

### `scripts/generate_sample_filter_downsample_data.py`

Stage-3 sample CSVs -> filtered/downsampled sample CSVs.

### `scripts/generate_sample_scaled_data.py`

Stage-4 sample CSVs -> globally scaled sample CSVs and scaler diagnostics.

Common option:

- `--dry-run` for preview without writing files.

## GUI Inspectors

### `scripts/inspect_raw_gui.py`

Interactive plotting for raw Arduino and OptiTrack streams loaded from the
manifest run list.

Key option:

- `--manifest-path`

### `scripts/inspect_hdf5_gui.py`

Interactive plotting and metadata browsing for exported HDF5.

Key option:

- `--hdf5`

## Test Command

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
