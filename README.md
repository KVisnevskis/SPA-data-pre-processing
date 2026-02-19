# SPA Data Pre-processing

End-to-end preprocessing pipeline for converting raw Arduino and OptiTrack trial
logs into synchronized, downsampled, globally scaled HDF5 datasets for model
development.

## Raw Data Access (Zenodo)

Raw trial data is not tracked in git. Download the dataset ZIP from Zenodo:

- DOI: `10.5281/zenodo.18697336`
- URL: https://zenodo.org/records/18697336

After download, extract the archive so this repository has:

- `data/arduino_raw/*.csv`
- `data/optitrack_raw/*.csv`

## What This Repository Does

The pipeline processes each run through these stages:

1. Decode and calibrate Arduino raw bytes.
2. Ingest OptiTrack pose streams and repair missing samples.
3. Compute relative orientation and displacement features.
4. Optionally trim restricted-motion segments.
5. Synchronize Arduino and OptiTrack streams.
6. Apply moving-average filtering and decimation.
7. Fit one global scaler across selected runs and export HDF5.

Output is a single HDF5 with per-run tables under `/runs/*` and reproducibility
metadata under `/meta/*`.

## Quick Start

### 1) Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Optional dependencies:

- `tables` for HDF5 export/read via `pandas.HDFStore`
- `matplotlib` for GUI inspectors

```powershell
python -m pip install tables matplotlib
```

### 2) Download and extract raw data

Download from Zenodo (links above) and extract into `data/` so manifest paths
in `configs/preprocessing_manifest_all_trials.json` resolve correctly.

### 3) Preview selected runs from the manifest

```powershell
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --dry-run
```

### 4) Run full export

```powershell
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --overwrite
```

Default output path: `outputs/preprocessed_all_trials.h5`

### 5) Inspect data

```powershell
.\.venv\Scripts\python.exe scripts\inspect_raw_gui.py
.\.venv\Scripts\python.exe scripts\inspect_hdf5_gui.py --hdf5 outputs/preprocessed_all_trials.h5
```

## Main Configuration Files

- `configs/preprocessing_manifest_all_trials.json`:
  run list and default preprocessing settings.
- `configs/restricted_motion_trim_config.json`:
  optional per-run trim bounds used when restricted-motion exclusion is enabled.
- `calibration/arduino_pressure_calibration.json`:
  pressure calibration parameters.
- `calibration/arduino_gyro_calibration.json`:
  gyro bias calibration parameters.

## Repository Layout

- `src/preprocessing/`: core pipeline implementation.
- `scripts/`: CLI entrypoints for export, calibration, sample generation, and inspection.
- `tests/`: unit/integration tests.
- `configs/`: manifest and preprocessing config files.
- `data/`: extracted raw dataset (from Zenodo) plus legacy reference artifact.
- `sample_data/`: small raw and processed sample files for quick validation.
- `outputs/`: generated exports and one-off analysis artifacts (gitignored).
- `docs/`: detailed documentation.
- `legacy/`: old prototypes and archive scripts, not part of the active pipeline.

## Documentation

- `docs/README.md`: documentation index.
- `docs/pipeline_specs.md`: stage-by-stage pipeline behavior.
- `docs/data_schema.md`: HDF5 output schema.
- `docs/manifest_reference.md`: manifest fields and examples.
- `docs/cli_reference.md`: command line script reference.

## Development

Run test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## License

This repository is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0). See `LICENSE`.
