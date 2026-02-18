## SPA Data Pre-processing

Pipeline for converting raw Arduino + OptiTrack trial logs into synchronized,
downsampled, scaled HDF5 datasets for model development.

### Canonical entrypoint

```powershell
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --dry-run
.\.venv\Scripts\python.exe scripts\export_dataset_hdf5.py --overwrite
```

`scripts/sc_export_dataset_hdf5.py` is kept as a backward-compatible alias.

### Key paths

- `configs/preprocessing_manifest_all_trials.json`: main preprocessing manifest
- `configs/restricted_motion_trim_config.json`: optional trim bounds
- `outputs/`: generated exports and analysis artifacts (gitignored)

### Typical workflow

1. Inspect raw data (`scripts/inspect_raw_gui.py`) and set trim bounds if needed.
2. Export with `scripts/export_dataset_hdf5.py`.
3. Inspect output HDF5 with `scripts/inspect_hdf5_gui.py`.
