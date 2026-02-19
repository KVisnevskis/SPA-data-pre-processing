## Data Directory

This folder contains full raw trial inputs used by the preprocessing pipeline.

- `arduino_raw/`: raw Arduino trial logs
- `optitrack_raw/`: raw OptiTrack trial logs
- `synced_48Hz_trim_LEGACY.h5`: legacy reference artifact for historical
  comparison

Use `configs/preprocessing_manifest_all_trials.json` to select and configure
which runs are exported.
