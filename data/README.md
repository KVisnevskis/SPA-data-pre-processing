## Data Directory

Contents in this folder are source inputs and legacy reference files used by the
pipeline and validation scripts.

- `arduino_raw/` and `optitrack_raw/`: raw trial logs
- `synced_48Hz_trim_LEGACY.h5`: legacy comparison target
- `preprocessing_manifest_all_trials.json`: compatibility copy of manifest

Canonical manifest editing should be done in:
`configs/preprocessing_manifest_all_trials.json`.
