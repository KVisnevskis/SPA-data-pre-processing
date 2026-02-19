## Data Directory

This folder stores raw trial inputs used by the preprocessing pipeline.

Raw data is distributed via Zenodo (not tracked in git):

- DOI: `10.5281/zenodo.18697336`
- URL: https://zenodo.org/records/18697336

Download the dataset ZIP and extract it into this directory so these paths
exist:

- `arduino_raw/`: raw Arduino trial logs
- `optitrack_raw/`: raw OptiTrack trial logs
- `synced_48Hz_trim_LEGACY.h5`: legacy reference artifact for historical
  comparison

Use `configs/preprocessing_manifest_all_trials.json` to select and configure
which runs are exported.
