## Sample Data

This folder contains compact fixtures for quick local validation of the
preprocessing pipeline. These are intentionally small excerpts; the full raw
dataset is distributed through Zenodo and is not tracked in git.

Raw sample inputs:

- `sample_arduino_fixed_orientation.csv`
- `sample_arduino_freehand_manipulation.csv`
- `sample_optitrack_fixed_orientation.csv`
- `sample_optitrack_freehand_manipulation.csv`

Processed fixture folders used by tests:

- `processed_synced/`
- `processed_downsample/`

Generated calibration, intermediate, scaled, and full-run artifacts should be
recreated locally from the Zenodo dataset when needed.
