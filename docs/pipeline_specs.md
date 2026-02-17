# Data Pre-processing Pipeline Specification

## Purpose
Transform raw Arduino sensor logs and raw OptiTrack motion-capture logs into a single, temporally aligned, filtered, downsampled, and consistently scaled dataset per run. The primary supervised target is the bending angle **ϕ(t)** derived from OptiTrack.

---

## Inputs (per run)
1) **Arduino raw log**
- Pressure
- Accelerometer (3-axis)
- (Optional/retained) Gyroscope, flex sensor
- Time base (sample index)

2) **OptiTrack raw log**
- Base rigid-body quaternion and position 
- Tip rigid-body quaternion and position 
- Time base (timestamps)

---

## Outputs
A dataset containing one entry per run (e.g., HDF5 key per run). Each run contains a time-series table at the final sampling rate with at minimum:
- `pressure`
- `acc_x`, `acc_y`, `acc_z`
- `phi` (ϕ)

Additional channels (e.g., Euler angles, positions, quaternions, displacement components) may be included as long as they are consistently defined.

---

## Pipeline stages

### Stage 0 — Ingest and standardise
**What it does**
- Reads raw Arduino and raw OptiTrack files.
- Produces two typed, consistently named tables with explicit units and a monotonic time/index.
- Ensures the required fields exist (pressure + accel for Arduino; base/tip pose for OptiTrack).

**Output**
- `arduino_raw_table`
- `optitrack_raw_table`

---

### Stage 1 — Repair missing OptiTrack samples (zero-order hold)
**What it does**
- Handles missing OptiTrack samples (e.g., occlusions) by forward-filling (zero-order hold) pose data so downstream computations are defined at every sample.
- (Optional) Renormalises quaternions after fill, if required by your convention.

**Output**
- `optitrack_repaired_table`

---

### Stage 2 — Compute ground-truth kinematics from OptiTrack
**What it does**
1) **Relative orientation**
- Computes the relative quaternion between base and tip:
  - `q_rel = q_base^{-1} ⊗ q_tip`

2) **Euler angles (ZYX)**
- Converts `q_rel` into Euler angles using a ZYX convention.
- Defines **ϕ** as the bending angle about the major bending axis.

3) **Relative displacement**
- Computes displacement components using your chosen convention (document the sign convention once and use it everywhere).

**Output**
- `optitrack_features_table` (includes `phi` and any supporting features)

---

### Stage 3 — Synchronise Arduino and OptiTrack (cross-correlation) + trim
**What it does**
- Estimates the time offset between Arduino and OptiTrack streams using cross-correlation:
  - `R_xy[k] = Σ x[n] y[n + k]`
  - `k* = argmax_k R_xy[k]`

**Fixed-orientation runs**
- Uses:
  - `x[n] = pressure[n]` (Arduino)
  - `y[n] = transform(phi[n])` (OptiTrack), where `transform` can be one of:
    - `none` (raw phi)
    - `invert` (-phi)
    - `abs` (|phi|)
    - `auto` (select the transform with highest cross-correlation)
- Applies the lag `k*` as a shift to align streams.
- Note: this implementation detail must be updated in the thesis text (Chapter 3) so method description and code are consistent.

**Freehand/dynamic runs**
- Builds a “virtual accelerometer” reference from OptiTrack base orientation by rotating the gravity vector from world into the base frame (exact rotation direction/sign must match your quaternion and IMU conventions).
- Synchronises measured accelerometer to this virtual accelerometer via cross-correlation, then applies the lag.

**Freehand manual trim (post-sync)**
- For freehand runs, perform an additional manual trim after synchronization to remove initial/final static periods where the actuator is resting on a surface.
- Inspect accelerometer channels (`acc_x`, `acc_y`, `acc_z`) on the synchronized run and select trim boundaries:
  - `trim_start_samples`: number of samples removed from the beginning.
  - `trim_end_samples`: number of samples removed from the end.
- Apply this trim after overlap trim and before Stage 4 filtering/downsampling.
- Store the trim decisions in a run-level trim manifest/log so the process is reproducible.
- Record per-run diagnostics:
  - `rows_before_manual_trim`
  - `trim_start_samples`
  - `trim_end_samples`
  - `rows_after_manual_trim`
  - `rows_trimmed_manual_total`
  - `first_kept_original_row`, `last_kept_original_row`

**Trim**
- After shifting, trims the combined data to the time interval where all required channels overlap.

**Output**
- `synced_table` (Arduino + OptiTrack features aligned on a common timeline)

---

### Stage 4 — Filter and downsample
**What it does**
- Applies a moving-average (sliding mean) filter to reduce high-frequency noise.
- Downsamples from the OptiTrack/merged rate to the desired final rate via decimation (e.g., 240 Hz → 48 Hz by factor 5).
- Produces the final time base used by the model.

**Implementation conventions (current code)**
- Moving-average window: `5` samples by default.
- Filter alignment: `centered` by default (optional `causal` mode).
- Edge handling: partial windows are averaged (`min_periods=1`), so no rows are dropped by filtering.
- Decimation policy: keep rows at indices `decimation_offset + n * decimation_factor` (default offset `0`).
- Default sample-rate conversion: `240 Hz -> 48 Hz` with decimation factor `5`.
- Time-base policy: after decimation, `Time` is rebased to start at `0` with step `1 / output_sample_rate_hz`.
- Optional diagnostics metadata includes row counts before/after, dropped rows, decimation settings, and effective sample rates.

**Output**
- `filtered_downsampled_table`

---

### Stage 5 — Global scaling to [-1, 1]
**What it does**
- Computes global minima and maxima across the dataset for the channels you scale (at minimum: pressure, accelerometer axes, and ϕ).
- Applies a consistent min–max scaling to map each scaled channel into `[-1, 1]`.
- Stores the scaler parameters (min/max per channel) for reproducibility.

**Implementation conventions (current code)**
- Fit one scaler across all runs in the same processing split (global min/max per scaled column).
- Default scaled columns: `pressure`, `acc_x`, `acc_y`, `acc_z`, `phi`.
- Scaled values overwrite the same column names (no `_scaled` suffix by default).
- Constant-column policy: if a column has zero global range, output zeros for that column.
- Persist scaler artifacts as JSON (per-column `min`, `max`, `range`, `is_constant`) plus a run-level scaling log.

**Output**
- `scaled_table`
- `scaler_parameters` (saved alongside the dataset)

---

### Stage 6 — Export
**What it does**
- Writes the final per-run tables to the chosen dataset format (HDF5), using a stable column schema.
- Stores run-level metadata as attributes or a sidecar file (e.g., run type, applied sync shift, final sampling rate).

**Output**
- Final dataset file (one entry per run)

---
