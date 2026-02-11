# Data Schema

Schema version: **v1.0**  
Applies to: **per-run pre-processed dataset tables** (one table per run in the final dataset file).

## Conventions

### Time
- `Time` is in **seconds**, measured from the start of the run **after** synchronisation and downsampling.
- Final sampling period is **1/48 s** (nominal **48 Hz**).

### Coordinate frames
- **OptiTrack global frame**: laboratory/world frame used by the motion-capture system.
- **IMU/base frame**: sensor frame rigidly attached to the SPA base (used by `acc_*` and `gyr_*`).

### Quaternion convention
- OptiTrack orientations are stored as **components** `*_X, *_Y, *_Z, *_W`.
- These represent a unit quaternion describing rigid-body orientation in the OptiTrack global frame.
- Unless explicitly stated elsewhere, treat the quaternion as **(w, x, y, z)** when performing quaternion algebra, but note that storage order is **X, Y, Z, W**.

### Euler angles and bending angle
- `phi`, `theta`, `psi` are the **ZYX Euler-angle** representation of the **relative** tip orientation with respect to the base.
- Units: **radians**.
- `phi` (ϕ) is the **bending angle** (major bending axis).

### Displacement convention
- `dx`, `dy`, `dz` are the **relative tip displacement components with respect to the base**.
- Convention for this schema (sign/direction):  
  **dx = BP_X − TP_X**, **dy = BP_Y − TP_Y**, **dz = BP_Z − TP_Z**  
  (i.e., base position minus tip position, in the OptiTrack global frame).  
  If this convention changes, bump schema version.

### Scaling
- The channels used for learning are typically normalised using **global min–max scaling to [-1, 1]**:
  - `pressure`
  - `acc_x`, `acc_y`, `acc_z`
  - `phi`
- Other channels are stored for reproducibility/analysis and are not necessarily scaled.

## Required columns (minimum)
A dataset is considered valid (v1.0) if it contains, at minimum:
- `pressure`, `acc_x`, `acc_y`, `acc_z`, `phi`, `Time`

## Column dictionary

| Column | Type | Units | Frame | Description | Notes |
|---|---:|---|---|---|---|
| `Unnamed: 0` | int | – | – | Legacy sample index from the original Arduino log. | Retained for traceability; not used for learning. |
| `acc_x` | float | m/s² | IMU/base | Accelerometer x-axis (gravity + linear acceleration). | Often scaled to [-1, 1]. |
| `acc_y` | float | m/s² | IMU/base | Accelerometer y-axis (gravity + linear acceleration). | Often scaled to [-1, 1]. |
| `acc_z` | float | m/s² | IMU/base | Accelerometer z-axis (gravity + linear acceleration). | Often scaled to [-1, 1]. |
| `gyr_x` | float | (as recorded) | IMU/base | Gyroscope x-axis angular velocity. | Stored; not required for baseline learning. |
| `gyr_y` | float | (as recorded) | IMU/base | Gyroscope y-axis angular velocity. | Stored; not required for baseline learning. |
| `gyr_z` | float | (as recorded) | IMU/base | Gyroscope z-axis angular velocity. | Stored; not required for baseline learning. |
| `flex` | float | (raw or calibrated) | local sensor | Flex sensor output, proportional to local curvature. | Stored; not required for baseline learning. |
| `pressure` | float | Pa | – | Internal chamber pressure (calibrated). | Often scaled to [-1, 1]. |
| `Time` | float | s | – | Timestamp from start of run after synchronisation/downsampling. | Monotonic; dt ≈ 1/48 s. |
| `BR_X` | float | – | OptiTrack global | Base rigid-body quaternion x component. | Stored order X,Y,Z,W. |
| `BR_Y` | float | – | OptiTrack global | Base rigid-body quaternion y component. |  |
| `BR_Z` | float | – | OptiTrack global | Base rigid-body quaternion z component. |  |
| `BR_W` | float | – | OptiTrack global | Base rigid-body quaternion w (scalar) component. |  |
| `BP_X` | float | m | OptiTrack global | Base rigid-body x position. |  |
| `BP_Y` | float | m | OptiTrack global | Base rigid-body y position. |  |
| `BP_Z` | float | m | OptiTrack global | Base rigid-body z position. |  |
| `TR_X` | float | – | OptiTrack global | Tip rigid-body quaternion x component. | Stored order X,Y,Z,W. |
| `TR_Y` | float | – | OptiTrack global | Tip rigid-body quaternion y component. |  |
| `TR_Z` | float | – | OptiTrack global | Tip rigid-body quaternion z component. |  |
| `TR_W` | float | – | OptiTrack global | Tip rigid-body quaternion w (scalar) component. |  |
| `TP_X` | float | m | OptiTrack global | Tip rigid-body x position. |  |
| `TP_Y` | float | m | OptiTrack global | Tip rigid-body y position. |  |
| `TP_Z` | float | m | OptiTrack global | Tip rigid-body z position. |  |
| `phi` | float | rad | relative base→tip | Bending angle ϕ from ZYX Euler of relative quaternion. | Learning target; often scaled to [-1, 1]. |
| `theta` | float | rad | relative base→tip | Secondary Euler angle θ from ZYX Euler of relative quaternion. | Stored for analysis. |
| `psi` | float | rad | relative base→tip | Secondary Euler angle ψ from ZYX Euler of relative quaternion. | Stored for analysis. |
| `dx` | float | m | OptiTrack global | Relative displacement x component (BP_X − TP_X). | Convention must remain consistent. |
| `dy` | float | m | OptiTrack global | Relative displacement y component (BP_Y − TP_Y). |  |
| `dz` | float | m | OptiTrack global | Relative displacement z component (BP_Z − TP_Z). |  |

## Validity rules (v1.0)
- `Time` must be strictly monotonic increasing.
- Required columns must contain **no missing values**.
- Quaternion components should be finite; unit-norm deviations should be small (tolerance defined in pipeline config).
- Units and conventions in this document must match the exported data; changes require a schema version bump.
