from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.optitrack_compute_angle import compute_optitrack_relative_features
from preprocessing.optitrack_raw import load_optitrack_raw_csv


def _pick_existing(repo_root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def test_compute_optitrack_relative_features_fixed_orientation_first_row():
    repo_root = Path(__file__).resolve().parents[1]
    path = _pick_existing(
        repo_root,
        [
            "sample_data/sample_optitrack_fixed_orientation.csv",
            "sample_data/sample_optitrack_fixed_orientaion.csv",
        ],
    )

    df_raw = load_optitrack_raw_csv(path)
    df = compute_optitrack_relative_features(df_raw)
    r0 = df.iloc[0]

    for c in ["phi", "theta", "psi", "dx", "dy", "dz"]:
        assert c in df.columns

    assert np.isclose(r0["phi"], 0.001276000691179604, atol=1e-12)
    assert np.isclose(r0["theta"], 0.00012200103565401483, atol=1e-12)
    assert np.isclose(r0["psi"], -4.1909985639417714e-05, atol=1e-12)
    assert np.isclose(r0["dx"], 0.01243199999999997, atol=1e-12)
    assert np.isclose(r0["dy"], 0.13662400000000002, atol=1e-12)
    assert np.isclose(r0["dz"], -0.019677000000000028, atol=1e-12)


def test_compute_optitrack_relative_features_requires_columns():
    df = pd.DataFrame({"BR_W": [1.0]})
    with pytest.raises(ValueError, match="Missing required OptiTrack columns"):
        compute_optitrack_relative_features(df)


def test_compute_optitrack_relative_features_normalizes_quaternions():
    repo_root = Path(__file__).resolve().parents[1]
    path = _pick_existing(
        repo_root,
        [
            "sample_data/sample_optitrack_fixed_orientation.csv",
            "sample_data/sample_optitrack_fixed_orientaion.csv",
        ],
    )
    df_raw = load_optitrack_raw_csv(path)

    scaled = df_raw.copy()
    for c in ["BR_W", "BR_X", "BR_Y", "BR_Z", "TR_W", "TR_X", "TR_Y", "TR_Z"]:
        scaled[c] = 3.0 * scaled[c]

    out_ref = compute_optitrack_relative_features(df_raw, normalize_quaternions=True)
    out_scaled = compute_optitrack_relative_features(scaled, normalize_quaternions=True)

    for c in ["phi", "theta", "psi", "dx", "dy", "dz"]:
        assert np.allclose(
            out_ref[c].to_numpy(),
            out_scaled[c].to_numpy(),
            atol=1e-12,
            equal_nan=True,
        )
