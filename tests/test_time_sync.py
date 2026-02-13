from pathlib import Path

import numpy as np

from preprocessing.arduino_raw import load_arduino_raw_csv
from preprocessing.optitrack_compute_angle import compute_optitrack_relative_features
from preprocessing.optitrack_raw import load_optitrack_raw_csv, repair_optitrack_missing_samples
from preprocessing.time_sync import (
    compute_virtual_accelerometer_from_optitrack,
    estimate_shift_from_signals,
    synchronize_fixed_orientation,
    synchronize_freehand,
)


def _pick_existing(repo_root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def _load_optitrack_features(path: Path):
    raw = load_optitrack_raw_csv(path)
    repaired = repair_optitrack_missing_samples(raw, strict=True)
    return compute_optitrack_relative_features(repaired, strict=True)


def test_estimate_shift_from_signals_known_delay():
    x = np.zeros(400, dtype=np.float64)
    x[20] = 1.0
    x[80] = 0.4
    x[170] = -0.8
    x[260] = 0.2

    delay = 17  # y is delayed wrt x
    y = np.zeros_like(x)
    y[delay:] = x[:-delay]

    sample_shift, lag, _ = estimate_shift_from_signals(x, y, max_lag=100)
    assert sample_shift == delay
    assert lag == delay


def test_synchronize_fixed_orientation_on_sample_data():
    repo_root = Path(__file__).resolve().parents[1]
    arduino_path = _pick_existing(
        repo_root,
        ["sample_data/sample_arduino_fixed_orientation.csv"],
    )
    optitrack_path = _pick_existing(
        repo_root,
        [
            "sample_data/sample_optitrack_fixed_orientation.csv",
            "sample_data/sample_optitrack_fixed_orientaion.csv",
        ],
    )

    arduino_df = load_arduino_raw_csv(arduino_path)
    optitrack_df = _load_optitrack_features(optitrack_path)

    synced_df, info = synchronize_fixed_orientation(
        arduino_df,
        optitrack_df,
        phi_transform="auto",
        return_info=True,
    )

    assert len(synced_df) > 0
    assert info["mode"] == "fixed"
    assert info["phi_transform_used"] in {"invert", "abs"}
    assert isinstance(info["sample_shift_arduino"], int)
    assert synced_df[["pressure", "phi", "acc_x", "acc_y", "acc_z"]].notna().all().all()


def test_synchronize_fixed_orientation_auto_vs_none_on_sample_data():
    repo_root = Path(__file__).resolve().parents[1]
    arduino_path = _pick_existing(
        repo_root,
        ["sample_data/sample_arduino_fixed_orientation.csv"],
    )
    optitrack_path = _pick_existing(
        repo_root,
        [
            "sample_data/sample_optitrack_fixed_orientation.csv",
            "sample_data/sample_optitrack_fixed_orientaion.csv",
        ],
    )

    arduino_df = load_arduino_raw_csv(arduino_path)
    optitrack_df = _load_optitrack_features(optitrack_path)

    _, info_none = synchronize_fixed_orientation(
        arduino_df,
        optitrack_df,
        phi_transform="none",
        return_info=True,
    )
    _, info_auto = synchronize_fixed_orientation(
        arduino_df,
        optitrack_df,
        phi_transform="auto",
        return_info=True,
    )

    # For this fixed sample, direct phi correlation lands on an unrealistic periodic peak.
    assert abs(info_none["sample_shift_arduino"]) > 30_000
    assert abs(info_auto["sample_shift_arduino"]) < 5_000


def test_virtual_accelerometer_shape_and_finite():
    repo_root = Path(__file__).resolve().parents[1]
    optitrack_path = _pick_existing(
        repo_root,
        ["sample_data/sample_optitrack_freehand_manipulation.csv"],
    )
    optitrack_df = _load_optitrack_features(optitrack_path)

    virt = compute_virtual_accelerometer_from_optitrack(optitrack_df)
    assert virt.shape == (len(optitrack_df), 3)
    assert np.isfinite(virt).all()


def test_synchronize_freehand_on_sample_data():
    repo_root = Path(__file__).resolve().parents[1]
    arduino_path = _pick_existing(
        repo_root,
        ["sample_data/sample_arduino_freehand_manipulation.csv"],
    )
    optitrack_path = _pick_existing(
        repo_root,
        ["sample_data/sample_optitrack_freehand_manipulation.csv"],
    )

    arduino_df = load_arduino_raw_csv(arduino_path)
    optitrack_df = _load_optitrack_features(optitrack_path)

    synced_df, info = synchronize_freehand(
        arduino_df,
        optitrack_df,
        return_info=True,
    )

    assert len(synced_df) > 0
    assert info["mode"] == "freehand"
    assert info["sync_variable"] in {"acc_x", "acc_y", "acc_z"}
    assert isinstance(info["sample_shift_arduino"], int)
    assert synced_df[["pressure", "phi", "acc_x", "acc_y", "acc_z"]].notna().all().all()
