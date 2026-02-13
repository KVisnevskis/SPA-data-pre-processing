from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.optitrack_raw import (
    DEFAULT_COLUMNS,
    POSE_COLUMNS,
    load_optitrack_raw_csv,
    repair_optitrack_missing_samples,
)


def _pick_existing(repo_root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"Expected numeric scalar, got {type(value).__name__}")


def test_repair_optitrack_missing_samples_forward_fills_pose_columns():
    data = {c: [1.0, 2.0, 3.0] for c in DEFAULT_COLUMNS}
    data["Time"] = [0.0, 1.0, 2.0]
    df = pd.DataFrame(data)

    df.loc[1, "BR_X"] = np.nan
    df.loc[1, "BP_Z"] = np.nan
    df.loc[1, "TR_W"] = np.nan
    df.loc[1, "TP_Y"] = np.nan

    repaired = repair_optitrack_missing_samples(df)

    assert _as_float(repaired.loc[1, "BR_X"]) == pytest.approx(_as_float(repaired.loc[0, "BR_X"]))
    assert _as_float(repaired.loc[1, "BP_Z"]) == pytest.approx(_as_float(repaired.loc[0, "BP_Z"]))
    assert _as_float(repaired.loc[1, "TR_W"]) == pytest.approx(_as_float(repaired.loc[0, "TR_W"]))
    assert _as_float(repaired.loc[1, "TP_Y"]) == pytest.approx(_as_float(repaired.loc[0, "TP_Y"]))


def test_repair_optitrack_missing_samples_strict_raises_for_leading_missing():
    data = {c: [1.0, 2.0, 3.0] for c in DEFAULT_COLUMNS}
    data["Time"] = [0.0, 1.0, 2.0]
    df = pd.DataFrame(data)
    df.loc[0, "BR_X"] = np.nan

    with pytest.raises(ValueError, match="still contain missing values"):
        repair_optitrack_missing_samples(df, strict=True)


def test_repair_optitrack_missing_samples_cleans_sample_file_pose_nans():
    repo_root = Path(__file__).resolve().parents[1]
    path = _pick_existing(
        repo_root,
        [
            "sample_data/sample_optitrack_fixed_orientation.csv",
            "sample_data/sample_optitrack_fixed_orientaion.csv",
        ],
    )

    raw_df = load_optitrack_raw_csv(path)
    assert raw_df[POSE_COLUMNS].isna().to_numpy().any()

    repaired_df = repair_optitrack_missing_samples(raw_df, strict=True)
    assert not repaired_df[POSE_COLUMNS].isna().to_numpy().any()


def test_repair_optitrack_missing_samples_returns_report():
    data = {c: [1.0, 2.0, 3.0, 4.0, 5.0] for c in DEFAULT_COLUMNS}
    data["Time"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    df = pd.DataFrame(data)

    df.loc[1, ["BR_X", "TP_Y"]] = np.nan
    df.loc[2, ["TR_W"]] = np.nan
    df.loc[4, ["BP_Z"]] = np.nan

    repaired_df, report = repair_optitrack_missing_samples(df, return_report=True)

    assert repaired_df.shape == df.shape
    assert report["rows_forward_filled_position"] == [1, 2, 4]
    assert report["rows_forward_filled_count"] == 3
    assert report["cells_forward_filled_count"] == 4
    assert report["longest_occlusion_stretch_length"] == 2
    assert report["longest_occlusion_stretch_start_position"] == 1
    assert report["longest_occlusion_stretch_end_position"] == 2
