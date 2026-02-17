from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.filter_downsample import filter_and_downsample_stage4


def _pick_existing(repo_root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def _expected_decimated_len(n_rows: int, factor: int, offset: int) -> int:
    if offset >= n_rows:
        return 0
    return ((n_rows - 1 - offset) // factor) + 1


def test_moving_average_centered_partial_window_policy():
    df = pd.DataFrame(
        {
            "sample_index": [0, 1, 2, 3, 4],
            "Time": [10.0, 10.1, 10.2, 10.3, 10.4],
            "pressure": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    out = filter_and_downsample_stage4(
        df,
        decimation_factor=1,
        moving_average_window=3,
        filter_alignment="centered",
        filter_columns=("pressure",),
        rebase_time=False,
    )

    assert np.allclose(out["pressure"].to_numpy(), [1.5, 2.0, 3.0, 4.0, 4.5], atol=1e-12)
    assert np.array_equal(out["sample_index"].to_numpy(), df["sample_index"].to_numpy())
    assert np.allclose(out["Time"].to_numpy(), df["Time"].to_numpy(), atol=1e-12)


def test_moving_average_causal_policy():
    df = pd.DataFrame({"pressure": [1.0, 2.0, 3.0, 4.0, 5.0], "Time": np.arange(5)})
    out = filter_and_downsample_stage4(
        df,
        decimation_factor=1,
        moving_average_window=3,
        filter_alignment="causal",
        filter_columns=("pressure",),
        rebase_time=False,
    )
    assert np.allclose(out["pressure"].to_numpy(), [1.0, 1.5, 2.0, 3.0, 4.0], atol=1e-12)


def test_decimation_anchor_policy_with_offset():
    df = pd.DataFrame(
        {
            "Time": np.arange(20, dtype=np.float64) / 240.0,
            "pressure": np.arange(20, dtype=np.float64),
        }
    )

    out, info = filter_and_downsample_stage4(
        df,
        decimation_factor=5,
        decimation_offset=2,
        moving_average_window=1,
        filter_columns=("pressure",),
        rebase_time=False,
        return_info=True,
    )

    assert np.array_equal(out["pressure"].to_numpy(), np.array([2.0, 7.0, 12.0, 17.0]))
    assert info["rows_before"] == 20
    assert info["rows_after"] == 4
    assert info["rows_dropped"] == 16
    assert info["kept_first_input_row"] == 2
    assert info["kept_last_input_row"] == 17
    assert info["decimation_factor"] == 5
    assert info["decimation_offset"] == 2


def test_stage4_rebases_time_column():
    df = pd.DataFrame(
        {
            "Time": np.linspace(12.0, 12.5, num=6),
            "pressure": np.arange(6, dtype=np.float64),
        }
    )
    out = filter_and_downsample_stage4(
        df,
        input_sample_rate_hz=240.0,
        decimation_factor=3,
        moving_average_window=1,
        filter_columns=("pressure",),
        rebase_time=True,
    )
    assert np.allclose(out["Time"].to_numpy(), [0.0, 1.0 / 80.0], atol=1e-12)


def test_filter_and_downsample_stage4_on_sample_stage3_data():
    repo_root = Path(__file__).resolve().parents[1]
    fixed_path = _pick_existing(
        repo_root,
        ["sample_data/processed_synced/sample_synced_fixed_orientation_stage3.csv"],
    )
    freehand_path = _pick_existing(
        repo_root,
        ["sample_data/processed_synced/sample_synced_freehand_manipulation_stage3.csv"],
    )

    required_cols = {"pressure", "acc_x", "acc_y", "acc_z", "phi", "Time"}

    for path in (fixed_path, freehand_path):
        stage3_df = pd.read_csv(path)
        stage4_df, info = filter_and_downsample_stage4(stage3_df, return_info=True)

        assert len(stage4_df) > 0
        assert set(stage3_df.columns) == set(stage4_df.columns)
        assert required_cols.issubset(stage4_df.columns)

        expected_rows = _expected_decimated_len(
            len(stage3_df),
            factor=info["decimation_factor"],
            offset=info["decimation_offset"],
        )
        assert len(stage4_df) == expected_rows
        assert info["rows_after"] == expected_rows
        assert info["rows_before"] == len(stage3_df)
        assert np.isclose(info["output_sample_rate_hz"], 48.0, atol=1e-12)

        finite = np.isfinite(
            stage4_df[["pressure", "acc_x", "acc_y", "acc_z", "phi"]].to_numpy(dtype=np.float64)
        )
        assert finite.all()
