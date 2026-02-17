from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.global_scaling import (
    DEFAULT_SCALE_COLUMNS,
    apply_global_min_max_scaler,
    fit_and_apply_global_min_max_scaling,
    fit_global_min_max_scaler,
)


def _pick_existing(base_dir: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = base_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {base_dir}: {candidates}")


def _pick_existing_dir(repo_root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these directories exist: {candidates}")


def test_fit_global_min_max_scaler_across_runs():
    run_tables = {
        "run_a": pd.DataFrame({"pressure": [1.0, 2.0], "acc_x": [0.0, 4.0]}),
        "run_b": pd.DataFrame({"pressure": [-2.0, 8.0], "acc_x": [-1.0, 3.0]}),
    }
    scaler = fit_global_min_max_scaler(run_tables, scale_columns=("pressure", "acc_x"))

    assert scaler["pressure"]["min"] == -2.0
    assert scaler["pressure"]["max"] == 8.0
    assert scaler["pressure"]["range"] == 10.0
    assert scaler["acc_x"]["min"] == -1.0
    assert scaler["acc_x"]["max"] == 4.0
    assert scaler["acc_x"]["range"] == 5.0


def test_apply_global_min_max_scaler_maps_bounds_to_unit_range():
    df = pd.DataFrame({"pressure": [-2.0, 3.0, 8.0]})
    scaler = {"pressure": {"min": -2.0, "max": 8.0, "range": 10.0, "is_constant": False}}
    out = apply_global_min_max_scaler(df, scaler_parameters=scaler, scale_columns=("pressure",))
    assert np.allclose(out["pressure"].to_numpy(), [-1.0, 0.0, 1.0], atol=1e-12)


def test_apply_global_min_max_scaler_constant_column_policy_zero():
    df = pd.DataFrame({"phi": [5.0, 5.0, 5.0]})
    scaler = {"phi": {"min": 5.0, "max": 5.0, "range": 0.0, "is_constant": True}}
    out = apply_global_min_max_scaler(df, scaler_parameters=scaler, scale_columns=("phi",))
    assert np.allclose(out["phi"].to_numpy(), [0.0, 0.0, 0.0], atol=1e-12)


def test_fit_and_apply_global_min_max_scaling_on_sample_downsample_data():
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = _pick_existing_dir(repo_root, ["sample_data/processed_downsample"])
    fixed_path = _pick_existing(
        input_dir,
        [
            "sample_filtered_downsampled_fixed_orientation.csv",
        ],
    )
    freehand_path = _pick_existing(
        input_dir,
        [
            "sample_filtered_downsampled_freehand_manipulation.csv",
        ],
    )

    run_tables = {
        "fixed": pd.read_csv(fixed_path),
        "freehand": pd.read_csv(freehand_path),
    }
    scaled_runs, scaler_params, info = fit_and_apply_global_min_max_scaling(
        run_tables,
        return_info=True,
    )

    assert set(scaler_params.keys()) == set(DEFAULT_SCALE_COLUMNS)
    required_cols = {"pressure", "acc_x", "acc_y", "acc_z", "phi", "Time"}

    for run_id in ("fixed", "freehand"):
        in_df = run_tables[run_id]
        out_df = scaled_runs[run_id]

        assert len(out_df) > 0
        assert len(out_df) == len(in_df)
        assert set(out_df.columns) == set(in_df.columns)
        assert required_cols.issubset(out_df.columns)

        scaled_values = out_df[list(DEFAULT_SCALE_COLUMNS)].to_numpy(dtype=np.float64)
        assert np.isfinite(scaled_values).all()
        assert np.all(scaled_values <= 1.0 + 1e-12)
        assert np.all(scaled_values >= -1.0 - 1e-12)

        unscaled_values = out_df[["Time"]].to_numpy(dtype=np.float64)
        assert np.isfinite(unscaled_values).all()

    assert set(info["runs"].keys()) == {"fixed", "freehand"}
