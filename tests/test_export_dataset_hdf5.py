from pathlib import Path
import json

from preprocessing.export_dataset_hdf5 import build_run_plan, preview_run_plan


def test_build_run_plan_normalizes_keys_and_filters_export_enabled(tmp_path):
    manifest = {
        "dataset_root": ".",
        "default_settings": {"sync": {"fixed_phi_transform": "auto"}},
        "runs": [
            {
                "run_id": "0roll_0pitch_tt_1",
                "arduino_raw_csv": "data/arduino_raw/m0roll_0pitch_tt_1.csv",
                "optitrack_raw_csv": "data/optitrack_raw/o0roll_0pitch_tt_1.csv",
                "sync_mode": "fixed",
                "hdf5_key": "runs/0roll_0pitch_tt_1",
                "export_enabled": "true",
            },
            {
                "run_id": "skip_me",
                "arduino_raw_csv": "data/arduino_raw/m_skip.csv",
                "optitrack_raw_csv": "data/optitrack_raw/o_skip.csv",
                "sync_mode": "fixed",
                "export_enabled": False,
            },
        ],
    }
    plan = build_run_plan(manifest)
    assert len(plan) == 1
    p = plan[0]
    assert p.run_id == "0roll_0pitch_tt_1"
    assert p.requested_hdf5_key == "runs/0roll_0pitch_tt_1"
    assert p.hdf5_key == "/runs/run_0roll_0pitch_tt_1"
    assert p.arduino_raw_csv.is_absolute()
    assert p.optitrack_raw_csv.is_absolute()


def test_preview_run_plan_counts_fixed_and_freehand(tmp_path):
    manifest = {
        "dataset_root": ".",
        "runs": [
            {
                "run_id": "fixed_1",
                "arduino_raw_csv": "data/arduino_raw/m_fixed_1.csv",
                "optitrack_raw_csv": "data/optitrack_raw/o_fixed_1.csv",
                "sync_mode": "fixed",
            },
            {
                "run_id": "freehand_1",
                "arduino_raw_csv": "data/arduino_raw/m_free_1.csv",
                "optitrack_raw_csv": "data/optitrack_raw/o_free_1.csv",
                "sync_mode": "freehand",
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    out = preview_run_plan(path)
    assert out["n_runs"] == 2
    assert out["sync_mode_counts"]["fixed"] == 1
    assert out["sync_mode_counts"]["freehand"] == 1
