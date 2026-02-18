import json

import pandas as pd

from preprocessing.restricted_motion_trim import (
    apply_restricted_motion_trim_for_run,
    load_restricted_motion_trim_config,
)


def test_load_trim_config_and_apply_for_run(tmp_path):
    payload = {
        "schema_version": "v1",
        "runs": [
            {
                "run_id": "run_1",
                "arduino": {"trim_up_to": 2, "trim_after": 6},
                "optitrack": {"trim_up_to": 1, "trim_after": 4},
                "notes": "exclude constrained segments",
            }
        ],
    }
    cfg_path = tmp_path / "trim_config.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    config_by_run = load_restricted_motion_trim_config(cfg_path)

    arduino_df = pd.DataFrame({"v": list(range(10))})
    optitrack_df = pd.DataFrame({"w": list(range(8))})
    arduino_out, optitrack_out, info = apply_restricted_motion_trim_for_run(
        run_id="run_1",
        arduino_df=arduino_df,
        optitrack_df=optitrack_df,
        trim_config_by_run=config_by_run,
        return_info=True,
    )

    assert list(arduino_out["v"]) == [2, 3, 4, 5, 6]
    assert list(optitrack_out["w"]) == [1, 2, 3, 4]
    assert info["applied"] is True
    assert info["run_id"] == "run_1"


def test_apply_trim_no_config_entry_is_noop_with_info():
    arduino_df = pd.DataFrame({"v": [10, 11, 12]})
    optitrack_df = pd.DataFrame({"w": [20, 21, 22, 23]})

    arduino_out, optitrack_out, info = apply_restricted_motion_trim_for_run(
        run_id="missing_run",
        arduino_df=arduino_df,
        optitrack_df=optitrack_df,
        trim_config_by_run={},
        return_info=True,
    )

    assert arduino_out.equals(arduino_df.reset_index(drop=True))
    assert optitrack_out.equals(optitrack_df.reset_index(drop=True))
    assert info["applied"] is False
    assert info["reason"] == "no_trim_entry_for_run"


def test_apply_trim_entry_with_null_bounds_is_noop_with_no_bounds_reason(tmp_path):
    payload = {
        "schema_version": "v1",
        "runs": [
            {
                "run_id": "run_1",
                "arduino": {"trim_up_to": None, "trim_after": None},
                "optitrack": {"trim_up_to": None, "trim_after": None},
                "notes": "",
            }
        ],
    }
    cfg_path = tmp_path / "trim_config.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    config_by_run = load_restricted_motion_trim_config(cfg_path)

    arduino_df = pd.DataFrame({"v": [1, 2, 3]})
    optitrack_df = pd.DataFrame({"w": [4, 5, 6]})
    arduino_out, optitrack_out, info = apply_restricted_motion_trim_for_run(
        run_id="run_1",
        arduino_df=arduino_df,
        optitrack_df=optitrack_df,
        trim_config_by_run=config_by_run,
        return_info=True,
    )

    assert arduino_out.equals(arduino_df.reset_index(drop=True))
    assert optitrack_out.equals(optitrack_df.reset_index(drop=True))
    assert info["applied"] is False
    assert info["reason"] == "no_bounds_for_run"
