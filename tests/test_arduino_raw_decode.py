from pathlib import Path
import numpy as np
import json

from preprocessing.arduino_raw import (
    PressureCalibration,
    load_arduino_raw_csv,
    load_pressure_calibration_json,
    pressure_adc_to_pa,
)

def test_load_arduino_fixed_orientation_first_row():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "sample_data" / "sample_arduino_fixed_orientation.csv"

    df = load_arduino_raw_csv(path)

    # First row decoded from:
    # 164,1,0,1,84,64,56,2,85,0,168,0,32,30,160,223
    r0 = df.iloc[0]

    # Time base
    assert r0["sample_index"] == 0
    assert np.isclose(r0["t_arduino_s"], 0.0)

    # Accel in m/s^2 (decoded g * 9.80665)
    assert np.isclose(r0["acc_x"], 0.25139117431640623, atol=1e-9)
    assert np.isclose(r0["acc_y"], 0.15322890625, atol=1e-9)
    assert np.isclose(r0["acc_z"], 9.85692823486328, atol=1e-9)

    # Gyro in rad/s
    assert np.isclose(r0["gyr_x"], 0.07567534466662436, atol=1e-9)
    assert np.isclose(r0["gyr_y"], 0.011324655451871604, atol=1e-9)
    assert np.isclose(r0["gyr_z"], 0.022382848422522698, atol=1e-9)

    # Raw ADC checks
    assert r0["pressure_adc"] == 482.0
    assert r0["flex_adc"] == 3578.0


def test_pressure_adc_to_pa_no_clip_by_default():
    calib = PressureCalibration(v_min_ratio=0.1, v_max_ratio=0.2, p_max_pa=1000.0)
    adc = np.array([0.0, 4095.0], dtype=np.float64)

    out = pressure_adc_to_pa(adc, calib=calib)
    assert np.isclose(out[0], -1000.0, atol=1e-12)
    assert np.isclose(out[1], 9000.0, atol=1e-12)

    out_clip = pressure_adc_to_pa(adc, calib=calib, clip=True)
    assert np.allclose(out_clip, [0.0, 1000.0], atol=1e-12)


def test_load_pressure_calibration_json_accepts_flat_and_nested(tmp_path):
    flat_path = tmp_path / "flat.json"
    flat_payload = {"v_min_ratio": 0.12, "v_max_ratio": 0.43, "p_max_pa": 76480.0}
    flat_path.write_text(json.dumps(flat_payload), encoding="utf-8")

    c1 = load_pressure_calibration_json(flat_path)
    assert np.isclose(c1.v_min_ratio, 0.12, atol=1e-12)
    assert np.isclose(c1.v_max_ratio, 0.43, atol=1e-12)
    assert np.isclose(c1.p_max_pa, 76480.0, atol=1e-12)

    nested_path = tmp_path / "nested.json"
    nested_payload = {
        "equivalent_ratiometric": {"v_min_ratio": 0.11, "v_max_ratio": 0.41, "p_max_pa": 70000.0}
    }
    nested_path.write_text(json.dumps(nested_payload), encoding="utf-8")

    c2 = load_pressure_calibration_json(nested_path)
    assert np.isclose(c2.v_min_ratio, 0.11, atol=1e-12)
    assert np.isclose(c2.v_max_ratio, 0.41, atol=1e-12)
    assert np.isclose(c2.p_max_pa, 70000.0, atol=1e-12)


def test_load_arduino_raw_applies_gyro_bias_json(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "sample_data" / "sample_arduino_fixed_orientation.csv"

    base = load_arduino_raw_csv(path)
    bias = {"bias_x_rads": 0.01, "bias_y_rads": -0.02, "bias_z_rads": 0.005}
    bias_path = tmp_path / "gyro_bias.json"
    bias_path.write_text(json.dumps(bias), encoding="utf-8")

    corrected = load_arduino_raw_csv(path, gyro_calib_json_path=bias_path)

    assert np.isclose(
        corrected.iloc[0]["gyr_x"], base.iloc[0]["gyr_x_rads"] - bias["bias_x_rads"], atol=1e-12
    )
    assert np.isclose(
        corrected.iloc[0]["gyr_y"], base.iloc[0]["gyr_y_rads"] - bias["bias_y_rads"], atol=1e-12
    )
    assert np.isclose(
        corrected.iloc[0]["gyr_z"], base.iloc[0]["gyr_z_rads"] - bias["bias_z_rads"], atol=1e-12
    )

    # Traceability columns remain raw decode outputs
    assert np.isclose(corrected.iloc[0]["gyr_x_rads"], base.iloc[0]["gyr_x_rads"], atol=1e-12)

