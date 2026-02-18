import json

import numpy as np
import pandas as pd

from preprocessing.gyro_calibration import (
    GyroCalibration,
    apply_gyro_bias_correction,
    fit_gyro_bias_from_arrays,
    fit_gyro_bias_from_dataframe,
    load_gyro_calibration_json,
)


def test_fit_gyro_bias_from_arrays_mean():
    x = np.array([0.11, 0.09, 0.10], dtype=np.float64)
    y = np.array([-0.02, -0.01, -0.03], dtype=np.float64)
    z = np.array([0.001, 0.002, 0.0], dtype=np.float64)

    calib, stats = fit_gyro_bias_from_arrays(x, y, z, method="mean")

    assert np.isclose(calib.bias_x_rads, np.mean(x), atol=1e-12)
    assert np.isclose(calib.bias_y_rads, np.mean(y), atol=1e-12)
    assert np.isclose(calib.bias_z_rads, np.mean(z), atol=1e-12)
    assert stats["n_samples"] == 3.0
    assert stats["method"] == "mean"


def test_fit_gyro_bias_from_dataframe_median():
    df = pd.DataFrame(
        {
            "gx": [1.0, 10.0, 2.0],
            "gy": [2.0, 20.0, 3.0],
            "gz": [3.0, 30.0, 4.0],
        }
    )
    calib, _ = fit_gyro_bias_from_dataframe(df, gyro_columns=("gx", "gy", "gz"), method="median")
    assert np.isclose(calib.bias_x_rads, 2.0, atol=1e-12)
    assert np.isclose(calib.bias_y_rads, 3.0, atol=1e-12)
    assert np.isclose(calib.bias_z_rads, 4.0, atol=1e-12)


def test_apply_gyro_bias_correction():
    calib = GyroCalibration(bias_x_rads=0.1, bias_y_rads=-0.2, bias_z_rads=0.05)
    x, y, z = apply_gyro_bias_correction(
        np.array([0.3, 0.4], dtype=np.float64),
        np.array([-0.1, -0.2], dtype=np.float64),
        np.array([0.0, 0.1], dtype=np.float64),
        calibration=calib,
    )
    assert np.allclose(x, [0.2, 0.3], atol=1e-12)
    assert np.allclose(y, [0.1, 0.0], atol=1e-12)
    assert np.allclose(z, [-0.05, 0.05], atol=1e-12)


def test_load_gyro_calibration_json_top_level_and_nested(tmp_path):
    top = tmp_path / "top.json"
    top.write_text(
        json.dumps({"bias_x_rads": 0.01, "bias_y_rads": -0.02, "bias_z_rads": 0.03}),
        encoding="utf-8",
    )
    c1 = load_gyro_calibration_json(top)
    assert np.isclose(c1.bias_x_rads, 0.01, atol=1e-12)
    assert np.isclose(c1.bias_y_rads, -0.02, atol=1e-12)
    assert np.isclose(c1.bias_z_rads, 0.03, atol=1e-12)

    nested = tmp_path / "nested.json"
    nested.write_text(
        json.dumps({"gyro_calibration": {"bias_x_rads": 0.1, "bias_y_rads": 0.2, "bias_z_rads": 0.3}}),
        encoding="utf-8",
    )
    c2 = load_gyro_calibration_json(nested)
    assert np.isclose(c2.bias_x_rads, 0.1, atol=1e-12)
    assert np.isclose(c2.bias_y_rads, 0.2, atol=1e-12)
    assert np.isclose(c2.bias_z_rads, 0.3, atol=1e-12)
