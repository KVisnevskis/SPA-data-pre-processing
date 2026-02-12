from pathlib import Path
import numpy as np

from preprocessing.arduino_raw import load_arduino_raw_csv

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

