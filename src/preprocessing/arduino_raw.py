from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import math


@dataclass(frozen=True)
class PressureCalibration:
    """
    Ratiometric min-max mapping commonly used for ABP-style sensors.
    You can adjust v_min_ratio/v_max_ratio/p_max_pa later without touching decoding.
    """
    v_min_ratio: float = 0.10
    v_max_ratio: float = 0.90
    p_max_pa: float = 206_000.0  # 206 kPa
    #pressure currently exported as ADC counts; conversion to Pa deferred (see Issue #1)


def _to_signed16(n: int) -> int:
    n = n & 0xFFFF
    return n | (-(n & 0x8000))


def decode_row_16bytes(row: Sequence[int]) -> dict[str, float]:
    """
    Decode one Arduino sample row consisting of 16 bytes:
      0..11  -> accel/gyro int16 little-endian (MPU6050 style)
      12..15 -> two 12-bit ADC values packed in uint16 then >>4 (pressure, flex)
    Output accel in g, gyro in rad/s, adc in counts.
    """
    if len(row) != 16:
        raise ValueError(f"Expected 16 integers per row, got {len(row)}")

    s = [int(x) for x in row]

    ax_g = _to_signed16(s[0] | (s[1] << 8)) / 16384.0
    ay_g = _to_signed16(s[2] | (s[3] << 8)) / 16384.0
    az_g = _to_signed16(s[4] | (s[5] << 8)) / 16384.0

    # Raw -> deg/s (MPU6050 @ ±250 dps => 131 LSB/(deg/s)), then deg/s -> rad/s
    dps_to_rads = math.pi / 180.0
    gx_rads = (_to_signed16(s[6]  | (s[7]  << 8)) / 131.0) * dps_to_rads
    gy_rads = (_to_signed16(s[8]  | (s[9]  << 8)) / 131.0) * dps_to_rads
    gz_rads = (_to_signed16(s[10] | (s[11] << 8)) / 131.0) * dps_to_rads

    pressure_adc = (s[12] | (s[13] << 8)) >> 4
    flex_adc = (s[14] | (s[15] << 8)) >> 4

    return {
        "acc_x_g": ax_g,
        "acc_y_g": ay_g,
        "acc_z_g": az_g,
        "gyr_x_rads": gx_rads,
        "gyr_y_rads": gy_rads,
        "gyr_z_rads": gz_rads,
        "pressure_adc": float(pressure_adc),
        "flex_adc": float(flex_adc),
    }


def pressure_adc_to_pa(
    pressure_adc: np.ndarray,
    calib: PressureCalibration = PressureCalibration(),
    adc_max: int = 4095,
) -> np.ndarray:
    """
    Map 12-bit ADC counts to pressure in Pa using a ratiometric min-max model.
    """
    ratio = pressure_adc / float(adc_max)
    denom = (calib.v_max_ratio - calib.v_min_ratio)
    if denom <= 0:
        raise ValueError("Invalid pressure calibration: v_max_ratio must be > v_min_ratio")

    p = ((ratio - calib.v_min_ratio) / denom) * calib.p_max_pa
    return np.clip(p, 0.0, calib.p_max_pa)


def load_arduino_raw_csv(
    path: str | Path,
    sample_rate_hz: float = 240.0,
    accel_to_mps2: bool = True,
    pressure_calib: PressureCalibration = PressureCalibration(),
) -> pd.DataFrame:
    """
    Load an Arduino raw CSV where each line contains 16 integers (bytes).
    Returns a standardized table with a sample index and a time base.
    """
    path = Path(path)

    raw = pd.read_csv(path, header=None)
    if raw.shape[1] != 16:
        raise ValueError(f"{path.name}: expected 16 columns, got {raw.shape[1]}")

    decoded = raw.apply(lambda r: decode_row_16bytes(r.values.tolist()), axis=1, result_type="expand")
    df = pd.DataFrame(decoded)

    # Time base
    df.insert(0, "sample_index", np.arange(len(df), dtype=np.int64))
    df.insert(1, "t_arduino_s", df["sample_index"].to_numpy() / float(sample_rate_hz))

    # Convert accel to m/s^2 if requested
    if accel_to_mps2:
        g0 = 9.80665
        df["acc_x"] = df["acc_x_g"] * g0
        df["acc_y"] = df["acc_y_g"] * g0
        df["acc_z"] = df["acc_z_g"] * g0
    else:
        df["acc_x"] = df["acc_x_g"]
        df["acc_y"] = df["acc_y_g"]
        df["acc_z"] = df["acc_z_g"]

    # Gyro: keep in rad/s (explicitly named in decode)
    df["gyr_x"] = df["gyr_x_rads"]
    df["gyr_y"] = df["gyr_y_rads"]
    df["gyr_z"] = df["gyr_z_rads"]

    # Pressure in Pa + keep ADC for traceability
    df["pressure"] = pressure_adc_to_pa(df["pressure_adc"].to_numpy(), calib=pressure_calib)

    # Flex: keep ADC for now; can calibrate later if needed
    df["flex"] = df["flex_adc"]

    # Final column order (Stage 0 standardised)
    keep = [
        "sample_index",
        "t_arduino_s",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "flex",
        "pressure",
        "pressure_adc",
        "flex_adc",
        "acc_x_g",
        "acc_y_g",
        "acc_z_g",
        "gyr_x_rads",
        "gyr_y_rads",
        "gyr_z_rads",
    ]
    return df[keep]

