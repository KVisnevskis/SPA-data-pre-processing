from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import math

from preprocessing.gyro_calibration import (
    GyroCalibration,
    apply_gyro_bias_correction,
    load_gyro_calibration_json,
)


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


def _pressure_calibration_from_payload(payload: dict[str, object]) -> PressureCalibration:
    """
    Accept either:
      1) flat schema: {"v_min_ratio": ..., "v_max_ratio": ..., "p_max_pa": ...}
      2) fit artifact schema with nested "equivalent_ratiometric"
    """
    if all(k in payload for k in ("v_min_ratio", "v_max_ratio", "p_max_pa")):
        source = payload
    elif "equivalent_ratiometric" in payload and isinstance(payload["equivalent_ratiometric"], dict):
        source = payload["equivalent_ratiometric"]  # type: ignore[assignment]
    else:
        raise ValueError(
            "Calibration JSON must contain either top-level keys "
            "v_min_ratio/v_max_ratio/p_max_pa or nested equivalent_ratiometric"
        )

    try:
        v_min_ratio = float(source["v_min_ratio"])  # type: ignore[index]
        v_max_ratio = float(source["v_max_ratio"])  # type: ignore[index]
        p_max_pa = float(source["p_max_pa"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid calibration JSON values; expected numeric v_min_ratio/v_max_ratio/p_max_pa") from exc

    if v_max_ratio <= v_min_ratio:
        raise ValueError("Invalid calibration JSON: v_max_ratio must be > v_min_ratio")
    if p_max_pa <= 0.0:
        raise ValueError("Invalid calibration JSON: p_max_pa must be > 0")

    return PressureCalibration(v_min_ratio=v_min_ratio, v_max_ratio=v_max_ratio, p_max_pa=p_max_pa)


def load_pressure_calibration_json(path: str | Path) -> PressureCalibration:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON root must be an object")
    return _pressure_calibration_from_payload(payload)


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
    clip: bool = False,
) -> np.ndarray:
    """
    Map 12-bit ADC counts to pressure in Pa using a ratiometric min-max model.
    By default this does not clip, so values outside the calibrated ADC span
    are linearly extrapolated.
    """
    ratio = pressure_adc / float(adc_max)
    denom = (calib.v_max_ratio - calib.v_min_ratio)
    if denom <= 0:
        raise ValueError("Invalid pressure calibration: v_max_ratio must be > v_min_ratio")

    p = ((ratio - calib.v_min_ratio) / denom) * calib.p_max_pa
    if clip:
        return np.clip(p, 0.0, calib.p_max_pa)
    return p


def load_arduino_raw_csv(
    path: str | Path,
    sample_rate_hz: float = 240.0,
    accel_to_mps2: bool = True,
    pressure_calib: PressureCalibration = PressureCalibration(),
    pressure_calib_json_path: str | Path | None = None,
    gyro_calib: GyroCalibration = GyroCalibration(),
    gyro_calib_json_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load an Arduino raw CSV where each line contains 16 integers (bytes).
    Returns a standardized table with a sample index and a time base.
    """
    path = Path(path)
    if pressure_calib_json_path is not None:
        pressure_calib = load_pressure_calibration_json(pressure_calib_json_path)
    if gyro_calib_json_path is not None:
        gyro_calib = load_gyro_calibration_json(gyro_calib_json_path)

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
    gyr_x, gyr_y, gyr_z = apply_gyro_bias_correction(
        df["gyr_x_rads"].to_numpy(dtype=np.float64),
        df["gyr_y_rads"].to_numpy(dtype=np.float64),
        df["gyr_z_rads"].to_numpy(dtype=np.float64),
        calibration=gyro_calib,
    )
    df["gyr_x"] = gyr_x
    df["gyr_y"] = gyr_y
    df["gyr_z"] = gyr_z

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

