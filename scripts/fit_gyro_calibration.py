from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.arduino_raw import load_arduino_raw_csv
from preprocessing.gyro_calibration import fit_gyro_bias_from_dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit gyro bias calibration from a stationary raw Arduino run "
            "(e.g., fixed orientation sample)."
        )
    )
    parser.add_argument(
        "--input-raw",
        type=Path,
        default=REPO_ROOT / "sample_data" / "sample_arduino_fixed_orientation.csv",
        help="Path to raw Arduino CSV used for stationary gyro bias fit.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=240.0,
        help="Arduino sample rate used during decoding.",
    )
    parser.add_argument(
        "--method",
        choices=("mean", "median"),
        default="mean",
        help="Bias estimator across stationary samples.",
    )
    parser.add_argument(
        "--fit-json",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_calibration" / "sample_gyro_calibration_fit.json",
        help="Detailed fit artifact path.",
    )
    parser.add_argument(
        "--module-calibration-json",
        type=Path,
        default=REPO_ROOT / "calibration" / "arduino_gyro_calibration.json",
        help="Minimal JSON for direct use by preprocessing.arduino_raw.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    df = load_arduino_raw_csv(args.input_raw, sample_rate_hz=args.sample_rate_hz, accel_to_mps2=True)
    calib, stats = fit_gyro_bias_from_dataframe(
        df,
        gyro_columns=("gyr_x_rads", "gyr_y_rads", "gyr_z_rads"),
        method=args.method,
    )

    fit_payload = {
        "input_raw": str(args.input_raw),
        "sample_rate_hz": float(args.sample_rate_hz),
        "method": args.method,
        "gyro_calibration": {
            "bias_x_rads": float(calib.bias_x_rads),
            "bias_y_rads": float(calib.bias_y_rads),
            "bias_z_rads": float(calib.bias_z_rads),
        },
        "stats": stats,
    }
    args.fit_json.parent.mkdir(parents=True, exist_ok=True)
    with args.fit_json.open("w", encoding="utf-8") as f:
        json.dump(fit_payload, f, indent=2)

    module_payload = {
        "bias_x_rads": float(calib.bias_x_rads),
        "bias_y_rads": float(calib.bias_y_rads),
        "bias_z_rads": float(calib.bias_z_rads),
        "source_fit_json": str(args.fit_json),
    }
    args.module_calibration_json.parent.mkdir(parents=True, exist_ok=True)
    with args.module_calibration_json.open("w", encoding="utf-8") as f:
        json.dump(module_payload, f, indent=2)

    print(f"[ok] fit json: {args.fit_json}")
    print(f"[ok] module calibration json: {args.module_calibration_json}")
    print(
        "[ok] gyro bias (rad/s): "
        f"x={calib.bias_x_rads:.8f}, y={calib.bias_y_rads:.8f}, z={calib.bias_z_rads:.8f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
