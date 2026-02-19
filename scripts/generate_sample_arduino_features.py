from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.arduino_raw import (
    PressureCalibration,
    load_arduino_raw_csv,
    load_pressure_calibration_json,
)
from preprocessing.gyro_calibration import GyroCalibration, load_gyro_calibration_json


def _pick_existing(base_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {base_dir}: {candidates}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate processed Arduino CSVs for sample fixed/freehand runs "
            "(raw ingest + decode/calibration)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "sample_data",
        help="Directory containing raw sample Arduino CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_arduino",
        help="Directory for generated processed CSV files.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=240.0,
        help="Arduino sample rate used to construct t_arduino_s.",
    )
    parser.add_argument(
        "--accel-in-g",
        action="store_true",
        help="Keep accel columns in g (default converts acc_x/y/z to m/s^2).",
    )
    parser.add_argument(
        "--pressure-v-min-ratio",
        type=float,
        default=0.10,
        help="Pressure calibration v_min_ratio (ratiometric).",
    )
    parser.add_argument(
        "--pressure-v-max-ratio",
        type=float,
        default=0.90,
        help="Pressure calibration v_max_ratio (ratiometric).",
    )
    parser.add_argument(
        "--pressure-max-pa",
        type=float,
        default=206_000.0,
        help="Pressure calibration full-scale pressure in Pa.",
    )
    parser.add_argument(
        "--pressure-calibration-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with v_min_ratio/v_max_ratio/p_max_pa. "
            "When provided, this overrides --pressure-v-min-ratio/--pressure-v-max-ratio/--pressure-max-pa."
        ),
    )
    parser.add_argument(
        "--gyro-calibration-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with gyro bias calibration "
            "(bias_x_rads/bias_y_rads/bias_z_rads)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be generated without writing CSV files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if args.pressure_calibration_json is not None:
        calib = load_pressure_calibration_json(args.pressure_calibration_json)
    else:
        calib = PressureCalibration(
            v_min_ratio=args.pressure_v_min_ratio,
            v_max_ratio=args.pressure_v_max_ratio,
            p_max_pa=args.pressure_max_pa,
        )
    gyro_calib = (
        load_gyro_calibration_json(args.gyro_calibration_json)
        if args.gyro_calibration_json is not None
        else GyroCalibration()
    )

    fixed_input = _pick_existing(
        input_dir,
        [
            "sample_arduino_fixed_orientation.csv",
        ],
    )
    freehand_input = _pick_existing(
        input_dir,
        ["sample_arduino_freehand_manipulation.csv"],
    )

    jobs: list[tuple[Path, str]] = [
        (fixed_input, "sample_arduino_fixed_orientation_processed.csv"),
        (freehand_input, "sample_arduino_freehand_manipulation_processed.csv"),
    ]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for in_path, out_name in jobs:
        processed_df = load_arduino_raw_csv(
            in_path,
            sample_rate_hz=args.sample_rate_hz,
            accel_to_mps2=not args.accel_in_g,
            pressure_calib=calib,
            gyro_calib=gyro_calib,
        )
        out_path = output_dir / out_name

        if args.dry_run:
            accel_units = "g" if args.accel_in_g else "m/s^2"
            print(f"[dry-run] {in_path.name} -> {out_path}")
            print(
                f"[dry-run] rows={len(processed_df)} cols={len(processed_df.columns)} "
                "pipeline=ingest->decode/calibrate "
                f"sample_rate_hz={args.sample_rate_hz} accel_units={accel_units}"
            )
        else:
            processed_df.to_csv(out_path, index=False)
            print(f"[ok] wrote {out_path} ({len(processed_df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
