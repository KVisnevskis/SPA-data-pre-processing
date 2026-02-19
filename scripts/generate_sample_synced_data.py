from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.time_sync import synchronize_fixed_orientation, synchronize_freehand


def _pick_existing(base_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {base_dir}: {candidates}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Stage-3 synchronized sample CSVs (fixed + freehand)."
    )
    parser.add_argument(
        "--arduino-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_arduino",
        help="Directory containing processed Arduino sample CSVs.",
    )
    parser.add_argument(
        "--optitrack-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_optitrack",
        help="Directory containing processed OptiTrack sample CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_synced",
        help="Directory for synchronized Stage-3 sample CSVs.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Optional absolute lag bound in samples for cross-correlation search.",
    )
    parser.add_argument(
        "--fixed-phi-transform",
        choices=["auto", "none", "invert", "abs"],
        default="auto",
        help=(
            "How to transform phi for fixed-run cross-correlation. "
            "'auto' picks the highest-correlation candidate among none/invert/abs."
        ),
    )
    parser.add_argument(
        "--freehand-rotation-direction",
        choices=["world_to_body", "body_to_world"],
        default="world_to_body",
        help="Rotation convention used to generate virtual accelerometer for freehand sync.",
    )
    parser.add_argument(
        "--sync-log-path",
        type=Path,
        default=None,
        help="Optional JSON path for synchronization diagnostics log.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be generated without writing CSV files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    arduino_dir: Path = args.arduino_dir
    optitrack_dir: Path = args.optitrack_dir
    output_dir: Path = args.output_dir
    log_path: Path | None = args.sync_log_path

    if log_path is None:
        log_path = output_dir / "sync_stage3_log.json"

    fixed_arduino = _pick_existing(
        arduino_dir,
        [
            "sample_arduino_fixed_orientation_processed.csv",
        ],
    )
    freehand_arduino = _pick_existing(
        arduino_dir,
        ["sample_arduino_freehand_manipulation_processed.csv"],
    )

    fixed_optitrack = _pick_existing(
        optitrack_dir,
        [
            "sample_optitrack_fixed_orientation_processed.csv",
        ],
    )
    freehand_optitrack = _pick_existing(
        optitrack_dir,
        ["sample_optitrack_freehand_manipulation_processed.csv"],
    )

    jobs: list[tuple[str, Path, Path, str]] = [
        (
            "fixed",
            fixed_arduino,
            fixed_optitrack,
            "sample_synced_fixed_orientation_stage3.csv",
        ),
        (
            "freehand",
            freehand_arduino,
            freehand_optitrack,
            "sample_synced_freehand_manipulation_stage3.csv",
        ),
    ]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    sync_log: dict[str, object] = {}

    for mode, arduino_path, optitrack_path, out_name in jobs:
        arduino_df = pd.read_csv(arduino_path)
        optitrack_df = pd.read_csv(optitrack_path)

        if mode == "fixed":
            synced_df, info = synchronize_fixed_orientation(
                arduino_df,
                optitrack_df,
                phi_transform=args.fixed_phi_transform,
                max_lag=args.max_lag,
                return_info=True,
            )
        else:
            synced_df, info = synchronize_freehand(
                arduino_df,
                optitrack_df,
                max_lag=args.max_lag,
                rotation_direction=args.freehand_rotation_direction,
                return_info=True,
            )

        out_path = output_dir / out_name
        sync_log[mode] = {
            "arduino_input": str(arduino_path),
            "optitrack_input": str(optitrack_path),
            "output": str(out_path),
            **info,
        }

        if args.dry_run:
            print(f"[dry-run] {mode}: {arduino_path.name} + {optitrack_path.name} -> {out_path}")
            print(
                f"[dry-run] rows={len(synced_df)} "
                f"shift={info['sample_shift_arduino']} "
                f"corr={info['max_cross_correlation']:.6f} "
                f"trimmed={info['rows_trimmed']}"
            )
        else:
            synced_df.to_csv(out_path, index=False)
            print(f"[ok] wrote {out_path} ({len(synced_df)} rows)")

    if args.dry_run:
        print(f"[dry-run] sync log -> {log_path}")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(sync_log, indent=2), encoding="utf-8")
        print(f"[ok] wrote {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
