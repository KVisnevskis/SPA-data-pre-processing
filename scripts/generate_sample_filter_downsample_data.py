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

from preprocessing.filter_downsample import filter_and_downsample


def _pick_existing(base_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {base_dir}: {candidates}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate downsampled sample CSVs (moving-average filter + decimation) "
            "from Stage-3 synchronized sample data."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_synced",
        help="Directory containing Stage-3 synchronized sample CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_downsample",
        help="Directory where downsampled CSVs will be written.",
    )
    parser.add_argument(
        "--input-sample-rate-hz",
        type=float,
        default=240.0,
        help="Input sample rate (Hz) used by Stage-3 synchronized data.",
    )
    parser.add_argument(
        "--decimation-factor",
        type=int,
        default=5,
        help="Integer decimation factor (default 5 gives 240 Hz -> 48 Hz).",
    )
    parser.add_argument(
        "--decimation-offset",
        type=int,
        default=0,
        help="Decimation anchor index in [0, decimation_factor - 1].",
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=5,
        help="Moving-average window size in samples.",
    )
    parser.add_argument(
        "--filter-alignment",
        choices=["centered", "causal"],
        default="centered",
        help="Moving-average alignment policy.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="Time",
        help="Time column to optionally rebase after downsampling.",
    )
    parser.add_argument(
        "--no-rebase-time",
        action="store_true",
        help="Do not overwrite the time column with a rebased 0..T timeline.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional JSON path for filter/downsample diagnostics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be generated without writing files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    log_path: Path | None = args.log_path

    if log_path is None:
        log_path = output_dir / "filter_downsample_log.json"

    fixed_input = _pick_existing(
        input_dir,
        [
            "sample_synced_fixed_orientation_stage3.csv",
        ],
    )
    freehand_input = _pick_existing(
        input_dir,
        ["sample_synced_freehand_manipulation_stage3.csv"],
    )

    jobs: list[tuple[str, Path, str]] = [
        ("fixed", fixed_input, "sample_filtered_downsampled_fixed_orientation.csv"),
        ("freehand", freehand_input, "sample_filtered_downsampled_freehand_manipulation.csv"),
    ]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    downsample_log: dict[str, object] = {}
    for run_mode, input_path, output_name in jobs:
        synced_df = pd.read_csv(input_path)
        downsampled_df, info = filter_and_downsample(
            synced_df,
            input_sample_rate_hz=args.input_sample_rate_hz,
            decimation_factor=args.decimation_factor,
            decimation_offset=args.decimation_offset,
            moving_average_window=args.moving_average_window,
            filter_alignment=args.filter_alignment,
            time_col=args.time_col,
            rebase_time=not args.no_rebase_time,
            return_info=True,
        )

        output_path = output_dir / output_name
        downsample_log[run_mode] = {
            "input": str(input_path),
            "output": str(output_path),
            "parameters": {
                "input_sample_rate_hz": args.input_sample_rate_hz,
                "decimation_factor": args.decimation_factor,
                "decimation_offset": args.decimation_offset,
                "moving_average_window": args.moving_average_window,
                "filter_alignment": args.filter_alignment,
                "time_col": args.time_col,
                "rebase_time": not args.no_rebase_time,
            },
            "diagnostics": info,
        }

        if args.dry_run:
            print(f"[dry-run] {run_mode}: {input_path.name} -> {output_path}")
            print(
                f"[dry-run] rows_before={info['rows_before']} "
                f"rows_after={info['rows_after']} "
                f"output_sample_rate_hz={info['output_sample_rate_hz']:.3f}"
            )
        else:
            downsampled_df.to_csv(output_path, index=False)
            print(f"[ok] wrote {output_path} ({len(downsampled_df)} rows)")

    if args.dry_run:
        print(f"[dry-run] downsample log -> {log_path}")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(downsample_log, indent=2), encoding="utf-8")
        print(f"[ok] wrote {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
