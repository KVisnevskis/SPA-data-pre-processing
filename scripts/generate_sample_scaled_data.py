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

from preprocessing.global_scaling import (
    DEFAULT_SCALE_COLUMNS,
    fit_and_apply_global_min_max_scaling,
)


def _pick_existing(base_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {base_dir}: {candidates}")


def _resolve_default_input_dir() -> Path:
    candidates = [
        REPO_ROOT / "sample_data" / "processed_downsample",
        REPO_ROOT / "sample_data" / "processed_stage4",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Stage-5 scaled sample CSVs from downsampled sample data "
            "using global min-max scaling to [-1, 1]."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_resolve_default_input_dir(),
        help="Directory containing Stage-4/downsampled sample CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_scaled",
        help="Directory where Stage-5 scaled CSVs will be written.",
    )
    parser.add_argument(
        "--scale-columns",
        nargs="+",
        default=list(DEFAULT_SCALE_COLUMNS),
        help="Columns to scale with one global scaler across all runs.",
    )
    parser.add_argument(
        "--scaler-params-path",
        type=Path,
        default=None,
        help="Optional JSON path for fitted global scaler parameters.",
    )
    parser.add_argument(
        "--scaling-log-path",
        type=Path,
        default=None,
        help="Optional JSON path for scaling diagnostics log.",
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

    scaler_params_path: Path | None = args.scaler_params_path
    scaling_log_path: Path | None = args.scaling_log_path
    if scaler_params_path is None:
        scaler_params_path = output_dir / "scaler_parameters.json"
    if scaling_log_path is None:
        scaling_log_path = output_dir / "scaling_log.json"

    fixed_input = _pick_existing(
        input_dir,
        [
            "sample_filtered_downsampled_fixed_orientation.csv",
            "sample_stage4_fixed_orientation.csv",
        ],
    )
    freehand_input = _pick_existing(
        input_dir,
        [
            "sample_filtered_downsampled_freehand_manipulation.csv",
            "sample_stage4_freehand_manipulation.csv",
        ],
    )

    run_tables = {
        "fixed": pd.read_csv(fixed_input),
        "freehand": pd.read_csv(freehand_input),
    }
    scaled_runs, scaler_parameters, scaling_info = fit_and_apply_global_min_max_scaling(
        run_tables,
        scale_columns=args.scale_columns,
        return_info=True,
    )

    jobs: list[tuple[str, str]] = [
        ("fixed", "sample_scaled_fixed_orientation.csv"),
        ("freehand", "sample_scaled_freehand_manipulation.csv"),
    ]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    scaling_log: dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "scale_columns": list(args.scale_columns),
        "runs": {},
    }

    for run_id, output_name in jobs:
        scaled_df = scaled_runs[run_id]
        output_path = output_dir / output_name
        run_info = scaling_info["runs"][run_id]
        scaling_log["runs"][run_id] = {
            "input": str(fixed_input if run_id == "fixed" else freehand_input),
            "output": str(output_path),
            "diagnostics": run_info,
        }

        if args.dry_run:
            print(
                f"[dry-run] {run_id}: "
                f"{Path(scaling_log['runs'][run_id]['input']).name} -> {output_path}"
            )
            print(
                f"[dry-run] rows={len(scaled_df)} "
                f"scaled_columns={','.join(run_info['scaled_columns'])}"
            )
        else:
            scaled_df.to_csv(output_path, index=False)
            print(f"[ok] wrote {output_path} ({len(scaled_df)} rows)")

    if args.dry_run:
        print(f"[dry-run] scaler parameters -> {scaler_params_path}")
        print(f"[dry-run] scaling log -> {scaling_log_path}")
    else:
        scaler_params_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_params_path.write_text(
            json.dumps(
                {
                    "scale_columns": list(args.scale_columns),
                    "scaler_parameters": scaler_parameters,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        scaling_log_path.parent.mkdir(parents=True, exist_ok=True)
        scaling_log_path.write_text(json.dumps(scaling_log, indent=2), encoding="utf-8")
        print(f"[ok] wrote {scaler_params_path}")
        print(f"[ok] wrote {scaling_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
