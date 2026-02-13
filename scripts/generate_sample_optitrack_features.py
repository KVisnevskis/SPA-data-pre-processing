from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.optitrack_compute_angle import compute_optitrack_relative_features
from preprocessing.optitrack_raw import load_optitrack_raw_csv, repair_optitrack_missing_samples


def _pick_existing(base_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist under {base_dir}: {candidates}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate processed OptiTrack CSVs for sample fixed/freehand runs "
            "(raw ingest + angle/displacement computation)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "sample_data",
        help="Directory containing raw sample OptiTrack CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sample_data" / "processed_optitrack",
        help="Directory for generated processed CSV files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if missing/non-finite values remain after Stage-1 repair "
            "(forward-fill)."
        ),
    )
    parser.add_argument(
        "--renormalize-quaternions",
        action="store_true",
        help="Renormalize base/tip quaternions during Stage-1 repair.",
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

    fixed_input = _pick_existing(
        input_dir,
        [
            "sample_optitrack_fixed_orientation.csv",
            "sample_optitrack_fixed_orientaion.csv",
        ],
    )
    freehand_input = _pick_existing(
        input_dir,
        ["sample_optitrack_freehand_manipulation.csv"],
    )

    jobs: list[tuple[Path, str]] = [
        (fixed_input, "sample_optitrack_fixed_orientation_processed.csv"),
        (freehand_input, "sample_optitrack_freehand_manipulation_processed.csv"),
    ]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for in_path, out_name in jobs:
        raw_df = load_optitrack_raw_csv(in_path)
        repaired_df = repair_optitrack_missing_samples(
            raw_df,
            renormalize_quaternions=args.renormalize_quaternions,
            strict=args.strict,
        )
        processed_df = compute_optitrack_relative_features(
            repaired_df,
            strict=args.strict,
        )
        out_path = output_dir / out_name

        if args.dry_run:
            print(f"[dry-run] {in_path.name} -> {out_path}")
            print(
                f"[dry-run] rows={len(processed_df)} cols={len(processed_df.columns)} "
                "pipeline=ingest->repair(ffill)->angles "
                "added=phi,theta,psi,dx,dy,dz"
            )
        else:
            processed_df.to_csv(out_path, index=False)
            print(f"[ok] wrote {out_path} ({len(processed_df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
