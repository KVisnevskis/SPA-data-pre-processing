from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.export_dataset_hdf5 import (
    export_dataset_hdf5_from_manifest,
    preview_run_plan,
)


def _default_manifest_path() -> Path:
    cfg = REPO_ROOT / "configs" / "preprocessing_manifest_all_trials.json"
    if cfg.exists():
        return cfg
    return REPO_ROOT / "data" / "preprocessing_manifest_all_trials.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export final preprocessed run tables to a single HDF5 file "
            "using a preprocessing manifest."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_default_manifest_path(),
        help="Path to preprocessing manifest JSON.",
    )
    parser.add_argument(
        "--output-hdf5",
        type=Path,
        default=REPO_ROOT / "outputs" / "preprocessed_all_trials.h5",
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=None,
        help="Optional subset of run IDs to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output HDF5 if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selected runs/settings without processing or writing HDF5.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.dry_run:
        preview = preview_run_plan(args.manifest_path, include_run_ids=args.run_ids)
        print(json.dumps(preview, indent=2))
        return 0

    report = export_dataset_hdf5_from_manifest(
        args.manifest_path,
        args.output_hdf5,
        include_run_ids=args.run_ids,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
