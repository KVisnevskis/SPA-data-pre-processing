#!/usr/bin/env python3
"""
Analyze missing samples in OptiTrack CSV data.

For each CSV file in a given folder:
- Count total samples (rows)
- Count rows containing at least one NaN (missing samples), considering ONLY
  columns up to Excel column 'BO' (first 67 columns, 1-based)
- Compute percentage of missing samples
- Find the longest run of consecutive missing samples
- List all contiguous runs ("gaps") of missing samples

Results are:
- Printed to the terminal
- Saved to a text log file: missing_samples_log.txt
"""

import argparse
from pathlib import Path

import pandas as pd


# Excel column 'BO' is the 67th column (1-based).
# We'll therefore use only the first 67 columns (0..66 in 0-based indexing).
EXCEL_BO_1BASED = 67


def analyze_file(path: Path):
    """
    Analyze a single CSV file for missing samples.

    A row is considered a "missing sample" if the 3rd column
    (data column after Frame and Time) is NaN.
    """
    try:
        # Use the proper header row (line 5 in the raw file: 0-based index)
        df = pd.read_csv(path, header=5)
    except Exception as e:
        return (
            {
                "file": path.name,
                "total_samples": 0,
                "missing_samples": 0,
                "pct_missing": 0.0,
                "longest_missing_run": 0,
                "error": str(e),
            },
            [],
        )

    total_samples = len(df)
    if total_samples == 0:
        return (
            {
                "file": path.name,
                "total_samples": 0,
                "missing_samples": 0,
                "pct_missing": 0.0,
                "longest_missing_run": 0,
                "error": "",
            },
            [],
        )

    # Now the columns should look like:
    # ['Frame', 'Time (Seconds)', 'X', 'Y', 'Z', 'W', 'X.1', 'Y.1', ...]
    if df.shape[1] < 3:
        missing_mask = pd.Series(False, index=df.index)
    else:
        third_col = df.columns[2]      # typically 'X'
        missing_mask = df[third_col].isna()

    missing_samples = int(missing_mask.sum())
    pct_missing = (missing_samples / total_samples) * 100.0 if total_samples > 0 else 0.0

    # --- contiguous run detection exactly as in your script ---
    runs = []
    in_run = False
    run_start = None
    run_len = 0
    run_id = 0

    for idx, is_missing in enumerate(missing_mask.values):
        if is_missing:
            if not in_run:
                in_run = True
                run_start = idx
                run_len = 1
                run_id += 1
            else:
                run_len += 1
        else:
            if in_run:
                runs.append(
                    {
                        "run_id": run_id,
                        "start_row": run_start,
                        "end_row": idx - 1,
                        "length": run_len,
                    }
                )
                in_run = False
                run_start = None
                run_len = 0

    if in_run:
        runs.append(
            {
                "run_id": run_id,
                "start_row": run_start,
                "end_row": total_samples - 1,
                "length": run_len,
            }
        )

    longest_missing_run = max((r["length"] for r in runs), default=0)

    summary = {
        "file": path.name,
        "total_samples": int(total_samples),
        "missing_samples": missing_samples,
        "pct_missing": pct_missing,
        "longest_missing_run": int(longest_missing_run),
        "error": "",
    }

    return summary, runs


def format_summary_table(summaries):
    """
    Create a nicely formatted text table for per-file summaries.
    Returns a list of strings (lines).
    """
    if not summaries:
        return ["No files to summarize."]

    headers = ["File", "Total", "Missing", "% Missing", "Longest Run", "Error"]

    file_width = max(len(headers[0]), max(len(s["file"]) for s in summaries))
    total_width = max(len(headers[1]), max(len(str(s["total_samples"])) for s in summaries))
    missing_width = max(len(headers[2]), max(len(str(s["missing_samples"])) for s in summaries))
    pct_width = len(headers[3])
    longest_width = max(len(headers[4]), max(len(str(s["longest_missing_run"])) for s in summaries))
    error_width = max(len(headers[5]), max(len(s["error"]) for s in summaries))

    line_fmt = (
        f"{{:<{file_width}}}  "
        f"{{:>{total_width}}}  "
        f"{{:>{missing_width}}}  "
        f"{{:>{pct_width}}}  "
        f"{{:>{longest_width}}}  "
        f"{{:<{error_width}}}"
    )

    lines = []
    lines.append(line_fmt.format(*headers))
    lines.append(
        "-" * (
            file_width
            + total_width
            + missing_width
            + pct_width
            + longest_width
            + error_width
            + 10
        )
    )

    for s in summaries:
        lines.append(
            line_fmt.format(
                s["file"],
                s["total_samples"],
                s["missing_samples"],
                f"{s['pct_missing']:.2f}",
                s["longest_missing_run"],
                s["error"],
            )
        )

    return lines


def format_runs_table(runs):
    """
    Create a formatted text table for per-run missing sequences.
    Returns a list of strings (lines).
    """
    if not runs:
        return ["No missing-sample runs found in any file."]

    headers = ["File", "Run ID", "Start Row", "End Row", "Length"]

    file_width = max(len(headers[0]), max(len(r["file"]) for r in runs))
    runid_width = max(len(headers[1]), max(len(str(r["run_id"])) for r in runs))
    start_width = max(len(headers[2]), max(len(str(r["start_row"])) for r in runs))
    end_width = max(len(headers[3]), max(len(str(r["end_row"])) for r in runs))
    len_width = max(len(headers[4]), max(len(str(r["length"])) for r in runs))

    line_fmt = (
        f"{{:<{file_width}}}  "
        f"{{:>{runid_width}}}  "
        f"{{:>{start_width}}}  "
        f"{{:>{end_width}}}  "
        f"{{:>{len_width}}}"
    )

    lines = []
    lines.append(line_fmt.format(*headers))
    lines.append(
        "-" * (
            file_width
            + runid_width
            + start_width
            + end_width
            + len_width
            + 8
        )
    )

    for r in runs:
        lines.append(
            line_fmt.format(
                r["file"],
                r["run_id"],
                r["start_row"],
                r["end_row"],
                r["length"],
            )
        )

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Investigate missing samples in OptiTrack CSV data."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing CSV files (one trial per file).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for files (default: *.csv).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="missing_samples_log.txt",
        help="Name of the text log file to create (default: missing_samples_log.txt).",
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory or does not exist.")
        return

    files = sorted(folder.glob(args.pattern))
    if not files:
        print(f"No files matching pattern '{args.pattern}' found in {folder}")
        return

    summaries = []
    runs = []

    for path in files:
        summary, file_runs = analyze_file(path)
        summaries.append(summary)
        for r in file_runs:
            r_with_file = {"file": path.name}
            r_with_file.update(r)
            runs.append(r_with_file)

    log_lines = []

    log_lines.append(f"Analyzing {len(files)} files in {folder.resolve()}\n")
    log_lines.append(
        f"NaN detection uses ONLY columns up to Excel 'BO' (first {EXCEL_BO_1BASED} columns).\n"
    )

    # Per-file summary
    log_lines.append("Per-file summary of missing samples:\n")
    summary_lines = format_summary_table(summaries)
    log_lines.extend(summary_lines)
    log_lines.append("")

    # Per-run missing sequences
    log_lines.append("Per-run (contiguous) missing-sample sequences:\n")
    runs_lines = format_runs_table(runs)
    log_lines.extend(runs_lines)

    # Print to terminal
    for line in log_lines:
        print(line)

    # Save to log file
    log_path = folder / args.log_file
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
