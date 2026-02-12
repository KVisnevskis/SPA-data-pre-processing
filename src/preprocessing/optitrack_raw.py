from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


# OptiTrack export column indices for:
# Time, Base quat (x,y,z,w), Base pos (x,y,z), Tip quat (x,y,z,w), Tip pos (x,y,z)
DEFAULT_USECOLS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 26, 27, 28, 29, 30, 31, 32]

DEFAULT_COLUMNS: list[str] = [
    "Time",
    "BR_X", "BR_Y", "BR_Z", "BR_W",
    "BP_X", "BP_Y", "BP_Z",
    "TR_X", "TR_Y", "TR_Z", "TR_W",
    "TP_X", "TP_Y", "TP_Z",
]


def find_optitrack_header_row(path: str | Path, marker: str = "Frame,Time (Seconds)") -> int:
    """
    Returns the line index (0-based) of the OptiTrack CSV header row that starts with:
    'Frame,Time (Seconds),...'
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith(marker):
                return i
            # Safety: OptiTrack header is typically very short
            if i > 200:
                break
    raise ValueError(f"Could not find OptiTrack header row starting with: {marker}")


def load_optitrack_raw_csv(
    path: str | Path,
    usecols: Sequence[int] = DEFAULT_USECOLS,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """
    Load OptiTrack CSV export and return a standardized table with:
      Time, BR_*, BP_*, TR_*, TP_*
    """
    path = Path(path)
    header_row = find_optitrack_header_row(path)

    df = pd.read_csv(path, skiprows=header_row, usecols=list(usecols))
    if len(columns) != df.shape[1]:
        raise ValueError(f"Expected {len(columns)} columns, got {df.shape[1]}")

    df = df.set_axis(list(columns), axis=1)

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
