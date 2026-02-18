from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.arduino_raw import load_arduino_raw_csv
from preprocessing.export_dataset_hdf5 import build_run_plan, load_manifest_json
from preprocessing.optitrack_raw import load_optitrack_raw_csv


MAX_POINTS = 20_000


@dataclass
class PlotSelection:
    source_name: tk.StringVar
    x_column: tk.StringVar
    y_column: tk.StringVar


def _default_manifest_path() -> Path:
    cfg = REPO_ROOT / "configs" / "preprocessing_manifest_all_trials.json"
    if cfg.exists():
        return cfg
    return REPO_ROOT / "data" / "preprocessing_manifest_all_trials.json"


def _resolve_dataset_root(manifest: dict[str, object]) -> Path:
    dataset_root_value = manifest.get("dataset_root", ".")
    root = Path(str(dataset_root_value))
    if root.is_absolute():
        return root
    return (REPO_ROOT / root).resolve()


def _resolve_optional_path(dataset_root: Path, value: object) -> Path | None:
    if value is None:
        return None
    as_str = str(value).strip()
    if not as_str:
        return None
    p = Path(as_str)
    if p.is_absolute():
        return p
    return (dataset_root / p).resolve()


class RawInspectorApp:
    def __init__(self, root: tk.Tk, initial_manifest_path: Path | None = None) -> None:
        self.root = root
        self.root.title("Raw Arduino + OptiTrack Inspector")
        self.root.geometry("1440x860")

        self.manifest_path_var = tk.StringVar(
            value=str(initial_manifest_path) if initial_manifest_path is not None else ""
        )
        self.run_id_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Load a manifest to begin.")

        self.plot_a = PlotSelection(tk.StringVar(value="arduino"), tk.StringVar(), tk.StringVar())
        self.plot_b = PlotSelection(tk.StringVar(value="optitrack"), tk.StringVar(), tk.StringVar())

        self.run_order: list[str] = []
        self.plan_by_run_id: dict[str, object] = {}
        self.run_cache: dict[str, dict[str, pd.DataFrame]] = {}
        self.current_data: dict[str, pd.DataFrame] = {}

        self.sample_rate_hz_arduino = 240.0
        self.pressure_calib_path: Path | None = None
        self.gyro_calib_path: Path | None = None

        self._build_layout()

        if initial_manifest_path is not None and initial_manifest_path.exists():
            self.load_manifest(initial_manifest_path)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Manifest:").pack(side="left")
        ttk.Entry(top, textvariable=self.manifest_path_var, width=90).pack(
            side="left",
            padx=(6, 6),
            fill="x",
            expand=True,
        )
        ttk.Button(top, text="Browse", command=self.on_browse_manifest).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Load", command=self.on_load_manifest).pack(side="left")

        run_row = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        run_row.pack(fill="x")
        ttk.Label(run_row, text="Run ID:").pack(side="left")
        self.run_combo = ttk.Combobox(run_row, textvariable=self.run_id_var, state="readonly", width=45)
        self.run_combo.pack(side="left", padx=(6, 8))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _evt: self.on_run_selected())
        ttk.Button(run_row, text="Reload Run", command=self.reload_current_run).pack(side="left")

        controls = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        controls.pack(fill="x")
        self._build_plot_controls(controls, "Plot A", self.plot_a, column=0)
        self._build_plot_controls(controls, "Plot B", self.plot_b, column=1)

        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(fill="x", padx=8, pady=(0, 8))

    def _build_plot_controls(
        self,
        parent: ttk.Frame,
        label: str,
        selection: PlotSelection,
        *,
        column: int,
    ) -> None:
        frame = ttk.LabelFrame(parent, text=label, padding=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0))
        parent.columnconfigure(column, weight=1)

        ttk.Label(frame, text="Source").grid(row=0, column=0, sticky="w")
        source_combo = ttk.Combobox(
            frame,
            textvariable=selection.source_name,
            state="readonly",
            width=22,
            values=["arduino", "optitrack"],
        )
        source_combo.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        source_combo.bind(
            "<<ComboboxSelected>>",
            lambda _evt: self.on_source_changed(selection),
        )

        ttk.Label(frame, text="X variable").grid(row=2, column=0, sticky="w")
        x_combo = ttk.Combobox(frame, textvariable=selection.x_column, state="readonly", width=35)
        x_combo.grid(row=3, column=0, sticky="ew", pady=(0, 4))
        x_combo.bind("<<ComboboxSelected>>", lambda _evt: self.redraw())

        ttk.Label(frame, text="Y variable").grid(row=4, column=0, sticky="w")
        y_combo = ttk.Combobox(frame, textvariable=selection.y_column, state="readonly", width=35)
        y_combo.grid(row=5, column=0, sticky="ew", pady=(0, 6))
        y_combo.bind("<<ComboboxSelected>>", lambda _evt: self.redraw())

        ttk.Button(frame, text="Update Plot", command=self.redraw).grid(row=6, column=0, sticky="w")
        frame.columnconfigure(0, weight=1)

        setattr(selection, "_x_combo", x_combo)
        setattr(selection, "_y_combo", y_combo)

    def on_browse_manifest(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select preprocessing manifest JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.manifest_path_var.set(selected)

    def on_load_manifest(self) -> None:
        path = Path(self.manifest_path_var.get().strip())
        self.load_manifest(path)

    def load_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            messagebox.showerror("Manifest not found", f"File does not exist:\n{manifest_path}")
            return

        try:
            manifest = load_manifest_json(manifest_path)
            run_plan = build_run_plan(manifest)
        except Exception as exc:
            messagebox.showerror("Manifest load failed", f"Could not parse manifest:\n{exc}")
            return

        if not run_plan:
            messagebox.showwarning("No runs", "No export-enabled runs found in this manifest.")
            return

        dataset_root = _resolve_dataset_root(manifest)
        default_settings = manifest.get("default_settings", {})
        sync_defaults = default_settings if isinstance(default_settings, dict) else {}
        self.sample_rate_hz_arduino = float(sync_defaults.get("sample_rate_hz_arduino", 240.0))

        calibration_paths = sync_defaults.get("calibration_paths", {})
        if isinstance(calibration_paths, dict):
            self.pressure_calib_path = _resolve_optional_path(dataset_root, calibration_paths.get("pressure"))
            self.gyro_calib_path = _resolve_optional_path(dataset_root, calibration_paths.get("gyro"))
        else:
            self.pressure_calib_path = None
            self.gyro_calib_path = None

        self.plan_by_run_id = {p.run_id: p for p in run_plan}
        self.run_order = [p.run_id for p in run_plan]
        self.run_cache.clear()
        self.current_data = {}

        self.run_combo["values"] = self.run_order
        self.run_id_var.set(self.run_order[0])
        self.on_run_selected()
        self.manifest_path_var.set(str(manifest_path))

    def reload_current_run(self) -> None:
        run_id = self.run_id_var.get().strip()
        if not run_id:
            return
        self.run_cache.pop(run_id, None)
        self.on_run_selected()

    def on_run_selected(self) -> None:
        run_id = self.run_id_var.get().strip()
        if not run_id:
            return
        if run_id not in self.plan_by_run_id:
            messagebox.showerror("Unknown run", f"Run ID not found in loaded manifest:\n{run_id}")
            return

        try:
            loaded = self._load_run_data(run_id)
        except Exception as exc:
            messagebox.showerror("Run load failed", f"Failed to load raw run data:\n{exc}")
            return

        self.current_data = loaded
        self.on_source_changed(self.plot_a)
        self.on_source_changed(self.plot_b)
        self.redraw()
        self.status_var.set(
            f"Loaded {run_id}: arduino={len(loaded['arduino'])} rows, "
            f"optitrack={len(loaded['optitrack'])} rows."
        )

    def _load_run_data(self, run_id: str) -> dict[str, pd.DataFrame]:
        if run_id in self.run_cache:
            return self.run_cache[run_id]

        plan = self.plan_by_run_id[run_id]
        arduino_df = load_arduino_raw_csv(
            plan.arduino_raw_csv,
            sample_rate_hz=self.sample_rate_hz_arduino,
            accel_to_mps2=True,
            pressure_calib_json_path=self.pressure_calib_path,
            gyro_calib_json_path=self.gyro_calib_path,
        )
        optitrack_df = load_optitrack_raw_csv(plan.optitrack_raw_csv)
        loaded = {"arduino": arduino_df, "optitrack": optitrack_df}
        self.run_cache[run_id] = loaded
        return loaded

    def on_source_changed(self, selection: PlotSelection) -> None:
        source = selection.source_name.get().strip()
        if source not in {"arduino", "optitrack"}:
            return
        if source not in self.current_data:
            return
        df = self.current_data[source]
        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numeric_cols:
            numeric_cols = list(df.columns)
        if not numeric_cols:
            return

        x_combo: ttk.Combobox = getattr(selection, "_x_combo")
        y_combo: ttk.Combobox = getattr(selection, "_y_combo")
        x_combo["values"] = numeric_cols
        y_combo["values"] = numeric_cols

        if source == "arduino":
            x_default = "sample_index" if "sample_index" in numeric_cols else numeric_cols[0]
            y_default = "acc_x" if "acc_x" in numeric_cols else numeric_cols[0]
        else:
            x_default = "Time" if "Time" in numeric_cols else numeric_cols[0]
            y_default = "BR_X" if "BR_X" in numeric_cols else numeric_cols[0]

        if y_default == x_default and len(numeric_cols) > 1:
            y_default = numeric_cols[1]

        selection.x_column.set(x_default)
        selection.y_column.set(y_default)
        self.redraw()

    def redraw(self) -> None:
        run_id = self.run_id_var.get().strip()
        self._draw_single_plot(self.axes[0], self.plot_a, label="Plot A", run_id=run_id)
        self._draw_single_plot(self.axes[1], self.plot_b, label="Plot B", run_id=run_id)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _draw_single_plot(
        self,
        ax: Axes,
        selection: PlotSelection,
        *,
        label: str,
        run_id: str,
    ) -> None:
        ax.clear()
        source = selection.source_name.get().strip()
        x_col = selection.x_column.get().strip()
        y_col = selection.y_column.get().strip()

        if source not in self.current_data:
            ax.set_title(f"{label}: load a run")
            ax.grid(alpha=0.2)
            return

        df = self.current_data[source]
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            ax.set_title(f"{label}: select x/y variables")
            ax.grid(alpha=0.2)
            return

        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            ax.set_title(f"{label}: no finite data")
            ax.grid(alpha=0.2)
            return

        x = x[mask]
        y = y[mask]
        if len(x) > MAX_POINTS:
            stride = max(1, len(x) // MAX_POINTS)
            x = x[::stride]
            y = y[::stride]

        ax.plot(x, y, linewidth=1.0)
        ax.set_title(f"{label}: {run_id} ({source})")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect raw Arduino and raw OptiTrack data from the preprocessing "
            "manifest with interactive dropdown plotting."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_default_manifest_path(),
        help="Path to preprocessing manifest JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    app = RawInspectorApp(root, initial_manifest_path=args.manifest_path)
    _ = app
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
