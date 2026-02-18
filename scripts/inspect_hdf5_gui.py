from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd


MAX_POINTS = 20_000


@dataclass
class PlotSelection:
    dataset_key: tk.StringVar
    x_column: tk.StringVar
    y_column: tk.StringVar


class HDF5InspectorApp:
    def __init__(self, root: tk.Tk, initial_hdf_path: Path | None = None) -> None:
        self.root = root
        self.root.title("HDF5 Trial Inspector")
        self.root.geometry("1400x820")

        self.hdf_path = tk.StringVar(value=str(initial_hdf_path) if initial_hdf_path else "")
        self.status_text = tk.StringVar(value="Load an HDF5 file to begin.")

        self.dataset_keys: list[str] = []
        self.df_cache: dict[str, pd.DataFrame] = {}

        self.plot_a = PlotSelection(tk.StringVar(), tk.StringVar(), tk.StringVar())
        self.plot_b = PlotSelection(tk.StringVar(), tk.StringVar(), tk.StringVar())

        self._build_layout()

        if initial_hdf_path and initial_hdf_path.exists():
            self.load_file(initial_hdf_path)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="HDF5 file:").pack(side="left")
        ttk.Entry(top, textvariable=self.hdf_path, width=100).pack(side="left", padx=(6, 6), fill="x", expand=True)
        ttk.Button(top, text="Browse", command=self.on_browse).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Load", command=self.on_load).pack(side="left")

        controls = ttk.Frame(self.root, padding=(8, 4, 8, 4))
        controls.pack(fill="x")

        self._build_plot_controls(controls, "Plot A", self.plot_a, 0)
        self._build_plot_controls(controls, "Plot B", self.plot_b, 1)

        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

        status = ttk.Label(self.root, textvariable=self.status_text, relief="sunken", anchor="w")
        status.pack(fill="x", padx=8, pady=(0, 8))

    def _build_plot_controls(
        self,
        parent: ttk.Frame,
        label: str,
        selection: PlotSelection,
        column: int,
    ) -> None:
        frame = ttk.LabelFrame(parent, text=label, padding=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0))
        parent.columnconfigure(column, weight=1)

        ttk.Label(frame, text="Dataset").grid(row=0, column=0, sticky="w")
        dataset_combo = ttk.Combobox(frame, textvariable=selection.dataset_key, state="readonly", width=55)
        dataset_combo.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        dataset_combo.bind("<<ComboboxSelected>>", lambda _evt: self.on_dataset_selected(selection))

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

        # Keep direct references for updates
        setattr(selection, "_dataset_combo", dataset_combo)
        setattr(selection, "_x_combo", x_combo)
        setattr(selection, "_y_combo", y_combo)

    def on_browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select HDF5 file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if selected:
            self.hdf_path.set(selected)

    def on_load(self) -> None:
        path = Path(self.hdf_path.get().strip())
        self.load_file(path)

    def load_file(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("File not found", f"HDF5 file does not exist:\n{path}")
            return
        try:
            with pd.HDFStore(path, mode="r") as store:
                keys = list(store.keys())
        except Exception as exc:
            messagebox.showerror("Load failed", f"Failed to open HDF5 file:\n{path}\n\n{exc}")
            return

        run_keys = [k for k in keys if k.startswith("/runs/")]
        self.dataset_keys = run_keys if run_keys else keys
        self.df_cache.clear()

        if not self.dataset_keys:
            messagebox.showwarning("No datasets", "No datasets found in the selected HDF5 file.")
            return

        self.hdf_path.set(str(path))
        self._set_dataset_options(self.plot_a, self.dataset_keys)
        self._set_dataset_options(self.plot_b, self.dataset_keys)
        self._init_plot_selection(self.plot_a)
        self._init_plot_selection(self.plot_b, prefer_second=True)
        self.redraw()

        self.status_text.set(
            f"Loaded {path.name} with {len(self.dataset_keys)} plottable datasets."
        )

    def _set_dataset_options(self, selection: PlotSelection, keys: list[str]) -> None:
        dataset_combo: ttk.Combobox = getattr(selection, "_dataset_combo")
        dataset_combo["values"] = keys

    def _set_column_options(self, selection: PlotSelection, columns: list[str]) -> None:
        x_combo: ttk.Combobox = getattr(selection, "_x_combo")
        y_combo: ttk.Combobox = getattr(selection, "_y_combo")
        x_combo["values"] = columns
        y_combo["values"] = columns

    def _init_plot_selection(self, selection: PlotSelection, prefer_second: bool = False) -> None:
        if not self.dataset_keys:
            return
        if prefer_second and len(self.dataset_keys) > 1:
            selection.dataset_key.set(self.dataset_keys[1])
        else:
            selection.dataset_key.set(self.dataset_keys[0])
        self.on_dataset_selected(selection)

    def on_dataset_selected(self, selection: PlotSelection) -> None:
        key = selection.dataset_key.get().strip()
        if not key:
            return
        try:
            df = self._load_dataframe(key)
        except Exception as exc:
            messagebox.showerror("Read failed", f"Failed to read dataset {key}:\n\n{exc}")
            return

        numeric_cols = [
            c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        if not numeric_cols:
            numeric_cols = list(df.columns)
        self._set_column_options(selection, numeric_cols)

        if "Time" in numeric_cols:
            selection.x_column.set("Time")
        else:
            selection.x_column.set(numeric_cols[0])

        y_default = "phi" if "phi" in numeric_cols else numeric_cols[0]
        if y_default == selection.x_column.get() and len(numeric_cols) > 1:
            y_default = numeric_cols[1]
        selection.y_column.set(y_default)
        self.redraw()

    def _load_dataframe(self, key: str) -> pd.DataFrame:
        if key not in self.df_cache:
            path = Path(self.hdf_path.get().strip())
            with pd.HDFStore(path, mode="r") as store:
                loaded = store[key]
            if isinstance(loaded, pd.Series):
                col_name = str(loaded.name) if loaded.name is not None else "value"
                loaded_df = loaded.to_frame(name=col_name)
            elif isinstance(loaded, pd.DataFrame):
                loaded_df = loaded
            else:
                raise TypeError(
                    f"Unsupported dataset type for key '{key}': {type(loaded).__name__}"
                )
            self.df_cache[key] = loaded_df
        return self.df_cache[key]

    def redraw(self) -> None:
        self._draw_single_plot(self.axes[0], self.plot_a, "Plot A")
        self._draw_single_plot(self.axes[1], self.plot_b, "Plot B")
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _draw_single_plot(self, ax: Axes, selection: PlotSelection, label: str) -> None:
        ax.clear()
        key = selection.dataset_key.get().strip()
        x_col = selection.x_column.get().strip()
        y_col = selection.y_column.get().strip()

        if not key or not x_col or not y_col:
            ax.set_title(f"{label}: select dataset and variables")
            ax.grid(alpha=0.2)
            return

        try:
            df = self._load_dataframe(key)
        except Exception as exc:
            ax.set_title(f"{label}: failed to load {key}\n{exc}")
            ax.grid(alpha=0.2)
            return

        if x_col not in df.columns or y_col not in df.columns:
            ax.set_title(f"{label}: invalid column selection")
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
        ax.set_title(f"{label}: {key}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an HDF5 file of processed trials and inspect two plots interactively "
            "with dataset/variable dropdown selectors."
        )
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path("outputs/preprocessed_all_trials.h5"),
        help="Path to HDF5 file (default: outputs/preprocessed_all_trials.h5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    app = HDF5InspectorApp(root, initial_hdf_path=args.hdf5)
    _ = app
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
