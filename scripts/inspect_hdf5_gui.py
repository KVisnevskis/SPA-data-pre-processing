from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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
MAX_META_ROWS = 500


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
        self.meta_keys: list[str] = []
        self.df_cache: dict[str, pd.DataFrame] = {}
        self.meta_cache: dict[str, pd.DataFrame] = {}

        self.plot_a = PlotSelection(tk.StringVar(), tk.StringVar(), tk.StringVar())
        self.plot_b = PlotSelection(tk.StringVar(), tk.StringVar(), tk.StringVar())
        self.meta_key_var = tk.StringVar()
        self.meta_filter_run_id = tk.StringVar()
        self.meta_status = tk.StringVar(value="No metadata loaded.")
        self.meta_row_json = tk.StringVar(value="")

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

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=4)

        plot_tab = ttk.Frame(notebook)
        meta_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="Plots")
        notebook.add(meta_tab, text="Metadata")

        controls = ttk.Frame(plot_tab, padding=(0, 0, 0, 4))
        controls.pack(fill="x")
        self._build_plot_controls(controls, "Plot A", self.plot_a, 0)
        self._build_plot_controls(controls, "Plot B", self.plot_b, 1)

        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        self._build_metadata_tab(meta_tab)

        status = ttk.Label(self.root, textvariable=self.status_text, relief="sunken", anchor="w")
        status.pack(fill="x", padx=8, pady=(0, 8))

    def _build_metadata_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, padding=(0, 0, 0, 4))
        top.pack(fill="x")

        ttk.Label(top, text="Meta table").pack(side="left")
        self.meta_combo = ttk.Combobox(top, textvariable=self.meta_key_var, state="readonly", width=45)
        self.meta_combo.pack(side="left", padx=(6, 10))
        self.meta_combo.bind("<<ComboboxSelected>>", lambda _evt: self.refresh_metadata_view())

        ttk.Label(top, text="run_id filter").pack(side="left")
        self.meta_filter_entry = ttk.Entry(top, textvariable=self.meta_filter_run_id, width=28)
        self.meta_filter_entry.pack(side="left", padx=(6, 6))
        self.meta_filter_entry.bind("<Return>", lambda _evt: self.refresh_metadata_view())
        ttk.Button(top, text="Apply", command=self.refresh_metadata_view).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Clear", command=self.clear_meta_filter).pack(side="left")

        self.meta_tree = ttk.Treeview(parent, show="headings")
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=self.meta_tree.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=self.meta_tree.xview)
        self.meta_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.meta_tree.pack(fill="both", expand=True, side="top")
        yscroll.pack(fill="y", side="right")
        xscroll.pack(fill="x", side="bottom")
        self.meta_tree.bind("<<TreeviewSelect>>", lambda _evt: self.on_meta_row_selected())

        meta_status_lbl = ttk.Label(parent, textvariable=self.meta_status, anchor="w")
        meta_status_lbl.pack(fill="x", pady=(4, 2))

        ttk.Label(parent, text="Selected row (JSON)").pack(anchor="w")
        self.meta_json_text = tk.Text(parent, height=8, wrap="none")
        self.meta_json_text.pack(fill="x", pady=(2, 0))

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
        meta_keys = [k for k in keys if k.startswith("/meta/")]
        self.dataset_keys = run_keys if run_keys else keys
        self.meta_keys = meta_keys
        self.df_cache.clear()
        self.meta_cache.clear()

        if not self.dataset_keys:
            messagebox.showwarning("No datasets", "No datasets found in the selected HDF5 file.")
            return

        self.hdf_path.set(str(path))
        self._set_dataset_options(self.plot_a, self.dataset_keys)
        self._set_dataset_options(self.plot_b, self.dataset_keys)
        self._init_plot_selection(self.plot_a)
        self._init_plot_selection(self.plot_b, prefer_second=True)
        self._init_metadata_selection()
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

    def _init_metadata_selection(self) -> None:
        self.meta_combo["values"] = self.meta_keys
        if self.meta_keys:
            self.meta_key_var.set(self.meta_keys[0])
            self.refresh_metadata_view()
        else:
            self.meta_key_var.set("")
            self._clear_tree()
            self.meta_status.set("No /meta/* tables in this HDF5.")

    def clear_meta_filter(self) -> None:
        self.meta_filter_run_id.set("")
        self.refresh_metadata_view()

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

    def _load_meta_dataframe(self, key: str) -> pd.DataFrame:
        if key not in self.meta_cache:
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
                    f"Unsupported metadata dataset type for key '{key}': {type(loaded).__name__}"
                )
            self.meta_cache[key] = loaded_df
        return self.meta_cache[key]

    def refresh_metadata_view(self) -> None:
        key = self.meta_key_var.get().strip()
        if not key:
            self._clear_tree()
            return
        try:
            df = self._load_meta_dataframe(key).copy()
        except Exception as exc:
            messagebox.showerror("Metadata load failed", f"Failed to load {key}:\n\n{exc}")
            return

        run_filter = self.meta_filter_run_id.get().strip()
        if run_filter and "run_id" in df.columns:
            df = df[df["run_id"].astype(str) == run_filter]

        shown = df.head(MAX_META_ROWS).copy()
        shown = shown.reset_index(drop=True)
        self._render_tree(shown)

        self.meta_status.set(
            f"{key}: showing {len(shown)} rows"
            f"{' (filtered)' if run_filter else ''}"
            + (f" of {len(df)}" if len(df) > MAX_META_ROWS else "")
        )
        self.meta_json_text.delete("1.0", tk.END)
        self.meta_json_text.insert(tk.END, "")

    def _clear_tree(self) -> None:
        self.meta_tree.delete(*self.meta_tree.get_children())
        self.meta_tree["columns"] = ()

    def _render_tree(self, df: pd.DataFrame) -> None:
        self._clear_tree()
        columns = list(df.columns)
        self.meta_tree["columns"] = columns
        for col in columns:
            self.meta_tree.heading(col, text=col)
            self.meta_tree.column(col, width=160, stretch=True, anchor="w")

        for i, row in df.iterrows():
            values = [self._stringify_cell(row[c]) for c in columns]
            self.meta_tree.insert("", "end", iid=str(i), values=values)

    def _stringify_cell(self, v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        s = str(v)
        return s if len(s) <= 140 else s[:137] + "..."

    def on_meta_row_selected(self) -> None:
        selected = self.meta_tree.selection()
        if not selected:
            return
        item_id = selected[0]
        values = self.meta_tree.item(item_id, "values")
        columns = self.meta_tree["columns"]
        row = {str(col): values[i] for i, col in enumerate(columns)}

        pretty = json.dumps(row, indent=2)
        self.meta_json_text.delete("1.0", tk.END)
        self.meta_json_text.insert(tk.END, pretty)

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
