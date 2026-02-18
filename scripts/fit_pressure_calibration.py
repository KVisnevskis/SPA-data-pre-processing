from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.pressure_calibration import (
    affine_to_ratiometric_params,
    decode_pressure_adc_from_raw_csv,
    detect_steady_pressure_segments,
    fit_affine_pressure_calibration,
    segments_to_dataframe,
)

DEFAULT_CALIB_DIR = REPO_ROOT / "sample_data" / "processed_calibration"


def _parse_pressures_kpa(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",")]
    values = [float(p) for p in parts if p]
    if not values:
        raise ValueError("No valid values found in --pressures-kpa")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit pressure calibration from steady-state ADC plateaus. "
            "Workflow: detect candidate steady segments, label them with known pressure "
            "states (kPa), then fit an affine model P_pa = m*ADC + b."
        )
    )
    parser.add_argument(
        "--input-raw",
        type=Path,
        default=REPO_ROOT / "sample_data" / "sample_arduino_fixed_orientation.csv",
        help="Path to raw Arduino CSV (16-byte rows).",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=240.0)

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=31,
        help="Rolling median window for derivative estimation.",
    )
    parser.add_argument(
        "--std-window",
        type=int,
        default=121,
        help="Rolling standard deviation window for local noise screening.",
    )
    parser.add_argument(
        "--derivative-threshold",
        type=float,
        default=1.2,
        help="Max abs(sample-to-sample change) in smoothed ADC for steady-state.",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=5.0,
        help="Max rolling std ADC for steady-state.",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=1.0,
        help="Minimum steady segment duration.",
    )

    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=DEFAULT_CALIB_DIR / "sample_pressure_steady_segments.csv",
        help=(
            "CSV path to write detected segments (and optionally read labels from). "
            "If this file exists and has pressure_kpa labels, those labels are used for fitting."
        ),
    )
    parser.add_argument(
        "--pressures-kpa",
        type=str,
        default=None,
        help=(
            "Comma-separated known pressure values (kPa) mapped in order to detected segments. "
            "Use this only when mapping by segment order is intentional."
        ),
    )
    parser.add_argument(
        "--fit-all-segments",
        action="store_true",
        help=(
            "By default, only segments with finite pressure_kpa labels are used. "
            "Set this flag only when every detected segment has a known state."
        ),
    )
    parser.add_argument(
        "--p-max-pa",
        type=float,
        default=None,
        help=(
            "Full-scale pressure for equivalent ratiometric parameters. "
            "Default is inferred from fitted model at observed max ADC in the input run."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_CALIB_DIR / "sample_pressure_calibration_fit.json",
        help="Path for calibration artifact JSON.",
    )
    parser.add_argument(
        "--module-calibration-json",
        type=Path,
        default=REPO_ROOT / "calibration" / "arduino_pressure_calibration.json",
        help=(
            "Path for a minimal JSON containing v_min_ratio/v_max_ratio/p_max_pa "
            "for direct use by preprocessing.arduino_raw."
        ),
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_CALIB_DIR / "sample_pressure_adc_steady_segments.png",
        help="Optional diagnostic plot path. If matplotlib is unavailable, plot is skipped.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable diagnostic plot generation.",
    )
    return parser


def _upsert_segments_csv(path: Path, detected: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if "segment_id" not in existing.columns:
            raise ValueError(f"{path} is missing required column 'segment_id'")
        merged = detected.merge(
            existing[["segment_id"] + [c for c in existing.columns if c == "pressure_kpa"]],
            on="segment_id",
            how="left",
        )
        if "pressure_kpa" not in merged.columns:
            merged["pressure_kpa"] = np.nan
    else:
        merged = detected.copy()
        merged["pressure_kpa"] = np.nan
    merged.to_csv(path, index=False)
    return merged


def _build_fit_points(
    segments_df: pd.DataFrame,
    *,
    pressures_kpa_arg: str | None,
    fit_all_segments: bool,
) -> pd.DataFrame:
    out = segments_df.copy()

    if pressures_kpa_arg is not None:
        provided = _parse_pressures_kpa(pressures_kpa_arg)
        if len(provided) != len(out):
            raise ValueError(
                f"--pressures-kpa count ({len(provided)}) does not match detected "
                f"segment count ({len(out)})"
            )
        out["pressure_kpa"] = np.asarray(provided, dtype=np.float64)

    if "pressure_kpa" not in out.columns:
        out["pressure_kpa"] = np.nan

    out["pressure_kpa"] = pd.to_numeric(out["pressure_kpa"], errors="coerce")

    if fit_all_segments:
        fit_df = out.copy()
        if not np.isfinite(fit_df["pressure_kpa"].to_numpy(dtype=np.float64)).all():
            raise ValueError(
                "--fit-all-segments was set, but some rows have missing/non-finite pressure_kpa"
            )
    else:
        fit_df = out[np.isfinite(out["pressure_kpa"].to_numpy(dtype=np.float64))].copy()

    if len(fit_df) < 2:
        raise ValueError(
            "Need at least 2 labeled steady segments to fit calibration. "
            "Edit segments CSV and fill pressure_kpa for known states."
        )

    fit_df["pressure_pa"] = fit_df["pressure_kpa"] * 1000.0
    return fit_df


def _segment_weights(fit_df: pd.DataFrame) -> np.ndarray:
    n = fit_df["n_samples"].to_numpy(dtype=np.float64)
    std = fit_df["adc_std"].to_numpy(dtype=np.float64)
    var = np.maximum(std**2, 1e-6)
    return n / var


def _try_plot(
    pressure_adc: np.ndarray,
    segments_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4), dpi=140)
    x = np.arange(len(pressure_adc))
    ax.plot(x, pressure_adc, lw=0.7, color="#1f77b4", alpha=0.9, label="pressure_adc")

    fit_ids = set(fit_df["segment_id"].astype(int).tolist())
    for _, row in segments_df.iterrows():
        sid = int(row["segment_id"])
        s = int(row["start_sample"])
        e = int(row["end_sample"])
        color = "#2ca02c" if sid in fit_ids else "#7f7f7f"
        alpha = 0.12 if sid in fit_ids else 0.08
        ax.axvspan(s, e, color=color, alpha=alpha)
        if sid in fit_ids and np.isfinite(row.get("pressure_kpa", np.nan)):
            ax.text(
                (s + e) / 2.0,
                row["adc_median"] + 12.0,
                f"{int(row['pressure_kpa'])} kPa",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
            )

    ax.set_title("Pressure ADC with detected steady-state segments")
    ax.set_xlabel("sample index")
    ax.set_ylabel("pressure_adc (counts)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> int:
    args = build_parser().parse_args()

    pressure_adc = decode_pressure_adc_from_raw_csv(args.input_raw)
    segments = detect_steady_pressure_segments(
        pressure_adc,
        sample_rate_hz=args.sample_rate_hz,
        smooth_window=args.smooth_window,
        std_window=args.std_window,
        derivative_threshold=args.derivative_threshold,
        std_threshold=args.std_threshold,
        min_duration_s=args.min_duration_s,
    )
    if not segments:
        raise ValueError("No steady segments detected. Relax thresholds or inspect the input signal.")

    segments_df = segments_to_dataframe(segments)
    segments_df = _upsert_segments_csv(args.segments_csv, segments_df)

    labeled_count = int(
        np.isfinite(pd.to_numeric(segments_df.get("pressure_kpa", np.nan), errors="coerce")).sum()
    )
    if args.pressures_kpa is None and labeled_count < 2:
        print(f"[ok] detected steady segments: {len(segments_df)}")
        print(f"[ok] segments csv: {args.segments_csv}")
        print(
            "[next] Fill 'pressure_kpa' for at least two known steady states in that CSV, "
            "then rerun this script to fit calibration."
        )
        if not args.no_plot:
            plotted = _try_plot(
                pressure_adc,
                segments_df,
                segments_df.iloc[0:0].copy(),
                args.plot_path,
            )
            if plotted:
                print(f"[ok] diagnostic plot: {args.plot_path}")
            else:
                print("[warn] matplotlib unavailable; skipped diagnostic plot")
        return 0

    fit_df = _build_fit_points(
        segments_df, pressures_kpa_arg=args.pressures_kpa, fit_all_segments=args.fit_all_segments
    )

    weights = _segment_weights(fit_df)
    model, metrics = fit_affine_pressure_calibration(
        fit_df["adc_median"].to_numpy(dtype=np.float64),
        fit_df["pressure_pa"].to_numpy(dtype=np.float64),
        weights=weights,
    )

    if args.p_max_pa is not None:
        p_max_pa = float(args.p_max_pa)
        p_max_source = "user_provided"
    else:
        max_labeled_pa = float(fit_df["pressure_pa"].max())
        observed_max_adc = float(np.max(pressure_adc))
        observed_max_pa = float(model.pressure_pa(np.array([observed_max_adc], dtype=np.float64))[0])
        # Avoid clipping at the highest labeled point when unlabeled higher-pressure states exist.
        p_max_pa = max(max_labeled_pa, observed_max_pa)
        p_max_source = "fitted_at_observed_max_adc"

    ratiometric = affine_to_ratiometric_params(model, p_max_pa=p_max_pa, adc_max=4095)

    fitted = model.pressure_pa(fit_df["adc_median"].to_numpy(dtype=np.float64))
    fit_df["predicted_pa"] = fitted
    fit_df["residual_pa"] = fit_df["pressure_pa"] - fit_df["predicted_pa"]
    fit_df["weight"] = weights

    artifact = {
        "input_raw": str(args.input_raw),
        "sample_rate_hz": float(args.sample_rate_hz),
        "steady_detection": {
            "smooth_window": int(args.smooth_window),
            "std_window": int(args.std_window),
            "derivative_threshold": float(args.derivative_threshold),
            "std_threshold": float(args.std_threshold),
            "min_duration_s": float(args.min_duration_s),
            "segments_detected": int(len(segments_df)),
        },
        "affine_calibration": {
            "slope_pa_per_count": float(model.slope_pa_per_count),
            "intercept_pa": float(model.intercept_pa),
        },
        "equivalent_ratiometric": ratiometric,
        "equivalent_ratiometric_p_max_source": p_max_source,
        "fit_metrics": metrics,
        "fit_points": fit_df[
            [
                "segment_id",
                "start_sample",
                "end_sample",
                "duration_s",
                "adc_median",
                "adc_std",
                "pressure_kpa",
                "pressure_pa",
                "predicted_pa",
                "residual_pa",
                "weight",
            ]
        ]
        .to_dict(orient="records"),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    module_payload = {
        "v_min_ratio": float(ratiometric["v_min_ratio"]),
        "v_max_ratio": float(ratiometric["v_max_ratio"]),
        "p_max_pa": float(ratiometric["p_max_pa"]),
        "source_fit_json": str(args.output_json),
    }
    args.module_calibration_json.parent.mkdir(parents=True, exist_ok=True)
    with args.module_calibration_json.open("w", encoding="utf-8") as f:
        json.dump(module_payload, f, indent=2)

    plotted = False
    if not args.no_plot:
        plotted = _try_plot(pressure_adc, segments_df, fit_df, args.plot_path)

    print(f"[ok] detected steady segments: {len(segments_df)}")
    print(f"[ok] segments csv: {args.segments_csv}")
    print(
        "[ok] fitted affine calibration: "
        f"P_pa = {model.slope_pa_per_count:.6f} * ADC + {model.intercept_pa:.3f}"
    )
    print(
        "[ok] equivalent PressureCalibration: "
        f"v_min_ratio={ratiometric['v_min_ratio']:.6f}, "
        f"v_max_ratio={ratiometric['v_max_ratio']:.6f}, "
        f"p_max_pa={ratiometric['p_max_pa']:.1f}"
    )
    print(f"[ok] artifact json: {args.output_json}")
    print(f"[ok] module calibration json: {args.module_calibration_json}")
    if not args.no_plot:
        if plotted:
            print(f"[ok] diagnostic plot: {args.plot_path}")
        else:
            print("[warn] matplotlib unavailable; skipped diagnostic plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
