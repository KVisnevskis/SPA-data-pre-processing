import numpy as np

from preprocessing.pressure_calibration import (
    AffinePressureCalibration,
    affine_to_ratiometric_params,
    detect_steady_pressure_segments,
    fit_affine_pressure_calibration,
)


def test_detect_steady_pressure_segments_finds_plateaus():
    rng = np.random.default_rng(7)

    a = 500.0 + rng.normal(0.0, 1.0, size=300)
    ramp = np.linspace(500.0, 1400.0, num=120)
    b = 1400.0 + rng.normal(0.0, 1.2, size=320)
    signal = np.concatenate([a, ramp, b])

    segments = detect_steady_pressure_segments(
        signal,
        sample_rate_hz=240.0,
        smooth_window=21,
        std_window=81,
        derivative_threshold=1.5,
        std_threshold=4.0,
        min_duration_s=1.0,
    )

    assert len(segments) >= 2
    medians = np.array([s.adc_median for s in segments], dtype=np.float64)
    assert np.any(np.isclose(medians, 500.0, atol=15.0))
    assert np.any(np.isclose(medians, 1400.0, atol=20.0))


def test_fit_affine_pressure_calibration_recovers_linear_mapping():
    adc = np.array([500.0, 700.0, 950.0, 1200.0, 1400.0], dtype=np.float64)
    slope_true = 61.1111111111
    intercept_true = -30_555.55555555
    pressure_pa = slope_true * adc + intercept_true
    weights = np.array([1.0, 3.0, 2.0, 2.0, 1.0], dtype=np.float64)

    model, metrics = fit_affine_pressure_calibration(adc, pressure_pa, weights=weights)

    assert np.isclose(model.slope_pa_per_count, slope_true, atol=1e-8)
    assert np.isclose(model.intercept_pa, intercept_true, atol=1e-6)
    assert np.isclose(metrics["weighted_rmse_pa"], 0.0, atol=1e-9)
    assert np.isclose(metrics["weighted_mae_pa"], 0.0, atol=1e-9)
    assert np.isclose(metrics["r2_weighted"], 1.0, atol=1e-12)


def test_affine_to_ratiometric_params_for_two_point_example():
    model = AffinePressureCalibration(
        slope_pa_per_count=(55_000.0 / (1400.0 - 500.0)),
        intercept_pa=-(55_000.0 / (1400.0 - 500.0)) * 500.0,
    )
    params = affine_to_ratiometric_params(model, p_max_pa=55_000.0, adc_max=4095)

    assert np.isclose(params["v_min_ratio"], 500.0 / 4095.0, atol=1e-12)
    assert np.isclose(params["v_max_ratio"], 1400.0 / 4095.0, atol=1e-12)
    assert np.isclose(params["p_max_pa"], 55_000.0, atol=1e-12)
