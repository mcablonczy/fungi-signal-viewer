import numpy as np

from plotting_gui.filters import (
    apply_filter_pipeline,
    apply_median_filter_3,
    compute_robust_car_median_reference,
)


def test_compute_robust_car_median_reference_uses_samplewise_median():
    segments = np.array([
        [1.0, 10.0, 3.0, 1000.0],
        [2.0, 20.0, 4.0, 5.0],
        [100.0, 30.0, 5.0, 6.0],
    ])

    ref = compute_robust_car_median_reference(segments)

    np.testing.assert_allclose(ref, np.array([2.0, 20.0, 4.0, 6.0]))


def test_apply_median_filter_3_removes_single_sample_spike():
    y = np.array([0.0, 100.0, 0.0, 1.0, 1.0])

    filtered = apply_median_filter_3(y)

    np.testing.assert_allclose(filtered, np.array([0.0, 0.0, 1.0, 1.0, 1.0]))


def test_apply_filter_pipeline_applies_median_stage_when_enabled():
    raw = np.array([2.0, 50.0, 3.0, 4.0, 100.0, 5.0])

    filtered = apply_filter_pipeline(
        raw=raw,
        fs=100.0,
        cm_enabled=False,
        cm_ref=None,
        butter_enabled=False,
        filter_type="lowpass",
        f1=0.1,
        f2=10.0,
        order=4,
        median_enabled=True,
        ma_enabled=False,
        ma_points=1,
        ma_passes=1,
    )

    np.testing.assert_allclose(filtered, np.array([2.0, 3.0, 4.0, 4.0, 5.0, 5.0]))
