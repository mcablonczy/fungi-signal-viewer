# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 18:00:16 2025

@author: markablonczy
"""

import numpy as np

from plotting_gui.peaks import (
    estimate_noise_sigma_mad,
    PeakThresholdConfig,
    build_peak_find_kwargs,
    find_pos_neg_peaks,
    analyze_peaks_full,
)


def test_estimate_noise_sigma_mad_simple():
    # Simple case: noise around zero
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1.0, size=10_000)
    sigma = estimate_noise_sigma_mad(y)
    assert 0.8 < sigma < 1.2   # rough sanity band

def test_find_peaks_on_single_spike():
    fs = 1000.0
    t = np.arange(0, 1.0, 1/fs)
    y = np.zeros_like(t)
    y[500] = 10.0  # one clear spike

    sigma = estimate_noise_sigma_mad(y)  # will be small
    cfg = PeakThresholdConfig(
        use_relative=True,
        k_prom=3.0,
        k_height=3.0,
        prom_abs=0.0,
        height_abs=0.0,
        min_dist_s=0.01,
        min_width_s=0.0,
    )

    kwargs = build_peak_find_kwargs(sigma=sigma, fs=fs, cfg=cfg)
    peaks_pos, peaks_neg = find_pos_neg_peaks(y, kwargs)

    assert len(peaks_pos) == 1
    assert len(peaks_neg) == 0
    assert peaks_pos[0] == 500

def test_analyze_peaks_full_single_spike():
    fs = 1000.0
    t = np.arange(0, 1.0, 1/fs)
    y = np.zeros_like(t)
    y[500] = 10.0

    cfg = PeakThresholdConfig(
        use_relative=True,
        k_prom=3.0,
        k_height=3.0,
        prom_abs=0.0,
        height_abs=0.0,
        min_dist_s=0.01,
        min_width_s=0.0,
    )

    result = analyze_peaks_full(
        t=t,
        y=y,
        fs=fs,
        cfg=cfg,
        biphasic_window_s=0.0,
    )

    idx = result["peak_idx"]
    amp = result["peak_amp"]

    assert idx.size == 1
    assert idx[0] == 500
    assert amp[0] == 10.0
    assert np.all(result["duration_full_s"] >= 0)
    assert np.all(result["pwhm_s"] >= 0)


