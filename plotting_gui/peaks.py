
import numpy as np
from dataclasses import dataclass
from typing import Dict
from scipy.signal import find_peaks

@dataclass
class PeakThresholdConfig:
    use_relative: bool
    k_prom: float
    k_height: float
    prom_abs: float
    height_abs: float
    min_dist_s: float
    min_width_s: float

def estimate_noise_sigma_mad(
    y: np.ndarray, min_sigma: float = 1e-12) -> float:
    """
    Robust noise estimate using Median Absolute Deviation (MAD).

    Parameters
    ----------
    y : np.ndarray
        1D array with signal samples (any baseline is fine).
    min_sigma : float
        Minimum sigma to return if the estimate degenerates.

    Returns
    -------
    float
        Estimated noise sigma.
    """
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        return min_sigma

    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))

    if mad > 0:
        sigma = 1.4826 * mad  # robust Gaussian-equivalent "std"
    else:
        sigma = float(np.std(y))

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(min_sigma)

    return sigma

def build_peak_find_kwargs(
    sigma: float,
    fs: float,
    cfg: PeakThresholdConfig,
) -> Dict:
    """
    Build the kwargs dict for scipy.signal.find_peaks based on the
    current settings and sigma.

    This is a direct extraction of the existing logic from viewer.py:
    - relative vs absolute thresholds
    - min distance, min width
    """
    prom_val = None
    height_val = None

    if cfg.use_relative:
        if cfg.k_prom > 0.0:
            prom_val = cfg.k_prom * sigma
        if cfg.k_height > 0.0:
            height_val = cfg.k_height * sigma
    else:
        if cfg.prom_abs > 0.0:
            prom_val = cfg.prom_abs
        if cfg.height_abs > 0.0:
            height_val = cfg.height_abs

    kwargs: Dict = {}
    if prom_val is not None and prom_val > 0.0:
        kwargs["prominence"] = prom_val
    if height_val is not None and height_val > 0.0:
        kwargs["height"] = height_val

    if cfg.min_dist_s > 0.0:
        dist_samples = int(round(cfg.min_dist_s * fs))
        if dist_samples > 0:
            kwargs["distance"] = dist_samples

    if cfg.min_width_s > 0.0:
        width_samples = int(round(cfg.min_width_s * fs))
        if width_samples > 0:
            kwargs["width"] = width_samples

    return kwargs

def find_pos_neg_peaks(
    y: np.ndarray,
    kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run scipy.signal.find_peaks on both y and -y with the same kwargs.

    Parameters
    ----------
    y : np.ndarray
        1D signal.
    kwargs : dict
        Keyword arguments for scipy.signal.find_peaks, typically built
        via build_peak_find_kwargs.

    Returns
    -------
    peaks_pos : np.ndarray
        Indices of positive-going peaks.
    peaks_neg : np.ndarray
        Indices of negative-going peaks.
    """
    y = np.asarray(y, dtype=float)

    # These two lines are exactly what you already do in viewer.py
    peaks_pos, _ = find_peaks(y, **kwargs)
    peaks_neg, _ = find_peaks(-y, **kwargs)

    return peaks_pos, peaks_neg













