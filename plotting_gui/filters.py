# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 18:15:09 2025

@author: markablonczy
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def apply_butterworth_filter(
    y: np.ndarray,
    fs: float,
    filter_type: str,
    f1: float,
    f2: float,
    order: int,
) -> np.ndarray:
    """
    Apply a Butterworth filter to a 1D signal y, mirroring the behavior used
    in the viewer:

    - Uses sosfiltfilt for zero-phase filtering.
    - Respects lowpass / highpass / bandpass modes.
    - Returns the input unchanged if:
        * the segment is too short for filtfilt,
        * cutoffs are invalid,
        * or an exception occurs.

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D).
    fs : float
        Sampling rate in Hz.
    filter_type : {"lowpass", "highpass", "bandpass"}
        Filter mode.
    f1 : float
        Lower cutoff (used for highpass or bandpass).
    f2 : float
        Upper cutoff (used for lowpass or bandpass).
    order : int
        Filter order (>= 1).

    Returns
    -------
    np.ndarray
        Filtered signal, same shape and dtype as input.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("apply_butterworth_filter expects a 1D array")

    # Keep original dtype for the output
    orig_dtype = y.dtype
    y_float = y.astype(float, copy=False)

    fs = float(fs) if fs > 0 else 1.0
    nyq = 0.5 * fs

    # Match your viewer's sanitization
    f1 = max(0.0, float(f1))
    f2 = max(0.0, float(f2))
    order = max(1, int(order))

    # filtfilt length sanity check (same as viewer)
    if y_float.size < (order + 1) * 3:
        # Too short → skip Butterworth, return input unchanged
        return y.astype(orig_dtype, copy=False)

    try:
        if filter_type == "lowpass":
            if f2 <= 0 or f2 >= nyq:
                # invalid cutoff → skip
                return y.astype(orig_dtype, copy=False)
            Wn = f2 / nyq
            sos = butter(order, Wn, btype="low", output="sos")
            out = sosfiltfilt(sos, y_float)

        elif filter_type == "highpass":
            if f1 <= 0 or f1 >= nyq:
                return y.astype(orig_dtype, copy=False)
            Wn = f1 / nyq
            sos = butter(order, Wn, btype="high", output="sos")
            out = sosfiltfilt(sos, y_float)

        else:  # "bandpass" or anything else
            lo = min(f1, f2)
            hi = max(f1, f2)
            if lo <= 0 or hi >= nyq or hi <= lo:
                return y.astype(orig_dtype, copy=False)
            Wn = (lo / nyq, hi / nyq)
            sos = butter(order, Wn, btype="band", output="sos")
            out = sosfiltfilt(sos, y_float)

    except Exception:
        # On any failure, fall back to the original signal
        return y.astype(orig_dtype, copy=False)

    return np.asarray(out, dtype=orig_dtype)

def apply_moving_average(
    y: np.ndarray,
    n_points: int,
    n_passes: int = 1,
) -> np.ndarray:
    """
    Apply a simple centered moving-average filter to a 1D array.

    - Uses 'same' convolution so the length is preserved.
    - Repeats the filter `n_passes` times.
    - Mirrors the behavior of the previous _apply_moving_average method
      (returns a float array).
    """
    y = np.asarray(y)

    if y.size == 0:
        return y

    n = int(n_points)
    if n <= 1:
        return y

    kernel = np.ones(n, dtype=float) / float(n)
    out = np.asarray(y, dtype=float)

    passes = max(1, int(n_passes))
    for _ in range(passes):
        out = np.convolve(out, kernel, mode="same")

    return out

def apply_common_mode(raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Subtract a reference (common-mode) signal from a raw signal.

    Mirrors the behavior of _get_common_mode_segment's subtraction:

    - raw and ref are 1D arrays of the same length.
    - Returns raw - ref when possible.
    - If dtype issues occur, falls back to float subtraction.
    """
    raw = np.asarray(raw)
    ref = np.asarray(ref)

    try:
        return raw - ref
    except Exception:
        # Fallback: do subtraction in float space
        return raw.astype(float) - ref.astype(float)

def apply_filter_pipeline(
    raw: np.ndarray,
    fs: float,
    cm_enabled: bool,
    cm_ref: np.ndarray | None,
    butter_enabled: bool,
    filter_type: str,
    f1: float,
    f2: float,
    order: int,
    ma_enabled: bool,
    ma_points: int,
    ma_passes: int,
) -> np.ndarray:
    """
    Apply the full filter chain to a 1D signal:

    1) Optional common-mode subtraction (raw - cm_ref).
    2) Optional Butterworth filter.
    3) Optional moving-average smoothing.

    This mirrors the behavior of the viewer's _get_filtered_segment logic,
    but without any GUI state or caching.
    """
    y = np.asarray(raw)

    # --- Common-mode ---
    if cm_enabled and cm_ref is not None:
        if cm_ref.shape == y.shape:
            y = apply_common_mode(y, cm_ref)
        else:
            # Shape mismatch → skip CM (same as implicitly doing nothing)
            pass

    # --- Butterworth ---
    if butter_enabled:
        y = apply_butterworth_filter(
            y=y,
            fs=fs,
            filter_type=filter_type,
            f1=f1,
            f2=f2,
            order=order,
        )

    # --- Moving average ---
    if ma_enabled:
        y = apply_moving_average(
            y=y,
            n_points=ma_points,
            n_passes=ma_passes,
        )

    return y


