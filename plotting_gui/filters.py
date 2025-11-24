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
