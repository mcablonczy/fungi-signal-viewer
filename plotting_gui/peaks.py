
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
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

@dataclass
class APFeatures:
    depol_start_time_s: float
    peak_time_s: float
    hyperpol_start_time_s: float
    hyperpol_end_time_s: float

    duration_full_s: float
    duration_depol_to_hyperpol_s: float
    duration_hyperpol_s: float

    fwhm_s: float
    rise_time_s: float
    decay_time_s: float

    net_area: float
    depol_area: float
    hyperpol_area: float
    energy: float

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

def extract_ap_features_for_peak(
    self, y: np.ndarray, idx_peak: int, fs: float):
    """
    Extract AP-like features around a single peak in a 1D signal y.

    Handles both polarities (upward or downward depolarization) by using sign logic.

    Returns a dict with:
      - dep/hyp times (samples + seconds)
      - durations
      - amplitudes (peak + hyperpol)
      - FWHM, rise, decay
      - areas (net, peak-only, hyperpol-only)
      - spike energy

    If something can't be determined (e.g. no hyperpolarization), corresponding
    entries are set to np.nan.
    """
    n = len(y)
    if n == 0 or idx_peak < 0 or idx_peak >= n:
        # return all NaNs
        nan = float("nan")
        return {
            "i_dep_start": np.nan,
            "i_peak": np.nan,
            "i_hyp_start": np.nan,
            "i_hyp_end": np.nan,
            "t_dep_start": nan,
            "t_peak": nan,
            "t_hyp_start": nan,
            "t_hyp_end": nan,
            "dur_full_s": nan,
            "dur_dep_to_hyp_start_s": nan,
            "dur_hyp_s": nan,
            "fwhm_s": nan,
            "rise_time_s": nan,
            "decay_time_s": nan,
            "peak_amp": nan,
            "hyp_amp": nan,
            "net_area": nan,
            "peak_area": nan,
            "hyp_area": nan,
            "spike_energy": nan,
        }

    dt = 1.0 / fs

    # ---------- 1) Baseline & noise estimate ----------
    # Pre-window (10 s or until start of signal)
    pre_s = 10.0
    pre_n = int(pre_s * fs)
    start_pre = max(0, idx_peak - pre_n)

    pre_seg = y[start_pre:idx_peak]
    if pre_seg.size == 0:
        baseline = float(np.median(y))
        seg_for_noise = y
    else:
        baseline = float(np.median(pre_seg))
        seg_for_noise = pre_seg

    yb = y - baseline  # baseline-subtracted

    # Robust sigma (noise)
    sigma = estimate_noise_sigma_mad(seg_for_noise)


    # ---------- 2) Peak amplitude & sign-handling ----------
    A_peak = float(yb[idx_peak])      # can be + or -
    if A_peak == 0.0:
        # avoid sign(0) being 0
        sign_peak = 1.0
    else:
        sign_peak = float(np.sign(A_peak))

    z = sign_peak * yb  # depolarization is always "up" in z
    abs_A_peak = float(z[idx_peak])

    if abs_A_peak <= 0:
        # degenerate spike
        nan = float("nan")
        return {
            "i_dep_start": np.nan,
            "i_peak": float(idx_peak),
            "i_hyp_start": np.nan,
            "i_hyp_end": np.nan,
            "t_dep_start": nan,
            "t_peak": idx_peak * dt,
            "t_hyp_start": nan,
            "t_hyp_end": nan,
            "dur_full_s": nan,
            "dur_dep_to_hyp_start_s": nan,
            "dur_hyp_s": nan,
            "fwhm_s": nan,
            "rise_time_s": nan,
            "decay_time_s": nan,
            "peak_amp": A_peak,  # we can still report amplitude
            "hyp_amp": nan,
            "net_area": nan,
            "peak_area": nan,
            "hyp_area": nan,
            "spike_energy": nan,
        }

    # ---------- 3) Depolarization start (5% of peak) ----------
    alpha = 0.05  # 5%
    L_dep = alpha * abs_A_peak

    i_dep_start = 0
    i = idx_peak
    # walk backward while we're clearly above the threshold
    while i > 0 and z[i] > L_dep:
        i -= 1
    # i is now at or below threshold; depolarization start is the next sample
    i_dep_start = max(0, min(idx_peak, i + 1))

    # ---------- 4) Rise time (10–90%) ----------
    L10 = 0.10 * abs_A_peak
    L90 = 0.90 * abs_A_peak

    i10 = i_dep_start
    for j in range(i_dep_start, idx_peak + 1):
        if z[j] >= L10:
            i10 = j
            break

    i90 = i10
    for j in range(i10, idx_peak + 1):
        if z[j] >= L90:
            i90 = j
            break

    rise_time_s = (i90 - i10) * dt

    # ---------- 5) FWHM (same as PWHM of depolarization peak) ----------
    Lhalf = 0.5 * abs_A_peak

    # left half-max
    i_left = idx_peak
    while i_left > i_dep_start and z[i_left] >= Lhalf:
        i_left -= 1

    # right half-max
    i_right = idx_peak
    while i_right < n - 1 and z[i_right] >= Lhalf:
        i_right += 1

    fwhm_s = (i_right - i_left) * dt

    # ---------- 6) Decay time (90–10% AFTER peak) ----------
    j90 = idx_peak
    for j in range(idx_peak, n):
        if z[j] <= L90:
            j90 = j
            break

    j10 = j90
    for j in range(j90, n):
        if z[j] <= L10:
            j10 = j
            break

    decay_time_s = (j10 - j90) * dt

    # ---------- 7) Hyperpolarization detection (opposite lobe) ----------
    # Opposite sign view: hyperpolarization (opposite to peak) becomes "positive" in z_opp.
    z_opp = -sign_peak * yb

    post_s = 10.0
    post_n = int(post_s * fs)
    end_post = min(n, idx_peak + post_n)

    if end_post <= idx_peak + 1:
        # no room to search
        hyp_amp = float("nan")
        i_hyp_start = np.nan
        i_hyp_end = np.nan
    else:
        z_opp_seg = z_opp[idx_peak:end_post]
        local_idx_trough = int(np.argmax(z_opp_seg))
        i_trough = idx_peak + local_idx_trough
        A_hyp = float(z_opp[i_trough])  # magnitude in opposite direction

        # noise-based criteria: A_hyp >= 1σ and prominence >= 4σ
        try:
            prominences, _, _ = peak_prominences(z_opp_seg, [local_idx_trough])
            prom = float(prominences[0]) if len(prominences) > 0 else 0.0
        except Exception:
            prom = 0.0

        if A_hyp < 1.0 * sigma or prom < 4.0 * sigma:
            # No valid hyperpolarization
            hyp_amp = float("nan")
            i_hyp_start = np.nan
            i_hyp_end = np.nan
        else:
            hyp_amp = A_hyp
            beta = 0.05  # 5% of hyperpol amplitude
            L_hyp = beta * A_hyp

            # hyperpol start: where z_opp rises above L_hyp on way to trough
            i_hyp_start = idx_peak
            for j in range(i_trough, idx_peak, -1):
                if z_opp[j] >= L_hyp:
                    i_hyp_start = j
                else:
                    break

            # hyperpol end: where z_opp falls below L_hyp after trough
            i_hyp_end = i_trough
            for j in range(i_trough, end_post):
                if z_opp[j] >= L_hyp:
                    i_hyp_end = j
                else:
                    break

    # convert possibly-NaN indices to integers carefully for durations
    def safe_idx(x):
        return int(x) if np.isfinite(x) else None

    i_dep = safe_idx(i_dep_start)
    i_hs = safe_idx(i_hyp_start)
    i_he = safe_idx(i_hyp_end)

    # ---------- 8) Durations ----------
    t_peak = idx_peak * dt
    t_dep_start = i_dep_start * dt
    t_hyp_start = i_hyp_start * dt if i_hs is not None else float("nan")
    t_hyp_end = i_hyp_end * dt if i_he is not None else float("nan")

    if i_dep is not None and i_he is not None and i_he >= i_dep:
        dur_full_s = (i_he - i_dep) * dt
    else:
        dur_full_s = float("nan")

    if i_dep is not None and i_hs is not None and i_hs >= i_dep:
        dur_dep_to_hyp_start_s = (i_hs - i_dep) * dt
    else:
        dur_dep_to_hyp_start_s = float("nan")

    if i_hs is not None and i_he is not None and i_he >= i_hs:
        dur_hyp_s = (i_he - i_hs) * dt
    else:
        dur_hyp_s = float("nan")

    # ---------- 9) Areas: net, peak-only, hyp-only ----------
    net_area = float("nan")
    peak_area = float("nan")
    hyp_area = float("nan")
    spike_energy = float("nan")

    if i_dep is not None and i_he is not None and i_he > i_dep:
        seg = yb[i_dep:i_he + 1]  # inclusive
        # net area (signed)
        net_area = float(np.sum(seg) * dt)

        # peak-only area (same direction as main peak)
        mask_peak = (sign_peak * seg) > 0
        if np.any(mask_peak):
            peak_area = float(np.sum(seg[mask_peak]) * dt)
        else:
            peak_area = 0.0

        # hyperpol-only area (opposite direction)
        mask_hyp = (sign_peak * seg) < 0
        if np.any(mask_hyp):
            hyp_area = float(np.sum(seg[mask_hyp]) * dt)
        else:
            hyp_area = 0.0

        # spike energy over full event
        spike_energy = float(np.sum(seg ** 2) * dt)

    # ---------- 10) Package up ----------
    return {
        "i_dep_start": float(i_dep_start),
        "i_peak": float(idx_peak),
        "i_hyp_start": float(i_hyp_start) if np.isfinite(i_hyp_start) else np.nan,
        "i_hyp_end": float(i_hyp_end) if np.isfinite(i_hyp_end) else np.nan,
        "t_dep_start": t_dep_start,
        "t_peak": t_peak,
        "t_hyp_start": t_hyp_start,
        "t_hyp_end": t_hyp_end,
        "dur_full_s": dur_full_s,
        "dur_dep_to_hyp_start_s": dur_dep_to_hyp_start_s,
        "dur_hyp_s": dur_hyp_s,
        "fwhm_s": fwhm_s,
        "rise_time_s": rise_time_s,
        "decay_time_s": decay_time_s,
        "peak_amp": A_peak,
        "hyp_amp": hyp_amp,
        "net_area": net_area,
        "peak_area": peak_area,
        "hyp_area": hyp_area,
        "spike_energy": spike_energy,
    }

def estimate_noise_sigma_mad(
    y: np.ndarray, min_sigma: float = 1e-12) -> float:
    """
    Robust noise estimate using Median Absolute Deviation (MAD).
    """
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        return min_sigma

    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))

    if mad > 0:
        sigma = 1.4826 * mad
    else:
        sigma = float(np.std(y))

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(min_sigma)

    return sigma

def analyze_peaks_full(
    t: np.ndarray,
    y: np.ndarray,
    fs: float,
    cfg: PeakThresholdConfig,
    biphasic_window_s: float,
) -> dict[str, np.ndarray]:
    """
    Full peak analysis pipeline for a single 1D signal y over time t.

    Returns a dict with numpy arrays (possibly empty):
        peak_idx         : indices into y
        peak_amp         : amplitude at peak (V, signed)
        start_idx        : left index where |y - median| >= 10% of peak amp
        end_idx          : right index same criterion
        duration_full_s  : (end_idx - start_idx) / fs
        pwhm_s           : width at half maximum (|y - median| >= 50% of peak amp)
    """

    empty = {
        "peak_idx": np.array([], dtype=int),
        "peak_amp": np.array([], dtype=float),
        "start_idx": np.array([], dtype=int),
        "end_idx": np.array([], dtype=int),
        "duration_full_s": np.array([], dtype=float),
        "pwhm_s": np.array([], dtype=float),
    }

    if t.size == 0 or y.size == 0:
        return empty

    y_float = np.asarray(y, dtype=float)

    # Baseline median used for duration/FWHM thresholds
    med = np.median(y_float)

    # ---------- 1) Estimate noise σ on this window ----------
    sigma = estimate_noise_sigma_mad(y_float)

    # ---------- Build thresholds ----------
    kwargs = build_peak_find_kwargs(
        sigma=sigma,
        fs=fs,
        cfg=cfg,
    )

    # ---------- Positive & negative peaks ----------
    try:
        peaks_pos, peaks_neg = find_pos_neg_peaks(y_float, kwargs)
    except Exception as e:
        print("[warn] find_peaks failed:", e)
        peaks_pos = np.array([], dtype=int)
        peaks_neg = np.array([], dtype=int)

    indices = np.concatenate([peaks_pos, peaks_neg])
    signs = np.concatenate([
        np.ones_like(peaks_pos, dtype=int),
        -np.ones_like(peaks_neg, dtype=int),
    ])

    if indices.size == 0:
        return empty

    order = np.argsort(indices)
    indices = indices[order]
    signs = signs[order]

    # ---------- Biphasic merge ----------
    keep = np.ones(len(indices), dtype=bool)
    window_s = biphasic_window_s
    if window_s > 0.0 and len(indices) > 1:
        max_dt_samples = int(round(window_s * fs))
        if max_dt_samples > 0:
            for i in range(len(indices) - 1):
                if not keep[i]:
                    continue
                j = i + 1
                if not keep[j]:
                    continue
                if signs[i] != signs[j] and (indices[j] - indices[i]) <= max_dt_samples:
                    # Drop the second lobe; keep the first
                    keep[j] = False

    final_idx = indices[keep]
    if final_idx.size == 0:
        return empty

    # ---------- Per-peak amplitude + durations ----------
    peak_amp = y_float[final_idx]  # signed amplitude relative to 0
    abs_amp = np.abs(peak_amp)

    n = y_float.size
    start_idx = np.zeros_like(final_idx)
    end_idx = np.zeros_like(final_idx)
    duration_full_s = np.zeros_like(final_idx, dtype=float)
    pwhm_s = np.zeros_like(final_idx, dtype=float)

    for k, idx in enumerate(final_idx):
        amp_k = abs_amp[k]
        if amp_k <= 0:
            # Degenerate, treat as 0-duration
            start_idx[k] = idx
            end_idx[k] = idx
            duration_full_s[k] = 0.0
            pwhm_s[k] = 0.0
            continue

        # Baseline-relative magnitude (use |y - med| for thresholds)
        thresh_full = 0.10 * amp_k
        thresh_half = 0.50 * amp_k

        # --- Full duration ---
        left = idx
        while left > 0 and np.abs(y_float[left] - med) >= thresh_full:
            left -= 1

        right = idx
        while right < n - 1 and np.abs(y_float[right] - med) >= thresh_full:
            right += 1

        start_idx[k] = left
        end_idx[k] = right
        duration_full_s[k] = max(0, right - left) / fs

        # --- Width at half maximum ---
        left_h = idx
        while left_h > 0 and np.abs(y_float[left_h] - med) >= thresh_half:
            left_h -= 1

        right_h = idx
        while right_h < n - 1 and np.abs(y_float[right_h] - med) >= thresh_half:
            right_h += 1

        pwhm_s[k] = max(0, right_h - left_h) / fs

    return {
        "peak_idx": final_idx,
        "peak_amp": peak_amp,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "duration_full_s": duration_full_s,
        "pwhm_s": pwhm_s,
    }













