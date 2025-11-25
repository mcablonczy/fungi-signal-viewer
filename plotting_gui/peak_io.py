# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:19:11 2025

@author: markablonczy
"""

import os
import csv
from typing import Iterable, Dict, Any, Sequence

import numpy as np

PEAK_STATS_HEADER = [
    "channel",
    "stat_name",
    "stat_value",
    "window_start_s",
    "window_end_s",
    "window_start_time",
    "window_end_time",
]

PEAKS_HEADER = [
    "channel",
    "peak_amplitude",
    "start_time_str",
    "peak_time_str",
    "end_time_str",
    "duration_full_s",
    "pwhm_s",
    "pwhm_freq_hz",
    "dep_to_hyp_start_s",
    "hyp_duration_s",
    "rise_time_s",
    "decay_time_s",
    "hyp_amplitude",
    "net_area",
    "peak_area",
    "hyp_area",
    "spike_energy",
]


def save_peak_windows_csv(file_path: str, df) -> None:
    """
    Save the long-format peak windows DataFrame to CSV.

    `df` is expected to be a pandas.DataFrame with columns like:
        channel, peak_id, sample_index, time_s, rel_time_s, value, peak_time_s
    """
    df.to_csv(file_path, index=False)


def save_peaks_csv_with_metadata(
    file_path: str,
    rows: Iterable[Dict[str, Any]],
    settings: Dict[str, Any],
) -> None:
    """
    Save:
      1) a 'peaks' CSV with one row per peak
      2) a 'metadata' CSV with key/value pairs of the current settings.

    `rows` is a list of dicts produced by _build_peaks_rows_for_current_window().
    `settings` is a dict produced by _current_peak_settings_snapshot().
    """
    # Normalize extension and derive metadata path
    root, ext = os.path.splitext(file_path)
    if not ext:
        ext = ".csv"
        file_path = root + ext
    meta_path = root + "_metadata" + ext

    # --------- 1) PEAKS CSV (one row per peak) ----------
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PEAKS_HEADER)

        for row in rows:
            # These keys are expected in each row dict
            amp = float(row["peak_amplitude"])
            dur = float(row["duration_full_s"])
            pwhm = float(row["pwhm_s"])
            freq = float(row["pwhm_freq_hz"])

            def _fmt_if_finite(key: str, fmt: str):
                val = row.get(key, np.nan)
                return fmt.format(float(val)) if np.isfinite(val) else ""

            writer.writerow([
                row["channel"],
                f"{amp:.1f}",
                row["start_time_str"],
                row["peak_time_str"],
                row["end_time_str"],
                f"{dur:.1f}",
                f"{pwhm:.1f}",
                f"{freq:.2f}",
                _fmt_if_finite("dep_to_hyp_start_s", "{:.1f}"),
                _fmt_if_finite("hyp_duration_s", "{:.1f}"),
                _fmt_if_finite("rise_time_s", "{:.1f}"),
                _fmt_if_finite("decay_time_s", "{:.1f}"),
                _fmt_if_finite("hyp_amplitude", "{:.1f}"),
                _fmt_if_finite("net_area", "{:.3e}"),
                _fmt_if_finite("peak_area", "{:.3e}"),
                _fmt_if_finite("hyp_area", "{:.3e}"),
                _fmt_if_finite("spike_energy", "{:.3e}"),
            ])

    # --------- 2) METADATA CSV (key/value pairs) ----------
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key in sorted(settings.keys()):
            writer.writerow([key, settings[key]])

def save_peak_stats_long_csv(file_path: str, rows: Iterable[Sequence]) -> None:
    """
    Write a long-format peak stats CSV.

    Parameters
    ----------
    file_path : str
        Output CSV path.
    rows : iterable of sequences
        Each row should already be a flat sequence matching PEAK_STATS_HEADER.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PEAK_STATS_HEADER)
        for r in rows:
            writer.writerow(r)