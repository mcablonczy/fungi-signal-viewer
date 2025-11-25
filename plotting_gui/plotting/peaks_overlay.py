# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:32:00 2025

@author: markablonczy
"""

from typing import Optional

import numpy as np
import pyqtgraph as pg


def set_peak_vertical_lines(
    peak_item: Optional[pg.PlotDataItem],
    t: np.ndarray,
    peak_indices: np.ndarray,
    y_min: float,
    y_max: float,
) -> None:
    """
    Draw vertical lines at peak times on the given PlotDataItem.

    Mirrors the existing behavior in viewer._update_peaks_for_channel:
    - For each peak time tp, draw a line from (tp, y_min) to (tp, y_max),
      separated by np.nan so pyqtgraph renders separate segments.
    - If no peaks or peak_item is None, clear the item.
    """
    if peak_item is None:
        return

    if peak_indices.size == 0 or t.size == 0:
        peak_item.setData([], [], _callSync="off")
        return

    # Guard against out-of-range peak indices
    valid_mask = (peak_indices >= 0) & (peak_indices < t.size)
    if not np.any(valid_mask):
        peak_item.setData([], [], _callSync="off")
        return

    peak_indices = peak_indices[valid_mask]
    t_peaks = t[peak_indices]

    x_vals = []
    y_vals = []
    for tp in t_peaks:
        x_vals.extend([tp, tp, np.nan])
        y_vals.extend([y_min, y_max, np.nan])

    peak_item.setData(x_vals, y_vals, _callSync="off")
