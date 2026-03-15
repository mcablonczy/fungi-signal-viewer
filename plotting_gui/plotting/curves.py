# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:24:58 2025

@author: markablonczy
"""

from typing import Tuple, Optional

import pyqtgraph as pg


Color = Tuple[int, int, int]  # (r, g, b)


def create_channel_curve(pw: pg.PlotWidget, color: Color) -> pg.PlotDataItem:
    """
    Create the main signal curve for a channel on the given PlotWidget.
    Applies the same downsampling / clipping settings as before.
    """
    curve = pw.plot([], [], pen=pg.mkPen(color, width=1))
    try:
        curve.setClipToView(True)
        curve.setDownsampling(auto=True, mode='peak')
        curve.setSkipFiniteCheck(True)
    except Exception:
        # Older pyqtgraph versions may not support all methods; fail quietly.
        pass
    return curve


def create_peak_overlay(
    pw: pg.PlotWidget,
    base_curve_z: Optional[float] = None,
) -> pg.PlotDataItem:
    """
    Create the vertical-lines / peak overlay item on the given PlotWidget.
    """
    peak_item = pw.plot([], [], pen=pg.mkPen(150, 150, 255, 160, width=1))
    if base_curve_z is not None:
        try:
            peak_item.setZValue(base_curve_z - 1)
        except Exception:
            pass
    return peak_item
