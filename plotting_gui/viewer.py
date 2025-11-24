# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:32:07 2025

@author: markablonczy
"""

import sys 
import os
import math
import h5py
import numpy as np
import pyqtgraph as pg
import math
import csv
import pandas as pd




from collections import OrderedDict

from scipy.signal import butter, sosfiltfilt, find_peaks, peak_prominences
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime, timedelta



from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QCheckBox, QLabel, QPushButton, QFileDialog, QScrollArea,
    QDoubleSpinBox, QFrame, QSizePolicy, QWidget as QtWidget,
    QTabWidget, QTableWidget, QTableWidgetItem,
    QDateTimeEdit, QComboBox, QSpinBox
)

from PyQt5.QtCore import Qt, QTimer, QPointF, QEvent, QDateTime
from PyQt5.QtGui import QFont

from plotting_gui.peaks import (
    estimate_noise_sigma_mad, PeakThresholdConfig, build_peak_find_kwargs, find_pos_neg_peaks,
    APFeatures, extract_ap_features_for_peak, analyze_peaks_full
)

from plotting_gui.filters import (
    apply_butterworth_filter, apply_moving_average, apply_common_mode,
)


# Your requested display order (trimmed, exact matches)
PREFERRED_ORDER = [
    "A-024","B-024","A-023","B-023","A-025","B-025","A-022","B-022",
    "B-026","A-026","B-021","A-021","B-027","A-027","B-020","A-020",
    "B-028","A-028","B-019","A-019","B-029","A-029","B-018","B-030",
    "A-018","B-017","A-030","B-031","A-017","B-016","A-031","B-000",
    "A-016","B-015","A-000","B-001","A-015","B-014","B-002","A-001",
    "B-012","A-014","B-003","A-002","B-011","A-013","B-004","A-003",
    "B-010","A-012","B-005","A-004","B-009","B-006","A-011","B-008",
    "A-005","B-007","A-010",
]


# ---------- Helpers: Smart formatting for time deltas ----------
def format_time_delta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 120.0:
        return f"{seconds:.2f} s"
    elif seconds < 7200.0:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:d}:{s:02d}"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:d}:{m:02d}:{s:02d}"

def format_time_hms(seconds: float) -> str:
    """Format time as hh:mm:ss from start (no fractions)."""
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def nice_125(value: float) -> float:
    """
    Round value to a 'nice' 1–2–5 × 10^n number.
    We still allow decimals here; caller can clamp to >= 1 if it wants integers.
    """
    v = abs(float(value))
    if v == 0:
        return 0.0
    mag = math.floor(math.log10(v))
    norm = v / (10 ** mag)

    if norm < 1.5:
        base = 1.0
    elif norm < 3.5:
        base = 2.0
    elif norm < 7.5:
        base = 5.0
    else:
        base = 10.0

    return base * (10 ** mag)


def get_viridis_colors(n):
    """Return n distinct colors from the viridis colormap as (r,g,b) tuples 0–255."""
    cmap = cm.get_cmap("viridis", n)
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors
    
def _format_no_sci(value: float, sig: int = 2) -> str:
    """Format with `sig` significant digits and NO scientific notation."""
    v = float(value)
    if v == 0:
        return "0"
    mag = math.floor(math.log10(abs(v)))
    decimals = max(0, sig - 1 - mag)
    s = f"{v:.{decimals}f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s


def format_time_compact(seconds: float) -> str:
    """2-sig-fig time with s/min/hr units, no scientific notation."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        val, unit = s, "s"
    elif s < 3600.0:
        val, unit = s / 60.0, "min"
    else:
        val, unit = s / 3600.0, "hr"
    return f"{_format_no_sci(val, 2)} {unit}"


def format_y_compact(delta_volts: float) -> str:
    """
    2-sig-fig amplitude with nV / uV / mV, no scientific notation.
    Assumes Y data are in VOLTS.
    """
    v = abs(float(delta_volts))
    if v < 1e-6:        # < 1000 nV
        return f"{_format_no_sci(v * 1e9, 2)} nV"
    elif v < 1e-3:      # < 1000 uV
        return f"{_format_no_sci(v * 1e6, 2)} uV"
    else:               # >= 1 mV
        return f"{_format_no_sci(v * 1e3, 2)} mV"



# ---------- Custom ViewBox: separate wheel zoom for X/Y ----------
class LinkedViewBox(pg.ViewBox):
    """Wheel = X-only; Shift+wheel = Y-only; Ctrl/Cmd+wheel = both."""
    def wheelEvent(self, ev):
        try:
            if hasattr(self, 'ignoreWheelEvent') and self.ignoreWheelEvent(ev):
                ev.ignore()
                return
        except Exception:
            pass

        delta = 0
        if hasattr(ev, "angleDelta"):
            delta = ev.angleDelta().y()
        elif hasattr(ev, "delta"):
            delta = ev.delta()
        if delta == 0:
            ev.ignore()
            return
        steps = delta / 120.0
        s = 0.9 ** steps

        mods = ev.modifiers()
        ctrl_or_cmd = (mods & Qt.ControlModifier) or (mods & Qt.MetaModifier)
        shift = bool(mods & Qt.ShiftModifier)

        do_x, do_y = True, False
        if ctrl_or_cmd:
            do_x, do_y = True, True
        elif shift:
            do_x, do_y = False, True

        pos = ev.position() if hasattr(ev, "position") else ev.pos()
        mousePoint = self.mapSceneToView(pos)
        center = QPointF(mousePoint.x(), mousePoint.y())

        self.scaleBy(x=(s if do_x else 1.0), y=(s if do_y else 1.0), center=center)
        ev.accept()


class _LRUCache:
    """Tiny per-channel LRU cache for HDF5 slices."""
    def __init__(self, max_segments: int = 12):
        self.max_segments = max_segments
        self._d = OrderedDict()

    def get(self, key):
        if key in self._d:
            v = self._d.pop(key)
            self._d[key] = v
            return v
        return None

    def put(self, key, value):
        if key in self._d:
            self._d.pop(key)
        self._d[key] = value
        while len(self._d) > self.max_segments:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()


class HDF5Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 Signal Viewer — Two-Column, L-Shape Cursor Scale")
        self.resize(1500, 900)
        
        self.display_order = []  # list of raw dataset indices in the desired UI order
        
        # Right panel / tabs
        self.tabs = None
        self.stats_table = None

        # Moving average filter
        self.ma_enable = False
        self.ma_points = 5      # number of points
        self.ma_passes = 1      # number of passes


        # Peak indices & extracted windows
        self._peak_indices = {}   # ch_idx -> np.ndarray of global sample indices
        self.peak_windows_df = None


        # Real-time window selection
        self.win_start_dtedit = None
        self.win_end_dtedit = None
        self.win_apply_button = None


        # Peak stats storage: ch_idx -> dict
        self._peak_stats = {}

        # Real start time (parsed from filename), or None if unknown
        self.start_datetime = None

        # Peak stats export
        self.classify_button = None

        # Derivative plotting
        self.derivative_checkbox = None
        self.show_derivative = False

        # Data-related
        self.h5file = None
        self.data = None            # h5py dataset (channels x samples)
        self.channel_names = []
        self.sample_rate = 100.0
        self.n_channels = 0
        self.n_samples = 0
        self.total_duration = 0.0

        # Selection / view state
        self.selected_channels = set()
        self.colors = {}
        self.file_name = ""
        self.default_view_secs = 10.0

        # Widgets
        self.file_label = None
        self.time_start_label = None
        self.time_end_label = None
        self.cursor_label = None
        self.time_window_spinbox = None
        self.select_all_checkbox = None
        self.unselect_all_button = None

        # Plot bookkeeping
        self._shared_plot = None
        self._curves = {}            # ch_idx -> PlotDataItem
        self._plot_widgets = []      # list[pg.PlotWidget]
        self._cell_widgets = []      # list[row-container widgets]
        self._name_labels = {}       # ch_idx -> QLabel

        # NEW: peak overlay items per channel
        self._peak_items = {}        # ch_idx -> PlotDataItem

        # Overlay scale bars per plot
        self._overlays = {}

        # Overlay scale bars per plot
        self._overlays = {}

        # Global Y limits for initial window and bounds
        self._locked_y_limits = (-1.0, 1.0)

        # Caching and throttling
        self._caches = {}            # ch_idx -> _LRUCache (raw segments)
        self._filter_caches = {}     # ch_idx -> _LRUCache (filtered segments)  # <-- NEW
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._fetch_and_render)
        self._update_delay_ms = 50

        # Default open folder
        self._default_open_dir = r"U:\MA PhD\Data\Intan\HDF5 Files\Experiment3"

        # Left panel (plots) containers
        self.plot_scroll_area = None
        self.grid_widget = None
        self.grid_layout = None

        # Right panel (controls)
        self.checkbox_group = None
        self.scroll_area = None

        # --- NEW: Filter widgets & state ---
        self.filter_enable_checkbox = None
        self.filter_type_combo = None
        self.filter_f1_spin = None
        self.filter_f2_spin = None
        self.filter_order_spin = None

        self.filter_enabled = False
        self.filter_type = "lowpass"   # "lowpass", "highpass", "bandpass"
        self.filter_order = 4
        self.filter_f1 = 0.1           # Hz (used for highpass / bandpass low)
        self.filter_f2 = 10.0          # Hz (used for lowpass / bandpass high)

        # --- Peak detection widgets & state ---
        self.peaks_enable_checkbox = None
        self.peaks_prominence_spin = None          # absolute prominence (units)
        self.peaks_min_dist_spin = None
        self.peaks_min_width_spin = None
        self.peaks_biphasic_window_spin = None

        self.peaks_enabled = False
        self.peaks_min_dist_s = 0.0         # seconds
        self.peaks_min_width_s = 0.0        # seconds
        self.peaks_biphasic_window_s = 0.5  # seconds

        # NEW: relative vs absolute thresholds
        self.peaks_use_relative_checkbox = None     # toggle
        self.peaks_k_prom_spin = None              # k for prominence
        self.peaks_k_height_spin = None            # k for height

        self.peaks_use_relative = True             # default: use k×σ
        self.peaks_k_prom = 4.0                    # prominence = 4σ
        self.peaks_k_height = 0.0                  # 0 => no height threshold

        # Absolute (fallback) thresholds
        self.peaks_prom_abs = 0.0                  # units of y
        self.peaks_height_abs = 0.0                # units of y


        # NEW: max time between opposite-sign peaks to treat as one biphasic event
        self.peaks_biphasic_window_spin = None
        self.peaks_biphasic_window_s = 0.5   # seconds (tweak as you like)

        # --- NEW: Common-mode removal widgets & state ---
        self.cm_enable_checkbox = None
        self.cm_ref_combo = None

        self.cm_enabled = False
        self.cm_ref_index = None   # raw dataset index of the reference channel


        self.init_ui()

    # -------------------------- UI SETUP --------------------------

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # === LEFT: File label + [channel panel | plots] + time labels + cursor ===
        left_layout = QVBoxLayout()
        
        # Top: file label
        self.file_label = QLabel("No file loaded")
        left_layout.addWidget(self.file_label)
        
        # Middle row: [channel scroll panel] | [plot grid scroll]
        mid_row = QHBoxLayout()
        
        # --- Channel checkbox scroll panel (moved from right side) ---
        self.checkbox_group = QVBoxLayout()
        self.checkbox_widget = QtWidget()
        self.checkbox_widget.setLayout(self.checkbox_group)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.checkbox_widget)
        self.scroll_area.setMinimumWidth(90)
        self.scroll_area.setMaximumWidth(140)
        
        mid_row.addWidget(self.scroll_area, stretch=0)
        
        # --- Plot area: scrollable grid of subplots (2 columns) ---
        self.grid_widget = QtWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(4)
        self.grid_layout.setVerticalSpacing(4)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        
        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidgetResizable(True)
        self.plot_scroll_area.setWidget(self.grid_widget)
        self.plot_scroll_area.setMinimumHeight(100)
        self.plot_scroll_area.setMaximumHeight(1200)
        
        mid_row.addWidget(self.plot_scroll_area, stretch=1)
        
        left_layout.addLayout(mid_row)
        
        # Bottom: absolute time range labels
        time_label_layout = QHBoxLayout()
        self.time_start_label = QLabel("Start: 0.00 s")
        self.time_end_label = QLabel("End: 0.00 s")
        f = QFont("Arial", 10)
        self.time_start_label.setFont(f)
        self.time_end_label.setFont(f)
        time_label_layout.addWidget(self.time_start_label)
        time_label_layout.addStretch()
        time_label_layout.addWidget(self.time_end_label)
        left_layout.addLayout(time_label_layout)
        
        # Cursor readout (bottom)
        self.cursor_label = QLabel("Cursor — t: –, y: –")
        self.cursor_label.setFont(QFont("Arial", 10))
        
        layout.addLayout(left_layout, stretch=5)
    
        # === RIGHT: Controls (will go inside Tab "Controls") ===
        control_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load File")
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setMinimumWidth(110)
        control_layout.addWidget(self.load_button)
        
        self.select_all_checkbox = QCheckBox("Select All")
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        control_layout.addWidget(self.select_all_checkbox)
        
        self.unselect_all_button = QPushButton("Unselect All")
        self.unselect_all_button.clicked.connect(self.unselect_all)
        control_layout.addWidget(self.unselect_all_button)
        
        # --- Real-time window selectors (dd/mm/yy hh:mm:ss) ---
        self.win_start_label = QLabel("Window start (dd/mm/yy hh:mm:ss):")
        self.win_start_dtedit = QDateTimeEdit()
        self.win_start_dtedit.setDisplayFormat("dd/MM/yy HH:mm:ss")
        self.win_start_dtedit.setCalendarPopup(True)
    
        self.win_end_label = QLabel("Window end (dd/mm/yy hh:mm:ss):")
        self.win_end_dtedit = QDateTimeEdit()
        self.win_end_dtedit.setDisplayFormat("dd/MM/yy HH:mm:ss")
        self.win_end_dtedit.setCalendarPopup(True)
    
        self.win_apply_button = QPushButton("Go to window")
        self.win_apply_button.clicked.connect(self._on_apply_window_times)
    
        control_layout.addWidget(self.win_start_label)
        control_layout.addWidget(self.win_start_dtedit)
        control_layout.addWidget(self.win_end_label)
        control_layout.addWidget(self.win_end_dtedit)
        control_layout.addWidget(self.win_apply_button)
    
        # Plot derivative instead of raw/filtered signal  (only ONCE)
        self.derivative_checkbox = QCheckBox("Plot derivative instead of signal")
        self.derivative_checkbox.stateChanged.connect(self._on_derivative_toggled)
        control_layout.addWidget(self.derivative_checkbox)
    
        # ---------- Common-mode removal controls ----------
        cm_title = QLabel("Common-mode removal")
        cm_title.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(cm_title)
        
        self.cm_enable_checkbox = QCheckBox("Enable common-mode")
        self.cm_enable_checkbox.stateChanged.connect(self._on_cm_params_changed)
        control_layout.addWidget(self.cm_enable_checkbox)
        
        cm_row = QHBoxLayout()
        cm_row.addWidget(QLabel("Reference:"))
        self.cm_ref_combo = QComboBox()
        self.cm_ref_combo.currentIndexChanged.connect(self._on_cm_params_changed)
        self.cm_ref_combo.setEnabled(False)  # enabled once file is loaded
        cm_row.addWidget(self.cm_ref_combo)
        cm_row.addStretch()
        control_layout.addLayout(cm_row)
        # ---------- end common-mode controls ----------
    
        # ---------- Butterworth filter controls ----------
        filter_title = QLabel("Butterworth filter")
        filter_title.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(filter_title)
    
        self.filter_enable_checkbox = QCheckBox("Enable filter")
        self.filter_enable_checkbox.stateChanged.connect(self._on_filter_params_changed)
        control_layout.addWidget(self.filter_enable_checkbox)
    
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Lowpass", "Highpass", "Bandpass"])
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_type_changed)
        type_row.addWidget(self.filter_type_combo)
        type_row.addStretch()
        control_layout.addLayout(type_row)
    
        f1_row = QHBoxLayout()
        f1_row.addWidget(QLabel("f1 (Hz):"))
        self.filter_f1_spin = QDoubleSpinBox()
        self.filter_f1_spin.setRange(0.001, 1e5)
        self.filter_f1_spin.setDecimals(3)
        self.filter_f1_spin.setSingleStep(0.1)
        self.filter_f1_spin.setValue(self.filter_f1)
        self.filter_f1_spin.valueChanged.connect(self._on_filter_params_changed)
        f1_row.addWidget(self.filter_f1_spin)
        control_layout.addLayout(f1_row)
    
        f2_row = QHBoxLayout()
        f2_row.addWidget(QLabel("f2 (Hz):"))
        self.filter_f2_spin = QDoubleSpinBox()
        self.filter_f2_spin.setRange(0.001, 1e5)
        self.filter_f2_spin.setDecimals(3)
        self.filter_f2_spin.setSingleStep(0.5)
        self.filter_f2_spin.setValue(self.filter_f2)
        self.filter_f2_spin.valueChanged.connect(self._on_filter_params_changed)
        f2_row.addWidget(self.filter_f2_spin)
        control_layout.addLayout(f2_row)
    
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Order:"))
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 10)
        self.filter_order_spin.setValue(self.filter_order)
        self.filter_order_spin.valueChanged.connect(self._on_filter_params_changed)
        order_row.addWidget(self.filter_order_spin)
        order_row.addStretch()
        control_layout.addLayout(order_row)
    
        self._update_filter_widget_states()
        # ---------- end Butterworth controls ----------

        # ---------- Moving average filter controls ----------
        ma_title = QLabel("Moving average filter")
        ma_title.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(ma_title)
    
        # Enable MA filter
        self.ma_enable_checkbox = QCheckBox("Enable moving average")
        self.ma_enable_checkbox.stateChanged.connect(self._on_ma_params_changed)
        control_layout.addWidget(self.ma_enable_checkbox)
    
        # Number of points
        ma_points_row = QHBoxLayout()
        ma_points_row.addWidget(QLabel("Points:"))
        self.ma_points_spin = QSpinBox()
        self.ma_points_spin.setRange(1, 10000)       # sensible limit
        self.ma_points_spin.setValue(self.ma_points) # default from __init__
        self.ma_points_spin.valueChanged.connect(self._on_ma_params_changed)
        ma_points_row.addWidget(self.ma_points_spin)
        ma_points_row.addStretch()
        control_layout.addLayout(ma_points_row)
    
        # Number of passes
        ma_passes_row = QHBoxLayout()
        ma_passes_row.addWidget(QLabel("Passes:"))
        self.ma_passes_spin = QSpinBox()
        self.ma_passes_spin.setRange(1, 10)
        self.ma_passes_spin.setValue(self.ma_passes)
        self.ma_passes_spin.valueChanged.connect(self._on_ma_params_changed)
        ma_passes_row.addWidget(self.ma_passes_spin)
        ma_passes_row.addStretch()
        control_layout.addLayout(ma_passes_row)
        # ---------- end Moving average filter controls ----------


        # ---------- Peak detection controls ----------
        peaks_title = QLabel("Peak detection")
        peaks_title.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(peaks_title)
        
        self.peaks_enable_checkbox = QCheckBox("Enable peak detection")
        self.peaks_enable_checkbox.stateChanged.connect(self._on_peaks_params_changed)
        control_layout.addWidget(self.peaks_enable_checkbox)
        
        self.peaks_use_relative_checkbox = QCheckBox("Use noise-based thresholds (k × σ)")
        self.peaks_use_relative_checkbox.setChecked(True)
        self.peaks_use_relative_checkbox.stateChanged.connect(self._on_peaks_params_changed)
        control_layout.addWidget(self.peaks_use_relative_checkbox)
        
        kprom_row = QHBoxLayout()
        kprom_row.addWidget(QLabel("k_prom (×σ):"))
        self.peaks_k_prom_spin = QDoubleSpinBox()
        self.peaks_k_prom_spin.setRange(0.0, 1e3)
        self.peaks_k_prom_spin.setDecimals(2)
        self.peaks_k_prom_spin.setSingleStep(0.5)
        self.peaks_k_prom_spin.setValue(self.peaks_k_prom)
        self.peaks_k_prom_spin.valueChanged.connect(self._on_peaks_params_changed)
        kprom_row.addWidget(self.peaks_k_prom_spin)
        control_layout.addLayout(kprom_row)
        
        kheight_row = QHBoxLayout()
        kheight_row.addWidget(QLabel("k_height (×σ):"))
        self.peaks_k_height_spin = QDoubleSpinBox()
        self.peaks_k_height_spin.setRange(0.0, 1e3)
        self.peaks_k_height_spin.setDecimals(2)
        self.peaks_k_height_spin.setSingleStep(0.5)
        self.peaks_k_height_spin.setValue(self.peaks_k_height)
        self.peaks_k_height_spin.valueChanged.connect(self._on_peaks_params_changed)
        kheight_row.addWidget(self.peaks_k_height_spin)
        control_layout.addLayout(kheight_row)
        
        prom_row = QHBoxLayout()
        prom_row.addWidget(QLabel("Prominence (abs units):"))
        self.peaks_prominence_spin = QDoubleSpinBox()
        self.peaks_prominence_spin.setRange(0.0, 1e9)
        self.peaks_prominence_spin.setDecimals(6)
        self.peaks_prominence_spin.setSingleStep(0.1)
        self.peaks_prominence_spin.setValue(50.0)
        self.peaks_prominence_spin.valueChanged.connect(self._on_peaks_params_changed)
        prom_row.addWidget(self.peaks_prominence_spin)
        control_layout.addLayout(prom_row)
        
        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("Min height (abs units):"))
        self.peaks_height_abs_spin = QDoubleSpinBox()
        self.peaks_height_abs_spin.setRange(0.0, 1e9)
        self.peaks_height_abs_spin.setDecimals(6)
        self.peaks_height_abs_spin.setSingleStep(0.1)
        self.peaks_height_abs_spin.setValue(50.0)
        self.peaks_height_abs_spin.valueChanged.connect(self._on_peaks_params_changed)
        height_row.addWidget(self.peaks_height_abs_spin)
        control_layout.addLayout(height_row)
    
        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("Min distance (s):"))
        self.peaks_min_dist_spin = QDoubleSpinBox()
        self.peaks_min_dist_spin.setRange(0.0, 1e6)
        self.peaks_min_dist_spin.setDecimals(3)
        self.peaks_min_dist_spin.setSingleStep(0.1)
        self.peaks_min_dist_spin.setValue(0.0)
        self.peaks_min_dist_spin.valueChanged.connect(self._on_peaks_params_changed)
        dist_row.addWidget(self.peaks_min_dist_spin)
        control_layout.addLayout(dist_row)
    
        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Min width (s):"))
        self.peaks_min_width_spin = QDoubleSpinBox()
        self.peaks_min_width_spin.setRange(0.0, 1e6)
        self.peaks_min_width_spin.setDecimals(3)
        self.peaks_min_width_spin.setSingleStep(0.1)
        self.peaks_min_width_spin.setValue(0.0)
        self.peaks_min_width_spin.valueChanged.connect(self._on_peaks_params_changed)
        width_row.addWidget(self.peaks_min_width_spin)
        control_layout.addLayout(width_row)
        
        bip_row = QHBoxLayout()
        bip_row.addWidget(QLabel("Biphasic merge (s):"))
        self.peaks_biphasic_window_spin = QDoubleSpinBox()
        self.peaks_biphasic_window_spin.setRange(0.0, 1e6)
        self.peaks_biphasic_window_spin.setDecimals(3)
        self.peaks_biphasic_window_spin.setSingleStep(0.05)
        self.peaks_biphasic_window_spin.setValue(self.peaks_biphasic_window_s)
        self.peaks_biphasic_window_spin.valueChanged.connect(self._on_peaks_params_changed)
        bip_row.addWidget(self.peaks_biphasic_window_spin)
        control_layout.addLayout(bip_row)
        
        self._update_peaks_widget_states()
        # ---------- end Peak detection controls ----------
    
        # Wrap controls in a widget for Tab 1
        controls_widget = QtWidget()
        controls_widget.setLayout(control_layout)
    
        # --- Peak stats tab ---
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(7)
        self.stats_table.setHorizontalHeaderLabels([
            "Channel", "Count", "Mean amp", "Median amp",
            "Mean dur (s)", "Median dur (s)", "Freq (peaks/min)"
        ])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.stats_table.setSelectionMode(QTableWidget.SingleSelection)

        stats_layout = QVBoxLayout()
        
        self.classify_button = QPushButton("Classify peaks")
        self.classify_button.clicked.connect(self._on_classify_peaks_clicked)
        stats_layout.addWidget(self.classify_button)
        
        self.extract_windows_button = QPushButton("Extract 4s peak windows")
        self.extract_windows_button.clicked.connect(self._on_extract_peak_windows_clicked)
        stats_layout.addWidget(self.extract_windows_button)
        
        # NEW: Save peaks button
        self.save_peaks_button = QPushButton("Save peaks")
        self.save_peaks_button.clicked.connect(self._on_save_peaks_clicked)
        stats_layout.addWidget(self.save_peaks_button)
        
        stats_layout.addWidget(self.stats_table)

           
        stats_widget = QtWidget()
        stats_widget.setLayout(stats_layout)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(controls_widget, "Controls")
        self.tabs.addTab(stats_widget, "Peak stats")
    
        layout.addWidget(self.tabs, stretch=0)
        
        # ★ NEW: shrink the right-side interface
        self.tabs.setMinimumWidth(270)
        self.tabs.setMaximumWidth(270)   # adjust as needed
        
        layout.addWidget(self.tabs, stretch=0)
    
        try:
            pg.setConfigOptions(antialias=True)
        except Exception:
            pass

    def _on_extract_peak_windows_clicked(self):
        if self.data is None:
            return
        if not self.selected_channels:
            return

        # Choose output file
        start_dir = os.path.dirname(self.file_name) if self.file_name else ""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save 10s Peak Windows CSV",
            os.path.join(start_dir, "peak_windows_10s.csv"),
            "CSV Files (*.csv)"
        )
        if not file_path:
            return

        df = self._build_peak_windows_dataframe(window_sec=4.0)
        if df is None or df.empty:
            print("[info] No peaks to export in current view.")
            return

        self.peak_windows_df = df
        try:
            df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"[warn] Failed to save peak windows CSV: {e}")

    def _on_ma_params_changed(self, *args, **kwargs):
        self.ma_enable = self.ma_enable_checkbox.isChecked()
        self.ma_points = int(self.ma_points_spin.value())
        self.ma_passes = int(self.ma_passes_spin.value())
    
        # Clear per-channel filter caches so we don't reuse old filtered segments
        if hasattr(self, "_filter_caches"):
            for cache in self._filter_caches.values():
                cache.clear()
    
        self._request_fetch()


    def _build_peak_windows_dataframe(self, window_sec: float = 10.0) -> pd.DataFrame:
        """
        For each selected channel and each detected peak (in the current view),
        extract a window of length `window_sec` centered on the peak time and
        return a long-format DataFrame.

        Rows: one sample per (channel, peak, time) within the window.
        Columns:
            channel         : channel name
            peak_id         : 0,1,2,... within that channel in current view
            sample_index    : global sample index
            time_s          : absolute time from recording start (s)
            rel_time_s      : time relative to peak center (s)
            value           : signal amplitude (µV)
            peak_time_s     : absolute peak time (s)
        """
        if self.data is None or not self.selected_channels:
            return None

        half_sec = 0.5 * float(window_sec)
        fs = float(self.sample_rate)
        n = self.n_samples

        rows = []

        for ch_idx in sorted(self.selected_channels):
            peaks = self._peak_indices.get(ch_idx)
            if peaks is None or len(peaks) == 0:
                continue

            ch_name = self.channel_names[ch_idx] if 0 <= ch_idx < len(self.channel_names) else f"ch{ch_idx}"

            for peak_id, p_idx in enumerate(peaks):
                peak_idx = int(p_idx)
                # Compute window in samples (centered on peak)
                half_samples = int(round(half_sec * fs))
                start = max(0, peak_idx - half_samples)
                end = min(n, peak_idx + half_samples)

                if end <= start:
                    continue

                # Get amplitude signal (filtered+CM-adjusted if available, but NOT derivative)
                if hasattr(self, "_get_filtered_segment"):
                    y_win = self._get_filtered_segment(ch_idx, start, end)
                else:
                    y_win = np.asarray(self.data[ch_idx, start:end], dtype=float)

                # Time vectors
                sample_idx = np.arange(start, end, dtype=int)
                time_s = sample_idx / fs
                peak_time_s = peak_idx / fs
                rel_time_s = time_s - peak_time_s

                for s_idx, t_abs, t_rel, val in zip(sample_idx, time_s, rel_time_s, y_win):
                    rows.append({
                        "channel": ch_name,
                        "peak_id": peak_id,
                        "sample_index": int(s_idx),
                        "time_s": float(t_abs),
                        "rel_time_s": float(t_rel),
                        "value": float(val),
                        "peak_time_s": float(peak_time_s),
                    })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


    def _on_derivative_toggled(self, state):
        self.show_derivative = (state == Qt.Checked)
        # Just re-fetch & redraw with the same view
        self._request_fetch()

    def _detect_peaks_in_signal(self, t: np.ndarray, y: np.ndarray):
        """
        Run peak detection on a single 1D signal y over time t and return
        per-peak details (dict of numpy arrays).
        """
        empty = {
            "peak_idx": np.array([], dtype=int),
            "peak_amp": np.array([], dtype=float),
            "start_idx": np.array([], dtype=int),
            "end_idx": np.array([], dtype=int),
            "duration_full_s": np.array([], dtype=float),
            "pwhm_s": np.array([], dtype=float),
        }
    
        if not getattr(self, "peaks_enabled", False):
            return empty
    
        if t.size == 0 or y.size == 0:
            return empty
    
        fs = float(self.sample_rate) if self.sample_rate > 0 else 1.0
    
        cfg = PeakThresholdConfig(
            use_relative=getattr(self, "peaks_use_relative", True),
            k_prom=getattr(self, "peaks_k_prom", 0.0),
            k_height=getattr(self, "peaks_k_height", 0.0),
            prom_abs=getattr(self, "peaks_prom_abs", 0.0),
            height_abs=getattr(self, "peaks_height_abs", 0.0),
            min_dist_s=getattr(self, "peaks_min_dist_s", 0.0),
            min_width_s=getattr(self, "peaks_min_width_s", 0.0),
        )
    
        biphasic_window_s = getattr(self, "peaks_biphasic_window_s", 0.0)
    
        return analyze_peaks_full(
            t=t,
            y=y,
            fs=fs,
            cfg=cfg,
            biphasic_window_s=biphasic_window_s,
        )

    
    def _build_peaks_rows_for_current_window(self):
        """
        Detect peaks in the current X window for ALL channels (0..n_channels-1)
        using the current CM/filter/MA + peak detection settings.
    
        Returns a list of dicts, each dict = one peak row.
        """
        if self.data is None or self._shared_plot is None:
            return []
    
        fs = float(self.sample_rate) if self.sample_rate > 0 else 1.0
    
        vb = self._shared_plot.getViewBox()
        (x0, x1), _ = vb.viewRange()
    
        # Clamp to recording range
        x0 = max(0.0, min(float(x0), self.total_duration))
        x1 = max(0.0, min(float(x1), self.total_duration))
        if x1 <= x0:
            x1 = min(self.total_duration, x0 + 1.0 / fs)
    
        sa = int(max(0, min(self.n_samples - 1, math.floor(x0 * fs))))
        sb = int(max(sa + 1, min(self.n_samples, math.ceil(x1 * fs))))
    
        t = np.arange(sa, sb, dtype=float) / fs
    
        rows = []
    
        for ch_idx in range(self.n_channels):
            y = self._get_filtered_segment(ch_idx, sa, sb)  # already using CM+filters+MA
            if y.size != t.size:
                continue
        
            peak_data = self._detect_peaks_in_signal(t, y)
            if peak_data is None or peak_data["peak_idx"].size == 0:
                continue
        
            ch_name = self.channel_names[ch_idx] if 0 <= ch_idx < len(self.channel_names) else f"ch{ch_idx}"
        
            peak_idx = peak_data["peak_idx"]  # indices in [0..len(y)-1] for the current window
        
            for k in range(peak_idx.size):
                local_idx = int(peak_idx[k])
        
                # --- AP features for this peak ---
                ap = extract_ap_features_for_peak(y, peak_idx, fs)
        
                # global sample index for the peak
                g_peak = sa + local_idx
                peak_time_s = g_peak / self.sample_rate
        
                if hasattr(self, "_format_cursor_timestamp"):
                    peak_time_str = self._format_cursor_timestamp(peak_time_s)
                else:
                    peak_time_str = ""
        
                # Depolarization / hyperpol times as formatted strings
                # (we know t_dep_start etc are seconds from recording start)
                if np.isfinite(ap["t_dep_start"]):
                    dep_time_str = self._format_cursor_timestamp(ap["t_dep_start"]) if hasattr(self, "_format_cursor_timestamp") else ""
                else:
                    dep_time_str = ""
        
                if np.isfinite(ap["t_hyp_start"]):
                    hyp_start_str = self._format_cursor_timestamp(ap["t_hyp_start"]) if hasattr(self, "_format_cursor_timestamp") else ""
                else:
                    hyp_start_str = ""
        
                if np.isfinite(ap["t_hyp_end"]):
                    hyp_end_str = self._format_cursor_timestamp(ap["t_hyp_end"]) if hasattr(self, "_format_cursor_timestamp") else ""
                else:
                    hyp_end_str = ""
        
                rows.append({
                    "channel": ch_name,
                    "peak_amplitude": ap["peak_amp"],       # baseline-relative, signed
                    "start_time_str": dep_time_str,
                    "peak_time_str": peak_time_str,
                    "end_time_str": hyp_end_str,
                    "duration_full_s": ap["dur_full_s"],
                    "pwhm_s": ap["fwhm_s"],                 # FWHM = PWHM for our purposes
                    "pwhm_freq_hz": (1.0 / ap["fwhm_s"]) if ap["fwhm_s"] > 0 else float("nan"),
        
                    # NEW AP metrics
                    "dep_to_hyp_start_s": ap["dur_dep_to_hyp_start_s"],
                    "hyp_duration_s": ap["dur_hyp_s"],
                    "rise_time_s": ap["rise_time_s"],
                    "decay_time_s": ap["decay_time_s"],
                    "hyp_amplitude": ap["hyp_amp"],
                    "net_area": ap["net_area"],
                    "peak_area": ap["peak_area"],
                    "hyp_area": ap["hyp_area"],
                    "spike_energy": ap["spike_energy"],
                })

    
        return rows


    def _current_peak_settings_snapshot(self):
        """Return a dict with the current CM/filter/MA/peak settings."""
        # CM
        cm_enabled = bool(self.cm_enable_checkbox.isChecked())
        cm_ref_name = ""
        if cm_enabled and getattr(self, "cm_ref_combo", None) is not None:
            idx = self.cm_ref_combo.currentIndex()
            if 0 <= idx < len(self.channel_names):
                cm_ref_name = self.channel_names[idx]
    
        # Butterworth
        butter_enabled = bool(getattr(self, "filter_enabled", False))
        butter_type = getattr(self, "filter_type", "bandpass")
        butter_f1 = float(getattr(self, "filter_f1", 0.0))
        butter_f2 = float(getattr(self, "filter_f2", 0.0))
        butter_order = int(getattr(self, "filter_order", 0))
    
        # Moving average
        ma_enabled = bool(getattr(self, "ma_enable", False))
        ma_points = int(getattr(self, "ma_points", 1))
        ma_passes = int(getattr(self, "ma_passes", 1))
    
        # Peaks
        peaks_use_relative = bool(getattr(self, "peaks_use_relative", True))
        peaks_k_prom = float(getattr(self, "peaks_k_prom", 0.0))
        peaks_k_height = float(getattr(self, "peaks_k_height", 0.0))
        peaks_prom_abs = float(getattr(self, "peaks_prom_abs", 0.0))
        peaks_height_abs = float(getattr(self, "peaks_height_abs", 0.0))
        peaks_min_dist_s = float(getattr(self, "peaks_min_dist_s", 0.0))
        peaks_min_width_s = float(getattr(self, "peaks_min_width_s", 0.0))
        peaks_biphasic_window_s = float(getattr(self, "peaks_biphasic_window_s", 0.0))
    
        show_derivative = bool(getattr(self, "show_derivative", False))
    
        return {
            "cm_enabled": cm_enabled,
            "cm_reference_channel": cm_ref_name,
            "butter_enabled": butter_enabled,
            "butter_type": butter_type,
            "butter_f1_Hz": butter_f1,
            "butter_f2_Hz": butter_f2,
            "butter_order": butter_order,
            "ma_enabled": ma_enabled,
            "ma_points": ma_points,
            "ma_passes": ma_passes,
            "peaks_use_relative": peaks_use_relative,
            "peaks_k_prom": peaks_k_prom,
            "peaks_k_height": peaks_k_height,
            "peaks_prom_abs": peaks_prom_abs,
            "peaks_height_abs": peaks_height_abs,
            "peaks_min_dist_s": peaks_min_dist_s,
            "peaks_min_width_s": peaks_min_width_s,
            "peaks_biphasic_window_s": peaks_biphasic_window_s,
            "show_derivative_for_display": show_derivative,
        }


    def _on_save_peaks_clicked(self):
        """Save one row per peak (minimal fields) + separate metadata CSV."""
        if self.data is None or self._shared_plot is None:
            return
    
        if not self.peaks_enable_checkbox.isChecked():
            print("[info] Peak detection is disabled; no peaks to save.")
            return
    
        # ---------- Build default filename based on current window ----------
        start_dir = os.path.dirname(self.file_name) if self.file_name else ""
    
        vb = self._shared_plot.getViewBox()
        (x0, x1), _ = vb.viewRange()
    
        # Clamp to recording range
        fs = float(self.sample_rate) if self.sample_rate > 0 else 1.0
        x0 = max(0.0, min(float(x0), self.total_duration))
        x1 = max(0.0, min(float(x1), self.total_duration))
        if x1 <= x0:
            x1 = min(self.total_duration, x0 + 1.0 / fs)
    
        # Use real timestamps if we know the recording start time
        if getattr(self, "start_datetime", None) is not None:
            from datetime import timedelta
            dt0 = self.start_datetime + timedelta(seconds=x0)
            dt1 = self.start_datetime + timedelta(seconds=x1)
            # ddmmyy_HHMMSS → "250811_004602"
            fn0 = dt0.strftime("%d%m%y_%H%M%S")
            fn1 = dt1.strftime("%d%m%y_%H%M%S")
        else:
            # Fallback: use seconds if no real time available
            fn0 = f"t{x0:.1f}s"
            fn1 = f"t{x1:.1f}s"
    
        default_filename = f"PeaksFrom_{fn0}_to_{fn1}.csv"
        default_path = os.path.join(start_dir, default_filename)
    
        # ---------- Ask user once where to save PEAKS CSV ----------
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save peaks CSV",
            default_path,
            "CSV Files (*.csv)"
        )
        if not file_path:
            return
    
        # Build peak rows for current window
        rows = self._build_peaks_rows_for_current_window()
        if not rows:
            print("[info] No peaks found in current window.")
            return
    
        # Derive metadata file path: same base name + "_metadata"
        root, ext = os.path.splitext(file_path)
        if not ext:
            ext = ".csv"
        meta_path = root + "_metadata" + ext
    
        # ------------------ 1) PEAKS CSV ------------------
        peaks_header = [
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
    
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(peaks_header)
    
                for row in rows:
                    amp = float(row["peak_amplitude"])
                    dur = float(row["duration_full_s"])
                    pwhm = float(row["pwhm_s"])
                    freq = float(row["pwhm_freq_hz"])
    
                    writer.writerow([
                        row["channel"],
                        f"{float(row['peak_amplitude']):.1f}",
                        row["start_time_str"],
                        row["peak_time_str"],
                        row["end_time_str"],
                        f"{float(row['duration_full_s']):.1f}",
                        f"{float(row['pwhm_s']):.1f}",
                        f"{float(row['pwhm_freq_hz']):.2f}",
                        f"{float(row['dep_to_hyp_start_s']):.1f}" if np.isfinite(row["dep_to_hyp_start_s"]) else "",
                        f"{float(row['hyp_duration_s']):.1f}" if np.isfinite(row["hyp_duration_s"]) else "",
                        f"{float(row['rise_time_s']):.1f}" if np.isfinite(row["rise_time_s"]) else "",
                        f"{float(row['decay_time_s']):.1f}" if np.isfinite(row["decay_time_s"]) else "",
                        f"{float(row['hyp_amplitude']):.1f}" if np.isfinite(row["hyp_amplitude"]) else "",
                        f"{float(row['net_area']):.3e}" if np.isfinite(row["net_area"]) else "",
                        f"{float(row['peak_area']):.3e}" if np.isfinite(row["peak_area"]) else "",
                        f"{float(row['hyp_area']):.3e}" if np.isfinite(row["hyp_area"]) else "",
                        f"{float(row['spike_energy']):.3e}" if np.isfinite(row["spike_energy"]) else "",
                    ])

        except Exception as e:
            print(f"[warn] Failed to save peaks CSV: {e}")
            return
    
        # ------------------ 2) METADATA CSV ------------------
        settings = self._current_peak_settings_snapshot()
    
        meta_cols = [
            "cm_enabled",
            "cm_reference_channel",
            "butter_enabled",
            "butter_type",
            "butter_f1_Hz",
            "butter_f2_Hz",
            "butter_order",
            "ma_enabled",
            "ma_points",
            "ma_passes",
            "peaks_use_relative",
            "peaks_k_prom",
            "peaks_k_height",
            "peaks_prom_abs",
            "peaks_height_abs",
            "peaks_min_dist_s",
            "peaks_min_width_s",
            "peaks_biphasic_window_s",
            "show_derivative_for_display",
        ]
    
        try:
            with open(meta_path, "w", newline="", encoding="utf-8") as fmeta:
                writer = csv.writer(fmeta)
                writer.writerow(meta_cols)
                writer.writerow([settings.get(k, "") for k in meta_cols])
        except Exception as e:
            print(f"[warn] Failed to save metadata CSV: {e}")


    def _compute_peak_stats_vector(self, t: np.ndarray, y: np.ndarray):
        """
        Compute peak stats (count, amplitude, duration, freq/min) for a single
        channel over the given window [t[0], t[-1]] using the current peak
        detection settings (relative/absolute thresholds, biphasic merge, etc.).

        Returns a dict with keys:
            count, mean_amp, median_amp, mean_dur, median_dur, freq_per_min
        """
        # Default "no peaks" stats
        empty_stats = {
            "count": 0,
            "mean_amp": np.nan,
            "median_amp": np.nan,
            "mean_dur": np.nan,
            "median_dur": np.nan,
            "freq_per_min": 0.0,
        }

        if not getattr(self, "peaks_enabled", False):
            return empty_stats

        if t.size == 0 or y.size == 0:
            return empty_stats

        fs = float(self.sample_rate) if self.sample_rate > 0 else 1.0
        y_float = np.asarray(y, dtype=float)

        # ---------- 1) Estimate noise σ on this window ----------
        sigma = estimate_noise_sigma_mad(y_float)


        # ---------- 2) Build thresholds ----------
        cfg = PeakThresholdConfig(
            use_relative=getattr(self, "peaks_use_relative", True),
            k_prom=getattr(self, "peaks_k_prom", 0.0),
            k_height=getattr(self, "peaks_k_height", 0.0),
            prom_abs=getattr(self, "peaks_prom_abs", 0.0),
            height_abs=getattr(self, "peaks_height_abs", 0.0),
            min_dist_s=getattr(self, "peaks_min_dist_s", 0.0),
            min_width_s=getattr(self, "peaks_min_width_s", 0.0),
        )
        
        kwargs = build_peak_find_kwargs(
            sigma=sigma,
            fs=fs,
            cfg=cfg,
        )


        # ---------- Positive & negative peaks ----------
        peaks_pos, peaks_neg = find_pos_neg_peaks(y_float, kwargs)

        if peaks_pos.size == 0 and peaks_neg.size == 0:
            return empty_stats

        indices = np.concatenate([peaks_pos, peaks_neg])
        signs = np.concatenate([
            np.ones_like(peaks_pos, dtype=int),
            -np.ones_like(peaks_neg, dtype=int),
        ])

        order = np.argsort(indices)
        indices = indices[order]
        signs = signs[order]

        # ---------- Biphasic merge ----------
        keep = np.ones(len(indices), dtype=bool)
        window_s = getattr(self, "peaks_biphasic_window_s", 0.0)
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
                        keep[j] = False

        final_indices = indices[keep]
        if final_indices.size == 0:
            return empty_stats

        # ---------- Amplitudes & durations ----------
        amps = np.abs(y_float[final_indices])

        durations = []
        for idx in final_indices:
            amp = np.abs(y_float[idx] - med)
            if amp <= 0:
                durations.append(0.0)
                continue
            thresh = 0.5 * amp

            left = idx
            while left > 0 and np.abs(y_float[left] - med) >= thresh:
                left -= 1

            right = idx
            n = len(y_float)
            while right < n - 1 and np.abs(y_float[right] - med) >= thresh:
                right += 1

            width_samples = max(1, right - left)
            durations.append(width_samples / fs)

        durations = np.asarray(durations, dtype=float)

        count = final_indices.size
        window_duration = max(1e-6, t[-1] - t[0])
        freq_per_min = count / (window_duration / 60.0)

        return {
            "count": int(count),
            "mean_amp": float(np.mean(amps)) if count > 0 else np.nan,
            "median_amp": float(np.median(amps)) if count > 0 else np.nan,
            "mean_dur": float(np.mean(durations)) if durations.size > 0 else np.nan,
            "median_dur": float(np.median(durations)) if durations.size > 0 else np.nan,
            "freq_per_min": float(freq_per_min),
        }

    def _on_classify_peaks_clicked(self):
        if self.data is None or self._shared_plot is None:
            return

        # Choose output path
        start_dir = os.path.dirname(self.file_name) if self.file_name else ""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Peak Stats CSV",
            os.path.join(start_dir, "peak_stats.csv"),
            "CSV Files (*.csv)"
        )
        if not file_path:
            return

        self._export_peak_stats_long_csv(file_path)

    def _on_derivative_toggled(self, state):
        self.show_derivative = (state == Qt.Checked)
        # Just re-fetch & redraw with the same view
        self._request_fetch()


    def _export_peak_stats_long_csv(self, file_path: str):
        """Export peak stats for ALL channels in current view window to a long-format CSV."""
        if self.data is None or self._shared_plot is None:
            return

        vb = self._shared_plot.getViewBox()
        (x0, x1), _ = vb.viewRange()

        # Clamp view range to valid time
        x0 = max(0.0, min(float(x0), self.total_duration))
        x1 = max(0.0, min(float(x1), self.total_duration))
        if x1 <= x0:
            x1 = min(self.total_duration, x0 + 1.0 / max(self.sample_rate, 1.0))

        # Convert to sample indices for THIS exact window (no margin)
        sa = int(max(0, min(self.n_samples - 1, math.floor(x0 * self.sample_rate))))
        sb = int(max(sa + 1, min(self.n_samples, math.ceil(x1 * self.sample_rate))))

        t = np.arange(sa, sb) / self.sample_rate

        # Human-readable window times (real time if available)
            
        if hasattr(self, "_format_cursor_timestamp"):
            win_start_str = self._format_cursor_timestamp(x0)
            win_end_str = self._format_cursor_timestamp(x1)
        else:
            win_start_str = f"{x0:.3f}"
            win_end_str = f"{x1:.3f}"

        # Prepare long-format rows
        rows = []
        stat_map = [
            ("count", "count"),
            ("mean_amp", "mean_amp"),
            ("median_amp", "median_amp"),
            ("mean_dur_s", "mean_dur"),
            ("median_dur_s", "median_dur"),
            ("freq_per_min", "freq_per_min"),
        ]

        for ch_idx in range(self.n_channels):
            # Use filtered/CM-adjusted data if you have a helper, otherwise raw segment
            if hasattr(self, "_get_filtered_segment"):
                y = self._get_filtered_segment(ch_idx, sa, sb)
            else:
                y = self._get_segment(ch_idx, sa, sb)

            stats = self._compute_peak_stats_vector(t, y)
            ch_name = self.channel_names[ch_idx] if 0 <= ch_idx < len(self.channel_names) else f"ch{ch_idx}"

            for csv_name, key in stat_map:
                val = stats.get(key, np.nan)
                rows.append([
                    ch_name,
                    csv_name,
                    val,
                    x0,
                    x1,
                    win_start_str,
                    win_end_str,
                ])

        # Write CSV (long format)
        header = [
            "channel",
            "stat_name",
            "stat_value",
            "window_start_s",
            "window_end_s",
            "window_start_time",
            "window_end_time",
        ]

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for r in rows:
                    writer.writerow(r)
        except Exception as e:
            # You can replace this with a QMessageBox if you want
            print(f"[warn] Failed to write CSV: {e}")


    def _init_start_datetime_from_filename(self, file_path: str):
        """
        Try to parse a start datetime from the HDF5 filename.

        Expected pattern (example):
            ..._250810_190430.h5  ->  YYMMDD_HHMMSS

        Interpreted as:
            year = 2000 + YY   (so 25 -> 2025)
        """
        base = os.path.basename(file_path)
        stem, _ = os.path.splitext(base)
        parts = stem.split("_")

        self.start_datetime = None  # default

        if len(parts) < 2:
            return

        date_token = parts[-2]
        time_token = parts[-1]

        if not (len(date_token) == 6 and len(time_token) == 6):
            return
        if not (date_token.isdigit() and time_token.isdigit()):
            return

        yy = int(date_token[0:2])
        mm = int(date_token[2:4])
        dd = int(date_token[4:6])
        HH = int(time_token[0:2])
        MM = int(time_token[2:4])
        SS = int(time_token[4:6])

        # Simple YY->YYYY mapping: 00–69 -> 2000–2069, 70–99 -> 1970–1999
        year = 2000 + yy if yy < 70 else 1900 + yy

        try:
            self.start_datetime = datetime(year, mm, dd, HH, MM, SS)
        except ValueError:
            # invalid date/time -> leave as None
            self.start_datetime = None


    def _update_filter_widget_states(self):
        """Enable/disable f2 depending on filter type."""
        if self.filter_type_combo is None or self.filter_f2_spin is None:
            return
        text = self.filter_type_combo.currentText().lower()
        is_band = "band" in text
        # For bandpass we use f1 and f2; for low/high just one cutoff
        self.filter_f2_spin.setEnabled(is_band or "low" in text)

    def _on_filter_type_changed(self):
        if self.filter_type_combo is None:
            return
        text = self.filter_type_combo.currentText().lower()
        if "low" in text:
            self.filter_type = "lowpass"
        elif "high" in text:
            self.filter_type = "highpass"
        else:
            self.filter_type = "bandpass"
        self._update_filter_widget_states()
        self._on_filter_params_changed()  # re-apply with new type

    def _clear_filter_caches(self):
        """Drop all cached filtered segments (called when params change)."""
        self._filter_caches = {i: _LRUCache(max_segments=12) for i in range(len(self.channel_names))}

    def _on_filter_params_changed(self):
        """Sync GUI → state, clear filter caches, trigger redraw."""
        if self.filter_enable_checkbox is not None:
            self.filter_enabled = self.filter_enable_checkbox.isChecked()
        if self.filter_order_spin is not None:
            self.filter_order = int(self.filter_order_spin.value())
        if self.filter_f1_spin is not None:
            self.filter_f1 = float(self.filter_f1_spin.value())
        if self.filter_f2_spin is not None:
            self.filter_f2 = float(self.filter_f2_spin.value())

        self._clear_filter_caches()
        self._request_fetch()  # will refetch & refilter visible range

    def _filter_signature(self):
        return (
            bool(self.filter_enabled),
            str(self.filter_type),
            float(self.filter_f1),
            float(self.filter_f2),
            int(self.filter_order),
    
            # Moving average parameters
            bool(getattr(self, "ma_enable", False)),
            int(getattr(self, "ma_points", 1)),
            int(getattr(self, "ma_passes", 1)),
        )


    def _on_peaks_params_changed(self):
        """Sync GUI → peak detection state and trigger redraw."""
        if self.peaks_enable_checkbox is not None:
            self.peaks_enabled = self.peaks_enable_checkbox.isChecked()

        if self.peaks_use_relative_checkbox is not None:
            self.peaks_use_relative = self.peaks_use_relative_checkbox.isChecked()

        if self.peaks_k_prom_spin is not None:
            self.peaks_k_prom = float(self.peaks_k_prom_spin.value())

        if self.peaks_k_height_spin is not None:
            self.peaks_k_height = float(self.peaks_k_height_spin.value())

        if self.peaks_prominence_spin is not None:
            self.peaks_prom_abs = float(self.peaks_prominence_spin.value())

        if getattr(self, "peaks_height_abs_spin", None) is not None:
            self.peaks_height_abs = float(self.peaks_height_abs_spin.value())

        if self.peaks_min_dist_spin is not None:
            self.peaks_min_dist_s = float(self.peaks_min_dist_spin.value())

        if self.peaks_min_width_spin is not None:
            self.peaks_min_width_s = float(self.peaks_min_width_spin.value())

        if self.peaks_biphasic_window_spin is not None:
            self.peaks_biphasic_window_s = float(self.peaks_biphasic_window_spin.value())

        # Update which widgets are active
        self._update_peaks_widget_states()

        self._request_fetch()



    
    # -------------------------- FILE I/O --------------------------
    def load_file(self):
        start_dir = self._default_open_dir if os.path.isdir(self._default_open_dir) else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open HDF5 File", start_dir, "HDF5 Files (*.h5 *.hdf5)"
        )
        if not file_path:
            return

        self.file_label.setText("Loading file… please wait")
        QApplication.processEvents()

        if self.h5file is not None:
            try:
                self.h5file.close()
            except Exception:
                pass
            self.h5file = None

        try:
            self.h5file = h5py.File(file_path, "r")
            self.data = self.h5file["amplifier_data"]  # shape: (channels, samples)
            
                        
            self.n_channels, self.n_samples = self.data.shape

            raw_names = self.h5file["channel_names"][:]
            self.channel_names = [n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n) for n in raw_names]

            # --- Build display order: map preferred names -> indices, then append leftovers ---
            name_to_idx = {str(n): i for i, n in enumerate(self.channel_names)}
            ordered = []
            seen = set()
            
            for nm in PREFERRED_ORDER:
                nm = nm.strip()
                if nm in name_to_idx:
                    idx = name_to_idx[nm]
                    if idx not in seen:
                        ordered.append(idx)
                        seen.add(idx)

            # Append any channels not specified, preserving their original order
            for i, nm in enumerate(self.channel_names):
                if i not in seen:
                    ordered.append(i)

            self.display_order = ordered


            self.sample_rate = float(self.h5file.attrs.get("sampling_rate_Hz", 1000.0))
            self.total_duration = self.n_samples / self.sample_rate
            self._populate_cm_ref_combo()

            
        except Exception as e:
            self.file_label.setText(f"Error loading file: {str(e)}")
            return

        self.file_name = os.path.basename(file_path)
        self._init_start_datetime_from_filename(file_path)
        self.file_label.setText(
            f"Loaded file: {self.file_name}  |  Channels: {self.n_channels}  |  Fs: {self.sample_rate:g} Hz  |  Duration: {self.total_duration:.2f} s"
        )
        
        # Initialize window start/end controls if we have a real start time
        if self.start_datetime is not None and self.win_start_dtedit is not None:
            # Start at recording start
            dt_start = self.start_datetime
            # End default = start + preferred view or end of recording
            dt_end = self.start_datetime + timedelta(
                seconds=min(self.default_view_secs, self.total_duration)
            )

            qdt_start = QDateTime(
                dt_start.year, dt_start.month, dt_start.day,
                dt_start.hour, dt_start.minute, dt_start.second
            )
            qdt_end = QDateTime(
                dt_end.year, dt_end.month, dt_end.day,
                dt_end.hour, dt_end.minute, dt_end.second
            )

            self.win_start_dtedit.blockSignals(True)
            self.win_end_dtedit.blockSignals(True)
            self.win_start_dtedit.setDateTime(qdt_start)
            self.win_end_dtedit.setDateTime(qdt_end)
            self.win_start_dtedit.blockSignals(False)
            self.win_end_dtedit.blockSignals(False)

            self.win_start_dtedit.setEnabled(True)
            self.win_end_dtedit.setEnabled(True)
            self.win_apply_button.setEnabled(True)
        else:
            # No real-time info → disable controls
            if self.win_start_dtedit is not None:
                self.win_start_dtedit.setEnabled(False)
            if self.win_end_dtedit is not None:
                self.win_end_dtedit.setEnabled(False)
            if self.win_apply_button is not None:
                self.win_apply_button.setEnabled(False)

        
        self.file_label.setText(
            f"Loaded file: {self.file_name}  |  Channels: {self.n_channels}  |  Fs: {self.sample_rate:g} Hz  |  Duration: {self.total_duration:.2f} s"
        )

        self._build_channel_ui()
        self._rebuild_plots()

        if self._shared_plot is not None:
            x0, x1 = 0.0, min(self.default_view_secs, self.total_duration)
            self._shared_plot.setXRange(x0, x1, padding=0)
        self._request_fetch()

    def closeEvent(self, event):
        if self.h5file:
            try:
                self.h5file.close()
            except Exception:
                pass
        event.accept()

    def _on_apply_window_times(self):
        """Apply the user-selected real-time window to the X range."""
        if self.data is None or self._shared_plot is None:
            return
        if self.start_datetime is None:
            return
        if self.win_start_dtedit is None or self.win_end_dtedit is None:
            return

        # Read QDateTime values
        qdt_start = self.win_start_dtedit.dateTime()
        qdt_end = self.win_end_dtedit.dateTime()

        # Convert to Python datetime (naive, same as start_datetime)
        dt_start = datetime(
            qdt_start.date().year(), qdt_start.date().month(), qdt_start.date().day(),
            qdt_start.time().hour(), qdt_start.time().minute(), qdt_start.time().second()
        )
        dt_end = datetime(
            qdt_end.date().year(), qdt_end.date().month(), qdt_end.date().day(),
            qdt_end.time().hour(), qdt_end.time().minute(), qdt_end.time().second()
        )

        # Convert to seconds-from-start
        t0 = (dt_start - self.start_datetime).total_seconds()
        t1 = (dt_end - self.start_datetime).total_seconds()

        # If user accidentally reversed them, fix order
        if t1 < t0:
            t0, t1 = t1, t0

        # Clamp to recording range
        t0 = max(0.0, min(t0, self.total_duration))
        t1 = max(0.0, min(t1, self.total_duration))

        # Ensure non-zero width (tiny epsilon if user picked same time)
        if t1 <= t0:
            t1 = min(self.total_duration, t0 + 1.0 / max(self.sample_rate, 1.0))

        # Apply to plot
        self._shared_plot.setXRange(t0, t1, padding=0)
        self._request_fetch()


    # -------------------------- CHANNEL UI --------------------------
    def _build_channel_ui(self):
        # Clear checkboxes
        for i in reversed(range(self.checkbox_group.count())):
            it = self.checkbox_group.itemAt(i)
            w = it.widget() if it is not None else None
            if w is not None:
                w.setParent(None)
    
        self.selected_channels.clear()
    
        # Colors & caches (unchanged)
        # Use viridis colormap for channel colors
        viridis_colors = get_viridis_colors(len(self.channel_names))
        self.colors = {
            name: viridis_colors[i]
            for i, name in enumerate(self.channel_names)
        }

        self._caches = {i: _LRUCache(max_segments=12) for i in range(len(self.channel_names))}
        self._filter_caches = {i: _LRUCache(max_segments=12) for i in range(len(self.channel_names))}  # <-- NEW

    
        # Create checkboxes in display order; stash raw index on the widget
        for raw_idx in self.display_order:
            name = self.channel_names[raw_idx]
            cb = QCheckBox(name)
            cb.setChecked(False)
            cb.setProperty("ch_idx", raw_idx)
            cb.stateChanged.connect(self._on_channel_toggled)
            self.checkbox_group.addWidget(cb)
    
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(False)
        self.select_all_checkbox.blockSignals(False)


    def _populate_cm_ref_combo(self):
        """Fill the common-mode reference combobox with channel names."""
        if self.cm_ref_combo is None:
            return

        self.cm_ref_combo.blockSignals(True)
        self.cm_ref_combo.clear()

        if not self.channel_names:
            self.cm_ref_combo.setEnabled(False)
            self.cm_ref_index = None
            self.cm_ref_combo.blockSignals(False)
            return

        # Use original channel order here; you could also use display_order.
        for idx, name in enumerate(self.channel_names):
            self.cm_ref_combo.addItem(name, userData=int(idx))

        self.cm_ref_combo.setEnabled(True)
        self.cm_ref_combo.setCurrentIndex(0)
        self.cm_ref_index = int(self.cm_ref_combo.currentData())

        self.cm_ref_combo.blockSignals(False)

    def _on_cm_params_changed(self):
        """Update cm_enabled and cm_ref_index from GUI and refresh."""
        if self.cm_enable_checkbox is not None:
            self.cm_enabled = self.cm_enable_checkbox.isChecked()

        if self.cm_ref_combo is not None and self.cm_ref_combo.isEnabled():
            data = self.cm_ref_combo.currentData()
            self.cm_ref_index = int(data) if data is not None else None
        else:
            self.cm_ref_index = None

        # Filtering caches depend on the effective input; safest is to clear them
        if hasattr(self, "_clear_filter_caches"):
            self._clear_filter_caches()

        self._request_fetch()


    def _on_channel_toggled(self):
        # preserve view (your existing code)
        prev_x = prev_y = None
        if self._shared_plot is not None:
            xr, yr = self._shared_plot.getViewBox().viewRange()
            prev_x = (float(xr[0]), float(xr[1]))
            prev_y = (float(yr[0]), float(yr[1]))
    
        # Recompute selection based on display-ordered checkboxes
        self.selected_channels.clear()
        for i in range(self.checkbox_group.count()):
            cb = self.checkbox_group.itemAt(i).widget()
            if cb and cb.isChecked():
                ch_idx = cb.property("ch_idx")
                if ch_idx is not None:
                    self.selected_channels.add(int(ch_idx))
    
        # Rebuild plots with preserved ranges
        self._rebuild_plots(preserve_xrange=prev_x, preserve_yrange=prev_y)
        self._request_fetch()




    def toggle_select_all(self, state):
        block = (state == Qt.Checked)
        for i in range(self.checkbox_group.count()):
            cb = self.checkbox_group.itemAt(i).widget()
            if cb is None:
                continue
            cb.blockSignals(True)
            cb.setChecked(block)
            cb.blockSignals(False)
        self._on_channel_toggled()

    def unselect_all(self):
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(False)
        self.select_all_checkbox.blockSignals(False)
        for i in range(self.checkbox_group.count()):
            cb = self.checkbox_group.itemAt(i).widget()
            if cb is None:
                continue
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._on_channel_toggled()

    # -------------------------- VIEW/ZOOM --------------------------
    def _on_pref_view_width(self, value: float):
        self.default_view_secs = float(value)
        if self._shared_plot is not None:
            vb = self._shared_plot.plotItem.vb
            x0, x1 = vb.viewRange()[0]
            cx = 0.5 * (x0 + x1)
            half = 0.5 * self.default_view_secs
            nx0, nx1 = max(0.0, cx - half), min(self.total_duration, cx + half)
            if nx1 - nx0 < 1e-6:
                nx1 = min(self.total_duration, nx0 + 1e-3)
            self._shared_plot.setXRange(nx0, nx1, padding=0)
            self._request_fetch()

    def _on_view_changed(self, *args, **kwargs):
        # Refresh scale bars because data-units-per-pixel changed
        for pw, ov in self._overlays.items():
            if ov.get('last_xy') is not None:
                self._update_scalebar_for_plot(pw, ov['last_xy'][0], ov['last_xy'][1])
        self._request_fetch()

        # NEW: update real-time window controls to match current view
        if self.start_datetime is not None and self.win_start_dtedit is not None:
            vb = self._shared_plot.getViewBox()
            (x0, x1), _ = vb.viewRange()

            x0 = max(0.0, min(float(x0), self.total_duration))
            x1 = max(0.0, min(float(x1), self.total_duration))

            dt_start = self.start_datetime + timedelta(seconds=x0)
            dt_end = self.start_datetime + timedelta(seconds=x1)

            qdt_start = QDateTime(
                dt_start.year, dt_start.month, dt_start.day,
                dt_start.hour, dt_start.minute, dt_start.second
            )
            qdt_end = QDateTime(
                dt_end.year, dt_end.month, dt_end.day,
                dt_end.hour, dt_end.minute, dt_end.second
            )

            self.win_start_dtedit.blockSignals(True)
            self.win_end_dtedit.blockSignals(True)
            self.win_start_dtedit.setDateTime(qdt_start)
            self.win_end_dtedit.setDateTime(qdt_end)
            self.win_start_dtedit.blockSignals(False)
            self.win_end_dtedit.blockSignals(False)


    def _request_fetch(self):
        self._update_timer.start(self._update_delay_ms)

    # -------------------------- PLOTS --------------------------
    def _clear_plots(self):
        for pw in self._plot_widgets:
            try:
                pw.removeEventFilter(self)
            except Exception:
                pass
            try:
                pw.scene().sigMouseMoved.disconnect()
            except Exception:
                pass
            ov = self._overlays.get(pw)
            if ov:
                for k in ('hbar', 'vbar', 'htext', 'vtext'):
                    try:
                        if ov.get(k) is not None:
                            pw.removeItem(ov[k])
                    except Exception:
                        pass
        self._plot_widgets.clear()
        self._curves.clear()
        self._name_labels.clear()
        self._overlays.clear()
        self._shared_plot = None
        
        # NEW: clear peak items & stats
        self._peak_items.clear()
        self._peak_stats.clear()
     
        for w in self._cell_widgets:
            try:
                w.setParent(None)
                w.deleteLater()
            except Exception:
                pass
        self._cell_widgets.clear()

        for i in reversed(range(self.grid_layout.count())):
            it = self.grid_layout.takeAt(i)
            w = it.widget()
            if w is not None:
                try:
                    w.setParent(None)
                    w.deleteLater()
                except Exception:
                    pass

    def _rebuild_plots(self, preserve_xrange=None, preserve_yrange=None):
        self._clear_plots()
        if self.data is None or not self.selected_channels:
            self.time_start_label.setText("Start: 0.00 s")
            self.time_end_label.setText("End: 0.00 s")
            self.cursor_label.setText("Cursor — t: –, y: –, channel: –")
            return

        label_font = QFont("Arial", 14)  # large channel labels
        label_font.setBold(True)

        sel_in_order = [idx for idx in self.display_order if idx in self.selected_channels]
        n = len(sel_in_order)
        rows = math.ceil(n / 2)

        first_pw = None

        for r in range(rows):
            left_i = 2 * r
            if left_i < n:
                ch_idx = sel_in_order[left_i]
                cell, pw = self._make_plot_cell(
                    ch_idx=ch_idx, label_font=label_font, label_side='left'
                )
                self.grid_layout.addWidget(cell, r, 0)
                self._cell_widgets.append(cell)
                first_pw = self._link_plotwidget(pw, first_pw)
        
            right_i = 2 * r + 1
            if right_i < n:
                ch_idx = sel_in_order[right_i]
                cell, pw = self._make_plot_cell(
                    ch_idx=ch_idx, label_font=label_font, label_side='right'
                )
                self.grid_layout.addWidget(cell, r, 1)
                self._cell_widgets.append(cell)
                first_pw = self._link_plotwidget(pw, first_pw)

        # --- Restore or set initial view BEFORE any fetch ---
        if first_pw is not None:
            # Disable autorange on X as well so pyqtgraph never jumps to full span
            first_pw.enableAutoRange(x=False, y=False)
        
            if preserve_xrange is not None:
                x0, x1 = preserve_xrange
            else:
                # First channel selected: force first 10 s (or self.default_view_secs)
                x0 = 0.0
                x1 = min(self.default_view_secs, self.total_duration)
            first_pw.setXRange(x0, x1, padding=0)
        
            if preserve_yrange is not None:
                y0, y1 = preserve_yrange
                first_pw.setYRange(y0, y1, padding=0)



    def _make_plot_cell(self, ch_idx, label_font, label_side='left'):
        """Creates a QWidget cell containing [label|plot] or [plot|label]."""
        cell = QtWidget()
        lay = QHBoxLayout(cell)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(3)

        vb = LinkedViewBox()
        pw = pg.PlotWidget(viewBox=vb)
        pw.setMouseEnabled(x=True, y=True)
        pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pw.setMinimumHeight(80)
        pw.plotItem.layout.setContentsMargins(0, 0, 0, 0)

        # Hide ALL axes inside the plot
        for ax in ('left', 'right', 'top', 'bottom'):
            pw.getPlotItem().hideAxis(ax)

        # Channel label outside the plot
        ch_name = self.channel_names[ch_idx]
        name_lbl = QLabel(ch_name)
        name_lbl.setFont(label_font)
        if label_side == 'left':
            name_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        else:
            name_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        name_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        name_lbl.setMinimumWidth(60)
        name_lbl.setToolTip(ch_name)
        self._name_labels[ch_idx] = name_lbl

        # Curve
        curve = pw.plot([], [], pen=pg.mkPen(self.colors[ch_name], width=1))
        try:
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, mode='peak')
            curve.setSkipFiniteCheck(True)
        except Exception:
            pass
        self._curves[ch_idx] = curve

        # NEW: peak vertical lines overlay for this channel
        peak_item = pw.plot([], [], pen=pg.mkPen(150, 150, 255, 160, width=1))
        peak_item.setZValue(curve.zValue() - 1)  # slightly behind the main curve if you prefer
        self._peak_items[ch_idx] = peak_item


        # Mouse move -> cursor readout + scale bars
        pw.scene().sigMouseMoved.connect(
            lambda pos, pw=pw, ch_idx=ch_idx: self._on_mouse_moved(pos, pw, ch_idx)
        )
        # Hide overlay when cursor leaves this plot
        pw.installEventFilter(self)

        # Assemble according to side
        if label_side == 'left':
            lay.addWidget(name_lbl, stretch=0)
            lay.addWidget(pw, stretch=1)
        else:
            lay.addWidget(pw, stretch=1)
            lay.addWidget(name_lbl, stretch=0)

        self._plot_widgets.append(pw)
        self._ensure_overlay(pw)
        return cell, pw

    def eventFilter(self, obj, event):
        # Hide the overlay for a plot when the cursor leaves that plot
        if event.type() == QEvent.Leave and obj in self._plot_widgets:
            self._hide_overlay(obj)
        return False  # continue normal processing

    def _link_plotwidget(self, pw: pg.PlotWidget, first_pw: pg.PlotWidget):
        """Link pw to first_pw for X/Y; connect signals on the master; set limits."""
        vb = pw.getViewBox()
    
        # No caps on Y (per your last change), but initialize a sane Y window.
        y_min, y_max = self._locked_y_limits
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min, y_max = -1.0, 1.0
    
        # Turn off BOTH X and Y autorange so setData() won't snap to full span.
        pw.enableAutoRange(x=False, y=False)
        pw.setYRange(y_min, y_max, padding=0)
    
        # Keep X clamped only; Y unlimited
        vb.setLimits(
            xMin=0.0,
            xMax=max(0.0, self.total_duration),
            # yMin/yMax omitted => unlimited Y
        )
    
        if first_pw is None:
            first_pw = pw
            self._shared_plot = pw
            vb.sigXRangeChanged.connect(self._on_view_changed)
            vb.sigYRangeChanged.connect(self._on_view_changed)
        else:
            pw.setXLink(first_pw)
            pw.setYLink(first_pw)
    
        return first_pw



    # -------------------------- Overlay (cursor L-shape scale bars) --------------------------
    def _ensure_overlay(self, pw: pg.PlotWidget):
        if pw in self._overlays:
            return
        hbar = pg.PlotDataItem([], [], pen=pg.mkPen(200, 200, 200, 220, width=3))
        vbar = pg.PlotDataItem([], [], pen=pg.mkPen(200, 200, 200, 220, width=3))
        hbar.setZValue(10)
        vbar.setZValue(10)
        pw.addItem(hbar)
        pw.addItem(vbar)
    
        # Labels with anchors for requested placement
        htext = pg.TextItem("", anchor=(0.5, 1.0), color=(230, 230, 230))  # above the right end
        vtext = pg.TextItem("", anchor=(1.0, 1.0), color=(230, 230, 230))  # top-left of vertical bar
        htext.setZValue(11)
        vtext.setZValue(11)
        pw.addItem(htext)
        pw.addItem(vtext)
    
        ctext = pg.TextItem("", anchor=(0.5, 0.0), color=(230, 230, 230))

        ctext.setZValue(12)
        pw.addItem(ctext)
    
        self._overlays[pw] = {
            'hbar': hbar,
            'vbar': vbar,
            'htext': htext,
            'vtext': vtext,
            'ctext': ctext,   # NEW
            'last_xy': None
        }


    def _hide_overlay(self, pw: pg.PlotWidget):
        ov = self._overlays.get(pw)
        if not ov:
            return
        ov['hbar'].setData([], [])
        ov['vbar'].setData([], [])
        ov['htext'].setText("")
        ov['vtext'].setText("")
        if ov.get('ctext') is not None:          # NEW
            ov['ctext'].setText("")
        ov['last_xy'] = None


    def _update_scalebar_for_plot(self, pw: pg.PlotWidget, mx: float, my: float):
        """Draw an L-shaped scalebar with 1–2–5 steps and correct units (x: s/ms, y: µV/mV/V/nV)."""
        ov = self._overlays.get(pw)
        if not ov:
            return

        vb = pw.getViewBox()
        px_dx, px_dy = vb.viewPixelSize()
        px_dx = abs(float(px_dx)) if np.isfinite(px_dx) and px_dx != 0 else 1e-9
        px_dy = abs(float(px_dy)) if np.isfinite(px_dy) and px_dy != 0 else 1e-9

        bar_px = 40.0   # target on-screen length
        off_px = 5.0    # label offset in pixels

        # ----------------- HORIZONTAL (TIME, in seconds) -----------------
        raw_dx = px_dx * bar_px  # seconds

        if raw_dx < 1.0:
            # milliseconds
            x_unit = "ms"
            unit_scale_x = 1e-3   # seconds per ms
            val_x = raw_dx / unit_scale_x
        elif raw_dx < 60.0:
            # seconds
            x_unit = "s"
            unit_scale_x = 1.0
            val_x = raw_dx
        elif raw_dx < 3600.0:
            # minutes
            x_unit = "min"
            unit_scale_x = 60.0   # seconds per minute
            val_x = raw_dx / 60.0
        else:
            # hours
            x_unit = "hr"
            unit_scale_x = 3600.0 # seconds per hour
            val_x = raw_dx / 3600.0

        nice_val_x = nice_125(val_x)
        nice_val_x = max(1.0, nice_val_x)
        dx = nice_val_x * unit_scale_x   # seconds

        offx = px_dx * off_px
        offy = px_dy * off_px

        # Horizontal bar: cursor -> right
        hx0, hy0 = mx, my
        hx1, hy1 = mx + dx, my
        ov['hbar'].setData([hx0, hx1], [hy0, hy1])
        h_label = f"{int(round(nice_val_x))} {x_unit}"
        ov['htext'].setText(h_label)
        ov['htext'].setPos(hx1, hy1 + offy)

        # ----------------- VERTICAL (AMPLITUDE, data in µV) -----------------
        # IMPORTANT: we assume self.data / y are in microvolts (µV)
        raw_dy_uV = px_dy * bar_px  # µV
        v_abs = abs(raw_dy_uV)

        if v_abs >= 1e6:
            # 1e6 µV = 1 V
            y_unit = "V"
            unit_scale_y_uV = 1e6      # µV per V
            val_y = v_abs / 1e6        # in V
        elif v_abs >= 1e3:
            # 1e3 µV = 1 mV
            y_unit = "mV"
            unit_scale_y_uV = 1e3      # µV per mV
            val_y = v_abs / 1e3        # in mV
        elif v_abs >= 1.0:
            # 1 µV and up
            y_unit = "µV"
            unit_scale_y_uV = 1.0      # µV per µV
            val_y = v_abs              # in µV
        else:
            # below 1 µV: show nV
            # 1 nV = 1e-3 µV
            y_unit = "nV"
            unit_scale_y_uV = 1e-3     # µV per nV
            val_y = v_abs / 1e-3       # in nV

        nice_val_y = nice_125(val_y)
        nice_val_y = max(1.0, nice_val_y)       # keep whole-number-ish label
        dy_uV = nice_val_y * unit_scale_y_uV    # µV to draw in data units

        # Vertical bar: cursor -> up
        vx0, vy0 = mx, my
        vx1, vy1 = mx, my + dy_uV   # note: y is in µV, so dy_uV is correct
        ov['vbar'].setData([vx0, vx1], [vy0, vy1])
        v_label = f"{int(round(nice_val_y))} {y_unit}"
        ov['vtext'].setText(v_label)
        ov['vtext'].setPos(vx1 - offx, vy1 + offy * 0.2)





    # -------------------------- DATA FETCH/RENDER --------------------------
    def _compute_needed_range(self):
        if self._shared_plot is None:
            return 0, 1
        vb = self._shared_plot.getViewBox()
        x0, x1 = vb.viewRange()[0]
        x0 = max(0.0, min(x0, self.total_duration))
        x1 = max(0.0, min(x1, self.total_duration))
        if x1 <= x0:
            x1 = min(self.total_duration, x0 + 1e-3)

        width = max(1e-6, x1 - x0)
        margin = 0.1 * width
        a = max(0.0, x0 - margin)
        b = min(self.total_duration, x1 + margin)

        sa = int(np.floor(a * self.sample_rate))
        sb = int(np.ceil(b * self.sample_rate))
        sa = max(0, min(sa, self.n_samples - 1))
        sb = max(sa + 1, min(sb, self.n_samples))

        chunk_sec = max(1.0, min(30.0, width * 1.5))
        chunk = max(1, int(round(chunk_sec * self.sample_rate)))
        sa = (sa // chunk) * chunk
        sb = ((sb + chunk - 1) // chunk) * chunk
        sb = min(sb, self.n_samples)
        return sa, sb

    def _get_segment(self, ch_idx: int, sa: int, sb: int):
        cache = self._caches.get(ch_idx)
        if cache is None:
            cache = _LRUCache()
            self._caches[ch_idx] = cache
        key = (sa, sb)
        arr = cache.get(key)
        if arr is not None:
            return arr
        arr = self.data[ch_idx, sa:sb]
        arr = np.asarray(arr)
        cache.put(key, arr)
        return arr

    def _get_common_mode_segment(self, ch_idx: int, sa: int, sb: int):
        """
        Return data segment for channel ch_idx in [sa:sb),
        after optional common-mode subtraction.
        """
        raw = self._get_segment(ch_idx, sa, sb)
    
        # If CM is disabled or ref channel invalid, just return raw
        if not getattr(self, "cm_enabled", False) or self.cm_ref_index is None:
            return raw
    
        # Make sure indices are sane
        if self.cm_ref_index < 0 or self.cm_ref_index >= self.n_channels:
            return raw
    
        # Reference segment
        ref = self._get_segment(self.cm_ref_index, sa, sb)
    
        # Subtract reference (reference channel will go near-zero itself)
        # The subtraction creates a new array; cached raw arrays are not modified.
        return apply_common_mode(raw, ref)



    def _get_filtered_segment(self, ch_idx: int, sa: int, sb: int):
        """
        Return data segment for channel ch_idx in [sa:sb),
        applying:
            1) common-mode removal
            2) optional Butterworth filter
            3) optional moving-average filter
        """
        # Always start from common-mode corrected segment
        raw_cm = self._get_common_mode_segment(ch_idx, sa, sb)
    
        use_butter = bool(getattr(self, "filter_enabled", False))
        use_ma = bool(getattr(self, "ma_enable", False))
    
        # If no filters at all, just return CM-only data
        if not use_butter and not use_ma:
            return raw_cm
    
        # Per-channel filter cache (for Butterworth + MA combined)
        cache = self._filter_caches.get(ch_idx)
        if cache is None:
            cache = _LRUCache(max_segments=12)
            self._filter_caches[ch_idx] = cache
    
        # IMPORTANT: _filter_signature() should include both Butterworth and MA params
        sig = self._filter_signature()
        key = (sa, sb, sig)
        arr = cache.get(key)
        if arr is not None:
            return arr
    
        y = raw_cm
    
        # ---------- Butterworth stage (if enabled) ----------
        if use_butter:
            fs = float(self.sample_rate) if self.sample_rate > 0 else 1.0
            y = apply_butterworth_filter(
                y=y,
                fs=fs,
                filter_type=self.filter_type,
                f1=self.filter_f1,
                f2=self.filter_f2,
                order=self.filter_order,
            )
    
        # ---------- Moving-average stage (if enabled) ----------
        if use_ma:
            y = apply_moving_average(
                y=y,
                n_points=int(self.ma_points),
                n_passes=int(self.ma_passes),
            )
    
        # Cache final result (CM + Butterworth + MA)
        cache.put(key, y)
        return y


    def _update_peaks_for_channel(self, ch_idx: int, t: np.ndarray, y: np.ndarray, y_min: float, y_max: float):
        """Update vertical peak lines + stats for a single channel in the current window."""
        peak_item = self._peak_items.get(ch_idx)
        if peak_item is None:
            return

        if not self.peaks_enabled:
            peak_item.setData([], [])
            # still store "no peaks" stats
            self._peak_stats[ch_idx] = {
                "count": 0,
                "mean_amp": np.nan,
                "median_amp": np.nan,
                "mean_dur": np.nan,
                "median_dur": np.nan,
                "freq_per_min": 0.0,
            }
            return

        if t.size == 0 or y.size == 0:
            peak_item.setData([], [])
            self._peak_stats[ch_idx] = {
                "count": 0,
                "mean_amp": np.nan,
                "median_amp": np.nan,
                "mean_dur": np.nan,
                "median_dur": np.nan,
                "freq_per_min": 0.0,
            }
            return

        fs = float(self.sample_rate) if self.sample_rate > 0 else 1.0
        y_float = np.asarray(y, dtype=float)
        med = np.median(y_float)

        # ---------- 1) Estimate noise σ on this window ----------
        sigma = estimate_noise_sigma_mad(y_float)


        # ---------- 2) Build prominence & height thresholds ----------
        prom_val = None
        height_val = None

        if getattr(self, "peaks_use_relative", True):
            if self.peaks_k_prom > 0.0:
                prom_val = self.peaks_k_prom * sigma
            if self.peaks_k_height > 0.0:
                height_val = self.peaks_k_height * sigma
        else:
            if self.peaks_prom_abs > 0.0:
                prom_val = self.peaks_prom_abs
            if self.peaks_height_abs > 0.0:
                height_val = self.peaks_height_abs

        kwargs = {}
        if prom_val is not None and prom_val > 0.0:
            kwargs["prominence"] = prom_val
        if height_val is not None and height_val > 0.0:
            kwargs["height"] = height_val

        if self.peaks_min_dist_s > 0.0:
            dist_samples = int(round(self.peaks_min_dist_s * fs))
            if dist_samples > 0:
                kwargs["distance"] = dist_samples

        if self.peaks_min_width_s > 0.0:
            width_samples = int(round(self.peaks_min_width_s * fs))
            if width_samples > 0:
                kwargs["width"] = width_samples

        # ---------- 3) Positive & negative peaks ----------
        try:
            peaks_pos, _props_pos = find_peaks(y_float, **kwargs)
        except Exception:
            peaks_pos = np.array([], dtype=int)

        try:
            peaks_neg, _props_neg = find_peaks(-y_float, **kwargs)
        except Exception:
            peaks_neg = np.array([], dtype=int)

        if peaks_pos.size == 0 and peaks_neg.size == 0:
            peak_item.setData([], [])
            self._peak_stats[ch_idx] = {
                "count": 0,
                "mean_amp": np.nan,
                "median_amp": np.nan,
                "mean_dur": np.nan,
                "median_dur": np.nan,
                "freq_per_min": 0.0,
            }
            return

        # Combine with sign: +1 for positive, -1 for negative
        indices = np.concatenate([peaks_pos, peaks_neg])
        signs = np.concatenate([
            np.ones_like(peaks_pos, dtype=int),
            -np.ones_like(peaks_neg, dtype=int),
        ])

        order = np.argsort(indices)
        indices = indices[order]
        signs = signs[order]

        # ---------- 4) Merge biphasic pairs (keep first extremum) ----------
        keep = np.ones(len(indices), dtype=bool)
        window_s = getattr(self, "peaks_biphasic_window_s", 0.0)
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
                        # biphasic event: keep first extremum only
                        keep[j] = False

        final_indices = indices[keep]
        if final_indices.size == 0:
            peak_item.setData([], [])
            self._peak_stats[ch_idx] = {
                "count": 0,
                "mean_amp": np.nan,
                "median_amp": np.nan,
                "mean_dur": np.nan,
                "median_dur": np.nan,
                "freq_per_min": 0.0,
            }
            return

        # final_indices: indices within current segment
        # Convert to global sample indices
        sa = getattr(self, "_current_sa", 0)
        global_indices = final_indices + int(sa)
        self._peak_indices[ch_idx] = global_indices


        # ---------- 5) Amplitude & rough duration ----------
        amps = np.abs(y_float[final_indices])

        durations = []
        for idx in final_indices:
            # simple FWHM-like duration around median baseline
            amp = np.abs(y_float[idx] - med)
            if amp <= 0:
                durations.append(0.0)
                continue
            thresh = 0.5 * amp

            # walk left
            left = idx
            while left > 0 and np.abs(y_float[left] - med) >= thresh:
                left -= 1

            # walk right
            right = idx
            n = len(y_float)
            while right < n - 1 and np.abs(y_float[right] - med) >= thresh:
                right += 1

            width_samples = max(1, right - left)
            durations.append(width_samples / fs)

        durations = np.asarray(durations, dtype=float)

        count = final_indices.size
        window_duration = max(1e-6, t[-1] - t[0])
        freq_per_min = count / (window_duration / 60.0)

        self._peak_stats[ch_idx] = {
            "count": int(count),
            "mean_amp": float(np.mean(amps)) if count > 0 else np.nan,
            "median_amp": float(np.median(amps)) if count > 0 else np.nan,
            "mean_dur": float(np.mean(durations)) if durations.size > 0 else np.nan,
            "median_dur": float(np.median(durations)) if durations.size > 0 else np.nan,
            "freq_per_min": float(freq_per_min),
        }

        # ---------- 6) Draw vertical lines ----------
        t_peaks = t[final_indices]
        x_vals = []
        y_vals = []
        for tp in t_peaks:
            x_vals.extend([tp, tp, np.nan])
            y_vals.extend([y_min, y_max, np.nan])

        peak_item.setData(x_vals, y_vals, _callSync='off')

    def _format_cursor_timestamp(self, seconds_from_start: float) -> str:
        """
        Format cursor time as dd/mm/yy hh:mm:ss.mmm using the file start time
        if available. If start_datetime is None, fall back to hh:mm:ss.
        """
        s = max(0.0, float(seconds_from_start))

        if self.start_datetime is None:
            # fallback: just hh:mm:ss
            whole = int(s)
            h = whole // 3600
            m = (whole % 3600) // 60
            sec = whole % 60
            return f"{h:02d}:{m:02d}:{sec:02d}"

        dt = self.start_datetime + timedelta(seconds=s)
        ms = dt.microsecond // 1000
        return dt.strftime("%d/%m/%y %H:%M:%S") + f".{ms:03d}"


    def _update_peaks_widget_states(self):
        """Enable/disable relative vs absolute controls based on mode."""
        use_rel = self.peaks_use_relative
        if self.peaks_k_prom_spin is not None:
            self.peaks_k_prom_spin.setEnabled(use_rel)
        if self.peaks_k_height_spin is not None:
            self.peaks_k_height_spin.setEnabled(use_rel)
        if self.peaks_prominence_spin is not None:
            self.peaks_prominence_spin.setEnabled(not use_rel)
        if getattr(self, "peaks_height_abs_spin", None) is not None:
            self.peaks_height_abs_spin.setEnabled(not use_rel)

    def _update_peak_stats_table(self):
        """Refresh the 'Peak stats' tab based on current window and selected channels."""
        if self.stats_table is None:
            return

        # Show stats only for selected channels, in display order
        rows = [idx for idx in self.display_order if idx in self.selected_channels]
        self.stats_table.setRowCount(len(rows))

        for row_idx, ch_idx in enumerate(rows):
            name = self.channel_names[ch_idx]
            stats = self._peak_stats.get(ch_idx, {})

            def _val(key, fmt="{:.3g}"):
                v = stats.get(key, np.nan)
                if v is None or not np.isfinite(v):
                    return ""
                try:
                    return fmt.format(v)
                except Exception:
                    return str(v)

            items = [
                QTableWidgetItem(name),
                QTableWidgetItem(str(stats.get("count", 0))),
                QTableWidgetItem(_val("mean_amp")),
                QTableWidgetItem(_val("median_amp")),
                QTableWidgetItem(_val("mean_dur")),
                QTableWidgetItem(_val("median_dur")),
                QTableWidgetItem(_val("freq_per_min")),
            ]

            for col, item in enumerate(items):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.stats_table.setItem(row_idx, col, item)


    def _fetch_and_render(self):
        if self.data is None or not self.selected_channels or self._shared_plot is None:
            return

        self._peak_stats.clear()

        sa, sb = self._compute_needed_range()
        self._current_sa = sa   # <--- ADD THIS
        t = np.arange(sa, sb) / self.sample_rate

        vb = self._shared_plot.getViewBox()
        x0, x1 = vb.viewRange()[0]
        self.time_start_label.setText(f"Start: {format_time_delta(x0)}")
        self.time_end_label.setText(f"End: {format_time_delta(x1)}")

        _x_range, y_range = vb.viewRange()
        y_min, y_max = float(y_range[0]), float(y_range[1])
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min, y_max = -1.0, 1.0

        for ch_idx in sorted(self.selected_channels):
            # 1) Get the base signal (with CM + filters if you have a wrapper)
            if hasattr(self, "_get_filtered_segment"):
                base_y = self._get_filtered_segment(ch_idx, sa, sb)
            else:
                base_y = self._get_segment(ch_idx, sa, sb)

            # 2) Optionally convert to derivative (dy/dt), keeping same length
            if self.show_derivative:
                # np.gradient keeps same length; multiply by fs to get per-second derivative
                y = np.gradient(base_y) * self.sample_rate
            else:
                y = base_y

            curve = self._curves.get(ch_idx)
            if curve is None:
                continue

            curve.setData(t, y, _callSync='off')
            self._update_peaks_for_channel(ch_idx, t, y, y_min, y_max)

        self._update_peak_stats_table()



    # ---------------------- Cursor readout + active-plot overlay control ------------------------
    def _on_mouse_moved(self, pos, pw: pg.PlotWidget, ch_idx: int):
        # Hide overlays on all OTHER plots
        for other_pw in list(self._overlays.keys()):
            if other_pw is not pw:
                self._hide_overlay(other_pw)
    
        vb = pw.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self._hide_overlay(pw)
            return
    
        mousePoint = vb.mapSceneToView(pos)
        x = float(mousePoint.x())
        y = float(mousePoint.y())
        x = max(0.0, min(x, self.total_duration))
    
        ts_str = self._format_cursor_timestamp(x)
    
        # Bottom label (optional)
        self.cursor_label.setText(
            f"Cursor — t: {ts_str}  |  y: {y:.6g}"
        )
    
        self._ensure_overlay(pw)
        ov = self._overlays[pw]
        ov['last_xy'] = (x, y)
        self._update_scalebar_for_plot(pw, x, y)
    
        # ---- Cursor text inside plot ----
        ctext = ov.get('ctext')
        if ctext is not None:
            # assuming data are in µV
            if self.show_derivative:
                y_str = f"{y:.2f} µV/s"
            else:
                y_str = f"{y:.2f} µV"
            
            label = f"{ts_str}\n{y_str}"

    
            # convert 10 px vertical offset into data units
            _, px_dy = vb.viewPixelSize()
            px_dy = float(px_dy) if np.isfinite(px_dy) and px_dy != 0 else 1e-9
            offset_data_y = 20.0 * px_dy   # 10 px downward
            
            # ---- Cursor text inside plot ----
            ctext = ov.get('ctext')
            if ctext is not None:
                # Format label
                ts_str = self._format_cursor_timestamp(x)   # real time dd/mm/yy hh:mm:ss.mmm
                y_str = f"{y:.2f} µV"                       # assuming data are in µV
                label = f"{ts_str}\n{y_str}"
            
                # Get pixel → data scale
                px_dx, px_dy = vb.viewPixelSize()
                px_dx = float(px_dx) if np.isfinite(px_dx) and px_dx != 0 else 1e-9
                px_dy = float(px_dy) if np.isfinite(px_dy) and px_dy != 0 else 1e-9
            
                # Vertical offset: keep it nicely below the cursor
                offset_data_y = 20.0 * px_dy   # ~20 px down
            
                # Determine horizontal anchoring based on proximity to edges
                (x_min, x_max), _ = vb.viewRange()
                margin_px = 60.0
                margin_data = margin_px * px_dx
            
                left_thresh  = x_min + margin_data
                right_thresh = x_max - margin_data
            
                if x <= left_thresh:
                    # Very close to left edge: left-align text, nudge right
                    anchor_x = 0.5
                    pos_x = x + margin_data
                elif x >= right_thresh:
                    # Very close to right edge: right-align text, nudge left
                    anchor_x = 0.5
                    pos_x = x - margin_data
                else:
                    # In the middle: center under cursor
                    anchor_x = 0.5
                    pos_x = x
            
                # Top of text box at (pos_x, y - offset_data_y)
                ctext.setAnchor((anchor_x, 0.0))   # (horizontal, vertical) in [0,1]
                ctext.setText(label)
                ctext.setPos(pos_x, y - offset_data_y)





    # ---------------------- Y-range estimation --------------------
    def _estimate_global_y_limits(self, channels, target_samples_per_ch: int = 20000):
        if not channels or self.data is None or self.n_samples == 0:
            return (-1.0, 1.0)

        y_min = np.inf
        y_max = -np.inf

        for ch_idx in channels:
            stride = max(1, self.n_samples // max(1, target_samples_per_ch))
            arr = np.asarray(self.data[ch_idx, 0:self.n_samples:stride])
            if arr.size == 0:
                continue
            ch_min = float(np.nanmin(arr))
            ch_max = float(np.nanmax(arr))
            y_min = min(y_min, ch_min)
            y_max = max(y_max, ch_max)

        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            return (-1.0, 1.0)

        pad = 0.05 * (y_max - y_min) if (y_max > y_min) else 0.05
        return (y_min - pad, y_max + pad)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec_())
    
