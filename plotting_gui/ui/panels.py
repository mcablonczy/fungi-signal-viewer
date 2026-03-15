# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:57:03 2025

@author: markablonczy
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QDateTimeEdit, QDoubleSpinBox, QSpinBox, QComboBox, QTableWidget, QTabWidget,
    QScrollArea, QGridLayout,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout


def build_controls_panel(viewer: object) -> QWidget:
    """
    Build the right-hand 'Controls' panel for the HDF5Viewer and attach
    all created widgets back onto the viewer instance.

    This is a direct transliteration of the original control_layout block
    from viewer.init_ui, using 'viewer' instead of 'self'.
    """
    controls_widget = QWidget()
    control_layout = QVBoxLayout(controls_widget)

    # --- File & selection controls ---
    viewer.load_button = QPushButton("Load File")
    viewer.load_button.clicked.connect(viewer.load_file)
    viewer.load_button.setMinimumWidth(110)
    control_layout.addWidget(viewer.load_button)

    viewer.select_all_checkbox = QCheckBox("Select All")
    viewer.select_all_checkbox.stateChanged.connect(viewer.toggle_select_all)
    control_layout.addWidget(viewer.select_all_checkbox)

    viewer.unselect_all_button = QPushButton("Unselect All")
    viewer.unselect_all_button.clicked.connect(viewer.unselect_all)
    control_layout.addWidget(viewer.unselect_all_button)

    # --- Real-time window selectors (dd/mm/yy hh:mm:ss) ---
    viewer.win_start_label = QLabel("Window start (dd/mm/yy hh:mm:ss):")
    viewer.win_start_dtedit = QDateTimeEdit()
    viewer.win_start_dtedit.setDisplayFormat("dd/MM/yy HH:mm:ss")
    viewer.win_start_dtedit.setCalendarPopup(True)

    viewer.win_end_label = QLabel("Window end (dd/mm/yy hh:mm:ss):")
    viewer.win_end_dtedit = QDateTimeEdit()
    viewer.win_end_dtedit.setDisplayFormat("dd/MM/yy HH:mm:ss")
    viewer.win_end_dtedit.setCalendarPopup(True)

    viewer.win_apply_button = QPushButton("Go to window")
    viewer.win_apply_button.clicked.connect(viewer._on_apply_window_times)

    control_layout.addWidget(viewer.win_start_label)
    control_layout.addWidget(viewer.win_start_dtedit)
    control_layout.addWidget(viewer.win_end_label)
    control_layout.addWidget(viewer.win_end_dtedit)
    control_layout.addWidget(viewer.win_apply_button)

    # Plot derivative instead of raw/filtered signal  (only ONCE)
    viewer.derivative_checkbox = QCheckBox("Plot derivative instead of signal")
    viewer.derivative_checkbox.stateChanged.connect(viewer._on_derivative_toggled)
    control_layout.addWidget(viewer.derivative_checkbox)

    # ---------- Common-mode removal controls ----------
    cm_title = QLabel("Common-mode removal")
    cm_title.setStyleSheet("font-weight: bold;")
    control_layout.addWidget(cm_title)

    viewer.cm_enable_checkbox = QCheckBox("Enable common-mode")
    viewer.cm_enable_checkbox.stateChanged.connect(viewer._on_cm_params_changed)
    control_layout.addWidget(viewer.cm_enable_checkbox)

    cm_mode_row = QHBoxLayout()
    cm_mode_row.addWidget(QLabel("Mode:"))
    viewer.cm_mode_combo = QComboBox()
    viewer.cm_mode_combo.addItem("Single channel", userData="single")
    viewer.cm_mode_combo.addItem("Robust CAR (median)", userData="robust_car_median")
    viewer.cm_mode_combo.currentIndexChanged.connect(viewer._on_cm_params_changed)
    cm_mode_row.addWidget(viewer.cm_mode_combo)
    cm_mode_row.addStretch()
    control_layout.addLayout(cm_mode_row)

    cm_row = QHBoxLayout()
    cm_row.addWidget(QLabel("Reference:"))
    viewer.cm_ref_combo = QComboBox()
    viewer.cm_ref_combo.currentIndexChanged.connect(viewer._on_cm_params_changed)
    viewer.cm_ref_combo.setEnabled(False)  # enabled once file is loaded
    cm_row.addWidget(viewer.cm_ref_combo)
    cm_row.addStretch()
    control_layout.addLayout(cm_row)
    viewer._update_cm_widget_states()

    # ---------- Butterworth filter controls ----------
    filter_title = QLabel("Butterworth filter")
    filter_title.setStyleSheet("font-weight: bold;")
    control_layout.addWidget(filter_title)

    viewer.filter_enable_checkbox = QCheckBox("Enable filter")
    viewer.filter_enable_checkbox.stateChanged.connect(viewer._on_filter_params_changed)
    control_layout.addWidget(viewer.filter_enable_checkbox)

    type_row = QHBoxLayout()
    type_row.addWidget(QLabel("Type:"))
    viewer.filter_type_combo = QComboBox()
    viewer.filter_type_combo.addItems(["Lowpass", "Highpass", "Bandpass"])
    viewer.filter_type_combo.currentIndexChanged.connect(viewer._on_filter_type_changed)
    type_row.addWidget(viewer.filter_type_combo)
    type_row.addStretch()
    control_layout.addLayout(type_row)

    f1_row = QHBoxLayout()
    f1_row.addWidget(QLabel("f1 (Hz):"))
    viewer.filter_f1_spin = QDoubleSpinBox()
    viewer.filter_f1_spin.setRange(0.001, 1e5)
    viewer.filter_f1_spin.setDecimals(3)
    viewer.filter_f1_spin.setSingleStep(0.5)
    viewer.filter_f1_spin.setValue(viewer.filter_f1)
    viewer.filter_f1_spin.valueChanged.connect(viewer._on_filter_params_changed)
    f1_row.addWidget(viewer.filter_f1_spin)
    control_layout.addLayout(f1_row)

    f2_row = QHBoxLayout()
    f2_row.addWidget(QLabel("f2 (Hz):"))
    viewer.filter_f2_spin = QDoubleSpinBox()
    viewer.filter_f2_spin.setRange(0.001, 1e5)
    viewer.filter_f2_spin.setDecimals(3)
    viewer.filter_f2_spin.setSingleStep(0.5)
    viewer.filter_f2_spin.setValue(viewer.filter_f2)
    viewer.filter_f2_spin.valueChanged.connect(viewer._on_filter_params_changed)
    f2_row.addWidget(viewer.filter_f2_spin)
    control_layout.addLayout(f2_row)

    order_row = QHBoxLayout()
    order_row.addWidget(QLabel("Order:"))
    viewer.filter_order_spin = QSpinBox()
    viewer.filter_order_spin.setRange(1, 10)
    viewer.filter_order_spin.setValue(viewer.filter_order)
    viewer.filter_order_spin.valueChanged.connect(viewer._on_filter_params_changed)
    order_row.addWidget(viewer.filter_order_spin)
    control_layout.addLayout(order_row)

    # Let the viewer sync widget enable/disable states
    viewer._update_filter_widget_states()

    # ---------- Median filter controls ----------
    median_title = QLabel("Median filter")
    median_title.setStyleSheet("font-weight: bold;")
    control_layout.addWidget(median_title)

    viewer.median_filter_checkbox = QCheckBox("Apply 3-point median filter")
    viewer.median_filter_checkbox.setChecked(viewer.median_filter_enabled)
    viewer.median_filter_checkbox.stateChanged.connect(viewer._on_median_filter_toggled)
    control_layout.addWidget(viewer.median_filter_checkbox)

    # ---------- Moving average filter controls ----------
    ma_title = QLabel("Moving average filter")
    ma_title.setStyleSheet("font-weight: bold;")
    control_layout.addWidget(ma_title)

    viewer.ma_enable_checkbox = QCheckBox("Enable moving average")
    viewer.ma_enable_checkbox.stateChanged.connect(viewer._on_ma_params_changed)
    control_layout.addWidget(viewer.ma_enable_checkbox)

    ma_points_row = QHBoxLayout()
    ma_points_row.addWidget(QLabel("Points:"))
    viewer.ma_points_spin = QSpinBox()
    viewer.ma_points_spin.setRange(1, 10000)
    viewer.ma_points_spin.setValue(viewer.ma_points)
    viewer.ma_points_spin.valueChanged.connect(viewer._on_ma_params_changed)
    ma_points_row.addWidget(viewer.ma_points_spin)
    control_layout.addLayout(ma_points_row)

    ma_passes_row = QHBoxLayout()
    ma_passes_row.addWidget(QLabel("Passes:"))
    viewer.ma_passes_spin = QSpinBox()
    viewer.ma_passes_spin.setRange(1, 10)
    viewer.ma_passes_spin.setValue(viewer.ma_passes)
    viewer.ma_passes_spin.valueChanged.connect(viewer._on_ma_params_changed)
    ma_passes_row.addWidget(viewer.ma_passes_spin)
    control_layout.addLayout(ma_passes_row)

    # ---------- Peak detection controls ----------
    peaks_title = QLabel("Peak detection")
    peaks_title.setStyleSheet("font-weight: bold;")
    control_layout.addWidget(peaks_title)

    viewer.peaks_enable_checkbox = QCheckBox("Enable peak detection")
    viewer.peaks_enable_checkbox.stateChanged.connect(viewer._on_peaks_params_changed)
    control_layout.addWidget(viewer.peaks_enable_checkbox)

    # Relative vs absolute thresholds
    viewer.peaks_use_relative_checkbox = QCheckBox("Use relative (Ïƒ-based) thresholds")
    viewer.peaks_use_relative_checkbox.setChecked(viewer.peaks_use_relative)
    viewer.peaks_use_relative_checkbox.stateChanged.connect(viewer._on_peaks_params_changed)
    control_layout.addWidget(viewer.peaks_use_relative_checkbox)

    kprom_row = QHBoxLayout()
    kprom_row.addWidget(QLabel("k_prom (Ã—Ïƒ):"))
    viewer.peaks_k_prom_spin = QDoubleSpinBox()
    viewer.peaks_k_prom_spin.setRange(0.0, 1e3)
    viewer.peaks_k_prom_spin.setDecimals(2)
    viewer.peaks_k_prom_spin.setSingleStep(0.5)
    viewer.peaks_k_prom_spin.setValue(viewer.peaks_k_prom)
    viewer.peaks_k_prom_spin.valueChanged.connect(viewer._on_peaks_params_changed)
    kprom_row.addWidget(viewer.peaks_k_prom_spin)
    control_layout.addLayout(kprom_row)

    kheight_row = QHBoxLayout()
    kheight_row.addWidget(QLabel("k_height (Ã—Ïƒ):"))
    viewer.peaks_k_height_spin = QDoubleSpinBox()
    viewer.peaks_k_height_spin.setRange(0.0, 1e3)
    viewer.peaks_k_height_spin.setDecimals(2)
    viewer.peaks_k_height_spin.setSingleStep(0.5)
    viewer.peaks_k_height_spin.setValue(viewer.peaks_k_height)
    viewer.peaks_k_height_spin.valueChanged.connect(viewer._on_peaks_params_changed)
    kheight_row.addWidget(viewer.peaks_k_height_spin)
    control_layout.addLayout(kheight_row)

    prom_row = QHBoxLayout()
    prom_row.addWidget(QLabel("Prominence (abs units):"))
    viewer.peaks_prominence_spin = QDoubleSpinBox()
    viewer.peaks_prominence_spin.setRange(0.0, 1e9)
    viewer.peaks_prominence_spin.setDecimals(6)
    viewer.peaks_prominence_spin.setSingleStep(0.1)
    viewer.peaks_prominence_spin.setValue(50.0)
    viewer.peaks_prominence_spin.valueChanged.connect(viewer._on_peaks_params_changed)
    prom_row.addWidget(viewer.peaks_prominence_spin)
    control_layout.addLayout(prom_row)

    height_row = QHBoxLayout()
    height_row.addWidget(QLabel("Min height (abs units):"))
    viewer.peaks_height_abs_spin = QDoubleSpinBox()
    viewer.peaks_height_abs_spin.setRange(0.0, 1e9)
    viewer.peaks_height_abs_spin.setDecimals(6)
    viewer.peaks_height_abs_spin.setSingleStep(0.1)
    viewer.peaks_height_abs_spin.setValue(50.0)
    viewer.peaks_height_abs_spin.valueChanged.connect(viewer._on_peaks_params_changed)
    height_row.addWidget(viewer.peaks_height_abs_spin)
    control_layout.addLayout(height_row)

    dist_row = QHBoxLayout()
    dist_row.addWidget(QLabel("Min distance (s):"))
    viewer.peaks_min_dist_spin = QDoubleSpinBox()
    viewer.peaks_min_dist_spin.setRange(0.0, 1e6)
    viewer.peaks_min_dist_spin.setDecimals(3)
    viewer.peaks_min_dist_spin.setSingleStep(0.1)
    viewer.peaks_min_dist_spin.setValue(0.0)
    viewer.peaks_min_dist_spin.valueChanged.connect(viewer._on_peaks_params_changed)
    dist_row.addWidget(viewer.peaks_min_dist_spin)
    control_layout.addLayout(dist_row)

    width_row = QHBoxLayout()
    width_row.addWidget(QLabel("Min width (s):"))
    viewer.peaks_min_width_spin = QDoubleSpinBox()
    viewer.peaks_min_width_spin.setRange(0.0, 1e6)
    viewer.peaks_min_width_spin.setDecimals(3)
    viewer.peaks_min_width_spin.setSingleStep(0.1)
    viewer.peaks_min_width_spin.setValue(0.0)
    viewer.peaks_min_width_spin.valueChanged.connect(viewer._on_peaks_params_changed)
    width_row.addWidget(viewer.peaks_min_width_spin)
    control_layout.addLayout(width_row)

    bip_row = QHBoxLayout()
    bip_row.addWidget(QLabel("Biphasic merge (s):"))
    viewer.peaks_biphasic_window_spin = QDoubleSpinBox()
    viewer.peaks_biphasic_window_spin.setRange(0.0, 1e6)
    viewer.peaks_biphasic_window_spin.setDecimals(3)
    viewer.peaks_biphasic_window_spin.setSingleStep(0.05)
    viewer.peaks_biphasic_window_spin.setValue(viewer.peaks_biphasic_window_s)
    viewer.peaks_biphasic_window_spin.valueChanged.connect(viewer._on_peaks_params_changed)
    bip_row.addWidget(viewer.peaks_biphasic_window_spin)
    control_layout.addLayout(bip_row)

    # Done
    return controls_widget

def build_peak_stats_panel(viewer: object) -> QWidget:
    """
    Build the 'Peak stats' tab panel and attach the relevant widgets
    (stats table + buttons) back to the viewer instance.
    """
    stats_widget = QWidget()
    stats_layout = QVBoxLayout(stats_widget)

    # --- Table ---
    viewer.stats_table = QTableWidget()
    viewer.stats_table.setColumnCount(7)
    viewer.stats_table.setHorizontalHeaderLabels([
        "Channel", "Count", "Mean amp", "Median amp",
        "Mean dur (s)", "Median dur (s)", "Freq (peaks/min)"
    ])
    viewer.stats_table.horizontalHeader().setStretchLastSection(True)
    viewer.stats_table.verticalHeader().setVisible(False)
    viewer.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
    viewer.stats_table.setSelectionBehavior(QTableWidget.SelectRows)
    viewer.stats_table.setSelectionMode(QTableWidget.SingleSelection)

    # --- Buttons ---
    viewer.classify_button = QPushButton("Classify peaks")
    viewer.classify_button.clicked.connect(viewer._on_classify_peaks_clicked)
    stats_layout.addWidget(viewer.classify_button)

    viewer.extract_windows_button = QPushButton("Extract 4s peak windows")
    viewer.extract_windows_button.clicked.connect(
        viewer._on_extract_peak_windows_clicked
    )
    stats_layout.addWidget(viewer.extract_windows_button)

    viewer.save_peaks_button = QPushButton("Save peaks")
    viewer.save_peaks_button.clicked.connect(viewer._on_save_peaks_clicked)
    stats_layout.addWidget(viewer.save_peaks_button)

    # Table last, fills remaining space
    stats_layout.addWidget(viewer.stats_table)

    return stats_widget

def build_right_tabs(viewer: object) -> QTabWidget:
    """
    Build the right-side QTabWidget containing:
      - Controls tab
      - Peak stats tab
    and attach it to the viewer.
    """
    controls_widget = build_controls_panel(viewer)
    stats_widget = build_peak_stats_panel(viewer)

    tabs = QTabWidget()
    tabs.addTab(controls_widget, "Controls")
    tabs.addTab(stats_widget, "Peak stats")

    # Right-side width constraint (same as you had in viewer.init_ui)
    tabs.setMinimumWidth(270)
    tabs.setMaximumWidth(270)

    return tabs

def build_left_panel(viewer: object) -> QWidget:
    """
    Build the left side of the UI:
      - file label
      - channel checkbox scroll panel
      - plot grid scroll area
      - time range labels
      - cursor readout

    Attaches created widgets as attributes on `viewer`.
    """
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)

    # --- Top: file label ---
    viewer.file_label = QLabel("No file loaded")
    left_layout.addWidget(viewer.file_label)

    # --- Middle row: [channel scroll panel] | [plot grid scroll] ---
    mid_row = QHBoxLayout()

    # Channel checkbox scroll panel
    viewer.checkbox_group = QVBoxLayout()
    viewer.checkbox_widget = QWidget()
    viewer.checkbox_widget.setLayout(viewer.checkbox_group)

    viewer.scroll_area = QScrollArea()
    viewer.scroll_area.setWidgetResizable(True)
    viewer.scroll_area.setWidget(viewer.checkbox_widget)
    viewer.scroll_area.setMinimumWidth(90)
    viewer.scroll_area.setMaximumWidth(140)

    mid_row.addWidget(viewer.scroll_area, stretch=0)

    # Plot area: scrollable grid of subplots (2 columns)
    viewer.grid_widget = QWidget()
    viewer.grid_layout = QGridLayout(viewer.grid_widget)
    viewer.grid_layout.setContentsMargins(0, 0, 0, 0)
    viewer.grid_layout.setHorizontalSpacing(4)
    viewer.grid_layout.setVerticalSpacing(4)
    viewer.grid_layout.setColumnStretch(0, 1)
    viewer.grid_layout.setColumnStretch(1, 1)

    viewer.plot_scroll_area = QScrollArea()
    viewer.plot_scroll_area.setWidgetResizable(True)
    viewer.plot_scroll_area.setWidget(viewer.grid_widget)
    viewer.plot_scroll_area.setMinimumHeight(100)
    viewer.plot_scroll_area.setMaximumHeight(1200)

    mid_row.addWidget(viewer.plot_scroll_area, stretch=1)

    left_layout.addLayout(mid_row)

    # --- Bottom: absolute time range labels ---
    time_label_layout = QHBoxLayout()
    viewer.time_start_label = QLabel("Start: 0.00 s")
    viewer.time_end_label = QLabel("End: 0.00 s")
    f = QFont("Arial", 10)
    viewer.time_start_label.setFont(f)
    viewer.time_end_label.setFont(f)
    time_label_layout.addWidget(viewer.time_start_label)
    time_label_layout.addStretch()
    time_label_layout.addWidget(viewer.time_end_label)
    left_layout.addLayout(time_label_layout)

    # --- Cursor readout (bottom) ---
    viewer.cursor_label = QLabel("Cursor â€” t: â€“, y: â€“")
    viewer.cursor_label.setFont(QFont("Arial", 10))
    left_layout.addWidget(viewer.cursor_label)

    return left_widget

def build_main_layout(viewer: object) -> QHBoxLayout:
    """
    Build the top-level UI layout:
      [ left panel ] --- [ right tabs ]
    """
    layout = QHBoxLayout()

    left_widget = build_left_panel(viewer)
    right_tabs = build_right_tabs(viewer)

    layout.addWidget(left_widget, stretch=5)
    layout.addWidget(right_tabs, stretch=0)

    return layout
