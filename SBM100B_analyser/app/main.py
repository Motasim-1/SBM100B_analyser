from __future__ import annotations

import csv
import queue
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import sounddevice as sd

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from core.audio_io import list_input_devices
from core.analysis import (
    analyze_array,
    compute_fft,
    load_wav,
    save_wav,
    analyze_thd,
    analyze_thdn_sinad,
    analyze_sweep_stft,
    analyze_sweep_known,
    analyze_external_sweep_response,
    rms_dbfs,
)
from core.calibration import load_calibration, save_calibration, dbfs_to_spl

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "recordings"
PLOTS = ROOT / "results" / "plots"
TABLES = ROOT / "results" / "tables"
CONFIG = ROOT / "config" / "calibration.json"
for d in (RECORDINGS, PLOTS, TABLES, CONFIG.parent):
    d.mkdir(parents=True, exist_ok=True)


class PlotWidget(QWidget):
    """Dark, dashboard-style matplotlib plot used by the main UI and plot windows."""

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.figure = Figure(figsize=(7.0, 3.6), facecolor="#0D141C")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(220)
        self.canvas.setStyleSheet("background: #0D141C;")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def _style_axis(self, ax):
        ax.set_facecolor("#111A23")
        ax.tick_params(colors="#C9D4DF", labelsize=10)
        ax.xaxis.label.set_color("#DDE7F0")
        ax.yaxis.label.set_color("#DDE7F0")
        ax.title.set_color("#F3F7FB")
        ax.title.set_fontsize(12)
        ax.title.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_color("#385064")
        ax.grid(True, which="both", color="#304457", alpha=0.55, linewidth=0.7)

    def waveform(self, data: np.ndarray, sr: int):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if len(data):
            step = max(1, len(data) // 16000)
            ax.plot(
                np.arange(len(data))[::step] / sr,
                data[::step],
                color="#2EA8FF",
                linewidth=1.05,
            )
        ax.set_title("Live Waveform")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        self._style_axis(ax)
        self.figure.tight_layout(pad=1.55)
        self.canvas.draw()

    def spectrum(self, data: np.ndarray, sr: int):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if len(data):
            max_samples = min(len(data), int(sr * 2))
            f, s = compute_fft(data[-max_samples:], sr)
            valid = (f >= 20) & (f <= 20000)
            ax.plot(f[valid], s[valid], color="#2EA8FF", linewidth=1.05)
        ax.set_title("FFT / Spectrum")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_xscale("log")
        ax.set_xlim(20, 20000)
        ax.set_ylim(-120, 60)
        self._style_axis(ax)
        self.figure.tight_layout(pad=1.55)
        self.canvas.draw()

    def spectrogram(self, data: np.ndarray, sr: int):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if len(data):
            max_samples = min(len(data), int(sr * 22))
            x = data[-max_samples:]
            nfft = 2048
            noverlap = 1536
            _, _, _, im = ax.specgram(
                x, NFFT=nfft, Fs=sr, noverlap=noverlap, scale="dB", cmap="inferno"
            )
            ax.set_yscale("log")
            ax.set_ylim(20, min(20000, sr / 2))
        ax.set_title("Live Spectrogram")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        self._style_axis(ax)
        self.figure.tight_layout(pad=1.55)
        self.canvas.draw()

    def xy(self, x, y, title: str, xlabel: str, ylabel: str, log_x: bool = False):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, marker="o", color="#2EA8FF", linewidth=1.8, markersize=5.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale("log")
        self._style_axis(ax)
        self.figure.tight_layout(pad=1.55)
        self.canvas.draw()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(
            path, dpi=180, facecolor=self.figure.get_facecolor(), bbox_inches="tight"
        )


class PlotDialog(QDialog):
    """Floating analysis window with zoom, pan, save and export toolbar."""

    def __init__(self, source_figure: Figure, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 680)
        self.figure = Figure(figsize=(10.0, 6.4), facecolor="#0D141C")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background: #0D141C;")
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(
            "QToolBar { background: #111820; border: 0; padding: 4px; } QToolButton { color: #EAF0F6; padding: 5px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setStyleSheet("QDialog { background: #0D141C; color: #EAF0F6; }")
        self.copy_from(source_figure)

    def _style_axis(self, ax):
        ax.set_facecolor("#111A23")
        ax.tick_params(colors="#C9D4DF", labelsize=10)
        ax.xaxis.label.set_color("#DDE7F0")
        ax.yaxis.label.set_color("#DDE7F0")
        ax.title.set_color("#F3F7FB")
        ax.title.set_fontsize(12)
        ax.title.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_color("#385064")
        ax.grid(True, which="both", color="#304457", alpha=0.55, linewidth=0.7)

    def copy_from(self, source_figure: Figure):
        self.figure.clear()
        src_axes = source_figure.axes
        if not src_axes:
            self.canvas.draw()
            return
        for i, src in enumerate(src_axes, start=1):
            dst = self.figure.add_subplot(1, len(src_axes), i)
            # Images from spectrograms.
            for img in src.images:
                try:
                    dst.imshow(
                        img.get_array(),
                        extent=img.get_extent(),
                        origin=img.origin,
                        aspect="auto",
                        cmap=img.get_cmap(),
                        norm=img.norm,
                    )
                except Exception:
                    pass
            # Lines from normal plots.
            for line in src.get_lines():
                dst.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    marker=line.get_marker(),
                    linestyle=line.get_linestyle(),
                    linewidth=max(line.get_linewidth(), 1.2),
                    color=line.get_color(),
                )
            dst.set_title(src.get_title())
            dst.set_xlabel(src.get_xlabel())
            dst.set_ylabel(src.get_ylabel())
            dst.set_xscale(src.get_xscale())
            dst.set_yscale(src.get_yscale())
            try:
                dst.set_xlim(src.get_xlim())
                dst.set_ylim(src.get_ylim())
            except Exception:
                pass
            self._style_axis(dst)
            if src.get_legend() is not None:
                dst.legend()
        self.figure.tight_layout(pad=1.4)
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SBM100B Analyzer")
        self.resize(1360, 860)
        self.setMinimumSize(1100, 720)

        self.offset_db = load_calibration(CONFIG).get("offset_db")
        self.stream = None
        self.audio_queue = queue.Queue()
        self.live_buffer = np.array([], dtype=np.float64)
        self.live_samplerate = 96000
        self.measurements: list[dict[str, Any]] = []
        self.hold_info = False

        self.timer = QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_live)

        self.build_ui()
        self.apply_theme()

    def apply_theme(self):
        """dark dashboard UI. Measurement logic is unchanged."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0B1118; color: #EAF0F6; font-size: 12px; font-family: Arial, Helvetica, sans-serif; }
            QFrame#sidebar { background: #081018; border-right: 1px solid #223140; }
            QLabel#appTitle { color: #FFFFFF; font-size: 16px; font-weight: 800; }
            QLabel#appSubTitle { color: #8EA4B7; font-size: 10px; }
            QLabel#sectionLabel { color: #7890A4; font-size: 10px; letter-spacing: 1px; }
            QLabel#actionHint { color: #AFC3D6; font-size: 11px; padding-left: 6px; }
            QFrame#topHeaderCard { background: #10263B; border: 1px solid #2EA8FF; border-radius: 12px; }
            QLabel#uiVersionLabel { background: #1769D1; color: white; border-radius: 8px; padding: 7px 12px; font-size: 13px; font-weight: 800; letter-spacing: 0.5px; }
            QLabel#uiWorkflowLabel { color: #EAF0F6; font-size: 14px; font-weight: 700; }
            QLabel#metricValue { color: #FFFFFF; font-size: 22px; font-weight: 800; }
            QLabel#metricSmall { color: #7DE28A; font-size: 11px; }
            QGroupBox { border: 1px solid #223140; border-radius: 11px; margin-top: 10px; padding: 10px; background: #111A23; }
            QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 7px; color: #9FB6C8; font-weight: 700; }
            QGroupBox#actionPanel { background: #0F1822; border: 1px solid #29445E; }
            QFrame#card { background: #101A24; border: 1px solid #223140; border-radius: 11px; }
            QTextEdit#infoCard { background: #0F2030; border: 1px solid #1F4F79; border-radius: 9px; padding: 8px; color: #EAF0F6; }
            QTabWidget::pane { border: 1px solid #223140; border-radius: 9px; background: #111A23; }
            QTabBar::tab { background: #131F2B; color: #9FB6C8; padding: 7px 16px; min-width: 105px; border-top-left-radius: 7px; border-top-right-radius: 7px; }
            QTabBar::tab:selected { background: #1769D1; color: white; font-weight: 700; }
            QPushButton { background: #172331; border: 1px solid #2A4055; border-radius: 8px; padding: 7px 10px; color: #EAF0F6; min-height: 28px; }
            QPushButton:hover { background: #203247; border-color: #3D8BFF; }
            QPushButton:pressed { background: #1769D1; }
            QPushButton:disabled { color: #657382; background: #121B25; border-color: #223140; }
            QPushButton#primaryButton { background: #1769D1; border-color: #3182F6; font-weight: 800; }
            QPushButton#successButton { background: #1D7A43; border-color: #2EBB68; font-weight: 800; }
            QPushButton#toolButton { background: #132030; border-color: #284158; }
            QPushButton#navButton { text-align: left; background: transparent; border: 0; border-radius: 8px; padding: 9px 10px; color: #D6E2EE; }
            QPushButton#navButton:hover { background: #132234; }
            QPushButton#navButtonChecked { text-align: left; background: #1769D1; border: 0; border-radius: 8px; padding: 9px 10px; color: white; font-weight: 800; }
            QComboBox, QDoubleSpinBox { background: #0D141C; border: 1px solid #2A4055; border-radius: 7px; padding: 5px; color: #EAF0F6; min-height: 24px; }
            QTextEdit { background: #0D141C; border: 1px solid #223140; border-radius: 8px; padding: 7px; color: #EAF0F6; }
            QTableWidget { background: #0D141C; alternate-background-color: #111A23; border: 1px solid #223140; border-radius: 9px; gridline-color: #2A4055; color: #EAF0F6; selection-background-color: #1769D1; }
            QHeaderView::section { background: #162230; color: #D7E1EA; padding: 7px; border: 1px solid #223140; font-weight: 700; }
            QTableWidget::item { padding: 5px; }
            QSplitter::handle { background: #1A2A3A; }
            QSplitter::handle:hover { background: #2C4A65; }
            QScrollBar:vertical, QScrollBar:horizontal { background: #0B1118; width: 10px; height: 10px; }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #34495E; border-radius: 5px; }
        """)

    def open_plot_window(self, plot_widget: PlotWidget, title: str):
        dialog = PlotDialog(plot_widget.figure, title, self)
        dialog.show()
        if not hasattr(self, "_plot_dialogs"):
            self._plot_dialogs = []
        self._plot_dialogs.append(dialog)

    def build_ui(self):
        main = QWidget()
        root = QHBoxLayout(main)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        def label(text: str) -> QLabel:
            w = QLabel(text)
            w.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return w

        # -----------------------------

        # -----------------------------
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(184)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(8)

        logo = QLabel("SBM100B")
        logo.setObjectName("appTitle")
        subtitle = QLabel("Measurement Analyzer")
        subtitle.setObjectName("appSubTitle")
        side.addWidget(logo)
        side.addWidget(subtitle)
        side.addSpacing(10)

        nav_label = QLabel("DASHBOARD")
        nav_label.setObjectName("sectionLabel")
        side.addWidget(nav_label)
        self.nav_buttons = []
        for txt in [
            "▣  Live & Capture",
            "▧  Frequency Response",
            "▧  Sweep Response",
            "▧  Linearity",
            "▧  Noise Floor",
            "▧  THD / SINAD",
        ]:
            b = QPushButton(txt)
            b.setObjectName("navButtonChecked" if "Live" in txt else "navButton")
            b.setMinimumHeight(34)
            side.addWidget(b)
            self.nav_buttons.append(b)
        side.addSpacing(8)
        settings_label = QLabel("SETTINGS")
        settings_label.setObjectName("sectionLabel")
        side.addWidget(settings_label)
        for txt in ["⚙  Device & Calibration", "⚙  Sweep Settings", "ⓘ  About"]:
            b = QPushButton(txt)
            b.setObjectName("navButton")
            b.setMinimumHeight(32)
            side.addWidget(b)
        side.addStretch(1)

        status_card = QFrame()
        status_card.setObjectName("card")
        st = QVBoxLayout(status_card)
        st.setContentsMargins(10, 10, 10, 10)
        self.sidebar_status_label = QLabel("Status: Ready")
        self.sidebar_status_label.setObjectName("metricSmall")
        self.sidebar_device_label = QLabel("Device: -")
        self.sidebar_sr_label = QLabel("SR: -")
        self.sidebar_rms_label = QLabel("RMS\n-- dBFS")
        self.sidebar_rms_label.setObjectName("metricValue")
        self.sidebar_spl_label = QLabel("SPL: not calibrated")
        self.sidebar_spl_label.setObjectName("metricSmall")
        for w in [
            self.sidebar_status_label,
            self.sidebar_device_label,
            self.sidebar_sr_label,
            self.sidebar_rms_label,
            self.sidebar_spl_label,
        ]:
            st.addWidget(w)
        side.addWidget(status_card)

        # -----------------------------
        # Main content
        # -----------------------------
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(9)

        header_card = QFrame()
        header_card.setObjectName("topHeaderCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(10)
        self.ui_version_label = QLabel("SBM100 Analyzer")
        self.ui_version_label.setObjectName("uiVersionLabel")
        self.ui_workflow_label = QLabel("SBM100B Measurement Dashboard")
        self.ui_workflow_label.setObjectName("uiWorkflowLabel")
        header_layout.addWidget(self.ui_version_label)
        header_layout.addWidget(self.ui_workflow_label)
        header_layout.addStretch(1)

        # Top controls in compact tabs
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setMaximumHeight(126)

        live_tab = QWidget()
        live_grid = QGridLayout(live_tab)
        live_grid.setContentsMargins(8, 6, 8, 6)
        live_grid.setHorizontalSpacing(8)
        live_grid.setVerticalSpacing(5)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(230)
        self.refresh_devices()

        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["96000", "48000"])
        self.sr_combo.setMaximumWidth(100)

        self.buffer_box = QDoubleSpinBox()
        self.buffer_box.setRange(1, 30)
        self.buffer_box.setValue(22)
        self.buffer_box.setSuffix(" s")
        self.buffer_box.setMaximumWidth(90)

        self.analysis_window_box = QDoubleSpinBox()
        self.analysis_window_box.setRange(0.5, 10.0)
        self.analysis_window_box.setDecimals(1)
        self.analysis_window_box.setSingleStep(0.5)
        self.analysis_window_box.setValue(3.0)
        self.analysis_window_box.setSuffix(" s")
        self.analysis_window_box.setMaximumWidth(90)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_devices)
        self.start_btn = QPushButton("Start Live")
        self.start_btn.clicked.connect(self.start_live)
        self.stop_btn = QPushButton("Stop Live")
        self.stop_btn.clicked.connect(self.stop_live)
        self.stop_btn.setEnabled(False)
        open_wav = QPushButton("Open WAV")
        open_wav.clicked.connect(self.open_wav)
        save_live = QPushButton("Save Live WAV")
        save_live.clicked.connect(self.save_live)

        self.open_analysis_windows = QCheckBox(
            "Open analysis plots in separate windows"
        )
        self.open_analysis_windows.setChecked(True)

        live_grid.addWidget(label("Device"), 0, 0)
        live_grid.addWidget(self.device_combo, 0, 1, 1, 3)
        live_grid.addWidget(label("Sample rate"), 0, 4)
        live_grid.addWidget(self.sr_combo, 0, 5)
        live_grid.addWidget(label("Live buffer"), 0, 6)
        live_grid.addWidget(self.buffer_box, 0, 7)
        live_grid.addWidget(label("Analysis window"), 0, 8)
        live_grid.addWidget(self.analysis_window_box, 0, 9)
        live_grid.addWidget(refresh, 1, 0)
        live_grid.addWidget(self.start_btn, 1, 1)
        live_grid.addWidget(self.stop_btn, 1, 2)
        live_grid.addWidget(open_wav, 1, 3)
        live_grid.addWidget(save_live, 1, 4)
        live_grid.addWidget(self.open_analysis_windows, 1, 5, 1, 4)
        live_grid.setColumnStretch(10, 1)

        cal_tab = QWidget()
        cal_grid = QGridLayout(cal_tab)
        cal_grid.setContentsMargins(8, 6, 8, 6)
        cal_grid.setHorizontalSpacing(8)
        cal_grid.setVerticalSpacing(5)

        self.ref_spl = QDoubleSpinBox()
        self.ref_spl.setRange(0, 160)
        self.ref_spl.setValue(72)
        self.ref_spl.setSuffix(" dB SPL")
        self.ref_spl.setMaximumWidth(140)

        self.measured_rms = QDoubleSpinBox()
        self.measured_rms.setRange(-180, 0)
        self.measured_rms.setValue(-62)
        self.measured_rms.setSuffix(" dBFS")
        self.measured_rms.setMaximumWidth(140)

        self.external_spl_box = QDoubleSpinBox()
        self.external_spl_box.setRange(0, 160)
        self.external_spl_box.setValue(72)
        self.external_spl_box.setSuffix(" dB SPL")
        self.external_spl_box.setMaximumWidth(140)

        self.use_external_spl = QCheckBox("Use external SPL for Linearity")
        self.use_external_spl.setChecked(False)
        self.auto_reset_after_capture = QCheckBox("Auto reset buffer after Capture")
        self.auto_reset_after_capture.setChecked(True)

        use_current_rms_btn = QPushButton("Use Current RMS")
        use_current_rms_btn.clicked.connect(self.use_current_rms_for_calibration)
        cal_btn = QPushButton("Set Calibration")
        cal_btn.clicked.connect(self.set_calibration)
        self.cal_label = QLabel(self.calibration_text())

        cal_grid.addWidget(label("Reference SPL"), 0, 0)
        cal_grid.addWidget(self.ref_spl, 0, 1)
        cal_grid.addWidget(label("Measured RMS"), 0, 2)
        cal_grid.addWidget(self.measured_rms, 0, 3)
        cal_grid.addWidget(use_current_rms_btn, 0, 4)
        cal_grid.addWidget(cal_btn, 0, 5)
        cal_grid.addWidget(self.cal_label, 0, 6, 1, 2)
        cal_grid.addWidget(label("External SPL now"), 1, 0)
        cal_grid.addWidget(self.external_spl_box, 1, 1)
        cal_grid.addWidget(self.use_external_spl, 1, 2, 1, 3)
        cal_grid.addWidget(self.auto_reset_after_capture, 1, 5, 1, 3)
        cal_grid.setColumnStretch(8, 1)

        sweep_tab = QWidget()
        sweep_grid = QGridLayout(sweep_tab)
        sweep_grid.setContentsMargins(8, 6, 8, 6)
        sweep_grid.setHorizontalSpacing(8)
        sweep_grid.setVerticalSpacing(5)

        self.sweep_duration_box = QDoubleSpinBox()
        self.sweep_duration_box.setRange(1.0, 30.0)
        self.sweep_duration_box.setDecimals(1)
        self.sweep_duration_box.setSingleStep(0.5)
        self.sweep_duration_box.setValue(14.0)
        self.sweep_duration_box.setSuffix(" s")
        self.sweep_duration_box.setMaximumWidth(90)

        self.sweep_min_freq_box = QDoubleSpinBox()
        self.sweep_min_freq_box.setRange(10, 5000)
        self.sweep_min_freq_box.setDecimals(0)
        self.sweep_min_freq_box.setValue(20)
        self.sweep_min_freq_box.setSuffix(" Hz")
        self.sweep_min_freq_box.setMaximumWidth(110)

        self.sweep_max_freq_box = QDoubleSpinBox()
        self.sweep_max_freq_box.setRange(100, 48000)
        self.sweep_max_freq_box.setDecimals(0)
        self.sweep_max_freq_box.setValue(20000)
        self.sweep_max_freq_box.setSuffix(" Hz")
        self.sweep_max_freq_box.setMaximumWidth(120)

        self.sweep_mode_combo = QComboBox()
        self.sweep_mode_combo.addItems(
            ["Old RMS mapping", "BTB65 known", "Known custom", "Auto tracking"]
        )
        self.sweep_mode_combo.setMinimumWidth(155)

        self.sweep_type_combo = QComboBox()
        self.sweep_type_combo.addItems(["Log", "Linear"])
        self.sweep_type_combo.setMaximumWidth(90)

        self.sweep_direction_combo = QComboBox()
        self.sweep_direction_combo.addItems(["Up", "Down"])
        self.sweep_direction_combo.setMaximumWidth(90)

        self.sweep_trim_start_box = QDoubleSpinBox()
        self.sweep_trim_start_box.setRange(0.0, 5.0)
        self.sweep_trim_start_box.setDecimals(2)
        self.sweep_trim_start_box.setSingleStep(0.1)
        self.sweep_trim_start_box.setValue(0.20)
        self.sweep_trim_start_box.setSuffix(" s")
        self.sweep_trim_start_box.setMaximumWidth(90)

        self.sweep_trim_end_box = QDoubleSpinBox()
        self.sweep_trim_end_box.setRange(0.0, 5.0)
        self.sweep_trim_end_box.setDecimals(2)
        self.sweep_trim_end_box.setSingleStep(0.1)
        self.sweep_trim_end_box.setValue(0.20)
        self.sweep_trim_end_box.setSuffix(" s")
        self.sweep_trim_end_box.setMaximumWidth(90)

        self.sweep_smooth_check = QCheckBox("Sweep smoothing")
        self.sweep_smooth_check.setChecked(True)

        sweep_grid.addWidget(label("Duration"), 0, 0)
        sweep_grid.addWidget(self.sweep_duration_box, 0, 1)
        sweep_grid.addWidget(label("Band"), 0, 2)
        sweep_grid.addWidget(self.sweep_min_freq_box, 0, 3)
        sweep_grid.addWidget(self.sweep_max_freq_box, 0, 4)
        sweep_grid.addWidget(label("Mode"), 0, 5)
        sweep_grid.addWidget(self.sweep_mode_combo, 0, 6)
        sweep_grid.addWidget(label("Type"), 1, 0)
        sweep_grid.addWidget(self.sweep_type_combo, 1, 1)
        sweep_grid.addWidget(label("Direction"), 1, 2)
        sweep_grid.addWidget(self.sweep_direction_combo, 1, 3)
        sweep_grid.addWidget(label("Trim start/end"), 1, 4)
        sweep_grid.addWidget(self.sweep_trim_start_box, 1, 5)
        sweep_grid.addWidget(self.sweep_trim_end_box, 1, 6)
        sweep_grid.addWidget(self.sweep_smooth_check, 1, 7)
        sweep_grid.setColumnStretch(8, 1)

        self.settings_tabs.addTab(live_tab, "Live / Device")
        self.settings_tabs.addTab(cal_tab, "Calibration")
        self.settings_tabs.addTab(sweep_tab, "Sweep settings")

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(105)
        self.info.setMaximumHeight(138)
        self.info.setObjectName("infoCard")

        action_panel = QGroupBox("Measurement actions - polished workflow")
        action_panel.setObjectName("actionPanel")
        action_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        action_panel.setMaximumHeight(148)
        action_grid = QGridLayout(action_panel)
        action_grid.setContentsMargins(10, 8, 10, 8)
        action_grid.setHorizontalSpacing(8)
        action_grid.setVerticalSpacing(7)

        self.test_type = QComboBox()
        self.test_type.addItems(
            [
                "Frequency response",
                "Sweep response",
                "Linearity",
                "Noise floor",
                "THD",
                "General",
            ]
        )
        self.test_type.setMinimumWidth(205)
        self.action_hint = QLabel("Selected workflow: Frequency response")
        self.action_hint.setObjectName("actionHint")

        self.capture_btn = QPushButton("Capture + Auto Detect")
        self.capture_btn.setObjectName("primaryButton")
        self.capture_btn.clicked.connect(self.capture_current)
        self.clear_info_btn = QPushButton("Resume Live Info")
        self.clear_info_btn.clicked.connect(self.resume_live_info)
        self.reset_buffer_btn = QPushButton("Reset Buffer")
        self.reset_buffer_btn.clicked.connect(self.reset_live_buffer)
        self.record_sweep_btn = QPushButton("Record Fresh Sweep")
        self.record_sweep_btn.clicked.connect(self.record_fresh_sweep)
        self.analyze_sweep_btn = QPushButton("Analyze Sweep")
        self.analyze_sweep_btn.clicked.connect(self.analyze_sweep_response)
        self.analyze_fr_btn = QPushButton("Analyze Frequency Response")
        self.analyze_fr_btn.clicked.connect(self.analyze_frequency_response)
        self.analyze_lin_btn = QPushButton("Analyze Linearity")
        self.analyze_lin_btn.clicked.connect(self.analyze_linearity)
        self.analyze_noise_btn = QPushButton("Analyze Noise Floor")
        self.analyze_noise_btn.clicked.connect(self.analyze_noise_floor)
        self.analyze_thd_btn = QPushButton("Analyze THD Auto")
        self.analyze_thd_btn.clicked.connect(self.analyze_thd_current)
        self.analyze_thdn_btn = QPushButton("Analyze THD+N / SINAD Auto")
        self.analyze_thdn_btn.clicked.connect(self.analyze_thdn_current)
        self.open_wave_btn = QPushButton("Open Waveform")
        self.open_wave_btn.setObjectName("toolButton")
        self.open_wave_btn.clicked.connect(
            lambda: self.open_plot_window(self.wave, "Live Waveform")
        )
        self.open_fft_btn = QPushButton("Plot FFT")
        self.open_fft_btn.setObjectName("toolButton")
        self.open_fft_btn.clicked.connect(
            lambda: self.open_plot_window(self.fft, "Live FFT / Spectrum")
        )
        self.open_spec_btn = QPushButton("Plot Spectrogram")
        self.open_spec_btn.setObjectName("toolButton")
        self.open_spec_btn.clicked.connect(self.open_spectrogram_window)
        self.export_btn = QPushButton("Export CSV + Plots")
        self.export_btn.setObjectName("toolButton")
        self.export_btn.clicked.connect(self.export_all)

        self.action_buttons = [
            self.capture_btn,
            self.reset_buffer_btn,
            self.record_sweep_btn,
            self.analyze_sweep_btn,
            self.clear_info_btn,
            self.analyze_fr_btn,
            self.analyze_lin_btn,
            self.analyze_noise_btn,
            self.analyze_thd_btn,
            self.analyze_thdn_btn,
            self.open_wave_btn,
            self.open_fft_btn,
            self.open_spec_btn,
            self.export_btn,
        ]
        for b in self.action_buttons:
            b.setMinimumWidth(112)
            b.setMinimumHeight(32)
            b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.analyze_fr_btn.setObjectName("successButton")
        self.analyze_sweep_btn.setObjectName("successButton")
        self.analyze_lin_btn.setObjectName("successButton")
        self.analyze_noise_btn.setObjectName("successButton")
        self.analyze_thd_btn.setObjectName("successButton")
        self.analyze_thdn_btn.setObjectName("successButton")
        self.capture_btn.setMinimumWidth(150)
        self.record_sweep_btn.setMinimumWidth(150)
        self.analyze_fr_btn.setMinimumWidth(190)
        self.analyze_thdn_btn.setMinimumWidth(190)

        action_grid.addWidget(QLabel("Measurement mode"), 0, 0)
        action_grid.addWidget(self.test_type, 0, 1)
        action_grid.addWidget(self.action_hint, 0, 2, 1, 5)
        action_grid.addWidget(self.export_btn, 0, 7)
        action_grid.addWidget(self.capture_btn, 1, 0)
        action_grid.addWidget(self.record_sweep_btn, 1, 1)
        action_grid.addWidget(self.analyze_sweep_btn, 1, 2)
        action_grid.addWidget(self.analyze_fr_btn, 1, 3)
        action_grid.addWidget(self.analyze_lin_btn, 1, 4)
        action_grid.addWidget(self.analyze_noise_btn, 1, 5)
        action_grid.addWidget(self.analyze_thd_btn, 1, 6)
        action_grid.addWidget(self.analyze_thdn_btn, 1, 7)
        action_grid.addWidget(self.reset_buffer_btn, 2, 0)
        action_grid.addWidget(self.clear_info_btn, 2, 1)
        action_grid.addWidget(self.open_wave_btn, 2, 2)
        action_grid.addWidget(self.open_fft_btn, 2, 3)
        action_grid.addWidget(self.open_spec_btn, 2, 4)
        action_grid.setColumnStretch(8, 1)
        self.test_type.currentTextChanged.connect(self.update_action_buttons)

        # Large waveform dashboard card + hidden FFT/result plot objects
        self.wave = PlotWidget("Wave")
        self.fft = PlotWidget("FFT")
        self.result_plot = PlotWidget("Result")
        for plot in (self.wave, self.fft, self.result_plot):
            plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        waveform_panel = QGroupBox("Live Waveform")
        waveform_layout = QVBoxLayout(waveform_panel)
        waveform_layout.setContentsMargins(10, 10, 10, 10)
        waveform_panel.setMinimumHeight(285)
        waveform_layout.addWidget(self.wave)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Test",
                "Auto Input",
                "Auto Frequency Hz",
                "Auto SPL dB",
                "External SPL dB",
                "Linearity SPL Used",
                "File",
                "RMS dBFS",
                "Peak dBFS",
                "Clipping",
                "Notes",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setMinimumHeight(235)
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 135)
        self.table.setColumnWidth(3, 105)
        self.table.setColumnWidth(4, 110)
        self.table.setColumnWidth(5, 135)
        self.table.setColumnWidth(6, 300)
        self.table.setColumnWidth(10, 320)

        lower_splitter = QSplitter(Qt.Horizontal)
        lower_splitter.addWidget(self.table)
        lower_splitter.addWidget(self.result_plot)
        lower_splitter.setStretchFactor(0, 3)
        lower_splitter.setStretchFactor(1, 2)
        lower_splitter.setChildrenCollapsible(False)
        lower_splitter.setSizes([820, 460])

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(waveform_panel)
        main_splitter.addWidget(lower_splitter)
        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setSizes([380, 390])

        outer.addWidget(header_card)
        outer.addWidget(self.settings_tabs)
        outer.addWidget(self.info)
        outer.addWidget(action_panel)
        outer.addWidget(main_splitter, 1)

        root.addWidget(sidebar)
        root.addWidget(content, 1)
        self.setCentralWidget(main)
        self.update_action_buttons(self.test_type.currentText())

    def update_action_buttons(self, test_name: str | None = None):
        """Show only the buttons relevant to the selected measurement mode.

        The analysis functions are unchanged; this only cleans up the interface so the
        active workflow is easier to understand on small screens.
        """
        test = test_name or self.test_type.currentText()

        # Common buttons that are useful for workflow.
        common = {
            self.reset_buffer_btn,
            self.clear_info_btn,
            self.open_wave_btn,
            self.open_fft_btn,
            self.open_spec_btn,
            self.export_btn,
        }

        mode_specific = {
            "Frequency response": {self.capture_btn, self.analyze_fr_btn},
            "Sweep response": {self.record_sweep_btn, self.analyze_sweep_btn},
            "Linearity": {self.capture_btn, self.analyze_lin_btn},
            "Noise floor": {self.capture_btn, self.analyze_noise_btn},
            "THD": {self.capture_btn, self.analyze_thd_btn, self.analyze_thdn_btn},
            "General": {self.capture_btn},
        }

        visible = common | mode_specific.get(test, {self.capture_btn})

        workflow_hints = {
            "Frequency response": "Capture stable tones, then run Frequency Response analysis.",
            "Sweep response": "Record a fresh sweep, then run Analyze Sweep.",
            "Linearity": "Capture each SPL point, then run Linearity.",
            "Noise floor": "Capture quiet-room audio, then run Noise Floor.",
            "THD": "Capture a clean tone, then run THD or THD+N / SINAD.",
            "General": "Capture and review the current audio window.",
        }
        if hasattr(self, "action_hint"):
            self.action_hint.setText(
                workflow_hints.get(test, "Selected workflow ready.")
            )

        for button in self.action_buttons:
            button.setVisible(button in visible)

        # Visual sidebar highlight only; the existing measurement logic is unchanged.
        if hasattr(self, "nav_buttons"):
            nav_lookup = {
                "Frequency response": "Frequency Response",
                "Sweep response": "Sweep Response",
                "Linearity": "Linearity",
                "Noise floor": "Noise Floor",
                "THD": "THD",
                "General": "Live & Capture",
            }
            wanted = nav_lookup.get(test, "Live & Capture")
            for nav in self.nav_buttons:
                nav.setObjectName(
                    "navButtonChecked" if wanted in nav.text() else "navButton"
                )
                nav.style().unpolish(nav)
                nav.style().polish(nav)

        # Move the user to the relevant settings tab automatically.
        if test == "Sweep response":
            self.settings_tabs.setCurrentIndex(2)
        elif test == "Linearity":
            self.settings_tabs.setCurrentIndex(1)
        else:
            self.settings_tabs.setCurrentIndex(0)

    def open_spectrogram_window(self):
        if len(self.live_buffer) == 0:
            QMessageBox.warning(
                self, "No audio", "Start live, record, or open a WAV file first."
            )
            return
        plot = PlotWidget("Spectrogram")
        plot.spectrogram(self.live_buffer, self.live_samplerate)
        self.open_plot_window(plot, "Live Spectrogram")

    def calibration_text(self):
        if self.offset_db is None:
            return "Calibration: not set - Auto SPL unavailable"
        return f"Calibration offset: {self.offset_db:.2f} dB"

    def spl_text_value(self, rms_value: float):
        spl = dbfs_to_spl(rms_value, self.offset_db)
        return None if spl is None else float(spl)

    def refresh_devices(self):
        self.device_combo.clear()
        for d in list_input_devices():
            self.device_combo.addItem(f"{d['index']}: {d['name']}", d["index"])
        for i in range(self.device_combo.count()):
            if "SB-AURORA" in self.device_combo.itemText(i):
                self.device_combo.setCurrentIndex(i)
                break

    def set_calibration(self):
        self.offset_db = float(self.ref_spl.value()) - float(self.measured_rms.value())
        save_calibration(CONFIG, self.offset_db)
        self.cal_label.setText(self.calibration_text())
        self.hold_info = False
        self.update_live()

    def drain_audio_queue(self):
        """Move all pending sounddevice blocks into live_buffer before capture/analysis.
        This prevents the buttons from analyzing an older buffer while new audio is waiting in the queue.
        """
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
        if chunks:
            self.live_buffer = np.concatenate(
                [self.live_buffer, np.concatenate(chunks)]
            )
            max_len = int(float(self.buffer_box.value()) * self.live_samplerate)
            if len(self.live_buffer) > max_len:
                self.live_buffer = self.live_buffer[-max_len:]

    def reset_live_buffer(self):
        """Clear current live audio so the next capture contains only the new tone/level."""
        while not self.audio_queue.empty():
            self.audio_queue.get()
        self.live_buffer = np.array([], dtype=np.float64)
        self.hold_info = False
        self.info.setText(
            "Live buffer reset\n"
            "Change the speaker frequency/level now, wait until the analysis window fills, then press Capture.\n"
            f"Current analysis window: {self.analysis_window_box.value():.1f} s"
        )
        self.wave.waveform(self.live_buffer, self.live_samplerate)
        self.fft.spectrum(self.live_buffer, self.live_samplerate)

    def prepare_fresh_audio_for_button(self):
        """Refresh the buffer from queued audio immediately before a button action."""
        self.drain_audio_queue()

    def get_analysis_buffer(self) -> np.ndarray:
        if len(self.live_buffer) == 0:
            return self.live_buffer
        n = int(float(self.analysis_window_box.value()) * self.live_samplerate)
        n = max(1, min(n, len(self.live_buffer)))
        return self.live_buffer[-n:].copy()

    def get_capture_buffer_for_test(self, test: str) -> np.ndarray:
        """Tone tests use the short analysis window; sweep uses its own sweep duration."""
        if len(self.live_buffer) == 0:
            return self.live_buffer
        if test == "Sweep response":
            n = int(float(self.sweep_duration_box.value()) * self.live_samplerate)
        else:
            n = int(float(self.analysis_window_box.value()) * self.live_samplerate)
        n = max(1, min(n, len(self.live_buffer)))
        return self.live_buffer[-n:].copy()

    def segment_stability_report(
        self, data: np.ndarray, sr: int
    ) -> tuple[float | None, str]:
        if len(data) < int(0.5 * sr):
            return None, "short segment"
        win = max(1, int(0.25 * sr))
        values = []
        for start in range(0, len(data) - win + 1, win):
            values.append(rms_dbfs(data[start : start + win]))
        if len(values) < 2:
            return 0.0, "stable check unavailable"
        spread = float(max(values) - min(values))
        if spread <= 1.0:
            msg = "stable"
        elif spread <= 3.0:
            msg = "acceptable, but not perfectly stable"
        else:
            msg = "unstable: re-record with constant tone/level"
        return spread, msg

    def use_current_rms_for_calibration(self):
        self.prepare_fresh_audio_for_button()
        data = self.get_analysis_buffer()
        if len(data) == 0:
            QMessageBox.warning(self, "No audio", "Start live or open WAV first.")
            return
        a = analyze_array(data, self.live_samplerate)
        self.measured_rms.setValue(float(a["rms_dbfs"]))
        self.hold_info = True
        self.info.setText(
            "Calibration helper\n"
            f"Analysis window used: last {self.analysis_window_box.value():.1f} s\n"
            f"Current RMS copied: {a['rms_dbfs']:.2f} dBFS\n"
            "Now enter the real SPL from your external SPL meter, then press Set Calibration."
        )

    def audio_callback(self, indata, frames, time, status):
        mono = indata[:, 0].astype(np.float64)
        if indata.dtype == np.int32:
            mono = mono / 2147483648.0
        self.audio_queue.put(mono)

    def start_live(self):
        if self.stream is not None:
            return
        self.live_samplerate = int(self.sr_combo.currentText())
        self.live_buffer = np.array([], dtype=np.float64)
        self.hold_info = False
        while not self.audio_queue.empty():
            self.audio_queue.get()
        try:
            self.stream = sd.InputStream(
                device=self.device_combo.currentData(),
                channels=2,
                samplerate=self.live_samplerate,
                dtype="int32",
                callback=self.audio_callback,
                blocksize=2048,
            )
            self.stream.start()
        except Exception as e:
            QMessageBox.critical(self, "Live monitor failed", str(e))
            self.stream = None
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start()

    def stop_live(self):
        self.timer.stop()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def resume_live_info(self):
        self.hold_info = False
        self.update_live()

    def update_live(self):
        self.drain_audio_queue()
        if len(self.live_buffer) == 0:
            return

        self.wave.waveform(self.live_buffer, self.live_samplerate)
        self.fft.spectrum(self.live_buffer, self.live_samplerate)

        if self.hold_info:
            return

        data = self.get_analysis_buffer()
        a = analyze_array(data, self.live_samplerate)
        spl = self.spl_text_value(a["rms_dbfs"])
        if hasattr(self, "sidebar_rms_label"):
            self.sidebar_status_label.setText(
                "Status: Live" if self.stream is not None else "Status: Ready"
            )
            self.sidebar_device_label.setText(
                f"Device: {self.device_combo.currentText()[:18]}"
            )
            self.sidebar_sr_label.setText(f"SR: {self.live_samplerate/1000:.1f} kHz")
            self.sidebar_rms_label.setText(f"RMS\n{a['rms_dbfs']:.1f} dBFS")
            self.sidebar_spl_label.setText(
                "SPL: not calibrated" if spl is None else f"SPL: {spl:.1f} dB"
            )
        spread, stable_msg = self.segment_stability_report(data, self.live_samplerate)
        spread_txt = "n/a" if spread is None else f"{spread:.2f} dB"
        self.info.setText(
            "Live auto analysis\n"
            f"Sample rate: {self.live_samplerate} Hz\n"
            "Input format: 2 ch 32-bit Integer, channel 1\n"
            f"Live buffer: {len(self.live_buffer) / self.live_samplerate:.2f} s\n"
            f"Analysis window used: last {len(data) / self.live_samplerate:.2f} s\n"
            f"Peak: {a['peak_dbfs']:.2f} dBFS\n"
            f"RMS: {a['rms_dbfs']:.2f} dBFS\n"
            f"Auto SPL: {'Not calibrated' if spl is None else f'{spl:.2f} dB SPL'}\n"
            f"Auto frequency: {a['dominant_frequency']:.2f} Hz\n"
            f"Stability: {stable_msg} (RMS spread {spread_txt})\n"
            f"Clipping: {'YES' if a['clipping'] else 'No'}"
        )

    def save_live(self):
        self.prepare_fresh_audio_for_button()
        if len(self.live_buffer) == 0:
            QMessageBox.warning(self, "No audio", "No live buffer to save.")
            return None
        path = (
            RECORDINGS / f"sbm100b_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        save_wav(path, self.live_buffer, self.live_samplerate)
        QMessageBox.information(self, "Saved", f"Saved full live buffer:\n{path}")
        return path

    def open_wav(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV", str(RECORDINGS), "WAV files (*.wav)"
        )
        if not path:
            return
        audio = load_wav(path)
        self.live_buffer = audio.data
        self.live_samplerate = audio.samplerate
        self.hold_info = False
        self.update_live()

    def auto_input_value(
        self,
        test: str,
        frequency_hz: float,
        spl_db: float | None,
        external_spl: float | None,
    ):
        if test == "Linearity":
            return external_spl if external_spl is not None else spl_db
        if test == "Noise floor":
            return spl_db
        if test == "Sweep response":
            return None
        return frequency_hz

    def record_fresh_sweep(self):
        """Dedicated sweep workflow: clear the old live buffer, then record a fresh window.
        This avoids timing errors where the speaker sweep starts after the live buffer already contains old audio.
        """
        if self.stream is None:
            self.start_live()
            if self.stream is None:
                return

        #  that make sure the live buffer can hold the full sweep plus time  to press Play.
        sweep_dur = float(self.sweep_duration_box.value())
        record_seconds = min(30.0, max(sweep_dur + 4.0, sweep_dur * 1.25))
        if float(self.buffer_box.value()) < record_seconds:
            self.buffer_box.setValue(record_seconds)

        while not self.audio_queue.empty():
            self.audio_queue.get()
        self.live_buffer = np.array([], dtype=np.float64)
        self.hold_info = True
        self.info.setText(
            "Fresh sweep recording armed\n"
            f"Recording window: {record_seconds:.1f} s\n"
            f"Expected sweep duration: {sweep_dur:.1f} s\n\n"
            "Press PLAY on the BTB65 sweep now.\n"
            "When recording finishes, the sweep will be saved as a Sweep response row. Then press Analyze Sweep."
        )
        QTimer.singleShot(int(record_seconds * 1000), self.finish_fresh_sweep_recording)

    def finish_fresh_sweep_recording(self):
        self.prepare_fresh_audio_for_button()
        data = self.live_buffer.copy()
        if len(data) == 0:
            QMessageBox.warning(
                self,
                "No sweep audio",
                "No audio was recorded during the fresh sweep window.",
            )
            return

        test = "Sweep response"
        path = (
            RECORDINGS
            / f"sweep_response_fresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        save_wav(path, data, self.live_samplerate)
        a = analyze_array(data, self.live_samplerate)
        spl = self.spl_text_value(a["rms_dbfs"])
        duration_s = len(data) / self.live_samplerate
        notes = (
            f"fresh sweep recording {duration_s:.2f}s; "
            f"expected sweep {float(self.sweep_duration_box.value()):.2f}s; "
            "Analyze Sweep uses this saved row"
        )
        row_data = {
            "test": test,
            "auto_input": None,
            "auto_frequency_hz": float(a["dominant_frequency"]),
            "auto_spl_db": spl,
            "external_spl_db": None,
            "linearity_spl_used": None,
            "file": str(path),
            "rms_dbfs": float(a["rms_dbfs"]),
            "peak_dbfs": float(a["peak_dbfs"]),
            "clipping": bool(a["clipping"]),
            "notes": notes,
        }
        self.measurements.append(row_data)
        self.add_table_row(row_data)
        self.wave.waveform(data, self.live_samplerate)
        self.fft.spectrum(data, self.live_samplerate)
        self.hold_info = True
        self.info.setText(
            "Fresh sweep recording finished\n"
            f"Saved duration: {duration_s:.2f} s\n"
            f"Saved file: {path}\n"
            f"Peak: {a['peak_dbfs']:.2f} dBFS\n"
            f"RMS: {a['rms_dbfs']:.2f} dBFS\n"
            f"Auto SPL: {'Not calibrated' if spl is None else f'{spl:.2f} dB SPL'}\n\n"
            "Now press Analyze Sweep.\n"
            "This workflow is preferred for BTB65 because the program starts a fresh recording before you press PLAY."
        )

    def capture_current(self):
        self.prepare_fresh_audio_for_button()
        test = self.test_type.currentText()
        data = self.get_capture_buffer_for_test(test)
        if len(data) == 0:
            QMessageBox.warning(self, "No audio", "Start live or open WAV first.")
            return
        required_seconds = (
            float(self.sweep_duration_box.value())
            if test == "Sweep response"
            else float(self.analysis_window_box.value())
        )
        min_required = int(required_seconds * self.live_samplerate * 0.90)
        if self.stream is not None and len(data) < min_required:
            QMessageBox.warning(
                self,
                "Fresh buffer not ready",
                f"The buffer has only {len(data) / self.live_samplerate:.2f} s of fresh audio.\n"
                f"Wait until it reaches about {required_seconds:.1f} s after changing/starting the signal, then capture again.",
            )
            return

        a = analyze_array(data, self.live_samplerate)
        spl = self.spl_text_value(a["rms_dbfs"])
        freq = float(a["dominant_frequency"])
        external_spl = (
            float(self.external_spl_box.value())
            if (test == "Linearity" and self.use_external_spl.isChecked())
            else None
        )
        linearity_spl_used = (
            external_spl
            if external_spl is not None
            else (spl if test == "Linearity" else None)
        )
        auto_input = self.auto_input_value(test, freq, spl, external_spl)
        spread, stable_msg = self.segment_stability_report(data, self.live_samplerate)
        spread_txt = "n/a" if spread is None else f"{spread:.2f} dB"

        safe_test = test.lower().replace(" ", "_")
        auto_tag = "na" if auto_input is None else f"{auto_input:.2f}".replace(".", "p")
        path = (
            RECORDINGS
            / f"{safe_test}_auto_{auto_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        save_wav(path, data, self.live_samplerate)

        notes = [
            f"analysis window {len(data) / self.live_samplerate:.2f}s",
            f"stability {stable_msg} ({spread_txt})",
        ]
        if spl is None:
            notes.append("Auto SPL unavailable: set calibration")
        if test == "Linearity":
            if external_spl is not None:
                notes.append("linearity uses external SPL")
            else:
                notes.append(
                    "linearity uses Auto SPL; external SPL recommended for scientific test"
                )
        elif test == "Frequency response":
            notes.append("auto input = detected frequency")
        elif test == "Sweep response":
            notes.append(
                f"sweep segment {len(data) / self.live_samplerate:.2f}s; Analyze Sweep for response curve"
            )
        elif test == "Noise floor":
            notes.append("record silence / quiet room")

        row_data = {
            "test": test,
            "auto_input": auto_input,
            "auto_frequency_hz": freq,
            "auto_spl_db": spl,
            "external_spl_db": external_spl,
            "linearity_spl_used": linearity_spl_used,
            "file": str(path),
            "rms_dbfs": float(a["rms_dbfs"]),
            "peak_dbfs": float(a["peak_dbfs"]),
            "clipping": bool(a["clipping"]),
            "notes": "; ".join(notes),
        }
        self.measurements.append(row_data)
        self.add_table_row(row_data)
        self.hold_info = True
        self.info.setText(
            "Captured with auto detection\n"
            f"Test: {test}\n"
            f"Saved segment: last {len(data) / self.live_samplerate:.2f} s\n"
            f"Auto frequency: {freq:.2f} Hz\n"
            f"Auto SPL: {'Not calibrated' if spl is None else f'{spl:.2f} dB SPL'}\n"
            f"External SPL: {'not used' if external_spl is None else f'{external_spl:.2f} dB SPL'}\n"
            f"Linearity SPL used: {'n/a' if linearity_spl_used is None else f'{linearity_spl_used:.2f} dB SPL'}\n"
            f"RMS: {a['rms_dbfs']:.2f} dBFS\n"
            f"Peak: {a['peak_dbfs']:.2f} dBFS\n"
            f"Stability: {stable_msg} (RMS spread {spread_txt})\n"
            f"Saved: {path}\n\n"
            "Next step: for tones, change tone/level and wait for the analysis window. For sweep, start the sweep and wait for the sweep-duration window before Capture."
        )
        if self.auto_reset_after_capture.isChecked() and self.stream is not None:
            while not self.audio_queue.empty():
                self.audio_queue.get()
            self.live_buffer = np.array([], dtype=np.float64)

    def add_table_row(self, r):
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            r["test"],
            "" if r["auto_input"] is None else f"{r['auto_input']:.2f}",
            f"{r['auto_frequency_hz']:.2f}",
            "" if r["auto_spl_db"] is None else f"{r['auto_spl_db']:.2f}",
            "" if r["external_spl_db"] is None else f"{r['external_spl_db']:.2f}",
            "" if r["linearity_spl_used"] is None else f"{r['linearity_spl_used']:.2f}",
            r["file"],
            f"{r['rms_dbfs']:.2f}",
            f"{r['peak_dbfs']:.2f}",
            "YES" if r["clipping"] else "No",
            r["notes"],
        ]
        for col, val in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(val))

    def read_measurements_from_table(self):
        rows = []
        for row in range(self.table.rowCount()):
            try:
                test_item = self.table.item(row, 0)
                if not test_item:
                    continue
                test = test_item.text().strip()
                if test.startswith("Frequency"):
                    test = "Frequency response"
                elif test.startswith("Sweep"):
                    test = "Sweep response"
                elif test.startswith("Linearity"):
                    test = "Linearity"
                elif test.startswith("Noise"):
                    test = "Noise floor"
                elif test.startswith("THD+N"):
                    test = "THD+N"
                elif test.startswith("THD"):
                    test = "THD"

                def get_float(col):
                    item = self.table.item(row, col)
                    if not item:
                        return None
                    txt = item.text().strip().replace(",", ".")
                    if txt == "":
                        return None
                    return float(txt)

                file_item = self.table.item(row, 6)
                clip_item = self.table.item(row, 9)
                note_item = self.table.item(row, 10)
                rows.append(
                    {
                        "test": test,
                        "auto_input": get_float(1),
                        "auto_frequency_hz": get_float(2),
                        "auto_spl_db": get_float(3),
                        "external_spl_db": get_float(4),
                        "linearity_spl_used": get_float(5),
                        "file": file_item.text() if file_item else "",
                        "rms_dbfs": get_float(7),
                        "peak_dbfs": get_float(8),
                        "clipping": (
                            (clip_item.text().strip().lower() == "yes")
                            if clip_item
                            else False
                        ),
                        "notes": note_item.text() if note_item else "",
                    }
                )
            except Exception:
                continue
        return rows

    def average_by_key(self, rows, key_name: str):
        grouped = {}
        for r in rows:
            key = r.get(key_name)
            if key is None:
                continue
            grouped.setdefault(round(float(key), 2), []).append(r)

        averaged = []
        for key, items in grouped.items():
            valid_rms = [x["rms_dbfs"] for x in items if x["rms_dbfs"] is not None]
            averaged.append(
                {
                    key_name: key,
                    "rms_dbfs": sum(valid_rms) / len(valid_rms) if valid_rms else None,
                    "auto_spl_db": (
                        None
                        if any(x["auto_spl_db"] is None for x in items)
                        else sum(x["auto_spl_db"] for x in items) / len(items)
                    ),
                    "auto_frequency_hz": (
                        None
                        if any(x["auto_frequency_hz"] is None for x in items)
                        else sum(x["auto_frequency_hz"] for x in items) / len(items)
                    ),
                    "count": len(items),
                }
            )
        return sorted(averaged, key=lambda x: x[key_name])

    def latest_sweep_row(self):
        rows = [
            m
            for m in self.read_measurements_from_table()
            if m["test"] == "Sweep response"
        ]
        rows = [r for r in rows if r.get("file")]
        return rows[-1] if rows else None

    def analyze_sweep_response(self):
        row = self.latest_sweep_row()
        source_text = "latest captured Sweep response row"
        if row is not None and Path(row["file"]).exists():
            audio = load_wav(row["file"])
            data = audio.data
            sr = audio.samplerate
        else:
            self.prepare_fresh_audio_for_button()
            data = self.get_capture_buffer_for_test("Sweep response")
            sr = self.live_samplerate
            source_text = "current live buffer"

        if len(data) == 0:
            QMessageBox.warning(
                self,
                "No sweep audio",
                "Select Test = Sweep response, record/capture a sweep, then Analyze Sweep.",
            )
            return

        mode = self.sweep_mode_combo.currentText()
        if mode == "Auto tracking":
            result = analyze_sweep_stft(
                data,
                sr,
                min_freq=float(self.sweep_min_freq_box.value()),
                max_freq=float(self.sweep_max_freq_box.value()),
                bins_per_octave=12,
            )
            analysis_method = "Auto STFT peak tracking"
        else:
            # BTB65 preset: official sweep is 20 Hz -> 20 kHz, 14 s, logarithmic.
            if mode in ("BTB65 known", "Old RMS mapping"):
                start_freq = 20.0
                stop_freq = 20000.0
                duration_s = 14.0
                direction = self.sweep_direction_combo.currentText()
                sweep_type = "Log"
            else:
                start_freq = float(self.sweep_min_freq_box.value())
                stop_freq = float(self.sweep_max_freq_box.value())
                duration_s = float(self.sweep_duration_box.value())
                direction = self.sweep_direction_combo.currentText()
                sweep_type = self.sweep_type_combo.currentText()

            if mode == "Old RMS mapping":
                # This uses the older sweep approach you provided: frequency -> expected time,
                # RMS in a small window, then normalize to 1 kHz. It avoids STFT peak tracking.
                result = analyze_external_sweep_response(
                    data,
                    sr,
                    start_hz=start_freq,
                    end_hz=stop_freq,
                    duration_s=duration_s,
                    bins=180,
                    sweep_type=sweep_type,
                    direction=direction,
                    window_ms=50.0,
                )
                analysis_method = "Old RMS time-mapped sweep method"
            else:
                result = analyze_sweep_known(
                    data,
                    sr,
                    start_freq=start_freq,
                    stop_freq=stop_freq,
                    duration_s=duration_s,
                    direction=direction,
                    sweep_type=sweep_type,
                    trim_start_s=float(self.sweep_trim_start_box.value()),
                    trim_end_s=float(self.sweep_trim_end_box.value()),
                    bins_per_octave=24,
                    smooth=self.sweep_smooth_check.isChecked(),
                )
                analysis_method = f"Known sweep mapping ({mode})"

        if not result["ok"]:
            QMessageBox.warning(
                self, "Sweep analysis failed", result.get("reason", "Unknown error")
            )
            return

        x = result["frequency_hz"]
        y = result["relative_db"]
        self.result_plot.xy(
            x,
            y,
            "Sweep Frequency Response",
            "Frequency [Hz]",
            "Level [dB re 1 kHz]",
            True,
        )
        if (
            getattr(self, "open_analysis_windows", None) is not None
            and self.open_analysis_windows.isChecked()
        ):
            self.open_plot_window(self.result_plot, "Sweep Frequency Response")

        csv_path = (
            TABLES / f"sweep_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frequency_hz", "level_db", "relative_db_re_1khz"])
            for fx, ly, ry in zip(
                result["frequency_hz"], result["level_db"], result["relative_db"]
            ):
                w.writerow([f"{fx:.6f}", f"{ly:.6f}", f"{ry:.6f}"])

        key_freqs = [20, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 20000]
        lines = [
            "Sweep response analysis",
            f"Source: {source_text}",
            f"Method: {analysis_method}",
            f"Sweep segment duration used: {result['duration_s']:.2f} s",
            f"Mapped/tracking points used: {result['points']}",
            f"Sweep direction: {result.get('detected_direction', 'known')}",
            f"Sweep start/stop: {result['detected_start_hz']:.1f} Hz -> {result['detected_stop_hz']:.1f} Hz",
            f"Sweep type: {result.get('sweep_type', self.sweep_type_combo.currentText())}",
            f"Trim: {result.get('trim_start_s', 0.0):.2f} s start, {result.get('trim_end_s', 0.0):.2f} s end",
            f"Auto segment used: {result.get('segment_start_s', 0.0):.2f} s -> {result.get('segment_end_s', result['duration_s']):.2f} s in the original buffer",
            f"CSV exported: {csv_path}",
            "",
            "Relative levels:",
        ]
        for kf in key_freqs:
            if kf >= float(x[0]) and kf <= float(x[-1]):
                idx = int(np.argmin(np.abs(x - kf)))
                lines.append(f"{kf:7.1f} Hz: {y[idx]:+6.2f} dB re 1 kHz")
        # Simple plausibility warning for wrong direction/trim. A small portable speaker
        # should not normally show extreme sub-bass levels relative to 1 kHz.
        warnings = result.get("warnings", []) or []
        if len(y):
            idx63 = int(np.argmin(np.abs(x - 63.0)))
            idx125 = int(np.argmin(np.abs(x - 125.0)))
            if y[idx63] > 18 or y[idx125] > 15:
                warnings.append(
                    "Very high low-frequency result detected. Try the opposite Direction or increase trim; result may be mapping/segment error."
                )
        if warnings:
            lines.extend(["", "Warnings:"])
            for wmsg in warnings:
                lines.append(f"- {wmsg}")
        lines.extend(
            [
                "",
                f"BTB65 preset uses: 20 Hz -> 20 kHz, 14 s, logarithmic sweep, Direction={direction}.",
                "Without a reference microphone this is the combined response of speaker + room + SBM100B/Aurora, not the microphone alone.",
                "Tone analysis modes are unchanged; Sweep response is only an additional mode. Old RMS mapping is the merged method from the earlier working sweep code.",
            ]
        )
        self.hold_info = True
        self.info.setText("\n".join(lines))

    def analyze_frequency_response(self):
        rows = [
            m
            for m in self.read_measurements_from_table()
            if m["test"] == "Frequency response"
        ]
        rows = [
            r
            for r in rows
            if r["auto_frequency_hz"] is not None and r["rms_dbfs"] is not None
        ]
        if not rows:
            QMessageBox.warning(
                self, "No data", "Capture frequency-response measurements first."
            )
            return
        averaged = self.average_by_key(rows, "auto_frequency_hz")
        levels = {
            m["auto_frequency_hz"]: m["rms_dbfs"]
            for m in averaged
            if m["rms_dbfs"] is not None
        }
        ref_freq = min(levels.keys(), key=lambda x: abs(x - 1000.0))
        ref = levels[ref_freq]
        x = sorted(levels.keys())
        y = [levels[f] - ref for f in x]
        self.result_plot.xy(
            x,
            y,
            "Frequency Response",
            "Auto-detected Frequency [Hz]",
            "Level [dB re 1 kHz]",
            True,
        )
        if (
            getattr(self, "open_analysis_windows", None) is not None
            and self.open_analysis_windows.isChecked()
        ):
            self.open_plot_window(self.result_plot, "Frequency Response")
        spl_values = [r["auto_spl_db"] for r in rows if r["auto_spl_db"] is not None]
        spl_spread_msg = "Auto SPL not available."
        if len(spl_values) >= 2:
            spread = max(spl_values) - min(spl_values)
            spl_spread_msg = f"Auto SPL spread across points: {spread:.2f} dB. Smaller is better; use reference mic for final lab test."
        self.hold_info = True
        self.info.setText(
            "Frequency response analysis\n"
            f"Rows used: {len(rows)}\n"
            f"Averaged points: {len(averaged)}\n"
            f"Reference frequency: {ref_freq:.2f} Hz\n"
            f"{spl_spread_msg}\n\n"
            "Auto frequency was used. Final scientific result still needs equalized input level or reference microphone."
        )

    def analyze_linearity(self):
        rows = [
            m for m in self.read_measurements_from_table() if m["test"] == "Linearity"
        ]
        rows = [
            r
            for r in rows
            if r["linearity_spl_used"] is not None and r["rms_dbfs"] is not None
        ]
        if not rows:
            QMessageBox.warning(
                self,
                "No usable linearity data",
                "Capture Linearity rows. Use external SPL if possible, or set calibration for Auto SPL.",
            )
            return
        averaged = self.average_by_key(rows, "linearity_spl_used")
        x = [r["linearity_spl_used"] for r in averaged]
        y = [r["rms_dbfs"] for r in averaged]
        self.result_plot.xy(x, y, "Linearity", "SPL Used [dB SPL]", "Output RMS [dBFS]")
        if (
            getattr(self, "open_analysis_windows", None) is not None
            and self.open_analysis_windows.isChecked()
        ):
            self.open_plot_window(self.result_plot, "Linearity")
        external_count = sum(1 for r in rows if r["external_spl_db"] is not None)
        method = "external SPL meter" if external_count else "Auto SPL from calibration"
        warning = ""
        if external_count == 0:
            warning = "\nWarning: Auto SPL is derived from RMS, so this is not an independent linearity proof. External SPL is recommended."
        self.hold_info = True
        self.info.setText(
            "Linearity analysis\n"
            f"Rows used: {len(rows)}\n"
            f"Averaged points: {len(averaged)}\n"
            f"X-axis method: {method}\n"
            "Keep frequency fixed, normally 1 kHz."
            f"{warning}"
        )

    def analyze_noise_floor(self):
        rows = [
            m for m in self.read_measurements_from_table() if m["test"] == "Noise floor"
        ]
        if not rows:
            QMessageBox.warning(
                self, "No noise data", "Capture Noise floor rows first."
            )
            return
        rms_values = [r["rms_dbfs"] for r in rows if r["rms_dbfs"] is not None]
        spl_values = [r["auto_spl_db"] for r in rows if r["auto_spl_db"] is not None]
        if not rms_values:
            return
        x = list(range(1, len(rms_values) + 1))
        self.result_plot.xy(
            x, rms_values, "Noise Floor", "Measurement #", "RMS Noise [dBFS]"
        )
        if (
            getattr(self, "open_analysis_windows", None) is not None
            and self.open_analysis_windows.isChecked()
        ):
            self.open_plot_window(self.result_plot, "Noise Floor")
        avg_rms = sum(rms_values) / len(rms_values)
        avg_spl_text = "Not calibrated"
        if spl_values:
            avg_spl_text = f"{sum(spl_values) / len(spl_values):.2f} dB SPL"
        self.hold_info = True
        self.info.setText(
            "Noise floor analysis\n"
            f"Rows used: {len(rows)}\n"
            f"Average noise RMS: {avg_rms:.2f} dBFS\n"
            f"Average estimated SPL: {avg_spl_text}\n"
            "Use a quiet room and no test tone. Capture several segments."
        )

    def analyze_thd_current(self):
        self.prepare_fresh_audio_for_button()
        data = self.get_analysis_buffer()
        if len(data) == 0:
            QMessageBox.warning(self, "No audio", "Start live or open WAV first.")
            return
        detected = analyze_array(data, self.live_samplerate)["dominant_frequency"]
        result = analyze_thd(
            data, self.live_samplerate, fundamental=float(detected), max_harmonic=5
        )
        harmonics = result["harmonics"]
        x = [h["order"] for h in harmonics]
        y = [h["db"] for h in harmonics]
        self.result_plot.xy(
            x, y, "THD / Harmonic Components", "Harmonic Order", "Magnitude [dB]", False
        )
        if (
            getattr(self, "open_analysis_windows", None) is not None
            and self.open_analysis_windows.isChecked()
        ):
            self.open_plot_window(self.result_plot, "THD / Harmonic Components")
        spread, stable_msg = self.segment_stability_report(data, self.live_samplerate)
        spread_txt = "n/a" if spread is None else f"{spread:.2f} dB"
        lines = [
            "THD auto analysis",
            f"Analysis window used: last {len(data) / self.live_samplerate:.2f} s",
            f"Auto fundamental: {result['fundamental_hz']:.2f} Hz",
            f"THD: {result['thd_percent']:.3f} %",
            f"Stability: {stable_msg} (RMS spread {spread_txt})",
            "",
            "Harmonics:",
        ]
        for h in harmonics:
            lines.append(
                f"{h['order']}x target {h['target_hz']:.2f} Hz -> peak {h['frequency']:.2f} Hz, {h['db']:.2f} dB"
            )
        self.hold_info = True
        self.info.setText("\n".join(lines))

    def analyze_thdn_current(self):
        self.prepare_fresh_audio_for_button()
        data = self.get_analysis_buffer()
        if len(data) == 0:
            QMessageBox.warning(self, "No audio", "Start live or open WAV first.")
            return
        detected = analyze_array(data, self.live_samplerate)["dominant_frequency"]
        result = analyze_thdn_sinad(
            data,
            self.live_samplerate,
            fundamental=float(detected),
            bandwidth_hz=20000.0,
            notch_width_hz=20.0,
        )
        labels = [1, 2]
        values = [result["signal_db"], result["noise_distortion_db"]]
        self.result_plot.xy(
            labels,
            values,
            "THD+N / SINAD",
            "1 = Signal, 2 = Noise + Distortion",
            "Magnitude [dB]",
            False,
        )
        if (
            getattr(self, "open_analysis_windows", None) is not None
            and self.open_analysis_windows.isChecked()
        ):
            self.open_plot_window(self.result_plot, "THD+N / SINAD")
        spread, stable_msg = self.segment_stability_report(data, self.live_samplerate)
        spread_txt = "n/a" if spread is None else f"{spread:.2f} dB"
        self.hold_info = True
        self.info.setText(
            "THD+N / SINAD auto analysis\n"
            f"Analysis window used: last {len(data) / self.live_samplerate:.2f} s\n"
            f"Auto fundamental: {result['fundamental_hz']:.2f} Hz\n"
            f"THD+N: {result['thdn_percent']:.3f} %\n"
            f"SINAD: {result['sinad_db']:.2f} dB\n"
            f"Signal level: {result['signal_db']:.2f} dB\n"
            f"Noise + distortion level: {result['noise_distortion_db']:.2f} dB\n"
            f"Stability: {stable_msg} (RMS spread {spread_txt})\n\n"
            "Estimator settings:\n"
            "Bandwidth: 20 Hz - 20 kHz\n"
            "Fundamental notch width: +/-20 Hz\n"
            "Only the selected analysis window is analyzed, not the whole live buffer."
        )

    def export_all(self):
        csv_path = (
            TABLES / f"measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        rows = self.read_measurements_from_table()
        fieldnames = [
            "test",
            "auto_input",
            "auto_frequency_hz",
            "auto_spl_db",
            "external_spl_db",
            "linearity_spl_used",
            "file",
            "rms_dbfs",
            "peak_dbfs",
            "clipping",
            "notes",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        self.wave.save(PLOTS / "live_waveform.png")
        self.fft.save(PLOTS / "live_fft.png")
        spec_plot = PlotWidget("Spectrogram export")
        spec_plot.spectrogram(self.live_buffer, self.live_samplerate)
        spec_plot.save(PLOTS / "live_spectrogram.png")
        self.result_plot.save(PLOTS / "result_plot.png")

        summary_path = (
            TABLES / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("SBM100B Analyzer Export\n")
            f.write("===========================================\n\n")
            f.write(self.info.toPlainText())
            f.write("\n\n")
            f.write(f"Rows exported: {len(rows)}\n")
            f.write(
                f"Analysis window setting: {self.analysis_window_box.value():.1f} s\n"
            )
            f.write(f"Calibration offset: {self.offset_db}\n")

        QMessageBox.information(
            self,
            "Exported",
            f"CSV:\n{csv_path}\n\nSummary:\n{summary_path}\n\nPlots:\n{PLOTS}",
        )

    def closeEvent(self, event):
        if self.stream is not None:
            self.stop_live()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
