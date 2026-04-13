"""
chart_widget.py — FreqTrade AI Bot Candlestick Chart Page
=========================================================

Architecture: Signal / Slot Bridge for Thread Safety
────────────────────────────────────────────────────
DataManager.load_async() runs its work in a background thread and invokes
the ``on_progress`` / ``on_done`` callbacks **from that worker thread**.
PyQt6 forbids touching any QWidget from a non-GUI thread.

Solution:
    1. We declare two ``pyqtSignal`` objects on the widget class:
       • ``_progress_signal(int, str)``  — carries (percent, message)
       • ``_done_signal(object)``        — carries the DataResult object

    2. We pass lightweight lambdas as callbacks to ``load_async``.
       Those lambdas do nothing except *emit* the signals above.
       ``emit()`` is thread-safe in Qt and simply posts an event to the
       main-thread event queue.

    3. In ``__init__`` we connect each signal to a normal *slot* method
       that freely updates QWidgets (progress bar, chart, labels).

This guarantees every pixel-level UI change happens on the main thread
while the heavy network / disk I/O stays off it.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.data_manager import DataManager, DataResult, ProgressEvent
from GUI.themes import get_colors


# ─── Custom Candlestick Graphics Item ────────────────────────────────────────

class CandlestickItem(pg.GraphicsObject):
    """
    Draws OHLCV candlesticks using QPainter primitives inside a
    pyqtgraph PlotWidget.  Each candle is drawn as:
      • A thin vertical line   (low  → high)  — the wick
      • A filled rectangle      (open → close) — the body

    Bullish candles (close ≥ open) are painted green; bearish ones red.
    """

    def __init__(self, ohlc_data: np.ndarray) -> None:
        """
        Args:
            ohlc_data: NumPy array with columns
                       [index, open, high, low, close].
                       ``index`` is a monotonic integer (row number)
                       used as the x-axis coordinate.
        """
        super().__init__()
        self._data = ohlc_data          # (N, 5)
        self._picture: pg.QtGui.QPicture | None = None
        self._generate_picture()

    # -- Pre-render every candle into a QPicture for fast repaints ----------

    def _generate_picture(self) -> None:
        self._picture = pg.QtGui.QPicture()
        painter = pg.QtGui.QPainter(self._picture)

        body_width = 0.6

        for row in self._data:
            idx, o, h, l, c = row

            # Colour selection
            if c >= o:
                brush = pg.mkBrush("#26A69A")   # bullish green
                pen   = pg.mkPen("#26A69A", width=1)
            else:
                brush = pg.mkBrush("#EF5350")   # bearish red
                pen   = pg.mkPen("#EF5350", width=1)

            painter.setPen(pen)
            painter.setBrush(brush)

            # Wick (high ↔ low)
            painter.drawLine(
                pg.QtCore.QPointF(idx, l),
                pg.QtCore.QPointF(idx, h),
            )

            # Body (open ↔ close)
            body_top    = max(o, c)
            body_bottom = min(o, c)
            body_height = body_top - body_bottom or (h - l) * 0.01  # avoid zero-height rect

            painter.drawRect(
                pg.QtCore.QRectF(
                    idx - body_width / 2,
                    body_bottom,
                    body_width,
                    body_height,
                )
            )

        painter.end()

    # -- pyqtgraph required overrides --------------------------------------

    def paint(self, painter, *_args) -> None:  # noqa: ANN001
        if self._picture is not None:
            self._picture.play(painter)

    def boundingRect(self) -> pg.QtCore.QRectF:
        if self._picture is None:
            return pg.QtCore.QRectF()
        return pg.QtCore.QRectF(self._picture.boundingRect())


# ─── Main Chart Widget ───────────────────────────────────────────────────────

class ChartWidget(QWidget):
    """
    Reusable candlestick chart widget.

    Can render a full-page chart with a built-in control bar (pair/timeframe
    selectors + load button) or operate in *external-control* mode where the
    caller drives data loading programmatically.

    Args:
        show_controls: If ``True`` (default) the control bar is displayed.
                       Set to ``False`` when embedding inside another page
                       that provides its own control UI.
    """

    # ── Signals (thread-safe bridge) ──────────────────────────────────────
    _progress_signal = pyqtSignal(int, str)   # (percent, status_message)
    _done_signal     = pyqtSignal(object)     # DataResult instance

    def __init__(self, parent: QWidget | None = None, show_controls: bool = True) -> None:
        super().__init__(parent)

        self._show_controls = show_controls
        self._dark = not show_controls  # embedded charts start dark by default
        self._bg_color = "#1A1A1A" if show_controls else "#FFFFFF"

        # -- Data layer ----------------------------------------------------
        self._data_manager = DataManager()

        # -- Build UI components -------------------------------------------
        self._init_ui()

        # -- Wire signals / slots ------------------------------------------
        self._progress_signal.connect(self._on_progress)
        self._done_signal.connect(self._on_data_loaded)

        if show_controls:
            self._btn_load.clicked.connect(self._start_loading)

    # ── UI Construction ───────────────────────────────────────────────────

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        # -- Control bar (optional) ----------------------------------------
        if self._show_controls:
            control_bar = QHBoxLayout()
            control_bar.setSpacing(10)

            self._combo_pair = QComboBox()
            self._combo_pair.addItems(DataManager.get_popular_pairs())
            self._combo_pair.setMinimumWidth(140)

            self._combo_tf = QComboBox()
            for key, label in DataManager.get_available_timeframes().items():
                self._combo_tf.addItem(f"{key}  ({label})", userData=key)
            idx_1h = self._combo_tf.findData("1h")
            if idx_1h >= 0:
                self._combo_tf.setCurrentIndex(idx_1h)

            self._btn_load = QPushButton("Load Data")
            self._btn_load.setFixedWidth(120)

            self._progress_bar = QProgressBar()
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
            self._progress_bar.setTextVisible(True)
            self._progress_bar.setFixedHeight(22)
            self._progress_bar.hide()

            self._status_label = QLabel("")
            self._status_label.setStyleSheet("color: #9E9E9E; font-size: 12px;")

            control_bar.addWidget(QLabel("Pair:"))
            control_bar.addWidget(self._combo_pair)
            control_bar.addWidget(QLabel("Timeframe:"))
            control_bar.addWidget(self._combo_tf)
            control_bar.addWidget(self._btn_load)
            control_bar.addWidget(self._progress_bar)
            control_bar.addWidget(self._status_label)
            control_bar.addStretch()

            root_layout.addLayout(control_bar)

        # -- Candlestick chart (pyqtgraph) ---------------------------------
        pg.setConfigOptions(antialias=True)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(self._bg_color)
        grid_alpha = 0.15 if self._show_controls else 0.1
        self._plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)
        label_color = "#9E9E9E" if self._show_controls else "#BBBBBB"
        self._plot_widget.setLabel("left", "Price", color=label_color)
        self._plot_widget.setLabel("bottom", "Bar Index", color=label_color)

        # Enable mouse panning and scroll-wheel zooming (pyqtgraph defaults)
        self._plot_widget.setMouseEnabled(x=True, y=True)

        # Add crosshair for better UX
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False,
                                            pen=pg.mkPen("#555555", width=0.8))
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False,
                                            pen=pg.mkPen("#555555", width=0.8))
        self._plot_widget.addItem(self._crosshair_v, ignoreBounds=True)
        self._plot_widget.addItem(self._crosshair_h, ignoreBounds=True)
        self._plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        root_layout.addWidget(self._plot_widget)

        # -- Styling --------------------------------------------------
        self._apply_styles()

    def _apply_styles(self) -> None:
        """Apply theme-aware stylesheet to this widget's controls."""
        C = get_colors(self._dark)
        bg = C['CARD_BG'] if not self._dark else '#121212'
        self.setStyleSheet(f"""
            ChartWidget {{
                background-color: {bg};
            }}
            QLabel {{
                color: {C['TEXT']};
                font-family: "Segoe UI";
                font-size: 13px;
            }}
            QComboBox {{
                background-color: {C['INPUT_BG']};
                color: {C['TEXT']};
                border: 1px solid {C['INPUT_BORDER']};
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 13px;
                min-width: 120px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {C['INPUT_BG']};
                color: {C['TEXT']};
                selection-background-color: {C['BLUE']};
                border: 1px solid {C['INPUT_BORDER']};
            }}
            QPushButton {{
                background-color: {C['BLUE']};
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: {C['BLUE_HOVER']}; }}
            QPushButton:disabled {{ background-color: {C['DIVIDER']}; color: {C['SUBTEXT']}; }}
            QProgressBar {{
                background-color: {C['INPUT_BG']};
                border: 1px solid {C['INPUT_BORDER']};
                border-radius: 4px;
                text-align: center;
                color: {C['TEXT']};
                font-size: 11px;
            }}
            QProgressBar::chunk {{
                background-color: {C['BLUE']};
                border-radius: 3px;
            }}
        """)

    def apply_theme(self, dark: bool) -> None:
        """Switch chart widget between light and dark theme."""
        self._dark = dark
        C = get_colors(dark)
        self._bg_color = C["CHART_BG"]
        self._plot_widget.setBackground(self._bg_color)
        # Update axis/grid colours
        label_color = C["SUBTEXT"]
        self._plot_widget.setLabel("left",   "Price",     color=label_color)
        self._plot_widget.setLabel("bottom", "Bar Index", color=label_color)
        self._apply_styles()

    # ── Loading Logic ─────────────────────────────────────────────────────

    def _start_loading(self) -> None:
        """Triggered by the 'Load Data' button click (main thread)."""
        pair = self._combo_pair.currentText()
        timeframe = self._combo_tf.currentData()
        self.load_pair(pair, timeframe)

    def load_pair(self, pair: str, timeframe: str, days: int = 365) -> None:
        """Public API: load data for a given pair/timeframe (can be called externally)."""
        if self._show_controls:
            self._btn_load.setEnabled(False)
            self._progress_bar.setValue(0)
            self._progress_bar.show()
            self._status_label.setText("Loading…")

        self._data_manager.load_async(
            pair=pair,
            timeframe=timeframe,
            days=days,
            on_progress=self._bridge_progress,
            on_done=self._bridge_done,
        )

    # ── Thread → Signal Bridge Callbacks ──────────────────────────────────
    # These tiny functions run on the WORKER THREAD.  Their only job is to
    # fire the corresponding signal so that the real work happens safely on
    # the GUI thread via the connected slot.

    def _bridge_progress(self, event: ProgressEvent) -> None:
        """Called on the worker thread — just emits the signal."""
        self._progress_signal.emit(event.percent, event.message)

    def _bridge_done(self, result: DataResult) -> None:
        """Called on the worker thread — just emits the signal."""
        self._done_signal.emit(result)

    # ── Slots (run on the GUI thread) ─────────────────────────────────────

    def _on_progress(self, percent: int, message: str) -> None:
        """Slot connected to ``_progress_signal``.  Safe to update widgets."""
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)

    def _on_data_loaded(self, result: DataResult) -> None:
        if self._show_controls:
            self._btn_load.setEnabled(True)

        if result.success:
            if self._show_controls:
                self._progress_bar.setValue(100)
                self._status_label.setText(
                    f"{result.pair} — {result.bars} bars loaded "
                    f"({result.source}, {result.duration_sec:.1f}s)"
                )
            self._render_candles(result.dataframe)
        else:
            if self._show_controls:
                self._progress_bar.setValue(0)
                self._progress_bar.hide()
                self._status_label.setText(f"Error: {result.user_message}")

    # ── Chart Rendering ───────────────────────────────────────────────────

    def _render_candles(self, df) -> None:
        """
        Clear the plot and draw a new CandlestickItem from the DataFrame.

        The DataFrame is expected to have columns: open, high, low, close
        (as produced by DataManager).  We use integer row indices as the
        x-axis to keep spacing uniform regardless of time gaps.
        """
        plot_item = self._plot_widget.getPlotItem()
        plot_item.clear()

        # Re-add crosshairs after clear
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False,
                                            pen=pg.mkPen("#555555", width=0.8))
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False,
                                            pen=pg.mkPen("#555555", width=0.8))
        plot_item.addItem(self._crosshair_v, ignoreBounds=True)
        plot_item.addItem(self._crosshair_h, ignoreBounds=True)

        if df is None or df.empty:
            return

        # Build the (index, open, high, low, close) array
        ohlc = np.column_stack([
            np.arange(len(df)),
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
        ])

        candles = CandlestickItem(ohlc)
        plot_item.addItem(candles)

        # Add a translucent volume bar overlay at the bottom
        self._draw_volume_bars(plot_item, df)

        # Auto-range to fit all data, then let user pan/zoom freely
        plot_item.enableAutoRange()

    def _draw_volume_bars(self, plot_item, df) -> None:
        """Draw semi-transparent volume bars anchored to the price y-axis bottom."""
        if "volume" not in df.columns:
            return

        volumes = df["volume"].values.astype(float)
        if volumes.max() == 0:
            return

        # Scale volume to occupy roughly 20% of the price range
        price_range = df["high"].max() - df["low"].min()
        if price_range == 0:
            return
        scale_factor = (price_range * 0.20) / volumes.max()
        scaled = volumes * scale_factor
        base = df["low"].min()

        colors = [
            pg.mkBrush("#26A69A50") if c >= o else pg.mkBrush("#EF535050")
            for o, c in zip(df["open"].values, df["close"].values)
        ]

        bar = pg.BarGraphItem(
            x=np.arange(len(df)),
            height=scaled,
            width=0.6,
            y0=base,
            brushes=colors,
            pen=pg.mkPen(None),
        )
        plot_item.addItem(bar)

    # ── Crosshair ─────────────────────────────────────────────────────────

    def _on_mouse_moved(self, pos) -> None:
        """Update crosshair lines to follow the mouse inside the plot."""
        vb = self._plot_widget.getPlotItem().vb
        if self._plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            self._crosshair_v.setPos(mouse_point.x())
            self._crosshair_h.setPos(mouse_point.y())
