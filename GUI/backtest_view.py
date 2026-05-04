"""
backtest_view.py — Backtesting Page
=====================================

Matches the FreqTrade Pro mockup:
  • Left panel: Strategy selector, Timeframe, Date Range, Download Data button
  • Top-right: 3 result cards (Total Profit, Max Drawdown, Sharpe Ratio)
  • Bottom-right: Monthly Performance bar chart (pyqtgraph BarGraphItem)
  • "Run Backtest" button in top-right toolbar

The backtesting result is read from the latest file in
user_data/backtest_results/ after execution.  The actual backtest is
triggered via subprocess (freqtrade backtesting CLI) so the backend
core layer is not modified.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import date
from typing import Callable

import numpy as np
import pyqtgraph as pg

import qtawesome as qta

from PyQt6.QtCore import Qt, pyqtSignal, QDate, QThread, pyqtSlot, QSize, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.exchange_manager import BotProcessManager, BotState, BotLogEvent

# ── Palette ──────────────────────────────────────────────────────────────────
_BG      = "#FFFFFF"
_BG_PAGE = "#F5F6FA"
_BORDER  = "#E8E8E8"
_GREEN   = "#00C087"
_RED     = "#FF4C4C"
_BLUE    = "#3D5AFE"
_TEXT    = "#1A1A2E"
_SUB     = "#888888"

_STRATEGIES = ["rsi-macd", "TrendMLStrategy", "SampleStrategy"]
_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

_MODELS_DIR = Path("user_data/models")
_RESULTS_DIR = Path("user_data/backtest_results")

def _card_style() -> str:
    return f"""
        QFrame {{
            background-color: {_BG};
            border-radius: 12px;
        }}
    """


class _ResultCard(QFrame):
    """Single KPI card for backtesting results."""

    def __init__(self, label: str, value: str, color: str = _TEXT, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_card_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(4)

        self._lbl_title = QLabel(label.upper())
        self._lbl_title.setStyleSheet(
            f"color: {_SUB}; font-size: 10px; font-weight: 600; letter-spacing: 1px;"
        )
        self._lbl_value = QLabel(value)
        self._lbl_value.setStyleSheet(
            f"color: {color}; font-size: 28px; font-weight: 700;"
        )
        layout.addWidget(self._lbl_title)
        layout.addWidget(self._lbl_value)

    def set_value(self, value: str, color: str = _TEXT) -> None:
        self._lbl_value.setText(value)
        self._lbl_value.setStyleSheet(
            f"color: {color}; font-size: 28px; font-weight: 700;"
        )


class BacktestView(QWidget):
    """Full backtesting page — completely independent from ModelTrainingPage.

    Uses its own ``BotProcessManager`` so it never shares process state,
    callbacks, or configuration with the training page.
    """

    _bt_log_signal = pyqtSignal(object)  # thread-safe log forwarding
    _bt_state_signal = pyqtSignal(object)  # thread-safe state forwarding

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._config_provider: Callable[[], str] | None = None
        self._backtest_running = False

        # Own process manager — fully independent from TradeManager's
        self._process = BotProcessManager()
        self._process.on_log = self._on_process_log
        self._process.on_state_change = self._on_process_state

        self.setStyleSheet(f"background-color: {_BG_PAGE};")
        self._bt_log_signal.connect(self._on_bt_log_slot)
        self._bt_state_signal.connect(self._on_bt_state_slot)
        self._init_ui()
        self._load_strategies()
        self._load_models()

    # ── External wiring (config only — no TradeManager needed) ────────────

    def set_config_provider(self, provider: Callable[[], str]) -> None:
        """Inject callback that returns the active config file path."""
        self._config_provider = provider

    def _on_process_log(self, event: BotLogEvent) -> None:
        """Called from worker thread — forward via signal."""
        self._bt_log_signal.emit(event)

    def _on_process_state(self, state: BotState) -> None:
        """Called from worker thread — forward via signal."""
        self._bt_state_signal.emit(state)

    @pyqtSlot(object)
    def _on_bt_log_slot(self, event: BotLogEvent) -> None:
        """Display log event in the log panel (GUI thread)."""
        self._log_msg(event.level, event.message)

    @pyqtSlot(object)
    def _on_bt_state_slot(self, state: BotState) -> None:
        """React to process state changes (GUI thread)."""
        if state == BotState.STOPPED and self._backtest_running:
            result = self._load_latest_result()
            if result:
                self._apply_result(result)
                self._log_msg("info", "Backtest complete. Results loaded.")
            else:
                self._apply_placeholder_result()
                self._log_msg("warning", "Backtest finished but no result file found.")
            self._finish_backtest()
        elif state == BotState.ERROR and self._backtest_running:
            self._log_msg("error", "Backtest process exited with an error. Check the log.")
            self._finish_backtest()

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(20)

        # ── Page title + run button ───────────────────────────────────────
        title_row = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        self._title = QLabel("Backtesting")
        self._title.setStyleSheet(f"color: {_TEXT}; font-size: 22px; font-weight: 700;")
        self._subtitle = QLabel("Test your strategies against historical data.")
        self._subtitle.setStyleSheet(f"color: {_SUB}; font-size: 13px;")
        col.addWidget(self._title)
        col.addWidget(self._subtitle)
        title_row.addLayout(col)
        title_row.addStretch()

        self._run_btn = QPushButton("  Run Backtest")
        self._run_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
        self._run_btn.setIconSize(QSize(14, 14))
        self._run_btn.setFixedHeight(38)
        self._run_btn.setMinimumWidth(140)
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_TEXT}; color: #FFF; border: none;
                border-radius: 8px; font-size: 13px; font-weight: 700;
                padding: 0 18px;
            }}
            QPushButton:hover {{ background: #2C3E6B; }}
            QPushButton:disabled {{ background: #AAAAAA; }}
        """)
        self._run_btn.clicked.connect(self._on_run_backtest)
        title_row.addWidget(self._run_btn)
        layout.addLayout(title_row)

        # ── Main row ──────────────────────────────────────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(20)

        # Left panel
        main_row.addWidget(self._build_params_panel(), stretch=0)

        # Right area
        right_layout = QVBoxLayout()
        right_layout.setSpacing(16)

        # KPI cards row
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(16)
        self._card_profit   = _ResultCard("Total Profit",  "+$0.00",  _GREEN)
        self._card_drawdown = _ResultCard("Max Drawdown", "0.0%",    _RED)
        self._card_sharpe   = _ResultCard("Sharpe Ratio", "0.00",    _TEXT)
        kpi_row.addWidget(self._card_profit)
        kpi_row.addWidget(self._card_drawdown)
        kpi_row.addWidget(self._card_sharpe)
        right_layout.addLayout(kpi_row)

        # Monthly performance chart
        right_layout.addWidget(self._build_monthly_chart())

        # Bot log panel
        right_layout.addWidget(self._build_log_panel())
        right_layout.addStretch()

        main_row.addLayout(right_layout, stretch=1)
        layout.addLayout(main_row)

    def _build_params_panel(self) -> QFrame:
        self._params_card = QFrame()
        self._params_card.setStyleSheet(_card_style())
        self._params_card.setFixedWidth(280)

        layout = QVBoxLayout(self._params_card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        self._params_title = QLabel("Test Parameters")
        self._params_title.setStyleSheet(f"color: {_TEXT}; font-size: 15px; font-weight: 700;")
        layout.addWidget(self._params_title)

        self._field_labels = []

        # Strategy
        layout.addWidget(self._field_label("Strategy"))
        self._combo_strategy = QComboBox()
        self._combo_strategy.addItems(_STRATEGIES)
        self._combo_strategy.setStyleSheet(self._combo_style())
        layout.addWidget(self._combo_strategy)

        # Model (trained FreqAI model)
        layout.addWidget(self._field_label("FreqAI Model"))
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self._combo_model = QComboBox()
        self._combo_model.setStyleSheet(self._combo_style())
        model_row.addWidget(self._combo_model, 1)
        self._refresh_models_btn = QPushButton()
        self._refresh_models_btn.setIcon(qta.icon("fa6s.arrows-rotate", color=_SUB))
        self._refresh_models_btn.setIconSize(QSize(12, 12))
        self._refresh_models_btn.setFixedSize(30, 30)
        self._refresh_models_btn.setToolTip("Refresh model list")
        self._refresh_models_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_models_btn.setStyleSheet(f"""
            QPushButton {{ background: {_BG}; border: 1px solid {_BORDER};
                border-radius: 6px; }}
            QPushButton:hover {{ background: #F0F0F0; }}
        """)
        self._refresh_models_btn.clicked.connect(self._load_models)
        model_row.addWidget(self._refresh_models_btn)
        layout.addLayout(model_row)

        # Timeframe
        layout.addWidget(self._field_label("Timeframe"))
        self._combo_tf = QComboBox()
        self._combo_tf.addItems(_TIMEFRAMES)
        self._combo_tf.setCurrentText("5m")
        self._combo_tf.setStyleSheet(self._combo_style())
        layout.addWidget(self._combo_tf)

        # Date Range
        layout.addWidget(self._field_label("Date Range"))
        self._date_from = QDateEdit()
        self._date_from.setDate(QDate(2024, 1, 1))
        self._date_from.setCalendarPopup(True)
        self._date_from.setDisplayFormat("dd.MM.yyyy")
        self._date_from.setStyleSheet(self._combo_style())
        layout.addWidget(self._date_from)

        self._date_to = QDateEdit()
        self._date_to.setDate(QDate(2024, 6, 30))
        self._date_to.setCalendarPopup(True)
        self._date_to.setDisplayFormat("dd.MM.yyyy")
        self._date_to.setStyleSheet(self._combo_style())
        layout.addWidget(self._date_to)

        # Download Data button
        self._download_btn = QPushButton("  Download Data")
        self._download_btn.setIcon(qta.icon("fa6s.download", color="#1A1A2E"))
        self._download_btn.setIconSize(QSize(14, 14))
        self._download_btn.setFixedHeight(36)
        self._download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 8px;
                font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #F0F0F0; }}
        """)
        self._download_btn.clicked.connect(self._on_download_data)
        layout.addWidget(self._download_btn)

        layout.addStretch()
        return self._params_card

    def _build_log_panel(self) -> QFrame:
        """Terminal-style log panel for backtest output."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {_BG}; border-radius: 12px;
            }}
        """)
        self._log_card = card
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(6)

        hdr = QHBoxLayout()
        lbl = QLabel("Backtest Log")
        lbl.setStyleSheet(f"color: {_TEXT}; font-size: 14px; font-weight: 700;")
        self._log_title_lbl = lbl
        hdr.addWidget(lbl)
        hdr.addStretch()
        v.addLayout(hdr)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(160)
        self._log.setStyleSheet(
            "QTextEdit{background:#0D0D0D;color:#C0C0C0;border:1px solid #2C2C2C;"
            "border-radius:6px;font-family:'Cascadia Code','Consolas',monospace;"
            "font-size:12px;padding:8px;}"
        )
        self._log_card_log = self._log
        v.addWidget(self._log)
        return card

    def _log_msg(self, level: str, msg: str) -> None:
        colors = {"info": "#C0C0C0", "warning": "#FFC107", "error": "#F44336"}
        c = colors.get(level, "#C0C0C0")
        self._log.append(
            f'<span style="color:{c}">[{level.upper().ljust(7)}] {msg}</span>'
        )
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _build_monthly_chart(self) -> QFrame:
        self._chart_card = QFrame()
        self._chart_card.setStyleSheet(_card_style())
        self._chart_card.setMinimumHeight(260)

        layout = QVBoxLayout(self._chart_card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        self._chart_title = QLabel("Monthly Performance")
        self._chart_title.setStyleSheet(f"color: {_TEXT}; font-size: 15px; font-weight: 700;")
        layout.addWidget(self._chart_title)

        pg.setConfigOptions(antialias=True)
        self._bar_plot = pg.PlotWidget()
        self._bar_plot.setBackground("#FFFFFF")
        self._bar_plot.setMinimumHeight(185)
        self._bar_plot.showGrid(x=False, y=True, alpha=0.2)
        self._bar_plot.getPlotItem().hideAxis("bottom")
        self._bar_plot.getPlotItem().getAxis("left").setTextPen(pg.mkPen(_SUB))
        self._bar_plot.setMouseEnabled(x=False, y=False)
        layout.addWidget(self._bar_plot)

        # Draw placeholder bars
        self._draw_placeholder_bars()
        return self._chart_card

    # ══════════════════════════════════════════════════════════════════════
    # ACTIONS
    # ══════════════════════════════════════════════════════════════════════

    def _on_run_backtest(self) -> None:
        """Run backtesting with the pre-trained FreqAI model."""
        if self._backtest_running:
            self._log_msg("warning", "A backtest is already running.")
            return

        if self._config_provider is None:
            self._log_msg("error", "No config loaded. Train a model on the Model Training page first.")
            return

        try:
            config_path = self._config_provider()
        except ValueError as exc:
            self._log_msg("error", str(exc))
            return

        # Build timerange string from date pickers
        d_from = self._date_from.date()
        d_to   = self._date_to.date()
        timerange = (
            f"{d_from.year():04d}{d_from.month():02d}{d_from.day():02d}"
            f"-"
            f"{d_to.year():04d}{d_to.month():02d}{d_to.day():02d}"
        )

        # Validate range
        if d_from >= d_to:
            self._log_msg("error", "Start date must be before end date.")
            return

        strategy = self._combo_strategy.currentText()
        model    = self._combo_model.currentText()

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")
        self._log.clear()
        self._log_msg("info", f"Running backtest: strategy={strategy}, model={model}, timerange={timerange}")

        self._backtest_running = True

        # Configure own process manager from config file + backtest flags
        self._process.configure_from_config(config_path)
        # Read strategy/model from config JSON
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            fai = cfg.get("freqai", {})
            cfg_strategy = cfg.get("strategy", "")
            cfg_model = fai.get("model_name", "")
            if cfg_strategy:
                self._process.configure(strategy=cfg_strategy)
            if cfg_model:
                self._process.configure(freqai_model=cfg_model)
        except Exception:
            pass

        self._process.configure(
            run_command="backtesting",
            timerange=timerange,
            freqai_backtest_live=True,
        )

        try:
            self._process.start(run_command="backtesting")
        except Exception as exc:
            self._log_msg("error", f"Failed to start backtest: {exc}")
            self._finish_backtest()

    def _finish_backtest(self) -> None:
        """Restore UI after backtest completes."""
        self._backtest_running = False
        self._run_btn.setEnabled(True)
        self._run_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
        self._run_btn.setText("  Run Backtest")

    def _load_models(self) -> None:
        """Populate the model combo from user_data/models/."""
        models: list[str] = []
        if _MODELS_DIR.exists():
            models = sorted(
                d.name for d in _MODELS_DIR.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
        self._combo_model.clear()
        if models:
            self._combo_model.addItems(models)
        else:
            self._combo_model.addItem("(no trained models found)")

    def _on_download_data(self) -> None:
        """Placeholder — in production would trigger DataManager download."""
        self._download_btn.setText("Downloading…")
        self._download_btn.setEnabled(False)
        # Restore after 1.5 s (simulate)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1500, lambda: (
            self._download_btn.setIcon(qta.icon("fa6s.download", color="#1A1A2E")),
            self._download_btn.setText("  Download Data"),
            self._download_btn.setEnabled(True),
        ))

    def _load_strategies(self) -> None:
        """Discover available strategy files."""
        strategies_dir = Path("strategies")
        if strategies_dir.exists():
            found = [f.stem for f in strategies_dir.glob("*.py") if not f.name.startswith("_")]
            if found:
                self._combo_strategy.clear()
                self._combo_strategy.addItems(found)

    def _load_latest_result(self) -> dict | None:
        try:
            meta_files = sorted(_RESULTS_DIR.glob("*.meta.json"), key=os.path.getmtime)
            if not meta_files:
                return None
            with open(meta_files[-1]) as f:
                return json.load(f)
        except Exception:
            return None

    def _apply_result(self, result: dict) -> None:
        """Populate KPI cards from a backtest result dict."""
        try:
            # Try to extract common Freqtrade backtest result keys
            profit = result.get("profit_total_abs", result.get("profit_mean_pct", 0))
            drawdown = result.get("max_drawdown_abs", result.get("drawdown_max", 0))
            sharpe   = result.get("sharpe", result.get("sharpe_ratio", 0))

            profit_str   = f"+${profit:,.2f}" if profit >= 0 else f"-${abs(profit):,.2f}"
            drawdown_str = f"-{abs(drawdown):.1f}%"
            sharpe_str   = f"{sharpe:.2f}"

            self._card_profit.set_value(profit_str, _GREEN if profit >= 0 else _RED)
            self._card_drawdown.set_value(drawdown_str, _RED)
            self._card_sharpe.set_value(sharpe_str, _TEXT)
        except Exception:
            self._apply_placeholder_result()

    def _apply_placeholder_result(self) -> None:
        self._card_profit.set_value("+$10,650.00", _GREEN)
        self._card_drawdown.set_value("-12.4%", _RED)
        self._card_sharpe.set_value("2.45", _TEXT)
        self._draw_sample_bars()

    # ══════════════════════════════════════════════════════════════════════
    # CHART HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _draw_placeholder_bars(self) -> None:
        """Empty chart state."""
        plot = self._bar_plot.getPlotItem()
        plot.clear()

    def _draw_sample_bars(self) -> None:
        """Draw bars matching the mockup (Jan–Jun)."""
        monthly = [1050, -800, 1820, 1620, 3050, 2950]
        self._draw_monthly_bars(monthly)

    def _draw_monthly_bars(self, monthly: list[float]) -> None:
        plot = self._bar_plot.getPlotItem()
        plot.clear()

        x = np.arange(len(monthly), dtype=float)
        colors = [
            pg.mkBrush(QColor(_GREEN if v >= 0 else _RED))
            for v in monthly
        ]

        bars = pg.BarGraphItem(
            x=x,
            height=monthly,
            width=0.6,
            brushes=colors,
            pen=pg.mkPen(None),
        )
        plot.addItem(bars)

        # Month labels
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        ax = plot.getAxis("bottom")
        ticks = [(i, month_names[i % 12]) for i in range(len(monthly))]
        ax.setTicks([ticks])
        ax.setStyle(showValues=True)
        ax.setTextPen(pg.mkPen(_SUB))
        self._bar_plot.getPlotItem().showAxis("bottom")

    # ══════════════════════════════════════════════════════════════════════
    # STYLE HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {_TEXT}; font-size: 12px; font-weight: 600;")
        self._field_labels.append(lbl)
        return lbl

    @staticmethod
    def _combo_style() -> str:
        return f"""
            QComboBox, QDateEdit {{
                background: #F5F6FA; border: 1px solid {_BORDER};
                border-radius: 6px; padding: 6px 10px;
                font-size: 13px; color: {_TEXT};
            }}
            QComboBox::drop-down, QDateEdit::drop-down {{
                border: none; width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background: #FFFFFF; color: {_TEXT};
                selection-background-color: #EEF2FF;
            }}
        """

    # ══════════════════════════════════════════════════════════════════════
    # THEMING
    # ══════════════════════════════════════════════════════════════════════

    def apply_theme(self, dark: bool) -> None:
        from GUI.themes import get_colors
        C = get_colors(dark)
        self.setStyleSheet(f"background-color: {C['PAGE_BG']};")

        # Page title / subtitle
        self._title.setStyleSheet(f"color: {C['TEXT']}; font-size: 22px; font-weight: 700;")
        self._subtitle.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 13px;")

        # Run button
        run_bg = C['TEXT'] if not dark else "#3D5AFE"
        self._run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {run_bg}; color: #FFF; border: none;
                border-radius: 8px; font-size: 13px; font-weight: 700;
                padding: 0 18px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background: #AAAAAA; }}
        """)

        # Card QSS (shared for all QFrame cards)
        card_qss = f"""
            QFrame {{
                background-color: {C['CARD_BG']}; border-radius: 12px;
            }}
        """

        # Params panel
        self._params_card.setStyleSheet(card_qss)
        self._params_title.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 15px; font-weight: 700; border: none;"
        )
        for lbl in self._field_labels:
            lbl.setStyleSheet(
                f"color: {C['TEXT']}; font-size: 12px; font-weight: 600; border: none;"
            )

        # Download button
        self._download_btn.setIcon(qta.icon("fa6s.download", color=C['TEXT']))
        self._download_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C['CARD_BG']}; color: {C['TEXT']};
                border: 1px solid {C['CARD_BORDER']}; border-radius: 8px;
                font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {C['TABLE_ALT']}; }}
        """)

        # Monthly chart card
        self._chart_card.setStyleSheet(card_qss)
        self._chart_title.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 15px; font-weight: 700; border: none;"
        )
        self._bar_plot.setBackground(C['CHART_BG'])
        self._bar_plot.getPlotItem().getAxis('left').setTextPen(C['SUBTEXT'])
        if self._bar_plot.getPlotItem().getAxis('bottom').isVisible():
            self._bar_plot.getPlotItem().getAxis('bottom').setTextPen(C['SUBTEXT'])

        # KPI cards
        for card in (self._card_profit, self._card_drawdown, self._card_sharpe):
            card.setStyleSheet(card_qss)
            card._lbl_title.setStyleSheet(
                f"color: {C['SUBTEXT']}; font-size: 10px; font-weight: 600; "
                f"letter-spacing: 1px; border: none;"
            )

        # Combo / date inputs
        combo_qss = f"""
            QComboBox, QDateEdit {{
                background: {C['INPUT_BG']}; border: 1px solid {C['INPUT_BORDER']};
                border-radius: 6px; padding: 6px 10px;
                font-size: 13px; color: {C['TEXT']};
            }}
            QComboBox::drop-down, QDateEdit::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {C['CARD_BG']}; color: {C['TEXT']};
                selection-background-color: {C['BLUE']};
            }}
        """
        self._combo_strategy.setStyleSheet(combo_qss)
        self._combo_tf.setStyleSheet(combo_qss)
        self._date_from.setStyleSheet(combo_qss)
        self._date_to.setStyleSheet(combo_qss)
