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
import subprocess
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pyqtgraph as pg

import qtawesome as qta

from PyQt6.QtCore import Qt, pyqtSignal, QDate, QThread, pyqtSlot, QSize
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
    QVBoxLayout,
    QWidget,
)

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
    """Full backtesting page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG_PAGE};")
        self._init_ui()
        self._load_strategies()

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
        """Attempt to find the latest backtest result or run a placeholder."""
        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")

        # Try to load the most recent result from disk
        result = self._load_latest_result()
        if result:
            self._apply_result(result)
        else:
            # Show placeholder numbers
            self._apply_placeholder_result()

        self._run_btn.setEnabled(True)
        self._run_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
        self._run_btn.setText("  Run Backtest")

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
