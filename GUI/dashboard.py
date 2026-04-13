"""
dashboard.py — Live Dashboard Page
====================================

Redesigned to match the FreqTrade Pro mockup:
  • 4 metric cards (balance, P/L, win rate, open trades)
  • Performance Overview area chart (real 30-day daily P&L from API)
  • Open Positions panel (real open trades from the bot; right column)
  • Recent Trade History table (bottom)

Updated via update_snapshot() called from the TradeManager signal bridge.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from GUI.components.metric_cards import MetricCard
from GUI.components.trade_table import TradeTable
from GUI.themes import get_colors
from core.trade_manager import DashboardSnapshot, OverallStatus

# -- Default palette (overridden by apply_theme) ---
_C = get_colors(dark=False)


def _card_style(C: dict) -> str:
    return f"""
        QFrame {{
            background-color: {C['CARD_BG']};
            border-radius: 12px;
        }}
    """


class _PositionRow(QWidget):
    def __init__(self, pair, side, open_rate, profit_pct, stake, C, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(2)
        lbl_pair = QLabel(pair)
        lbl_pair.setStyleSheet(f"color: {C['TEXT']}; font-size: 13px; font-weight: 600;")
        side_color = C["GREEN"] if side in ("LONG", "BUY") else C["RED"]
        lbl_side = QLabel(side)
        lbl_side.setStyleSheet(f"color: {side_color}; font-size: 11px; font-weight: 600;")
        left.addWidget(lbl_pair)
        left.addWidget(lbl_side)
        layout.addLayout(left)
        layout.addStretch()
        right = QVBoxLayout()
        right.setSpacing(2)
        right.setAlignment(Qt.AlignmentFlag.AlignRight)
        lbl_price = QLabel(f"")
        lbl_price.setAlignment(Qt.AlignmentFlag.AlignRight)
        lbl_price.setStyleSheet(f"color: {C['TEXT']}; font-size: 12px;")
        pl_color = C["GREEN"] if profit_pct >= 0 else C["RED"]
        lbl_pl = QLabel(f"{profit_pct:+.2f}%")
        lbl_pl.setAlignment(Qt.AlignmentFlag.AlignRight)
        lbl_pl.setStyleSheet(f"color: {pl_color}; font-size: 12px; font-weight: 600;")
        right.addWidget(lbl_price)
        right.addWidget(lbl_pl)
        layout.addLayout(right)


class DashboardPage(QWidget):
    _snapshot_signal = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dark = False
        self.setStyleSheet(f"background-color: {_C['PAGE_BG']};")
        self._snapshot_signal.connect(self.update_snapshot)
        self._init_ui()

    def _init_ui(self) -> None:
        C = get_colors(self._dark)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._container = QWidget()
        self._container.setStyleSheet(f"background-color: {C['PAGE_BG']};")
        scroll.setWidget(self._container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(20)
        # title row
        title_row = QHBoxLayout()
        self._page_title = QLabel("Dashboard")
        self._page_title.setStyleSheet(f"color: {C['TEXT']}; font-size: 22px; font-weight: 700;")
        title_row.addWidget(self._page_title)
        title_row.addStretch()
        self._status_badge = QLabel("● Disconnected")
        self._status_badge.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 12px; font-weight: 600;")
        title_row.addWidget(self._status_badge)
        layout.addLayout(title_row)
        # metric cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)
        self.balance_card     = MetricCard("Total Balance",     ".00", "--",                    icon="fa6s.wallet")
        self.pnl_card         = MetricCard("Total Profit/Loss", ".00", "Awaiting data...",      icon="fa6s.chart-line")
        self.winrate_card     = MetricCard("Win Rate",          "--",    "No closed trades yet",  icon="fa6s.percent")
        self.open_trades_card = MetricCard("Open Trades",       "0",     "Bot not connected",     icon="fa6s.arrow-trend-up")
        for card in (self.balance_card, self.pnl_card, self.winrate_card, self.open_trades_card):
            cards_row.addWidget(card)
        layout.addLayout(cards_row)
        # mid row
        mid_row = QHBoxLayout()
        mid_row.setSpacing(16)
        # chart card
        self._chart_card = QFrame()
        self._chart_card.setStyleSheet(_card_style(C))
        chart_layout = QVBoxLayout(self._chart_card)
        chart_layout.setContentsMargins(20, 16, 20, 16)
        chart_layout.setSpacing(8)
        self._lbl_chart_title = QLabel("Performance Overview")
        self._lbl_chart_title.setStyleSheet(f"color: {C['TEXT']}; font-size: 15px; font-weight: 700;")
        self._lbl_chart_sub = QLabel("30-day cumulative profit")
        self._lbl_chart_sub.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 11px;")
        chart_layout.addWidget(self._lbl_chart_title)
        chart_layout.addWidget(self._lbl_chart_sub)
        pg.setConfigOptions(antialias=True)
        self._perf_plot = pg.PlotWidget()
        self._perf_plot.setBackground(C["CHART_BG"])
        self._perf_plot.setMinimumHeight(220)
        self._perf_plot.getPlotItem().hideAxis("left")
        self._perf_plot.getPlotItem().hideAxis("bottom")
        self._perf_plot.setMouseEnabled(x=False, y=False)
        self._perf_plot.showGrid(x=False, y=False)
        chart_layout.addWidget(self._perf_plot)
        self._draw_empty_chart(C)
        mid_row.addWidget(self._chart_card, stretch=3)
        # open positions card
        self._pos_card = QFrame()
        self._pos_card.setStyleSheet(_card_style(C))
        self._pos_card.setMinimumWidth(280)
        pos_layout = QVBoxLayout(self._pos_card)
        pos_layout.setContentsMargins(18, 16, 18, 16)
        pos_layout.setSpacing(0)
        self._lbl_positions = QLabel("Open Positions")
        self._lbl_positions.setStyleSheet(f"color: {C['TEXT']}; font-size: 15px; font-weight: 700;")
        pos_layout.addWidget(self._lbl_positions)
        pos_layout.addSpacing(10)
        self._positions_container = QVBoxLayout()
        self._positions_container.setSpacing(0)
        pos_layout.addLayout(self._positions_container)
        self._no_positions_lbl = QLabel("No open positions")
        self._no_positions_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_positions_lbl.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 12px; font-style: italic;")
        self._positions_container.addWidget(self._no_positions_lbl)
        pos_layout.addStretch()
        mid_row.addWidget(self._pos_card, stretch=1)
        layout.addLayout(mid_row)
        # trade table
        self._trade_table = TradeTable(title="Recent Trade History")
        self._trade_table.set_subtitle("Latest executions from active strategy")
        layout.addWidget(self._trade_table)
        layout.addStretch()

    # -- Chart --

    def _draw_empty_chart(self, C: dict) -> None:
        x = np.arange(30, dtype=float)
        y = np.zeros(30)
        self._redraw_chart(x, y, C)

    def _redraw_chart(self, x, y, C: dict) -> None:
        plot = self._perf_plot.getPlotItem()
        plot.clear()
        color = C["GREEN"]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=2)),
            pg.PlotDataItem(x, np.zeros_like(y), pen=pg.mkPen(None)),
            brush=pg.mkBrush(QColor(r, g, b, 40)),
        )
        plot.addItem(fill)
        plot.plot(x, y, pen=pg.mkPen(color, width=2))

    # -- Positions --

    def _update_positions(self, open_trades: list, C: dict) -> None:
        while self._positions_container.count():
            item = self._positions_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not open_trades:
            lbl = QLabel("No open positions")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 12px; font-style: italic;")
            self._positions_container.addWidget(lbl)
            return
        for i, trade in enumerate(open_trades):
            pair = trade.get("pair", "--")
            is_short = trade.get("is_short", False)
            side = "SHORT" if is_short else "LONG"
            open_rate = trade.get("open_rate", 0.0)
            profit_pct = trade.get("profit_pct", trade.get("profit_ratio", 0.0))
            if isinstance(profit_pct, float) and abs(profit_pct) < 1.0 and profit_pct != 0.0:
                profit_pct *= 100.0
            stake = trade.get("stake_amount", 0.0)
            row = _PositionRow(pair, side, open_rate, profit_pct, stake, C)
            self._positions_container.addWidget(row)
            if i < len(open_trades) - 1:
                div = QFrame()
                div.setFrameShape(QFrame.Shape.HLine)
                div.setStyleSheet(f"color: {C['DIVIDER']};")
                self._positions_container.addWidget(div)

    # -- Public API --

    def update_snapshot(self, snap: DashboardSnapshot) -> None:
        C = get_colors(self._dark)
        # balance
        direction = "↑" if snap.balance_change_pct >= 0 else "↓"
        self.balance_card.update_value(
            f"",
            subtitle=f"{direction} {snap.balance_change_pct:+.2f}% from start",
            positive=snap.balance_change_pct >= 0 if snap.has_data else None,
        )
        self.balance_card.set_title(
            f"Total Balance ({snap.stake_currency}" + (" · Dry Run)" if snap.dry_run else ")")
        )
        # pnl
        self.pnl_card.update_value(
            f"",
            subtitle=f"{snap.profit_all_pct:+.2f}%  all time",
            positive=snap.profit_all_coin >= 0 if snap.has_data else None,
        )
        # winrate
        wr = snap.win_rate
        total_closed = snap.winning_trades + snap.losing_trades
        self.winrate_card.update_value(
            f"{wr:.1f}%",
            subtitle=f"{total_closed} closed  ({snap.winning_trades}W / {snap.losing_trades}L)",
            positive=wr >= 50.0 if total_closed > 0 else None,
        )
        # open trades
        max_str = str(snap.max_open_trades) if snap.max_open_trades > 0 else "∞"
        self.open_trades_card.update_value(
            str(snap.open_trade_count),
            subtitle=f"Max {max_str}  •  {snap.strategy or 'N/A'}",
            positive=None,
        )
        # status badge
        if snap.open_trade_count > 0:
            self._status_badge.setText(f"● Bot Active: {snap.open_trade_count} open")
            self._status_badge.setStyleSheet(f"color: {C['GREEN']}; font-size: 12px; font-weight: 600;")
        else:
            self._status_badge.setText("● Bot Running — idle")
            self._status_badge.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 12px; font-weight: 600;")
        # chart + positions + table
        self.update_performance_chart(snap.daily_profits)
        self._update_positions(snap.open_trades, C)
        self._trade_table.update_trades(snap.open_trades, "open")

    def update_performance_chart(self, daily_profits: list) -> None:
        C = get_colors(self._dark)
        self._perf_plot.setBackground(C["CHART_BG"])
        if not daily_profits:
            self._draw_empty_chart(C)
            return
        try:
            profits = [float(d.get("abs_profit", 0.0)) for d in daily_profits]
            cumulative = np.cumsum(profits)
            x = np.arange(len(cumulative), dtype=float)
            self._redraw_chart(x, cumulative, C)
        except Exception:
            self._draw_empty_chart(C)

    def update_connection_status(self, status: OverallStatus) -> None:
        C = get_colors(self._dark)
        colours = {
            OverallStatus.DISCONNECTED: (C["RED"],    "Bot Offline"),
            OverallStatus.CONNECTING:   (C["ORANGE"], "Connecting..."),
            OverallStatus.CONNECTED:    (C["BLUE"],   "REST Connected"),
            OverallStatus.LIVE:         (C["GREEN"],  "Bot Online"),
            OverallStatus.ERROR:        (C["RED"],    "Bot Offline - check Docker"),
        }
        color, text = colours.get(status, (C["SUBTEXT"], "Unknown"))
        self._status_badge.setText(f"● {text}")
        self._status_badge.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")

    def apply_theme(self, dark: bool) -> None:
        self._dark = dark
        C = get_colors(dark)
        self.setStyleSheet(f"background-color: {C['PAGE_BG']};")
        self._container.setStyleSheet(f"background-color: {C['PAGE_BG']};")
        self._page_title.setStyleSheet(f"color: {C['TEXT']}; font-size: 22px; font-weight: 700;")
        for card in (self.balance_card, self.pnl_card, self.winrate_card, self.open_trades_card):
            card.apply_theme(dark)
        self._chart_card.setStyleSheet(_card_style(C))
        self._lbl_chart_title.setStyleSheet(f"color: {C['TEXT']}; font-size: 15px; font-weight: 700;")
        self._lbl_chart_sub.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 11px;")
        self._perf_plot.setBackground(C["CHART_BG"])
        self._pos_card.setStyleSheet(_card_style(C))
        self._lbl_positions.setStyleSheet(f"color: {C['TEXT']}; font-size: 15px; font-weight: 700;")
        self._trade_table.apply_theme(dark)
