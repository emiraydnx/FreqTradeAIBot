"""
trade_table.py — Open / Closed Trades Table Widget
====================================================

A styled QTableWidget that displays trade data from the Freqtrade REST API.
Supports two modes:

    • **Open trades** — columns: Pair, Side, Profit %, Stake, Duration, Actions
    • **Closed trades** — columns: Pair, Side, Profit %, Stake, Duration, Exit Reason

Call ``update_trades(rows)`` with a list of trade dicts from the API.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


from GUI.themes import get_colors


# ── Colours ──────────────────────────────────────────────────────────────────
_GREEN = "#00C087"
_RED   = "#FF4C4C"
_WHITE = "#1A1A2E"
_GREY  = "#888888"

OPEN_COLUMNS   = ["Pair", "Type", "Price", "Amount", "Status", "Profit/Loss"]
CLOSED_COLUMNS = ["Pair", "Type", "Price", "Amount", "Status", "Profit/Loss"]


class TradeTable(QFrame):
    """
    Styled trade table widget.

    Signals:
        close_trade_requested(trade_id):  Emitted when user clicks the
            close button on an open trade row.
    """

    close_trade_requested = pyqtSignal(int)

    def __init__(self, title: str = "Recent Trade History", parent=None) -> None:
        super().__init__(parent)
        self._dark = False

        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E8E8E8;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # ── Header row ────────────────────────────────────────────────────
        header_layout = QHBoxLayout()
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            "color: #1A1A2E; font-size: 15px; font-weight: 700; border: none;"
        )
        self._subtitle_label = QLabel("")
        self._subtitle_label.setStyleSheet(
            "color: #888888; font-size: 11px; border: none;"
        )
        self._view_all_btn = QPushButton("View All")
        self._view_all_btn.setStyleSheet("""
            QPushButton {
                color: #3D5AFE; background: transparent; border: none;
                font-size: 12px; font-weight: 600;
            }
            QPushButton:hover { color: #536DFE; }
        """)
        header_layout.addWidget(self._title_label)
        header_layout.addSpacing(8)
        header_layout.addWidget(self._subtitle_label)
        header_layout.addStretch()
        header_layout.addWidget(self._view_all_btn)
        layout.addLayout(header_layout)

        # ── Table ─────────────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(len(OPEN_COLUMNS))
        self._table.setHorizontalHeaderLabels(OPEN_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setStyleSheet(self._table_style(self._dark))
        self._table.setMinimumHeight(160)
        layout.addWidget(self._table)

        # ── Empty state ───────────────────────────────────────────────────
        self._empty_label = QLabel("No trades to display")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #757575; font-size: 14px; font-style: italic; border: none;"
        )
        layout.addWidget(self._empty_label)
        self._empty_label.setVisible(True)
        self._table.setVisible(False)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def update_trades(self, trades: list[dict[str, Any]], mode: str = "open") -> None:
        """
        Populate the table with trade data.

        Args:
            trades: List of trade dicts from the Freqtrade API.
            mode:   ``"open"`` or ``"closed"`` — controls which columns are shown.
        """
        columns = OPEN_COLUMNS if mode == "open" else CLOSED_COLUMNS
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)

        self._table.setRowCount(len(trades))

        if not trades:
            self._table.setVisible(False)
            self._empty_label.setVisible(True)
            return

        self._table.setVisible(True)
        self._empty_label.setVisible(False)

        for row, trade in enumerate(trades):
            if mode == "open":
                self._fill_open_row(row, trade)
            else:
                self._fill_closed_row(row, trade)

    def clear(self) -> None:
        self._table.setRowCount(0)
        self._table.setVisible(False)
        self._empty_label.setVisible(True)

    # ══════════════════════════════════════════════════════════════════════
    # ROW BUILDERS
    # ══════════════════════════════════════════════════════════════════════

    def set_subtitle(self, text: str) -> None:
        self._subtitle_label.setText(text)

    def apply_theme(self, dark: bool) -> None:
        """Switch between light and dark theme."""
        self._dark = dark
        C = get_colors(dark)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {C['CARD_BG']};
                border-radius: 12px;
                border: 1px solid {C['CARD_BORDER']};
            }}
        """)
        self._title_label.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 15px; font-weight: 700; border: none;"
        )
        self._subtitle_label.setStyleSheet(
            f"color: {C['SUBTEXT']}; font-size: 11px; border: none;"
        )
        self._table.setStyleSheet(self._table_style(dark))
        self._empty_label.setStyleSheet(
            f"color: {C['SUBTEXT']}; font-size: 14px; font-style: italic; border: none;"
        )

    def _fill_open_row(self, row: int, trade: dict) -> None:
        pair = trade.get("pair", "—")
        side = trade.get("trade_direction", trade.get("side", "long")).upper()
        profit_pct = trade.get("profit_pct", trade.get("profit_ratio", 0.0))
        if isinstance(profit_pct, (int, float)):
            if -1 < profit_pct < 1 and profit_pct != 0:
                profit_pct *= 100
        stake = trade.get("stake_amount", 0.0)
        open_rate = trade.get("open_rate", 0.0)
        is_open = trade.get("is_open", True)
        status = "OPEN" if is_open else "CLOSED"
        profit_str = f"{profit_pct:+.2f}%"

        # Pair
        self._table.setItem(row, 0, self._make_item(pair, self._text_color()))

        # Type badge (BUY=green, SELL=red)
        type_item = QTableWidgetItem(side)
        type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        type_item.setForeground(QColor(_GREEN if side == "BUY" or side == "LONG" else _RED))
        self._table.setItem(row, 1, type_item)

        # Price
        self._table.setItem(row, 2, self._make_item(f"${open_rate:,.2f}", self._text_color()))

        # Amount
        self._table.setItem(row, 3, self._make_item(f"{stake:.2f}", self._text_color()))

        # Status
        status_item = QTableWidgetItem(f"● {status}")
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        color = _GREEN if status == "OPEN" else _GREY
        status_item.setForeground(QColor(color))
        self._table.setItem(row, 4, status_item)

        # Profit/Loss
        pl_item = QTableWidgetItem(profit_str)
        pl_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_item.setForeground(QColor(_GREEN if profit_pct >= 0 else _RED))
        self._table.setItem(row, 5, pl_item)

    def _fill_closed_row(self, row: int, trade: dict) -> None:
        pair = trade.get("pair", "—")
        side = trade.get("trade_direction", trade.get("side", "long")).upper()
        profit_pct = trade.get("profit_pct", trade.get("profit_ratio", 0.0))
        if isinstance(profit_pct, (int, float)):
            if -1 < profit_pct < 1 and profit_pct != 0:
                profit_pct *= 100
        close_rate = trade.get("close_rate", trade.get("open_rate", 0.0))
        stake = trade.get("stake_amount", 0.0)
        profit_str = f"{profit_pct:+.2f}%"

        self._table.setItem(row, 0, self._make_item(pair, self._text_color()))

        type_item = QTableWidgetItem(side)
        type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        type_item.setForeground(QColor(_GREEN if side in ("BUY", "LONG") else _RED))
        self._table.setItem(row, 1, type_item)

        self._table.setItem(row, 2, self._make_item(f"${close_rate:,.2f}", self._text_color()))
        self._table.setItem(row, 3, self._make_item(f"{stake:.2f}", self._text_color()))

        status_item = QTableWidgetItem("● CLOSED")
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        status_item.setForeground(QColor(_GREY))
        self._table.setItem(row, 4, status_item)

        pl_item = QTableWidgetItem(profit_str)
        pl_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_item.setForeground(QColor(_GREEN if profit_pct >= 0 else _RED))
        self._table.setItem(row, 5, pl_item)

    def _text_color(self) -> str:
        return "#E0E0E0" if self._dark else "#1A1A2E"

    @staticmethod
    def _make_item(text: str, color: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setForeground(QColor(color))
        return item

    # ══════════════════════════════════════════════════════════════════════
    # STYLING
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _table_style(dark: bool = False) -> str:
        C = get_colors(dark)
        return f"""
            QTableWidget {{
                background-color: {C['TABLE_BG']};
                color: {C['TEXT']};
                border: none;
                font-size: 13px;
                gridline-color: transparent;
            }}
            QTableWidget::item {{
                padding: 10px 4px;
                border-bottom: 1px solid {C['DIVIDER']};
            }}
            QTableWidget::item:selected {{
                background-color: {'#1E2A4A' if dark else '#EEF2FF'};
            }}
            QHeaderView::section {{
                background-color: {C['TABLE_HEADER']};
                color: {C['SUBTEXT']};
                font-weight: bold;
                font-size: 12px;
                padding: 8px 4px;
                border: none;
                border-bottom: 1px solid {C['CARD_BORDER']};
            }}
        """
