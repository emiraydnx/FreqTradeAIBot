"""
technical_analysis.py — Technical Analysis / Charts Page
==========================================================

Full-page chart view matching the FreqTrade Pro mockup:
  • Pair search input + timeframe toggle buttons (1m 5m 15m 1h 4h 1d)
  • Toolbar icons (layers, download, share, fullscreen)
  • Pair header: exchange, price, 24h change, 24h High/Low, Buy/Sell
  • Main: CandlestickChart (reuses ChartWidget with show_controls=False)
  • Right-top: Indicators panel (RSI, MACD, Volatility)
  • Right-bottom: Order Book panel (ask/bid levels)

All heavy data loading is done via DataManager.load_async() which keeps
the GUI thread responsive.
"""

from __future__ import annotations

import numpy as np

import qtawesome as qta

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from GUI.chart_widget import ChartWidget
from core.data_manager import DataManager, DataResult, ProgressEvent
from core.trade_manager import DashboardSnapshot, OverallStatus

# ── Palette ──────────────────────────────────────────────────────────────────
_BG      = "#FFFFFF"
_BG_PAGE = "#F5F6FA"
_BORDER  = "#E8E8E8"
_GREEN   = "#00C087"
_RED     = "#FF4C4C"
_BLUE    = "#3D5AFE"
_TEXT    = "#1A1A2E"
_SUB     = "#888888"
_BULLISH = "#26A69A"
_BEARISH = "#EF5350"


def _card(min_height: int = 0) -> QFrame:
    f = QFrame()
    f.setStyleSheet(f"""
        QFrame {{
            background-color: {_BG};
            border-radius: 12px;
        }}
    """)
    if min_height:
        f.setMinimumHeight(min_height)
    return f


class TechnicalAnalysisPage(QWidget):
    """Charts page with candlestick chart, indicators panel and order book."""

    # Thread-safe progress / done bridge
    _progress_signal = pyqtSignal(int, str)
    _done_signal     = pyqtSignal(object)

    _TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG_PAGE};")

        self._data_manager = DataManager()
        self._current_pair = "BTC/USDT"
        self._current_tf   = "1h"
        self._tf_buttons: list[QPushButton] = []
        self._toolbar_buttons: list[QPushButton] = []
        self._dark = False
        self._status_badge_color = _SUB

        self._progress_signal.connect(self._on_progress)
        self._done_signal.connect(self._on_loaded)

        self._init_ui()
        # Load default pair on start
        QTimer.singleShot(200, self._trigger_load)

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # ── Page title ────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        self._title_lbl = QLabel("Charts")
        self._title_lbl.setStyleSheet(
            f"color: {_TEXT}; font-size: 22px; font-weight: 700;"
        )
        title_row.addWidget(self._title_lbl)
        title_row.addStretch()
        self._status_badge = QLabel("● Disconnected")
        self._status_badge.setStyleSheet(
            f"color: {_SUB}; font-size: 12px; font-weight: 600;"
        )
        title_row.addWidget(self._status_badge)
        layout.addLayout(title_row)

        # ── Search + timeframe bar ────────────────────────────────────────
        ctrl_bar = self._build_control_bar()
        layout.addWidget(ctrl_bar)

        # ── Main content row ──────────────────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setSpacing(14)

        # Left: chart card
        self._chart_card = _card()
        chart_card_layout = QVBoxLayout(self._chart_card)
        chart_card_layout.setContentsMargins(16, 14, 16, 14)
        chart_card_layout.setSpacing(10)

        # Pair header
        pair_header = self._build_pair_header()
        chart_card_layout.addLayout(pair_header)

        # Progress bar (hidden until loading)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar { background: #F0F0F0; border: none; border-radius: 2px; }
            QProgressBar::chunk { background: #3D5AFE; border-radius: 2px; }
        """)
        self._progress_bar.hide()
        chart_card_layout.addWidget(self._progress_bar)

        # CandlestickChart (no built-in controls — we drive it externally)
        self._chart = ChartWidget(show_controls=False)
        self._chart.setMinimumHeight(340)
        chart_card_layout.addWidget(self._chart)

        content_row.addWidget(self._chart_card, stretch=3)

        # Right panels
        right_col = QVBoxLayout()
        right_col.setSpacing(14)
        right_col.addWidget(self._build_indicators_panel())
        right_col.addWidget(self._build_order_book_panel())
        right_col.addStretch()

        right_widget = QWidget()
        right_widget.setLayout(right_col)
        right_widget.setFixedWidth(240)
        content_row.addWidget(right_widget, stretch=0)

        layout.addLayout(content_row)

    def _build_control_bar(self) -> QWidget:
        self._control_bar = QWidget()
        self._control_bar.setStyleSheet(f"background: {_BG}; border-radius: 8px;")
        layout = QHBoxLayout(self._control_bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Search input
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search pair (e.g. BTC/USDT)")
        self._search_input.setFixedWidth(220)
        self._search_input.setStyleSheet(f"""
            QLineEdit {{
                background: #F5F6FA; border: 1px solid {_BORDER};
                border-radius: 6px; padding: 6px 10px;
                font-size: 13px; color: {_TEXT};
            }}
        """)
        self._search_input.returnPressed.connect(self._on_search_enter)
        layout.addWidget(self._search_input)

        layout.addSpacing(8)

        # Timeframe buttons
        for tf in self._TIMEFRAMES:
            btn = QPushButton(tf)
            btn.setFixedSize(40, 30)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, t=tf: self._on_tf_click(t))
            self._tf_buttons.append(btn)
            layout.addWidget(btn)
        self._set_tf_selected(self._current_tf)

        layout.addStretch()

        # Toolbar icons
        _toolbar_defs = [
            ("fa6s.layer-group", "Indicators"),
            ("fa6s.download",    "Download"),
            ("fa6s.share-nodes", "Share"),
            ("fa6s.expand",      "Fullscreen"),
        ]
        for qta_name, tooltip in _toolbar_defs:
            tb = QPushButton()
            tb.setIcon(qta.icon(qta_name, color=_SUB))
            tb.setIconSize(QSize(16, 16))
            tb.setFixedSize(32, 32)
            tb.setToolTip(tooltip)
            tb.setProperty("qta_name", qta_name)
            tb.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: 1px solid {_BORDER};
                    border-radius: 6px;
                }}
                QPushButton:hover {{ background: #F0F0F0; }}
            """)
            self._toolbar_buttons.append(tb)
            layout.addWidget(tb)

        return self._control_bar

    def _build_pair_header(self) -> QHBoxLayout:
        row = QHBoxLayout()

        pair_col = QVBoxLayout()
        pair_col.setSpacing(2)
        self._lbl_pair = QLabel("BTC/USDT")
        self._lbl_pair.setStyleSheet(
            f"color: {_TEXT}; font-size: 16px; font-weight: 700;"
        )
        self._lbl_exchange = QLabel("Binance · Perpetual")
        self._lbl_exchange.setStyleSheet(f"color: {_SUB}; font-size: 11px;")
        pair_col.addWidget(self._lbl_pair)
        pair_col.addWidget(self._lbl_exchange)
        row.addLayout(pair_col)

        row.addSpacing(16)

        # Price + change
        price_col = QVBoxLayout()
        price_col.setSpacing(2)
        self._lbl_price = QLabel("—")
        self._lbl_price.setStyleSheet(
            f"color: {_GREEN}; font-size: 18px; font-weight: 700;"
        )
        self._lbl_change = QLabel("+0.00%")
        self._lbl_change.setStyleSheet(f"color: {_GREEN}; font-size: 11px;")
        price_col.addWidget(self._lbl_price)
        price_col.addWidget(self._lbl_change)
        row.addLayout(price_col)

        row.addSpacing(20)

        # 24h High/Low
        self._high_low_titles: list[QLabel] = []
        for label, attr_name in [("24h High", "_lbl_high"), ("24h Low", "_lbl_low")]:
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl_title = QLabel(label)
            lbl_title.setStyleSheet(f"color: {_SUB}; font-size: 10px;")
            self._high_low_titles.append(lbl_title)
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet(f"color: {_TEXT}; font-size: 12px; font-weight: 600;")
            setattr(self, attr_name, lbl_val)
            col.addWidget(lbl_title)
            col.addWidget(lbl_val)
            row.addLayout(col)
            row.addSpacing(12)

        row.addStretch()

        return row

    def _build_indicators_panel(self) -> QFrame:
        self._indicators_card = _card(min_height=140)
        layout = QVBoxLayout(self._indicators_card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self._indicators_title = QLabel("Indicators")
        self._indicators_title.setStyleSheet(f"color: {_TEXT}; font-size: 14px; font-weight: 700;")
        layout.addWidget(self._indicators_title)

        self._indicator_rows: dict[str, QLabel] = {}
        self._indicator_name_labels: list[QLabel] = []
        for name, default_val, default_color in [
            ("RSI (14)", "—", _SUB),
            ("MACD", "—", _SUB),
            ("Volatility", "—", _SUB),
        ]:
            row = QHBoxLayout()
            title = QLabel(name)
            title.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
            self._indicator_name_labels.append(title)
            val = QLabel(default_val)
            val.setStyleSheet(f"color: {default_color}; font-size: 12px; font-weight: 600;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(title)
            row.addStretch()
            row.addWidget(val)
            layout.addLayout(row)
            self._indicator_rows[name] = val

        layout.addStretch()
        return self._indicators_card

    def _build_order_book_panel(self) -> QFrame:
        self._orderbook_card = _card(min_height=200)
        layout = QVBoxLayout(self._orderbook_card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self._orderbook_title = QLabel("Order Book")
        self._orderbook_title.setStyleSheet(f"color: {_TEXT}; font-size: 14px; font-weight: 700;")
        layout.addWidget(self._orderbook_title)

        # Placeholder ask levels
        self._ask_labels: list[tuple[QLabel, QLabel]] = []
        for price, size in [
            ("—", "—"), ("—", "—"), ("—", "—"), ("—", "—"), ("—", "—"),
        ]:
            row = QHBoxLayout()
            lp = QLabel(price)
            lp.setStyleSheet(f"color: {_RED}; font-size: 12px;")
            ls = QLabel(size)
            ls.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
            ls.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lp)
            row.addStretch()
            row.addWidget(ls)
            layout.addLayout(row)
            self._ask_labels.append((lp, ls))

        # Mid price
        self._mid_price_lbl = QLabel("—")
        self._mid_price_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mid_price_lbl.setStyleSheet(
            f"color: {_TEXT}; font-size: 14px; font-weight: 700; "
            f"border-top: 1px solid {_BORDER}; border-bottom: 1px solid {_BORDER}; "
            f"padding: 4px 0;"
        )
        layout.addWidget(self._mid_price_lbl)

        # Placeholder bid levels
        self._bid_labels: list[tuple[QLabel, QLabel]] = []
        for price, size in [
            ("—", "—"), ("—", "—"), ("—", "—"), ("—", "—"), ("—", "—"),
        ]:
            row = QHBoxLayout()
            lp = QLabel(price)
            lp.setStyleSheet(f"color: {_GREEN}; font-size: 12px;")
            ls = QLabel(size)
            ls.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
            ls.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lp)
            row.addStretch()
            row.addWidget(ls)
            layout.addLayout(row)
            self._bid_labels.append((lp, ls))

        layout.addStretch()
        return self._orderbook_card

    # ══════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_search_enter(self) -> None:
        text = self._search_input.text().strip().upper()
        if text:
            self._current_pair = text
            self._lbl_pair.setText(self._current_pair)
            self._trigger_load()

    def _on_tf_click(self, tf: str) -> None:
        self._current_tf = tf
        self._set_tf_selected(tf)
        self._trigger_load()

    def _set_tf_selected(self, tf: str) -> None:
        from GUI.themes import get_colors
        C = get_colors(self._dark)
        for btn in self._tf_buttons:
            selected = btn.text() == tf
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {"#3D5AFE" if selected else "transparent"};
                    color: {"#FFF" if selected else C['TEXT']};
                    border: {"none" if selected else f"1px solid {C['CARD_BORDER']}"};
                    border-radius: 6px; font-size: 12px; font-weight: 600;
                }}
                QPushButton:hover {{
                    background: {"#536DFE" if selected else C['TABLE_ALT']};
                }}
            """)

    def _trigger_load(self) -> None:
        """Start async data load for current pair/timeframe."""
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._data_manager.load_async(
            pair=self._current_pair,
            timeframe=self._current_tf,
            days=180,
            on_progress=lambda e: self._progress_signal.emit(e.percent, e.message),
            on_done=lambda r: self._done_signal.emit(r),
        )

    # ══════════════════════════════════════════════════════════════════════
    # SLOTS (GUI thread)
    # ══════════════════════════════════════════════════════════════════════

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress_bar.setValue(percent)

    def _set_status_badge(self, text: str, color: str) -> None:
        self._status_badge_color = color
        self._status_badge.setText(f"● {text}")
        self._status_badge.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 600;"
        )

    @staticmethod
    def _format_price(value: float) -> str:
        return f"${value:,.2f}"

    @staticmethod
    def _format_level_with_pct(level: float, reference_price: float) -> str:
        if not reference_price:
            return TechnicalAnalysisPage._format_price(level)
        delta_pct = ((level - reference_price) / reference_price) * 100
        sign = "+" if delta_pct >= 0 else ""
        return f"{TechnicalAnalysisPage._format_price(level)} ({sign}{delta_pct:.2f}%)"

    def _on_loaded(self, result: DataResult) -> None:
        self._progress_bar.hide()
        if not result.success:
            return

        df = result.dataframe
        self._chart._render_candles(df)

        # Update pair header
        if not df.empty:
            self._current_pair = result.pair
            self._lbl_pair.setText(result.pair)
            self._search_input.setText(result.pair)
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            price = last["close"]
            change_pct = (price - prev["close"]) / prev["close"] * 100 if prev["close"] else 0
            high24 = df["high"].tail(24).max() if len(df) >= 24 else df["high"].max()
            low24  = df["low"].tail(24).min()  if len(df) >= 24 else df["low"].min()

            color = _GREEN if change_pct >= 0 else _RED
            self._lbl_price.setText(self._format_price(price))
            self._lbl_price.setStyleSheet(
                f"color: {color}; font-size: 18px; font-weight: 700;"
            )
            sign = "+" if change_pct >= 0 else ""
            self._lbl_change.setText(f"{sign}{change_pct:.2f}%")
            self._lbl_change.setStyleSheet(f"color: {color}; font-size: 11px;")
            self._lbl_high.setText(self._format_level_with_pct(high24, price))
            self._lbl_low.setText(self._format_level_with_pct(low24, price))
            self._mid_price_lbl.setText(f"{price:,.2f}")

            # Update indicators (calculated from OHLCV)
            self._update_indicators(df)

    def update_snapshot(self, snap: DashboardSnapshot) -> None:
        from GUI.themes import get_colors
        C = get_colors(self._dark)
        if snap.exchange:
            market_type = "Spot" if "/" in self._current_pair else "Market"
            self._lbl_exchange.setText(f"{snap.exchange.title()} · {market_type}")

        if snap.open_trade_count > 0:
            self._set_status_badge(f"Bot Active: {snap.open_trade_count} open", C["GREEN"])
        elif snap.has_data:
            self._set_status_badge("Bot Running - idle", C["SUBTEXT"])

    def update_connection_status(self, status: OverallStatus) -> None:
        from GUI.themes import get_colors
        C = get_colors(self._dark)
        colours = {
            OverallStatus.DISCONNECTED: (C["RED"], "Bot Offline"),
            OverallStatus.CONNECTING: (C["ORANGE"], "Connecting..."),
            OverallStatus.CONNECTED: (C["BLUE"], "REST Connected"),
            OverallStatus.LIVE: (C["GREEN"], "Bot Online"),
            OverallStatus.ERROR: (C["RED"], "Bot Offline - check Docker"),
        }
        color, text = colours.get(status, (C["SUBTEXT"], "Unknown"))
        self._set_status_badge(text, color)

    def _update_indicators(self, df) -> None:
        """Calculate basic indicators from OHLCV dataframe and update panel."""
        try:
            # RSI (14)
            closes = df["close"].values.astype(float)
            rsi = self._calc_rsi(closes, 14)
            rsi_val = f"{rsi:.1f}" if rsi is not None else "—"
            rsi_color = _GREEN if rsi and rsi < 70 else _RED if rsi and rsi > 70 else _SUB
            self._indicator_rows["RSI (14)"].setText(rsi_val)
            self._indicator_rows["RSI (14)"].setStyleSheet(
                f"color: {rsi_color}; font-size: 12px; font-weight: 600;"
            )

            # MACD (simplified: 12 EMA > 26 EMA → Bullish)
            if len(closes) >= 26:
                ema12 = self._calc_ema(closes, 12)
                ema26 = self._calc_ema(closes, 26)
                macd_text = "Bullish" if ema12 > ema26 else "Bearish"
                macd_color = _GREEN if ema12 > ema26 else _RED
            else:
                macd_text, macd_color = "—", _SUB
            self._indicator_rows["MACD"].setText(macd_text)
            self._indicator_rows["MACD"].setStyleSheet(
                f"color: {macd_color}; font-size: 12px; font-weight: 600;"
            )

            # Volatility (std of returns over last 20 bars)
            if len(closes) >= 20:
                returns = np.diff(closes[-21:]) / closes[-21:-1]
                vol = float(np.std(returns) * 100)
                vol_text = "Low" if vol < 1 else "High" if vol > 3 else "Medium"
                vol_color = _GREEN if vol_text == "Low" else _RED if vol_text == "High" else _SUB
            else:
                vol_text, vol_color = "—", _SUB
            self._indicator_rows["Volatility"].setText(vol_text)
            self._indicator_rows["Volatility"].setStyleSheet(
                f"color: {vol_color}; font-size: 12px; font-weight: 600;"
            )
        except Exception:
            pass

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> float | None:
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _calc_ema(closes: np.ndarray, period: int) -> float:
        if len(closes) < period:
            return float(closes[-1])
        k = 2.0 / (period + 1)
        ema = float(closes[:period].mean())
        for price in closes[period:]:
            ema = price * k + ema * (1 - k)
        return ema

    # ══════════════════════════════════════════════════════════════════════
    # THEMING
    # ══════════════════════════════════════════════════════════════════════

    def apply_theme(self, dark: bool) -> None:
        from GUI.themes import get_colors
        self._dark = dark
        C = get_colors(dark)
        self.setStyleSheet(f"background-color: {C['PAGE_BG']};")

        # Page title
        self._title_lbl.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 22px; font-weight: 700;"
        )
        self._status_badge.setStyleSheet(
            f"color: {self._status_badge_color}; font-size: 12px; font-weight: 600;"
        )

        # Card QSS
        card_qss = f"""
            QFrame {{
                background-color: {C['CARD_BG']};
                border-radius: 12px;
            }}
        """

        # Control bar
        self._control_bar.setStyleSheet(
            f"background: {C['CARD_BG']}; border-radius: 8px;"
        )

        # Search input
        self._search_input.setStyleSheet(f"""
            QLineEdit {{
                background: {C['INPUT_BG']}; border: 1px solid {C['INPUT_BORDER']};
                border-radius: 6px; padding: 6px 10px;
                font-size: 13px; color: {C['TEXT']};
            }}
        """)

        # Toolbar icons
        for tb in self._toolbar_buttons:
            icon_name = tb.property("qta_name")
            if icon_name:
                tb.setIcon(qta.icon(icon_name, color=C['SUBTEXT']))
            tb.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: 1px solid {C['CARD_BORDER']};
                    border-radius: 6px;
                }}
                QPushButton:hover {{ background: {C['TABLE_ALT']}; }}
            """)

        # Chart card
        self._chart_card.setStyleSheet(card_qss)

        # Pair header labels
        self._lbl_pair.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 16px; font-weight: 700;"
        )
        self._lbl_exchange.setStyleSheet(
            f"color: {C['SUBTEXT']}; font-size: 11px;"
        )
        self._lbl_high.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 12px; font-weight: 600;"
        )
        self._lbl_low.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 12px; font-weight: 600;"
        )
        for lbl in self._high_low_titles:
            lbl.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 10px;")

        # Mid price
        self._mid_price_lbl.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 14px; font-weight: 700; "
            f"border-top: 1px solid {C['DIVIDER']}; "
            f"border-bottom: 1px solid {C['DIVIDER']}; padding: 4px 0;"
        )

        # Indicators panel
        self._indicators_card.setStyleSheet(card_qss)
        self._indicators_title.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 14px; font-weight: 700; border: none;"
        )
        for lbl in self._indicator_name_labels:
            lbl.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 12px; border: none;")

        # Order book panel
        self._orderbook_card.setStyleSheet(card_qss)
        self._orderbook_title.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 14px; font-weight: 700; border: none;"
        )
        for lp, ls in self._ask_labels:
            ls.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 12px; border: none;")
        for lp, ls in self._bid_labels:
            ls.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 12px; border: none;")

        # Chart widget
        self._chart.apply_theme(dark)

        # Timeframe buttons
        self._set_tf_selected(self._current_tf)
