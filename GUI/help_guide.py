"""
help_guide.py — Help & Guide Page
===================================

Static help page with usage guides, strategy tips, and API setup info.
No backend connections required.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# ── Palette ──────────────────────────────────────────────────────────────────
_BG      = "#FFFFFF"
_BG_PAGE = "#F5F6FA"
_BORDER  = "#E8E8E8"
_TEXT    = "#1A1A2E"
_BLUE    = "#3D5AFE"
_SUB     = "#888888"

_HELP_HTML = """
<html><body style="font-family: 'Segoe UI', sans-serif; color: #1A1A2E; font-size: 13px; line-height: 1.7;">

<h2 style="color: #1A1A2E; margin-bottom: 4px;">Getting Started</h2>
<p style="color: #888; margin-top: 0;">FreqTrade Pro helps you run, monitor and train algorithmic trading bots.</p>

<hr style="border: none; border-top: 1px solid #E8E8E8; margin: 16px 0;">

<h3 style="color: #3D5AFE;">1. Connecting to Your Bot</h3>
<ol>
  <li>Open <b>Bot Configuration</b> and fill in your exchange API credentials.</li>
  <li>Set the REST API host/port (default: <code>localhost:8080</code>).</li>
  <li>Click <b>Start Bot</b> to launch the Freqtrade process.</li>
  <li>The header badge will change to <span style="color: #00C087;">● Bot Online</span> once connected.</li>
</ol>

<h3 style="color: #3D5AFE;">2. Dashboard</h3>
<p>
The Dashboard shows live trading metrics: total balance, profit/loss, win rate and open trades.
The <b>Performance Overview</b> chart updates daily.
The <b>Recent Trade History</b> table shows the latest executions across all active bots.
</p>

<h3 style="color: #3D5AFE;">3. Technical Analysis</h3>
<p>
Navigate to <b>Charts</b> to view candlestick charts.
Type a pair in the search box (e.g. <code>BTC/USDT</code>) and select a timeframe.
The right panel shows live RSI, MACD and Volatility indicators.
</p>

<h3 style="color: #3D5AFE;">4. Backtesting</h3>
<ol>
  <li>Select a strategy, timeframe and date range in the left panel.</li>
  <li>Click <b>Download Data</b> if you need to fetch historical OHLCV data first.</li>
  <li>Click <b>Run Backtest</b> to execute the strategy against historical data.</li>
  <li>Results (Total Profit, Max Drawdown, Sharpe Ratio) are shown on the right.</li>
</ol>

<h3 style="color: #3D5AFE;">5. Model Training (FreqAI)</h3>
<ol>
  <li>Go to <b>Model Training</b> and click <b>Prepare Dataset</b> first.</li>
  <li>Click <b>Start Training</b> to begin ML model training.</li>
  <li>Monitor epoch progress, loss and accuracy in real time.</li>
  <li>Trained models appear in <b>Model History</b> on the right.</li>
</ol>

<hr style="border: none; border-top: 1px solid #E8E8E8; margin: 16px 0;">

<h2 style="color: #1A1A2E;">Strategy Tips</h2>
<ul>
  <li>Use at least <b>6 months</b> of historical data for reliable backtests.</li>
  <li>Combine both bull and bear market cycles in your training data.</li>
  <li>Always run a dry-run first before live trading.</li>
  <li>Set a stop-loss in your strategy file (<code>stoploss = -0.05</code>).</li>
  <li>Use <code>minimal_roi</code> to lock in profits at pre-defined levels.</li>
</ul>

<hr style="border: none; border-top: 1px solid #E8E8E8; margin: 16px 0;">

<h2 style="color: #1A1A2E;">Supported Exchanges</h2>
<p>Binance, Kraken, OKX, Bybit, Gate.io, and any CCXT-compatible exchange.</p>

<h2 style="color: #1A1A2E;">Config Files</h2>
<p>All configuration files are stored in the <code>config/</code> folder.
Load or save them from the <b>Bot Configuration</b> page.</p>

<h2 style="color: #1A1A2E;">Logs</h2>
<p>Bot logs are stored in <code>user_data/logs/</code> and are also visible
in the log panel at the bottom of the <b>Bot Configuration</b> page.</p>

<hr style="border: none; border-top: 1px solid #E8E8E8; margin: 16px 0;">

<p style="color: #888; font-size: 12px;">
FreqTrade Pro &mdash; Built on top of the open-source
<a href="https://www.freqtrade.io" style="color: #3D5AFE;">Freqtrade</a> framework.
</p>
</body></html>
"""


class HelpGuidePage(QWidget):
    """Static help and documentation page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG_PAGE};")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)

        # Title
        title = QLabel("Help & Guide")
        title.setStyleSheet(f"color: {_TEXT}; font-size: 22px; font-weight: 700;")
        layout.addWidget(title)

        # Card
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {_BG};
                border-radius: 12px;
                border: 1px solid {_BORDER};
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(_HELP_HTML)
        browser.setStyleSheet(f"""
            QTextBrowser {{
                background: {_BG}; border: none;
                padding: 20px 28px;
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
                color: {_TEXT};
            }}
            QScrollBar:vertical {{
                background: #F5F6FA; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #CCCCCC; border-radius: 4px;
            }}
        """)
        self._browser = browser
        card_layout.addWidget(browser)
        layout.addWidget(card)

    # ══════════════════════════════════════════════════════════════════════
    # THEMING
    # ══════════════════════════════════════════════════════════════════════

    def apply_theme(self, dark: bool) -> None:
        from GUI.themes import get_colors
        C = get_colors(dark)
        self.setStyleSheet(f"background-color: {C['PAGE_BG']};")
        self._browser.setStyleSheet(f"""
            QTextBrowser {{
                background: {C['CARD_BG']}; border: none;
                padding: 20px 28px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                color: {C['TEXT']};
            }}
            QScrollBar:vertical {{
                background: {C['SCROLLBAR_BG']}; width: 8px; border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {C['SCROLLBAR']}; border-radius: 4px;
            }}
        """)
