"""
main_window.py — Application Main Window
==========================================

Creates the TradeManager (core layer) and wires it to all GUI pages
via pyqtSignals so that every widget update happens on the GUI thread.

Sidebar collapses between full (220 px) and icon-only (60 px) modes
with a smooth animation.
"""

import sys
import logging

import qtawesome as qta

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QLabel, QPushButton, QSizePolicy, QFrame,
    QGraphicsOpacityEffect,
)
from PyQt6.QtCore import (
    Qt, QSize, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QParallelAnimationGroup, QSequentialAnimationGroup,
)
from PyQt6.QtGui import QFont, QIcon

from core.trade_manager import TradeManager, DashboardSnapshot, OverallStatus
from core.exchange_manager import BotState
from API.ws_client import WSMessage

from GUI.dashboard import DashboardPage
from GUI.bot_config_page import BotConfigPage
from GUI.backtest_view import BacktestView
from GUI.technical_analysis import TechnicalAnalysisPage
from GUI.model_training import ModelTrainingPage
from GUI.help_guide import HelpGuidePage
from GUI.settings_panel import SettingsPanel
from GUI.themes import get_colors

logger = logging.getLogger(__name__)

# ── Sidebar dimensions ───────────────────────────────────────────────────────
_EXPANDED_W = 220
_COLLAPSED_W = 60

# Nav entries: (qta_icon_name, label)
_NAV = [
    ("fa5s.tachometer-alt",      "Dashboard"),
    ("fa5s.history",      "Backtesting"),
    ("fa5s.chart-line",      "Technical Analysis"),
    ("fa6s.sliders",            "Bot Configuration"),
    ("fa5s.brain",           "Model Training"),
    ("fa6s.circle-question", "Help & Guide"),
    ("fa6s.gear",         "Settings"),
]


class _NavButton(QPushButton):
    """Single sidebar navigation button with qtawesome icon + optional label."""

    def __init__(self, qta_icon_name: str, label: str, parent=None) -> None:
        super().__init__(parent)
        self._qta_name = qta_icon_name
        self._label = label
        self._selected = False
        self._dark = False
        self._expanded = True
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(46)
        self.setIconSize(QSize(18, 18))
        self._refresh()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._refresh()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._refresh()

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self._refresh()

    def _refresh(self) -> None:
        color = "#FFFFFF" if self._selected else ("#A0A0A0" if self._dark else "#555555")
        self.setIcon(qta.icon(self._qta_name, color=color))
        if self._expanded:
            self.setText(f"  {self._label}")
        else:
            self.setText("")
        bg = "#3D5AFE" if self._selected else "transparent"
        bg_hover = "#536DFE" if self._selected else ("#2A2A2A" if self._dark else "#F0F0F0")
        align = "left" if self._expanded else "center"
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {color};
                border: none;
                border-radius: 8px;
                padding: 0 12px;
                font-family: "Segoe UI", sans-serif;
                font-size: 14px;
                font-weight: 500;
                text-align: {align};
            }}
            QPushButton:hover {{
                background-color: {bg_hover};
                color: #FFFFFF;
            }}
        """)


class MainWindow(QMainWindow):
    """
    Top-level window.

    Owns the ``TradeManager`` and bridges its worker-thread callbacks
    to GUI-thread slots via pyqtSignals.
    """

    # ── Bridge signals (worker thread → GUI thread) ───────────────────────
    _dashboard_signal   = pyqtSignal(object)   # DashboardSnapshot
    _connection_signal  = pyqtSignal(object)   # OverallStatus
    _trade_event_signal = pyqtSignal(object)   # WSMessage

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("FreqTrade Pro")
        self.setMinimumSize(1280, 800)

        # ── Core layer ────────────────────────────────────────────────────
        self._tm = TradeManager()
        self._dark_mode = False
        self._sidebar_expanded = True

        self.setStyleSheet(self._get_stylesheet(False))
        self._init_ui()
        self._connect_signals()
        self._try_connect()

        logger.info("MainWindow ready")

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Top header bar ────────────────────────────────────────────────
        header = self._build_header()
        root_layout.addWidget(header)

        # ── Main content row (sidebar + pages) ────────────────────────────
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._sidebar = self._build_sidebar()
        body_layout.addWidget(self._sidebar)

        self.content_area = QStackedWidget()
        self.content_area.setStyleSheet("background-color: #F5F6FA;")
        body_layout.addWidget(self.content_area)

        root_layout.addLayout(body_layout)

        # ── Pages ─────────────────────────────────────────────────────────
        self.page_dashboard       = DashboardPage()
        self.page_backtest        = BacktestView()
        self.page_technical       = TechnicalAnalysisPage()
        self.page_bot_config      = BotConfigPage(trade_manager=self._tm)
        self.page_model_training  = ModelTrainingPage()
        self.page_help            = HelpGuidePage()
        self.page_settings        = SettingsPanel()

        self.content_area.addWidget(self.page_dashboard)       # index 0
        self.content_area.addWidget(self.page_backtest)        # index 1
        self.content_area.addWidget(self.page_technical)       # index 2
        self.content_area.addWidget(self.page_bot_config)      # index 3
        self.content_area.addWidget(self.page_model_training)  # index 4
        self.content_area.addWidget(self.page_help)            # index 5
        self.content_area.addWidget(self.page_settings)        # index 6

        self._select_nav(0)

        # Apply initial theme to all pages
        self._apply_theme(self._dark_mode)

    def _build_header(self) -> QWidget:
        """Build the top navigation bar with logo, status badge, balance, avatar."""
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(
            "background-color: #FFFFFF; border-bottom: 1px solid #E8E8E8;"
        )
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 20, 0)
        layout.setSpacing(12)

        # Hamburger toggle
        self._toggle_btn = QPushButton()
        self._toggle_btn.setIcon(qta.icon("fa6s.bars", color="#555"))
        self._toggle_btn.setIconSize(QSize(18, 18))
        self._toggle_btn.setFixedSize(36, 36)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent; border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #F0F0F0; }
        """)
        self._toggle_btn.clicked.connect(self._toggle_sidebar)
        layout.addWidget(self._toggle_btn)

        # Logo / app name
        logo = QLabel("FreqTrade Pro")
        logo.setStyleSheet(
            "color: #1A1A2E; font-size: 17px; font-weight: 700; "
            "font-family: 'Segoe UI';"
        )
        self._logo_lbl = logo
        layout.addWidget(logo)

        layout.addSpacing(16)

        # Connection status badge
        self._header_status = QLabel("● Bot Online: 0 Active")
        self._header_status.setStyleSheet(
            "color: #00C087; font-size: 12px; font-weight: 600;"
        )
        layout.addWidget(self._header_status)

        layout.addStretch()

        # Total balance
        self._header_balance = QLabel("Total Balance   $0.00")
        self._header_balance.setStyleSheet(
            "color: #555; font-size: 13px;"
        )
        layout.addWidget(self._header_balance)

        layout.addSpacing(12)

        # User avatar / initials
        avatar = QLabel("EA")
        avatar.setFixedSize(34, 34)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            "background-color: #1A1A2E; color: #FFFFFF; border-radius: 17px; "
            "font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(avatar)

        self._header_bar = header
        return header

    def _build_sidebar(self) -> QWidget:
        """Build collapsible left sidebar."""
        sidebar = QWidget()
        sidebar.setFixedWidth(_EXPANDED_W)
        sidebar.setStyleSheet(
            "background-color: #FFFFFF; border-right: 1px solid #E8E8E8;"
        )
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(2)

        # Nav buttons
        self._nav_buttons: list[_NavButton] = []
        for icon, label in _NAV:
            btn = _NavButton(icon, label)
            btn.clicked.connect(lambda checked, b=btn: self._on_nav_click(b))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addStretch()

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #E8E8E8;")
        layout.addWidget(divider)

        # Dark mode toggle
        self._dark_mode_btn = QPushButton("  Light Mode")
        self._dark_mode_btn.setIcon(qta.icon("fa6s.sun", color="#888"))
        self._dark_mode_btn.setIconSize(QSize(16, 16))
        self._dark_mode_btn.setFixedHeight(40)
        self._dark_mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dark_mode_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #888;
                border: none; border-radius: 8px;
                padding: 0 12px; font-size: 13px; text-align: left;
            }
            QPushButton:hover { background-color: #F5F5F5; color: #333; }
        """)
        self._dark_mode_btn.clicked.connect(self._toggle_dark_mode)
        layout.addWidget(self._dark_mode_btn)

        return sidebar

    # ══════════════════════════════════════════════════════════════════════
    # SIDEBAR COLLAPSE / EXPAND
    # ══════════════════════════════════════════════════════════════════════

    def _toggle_sidebar(self) -> None:
        self._sidebar_expanded = not self._sidebar_expanded
        target = _EXPANDED_W if self._sidebar_expanded else _COLLAPSED_W

        anim = QPropertyAnimation(self._sidebar, b"minimumWidth")
        anim.setDuration(200)
        anim.setStartValue(self._sidebar.width())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        anim2 = QPropertyAnimation(self._sidebar, b"maximumWidth")
        anim2.setDuration(200)
        anim2.setStartValue(self._sidebar.width())
        anim2.setEndValue(target)
        anim2.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self._anim1 = anim
        self._anim2 = anim2

        anim.finished.connect(self._on_anim_done)
        anim.start()
        anim2.start()

    def _on_anim_done(self) -> None:
        for btn in self._nav_buttons:
            btn.set_expanded(self._sidebar_expanded)
        if self._sidebar_expanded:
            self._dark_mode_btn.setVisible(True)
        else:
            self._dark_mode_btn.setVisible(False)

    # ══════════════════════════════════════════════════════════════════════
    # DARK MODE
    # ══════════════════════════════════════════════════════════════════════

    def _toggle_dark_mode(self) -> None:
        self._dark_mode = not self._dark_mode
        dark = self._dark_mode

        # Animate content opacity: fade out → apply theme → fade in
        current = self.content_area.currentWidget()
        if not current.graphicsEffect():
            effect = QGraphicsOpacityEffect(current)
            effect.setOpacity(1.0)
            current.setGraphicsEffect(effect)

        effect = current.graphicsEffect()

        fade_out = QPropertyAnimation(effect, b"opacity")
        fade_out.setDuration(120)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InQuad)

        fade_in = QPropertyAnimation(effect, b"opacity")
        fade_in.setDuration(250)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.OutQuad)

        def on_faded_out():
            self._apply_theme(dark)
            self._theme_fade_in = fade_in
            fade_in.start()

        fade_out.finished.connect(on_faded_out)
        self._theme_fade_out = fade_out
        fade_out.start()

    def _apply_theme(self, dark: bool) -> None:
        C = get_colors(dark)

        # Global app stylesheet (scrollbars, fonts)
        self.setStyleSheet(self._get_stylesheet(dark))

        # Header bar
        self._header_bar.setStyleSheet(
            f"background-color: {C['HEADER_BG']}; border-bottom: 1px solid {C['HEADER_BORDER']};"
        )
        self._toggle_btn.setIcon(qta.icon("fa6s.bars", color=C['SUBTEXT']))
        self._logo_lbl.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 17px; font-weight: 700; "
            f"font-family: 'Segoe UI';"
        )
        self._header_balance.setStyleSheet(f"color: {C['SUBTEXT']}; font-size: 13px;")

        # Sidebar
        self._sidebar.setStyleSheet(
            f"background-color: {C['SIDEBAR_BG']}; border-right: 1px solid {C['SIDEBAR_BORDER']};"
        )
        icon_name = "fa6s.sun" if dark else "fa6s.moon"
        label_text = "  Light Mode" if dark else "  Dark Mode"
        self._dark_mode_btn.setIcon(qta.icon(icon_name, color=C['TOGGLE_TEXT']))
        self._dark_mode_btn.setText(label_text)
        self._dark_mode_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C['TOGGLE_TEXT']};
                border: none; border-radius: 8px;
                padding: 0 12px; font-size: 13px; text-align: left;
            }}
            QPushButton:hover {{ background-color: {C['TOGGLE_HOVER']}; color: {C['TEXT']}; }}
        """)

        # Nav buttons
        for btn in self._nav_buttons:
            btn.set_dark(dark)

        # Content area
        self.content_area.setStyleSheet(f"background-color: {C['PAGE_BG']};")

        # All page widgets
        self.page_dashboard.apply_theme(dark)
        self.page_backtest.apply_theme(dark)
        self.page_technical.apply_theme(dark)
        self.page_bot_config.apply_theme(dark)
        self.page_model_training.apply_theme(dark)
        self.page_help.apply_theme(dark)
        self.page_settings.apply_theme(dark)

    # ══════════════════════════════════════════════════════════════════════
    # NAVIGATION
    # ══════════════════════════════════════════════════════════════════════

    def _on_nav_click(self, clicked_btn: _NavButton) -> None:
        idx = self._nav_buttons.index(clicked_btn)
        self._select_nav(idx)

    def _select_nav(self, index: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            btn.set_selected(i == index)
        self._fade_to_page(index)

    def _fade_to_page(self, index: int) -> None:
        """Fade-out current page, switch, fade-in new page."""
        if index == self.content_area.currentIndex():
            self.content_area.setCurrentIndex(index)
            return

        old_widget = self.content_area.currentWidget()
        new_widget = self.content_area.widget(index)

        # Ensure both have opacity effects
        for w in (old_widget, new_widget):
            if not w.graphicsEffect():
                effect = QGraphicsOpacityEffect(w)
                effect.setOpacity(1.0)
                w.setGraphicsEffect(effect)

        old_effect = old_widget.graphicsEffect()
        new_effect = new_widget.graphicsEffect()
        new_effect.setOpacity(0.0)

        # Fade out old
        fade_out = QPropertyAnimation(old_effect, b"opacity")
        fade_out.setDuration(150)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InQuad)

        # Fade in new
        fade_in = QPropertyAnimation(new_effect, b"opacity")
        fade_in.setDuration(200)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.OutQuad)

        def on_fade_out_done():
            self.content_area.setCurrentIndex(index)
            self._fade_in_anim = fade_in
            fade_in.start()

        fade_out.finished.connect(on_fade_out_done)
        self._fade_out_anim = fade_out
        fade_out.start()

    # ══════════════════════════════════════════════════════════════════════
    # SIGNAL WIRING
    # ══════════════════════════════════════════════════════════════════════

    def _connect_signals(self) -> None:
        self._dashboard_signal.connect(self._slot_dashboard_update)
        self._connection_signal.connect(self._slot_connection_change)
        self._trade_event_signal.connect(self._slot_trade_event)

        self._tm.on_dashboard_update  = self._dashboard_signal.emit
        self._tm.on_connection_change = self._connection_signal.emit
        self._tm.on_trade_event       = self._trade_event_signal.emit

    def _try_connect(self) -> None:
        """Auto-connect to the FreqTrade REST API on startup."""
        try:
            self._tm.configure_from_config("config/config_binance.json")
            self._tm.connect()
            self._tm.start_polling(interval_sec=5.0)
            logger.info("Auto-connect started")
        except FileNotFoundError:
            logger.warning("Config not found — running GUI-only mode")
            self._connection_signal.emit(OverallStatus.DISCONNECTED)
        except Exception as exc:
            logger.warning("Auto-connect failed: %s", exc)
            self._connection_signal.emit(OverallStatus.ERROR)

    # ══════════════════════════════════════════════════════════════════════
    # SLOTS (GUI thread)
    # ══════════════════════════════════════════════════════════════════════

    def _slot_dashboard_update(self, snapshot: DashboardSnapshot) -> None:
        self.page_dashboard.update_snapshot(snapshot)
        self.page_technical.update_snapshot(snapshot)
        # Update header balance
        self._header_balance.setText(
            f"Total Balance   ${snapshot.total_balance:,.2f}"
        )

        if snapshot.open_trade_count > 0:
            self._header_status.setText(f"● Bot Active: {snapshot.open_trade_count} open")
            self._header_status.setStyleSheet(
                "color: #00C087; font-size: 12px; font-weight: 600;"
            )
        elif snapshot.has_data and self._tm.status in (OverallStatus.CONNECTED, OverallStatus.LIVE):
            self._header_status.setText("● Bot Running - idle")
            self._header_status.setStyleSheet(
                "color: #A0A0A0; font-size: 12px; font-weight: 600;"
            )

    def _slot_connection_change(self, status: OverallStatus) -> None:
        self.page_dashboard.update_connection_status(status)
        self.page_technical.update_connection_status(status)
        colours = {
            OverallStatus.DISCONNECTED: ("#F44336", "Bot Offline"),
            OverallStatus.CONNECTING:   ("#FFC107", "Connecting…"),
            OverallStatus.CONNECTED:    ("#2196F3", "Bot Connected"),
            OverallStatus.LIVE:         ("#00C087", "Bot Online"),
            OverallStatus.ERROR:        ("#F44336", "Bot Error"),
        }
        color, text = colours.get(status, ("#A0A0A0", "Unknown"))
        self._header_status.setText(f"● {text}")
        self._header_status.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 600;"
        )

    def _slot_trade_event(self, msg: WSMessage) -> None:
        logger.debug("Trade event: %s", msg.msg_type)

    # ══════════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ══════════════════════════════════════════════════════════════════════

    @property
    def trade_manager(self) -> TradeManager:
        return self._tm

    @property
    def sidebar(self):
        return self._sidebar

    # ══════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    def closeEvent(self, event) -> None:
        logger.info("Shutting down…")
        self._tm.close()
        super().closeEvent(event)

    @staticmethod
    def _get_stylesheet(dark: bool = False) -> str:
        bg = "#121212" if dark else "#F5F6FA"
        sb_bg = "#1E1E1E" if dark else "#F0F0F0"
        sb_handle = "#444444" if dark else "#CCCCCC"
        return f"""
        QMainWindow {{
            background-color: {bg};
        }}
        QStackedWidget {{
            background-color: {bg};
        }}
        QWidget {{
            font-family: "Segoe UI", sans-serif;
        }}
        QScrollBar:vertical {{
            background: {sb_bg}; width: 8px; border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {sb_handle}; border-radius: 4px; min-height: 20px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """


# Allow running this file directly for quick testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())