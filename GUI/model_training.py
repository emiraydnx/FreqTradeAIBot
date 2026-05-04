"""
model_training.py — Model Training & Bot Lifecycle Page
========================================================

Unified page for:
  • Loading a saved config file and previewing FreqAI ML parameters
  • API connection controls (connect / disconnect)
  • Prepare Dataset and Run & Train Bot actions
  • Real-time training progress via bot log parsing
  • Model History from user_data/models/
  • Training Tips
"""

from __future__ import annotations

import json
import logging
import re
import html
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import qtawesome as qta

import pandas as pd

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QDate
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDateEdit,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from API.rest_client import ApiResult
from API.ws_client import WSState
from core.exchange_manager import BotState, BotLogEvent, BotProcessManager
from core.trade_manager import TradeManager, OverallStatus
from GUI.themes import get_colors

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

logger = logging.getLogger(__name__)

# ── Palette ──────────────────────────────────────────────────────────────────
_BG      = "#FFFFFF"
_BG_PAGE = "#F5F6FA"
_BORDER  = "#E8E8E8"
_GREEN   = "#00C087"
_RED     = "#FF4C4C"
_ORANGE  = "#FF9A00"
_BLUE    = "#3D5AFE"
_TEXT    = "#1A1A2E"
_SUB     = "#888888"

_MODELS_DIR = Path("user_data/models")

_CANDLES_PER_DAY: dict[str, int] = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "30m": 48,
    "1h": 24,
    "4h": 6,
    "1d": 1,
}

# ── Status badge constants ───────────────────────────────────────────────────
_DOT = (
    "font-size: 14px; font-weight: bold; border: none; "
    "padding: 2px 8px; border-radius: 4px;"
)
_CONN_COLOURS = {
    OverallStatus.DISCONNECTED: ("#F44336", "Disconnected"),
    OverallStatus.CONNECTING:   ("#FFC107", "Connecting…"),
    OverallStatus.CONNECTED:    ("#2196F3", "REST Connected"),
    OverallStatus.LIVE:         ("#4CAF50", "Live (REST + WS)"),
    OverallStatus.ERROR:        ("#F44336", "Error"),
}
_BOT_COLOURS = {
    BotState.STOPPED:  ("#A0A0A0", "Stopped"),
    BotState.STARTING: ("#FFC107", "Starting…"),
    BotState.RUNNING:  ("#4CAF50", "Running"),
    BotState.STOPPING: ("#FFC107", "Stopping…"),
    BotState.ERROR:    ("#F44336", "Error"),
}


def _card_style(min_height: int = 0) -> str:
    extra = f"min-height: {min_height}px;" if min_height else ""
    return f"""
        QFrame {{
            background-color: {_BG};
            border-radius: 12px;
            {extra}
        }}
    """


def _btn_qss(color: str) -> str:
    return f"""
        QPushButton {{
            background-color: {color}; color: #FFFFFF; border: none;
            border-radius: 6px; padding: 9px 18px;
            font-size: 13px; font-weight: bold;
        }}
        QPushButton:hover {{ opacity: 0.85; }}
        QPushButton:disabled {{ background-color: #2C2C2C; color: #555555; }}
    """


class _MetricRow(QWidget):
    """Label + value pair for training metrics."""

    def __init__(self, label: str, value: str, value_color: str = _TEXT, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        lbl = QLabel(label.upper())
        lbl.setStyleSheet(
            f"color: {_SUB}; font-size: 10px; font-weight: 600; letter-spacing: 1px;"
        )
        layout.addWidget(lbl)
        layout.addStretch()
        self._val_lbl = QLabel(value)
        self._val_lbl.setStyleSheet(
            f"color: {value_color}; font-size: 18px; font-weight: 700;"
        )
        layout.addWidget(self._val_lbl)

    def set_value(self, value: str, color: str | None = None) -> None:
        self._val_lbl.setText(value)
        if color:
            self._val_lbl.setStyleSheet(
                f"color: {color}; font-size: 18px; font-weight: 700;"
            )


class _ModelHistoryRow(QWidget):
    """Single row in the Model History list."""

    def __init__(self, name: str, date_str: str, accuracy: str, status: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(6)

        left = QVBoxLayout()
        left.setSpacing(2)
        lbl_name = QLabel(name)
        lbl_name.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 600;")
        lbl_date = QLabel(date_str)
        lbl_date.setStyleSheet(f"color: {_SUB}; font-size: 11px;")
        left.addWidget(lbl_name)
        left.addWidget(lbl_date)
        layout.addLayout(left)
        layout.addStretch()

        lbl_acc = QLabel(accuracy)
        lbl_acc.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 600;")
        layout.addWidget(lbl_acc)
        layout.addSpacing(8)

        color   = _BLUE if status == "Deployed" else _SUB
        bg      = "#EEF2FF" if status == "Deployed" else "#F5F6FA"
        badge   = QLabel(status)
        badge.setFixedWidth(72)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"color: {color}; background: {bg}; border-radius: 4px; "
            f"font-size: 11px; font-weight: 600; padding: 2px 6px;"
        )
        layout.addWidget(badge)


# ── Preview row helper ───────────────────────────────────────────────────────

class _PreviewRow(QWidget):
    """Single key-value row for the FreqAI parameter preview."""

    def __init__(self, key: str, value: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        self._key_lbl = QLabel(key)
        self._key_lbl.setStyleSheet(
            f"color: {_SUB}; font-size: 12px; font-weight: 600;"
        )
        self._key_lbl.setFixedWidth(200)
        layout.addWidget(self._key_lbl)
        self._val_lbl = QLabel(value)
        self._val_lbl.setStyleSheet(
            f"color: {_TEXT}; font-size: 12px; font-weight: 500;"
        )
        self._val_lbl.setWordWrap(True)
        layout.addWidget(self._val_lbl, 1)

    def set_value(self, text: str) -> None:
        self._val_lbl.setText(text)


# ══════════════════════════════════════════════════════════════════════════════
# PREPARE DATASET DIALOG
# ══════════════════════════════════════════════════════════════════════════════


class PrepareDatasetDialog(QDialog):
    """Dialog for selecting only timerange; pairs/timeframes come from config."""

    _dl_log_signal = pyqtSignal(str, str)     # (level, message)
    _dl_done_signal = pyqtSignal(bool)        # success?

    def __init__(
        self,
        config_path: str,
        coverage_validator: Callable[[str], tuple[bool, str]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config_path = config_path
        self._coverage_validator = coverage_validator
        self._downloading = False
        self._download_ok = False
        self._selection_source = ""
        self._process: subprocess.Popen | None = None

        self.setWindowTitle("Prepare Dataset — Download OHLCV Data")
        self.setMinimumSize(560, 480)
        self.setStyleSheet(f"background-color: {_BG_PAGE};")

        self._dl_log_signal.connect(self._on_log)
        self._dl_done_signal.connect(self._on_done)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Download Historical Data")
        title.setStyleSheet(f"color: {_TEXT}; font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        pairs = self._read_config_pairs()
        timeframes = self._read_download_timeframes()
        desc = QLabel(
            "Select only date range.\n"
            "Pairs and timeframes are read from the active config file.\n"
            "Use Existing Data to continue without downloading.\n"
            f"Pairs: {', '.join(pairs) if pairs else '(none)'}\n"
            f"Timeframes: {', '.join(timeframes) if timeframes else '(none)'}"
        )
        desc.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        date_lbl = QLabel("Date Range")
        date_lbl.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 600;")
        layout.addWidget(date_lbl)

        date_row = QHBoxLayout()
        date_row.setSpacing(12)

        date_style = f"""
            QDateEdit {{
                background: {_BG}; border: 1px solid {_BORDER};
                border-radius: 6px; padding: 6px 10px;
                font-size: 13px; color: {_TEXT};
            }}
            QDateEdit::drop-down {{ border: none; width: 20px; }}
        """

        lbl_from = QLabel("From:")
        lbl_from.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
        date_row.addWidget(lbl_from)
        self._date_from = QDateEdit()
        self._date_from.setDate(QDate(2023, 1, 1))
        self._date_from.setCalendarPopup(True)
        self._date_from.setDisplayFormat("dd.MM.yyyy")
        self._date_from.setStyleSheet(date_style)
        date_row.addWidget(self._date_from)

        lbl_to = QLabel("To:")
        lbl_to.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
        date_row.addWidget(lbl_to)
        self._date_to = QDateEdit()
        self._date_to.setDate(QDate.currentDate())
        self._date_to.setCalendarPopup(True)
        self._date_to.setDisplayFormat("dd.MM.yyyy")
        self._date_to.setStyleSheet(date_style)
        date_row.addWidget(self._date_to)
        date_row.addStretch()
        layout.addLayout(date_row)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(140)
        self._log.setStyleSheet(
            "QTextEdit{background:#0D0D0D;color:#C0C0C0;border:1px solid #2C2C2C;"
            "border-radius:6px;font-family:'Cascadia Code','Consolas',monospace;"
            "font-size:11px;padding:8px;}"
        )
        layout.addWidget(self._log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.setMinimumWidth(100)
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 8px;
                font-size: 13px; font-weight: 600; padding: 0 16px;
            }}
            QPushButton:hover {{ background: #F0F0F0; }}
        """)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._use_existing_btn = QPushButton("  Use Existing Data")
        self._use_existing_btn.setIcon(qta.icon("fa6s.hard-drive", color=_TEXT))
        self._use_existing_btn.setIconSize(QSize(14, 14))
        self._use_existing_btn.setFixedHeight(36)
        self._use_existing_btn.setMinimumWidth(170)
        self._use_existing_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._use_existing_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 8px;
                font-size: 13px; font-weight: 700;
                padding: 0 16px;
            }}
            QPushButton:hover {{ background: #F0F0F0; }}
        """)
        self._use_existing_btn.clicked.connect(self._on_use_existing_data)
        btn_row.addWidget(self._use_existing_btn)

        self._download_btn = QPushButton("  Download Data")
        self._download_btn.setIcon(qta.icon("fa6s.download", color="#FFFFFF"))
        self._download_btn.setIconSize(QSize(14, 14))
        self._download_btn.setFixedHeight(36)
        self._download_btn.setMinimumWidth(160)
        self._download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_TEXT}; color: #FFF; border: none;
                border-radius: 8px; font-size: 13px; font-weight: 700;
                padding: 0 18px;
            }}
            QPushButton:hover {{ background: #2C3E6B; }}
        """)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)

        layout.addLayout(btn_row)

    def _read_config(self) -> dict:
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _read_config_pairs(self) -> list[str]:
        cfg = self._read_config()
        return cfg.get("exchange", {}).get("pair_whitelist", [])

    def _read_download_timeframes(self) -> list[str]:
        cfg = self._read_config()
        feature = cfg.get("freqai", {}).get("feature_parameters", {})
        include_tfs = feature.get("include_timeframes", []) or []
        base_tf = cfg.get("timeframe", "1h")
        result: list[str] = []
        for tf in [base_tf, *include_tfs]:
            if tf and tf not in result:
                result.append(tf)
        return result

    def _log_msg(self, level: str, msg: str) -> None:
        colors = {
            "info": "#C0C0C0",
            "warning": "#FFC107",
            "error": "#F44336",
            "success": "#4CAF50",
        }
        c = colors.get(level, "#C0C0C0")
        safe = html.escape(str(msg))
        self._log.append(f'<span style="color:{c}">[{level.upper().ljust(7)}] {safe}</span>')
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_log(self, level: str, msg: str) -> None:
        self._log_msg(level, msg)

    @property
    def selected_timerange(self) -> str:
        d_from = self._date_from.date()
        d_to = self._date_to.date()
        return (
            f"{d_from.year():04d}{d_from.month():02d}{d_from.day():02d}"
            f"-"
            f"{d_to.year():04d}{d_to.month():02d}{d_to.day():02d}"
        )

    @property
    def download_succeeded(self) -> bool:
        return self._download_ok

    @property
    def selection_source(self) -> str:
        return self._selection_source

    def _on_done(self, success: bool) -> None:
        self._downloading = False
        self._download_ok = success
        self._download_btn.setEnabled(True)
        self._use_existing_btn.setEnabled(True)
        self._download_btn.setText("  Download Data")
        self._download_btn.setIcon(qta.icon("fa6s.download", color="#FFFFFF"))
        if success:
            self._selection_source = "download"
            self._log_msg("success", "Data download completed successfully!")
            self.accept()
        else:
            self._log_msg("error", "Data download failed. Check the log above.")

    def _on_use_existing_data(self) -> None:
        """Use already-downloaded disk data for the selected timerange."""
        d_from = self._date_from.date()
        d_to = self._date_to.date()
        if d_from >= d_to:
            self._log_msg("error", "Start date must be before end date.")
            return

        timerange = self.selected_timerange
        if self._coverage_validator is not None:
            ok, reason = self._coverage_validator(timerange)
            if not ok:
                self._log_msg("error", reason)
                return

        self._download_ok = True
        self._selection_source = "existing"
        self._log_msg("success", f"Using existing disk data for timerange={timerange}")
        self.accept()

    def _on_cancel(self) -> None:
        if self._downloading and self._process is not None:
            self._process.terminate()
            self._downloading = False
        self.reject()

    def closeEvent(self, event) -> None:
        if self._downloading and self._process is not None:
            self._process.terminate()
        super().closeEvent(event)

    def _on_download(self) -> None:
        pairs = self._read_config_pairs()
        if not pairs:
            self._log_msg("error", "pair_whitelist is empty in config file.")
            return

        timeframes = self._read_download_timeframes()
        if not timeframes:
            self._log_msg("error", "No timeframe found in config.")
            return

        d_from = self._date_from.date()
        d_to = self._date_to.date()
        if d_from >= d_to:
            self._log_msg("error", "Start date must be before end date.")
            return

        timerange = self.selected_timerange

        self._downloading = True
        self._download_btn.setEnabled(False)
        self._use_existing_btn.setEnabled(False)
        self._download_btn.setText("Downloading…")
        self._log.clear()
        self._log_msg(
            "info",
            f"Downloading data for config pairs ({len(pairs)}) with timerange={timerange}",
        )

        config_name = Path(self._config_path).name

        threading.Thread(
            target=self._run_download,
            args=(config_name, timerange, pairs, timeframes),
            daemon=True,
        ).start()

    def _run_download(
        self,
        config_name: str,
        timerange: str,
        pairs: list[str],
        timeframes: list[str],
    ) -> None:
        """Run docker compose download-data in a background thread."""
        project_root = Path(__file__).resolve().parent.parent

        docker_cmd = BotProcessManager._find_docker_compose()
        if docker_cmd is None:
            self._dl_log_signal.emit("error", "Docker not found. Install Docker Desktop.")
            self._dl_done_signal.emit(False)
            return

        cmd = [
            *docker_cmd,
            "-f",
            str(project_root / "docker-compose.yml"),
            "run",
            "--rm",
            "freqtrade",
            "download-data",
            "--config",
            f"/freqtrade/config/{config_name}",
            "--timerange",
            timerange,
            "--timeframes",
            *timeframes,
            "--pairs",
            *pairs,
        ]
        self._dl_log_signal.emit("info", f"$ {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            self._dl_log_signal.emit("error", f"Failed to start Docker: {exc}")
            self._dl_done_signal.emit(False)
            return

        try:
            for line in self._process.stdout:
                stripped = line.rstrip("\n\r")
                if stripped:
                    self._dl_log_signal.emit("info", stripped)
        except Exception:
            pass

        rc = self._process.wait()
        self._process = None
        self._dl_done_signal.emit(rc == 0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainingPage(QWidget):
    """Model Training & Bot Lifecycle page."""

    # Bridge signals (worker → GUI thread)
    _login_done_signal = pyqtSignal(object)
    _bot_state_signal  = pyqtSignal(object)
    _ws_state_signal   = pyqtSignal(object)
    _log_signal        = pyqtSignal(object)

    def __init__(self, trade_manager: TradeManager, parent=None) -> None:
        super().__init__(parent)
        self._tm = trade_manager
        self._dark = False
        self._config_path: str = ""
        self._config_dict: dict = {}
        self._config_provider: Callable[[], str] | None = None
        self._dataset_timerange: str = ""
        self._training_active = False
        self._train_started_ts = 0.0
        self._epoch = 0
        self._max_epoch = 100
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(1500)
        self._metrics_timer.timeout.connect(self._refresh_runtime_metrics)

        self.setStyleSheet(f"background-color: {_BG_PAGE};")
        self._init_ui()
        self._connect_signals()
        self._load_model_history()

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        root.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)

        # ── Page title ────────────────────────────────────────────────────
        title = QLabel("Model Training")
        title.setStyleSheet(f"color: {_TEXT}; font-size: 22px; font-weight: 700;")
        self._title_lbl = title
        subtitle = QLabel(
            "Train your FreqAI model from Bot Configuration settings."
        )
        subtitle.setStyleSheet(f"color: {_SUB}; font-size: 13px;")
        self._subtitle_lbl = subtitle
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # ── Step 1: Load Config ───────────────────────────────────────────
        layout.addWidget(self._build_config_card())

        # ── Step 2: FreqAI Parameters Preview ────────────────────────────
        layout.addWidget(self._build_preview_card())

        # ── Step 3: Toolbar (Prepare Dataset + Learn) ──────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        self._prepare_btn = QPushButton("  Prepare Dataset")
        self._prepare_btn.setIcon(qta.icon("fa6s.database", color=_TEXT))
        self._prepare_btn.setIconSize(QSize(14, 14))
        self._prepare_btn.setFixedHeight(38)
        self._prepare_btn.setMinimumWidth(160)
        self._prepare_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._prepare_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 8px;
                font-size: 13px; font-weight: 700; padding: 0 18px;
            }}
            QPushButton:hover {{ background: #F0F0F0; }}
        """)
        self._prepare_btn.setEnabled(False)
        self._prepare_btn.clicked.connect(self._on_prepare_dataset)
        toolbar.addWidget(self._prepare_btn)

        self._train_btn = QPushButton("  Start Learning")
        self._train_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
        self._train_btn.setIconSize(QSize(14, 14))
        self._train_btn.setFixedHeight(38)
        self._train_btn.setMinimumWidth(160)
        self._train_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._train_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_TEXT}; color: #FFF; border: none;
                border-radius: 8px; font-size: 13px; font-weight: 700;
                padding: 0 18px;
            }}
            QPushButton:hover {{ background: #2C3E6B; }}
        """)
        self._train_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._on_train_clicked)
        toolbar.addWidget(self._train_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # ── Main row (Training Progress + right column) ──────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(20)

        main_row.addWidget(self._build_progress_card(), stretch=2)

        right_col = QVBoxLayout()
        right_col.setSpacing(16)
        right_col.addWidget(self._build_history_card(), stretch=1)
        right_col.addWidget(self._build_tips_card(), stretch=0)

        right_widget = QWidget()
        right_widget.setLayout(right_col)
        main_row.addWidget(right_widget, stretch=1)

        layout.addLayout(main_row)

        # ── Bot Log ──────────────────────────────────────────────────────
        layout.addWidget(self._build_log_card())

        layout.addStretch()

    # ── Config loading card ────────────────────────────────────────────────

    def _build_config_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_card_style())
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(
            qta.icon("fa6s.file-code", color=_SUB).pixmap(QSize(18, 18))
        )
        icon_lbl.setStyleSheet("border: none;")
        self._config_icon = icon_lbl
        title = QLabel("Config")
        title.setStyleSheet(
            f"color: {_TEXT}; font-size: 15px; font-weight: 700;"
        )
        hdr.addWidget(icon_lbl)
        hdr.addWidget(title)
        hdr.addStretch()

        # Bot status badge (inline in header)
        self._conn_badge = QLabel("Disconnected")
        self._conn_badge.setStyleSheet(
            f"{_DOT} background:#2C2C2C; color:#F44336;"
        )
        hdr.addWidget(self._conn_badge)

        self._bot_badge = QLabel("Stopped")
        self._bot_badge.setStyleSheet(
            f"{_DOT} background:#2C2C2C; color:#A0A0A0;"
        )
        hdr.addWidget(self._bot_badge)
        lay.addLayout(hdr)

        # Load config row
        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(10)

        self._load_btn = QPushButton("  Load Config File")
        self._load_btn.setIcon(qta.icon("fa6s.folder-open", color="#FFFFFF"))
        self._load_btn.setIconSize(QSize(14, 14))
        self._load_btn.setFixedHeight(36)
        self._load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._load_btn.setStyleSheet(_btn_qss(_BLUE))
        self._load_btn.clicked.connect(self._on_load_config)
        cfg_row.addWidget(self._load_btn)

        self._config_path_lbl = QLabel("No config loaded")
        self._config_path_lbl.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
        cfg_row.addWidget(self._config_path_lbl, 1)

        lay.addLayout(cfg_row)

        # Dataset info badge (populated after Prepare Dataset dialog)
        self._dataset_info_lbl = QLabel("No dataset prepared. Click 'Prepare Dataset' first.")
        self._dataset_info_lbl.setStyleSheet(f"color: {_SUB}; font-size: 11px; padding: 2px 0;")
        self._dataset_info_lbl.setWordWrap(True)
        lay.addWidget(self._dataset_info_lbl)

        return card
    # ── FreqAI Parameters Preview card ────────────────────────────────────

    def _build_preview_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_card_style())
        self._preview_card = card

        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(
            qta.icon("fa6s.brain", color=_BLUE).pixmap(QSize(18, 18))
        )
        icon_lbl.setStyleSheet("border: none;")
        self._preview_icon = icon_lbl
        title = QLabel("FreqAI Parameters Preview")
        title.setStyleSheet(
            f"color: {_TEXT}; font-size: 15px; font-weight: 700;"
        )
        hdr.addWidget(icon_lbl)
        hdr.addWidget(title)
        hdr.addStretch()
        lay.addLayout(hdr)

        self._preview_placeholder = QLabel(
            "Load a config file to see FreqAI parameters."
        )
        self._preview_placeholder.setStyleSheet(
            f"color: {_SUB}; font-size: 12px; padding: 12px 0;"
        )
        lay.addWidget(self._preview_placeholder)

        # Grid for preview rows (hidden until config loaded)
        self._preview_grid = QWidget()
        grid_lay = QVBoxLayout(self._preview_grid)
        grid_lay.setContentsMargins(0, 4, 0, 0)
        grid_lay.setSpacing(0)

        self._pv_enabled      = _PreviewRow("Enabled", "—")
        self._pv_model        = _PreviewRow("ML Model", "—")
        self._pv_identifier   = _PreviewRow("Identifier", "—")
        self._pv_train_days   = _PreviewRow("Train Period (days)", "—")
        self._pv_purge        = _PreviewRow("Purge Old Models", "—")
        self._pv_timeframes   = _PreviewRow("Include Timeframes", "—")
        self._pv_corr_pairs   = _PreviewRow("Correlation Pairlist", "—")
        self._pv_label_period = _PreviewRow("Label Period (candles)", "—")
        self._pv_shifted      = _PreviewRow("Shifted Candles", "—")
        self._pv_indicators   = _PreviewRow("Indicator Periods", "—")
        self._pv_svm          = _PreviewRow("SVM Outlier Removal", "—")
        self._pv_test_size    = _PreviewRow("Test Size", "—")
        self._pv_shuffle      = _PreviewRow("Shuffle", "—")

        for row in [
            self._pv_enabled, self._pv_model, self._pv_identifier,
            self._pv_train_days, self._pv_purge,
            self._pv_timeframes, self._pv_corr_pairs,
            self._pv_label_period, self._pv_shifted,
            self._pv_indicators, self._pv_svm,
            self._pv_test_size, self._pv_shuffle,
        ]:
            grid_lay.addWidget(row)

        self._preview_grid.setVisible(False)
        lay.addWidget(self._preview_grid)

        return card

    # ── Training Progress card ────────────────────────────────────────────

    def _build_progress_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_card_style())
        card.setMinimumHeight(280)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Header
        hdr = QHBoxLayout()
        lbl_icon = QLabel()
        lbl_icon.setPixmap(
            qta.icon("fa6s.gear", color=_SUB).pixmap(QSize(18, 18))
        )
        lbl_icon.setStyleSheet("border: none;")
        self._progress_icon = lbl_icon
        lbl_title = QLabel("Training Progress")
        lbl_title.setStyleSheet(
            f"color: {_TEXT}; font-size: 15px; font-weight: 700;"
        )
        hdr.addWidget(lbl_icon)
        hdr.addWidget(lbl_title)
        hdr.addStretch()
        layout.addLayout(hdr)

        self._lbl_model_name = QLabel("Model: —")
        self._lbl_model_name.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
        layout.addWidget(self._lbl_model_name)

        # Epoch progress
        epoch_row = QHBoxLayout()
        self._lbl_epoch = QLabel("Epoch 0/100")
        self._lbl_epoch.setStyleSheet(
            f"color: {_TEXT}; font-size: 13px; font-weight: 600;"
        )
        epoch_row.addWidget(self._lbl_epoch)
        epoch_row.addStretch()
        self._lbl_epoch_pct = QLabel("0%")
        self._lbl_epoch_pct.setStyleSheet(f"color: {_SUB}; font-size: 13px;")
        epoch_row.addWidget(self._lbl_epoch_pct)
        layout.addLayout(epoch_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #F0F0F0; border: none; border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {_TEXT}; border-radius: 4px;
            }}
        """)
        layout.addWidget(self._progress_bar)

        # Metrics grid (2×2)
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(32)

        left_metrics = QVBoxLayout()
        self._row_loss = _MetricRow("Current Loss", "—", _TEXT)
        self._row_time = _MetricRow("Elapsed Time", "00:00:00", _TEXT)
        left_metrics.addWidget(self._row_loss)
        left_metrics.addWidget(self._row_time)

        right_metrics = QVBoxLayout()
        self._row_cpu = _MetricRow("CPU Usage", "—", _GREEN)
        self._row_ram = _MetricRow("RAM Usage", "—", _ORANGE)
        right_metrics.addWidget(self._row_cpu)
        right_metrics.addWidget(self._row_ram)

        metrics_layout.addLayout(left_metrics)
        metrics_layout.addLayout(right_metrics)
        layout.addLayout(metrics_layout)

        layout.addStretch()

        # Stop / Pause buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._stop_train_btn = QPushButton("  Stop")
        self._stop_train_btn.setIcon(
            qta.icon("fa6s.circle-stop", color=_TEXT)
        )
        self._stop_train_btn.setIconSize(QSize(14, 14))
        self._stop_train_btn.setFixedSize(90, 34)
        self._stop_train_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_train_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 6px;
                font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #FFE5E5; color: {_RED}; }}
        """)
        self._stop_train_btn.clicked.connect(self._on_stop_training)

        self._pause_btn = QPushButton("  Pause")
        self._pause_btn.setIcon(qta.icon("fa6s.pause", color=_TEXT))
        self._pause_btn.setIconSize(QSize(14, 14))
        self._pause_btn.setFixedSize(90, 34)
        self._pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pause_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 6px;
                font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #FFF3E0; color: {_ORANGE}; }}
        """)

        btn_row.addWidget(self._stop_train_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(self._pause_btn)
        layout.addLayout(btn_row)

        return card

    # ── Model History card ────────────────────────────────────────────────

    def _build_history_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_card_style())

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        lbl = QLabel("Model History")
        lbl.setStyleSheet(
            f"color: {_TEXT}; font-size: 15px; font-weight: 700;"
        )
        layout.addWidget(lbl)
        layout.addSpacing(4)

        self._history_layout = QVBoxLayout()
        self._history_layout.setSpacing(0)
        layout.addLayout(self._history_layout)
        layout.addStretch()

        return card

    # ── Tips card ─────────────────────────────────────────────────────────

    def _build_tips_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #F8F9FF;
                border-radius: 12px;
                border: 1px solid #DDE3FF;
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(6)

        hdr = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(
            qta.icon("fa6s.lightbulb", color=_BLUE).pixmap(QSize(16, 16))
        )
        icon.setStyleSheet("border: none;")
        self._tips_icon = icon
        lbl = QLabel("Training Tips")
        lbl.setStyleSheet(
            f"color: {_TEXT}; font-size: 13px; font-weight: 700;"
        )
        hdr.addWidget(icon)
        hdr.addWidget(lbl)
        hdr.addStretch()
        layout.addLayout(hdr)

        tip = QLabel(
            "Ensure your dataset covers at least 6 months of historical data "
            "for better trend prediction accuracy. Use a mix of bull and bear "
            "market cycles."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
        layout.addWidget(tip)

        return card

    # ── Bot Log card ──────────────────────────────────────────────────────

    def _build_log_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_card_style())
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(8)

        hdr = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(
            qta.icon("fa6s.terminal", color=_SUB).pixmap(QSize(16, 16))
        )
        icon_lbl.setStyleSheet("border: none;")
        self._log_icon = icon_lbl
        lbl = QLabel("Bot Log")
        lbl.setStyleSheet(
            f"color: {_TEXT}; font-size: 15px; font-weight: 700;"
        )
        hdr.addWidget(icon_lbl)
        hdr.addWidget(lbl)
        hdr.addStretch()
        v.addLayout(hdr)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet(
            "QTextEdit{background:#0D0D0D;color:#C0C0C0;border:1px solid #2C2C2C;"
            "border-radius:6px;font-family:'Cascadia Code','Consolas',monospace;"
            "font-size:12px;padding:8px;}"
        )
        v.addWidget(self._log)
        return card

    def set_config_provider(self, provider: Callable[[], str]) -> None:
        """Inject callback that writes Bot Configuration and returns config path."""
        self._config_provider = provider
        self._train_btn.setEnabled(True)

    def get_selected_config_path(self) -> str:
        """Return config selected on Model Training page for live trading startup."""
        if self._config_path and Path(self._config_path).exists():
            return self._config_path
        raise ValueError("Please load a config file on the Model Training page first.")

    # ══════════════════════════════════════════════════════════════════════
    # SIGNAL WIRING
    # ══════════════════════════════════════════════════════════════════════

    def _connect_signals(self) -> None:
        self._login_done_signal.connect(self._slot_login)
        self._bot_state_signal.connect(self._slot_bot_state)
        self._ws_state_signal.connect(self._slot_ws_state)
        self._log_signal.connect(self._slot_log)

        # Wire TradeManager callbacks that MainWindow doesn't use
        self._tm.on_bot_state_change = self._bot_state_signal.emit
        self._tm.on_ws_state_change  = self._ws_state_signal.emit
        self._tm.on_log              = self._log_signal.emit

    # ══════════════════════════════════════════════════════════════════════
    # MODEL HISTORY FROM DISK
    # ══════════════════════════════════════════════════════════════════════

    def _load_model_history(self) -> None:
        """Scan user_data/models/ and populate the history panel."""
        rows: list[tuple[str, str, str, str]] = []
        if _MODELS_DIR.exists():
            for model_dir in sorted(_MODELS_DIR.iterdir()):
                if not model_dir.is_dir():
                    continue
                meta_file = model_dir / "global_metadata.json"
                run_file  = model_dir / "run_params.json"
                try:
                    meta = (
                        json.loads(meta_file.read_text())
                        if meta_file.exists() else {}
                    )
                    run = (
                        json.loads(run_file.read_text())
                        if run_file.exists() else {}
                    )
                    accuracy = meta.get(
                        "accuracy",
                        run.get("model_validation_score", None),
                    )
                    acc_str = (
                        f"{accuracy * 100:.1f}%"
                        if accuracy and accuracy <= 1
                        else (
                            f"{accuracy:.1f}%" if accuracy else "—"
                        )
                    )
                    date_str = meta.get("train_end_date", "")[:10] or "—"
                    rows.append((model_dir.name, date_str, acc_str, "Archived"))
                except Exception:
                    rows.append((model_dir.name, "—", "—", "Archived"))

        # Mark newest as Deployed
        if rows:
            rows[-1] = (rows[-1][0], rows[-1][1], rows[-1][2], "Deployed")

        # Fallback
        if not rows:
            rows = [
                ("LSTM-v1",         "2024-03-15", "84.2%", "Deployed"),
                ("XGBoost-Trend",   "2024-03-10", "79.5%", "Archived"),
                ("RandomForest-v4", "2024-03-05", "81.1%", "Archived"),
            ]

        for i, (name, date_str, acc, status) in enumerate(reversed(rows)):
            row_widget = _ModelHistoryRow(name, date_str, acc, status)
            self._history_layout.addWidget(row_widget)
            if i < len(rows) - 1:
                div = QFrame()
                div.setFrameShape(QFrame.Shape.HLine)
                div.setStyleSheet(f"color: {_BORDER};")
                self._history_layout.addWidget(div)

    # ══════════════════════════════════════════════════════════════════════
    # ACTION HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_load_config(self) -> None:
        start = str(Path(__file__).resolve().parent.parent / "config")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Freqtrade Config", start, "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            self._log_msg("error", f"Failed to read config: {exc}")
            return
        self._apply_config(path, cfg, source_label="Loaded")

    def _apply_config(self, path: str, cfg: dict, source_label: str = "Loaded") -> None:
        self._config_path = path
        self._config_dict = cfg
        self._dataset_timerange = ""
        self._config_path_lbl.setText(Path(path).name)
        self._dataset_info_lbl.setText("No dataset prepared. Click 'Prepare Dataset' first.")
        self._dataset_info_lbl.setStyleSheet(f"color: {_SUB}; font-size: 11px; padding: 2px 0;")
        self._log_msg("info", f"{source_label} config: {Path(path).name}")

        fai = cfg.get("freqai", {})
        if fai:
            self._populate_preview(fai)
            self._preview_placeholder.setVisible(False)
            self._preview_grid.setVisible(True)
        else:
            self._preview_placeholder.setText("No FreqAI section found in config.")
            self._preview_placeholder.setVisible(True)
            self._preview_grid.setVisible(False)

        # If a provider exists, user can still click and we'll validate at runtime.
        self._train_btn.setEnabled(bool(fai.get("enabled", False)) or self._config_provider is not None)
        self._prepare_btn.setEnabled(True)

        model_name = fai.get("model_name", fai.get("identifier", "—"))
        identifier = fai.get("identifier", "")
        display = f"{model_name} ({identifier})" if identifier else model_name
        self._lbl_model_name.setText(f"Model: {display}")

    def _populate_preview(self, fai: dict) -> None:
        """Fill the FreqAI preview card from parsed config dict."""
        fp = fai.get("feature_parameters", {})
        dsp = fai.get("data_split_parameters", {})

        self._pv_enabled.set_value(
            "Yes" if fai.get("enabled", False) else "No"
        )
        self._pv_model.set_value(fai.get("model_name", "—"))
        self._pv_identifier.set_value(fai.get("identifier", "—"))
        self._pv_train_days.set_value(
            str(fai.get("train_period_days", "—"))
        )
        self._pv_purge.set_value(
            "Yes" if fai.get("purge_old_models", False) else "No"
        )

        tfs = fp.get("include_timeframes", [])
        self._pv_timeframes.set_value(
            ", ".join(tfs) if isinstance(tfs, list) else str(tfs)
        )
        cp = fp.get("include_corr_pairlist", [])
        self._pv_corr_pairs.set_value(
            ", ".join(cp) if isinstance(cp, list) else str(cp)
        )
        self._pv_label_period.set_value(
            str(fp.get("label_period_candles", "—"))
        )
        self._pv_shifted.set_value(
            str(fp.get("include_shifted_candles", "—"))
        )
        ip = fp.get("indicator_periods_candles", [])
        self._pv_indicators.set_value(
            ", ".join(str(x) for x in ip)
            if isinstance(ip, list) else str(ip)
        )
        self._pv_svm.set_value(
            "Yes"
            if fp.get("use_SVM_to_remove_outliers", False)
            else "No"
        )
        self._pv_test_size.set_value(str(dsp.get("test_size", "—")))
        self._pv_shuffle.set_value(
            "Yes" if dsp.get("shuffle", False) else "No"
        )

    def _on_prepare_dataset(self) -> None:
        """Open dataset dialog and keep the chosen timerange for training."""
        if not self._config_path or not Path(self._config_path).exists():
            self._log_msg("error", "Load a config file first.")
            return

        dlg = PrepareDatasetDialog(
            self._config_path,
            coverage_validator=self._validate_dataset_coverage,
            parent=self,
        )
        dlg.exec()

        if dlg.download_succeeded:
            self._dataset_timerange = dlg.selected_timerange
            source_text = "downloaded" if dlg.selection_source == "download" else "existing"
            self._dataset_info_lbl.setText(
                f"Dataset ready ({source_text}) for timerange: {self._dataset_timerange}"
            )
            self._dataset_info_lbl.setStyleSheet(
                f"color: {_GREEN}; font-size: 11px; font-weight: 600; padding: 2px 0;"
            )

    def _validate_dataset_coverage(self, timerange: str) -> tuple[bool, str]:
        """Validate local data files cover the required FreqAI training horizon."""
        if not re.match(r"^\d{8}-\d{8}$", timerange):
            return False, "Invalid timerange format. Expected YYYYMMDD-YYYYMMDD."

        timerange_start = datetime.strptime(timerange.split("-")[0], "%Y%m%d")

        cfg = self._config_dict
        exchange = cfg.get("exchange", {}).get("name", "binance")
        pairs: list[str] = cfg.get("exchange", {}).get("pair_whitelist", [])
        if not pairs:
            return False, "pair_whitelist is empty in config."

        freqai_cfg = cfg.get("freqai", {})
        feature_cfg = freqai_cfg.get("feature_parameters", {})
        train_days = int(freqai_cfg.get("train_period_days", 0) or 0)

        # Strategy startup candles (warmup before first usable signal).
        # Keep this configurable with a safe default when absent in config.
        strategy_startup_candles = int(cfg.get("startup_candle_count", 200) or 200)

        base_tf = cfg.get("timeframe", "1h")
        include_tfs = feature_cfg.get("include_timeframes", []) or []
        timeframes: list[str] = []
        for tf in [base_tf, *include_tfs]:
            if tf and tf not in timeframes:
                timeframes.append(tf)

        if not timeframes:
            return False, "No timeframe found in config."

        data_dir = Path("user_data") / "data" / exchange
        if not data_dir.exists():
            return False, f"Data directory not found: {data_dir}"

        missing_files: list[str] = []
        insufficient: list[str] = []

        for pair in pairs:
            token = pair.replace("/", "_")
            for tf in timeframes:
                candles_per_day = _CANDLES_PER_DAY.get(tf)
                if not candles_per_day:
                    return False, f"Unsupported timeframe in config: {tf}"

                # Required history before timerange start:
                # 1) train_period_days (FreqAI training window)
                # 2) strategy startup candles converted to days for this timeframe
                startup_days_for_tf = strategy_startup_candles / candles_per_day
                required_start = timerange_start - timedelta(days=(train_days + startup_days_for_tf))

                fpath = data_dir / f"{token}-{tf}.feather"
                if not fpath.exists():
                    missing_files.append(f"{pair} {tf} (file missing)")
                    continue

                try:
                    df = pd.read_feather(fpath, columns=["date"])
                    if df.empty:
                        missing_files.append(f"{pair} {tf} (empty file)")
                        continue
                    first = pd.to_datetime(df["date"].iloc[0]).to_pydatetime().replace(tzinfo=None)
                except Exception as exc:
                    missing_files.append(f"{pair} {tf} (read error: {exc})")
                    continue

                if first > required_start:
                    insufficient.append(
                        f"{pair} {tf}: starts {first.date()} but needs <= {required_start.date()}"
                    )

        if missing_files or insufficient:
            parts = [
                "Selected timerange does not have enough local history for training.",
                "Run Prepare Dataset with an earlier start date.",
            ]
            if missing_files:
                parts.append("Missing data:")
                parts.extend(f"  - {x}" for x in missing_files[:8])
            if insufficient:
                parts.append("Insufficient coverage:")
                parts.extend(f"  - {x}" for x in insufficient[:8])
            return False, "\n".join(parts)

        return True, ""

    def _on_train_clicked(self) -> None:
        """Train FreqAI model via ``freqtrade backtesting --freqai``."""
        if self._training_active:
            return

        # Use the config explicitly loaded in this page first.
        # Fall back to provider-generated runtime config only if none is loaded.
        has_loaded_config = bool(self._config_path) and Path(self._config_path).exists()
        if not has_loaded_config and self._config_provider is not None:
            try:
                config_path = self._config_provider()
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self._apply_config(config_path, cfg, source_label="Synced")
            except Exception as exc:
                self._log_msg("error", f"Bot config sync failed: {exc}")
                return

        if not self._config_path:
            self._log_msg("error", "Load a config file first or configure Bot Configuration page.")
            return

        fai = self._config_dict.get("freqai", {})
        if not fai.get("enabled", False):
            self._log_msg("error", "FreqAI is not enabled in the active config.")
            return

        # Use timerange from Prepare Dataset dialog
        timerange = getattr(self, "_dataset_timerange", "")
        if not timerange:
            self._log_msg(
                "error",
                "No dataset prepared. Click 'Prepare Dataset' to download data and set the timerange first.",
            )
            return

        self._training_active = True
        self._train_started_ts = time.monotonic()
        self._reset_progress()
        self._set_training_ui_state(active=True)
        self._metrics_timer.start()

        self._log_msg(
            "info",
            f"Starting FreqAI model training via freqtrade backtesting "
            f"(timerange={timerange}, config={Path(self._config_path).name})",
        )

        try:
            self._tm.configure_from_config(self._config_path)
            self._tm.start_learning(timerange=timerange)
        except Exception as exc:
            self._log_msg("error", f"Failed to start training: {exc}")
            self._training_active = False
            self._metrics_timer.stop()
            self._set_training_ui_state(active=False)

    def _on_stop_training(self) -> None:
        if self._training_active:
            self._tm.stop_bot()
            self._training_active = False
            self._metrics_timer.stop()
            self._set_training_ui_state(active=False)
            self._log_msg("info", "Training stopped by user.")

    def _set_training_ui_state(self, active: bool) -> None:
        self._train_btn.setEnabled(not active)
        self._train_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
        self._train_btn.setText("  Learning…" if active else "  Start Learning")

    def _reset_progress(self) -> None:
        self._epoch = 0
        self._max_epoch = 100
        self._progress_bar.setValue(0)
        self._lbl_epoch.setText("Epoch 0/100")
        self._lbl_epoch_pct.setText("0%")
        self._row_loss.set_value("—")
        self._row_time.set_value("00:00:00")
        self._row_cpu.set_value("—")
        self._row_ram.set_value("—")

    def _refresh_runtime_metrics(self) -> None:
        if not self._training_active:
            return

        elapsed = max(0.0, time.monotonic() - self._train_started_ts)
        self._row_time.set_value(self._format_elapsed(elapsed))

        if psutil is None:
            self._row_cpu.set_value("N/A")
            self._row_ram.set_value("N/A")
            return

        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            self._row_cpu.set_value(f"{cpu:.1f}%", _GREEN)
            self._row_ram.set_value(
                f"{mem.percent:.1f}% ({used_gb:.1f}/{total_gb:.1f} GB)",
                _ORANGE,
            )
        except Exception:
            self._row_cpu.set_value("N/A")
            self._row_ram.set_value("N/A")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = int(seconds)
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    # ══════════════════════════════════════════════════════════════════════
    # SLOTS (GUI thread)
    # ══════════════════════════════════════════════════════════════════════

    def _slot_login(self, result: ApiResult) -> None:
        if result.success:
            self._log_msg("info", "Login successful!")
        else:
            self._log_msg(
                "error", f"Login failed: {result.user_message}"
            )

    def update_connection_status(self, status: OverallStatus) -> None:
        """Called by MainWindow when connection status changes."""
        col, txt = _CONN_COLOURS.get(status, ("#A0A0A0", "Unknown"))
        self._conn_badge.setText(txt)
        self._conn_badge.setStyleSheet(
            f"{_DOT} background:#2C2C2C; color:{col};"
        )

    def _slot_bot_state(self, state: BotState) -> None:
        col, txt = _BOT_COLOURS.get(state, ("#A0A0A0", "Unknown"))
        self._bot_badge.setText(txt)
        self._bot_badge.setStyleSheet(
            f"{_DOT} background:#2C2C2C; color:{col};"
        )

        if state == BotState.RUNNING:
            self._log_msg(
                "info",
                "freqtrade backtesting is running — collecting training metrics.",
            )
        elif state == BotState.STOPPED and self._training_active:
            self._training_active = False
            self._metrics_timer.stop()
            self._set_training_ui_state(active=False)
            self._log_msg("info", "Training command finished.")
        elif state == BotState.ERROR:
            self._training_active = False
            self._metrics_timer.stop()
            self._set_training_ui_state(active=False)
            self._log_msg("error", "Bot process encountered an error.")

    def _slot_ws_state(self, state: WSState) -> None:
        self._log_msg("info", f"WebSocket: {state.name}")

    def _slot_log(self, event: BotLogEvent) -> None:
        self._log_msg(event.level, event.message)
        self._parse_training_log(event.message)

    def _parse_training_log(self, message: str) -> None:
        """Parse FreqAI log lines for training progress metrics."""
        # Epoch: "epoch X/Y"
        m = re.search(r"(?:epoch|iter(?:ation)?)\s+(\d+)\s*/\s*(\d+)", message, re.IGNORECASE)
        if m:
            epoch = int(m.group(1))
            total = int(m.group(2))
            self._epoch = epoch
            self._max_epoch = total
            pct = int(epoch / total * 100) if total > 0 else 0
            self._progress_bar.setValue(pct)
            self._lbl_epoch.setText(f"Epoch {epoch}/{total}")
            self._lbl_epoch_pct.setText(f"{pct}%")

        # Loss: "loss: 0.0345" or "loss=0.0345"
        m = re.search(
            r"loss[=:\s]+(\d+\.?\d*)", message, re.IGNORECASE
        )
        if m:
            self._row_loss.set_value(m.group(1))

        # CPU/RAM lines if backend logs them explicitly.
        m = re.search(r"cpu[^\d]*(\d+(?:\.\d+)?)\s*%", message, re.IGNORECASE)
        if m:
            self._row_cpu.set_value(f"{float(m.group(1)):.1f}%", _GREEN)

        m = re.search(r"ram[^\d]*(\d+(?:\.\d+)?)\s*%", message, re.IGNORECASE)
        if m:
            self._row_ram.set_value(f"{float(m.group(1)):.1f}%", _ORANGE)

        if "completed successfully" in message.lower() and self._training_active:
            self._progress_bar.setValue(100)
            self._lbl_epoch_pct.setText("100%")

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _log_msg(self, level: str, msg: str) -> None:
        colors = {
            "info": "#C0C0C0",
            "warning": "#FFC107",
            "error": "#F44336",
        }
        c = colors.get(level, "#C0C0C0")
        safe = html.escape(str(msg))
        self._log.append(
            f'<span style="color:{c}">'
            f"[{level.upper().ljust(7)}] {safe}</span>"
        )
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ══════════════════════════════════════════════════════════════════════
    # THEMING
    # ══════════════════════════════════════════════════════════════════════

    def apply_theme(self, dark: bool) -> None:
        self._dark = dark
        C = get_colors(dark)
        self.setStyleSheet(f"background-color: {C['PAGE_BG']};")

        # Update all card frames
        card_qss = f"""
            QFrame {{
                background-color: {C['CARD_BG']}; border-radius: 12px;
            }}
        """
        for frame in self.findChildren(QFrame):
            ss = frame.styleSheet()
            if "border-radius: 12px" in ss:
                frame.setStyleSheet(card_qss)

        # Tips card
        tips_bg = "#1A2040" if dark else "#F8F9FF"
        for frame in self.findChildren(QFrame):
            ss = frame.styleSheet()
            if "F8F9FF" in ss or "1A2040" in ss:
                frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: {tips_bg}; border-radius: 12px;
                    }}
                """)
                break

        # Progress bar
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C['INPUT_BG']}; border: none; border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {C['BLUE']}; border-radius: 4px;
            }}
        """)

        # Title labels
        self._title_lbl.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 22px; font-weight: 700;"
        )
        self._subtitle_lbl.setStyleSheet(
            f"color: {C['SUBTEXT']}; font-size: 13px;"
        )

        for lbl in self.findChildren(QLabel):
            ss = lbl.styleSheet()
            if "font-size: 15px" in ss and "font-weight: 700" in ss:
                lbl.setStyleSheet(
                    ss.replace("#1A1A2E", C["TEXT"]).replace(
                        "#E0E0E0", C["TEXT"]
                    )
                )
            elif (
                "font-size: 13px; font-weight: 600" in ss
                and "#1A1A2E" in ss
            ):
                lbl.setStyleSheet(ss.replace("#1A1A2E", C["TEXT"]))

        # Button icons
        self._stop_train_btn.setIcon(
            qta.icon("fa6s.circle-stop", color=C["TEXT"])
        )
        self._pause_btn.setIcon(
            qta.icon("fa6s.pause", color=C["TEXT"])
        )
        self._progress_icon.setPixmap(
            qta.icon("fa6s.microchip", color=C["SUBTEXT"]).pixmap(
                QSize(18, 18)
            )
        )
        self._tips_icon.setPixmap(
            qta.icon("fa6s.lightbulb", color=C["BLUE"]).pixmap(
                QSize(16, 16)
            )
        )
        self._config_icon.setPixmap(
            qta.icon("fa6s.file-code", color=C["SUBTEXT"]).pixmap(
                QSize(18, 18)
            )
        )
        self._preview_icon.setPixmap(
            qta.icon("fa6s.brain", color=C["BLUE"]).pixmap(
                QSize(18, 18)
            )
        )
        self._log_icon.setPixmap(
            qta.icon("fa6s.terminal", color=C["SUBTEXT"]).pixmap(
                QSize(16, 16)
            )
        )

        # Log terminal (always dark)
        self._log.setStyleSheet(
            "QTextEdit{background:#0D0D0D;color:#C0C0C0;"
            "border:1px solid #2C2C2C;"
            "border-radius:6px;"
            "font-family:'Cascadia Code','Consolas',monospace;"
            "font-size:12px;padding:8px;}"
        )

        # Badges
        badge_bg = C["CARD_BG"]
        old_ss = self._bot_badge.styleSheet()
        col_match = (
            old_ss.split("color:")[-1].split(";")[0].strip()
            if "color:" in old_ss
            else "#A0A0A0"
        )
        self._bot_badge.setStyleSheet(
            f"{_DOT} background:{badge_bg}; color:{col_match};"
        )

        conn_ss = self._conn_badge.styleSheet()
        conn_col = (
            conn_ss.split("color:")[-1].split(";")[0].strip()
            if "color:" in conn_ss
            else "#A0A0A0"
        )
        self._conn_badge.setStyleSheet(
            f"{_DOT} background:{badge_bg}; color:{conn_col};"
        )
