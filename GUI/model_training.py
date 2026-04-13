"""
model_training.py — Model Training Page
=========================================

Matches the FreqTrade Pro mockup:
  • Top toolbar: "Prepare Dataset" + "Start Training" buttons
  • Left: Training Progress card (model name, epoch progress bar, metrics, Stop/Pause)
  • Right-top: Model History card (models from user_data/models/)
  • Right-bottom: Training Tips card
"""

from __future__ import annotations

import json
from pathlib import Path

import qtawesome as qta

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
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
_ORANGE  = "#FF9A00"
_BLUE    = "#3D5AFE"
_TEXT    = "#1A1A2E"
_SUB     = "#888888"

_MODELS_DIR = Path("user_data/models")


def _card_style(min_height: int = 0) -> str:
    extra = f"min-height: {min_height}px;" if min_height else ""
    return f"""
        QFrame {{
            background-color: {_BG};
            border-radius: 12px;
            {extra}
        }}
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


class ModelTrainingPage(QWidget):
    """Model Training page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG_PAGE};")
        self._training_active = False
        self._epoch = 0
        self._max_epoch = 100

        self._init_ui()
        self._load_model_history()

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(20)

        # ── Page title + toolbar ──────────────────────────────────────────
        title_row = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel("Bot Model Training")
        title.setStyleSheet(f"color: {_TEXT}; font-size: 22px; font-weight: 700;")
        subtitle = QLabel("Train machine learning models for predictive trading.")
        subtitle.setStyleSheet(f"color: {_SUB}; font-size: 13px;")
        col.addWidget(title)
        col.addWidget(subtitle)
        title_row.addLayout(col)
        title_row.addStretch()

        self._prepare_btn = QPushButton("  Prepare Dataset")
        self._prepare_btn.setIcon(qta.icon("fa6s.database", color=_TEXT))
        self._prepare_btn.setIconSize(QSize(14, 14))
        self._prepare_btn.setFixedHeight(38)
        self._prepare_btn.setMinimumWidth(148)
        self._prepare_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._prepare_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 8px;
                font-size: 13px; font-weight: 600; padding: 0 14px;
            }}
            QPushButton:hover {{ background: #F0F0F0; }}
        """)
        title_row.addWidget(self._prepare_btn)
        title_row.addSpacing(8)

        self._train_btn = QPushButton("  Start Training")
        self._train_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
        self._train_btn.setIconSize(QSize(14, 14))
        self._train_btn.setFixedHeight(38)
        self._train_btn.setMinimumWidth(140)
        self._train_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._train_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_TEXT}; color: #FFF; border: none;
                border-radius: 8px; font-size: 13px; font-weight: 700;
                padding: 0 18px;
            }}
            QPushButton:hover {{ background: #2C3E6B; }}
        """)
        self._train_btn.clicked.connect(self._on_train_clicked)
        title_row.addWidget(self._train_btn)
        layout.addLayout(title_row)

        # ── Main row ──────────────────────────────────────────────────────
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
        layout.addStretch()

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
        lbl_icon.setPixmap(qta.icon("fa6s.gear", color=_SUB).pixmap(QSize(18, 18)))
        lbl_icon.setStyleSheet("border: none;")
        self._progress_icon = lbl_icon
        lbl_title = QLabel("Training Progress")
        lbl_title.setStyleSheet(f"color: {_TEXT}; font-size: 15px; font-weight: 700;")
        hdr.addWidget(lbl_icon)
        hdr.addWidget(lbl_title)
        hdr.addStretch()
        layout.addLayout(hdr)

        self._lbl_model_name = QLabel("Model: LSTM-v2-TrendPredictor")
        self._lbl_model_name.setStyleSheet(f"color: {_SUB}; font-size: 12px;")
        layout.addWidget(self._lbl_model_name)

        # Epoch progress
        epoch_row = QHBoxLayout()
        self._lbl_epoch = QLabel("Epoch 0/100")
        self._lbl_epoch.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 600;")
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

        # Metrics grid (2x2)
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(32)

        left_metrics = QVBoxLayout()
        self._row_loss    = _MetricRow("Current Loss",    "—",        _TEXT)
        self._row_time    = _MetricRow("Time Remaining",  "—:—:—",   _TEXT)
        left_metrics.addWidget(self._row_loss)
        left_metrics.addWidget(self._row_time)

        right_metrics = QVBoxLayout()
        self._row_acc     = _MetricRow("Accuracy",       "—",        _GREEN)
        self._row_gpu     = _MetricRow("GPU Usage",      "—",        _ORANGE)
        right_metrics.addWidget(self._row_acc)
        right_metrics.addWidget(self._row_gpu)

        metrics_layout.addLayout(left_metrics)
        metrics_layout.addLayout(right_metrics)
        layout.addLayout(metrics_layout)

        layout.addStretch()

        # Stop / Pause buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._stop_btn = QPushButton("  Stop")
        self._stop_btn.setIcon(qta.icon("fa6s.circle-stop", color=_TEXT))
        self._stop_btn.setIconSize(QSize(14, 14))
        self._stop_btn.setFixedSize(90, 34)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_BG}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 6px;
                font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #FFE5E5; color: {_RED}; }}
        """)
        self._stop_btn.clicked.connect(self._on_stop)

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

        btn_row.addWidget(self._stop_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(self._pause_btn)
        layout.addLayout(btn_row)

        return card

    def _build_history_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(_card_style())

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        lbl = QLabel("Model History")
        lbl.setStyleSheet(f"color: {_TEXT}; font-size: 15px; font-weight: 700;")
        layout.addWidget(lbl)
        layout.addSpacing(4)

        self._history_layout = QVBoxLayout()
        self._history_layout.setSpacing(0)
        layout.addLayout(self._history_layout)
        layout.addStretch()

        return card

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
        icon.setPixmap(qta.icon("fa6s.lightbulb", color=_BLUE).pixmap(QSize(16, 16)))
        icon.setStyleSheet("border: none;")
        self._tips_icon = icon
        lbl = QLabel("Training Tips")
        lbl.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 700;")
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

    # ══════════════════════════════════════════════════════════════════════
    # MODEL HISTORY FROM DISK
    # ══════════════════════════════════════════════════════════════════════

    def _load_model_history(self) -> None:
        """Scan user_data/models/ and populate the history panel."""
        rows = []
        if _MODELS_DIR.exists():
            for model_dir in sorted(_MODELS_DIR.iterdir()):
                if not model_dir.is_dir():
                    continue
                meta_file = model_dir / "global_metadata.json"
                run_file  = model_dir / "run_params.json"
                try:
                    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
                    run  = json.loads(run_file.read_text())  if run_file.exists()  else {}
                    accuracy = meta.get("accuracy", run.get("model_validation_score", None))
                    acc_str = f"{accuracy * 100:.1f}%" if accuracy and accuracy <= 1 else (
                        f"{accuracy:.1f}%" if accuracy else "—"
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
                ("LSTM-v1",          "2024-03-15", "84.2%", "Deployed"),
                ("XGBoost-Trend",    "2024-03-10", "79.5%", "Archived"),
                ("RandomForest-v4",  "2024-03-05", "81.1%", "Archived"),
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
    # TRAINING SIMULATION (DEMO)
    # ══════════════════════════════════════════════════════════════════════

    def _on_train_clicked(self) -> None:
        if self._training_active:
            return
        self._training_active = True
        self._epoch = 0
        self._train_btn.setEnabled(False)
        self._train_btn.setText("Training…")
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick_training)
        self._tick_timer.start(150)

    def _tick_training(self) -> None:
        """Simulate epoch progress for demo."""
        self._epoch += 1
        pct = int(self._epoch / self._max_epoch * 100)
        remaining_s = int((self._max_epoch - self._epoch) * 0.15)
        m, s = divmod(remaining_s, 60)
        h, m = divmod(m, 60)

        self._progress_bar.setValue(pct)
        self._lbl_epoch.setText(f"Epoch {self._epoch}/{self._max_epoch}")
        self._lbl_epoch_pct.setText(f"{pct}%")
        self._row_loss.set_value(f"{0.08 - self._epoch * 0.0005:.4f}" if self._epoch < 100 else "0.0245")
        self._row_acc.set_value(f"{60 + self._epoch * 0.3:.1f}%", _GREEN)
        self._row_gpu.set_value(f"{70 + (self._epoch % 15):.0f}%", _ORANGE)
        self._row_time.set_value(f"{h:02d}:{m:02d}:{s:02d}")

        if self._epoch >= self._max_epoch:
            self._tick_timer.stop()
            self._training_active = False
            self._train_btn.setEnabled(True)
            self._train_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
            self._train_btn.setText("  Start Training")

    def _on_stop(self) -> None:
        if self._training_active:
            self._tick_timer.stop()
            self._training_active = False
            self._train_btn.setEnabled(True)
            self._train_btn.setIcon(qta.icon("fa6s.play", color="#FFFFFF"))
            self._train_btn.setText("  Start Training")

    # ══════════════════════════════════════════════════════════════════════
    # THEMING
    # ══════════════════════════════════════════════════════════════════════

    def apply_theme(self, dark: bool) -> None:
        from GUI.themes import get_colors
        from PyQt6.QtWidgets import QFrame, QLabel
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
        # Tips card has its own blue-tint bg
        tips_bg   = "#1A2040" if dark else "#F8F9FF"
        tips_border = "#2A3060" if dark else "#DDE3FF"
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
        for lbl in self.findChildren(QLabel):
            ss = lbl.styleSheet()
            if "font-size: 22px" in ss or "font-size: 15px" in ss:
                lbl.setStyleSheet(
                    ss.replace("#1A1A2E", C['TEXT']).replace("#E0E0E0", C['TEXT'])
                )
            elif "font-size: 13px; font-weight: 600" in ss and "#1A1A2E" in ss:
                lbl.setStyleSheet(ss.replace("#1A1A2E", C['TEXT']))
        # Button icons
        self._prepare_btn.setIcon(qta.icon("fa6s.database", color=C['TEXT']))
        self._stop_btn.setIcon(qta.icon("fa6s.circle-stop", color=C['TEXT']))
        self._pause_btn.setIcon(qta.icon("fa6s.pause", color=C['TEXT']))
        self._progress_icon.setPixmap(
            qta.icon("fa6s.microchip", color=C['SUBTEXT']).pixmap(QSize(18, 18))
        )
        self._tips_icon.setPixmap(
            qta.icon("fa6s.lightbulb", color=C['BLUE']).pixmap(QSize(16, 16))
        )
