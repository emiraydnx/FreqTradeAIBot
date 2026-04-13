"""
metric_cards.py — Reusable Dashboard Metric Card Widgets
=========================================================

Provides ``MetricCard`` — a styled card showing a title, value, and
subtitle.  Designed to be updated live from ``DashboardSnapshot`` data.
"""

from __future__ import annotations

import qtawesome as qta

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from GUI.themes import get_colors


# ── Colours ──────────────────────────────────────────────────────────────────
COLOR_POSITIVE = "#00C087"   # green
COLOR_NEGATIVE = "#FF4C4C"   # red
COLOR_NEUTRAL  = "#888888"   # grey


class MetricCard(QFrame):
    """
    A single metric card displaying title / value / subtitle.

    Layout::

        ┌───────────────────────────────┐
        │ Title (grey)      [icon] (rt) │
        │ Value (bold, large)           │
        │ Subtitle (coloured, small)    │
        └───────────────────────────────┘
    """

    def __init__(
        self,
        title: str = "",
        value: str = "",
        subtitle: str = "",
        icon: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._dark = False

        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
                border: 1px solid #E8E8E8;
            }
        """)
        self.setMinimumHeight(100)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(4)

        # Title row with optional icon
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            "color: #888888; font-size: 12px; font-weight: 500; border: none;"
        )
        title_row.addWidget(self._title_label)
        title_row.addStretch()
        self._icon_name = icon
        self._icon_lbl: QLabel | None = None
        if icon:
            self._icon_lbl = QLabel()
            self._icon_lbl.setPixmap(
                qta.icon(icon, color="#AAAAAA").pixmap(QSize(18, 18))
            )
            self._icon_lbl.setStyleSheet("border: none;")
            title_row.addWidget(self._icon_lbl)
        root.addLayout(title_row)

        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(
            "color: #1A1A2E; font-size: 24px; font-weight: 700; border: none;"
        )
        root.addWidget(self._value_label)

        self._subtitle_label = QLabel(subtitle)
        self._subtitle_label.setStyleSheet(
            f"color: {COLOR_NEUTRAL}; font-size: 11px; border: none;"
        )
        root.addWidget(self._subtitle_label)

    # ── Public API ────────────────────────────────────────────────────────

    def update_value(
        self,
        value: str,
        subtitle: str = "",
        positive: bool | None = None,
    ) -> None:
        """
        Update the displayed value and subtitle.

        Args:
            value:    Main display value (e.g. "$1,250.00").
            subtitle: Secondary text (e.g. "+25.0%").
            positive: ``True`` → green, ``False`` → red, ``None`` → grey.
        """
        self._value_label.setText(value)
        if subtitle:
            self._subtitle_label.setText(subtitle)

        if positive is True:
            color = COLOR_POSITIVE
        elif positive is False:
            color = COLOR_NEGATIVE
        else:
            color = COLOR_NEUTRAL
        self._subtitle_label.setStyleSheet(
            f"color: {color}; font-size: 11px; border: none;"
        )

    def set_title(self, title: str) -> None:
        self._title_label.setText(title)

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
            f"color: {C['SUBTEXT']}; font-size: 12px; font-weight: 500; border: none;"
        )
        self._value_label.setStyleSheet(
            f"color: {C['TEXT']}; font-size: 24px; font-weight: 700; border: none;"
        )
        # Preserve existing subtitle colour (positive/negative/neutral)
        cur_style = self._subtitle_label.styleSheet()
        if "#00C087" in cur_style:
            sub_color = COLOR_POSITIVE
        elif "#FF4C4C" in cur_style:
            sub_color = COLOR_NEGATIVE
        else:
            sub_color = COLOR_NEUTRAL
        self._subtitle_label.setStyleSheet(
            f"color: {sub_color}; font-size: 11px; border: none;"
        )
        # Update icon colour
        if self._icon_lbl and self._icon_name:
            self._icon_lbl.setPixmap(
                qta.icon(self._icon_name, color=C['SUBTEXT']).pixmap(QSize(18, 18))
            )
