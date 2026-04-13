"""
settings_panel.py — Application Settings Page
===============================================

Interface and application-level preferences only.
Bot configuration has been moved to ``BotConfigPage``.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class SettingsPanel(QWidget):
    """Application settings — interface preferences, theme, language, etc."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dark = False
        self._init_ui()

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Settings")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #1A1A2E;")
        self._title_lbl = title
        layout.addWidget(title)

        layout.addWidget(self._build_appearance_group())
        layout.addStretch()

    # ── Appearance ────────────────────────────────────────────────────────

    def _build_appearance_group(self) -> QGroupBox:
        grp = QGroupBox("Appearance")
        grp.setStyleSheet(self._group_style())
        grid = QGridLayout(grp)
        grid.setSpacing(12)

        grid.addWidget(self._label("Theme"), 0, 0)
        self._theme_combo = QComboBox()
        self._theme_combo.setStyleSheet(self._combo_style())
        self._theme_combo.addItem("Dark (default)", "dark")
        self._theme_combo.addItem("Light", "light")
        grid.addWidget(self._theme_combo, 0, 1)

        grid.addWidget(self._label("Language"), 1, 0)
        self._lang_combo = QComboBox()
        self._lang_combo.setStyleSheet(self._combo_style())
        self._lang_combo.addItem("English", "en")
        self._lang_combo.addItem("Türkçe", "tr")
        grid.addWidget(self._lang_combo, 1, 1)

        return grp

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #A0A0A0; font-size: 14px; border: none;")
        return lbl

    def _group_style(self) -> str:
        from GUI.themes import get_colors
        C = get_colors(self._dark)
        return f"""
            QGroupBox {{
                color: {C['GROUP_TITLE']}; font-size: 15px; font-weight: bold;
                border: 1px solid {C['CARD_BORDER']}; border-radius: 10px;
                margin-top: 10px; padding-top: 18px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 15px; padding: 0 5px;
            }}
        """

    def _combo_style(self) -> str:
        from GUI.themes import get_colors
        C = get_colors(self._dark)
        return f"""
            QComboBox {{
                background-color: {C['INPUT_BG']}; color: {C['TEXT']};
                border: 1px solid {C['INPUT_BORDER']}; border-radius: 6px;
                padding: 8px 12px; font-size: 13px; min-width: 250px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {C['CARD_BG']}; color: {C['TEXT']};
                selection-background-color: #3D5AFE;
            }}
        """

    # ══════════════════════════════════════════════════════════════════════
    # THEMING
    # ══════════════════════════════════════════════════════════════════════

    def apply_theme(self, dark: bool) -> None:
        from GUI.themes import get_colors
        from PyQt6.QtWidgets import QGroupBox
        self._dark = dark
        C = get_colors(dark)
        self.setStyleSheet(f"background-color: {C['PAGE_BG']};")
        self._title_lbl.setStyleSheet(f"color: {C['TEXT']};")
        for grp in self.findChildren(QGroupBox):
            grp.setStyleSheet(self._group_style())
        self._theme_combo.setStyleSheet(self._combo_style())
        self._lang_combo.setStyleSheet(self._combo_style())
