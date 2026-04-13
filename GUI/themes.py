"""
themes.py — Centralized Color Tokens for Light / Dark Mode
===========================================================

Usage::

    from GUI.themes import get_colors

    C = get_colors(dark=True)
    self.setStyleSheet(f"background-color: {C['PAGE_BG']};")

Every GUI widget that supports theming calls ``apply_theme(dark: bool)``
which re-applies its own stylesheets using tokens from this module.
The MainWindow calls ``apply_theme`` on all pages when the user toggles
dark mode.
"""

from __future__ import annotations

# ── Accent colours (same in both modes) ─────────────────────────────────────
GREEN  = "#00C087"
RED    = "#FF4C4C"
ORANGE = "#FF9A00"
BLUE   = "#3D5AFE"
BLUE_HOVER = "#536DFE"

# ── Light theme ──────────────────────────────────────────────────────────────
LIGHT: dict[str, str] = {
    "PAGE_BG":       "#F5F6FA",
    "CARD_BG":       "#FFFFFF",
    "CARD_BORDER":   "#E8E8E8",
    "TEXT":          "#1A1A2E",
    "SUBTEXT":       "#888888",
    "HEADER_BG":     "#FFFFFF",
    "HEADER_BORDER": "#E8E8E8",
    "SIDEBAR_BG":    "#FFFFFF",
    "SIDEBAR_BORDER":"#E8E8E8",
    "INPUT_BG":      "#F5F6FA",
    "INPUT_BORDER":  "#E0E0E0",
    "TABLE_BG":      "#FFFFFF",
    "TABLE_ALT":     "#F8F9FB",
    "TABLE_HEADER":  "#F0F1F5",
    "DIVIDER":       "#E8E8E8",
    "CHART_BG":      "#FFFFFF",
    "CHART_GRID":    "#E8E8E8",
    "LABEL":         "#9E9E9E",
    "TOGGLE_TEXT":   "#888888",
    "TOGGLE_HOVER":  "#F5F5F5",
    "NAV_UNSEL":     "#555555",
    "NAV_HOVER":     "#F0F0F0",
    "TIPS_BG":       "#F0F4FF",
    "SCROLLBAR":     "#CCCCCC",
    "SCROLLBAR_BG":  "#F0F0F0",
    "GROUP_TITLE":   "#1A1A2E",
}

# ── Dark theme ───────────────────────────────────────────────────────────────
DARK: dict[str, str] = {
    "PAGE_BG":       "#121212",
    "CARD_BG":       "#1E1E1E",
    "CARD_BORDER":   "#2C2C2C",
    "TEXT":          "#E0E0E0",
    "SUBTEXT":       "#888888",
    "HEADER_BG":     "#1A1A2E",
    "HEADER_BORDER": "#2C2C2C",
    "SIDEBAR_BG":    "#1A1A2E",
    "SIDEBAR_BORDER":"#2C2C2C",
    "INPUT_BG":      "#252525",
    "INPUT_BORDER":  "#3C3C3C",
    "TABLE_BG":      "#1E1E1E",
    "TABLE_ALT":     "#252525",
    "TABLE_HEADER":  "#252525",
    "DIVIDER":       "#2C2C2C",
    "CHART_BG":      "#1A1A1A",
    "CHART_GRID":    "#2C2C2C",
    "LABEL":         "#9E9E9E",
    "TOGGLE_TEXT":   "#A0A0A0",
    "TOGGLE_HOVER":  "#2A2A2A",
    "NAV_UNSEL":     "#A0A0A0",
    "NAV_HOVER":     "#2A2A2A",
    "TIPS_BG":       "#1A2040",
    "SCROLLBAR":     "#444444",
    "SCROLLBAR_BG":  "#1E1E1E",
    "GROUP_TITLE":   "#E0E0E0",
}


def get_colors(dark: bool = False) -> dict[str, str]:
    """Return the active color token dict.

    Also injects the accent colours (GREEN, RED, etc.) so callers can
    reference them via the same dict for convenience.
    """
    base = dict(DARK if dark else LIGHT)
    base.update({
        "GREEN":      GREEN,
        "RED":        RED,
        "ORANGE":     ORANGE,
        "BLUE":       BLUE,
        "BLUE_HOVER": BLUE_HOVER,
    })
    return base
