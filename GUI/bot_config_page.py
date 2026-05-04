"""
bot_config_page.py — Comprehensive Freqtrade Configuration Editor
==================================================================

Five-tab page that covers every runtime parameter the Freqtrade /
FreqAI bot needs, plus a *Generate & Save Config* action that writes
a valid ``config.json``.

Tabs
----
1. **Exchange & Pairlist** — exchange name, API key/secret, whitelist,
   blacklist.
2. **Stake Parameters** — stake currency, stake amount, max open trades,
   dry run.
3. **Strategy & Risk** — strategy file picker (``minimal_roi``,
   ``stoploss``, and ``trailing_stop`` are intentionally OMITTED from
   the generated JSON so the bot reads them from the strategy file).
4. **REST API** — listen address/port, JWT secret, enable flag, username,
   password.
5. **FreqAI (ML)** — enable flag, model selector, identifier, data-split
   & feature parameters.

Bottom action bar: **Generate & Save Config**.

Note: Bot lifecycle controls (API connection, Start/Stop, log) have been
moved to the **Model Training** page.
"""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path

import qtawesome as qta

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from GUI.themes import get_colors

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

class BotConfigPage(QWidget):
    """Tabbed configuration editor — pure config creation & saving."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._strategy_path: str = ""
        self._config_path: str = ""
        self._dark = True
        self._init_ui()

    # ──────────────────────────────────────────────────────────────────────
    # UI SKELETON
    # ──────────────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        root.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        # Title
        self._title = QLabel("Bot Configuration")
        self._title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        self._title.setStyleSheet("color:#FFFFFF;")
        lay.addWidget(self._title)

        # ── Tab widget ────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(_tab_widget_qss())
        lay.addWidget(self._tabs)

        self._tabs.addTab(self._build_exchange_tab(),  "Exchange && Pairlist")
        self._tabs.addTab(self._build_stake_tab(),     "Stake Parameters")
        self._tabs.addTab(self._build_strategy_tab(),  "Strategy && Risk")
        self._tabs.addTab(self._build_api_tab(),       "REST API")
        self._tabs.addTab(self._build_freqai_tab(),    "FreqAI (ML)")

        # ── Action bar ────────────────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(12)

        action_row.addStretch()

        self._save_btn = QPushButton("Generate && Save Config")
        self._save_btn.setStyleSheet(_btn_qss("#4CAF50"))
        self._save_btn.clicked.connect(self._on_save_config)
        action_row.addWidget(self._save_btn)

        lay.addLayout(action_row)
        lay.addStretch()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — Exchange & Pairlist
    # ══════════════════════════════════════════════════════════════════════

    def _build_exchange_tab(self) -> QWidget:
        page = QWidget()
        g = QGridLayout(page)
        g.setSpacing(12)
        g.setContentsMargins(16, 16, 16, 16)

        row = 0
        g.addWidget(_label("Exchange"), row, 0)
        self._exchange_combo = QComboBox()
        self._exchange_combo.setStyleSheet(_combo_qss())
        for ex in ["binance", "kraken", "kucoin", "okx", "bybit", "gate"]:
            self._exchange_combo.addItem(ex.title(), ex)
        g.addWidget(self._exchange_combo, row, 1)

        row += 1
        g.addWidget(_label("API Key"), row, 0)
        self._api_key = QLineEdit()
        self._api_key.setPlaceholderText("Your exchange API key")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setStyleSheet(_line_qss())
        g.addWidget(self._api_key, row, 1)

        row += 1
        g.addWidget(_label("API Secret"), row, 0)
        self._api_secret = QLineEdit()
        self._api_secret.setPlaceholderText("Your exchange API secret")
        self._api_secret.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_secret.setStyleSheet(_line_qss())
        g.addWidget(self._api_secret, row, 1)

        row += 1
        g.addWidget(_label("Pair Whitelist"), row, 0, Qt.AlignmentFlag.AlignTop)
        self._whitelist = QTextEdit()
        self._whitelist.setPlaceholderText("BTC/USDT\nETH/USDT\nSOL/USDT")
        self._whitelist.setMaximumHeight(100)
        self._whitelist.setStyleSheet(_textedit_qss())
        g.addWidget(self._whitelist, row, 1)

        row += 1
        g.addWidget(_label("Pair Blacklist"), row, 0, Qt.AlignmentFlag.AlignTop)
        self._blacklist = QTextEdit()
        self._blacklist.setPlaceholderText("BNB/.*")
        self._blacklist.setMaximumHeight(80)
        self._blacklist.setStyleSheet(_textedit_qss())
        g.addWidget(self._blacklist, row, 1)

        row += 1
        g.addWidget(_label("Trading Mode"), row, 0)
        self._trading_mode = QComboBox()
        self._trading_mode.setStyleSheet(_combo_qss())
        for m in ["spot", "futures"]:
            self._trading_mode.addItem(m.title(), m)
        g.addWidget(self._trading_mode, row, 1)

        g.setRowStretch(row + 1, 1)
        return page

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — Stake Parameters
    # ══════════════════════════════════════════════════════════════════════

    def _build_stake_tab(self) -> QWidget:
        page = QWidget()
        g = QGridLayout(page)
        g.setSpacing(12)
        g.setContentsMargins(16, 16, 16, 16)

        row = 0
        g.addWidget(_label("Stake Currency"), row, 0)
        self._stake_combo = QComboBox()
        self._stake_combo.setStyleSheet(_combo_qss())
        self._stake_combo.setEditable(True)
        for c in ["USDT", "USD", "BTC", "ETH", "BUSD", "USDC"]:
            self._stake_combo.addItem(c, c)
        g.addWidget(self._stake_combo, row, 1)

        row += 1
        g.addWidget(_label("Stake Amount"), row, 0)
        self._stake_amount = QLineEdit()
        self._stake_amount.setPlaceholderText('100  or  "unlimited"')
        self._stake_amount.setText("100")
        self._stake_amount.setStyleSheet(_line_qss())
        g.addWidget(self._stake_amount, row, 1)

        row += 1
        g.addWidget(_label("Max Open Trades"), row, 0)
        self._max_trades = QSpinBox()
        self._max_trades.setRange(-1, 100)
        self._max_trades.setValue(3)
        self._max_trades.setSpecialValueText("unlimited (-1)")
        self._max_trades.setStyleSheet(_spin_qss())
        g.addWidget(self._max_trades, row, 1)

        row += 1
        g.addWidget(_label("Dry Run"), row, 0)
        self._dry_run = QCheckBox("Enabled (paper trading)")
        self._dry_run.setChecked(True)
        self._dry_run.setStyleSheet(_check_qss())
        g.addWidget(self._dry_run, row, 1)

        row += 1
        g.addWidget(_label("Dry Run Wallet"), row, 0)
        self._dry_run_wallet = QSpinBox()
        self._dry_run_wallet.setRange(0, 10_000_000)
        self._dry_run_wallet.setValue(1000)
        self._dry_run_wallet.setPrefix("$ ")
        self._dry_run_wallet.setStyleSheet(_spin_qss())
        g.addWidget(self._dry_run_wallet, row, 1)

        g.setRowStretch(row + 1, 1)
        return page

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — Strategy & Risk
    # ══════════════════════════════════════════════════════════════════════

    def _build_strategy_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # Strategy file picker
        grp = QGroupBox("Strategy File")
        grp.setStyleSheet(_group_qss())
        g = QGridLayout(grp)
        g.setSpacing(12)

        g.addWidget(_label("Strategy .py"), 0, 0)
        self._strat_path_edit = QLineEdit()
        self._strat_path_edit.setReadOnly(True)
        self._strat_path_edit.setPlaceholderText("No strategy selected")
        self._strat_path_edit.setStyleSheet(_line_qss())
        g.addWidget(self._strat_path_edit, 0, 1)

        self._strat_browse_btn = QPushButton("Browse…")
        self._strat_browse_btn.setStyleSheet(_btn_qss("#2196F3"))
        self._strat_browse_btn.clicked.connect(self._on_browse_strategy)
        g.addWidget(self._strat_browse_btn, 0, 2)

        lay.addWidget(grp)

        # Info note
        info_frame = QFrame()
        info_frame.setStyleSheet(
            "background:#1A237E; border-radius:6px; border:none;"
        )
        info_row = QHBoxLayout(info_frame)
        info_row.setContentsMargins(10, 10, 10, 10)
        info_row.setSpacing(8)
        self._info_icon = QLabel()
        self._info_icon.setPixmap(
            qta.icon("fa6s.circle-info", color="#90CAF9").pixmap(QSize(16, 16))
        )
        self._info_icon.setStyleSheet("border: none; background: transparent;")
        self._info_icon.setFixedSize(16, 16)
        info_row.addWidget(self._info_icon, 0, Qt.AlignmentFlag.AlignTop)
        self._info_note = QLabel(
            "<i>minimal_roi</i>, <i>stoploss</i>, and <i>trailing_stop</i> "
            "are intentionally <b>not</b> included in the generated config JSON. "
            "Freqtrade will read them directly from your strategy file."
        )
        self._info_note.setWordWrap(True)
        self._info_note.setStyleSheet(
            "color:#90CAF9; font-size:12px; background:transparent; border:none;"
        )
        info_row.addWidget(self._info_note, 1)
        self._info_frame = info_frame
        lay.addWidget(info_frame)

        # Quick strategy selector (discovered strategies)
        grp2 = QGroupBox("Quick Select (discovered strategies)")
        grp2.setStyleSheet(_group_qss())
        g2 = QGridLayout(grp2)
        g2.setSpacing(12)

        g2.addWidget(_label("Strategy"), 0, 0)
        self._strat_combo = QComboBox()
        self._strat_combo.setStyleSheet(_combo_qss())
        self._discover_strategies()
        self._strat_combo.currentIndexChanged.connect(self._on_quick_strategy)
        g2.addWidget(self._strat_combo, 0, 1)

        lay.addWidget(grp2)
        lay.addStretch()
        return page

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — REST API
    # ══════════════════════════════════════════════════════════════════════

    def _build_api_tab(self) -> QWidget:
        page = QWidget()
        g = QGridLayout(page)
        g.setSpacing(12)
        g.setContentsMargins(16, 16, 16, 16)

        row = 0
        g.addWidget(_label("Enable API Server"), row, 0)
        self._api_enabled = QCheckBox("Enabled")
        self._api_enabled.setChecked(True)
        self._api_enabled.setStyleSheet(_check_qss())
        g.addWidget(self._api_enabled, row, 1)

        row += 1
        g.addWidget(_label("Listen IP"), row, 0)
        self._api_ip = QLineEdit("127.0.0.1")
        self._api_ip.setStyleSheet(_line_qss())
        g.addWidget(self._api_ip, row, 1)

        row += 1
        g.addWidget(_label("Listen Port"), row, 0)
        self._api_port = QSpinBox()
        self._api_port.setRange(1, 65535)
        self._api_port.setValue(8080)
        self._api_port.setStyleSheet(_spin_qss())
        g.addWidget(self._api_port, row, 1)

        row += 1
        g.addWidget(_label("Username"), row, 0)
        self._api_user = QLineEdit()
        self._api_user.setPlaceholderText("freqtrader")
        self._api_user.setStyleSheet(_line_qss())
        g.addWidget(self._api_user, row, 1)

        row += 1
        g.addWidget(_label("Password"), row, 0)
        self._api_pass = QLineEdit()
        self._api_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_pass.setPlaceholderText("••••••••")
        self._api_pass.setStyleSheet(_line_qss())
        g.addWidget(self._api_pass, row, 1)

        row += 1
        g.addWidget(_label("JWT Secret Key"), row, 0)
        jwt_row = QHBoxLayout()
        self._jwt_key = QLineEdit()
        self._jwt_key.setPlaceholderText("Auto-generate or paste")
        self._jwt_key.setStyleSheet(_line_qss())
        jwt_row.addWidget(self._jwt_key)

        gen_btn = QPushButton("Generate")
        gen_btn.setStyleSheet(_btn_qss("#FF9800"))
        gen_btn.setFixedWidth(100)
        gen_btn.clicked.connect(self._generate_jwt)
        jwt_row.addWidget(gen_btn)
        g.addLayout(jwt_row, row, 1)

        g.setRowStretch(row + 1, 1)
        return page

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — FreqAI (ML)
    # ══════════════════════════════════════════════════════════════════════

    def _build_freqai_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        # ── Basic settings ────────────────────────────────────────────────
        grp = QGroupBox("General")
        grp.setStyleSheet(_group_qss())
        g = QGridLayout(grp)
        g.setSpacing(12)

        row = 0
        g.addWidget(_label("Enable FreqAI"), row, 0)
        self._fai_enabled = QCheckBox("Enabled")
        self._fai_enabled.setChecked(True)
        self._fai_enabled.setStyleSheet(_check_qss())
        g.addWidget(self._fai_enabled, row, 1)

        row += 1
        g.addWidget(_label("ML Model"), row, 0)
        self._fai_model = QComboBox()
        self._fai_model.setStyleSheet(_combo_qss())
        for m in [
            "LightGBMRegressor", "XGBoostRegressor", "RandomForestRegressor",
            "LightGBMClassifier", "XGBoostClassifier", "CatboostRegressor",
        ]:
            self._fai_model.addItem(m, m)
        g.addWidget(self._fai_model, row, 1)

        row += 1
        g.addWidget(_label("Identifier"), row, 0)
        self._fai_identifier = QLineEdit("trend_model_v1")
        self._fai_identifier.setStyleSheet(_line_qss())
        g.addWidget(self._fai_identifier, row, 1)

        row += 1
        g.addWidget(_label("Train Period (days)"), row, 0)
        self._fai_train_days = QSpinBox()
        self._fai_train_days.setRange(1, 365)
        self._fai_train_days.setValue(30)
        self._fai_train_days.setStyleSheet(_spin_qss())
        g.addWidget(self._fai_train_days, row, 1)

        row += 1
        g.addWidget(_label("Purge Old Models"), row, 0)
        self._fai_purge = QCheckBox("Enabled")
        self._fai_purge.setChecked(True)
        self._fai_purge.setStyleSheet(_check_qss())
        g.addWidget(self._fai_purge, row, 1)

        lay.addWidget(grp)

        # ── Data split parameters ─────────────────────────────────────────
        grp2 = QGroupBox("Data Split Parameters")
        grp2.setStyleSheet(_group_qss())
        g2 = QGridLayout(grp2)
        g2.setSpacing(12)

        g2.addWidget(_label("Test Size"), 0, 0)
        self._fai_test_size = QDoubleSpinBox()
        self._fai_test_size.setRange(0.01, 0.50)
        self._fai_test_size.setSingleStep(0.05)
        self._fai_test_size.setValue(0.15)
        self._fai_test_size.setStyleSheet(_spin_qss())
        g2.addWidget(self._fai_test_size, 0, 1)

        g2.addWidget(_label("Shuffle"), 1, 0)
        self._fai_shuffle = QCheckBox("Enabled")
        self._fai_shuffle.setChecked(False)
        self._fai_shuffle.setStyleSheet(_check_qss())
        g2.addWidget(self._fai_shuffle, 1, 1)

        lay.addWidget(grp2)

        # ── Feature parameters ────────────────────────────────────────────
        grp3 = QGroupBox("Feature Parameters")
        grp3.setStyleSheet(_group_qss())
        g3 = QGridLayout(grp3)
        g3.setSpacing(12)

        r = 0
        g3.addWidget(_label("Include Timeframes"), r, 0, Qt.AlignmentFlag.AlignTop)
        self._fai_timeframes = QLineEdit("1h, 4h")
        self._fai_timeframes.setStyleSheet(_line_qss())
        self._fai_timeframes.setPlaceholderText("1h, 4h, 1d")
        g3.addWidget(self._fai_timeframes, r, 1)

        r += 1
        g3.addWidget(_label("Correlation Pairlist"), r, 0, Qt.AlignmentFlag.AlignTop)
        self._fai_corr_pairs = QLineEdit("BTC/USDT, ETH/USDT")
        self._fai_corr_pairs.setStyleSheet(_line_qss())
        g3.addWidget(self._fai_corr_pairs, r, 1)

        r += 1
        g3.addWidget(_label("Label Period (candles)"), r, 0)
        self._fai_label_period = QSpinBox()
        self._fai_label_period.setRange(1, 200)
        self._fai_label_period.setValue(24)
        self._fai_label_period.setStyleSheet(_spin_qss())
        g3.addWidget(self._fai_label_period, r, 1)

        r += 1
        g3.addWidget(_label("Shifted Candles"), r, 0)
        self._fai_shifted = QSpinBox()
        self._fai_shifted.setRange(0, 20)
        self._fai_shifted.setValue(2)
        self._fai_shifted.setStyleSheet(_spin_qss())
        g3.addWidget(self._fai_shifted, r, 1)

        r += 1
        g3.addWidget(_label("Indicator Periods"), r, 0)
        self._fai_indicator_periods = QLineEdit("10, 20, 50")
        self._fai_indicator_periods.setStyleSheet(_line_qss())
        self._fai_indicator_periods.setPlaceholderText("10, 20, 50")
        g3.addWidget(self._fai_indicator_periods, r, 1)

        r += 1
        g3.addWidget(_label("SVM Outlier Removal"), r, 0)
        self._fai_svm = QCheckBox("Enabled")
        self._fai_svm.setChecked(True)
        self._fai_svm.setStyleSheet(_check_qss())
        g3.addWidget(self._fai_svm, r, 1)

        lay.addWidget(grp3)
        lay.addStretch()
        return page

    # ══════════════════════════════════════════════════════════════════════
    # ACTION HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    # ── Strategy browse ───────────────────────────────────────────────────

    def _on_browse_strategy(self) -> None:
        start = str(Path(__file__).resolve().parent.parent / "strategies")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Strategy File", start, "Python Files (*.py)"
        )
        if path:
            self._strategy_path = path
            self._strat_path_edit.setText(path)

    def _on_quick_strategy(self) -> None:
        name = self._strat_combo.currentData()
        if not name:
            return
        root = Path(__file__).resolve().parent.parent
        for d in [root / "strategies", root / "user_data" / "strategies"]:
            candidate = d / f"{name}.py"
            if candidate.exists():
                self._strategy_path = str(candidate)
                self._strat_path_edit.setText(str(candidate))
                return

    # ── JWT generation ────────────────────────────────────────────────────

    def _generate_jwt(self) -> None:
        self._jwt_key.setText(secrets.token_hex(32))

    # ── Populate all tabs from a config dict ──────────────────────────────

    def _populate_from_config(self, cfg: dict) -> None:
        # Exchange tab
        ex = cfg.get("exchange", {})
        _set_combo(self._exchange_combo, ex.get("name", ""))
        self._api_key.setText(ex.get("key", ""))
        self._api_secret.setText(ex.get("secret", ""))
        wl = ex.get("pair_whitelist", [])
        self._whitelist.setPlainText("\n".join(wl) if isinstance(wl, list) else str(wl))
        bl = ex.get("pair_blacklist", [])
        self._blacklist.setPlainText("\n".join(bl) if isinstance(bl, list) else str(bl))
        _set_combo(self._trading_mode, cfg.get("trading_mode", "spot"))

        # Stake tab
        _set_combo(self._stake_combo, cfg.get("stake_currency", "USDT"))
        sa = cfg.get("stake_amount", 100)
        self._stake_amount.setText(str(sa))
        mot = cfg.get("max_open_trades", 3)
        self._max_trades.setValue(int(mot) if isinstance(mot, (int, float)) else 3)
        self._dry_run.setChecked(cfg.get("dry_run", True))
        self._dry_run_wallet.setValue(int(cfg.get("dry_run_wallet", 1000)))

        # Strategy tab
        strategy = cfg.get("strategy", "")
        if strategy:
            idx = self._strat_combo.findData(strategy)
            if idx >= 0:
                self._strat_combo.setCurrentIndex(idx)

        # API tab
        api = cfg.get("api_server", {})
        self._api_enabled.setChecked(api.get("enabled", True))
        raw_ip = api.get("listen_ip_address", "127.0.0.1")
        self._api_ip.setText("127.0.0.1" if raw_ip in ("0.0.0.0", "::", "") else raw_ip)
        self._api_port.setValue(api.get("listen_port", 8080))
        self._api_user.setText(api.get("username", ""))
        self._api_pass.setText(api.get("password", ""))
        self._jwt_key.setText(api.get("jwt_secret_key", ""))

        # FreqAI tab
        fai = cfg.get("freqai", {})
        self._fai_enabled.setChecked(fai.get("enabled", False))
        _set_combo(self._fai_model, fai.get("model_name", "LightGBMRegressor"))
        self._fai_identifier.setText(fai.get("identifier", "trend_model_v1"))
        self._fai_train_days.setValue(fai.get("train_period_days", 30))
        self._fai_purge.setChecked(fai.get("purge_old_models", True))

        dsp = fai.get("data_split_parameters", {})
        self._fai_test_size.setValue(dsp.get("test_size", 0.15))
        self._fai_shuffle.setChecked(dsp.get("shuffle", False))

        fp = fai.get("feature_parameters", {})
        tfs = fp.get("include_timeframes", ["1h", "4h"])
        self._fai_timeframes.setText(", ".join(tfs) if isinstance(tfs, list) else str(tfs))
        cp = fp.get("include_corr_pairlist", ["BTC/USDT", "ETH/USDT"])
        self._fai_corr_pairs.setText(", ".join(cp) if isinstance(cp, list) else str(cp))
        self._fai_label_period.setValue(fp.get("label_period_candles", 24))
        self._fai_shifted.setValue(fp.get("include_shifted_candles", 2))
        ip = fp.get("indicator_periods_candles", [10, 20, 50])
        self._fai_indicator_periods.setText(", ".join(str(x) for x in ip) if isinstance(ip, list) else str(ip))
        self._fai_svm.setChecked(fp.get("use_SVM_to_remove_outliers", True))

    # ── Generate & save config ────────────────────────────────────────────

    def _on_save_config(self) -> None:
        valid, err = self._validate_before_build()
        if not valid:
            QMessageBox.warning(self, "Validation Error", err)
            self._tabs.setCurrentIndex(0)
            return

        cfg = self._build_config_dict()

        default_dir = str(Path(__file__).resolve().parent.parent / "config")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Freqtrade Config", default_dir + "/config.json",
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)
            self._config_path = path
            QMessageBox.information(self, "Saved", f"Config saved → {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Save failed: {exc}")

    def get_or_generate_config_path(self, filename: str = "config_runtime.json") -> str:
        """
        Build a runtime config from the current form state and persist it
        under ``./config`` without opening a file dialog.
        """
        valid, err = self._validate_before_build()
        if not valid:
            raise ValueError(err)

        cfg = self._build_config_dict()
        config_dir = Path(__file__).resolve().parent.parent / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        out_path = config_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        self._config_path = str(out_path)
        return self._config_path

    def _validate_before_build(self) -> tuple[bool, str]:
        if not self._dry_run.isChecked():
            if not self._api_key.text().strip() or not self._api_secret.text().strip():
                return (
                    False,
                    "Dry Run is disabled but exchange API Key / Secret are empty.\n"
                    "Please provide valid credentials for live trading.",
                )
        return True, ""

    def _build_config_dict(self) -> dict:
        """Assemble a Freqtrade-compatible config dict from all tabs."""
        # Parse stake_amount
        sa_text = self._stake_amount.text().strip()
        if sa_text.lower() == "unlimited":
            stake_amount: int | float | str = "unlimited"
        else:
            try:
                stake_amount = float(sa_text)
                if stake_amount == int(stake_amount):
                    stake_amount = int(stake_amount)
            except ValueError:
                stake_amount = 100

        # Parse whitelist / blacklist
        whitelist = _parse_list(self._whitelist.toPlainText())
        blacklist = _parse_list(self._blacklist.toPlainText())

        # Strategy name (derived from filename)
        strategy_name = ""
        if self._strategy_path:
            strategy_name = Path(self._strategy_path).stem
        elif self._strat_combo.currentData():
            strategy_name = self._strat_combo.currentData()

        cfg: dict = {
            "max_open_trades": self._max_trades.value(),
            "stake_currency": self._stake_combo.currentText().strip(),
            "stake_amount": stake_amount,
            "tradable_balance_ratio": 0.99,
            "fiat_display_currency": "USD",
            "dry_run": self._dry_run.isChecked(),
            "dry_run_wallet": self._dry_run_wallet.value(),
            "cancel_open_orders_on_exit": False,
            "trading_mode": self._trading_mode.currentData() or "spot",
            "margin_mode": "",
            "unfilledtimeout": {
                "entry": 10, "exit": 10,
                "exit_timeout_count": 0, "unit": "minutes",
            },
            "entry_pricing": {
                "price_side": "same",
                "use_order_book": True,
                "order_book_top": 1,
                "price_last_balance": 0.0,
                "check_depth_of_market": {"enabled": False, "bids_to_ask_delta": 1},
            },
            "exit_pricing": {
                "price_side": "same",
                "use_order_book": True,
                "order_book_top": 1,
            },
            "exchange": {
                "name": self._exchange_combo.currentData() or "binance",
                "key": self._api_key.text().strip(),
                "secret": self._api_secret.text().strip(),
                "ccxt_config": {},
                "ccxt_async_config": {},
                "pair_whitelist": whitelist,
                "pair_blacklist": blacklist,
            },
            "pairlists": [{"method": "StaticPairList"}],
            "telegram": {"enabled": False, "token": "", "chat_id": ""},
            "api_server": {
                "enabled": self._api_enabled.isChecked(),
                "listen_ip_address": self._api_ip.text().strip() or "127.0.0.1",
                "listen_port": self._api_port.value(),
                "verbosity": "error",
                "enable_openapi": False,
                "jwt_secret_key": self._jwt_key.text().strip(),
                "ws_token": secrets.token_urlsafe(24),
                "CORS_origins": [],
                "username": self._api_user.text().strip(),
                "password": self._api_pass.text().strip(),
            },
            "bot_name": "freqtrade",
            "initial_state": "running",
            "force_entry_enable": False,
            "internals": {"process_throttle_secs": 5},
        }

        # Strategy (ROI / stoploss / trailing intentionally OMITTED)
        if strategy_name:
            cfg["strategy"] = strategy_name

        # FreqAI block
        if self._fai_enabled.isChecked():
            cfg["freqai"] = {
                "enabled": True,
                "purge_old_models": self._fai_purge.isChecked(),
                "train_period_days": self._fai_train_days.value(),
                "identifier": self._fai_identifier.text().strip() or "trend_model_v1",
                "feature_parameters": {
                    "include_timeframes": _parse_list(self._fai_timeframes.text()),
                    "include_corr_pairlist": _parse_list(self._fai_corr_pairs.text()),
                    "label_period_candles": self._fai_label_period.value(),
                    "include_shifted_candles": self._fai_shifted.value(),
                    "indicator_periods_candles": _parse_int_list(
                        self._fai_indicator_periods.text()
                    ),
                    "use_SVM_to_remove_outliers": self._fai_svm.isChecked(),
                },
                "data_split_parameters": {
                    "test_size": round(self._fai_test_size.value(), 2),
                    "shuffle": self._fai_shuffle.isChecked(),
                },
            }
            cfg["freqai"]["model_name"] = self._fai_model.currentData()

        return cfg

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def apply_theme(self, dark: bool) -> None:
        """Re-apply all stylesheets for light / dark mode."""
        self._dark = dark
        C = get_colors(dark)

        # Title
        self._title.setStyleSheet(f"color:{C['TEXT']};")

        # Tabs
        self._tabs.setStyleSheet(_tab_widget_qss(dark))

        # All input widgets
        combo_qss = _combo_qss(dark)
        line_qss = _line_qss(dark)
        spin_qss = _spin_qss(dark)
        check_qss = _check_qss(dark)
        textedit_qss = _textedit_qss(dark)
        group_qss = _group_qss(dark)

        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet(combo_qss)
        for le in self.findChildren(QLineEdit):
            # Skip internal QLineEdits inside QComboBox / QSpinBox
            parent = le.parent()
            if isinstance(parent, (QComboBox, QSpinBox, QDoubleSpinBox)):
                continue
            le.setStyleSheet(line_qss)
        for sb in self.findChildren(QSpinBox):
            sb.setStyleSheet(spin_qss)
        for dsb in self.findChildren(QDoubleSpinBox):
            dsb.setStyleSheet(spin_qss)
        for cb in self.findChildren(QCheckBox):
            cb.setStyleSheet(check_qss)
        for grp in self.findChildren(QGroupBox):
            grp.setStyleSheet(group_qss)

        # Whitelist / blacklist text edits
        self._whitelist.setStyleSheet(textedit_qss)
        self._blacklist.setStyleSheet(textedit_qss)

        # Info note
        note_bg = "#1A237E" if dark else "#E3F2FD"
        note_fg = "#90CAF9" if dark else "#1565C0"
        self._info_frame.setStyleSheet(
            f"background:{note_bg}; border-radius:6px; border:none;"
        )
        self._info_note.setStyleSheet(
            f"color:{note_fg}; font-size:12px; background:transparent; border:none;"
        )
        self._info_icon.setPixmap(
            qta.icon("fa6s.circle-info", color=note_fg).pixmap(QSize(16, 16))
        )

        # Field labels (all QLabels except special ones)
        label_color = C['LABEL']
        special = {id(self._title), id(self._info_note), id(self._info_icon)}
        for lbl in self.findChildren(QLabel):
            if id(lbl) not in special:
                lbl.setStyleSheet(f"color: {label_color}; font-size: 13px; border: none;")

    def _discover_strategies(self) -> None:
        root = Path(__file__).resolve().parent.parent
        seen: set[str] = set()
        for d in [root / "strategies", root / "user_data" / "strategies"]:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.py")):
                if p.name.startswith("_"):
                    continue
                name = p.stem
                if name not in seen and name != "__init__":
                    seen.add(name)
                    self._strat_combo.addItem(name, name)
        idx = self._strat_combo.findData("TrendMLStrategy")
        if idx >= 0:
            self._strat_combo.setCurrentIndex(idx)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _set_combo(combo: QComboBox, value: str) -> None:
    """Select a combo item by its data value (case-insensitive)."""
    for i in range(combo.count()):
        if str(combo.itemData(i)).lower() == value.lower():
            combo.setCurrentIndex(i)
            return
    # For editable combos, just set the text
    if combo.isEditable():
        combo.setEditText(value)


def _parse_list(text: str) -> list[str]:
    """Split comma-or-newline-separated text into a clean list of strings."""
    items: list[str] = []
    for chunk in text.replace("\n", ",").split(","):
        s = chunk.strip()
        if s:
            items.append(s)
    return items


def _parse_int_list(text: str) -> list[int]:
    """Parse '10, 20, 50' into [10, 20, 50]."""
    result: list[int] = []
    for chunk in text.replace("\n", ",").split(","):
        s = chunk.strip()
        if s:
            try:
                result.append(int(s))
            except ValueError:
                pass
    return result


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #A0A0A0; font-size: 13px; border: none;")
    return lbl


# ── QSS helpers ──────────────────────────────────────────────────────────────

def _tab_widget_qss(dark: bool = True) -> str:
    C = get_colors(dark)
    return f"""
        QTabWidget::pane {{
            border: 1px solid {C['CARD_BORDER']}; border-radius: 8px;
            background: {C['CARD_BG']}; top: -1px;
        }}
        QTabBar::tab {{
            background: {C['INPUT_BG']}; color: {C['SUBTEXT']};
            border: 1px solid {C['CARD_BORDER']}; border-bottom: none;
            padding: 10px 18px; font-size: 13px; font-weight: 600;
            border-top-left-radius: 6px; border-top-right-radius: 6px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background: {C['CARD_BG']}; color: {C['TEXT']};
            border-bottom: 2px solid {C['BLUE']};
        }}
        QTabBar::tab:hover:!selected {{
            background: {C['TABLE_ALT']}; color: {C['TEXT']};
        }}
    """


def _group_qss(dark: bool = True) -> str:
    C = get_colors(dark)
    return f"""
        QGroupBox {{
            color: {C['GROUP_TITLE']}; font-size: 14px; font-weight: bold;
            border: 1px solid {C['CARD_BORDER']}; border-radius: 8px;
            margin-top: 10px; padding-top: 16px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: 12px; padding: 0 4px;
        }}
    """


def _combo_qss(dark: bool = True) -> str:
    C = get_colors(dark)
    return f"""
        QComboBox {{
            background-color: {C['INPUT_BG']}; color: {C['TEXT']};
            border: 1px solid {C['INPUT_BORDER']}; border-radius: 6px;
            padding: 7px 10px; font-size: 13px; min-width: 220px; min-height: 20px;
        }}
        QComboBox::drop-down {{ border: none; }}
        QComboBox QAbstractItemView {{
            background-color: {C['INPUT_BG']}; color: {C['TEXT']};
            selection-background-color: {C['BLUE']};
        }}
    """


def _spin_qss(dark: bool = True) -> str:
    C = get_colors(dark)
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {C['INPUT_BG']}; color: {C['TEXT']};
            border: 1px solid {C['INPUT_BORDER']}; border-radius: 6px;
            padding: 7px 10px; font-size: 13px; min-width: 220px; min-height: 20px;
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: {C['CARD_BORDER']}; border: none; width: 18px;
        }}
    """


def _line_qss(dark: bool = True) -> str:
    C = get_colors(dark)
    return f"""
        QLineEdit {{
            background-color: {C['INPUT_BG']}; color: {C['TEXT']};
            border: 1px solid {C['INPUT_BORDER']}; border-radius: 6px;
            padding: 7px 10px; font-size: 13px; min-width: 220px;
        }}
        QLineEdit:focus {{ border: 1px solid {C['BLUE']}; }}
    """


def _textedit_qss(dark: bool = True) -> str:
    C = get_colors(dark)
    return f"""
        QTextEdit {{
            background-color: {C['INPUT_BG']}; color: {C['TEXT']};
            border: 1px solid {C['INPUT_BORDER']}; border-radius: 6px;
            padding: 7px 10px; font-size: 13px; min-width: 220px;
        }}
        QTextEdit:focus {{ border: 1px solid {C['BLUE']}; }}
    """


def _check_qss(dark: bool = True) -> str:
    C = get_colors(dark)
    return (
        f"QCheckBox{{color:{C['LABEL']};font-size:13px;border:none;}}"
        f"QCheckBox::indicator{{width:18px;height:18px;}}"
    )


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
