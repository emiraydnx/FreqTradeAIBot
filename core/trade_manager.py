"""
trade_manager.py — High-Level Business Logic Layer
====================================================

Orchestrates the REST client, WebSocket client, and Bot Process Manager
into a single facade that the GUI can consume.

Responsibilities:
    • Owns the ``FreqtradeClient`` (REST), ``FreqtradeWSClient`` (WS),
      and ``BotProcessManager`` (process lifecycle) instances.
    • Aggregates data for the dashboard (balance, PnL, trades, …).
    • Merges real-time WS pushes into a local state cache so the GUI
      always reads from a single, consistent snapshot.
    • Provides all trade actions (force entry/exit, start/stop) with
      unified async callbacks.

Architecture (GUI-safe):
────────────────────────
Nothing in this module touches a QWidget directly.  Every callback fires
from a worker thread; the GUI layer wires them to ``pyqtSignal.emit()``
so slots execute on the main thread.

Usage from GUI::

    from core.trade_manager import TradeManager

    mgr = TradeManager()
    mgr.configure_from_config("config/config_binance.json")

    # Wire signals
    mgr.on_dashboard_update   = self._dashboard_signal.emit
    mgr.on_trade_event        = self._trade_event_signal.emit
    mgr.on_bot_state_change   = self._bot_state_signal.emit
    mgr.on_connection_change  = self._connection_signal.emit
    mgr.on_log                = self._log_signal.emit

    # Connect + start polling
    mgr.connect()
    mgr.start_polling(interval_sec=5.0)

    # Force a trade
    mgr.force_entry_async("BTC/USDT", side="long")

    # Shutdown
    mgr.stop_polling()
    mgr.disconnect()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

from API.rest_client import (
    FreqtradeClient,
    ConnectionState,
    ApiResult,
    ServerInfo,
)
from API.ws_client import (
    FreqtradeWSClient,
    WSState,
    WSMessage,
    MessageType,
)
from core.exchange_manager import (
    BotProcessManager,
    BotState,
    LaunchMode,
    BotLogEvent,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Connection Status (unified) ──────────────────────────────────────────────

class OverallStatus(Enum):
    """High-level status shown in the GUI header / status bar."""
    DISCONNECTED = auto()   # REST not connected
    CONNECTING   = auto()   # login in progress
    CONNECTED    = auto()   # REST connected, WS may be connecting
    LIVE         = auto()   # REST + WS both connected
    ERROR        = auto()   # authentication or network failure


# ── Dashboard Snapshot ───────────────────────────────────────────────────────

@dataclass
class DashboardSnapshot:
    """
    Immutable data object that the GUI reads to paint the dashboard.

    Updated either by a REST poll (``refresh_dashboard``) or incrementally
    by a WebSocket trade event.
    """
    # Balance
    total_balance: float = 0.0
    stake_currency: str = "USDT"
    balance_change_pct: float = 0.0

    # Profit / Loss
    profit_closed_coin: float = 0.0
    profit_closed_pct: float = 0.0
    profit_all_coin: float = 0.0
    profit_all_pct: float = 0.0
    profit_factor: float = 0.0

    # Counts
    trade_count: int = 0
    closed_trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    open_trade_count: int = 0
    max_open_trades: int = 0

    # Open trades detail
    open_trades: list[dict[str, Any]] = field(default_factory=list)

    # Recent trade history (closed or latest trades)
    recent_trades: list[dict[str, Any]] = field(default_factory=list)

    # Performance per pair
    performance: list[dict[str, Any]] = field(default_factory=list)

    # Daily profit history (last 30 days)
    daily_profits: list[dict[str, Any]] = field(default_factory=list)

    # Bot metadata
    bot_state: str = ""          # "running" | "stopped" | …
    strategy: str = ""
    version: str = ""
    dry_run: bool = True
    exchange: str = ""

    @property
    def win_rate(self) -> float:
        """Win rate as 0–100 percentage.  0 if no closed trades."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return (self.winning_trades / total) * 100.0

    @property
    def has_data(self) -> bool:
        return self.bot_state != ""


# ── TradeManager ─────────────────────────────────────────────────────────────

class TradeManager:
    """
    Single point of control for the GUI.

    Combines REST polling, WebSocket push events, and process lifecycle
    into one unified interface with callback-based async delivery.
    """

    def __init__(self) -> None:
        # ── Sub-components ────────────────────────────────────────────────
        self._rest = FreqtradeClient()
        self._ws = FreqtradeWSClient()
        self._process = BotProcessManager()

        # ── Cached state ──────────────────────────────────────────────────
        self._snapshot = DashboardSnapshot()
        self._snapshot_lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._last_ws_refresh_ts: float = 0.0

        # ── Polling ───────────────────────────────────────────────────────
        self._poll_interval: float = 5.0
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

        # ── Overall status ────────────────────────────────────────────────
        self.status: OverallStatus = OverallStatus.DISCONNECTED

        # ── GUI Callbacks (all fire from worker threads) ──────────────────
        self.on_dashboard_update: Optional[Callable[[DashboardSnapshot], None]] = None
        self.on_trade_event: Optional[Callable[[WSMessage], None]] = None
        self.on_bot_state_change: Optional[Callable[[BotState], None]] = None
        self.on_connection_change: Optional[Callable[[OverallStatus], None]] = None
        self.on_ws_state_change: Optional[Callable[[WSState], None]] = None
        self.on_log: Optional[Callable[[BotLogEvent], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # ── Wire internal sub-component callbacks ─────────────────────────
        self._rest.set_state_callback(self._on_rest_state)
        self._ws.on_state_change = self._on_ws_state
        self._ws.on_message = self._on_ws_message
        self._process.on_state_change = self._on_bot_state
        self._process.on_log = self._on_bot_log

        logger.info("TradeManager initialized")

    # ══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════

    def configure(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        username: str = "",
        password: str = "",
        ws_token: str = "",
    ) -> None:
        """Configure REST + WS connection parameters manually."""
        host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        self._rest.configure(host=host, port=port,
                             username=username, password=password)
        self._ws.configure(host=host, port=port, ws_token=ws_token)
        self._process.configure(api_host=host, api_port=port,
                                api_username=username, api_password=password)
        logger.info("TradeManager configured → %s:%d", host, port)

    def configure_from_config(self, config_path: str | Path) -> None:
        """
        One-shot configuration from a Freqtrade JSON config file.
        Populates REST client, WS client, and process manager credentials.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            cfg: dict = json.load(f)

        api_cfg = cfg.get("api_server", {})
        raw_host = api_cfg.get("listen_ip_address", "127.0.0.1")
        host = "127.0.0.1" if raw_host in ("0.0.0.0", "::", "") else raw_host
        port = api_cfg.get("listen_port", 8080)
        username = api_cfg.get("username", "")
        password = api_cfg.get("password", "")
        ws_token = api_cfg.get("ws_token", "")

        self._rest.configure(host=host, port=port,
                             username=username, password=password)
        self._ws.configure(host=host, port=port, ws_token=ws_token)
        self._process.configure(
            api_host=host, api_port=port,
            api_username=username, api_password=password,
            config_path=str(path),
        )

        # Also read strategy/model for the process manager
        freqai_cfg = cfg.get("freqai", {})
        strategy = cfg.get("strategy", "")
        model = freqai_cfg.get("model_name", "")
        if strategy:
            self._process.configure(strategy=strategy)
        if model:
            self._process.configure(freqai_model=model)

        logger.info("TradeManager configured from %s", path)

    # ══════════════════════════════════════════════════════════════════════
    # CONNECT / DISCONNECT
    # ══════════════════════════════════════════════════════════════════════

    def connect(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        """
        Login via REST then connect WebSocket.  Non-blocking.

        ``on_done`` receives the login ``ApiResult``.
        """
        # Disconnect any stale WS/REST session before reconnecting
        self._ws.disconnect()

        self._set_status(OverallStatus.CONNECTING)

        def _after_login(result: ApiResult) -> None:
            if result.success:
                logger.info("REST login OK — fetching ws_token from running bot…")
                # Retrieve ws_token from the live bot via /show_config so it
                # always matches the running instance (even if the local JSON
                # config file differs).
                try:
                    cfg_result = self._rest.get_show_config()
                    if cfg_result.success and isinstance(cfg_result.data, dict):
                        live_ws_token = cfg_result.data.get("ws_token", "")
                        if live_ws_token:
                            self._ws.configure(
                                host=self._ws._host,
                                port=self._ws._port,
                                ws_token=live_ws_token,
                            )
                except Exception as exc:
                    logger.warning("Could not fetch ws_token from bot: %s", exc)

                self._ws.connect()
                self._set_status(OverallStatus.CONNECTED)
            else:
                logger.warning("REST login failed: %s", result.user_message)
                self._set_status(OverallStatus.ERROR)
                if self.on_error:
                    self.on_error(result.user_message)

            if on_done:
                on_done(result)

        self._rest.login_async(on_done=_after_login)

    def disconnect(self) -> None:
        """Cleanly shut down WS + REST.  Does NOT stop the bot process."""
        self.stop_polling()
        self._ws.disconnect()
        self._rest.close()
        self._set_status(OverallStatus.DISCONNECTED)
        logger.info("TradeManager disconnected")

    @property
    def is_connected(self) -> bool:
        return self._rest.is_connected

    # ══════════════════════════════════════════════════════════════════════
    # BOT PROCESS LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    def start_bot(self, run_command: str = "trade") -> None:
        """Start the Freqtrade bot process with the requested CLI command."""
        self._process.start(run_command=run_command)

    def start_learning(self, timerange: str = "") -> None:
        """Train a FreqAI model via ``freqtrade backtesting --freqai``.

        Runs in train-only mode: the process is stopped automatically once
        model training finishes, before the backtesting phase begins.
        """
        self._process.configure(
            run_command="backtesting",
            timerange=timerange,
            freqai_backtest_live=False,
            train_only=True,
        )
        self._process.start(run_command="backtesting")

    def run_backtesting(self, timerange: str = "") -> None:
        """Backtest a pre-trained FreqAI model via
        ``freqtrade backtesting --freqai-backtest-live-models``.

        Requires that a model was previously trained and saved under
        ``user_data/models/``.
        """
        self._process.configure(
            run_command="backtesting",
            timerange=timerange,
            freqai_backtest_live=True,
        )
        self._process.start(run_command="backtesting")

    def stop_bot(self) -> None:
        """Stop the Freqtrade bot process."""
        self._process.stop()

    def restart_bot(self) -> None:
        """Restart the Freqtrade bot process."""
        self._process.restart()

    @property
    def bot_state(self) -> BotState:
        return self._process.state

    @property
    def bot_run_command(self) -> str:
        """Current run command of the bot process ('trade', 'backtesting', etc.)."""
        return self._process.run_command

    @property
    def bot_is_running(self) -> bool:
        return self._process.is_running

    def configure_launch_mode(self, mode: LaunchMode) -> None:
        """Switch between Docker and subprocess modes."""
        self._process.configure(launch_mode=mode)

    def get_available_launch_modes(self) -> list[LaunchMode]:
        return self._process.get_available_modes()

    def get_process_info(self) -> dict[str, Any]:
        return self._process.get_process_info()

    # ══════════════════════════════════════════════════════════════════════
    # DASHBOARD POLLING
    # ══════════════════════════════════════════════════════════════════════

    def start_polling(self, interval_sec: float = 5.0) -> None:
        """
        Begin periodic dashboard refresh via REST API.

        Polls ``get_dashboard_state`` every ``interval_sec`` seconds in a
        background thread.  Each poll updates the internal snapshot and
        fires ``on_dashboard_update``.
        """
        if self._poll_thread is not None and self._poll_thread.is_alive():
            logger.warning("Polling already running")
            return

        self._poll_interval = interval_sec
        self._poll_stop.clear()

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="DashboardPoll",
        )
        self._poll_thread.start()
        logger.info("Dashboard polling started (%.1fs interval)", interval_sec)

    def stop_polling(self) -> None:
        """Stop the periodic dashboard refresh."""
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=self._poll_interval + 2)
            self._poll_thread = None
            logger.info("Dashboard polling stopped")

    @property
    def is_polling(self) -> bool:
        return self._poll_thread is not None and self._poll_thread.is_alive()

    def refresh_dashboard(
        self,
        on_done: Optional[Callable[[DashboardSnapshot], None]] = None,
    ) -> None:
        """
        One-shot async dashboard refresh.

        Fetches all dashboard data from REST, updates the internal snapshot,
        fires ``on_dashboard_update`` and the optional ``on_done`` callback.
        """
        def _worker() -> None:
            snapshot = self._fetch_and_build_snapshot()
            if on_done:
                on_done(snapshot)

        threading.Thread(target=_worker, daemon=True, name="DashRefresh").start()

    @property
    def snapshot(self) -> DashboardSnapshot:
        """Current dashboard state.  Thread-safe read."""
        with self._snapshot_lock:
            return self._snapshot

    # ══════════════════════════════════════════════════════════════════════
    # TRADE ACTIONS (async, GUI-safe)
    # ══════════════════════════════════════════════════════════════════════

    def force_entry_async(
        self,
        pair: str,
        side: str = "long",
        price: Optional[float] = None,
        stake_amount: Optional[float] = None,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        """Open a trade immediately.  Non-blocking."""
        self._rest.force_enter_async(
            pair=pair,
            side=side,
            on_done=self._wrap_action_callback("force_entry", on_done),
            price=price,
            stake_amount=stake_amount,
        )

    def force_exit_async(
        self,
        trade_id: int | str,
        order_type: str = "market",
        amount: Optional[float] = None,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        """Close a trade immediately.  Non-blocking."""
        self._rest.force_exit_async(
            trade_id=trade_id,
            order_type=order_type,
            amount=amount,
            on_done=self._wrap_action_callback("force_exit", on_done),
        )

    def start_trading_async(
        self,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        """Send /start to the bot (resume trading).  Non-blocking."""
        self._rest.start_async(
            on_done=self._wrap_action_callback("start_trading", on_done),
        )

    def stop_trading_async(
        self,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        """Send /stop to the bot (pause trading).  Non-blocking."""
        self._rest.stop_async(
            on_done=self._wrap_action_callback("stop_trading", on_done),
        )

    def reload_config_async(
        self,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        """Hot-reload the bot's config.  Non-blocking."""
        self._rest.reload_config_async(
            on_done=self._wrap_action_callback("reload_config", on_done),
        )

    # ── Sync convenience (for scripts / tests) ───────────────────────────

    def get_open_trades(self) -> ApiResult:
        return self._rest.get_status()

    def get_trade_history(self, limit: int = 50, offset: int = 0) -> ApiResult:
        return self._rest.get_trades(limit=limit, offset=offset)

    def get_profit(self) -> ApiResult:
        return self._rest.get_profit()

    def get_balance(self) -> ApiResult:
        return self._rest.get_balance()

    def get_performance(self) -> ApiResult:
        return self._rest.get_performance()

    def get_daily(self, days: int = 7) -> ApiResult:
        return self._rest.get_daily(timescale=days)

    def get_whitelist(self) -> ApiResult:
        return self._rest.get_whitelist()

    def get_logs(self, limit: int = 50) -> ApiResult:
        return self._rest.get_logs(limit=limit)

    # ══════════════════════════════════════════════════════════════════════
    # SUB-COMPONENT ACCESS (for advanced GUI panels)
    # ══════════════════════════════════════════════════════════════════════

    @property
    def rest_client(self) -> FreqtradeClient:
        """Direct REST client access — use for endpoints not wrapped here."""
        return self._rest

    @property
    def ws_client(self) -> FreqtradeWSClient:
        """Direct WebSocket client access."""
        return self._ws

    @property
    def process_manager(self) -> BotProcessManager:
        """Direct process manager access."""
        return self._process

    @property
    def server_info(self) -> ServerInfo:
        return self._rest.server_info

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL — Polling Loop
    # ══════════════════════════════════════════════════════════════════════

    def _poll_loop(self) -> None:
        """Background thread: periodically fetch dashboard data."""
        while not self._poll_stop.is_set():
            if self._rest.is_connected:
                try:
                    self._fetch_and_build_snapshot()
                except Exception as exc:
                    logger.warning("Dashboard poll error: %s", exc)

            self._poll_stop.wait(timeout=self._poll_interval)

    def _fetch_and_build_snapshot(self) -> DashboardSnapshot:
        """
        Call ``get_dashboard_state()`` and transform the result into a
        ``DashboardSnapshot``.  Stores it internally and fires the callback.
        """
        with self._fetch_lock:
            result = self._rest.get_dashboard_state()

            with self._snapshot_lock:
                if result.success or (result.data is not None):
                    self._snapshot = self._build_snapshot(result.data or {})

                    # Fetch daily P&L for the performance chart.
                    daily_result = self._rest.get_daily(timescale=30)
                    if daily_result.success and isinstance(daily_result.data, dict):
                        self._snapshot.daily_profits = daily_result.data.get("data", [])

                    # Fetch latest trade history for the "Recent Trade History" table.
                    trades_result = self._rest.get_trades(limit=20, offset=0)
                    if trades_result.success and isinstance(trades_result.data, list):
                        self._snapshot.recent_trades = trades_result.data

                    # Fetch running bot metadata (strategy, state, mode, etc.).
                    cfg_result = self._rest.get_show_config()
                    if cfg_result.success and isinstance(cfg_result.data, dict):
                        cfg = cfg_result.data
                        self._snapshot.bot_state = str(cfg.get("state", "") or "")
                        self._snapshot.strategy = str(cfg.get("strategy", "") or "")
                        self._snapshot.version = str(self._rest.server_info.version or "")
                        self._snapshot.dry_run = bool(cfg.get("dry_run", True))
                        self._snapshot.exchange = str(cfg.get("exchange", "") or "")
                        if self._snapshot.max_open_trades <= 0:
                            self._snapshot.max_open_trades = int(cfg.get("max_open_trades", 0) or 0)

                snapshot = self._snapshot

        if self.on_dashboard_update:
            try:
                self.on_dashboard_update(snapshot)
            except Exception:
                pass

        return snapshot

    @staticmethod
    def _build_snapshot(data: dict[str, Any]) -> DashboardSnapshot:
        """
        Transform the raw ``get_dashboard_state`` dict into a typed snapshot.

        Expected keys: status, profit, balance, count, performance
        """
        snap = DashboardSnapshot()

        # ── Profit ────────────────────────────────────────────────────────
        profit = data.get("profit")
        if isinstance(profit, dict):
            snap.profit_closed_coin = profit.get("profit_closed_coin", 0.0)
            snap.profit_closed_pct = profit.get("profit_closed_percent", 0.0)
            snap.profit_all_coin = profit.get("profit_all_coin", 0.0)
            snap.profit_all_pct = profit.get("profit_all_percent", 0.0)
            snap.profit_factor = profit.get("profit_factor", 0.0)
            snap.trade_count = profit.get("trade_count", 0)
            snap.closed_trade_count = profit.get("closed_trade_count", 0)
            snap.winning_trades = profit.get("winning_trades", 0)
            snap.losing_trades = profit.get("losing_trades", 0)

        # ── Balance ───────────────────────────────────────────────────────
        balance = data.get("balance")
        if isinstance(balance, dict):
            snap.total_balance = balance.get("total", 0.0)
            snap.stake_currency = balance.get("stake", "USDT")
            # If we have starting capital, compute change %
            starting = balance.get("starting_capital", 0.0)
            if starting > 0:
                snap.balance_change_pct = (
                    (snap.total_balance - starting) / starting
                ) * 100.0

        # ── Trade count ───────────────────────────────────────────────────
        count = data.get("count")
        if isinstance(count, dict):
            snap.open_trade_count = count.get("current", 0)
            snap.max_open_trades = count.get("max", 0)

        # ── Open trades (status) ──────────────────────────────────────────
        status = data.get("status")
        if isinstance(status, list):
            snap.open_trades = status

        # ── Performance ───────────────────────────────────────────────────
        perf = data.get("performance")
        if isinstance(perf, list):
            snap.performance = perf

        return snap

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL — WebSocket Message Handler
    # ══════════════════════════════════════════════════════════════════════

    def _on_ws_message(self, msg: WSMessage) -> None:
        """
        Dispatch incoming WS messages.

        Trade events trigger an immediate dashboard refresh so the GUI
        updates without waiting for the next poll cycle.
        """
        logger.debug("WS message: %s", msg.msg_type)

        # Forward every message to the GUI
        if self.on_trade_event:
            try:
                self.on_trade_event(msg)
            except Exception:
                pass

        # On trade-related events, refresh dashboard quickly (debounced).
        if msg.msg_type in (
            MessageType.ENTRY,
            MessageType.ENTRY_FILL,
            MessageType.ENTRY_CANCEL,
            MessageType.EXIT,
            MessageType.EXIT_FILL,
            MessageType.EXIT_CANCEL,
        ):
            now = time.monotonic()
            if (now - self._last_ws_refresh_ts) >= 1.0:
                self._last_ws_refresh_ts = now
                logger.info("Trade WS event detected — refreshing dashboard")
                self.refresh_dashboard()

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL — Sub-component State Callbacks
    # ══════════════════════════════════════════════════════════════════════

    def _on_rest_state(self, state: ConnectionState) -> None:
        """REST client state changed → recompute overall status."""
        logger.debug("REST state → %s", state.name)
        self._recompute_status()

    def _on_ws_state(self, state: WSState) -> None:
        """WS client state changed → recompute overall status."""
        logger.debug("WS state → %s", state.name)
        if self.on_ws_state_change:
            try:
                self.on_ws_state_change(state)
            except Exception:
                pass
        self._recompute_status()

    def _on_bot_state(self, state: BotState) -> None:
        """Bot process state changed → forward to GUI."""
        logger.debug("Bot state → %s", state.name)
        if self.on_bot_state_change:
            try:
                self.on_bot_state_change(state)
            except Exception:
                pass

        # Auto-connect when bot comes online
        if state == BotState.RUNNING and not self._rest.is_connected:
            logger.info("Bot is RUNNING — auto-connecting REST + WS…")
            self.connect()

    def _on_bot_log(self, event: BotLogEvent) -> None:
        """Forward process manager log events to GUI."""
        if self.on_log:
            try:
                self.on_log(event)
            except Exception:
                pass

    def _recompute_status(self) -> None:
        """Derive ``OverallStatus`` from REST + WS states."""
        rest_state = self._rest.state
        ws_state = self._ws.state

        if rest_state == ConnectionState.ERROR:
            new_status = OverallStatus.ERROR
        elif rest_state == ConnectionState.CONNECTING:
            new_status = OverallStatus.CONNECTING
        elif rest_state == ConnectionState.CONNECTED:
            if ws_state == WSState.CONNECTED:
                new_status = OverallStatus.LIVE
            else:
                new_status = OverallStatus.CONNECTED
        else:
            new_status = OverallStatus.DISCONNECTED

        self._set_status(new_status)

    def _set_status(self, new_status: OverallStatus) -> None:
        if self.status != new_status:
            old = self.status
            self.status = new_status
            logger.debug("OverallStatus: %s → %s", old.name, new_status.name)
            if self.on_connection_change:
                try:
                    self.on_connection_change(new_status)
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL — Action Callback Wrapper
    # ══════════════════════════════════════════════════════════════════════

    def _wrap_action_callback(
        self,
        action_name: str,
        on_done: Optional[Callable[[ApiResult], None]],
    ) -> Callable[[ApiResult], None]:
        """
        Wrap a user-provided ``on_done`` callback with logging and
        an automatic dashboard refresh on success.
        """
        def _wrapper(result: ApiResult) -> None:
            if result.success:
                logger.info("Action '%s' succeeded", action_name)
                # Refresh dashboard after successful actions
                self.refresh_dashboard()
            else:
                logger.warning("Action '%s' failed: %s",
                               action_name, result.user_message)

            if on_done:
                on_done(result)

        return _wrapper

    # ══════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """
        Release all resources.  Does NOT stop the bot process.
        Call ``stop_bot()`` first if you want to shut down the bot.
        """
        self.stop_polling()
        self._ws.disconnect()
        self._rest.close()
        self._process.close()
        self._set_status(OverallStatus.DISCONNECTED)
        logger.info("TradeManager closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
