"""
rest_client.py — Freqtrade REST API Client
===========================================

Thread-safe, GUI-friendly HTTP client wrapping Freqtrade's REST API.

Architecture (follows the same Signal/Slot bridge pattern as chart_widget.py):
──────────────────────────────────────────────────────────────────────────────
All network calls run in background threads via ``_request_async``.  The caller
provides an ``on_done`` callback which is invoked from the worker thread.
In a PyQt6 context the callback should be a ``pyqtSignal.emit`` so that the
actual slot runs on the GUI thread.

JWT Authentication Flow:
    1. ``login()`` sends POST /api/v1/token/login with HTTP Basic Auth.
    2. The server returns ``access_token`` (15 min) + ``refresh_token``.
    3. Every subsequent request adds ``Authorization: Bearer <access_token>``.
    4. When a request gets 401, the client automatically refreshes the
       token via ``/api/v1/token/refresh`` and retries once.

Connection State Machine:
    DISCONNECTED  →  CONNECTING  →  CONNECTED
          ↑                              │
          └───────── ERROR ◄─────────────┘

Usage from GUI:
    client = FreqtradeClient()
    client.configure("127.0.0.1", 8080, "emirhan", "3696")

    # Async login (GUI-safe)
    client.login_async(on_done=self._login_done_signal.emit)

    # Async data fetch
    client.get_status_async(on_done=self._status_signal.emit)

    # Sync usage (scripts / testing — blocks the caller)
    result = client.get_profit()
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

import requests
from requests.auth import HTTPBasicAuth

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────
API_BASE_PATH = "/api/v1"
DEFAULT_TIMEOUT = 10          # seconds per HTTP request
TOKEN_REFRESH_MARGIN = 60     # refresh token this many seconds before expiry


# ── Connection State ─────────────────────────────────────────────────────────

class ConnectionState(Enum):
    """Observable state that the GUI can bind to for status indicators."""
    DISCONNECTED = auto()
    CONNECTING   = auto()
    CONNECTED    = auto()
    ERROR        = auto()


# ── Result Objects ───────────────────────────────────────────────────────────

@dataclass
class ApiResult:
    """
    Uniform result object returned by every API call.

    GUI checks ``success`` first, then reads ``data`` or ``user_message``.
    """
    success: bool
    data: Any = None                  # Parsed JSON response body (dict | list)
    status_code: int = 0              # HTTP status code
    error_code: str = ""              # Machine-readable error key
    user_message: str = ""            # Human-readable message for the UI
    endpoint: str = ""                # Which endpoint was called


@dataclass
class ServerInfo:
    """Cached server metadata populated after a successful login."""
    version: str = ""
    strategy: str = ""
    state: str = ""                    # "running" | "stopped" | "paused"
    dry_run: bool = True
    exchange: str = ""
    trading_mode: str = ""
    stake_currency: str = ""
    max_open_trades: int = 0


# ── FreqtradeClient ─────────────────────────────────────────────────────────

class FreqtradeClient:
    """
    Stateful HTTP client for the Freqtrade REST API.

    Design principles matching the existing DataManager:
      • Non-blocking ``*_async`` methods for GUI usage
      • Synchronous counterparts for scripts / tests
      • User-friendly error messages in every ApiResult
      • Thread-safe JWT token storage with automatic refresh
    """

    def __init__(self) -> None:
        # ── Connection config ─────────────────────────────────────────────
        self._host: str = "127.0.0.1"
        self._port: int = 8080
        self._username: str = ""
        self._password: str = ""

        # ── JWT tokens (guarded by _token_lock) ──────────────────────────
        self._access_token: str = ""
        self._refresh_token: str = ""
        self._token_lock = threading.Lock()

        # ── Observable state ──────────────────────────────────────────────
        self.state: ConnectionState = ConnectionState.DISCONNECTED
        self.server_info: ServerInfo = ServerInfo()

        # ── Shared requests session (connection pooling) ──────────────────
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # ── State change callback (GUI can observe) ──────────────────────
        self._on_state_change: Optional[Callable[[ConnectionState], None]] = None

        logger.info("FreqtradeClient initialized (disconnected)")

    # ══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════

    def configure(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        username: str = "",
        password: str = "",
    ) -> None:
        """Set connection parameters.  Call before ``login()``."""
        self._host = host.strip()
        self._port = port
        self._username = username
        self._password = password
        self._clear_tokens()
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("Configured → %s:%d user=%s", self._host, self._port, self._username)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "FreqtradeClient":
        """
        Build a client pre-configured from a Freqtrade JSON config file.

        Reads ``api_server.listen_ip_address``, ``listen_port``,
        ``username``, ``password`` from the config.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            cfg: dict = json.load(f)

        api_cfg = cfg.get("api_server", {})
        instance = cls()
        instance.configure(
            host=api_cfg.get("listen_ip_address", "127.0.0.1"),
            port=api_cfg.get("listen_port", 8080),
            username=api_cfg.get("username", ""),
            password=api_cfg.get("password", ""),
        )
        return instance

    def set_state_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """Register a callback invoked whenever ``self.state`` changes."""
        self._on_state_change = callback

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}{API_BASE_PATH}"

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED

    # ══════════════════════════════════════════════════════════════════════
    # AUTHENTICATION
    # ══════════════════════════════════════════════════════════════════════

    def login(self) -> ApiResult:
        """
        Synchronous login.  Acquires JWT tokens via HTTP Basic Auth.
        On success, also fetches ``show_config`` to populate ``server_info``.
        """
        self._set_state(ConnectionState.CONNECTING)

        url = f"{self.base_url}/token/login"
        try:
            resp = self._session.post(
                url,
                auth=HTTPBasicAuth(self._username, self._password),
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.ConnectionError:
            return self._connection_error("login")
        except requests.Timeout:
            return self._timeout_error("login")
        except requests.RequestException as exc:
            return self._generic_request_error("login", exc)

        if resp.status_code == 200:
            body = resp.json()
            with self._token_lock:
                self._access_token = body.get("access_token", "")
                self._refresh_token = body.get("refresh_token", "")
            self._set_state(ConnectionState.CONNECTED)
            logger.info("Login successful → %s:%d", self._host, self._port)

            # Fetch server info right after login
            self._fetch_server_info()

            return ApiResult(
                success=True,
                data=body,
                status_code=200,
                endpoint="token/login",
            )

        self._set_state(ConnectionState.ERROR)
        return ApiResult(
            success=False,
            status_code=resp.status_code,
            error_code="AUTH_FAILED",
            user_message=self._auth_error_message(resp.status_code),
            endpoint="token/login",
        )

    def login_async(
        self,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        """Non-blocking login.  ``on_done`` fires from the worker thread."""
        self._run_in_thread(self.login, on_done)

    def logout(self) -> None:
        """Clear tokens and reset state."""
        self._clear_tokens()
        self._set_state(ConnectionState.DISCONNECTED)
        self.server_info = ServerInfo()
        logger.info("Logged out")

    def refresh_token(self) -> bool:
        """
        Refresh the access token using the stored refresh token.
        Returns True on success.
        """
        with self._token_lock:
            rt = self._refresh_token
        if not rt:
            return False

        url = f"{self.base_url}/token/refresh"
        try:
            resp = self._session.post(
                url,
                headers={"Authorization": f"Bearer {rt}"},
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.RequestException:
            return False

        if resp.status_code == 200:
            body = resp.json()
            with self._token_lock:
                self._access_token = body.get("access_token", self._access_token)
            logger.debug("Token refreshed")
            return True

        logger.warning("Token refresh failed (HTTP %d)", resp.status_code)
        return False

    # ══════════════════════════════════════════════════════════════════════
    # CORE HTTP TRANSPORT
    # ══════════════════════════════════════════════════════════════════════

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
        params: dict | None = None,
        retry_auth: bool = True,
    ) -> ApiResult:
        """
        Central HTTP dispatcher.  All public endpoint methods go through here.

        Handles:
          • Bearer token injection
          • Automatic 401 → token refresh → retry (once)
          • Connection / timeout / generic error wrapping
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        with self._token_lock:
            token = self._access_token

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            resp = self._session.request(
                method=method,
                url=url,
                json=payload,
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.ConnectionError:
            self._set_state(ConnectionState.ERROR)
            return self._connection_error(endpoint)
        except requests.Timeout:
            return self._timeout_error(endpoint)
        except requests.RequestException as exc:
            return self._generic_request_error(endpoint, exc)

        # ── Auto-refresh on 401 (one retry) ──────────────────────────────
        if resp.status_code == 401 and retry_auth:
            if self.refresh_token():
                return self._request(method, endpoint, payload, params, retry_auth=False)
            # Refresh also failed → force re-login
            self._set_state(ConnectionState.ERROR)
            return ApiResult(
                success=False,
                status_code=401,
                error_code="TOKEN_EXPIRED",
                user_message="Session expired. Please log in again.",
                endpoint=endpoint,
            )

        # ── Parse response ────────────────────────────────────────────────
        try:
            body = resp.json()
        except ValueError:
            body = resp.text

        if 200 <= resp.status_code < 300:
            return ApiResult(
                success=True,
                data=body,
                status_code=resp.status_code,
                endpoint=endpoint,
            )

        return ApiResult(
            success=False,
            data=body,
            status_code=resp.status_code,
            error_code=f"HTTP_{resp.status_code}",
            user_message=self._error_message_for_status(resp.status_code, body),
            endpoint=endpoint,
        )

    def _get(self, endpoint: str, params: dict | None = None) -> ApiResult:
        return self._request("GET", endpoint, params=params)

    def _post(self, endpoint: str, payload: dict | None = None) -> ApiResult:
        return self._request("POST", endpoint, payload=payload)

    def _delete(self, endpoint: str, payload: dict | None = None) -> ApiResult:
        return self._request("DELETE", endpoint, payload=payload)

    # ══════════════════════════════════════════════════════════════════════
    # ASYNC WRAPPER (Thread Bridge)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _run_in_thread(
        func: Callable[..., ApiResult],
        on_done: Optional[Callable[[ApiResult], None]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Execute ``func`` in a daemon thread and pass the result to ``on_done``.

        ``on_done`` is called from the worker thread — in PyQt6 this should
        be a ``pyqtSignal.emit`` so the connected slot runs on the GUI thread.
        """
        def _worker() -> None:
            result = func(*args, **kwargs)
            if on_done:
                on_done(result)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _request_async(
        self,
        method: str,
        endpoint: str,
        on_done: Optional[Callable[[ApiResult], None]] = None,
        payload: dict | None = None,
        params: dict | None = None,
    ) -> None:
        """Fire an async HTTP request — thin wrapper around ``_run_in_thread``."""
        self._run_in_thread(
            self._request, on_done,
            method, endpoint, payload, params,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — HEALTH / INFO
    # ══════════════════════════════════════════════════════════════════════

    def ping(self) -> ApiResult:
        """GET /ping — no auth required.  Quick connectivity test."""
        url = f"{self.base_url}/ping"
        try:
            resp = self._session.get(url, timeout=DEFAULT_TIMEOUT)
            return ApiResult(
                success=resp.status_code == 200,
                data=resp.json() if resp.status_code == 200 else None,
                status_code=resp.status_code,
                endpoint="ping",
            )
        except requests.ConnectionError:
            return self._connection_error("ping")
        except requests.Timeout:
            return self._timeout_error("ping")
        except requests.RequestException as exc:
            return self._generic_request_error("ping", exc)

    def ping_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._run_in_thread(self.ping, on_done)

    def get_version(self) -> ApiResult:
        """GET /version"""
        return self._get("version")

    def get_version_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "version", on_done)

    def get_show_config(self) -> ApiResult:
        """GET /show_config — returns the running bot's configuration."""
        return self._get("show_config")

    def get_show_config_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "show_config", on_done)

    def get_health(self) -> ApiResult:
        """GET /health — last bot processing loop timestamp."""
        return self._get("health")

    def get_health_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "health", on_done)

    def get_sysinfo(self) -> ApiResult:
        """GET /sysinfo — CPU / RAM usage of the bot host."""
        return self._get("sysinfo")

    def get_sysinfo_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "sysinfo", on_done)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — TRADING STATE
    # ══════════════════════════════════════════════════════════════════════

    def get_status(self) -> ApiResult:
        """GET /status — list of all open trades."""
        return self._get("status")

    def get_status_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "status", on_done)

    def get_count(self) -> ApiResult:
        """GET /count — open trade count vs max."""
        return self._get("count")

    def get_count_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "count", on_done)

    def get_profit(self) -> ApiResult:
        """GET /profit — cumulative profit/loss stats."""
        return self._get("profit")

    def get_profit_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "profit", on_done)

    def get_balance(self) -> ApiResult:
        """GET /balance — account balances per currency."""
        return self._get("balance")

    def get_balance_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "balance", on_done)

    def get_performance(self) -> ApiResult:
        """GET /performance — per-pair performance of closed trades."""
        return self._get("performance")

    def get_performance_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "performance", on_done)

    def get_daily(self, timescale: int = 7) -> ApiResult:
        """GET /daily — profit/loss per day for the last N days."""
        return self._get("daily", params={"timescale": timescale})

    def get_daily_async(
        self,
        timescale: int = 7,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        self._request_async("GET", "daily", on_done, params={"timescale": timescale})

    def get_weekly(self, timescale: int = 4) -> ApiResult:
        """GET /weekly"""
        return self._get("weekly", params={"timescale": timescale})

    def get_monthly(self, timescale: int = 3) -> ApiResult:
        """GET /monthly"""
        return self._get("monthly", params={"timescale": timescale})

    def get_stats(self) -> ApiResult:
        """GET /stats — hold duration / exit reason stats."""
        return self._get("stats")

    def get_stats_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "stats", on_done)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — TRADE HISTORY
    # ══════════════════════════════════════════════════════════════════════

    def get_trades(self, limit: int = 50, offset: int = 0) -> ApiResult:
        """GET /trades — paginated trade history."""
        return self._get("trades", params={"limit": limit, "offset": offset})

    def get_trades_async(
        self,
        limit: int = 50,
        offset: int = 0,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        self._request_async("GET", "trades", on_done, params={"limit": limit, "offset": offset})

    def get_trade(self, trade_id: int) -> ApiResult:
        """GET /trade/<trade_id> — single trade detail."""
        return self._get(f"trade/{trade_id}")

    def get_trade_async(
        self,
        trade_id: int,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        self._request_async("GET", f"trade/{trade_id}", on_done)

    def delete_trade(self, trade_id: int) -> ApiResult:
        """DELETE /trades/<trade_id> — remove trade from DB."""
        return self._delete(f"trades/{trade_id}")

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — BOT CONTROLS
    # ══════════════════════════════════════════════════════════════════════

    def start(self) -> ApiResult:
        """POST /start — start trading."""
        return self._post("start")

    def start_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("POST", "start", on_done)

    def stop(self) -> ApiResult:
        """POST /stop — stop trading."""
        return self._post("stop")

    def stop_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("POST", "stop", on_done)

    def pause(self) -> ApiResult:
        """POST /pause — pause trading (handle open trades, no new entries)."""
        return self._post("pause")

    def pause_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("POST", "pause", on_done)

    def stopbuy(self) -> ApiResult:
        """POST /stopbuy — stop entering new trades, let open trades exit normally."""
        return self._post("stopbuy")

    def stopbuy_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("POST", "stopbuy", on_done)

    def reload_config(self) -> ApiResult:
        """POST /reload_config — hot-reload the bot's configuration."""
        result = self._post("reload_config")
        if result.success:
            self._fetch_server_info()
        return result

    def reload_config_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("POST", "reload_config", on_done)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — FORCE ENTRY / EXIT
    # ══════════════════════════════════════════════════════════════════════

    def force_enter(
        self,
        pair: str,
        side: str = "long",
        price: Optional[float] = None,
        order_type: Optional[str] = None,
        stake_amount: Optional[float] = None,
        leverage: Optional[float] = None,
        entry_tag: str = "force_enter",
    ) -> ApiResult:
        """POST /forceenter — open a trade immediately."""
        payload: dict[str, Any] = {
            "pair": pair.strip().upper(),
            "side": side,
        }
        if price is not None:
            payload["price"] = price
        if order_type is not None:
            payload["ordertype"] = order_type
        if stake_amount is not None:
            payload["stakeamount"] = stake_amount
        if leverage is not None:
            payload["leverage"] = leverage
        if entry_tag:
            payload["entry_tag"] = entry_tag

        return self._post("forceenter", payload=payload)

    def force_enter_async(
        self,
        pair: str,
        side: str = "long",
        on_done: Optional[Callable[[ApiResult], None]] = None,
        **kwargs: Any,
    ) -> None:
        self._run_in_thread(self.force_enter, on_done, pair, side, **kwargs)

    def force_exit(
        self,
        trade_id: int | str,
        order_type: str = "market",
        amount: Optional[float] = None,
    ) -> ApiResult:
        """POST /forceexit — close a trade immediately."""
        payload: dict[str, Any] = {
            "tradeid": str(trade_id),
            "ordertype": order_type,
        }
        if amount is not None:
            payload["amount"] = amount

        return self._post("forceexit", payload=payload)

    def force_exit_async(
        self,
        trade_id: int | str,
        order_type: str = "market",
        amount: Optional[float] = None,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        self._run_in_thread(self.force_exit, on_done, trade_id, order_type, amount)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — PAIR LISTS
    # ══════════════════════════════════════════════════════════════════════

    def get_whitelist(self) -> ApiResult:
        """GET /whitelist"""
        return self._get("whitelist")

    def get_whitelist_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "whitelist", on_done)

    def get_blacklist(self) -> ApiResult:
        """GET /blacklist"""
        return self._get("blacklist")

    def get_blacklist_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "blacklist", on_done)

    def add_blacklist(self, pairs: list[str]) -> ApiResult:
        """POST /blacklist — add pairs to the blacklist."""
        return self._post("blacklist", payload={"blacklist": pairs})

    def delete_blacklist(self, pairs: list[str]) -> ApiResult:
        """DELETE /blacklist — remove pairs from the blacklist."""
        return self._delete("blacklist", payload={"blacklist": pairs})

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — LOGS / DATA
    # ══════════════════════════════════════════════════════════════════════

    def get_logs(self, limit: int = 50) -> ApiResult:
        """GET /logs — tail of the bot's log."""
        return self._get("logs", params={"limit": limit})

    def get_logs_async(
        self,
        limit: int = 50,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        self._request_async("GET", "logs", on_done, params={"limit": limit})

    def get_locks(self) -> ApiResult:
        """GET /locks — currently locked pairs."""
        return self._get("locks")

    def get_locks_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "locks", on_done)

    def get_entries(self, pair: str = "") -> ApiResult:
        """GET /entries — entry tag performance."""
        params = {"pair": pair} if pair else None
        return self._get("entries", params=params)

    def get_exits(self, pair: str = "") -> ApiResult:
        """GET /exits — exit reason performance."""
        params = {"pair": pair} if pair else None
        return self._get("exits", params=params)

    def get_mix_tags(self, pair: str = "") -> ApiResult:
        """GET /mix_tags — combined entry_tag + exit_reason stats."""
        params = {"pair": pair} if pair else None
        return self._get("mix_tags", params=params)

    def get_strategies(self) -> ApiResult:
        """GET /strategies — list available strategies."""
        return self._get("strategies")

    def get_strategies_async(self, on_done: Optional[Callable[[ApiResult], None]] = None) -> None:
        self._request_async("GET", "strategies", on_done)

    def get_pair_candles(
        self,
        pair: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> ApiResult:
        """GET /pair_candles — live analyzed dataframe for a pair."""
        return self._get("pair_candles", params={
            "pair": pair,
            "timeframe": timeframe,
            "limit": limit,
        })

    def get_available_pairs(self, timeframe: str = "") -> ApiResult:
        """GET /available_pairs — backtest data availability."""
        params = {"timeframe": timeframe} if timeframe else None
        return self._get("available_pairs", params=params)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — LOCK MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def add_lock(
        self,
        pair: str,
        until: str,
        side: str = "long",
        reason: str = "",
    ) -> ApiResult:
        """POST /locks — lock a pair until a given datetime."""
        payload: dict[str, Any] = {"pair": pair, "until": until, "side": side}
        if reason:
            payload["reason"] = reason
        return self._post("locks", payload=payload)

    def delete_lock(self, lock_id: int) -> ApiResult:
        """DELETE /locks/<lock_id>"""
        return self._delete(f"locks/{lock_id}")

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API — CANCEL ORDER
    # ══════════════════════════════════════════════════════════════════════

    def cancel_open_order(self, trade_id: int) -> ApiResult:
        """DELETE /trades/<trade_id>/open-order"""
        return self._delete(f"trades/{trade_id}/open-order")

    def reload_trade(self, trade_id: int) -> ApiResult:
        """POST /trades/<trade_id>/reload — re-sync trade from exchange."""
        return self._post(f"trades/{trade_id}/reload")

    # ══════════════════════════════════════════════════════════════════════
    # CONVENIENCE — AGGREGATED DASHBOARD STATE
    # ══════════════════════════════════════════════════════════════════════

    def get_dashboard_state(self) -> ApiResult:
        """
        Fetch multiple endpoints and merge into a single dict for the Dashboard.

        Returns ``ApiResult.data`` as::

            {
                "status": [...],       # open trades
                "profit": {...},       # cumulative P/L
                "balance": {...},      # wallet balances
                "count": {...},        # open / max trade count
                "performance": [...],  # per-pair perf
            }

        If any sub-request fails, its key will be ``None`` and the
        ``user_message`` will note which one failed.
        """
        results: dict[str, Any] = {}
        errors: list[str] = []

        for key, fetch_fn in [
            ("status", self.get_status),
            ("profit", self.get_profit),
            ("balance", self.get_balance),
            ("count", self.get_count),
            ("performance", self.get_performance),
        ]:
            r = fetch_fn()
            if r.success:
                results[key] = r.data
            else:
                results[key] = None
                errors.append(key)

        if errors:
            return ApiResult(
                success=False,
                data=results,
                error_code="PARTIAL_FAILURE",
                user_message=f"Failed to fetch: {', '.join(errors)}",
                endpoint="dashboard_state",
            )

        return ApiResult(success=True, data=results, endpoint="dashboard_state")

    def get_dashboard_state_async(
        self,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        self._run_in_thread(self.get_dashboard_state, on_done)

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _fetch_server_info(self) -> None:
        """Populate ``server_info`` from /show_config and /version."""
        try:
            cfg_result = self.get_show_config()
            if cfg_result.success and isinstance(cfg_result.data, dict):
                d = cfg_result.data
                self.server_info = ServerInfo(
                    strategy=d.get("strategy", ""),
                    state=d.get("state", ""),
                    dry_run=d.get("dry_run", True),
                    exchange=d.get("exchange", ""),
                    trading_mode=d.get("trading_mode", ""),
                    stake_currency=d.get("stake_currency", ""),
                    max_open_trades=d.get("max_open_trades", 0),
                )
            ver_result = self.get_version()
            if ver_result.success and isinstance(ver_result.data, dict):
                self.server_info.version = ver_result.data.get("version", "")
        except Exception as exc:
            logger.warning("Could not fetch server info: %s", exc)

    def _set_state(self, new_state: ConnectionState) -> None:
        if self.state != new_state:
            self.state = new_state
            logger.debug("State → %s", new_state.name)
            if self._on_state_change:
                try:
                    self._on_state_change(new_state)
                except Exception:
                    pass

    def _clear_tokens(self) -> None:
        with self._token_lock:
            self._access_token = ""
            self._refresh_token = ""

    # ── Error Factories ───────────────────────────────────────────────────

    def _connection_error(self, endpoint: str) -> ApiResult:
        self._set_state(ConnectionState.ERROR)
        return ApiResult(
            success=False,
            error_code="CONNECTION_ERROR",
            user_message=(
                f"Cannot connect to Freqtrade at {self._host}:{self._port}. "
                "Make sure the bot is running and the API server is enabled."
            ),
            endpoint=endpoint,
        )

    def _timeout_error(self, endpoint: str) -> ApiResult:
        return ApiResult(
            success=False,
            error_code="TIMEOUT",
            user_message=(
                f"Request to /{endpoint} timed out after {DEFAULT_TIMEOUT}s. "
                "The bot may be under heavy load — try again."
            ),
            endpoint=endpoint,
        )

    def _generic_request_error(self, endpoint: str, exc: Exception) -> ApiResult:
        logger.error("Request error [%s]: %s", endpoint, exc)
        return ApiResult(
            success=False,
            error_code="REQUEST_ERROR",
            user_message=f"Network error while calling /{endpoint}. Check your connection.",
            endpoint=endpoint,
        )

    @staticmethod
    def _auth_error_message(status_code: int) -> str:
        if status_code == 401:
            return "Invalid username or password. Check your API credentials."
        if status_code == 403:
            return "Access forbidden. The bot may have a different authentication config."
        return f"Authentication failed (HTTP {status_code})."

    @staticmethod
    def _error_message_for_status(status_code: int, body: Any) -> str:
        """Map HTTP status codes to user-friendly messages."""
        if isinstance(body, dict) and "error" in body:
            return str(body["error"])

        messages = {
            400: "Bad request — check the parameters.",
            401: "Session expired. Please log in again.",
            403: "Access forbidden.",
            404: "Endpoint not found — is the bot version compatible?",
            500: "Internal bot error. Check the bot logs.",
            502: "Bot is unreachable (bad gateway).",
            503: "Bot service unavailable — it may be starting up.",
        }
        return messages.get(status_code, f"Unexpected error (HTTP {status_code}).")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying requests session."""
        self._session.close()
        self._clear_tokens()
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("FreqtradeClient closed")

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
