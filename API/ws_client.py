"""
ws_client.py — Freqtrade WebSocket Client
==========================================

Persistent WebSocket connection to Freqtrade's message endpoint for
real-time push events (trade opened/closed, whitelist changes, etc.).

Architecture (mirrors the Signal/Slot bridge used in chart_widget.py):
──────────────────────────────────────────────────────────────────────
The WebSocket event loop runs in a dedicated ``threading.Thread`` with its
own ``asyncio`` event loop.  Incoming messages are parsed and forwarded to
caller-provided callbacks that should be ``pyqtSignal.emit()`` calls —
keeping all QWidget updates on the GUI thread.

Connection lifecycle::

    DISCONNECTED  ──connect()──►  CONNECTING  ──handshake──►  CONNECTED
          ▲                                                       │
          └──────── RECONNECTING ◄──── on error / close ─────────┘
                        │
                    (max retries exceeded)
                        │
                        ▼
                      ERROR

Auto-reconnect with exponential backoff:
    Delays: 1s → 2s → 4s → 8s → 16s → 30s (capped)
    Resets to 1s on every successful connection.

Freqtrade WebSocket protocol:
    Endpoint:  ws://<host>:<port>/api/v1/message/ws?token=<ws_token>
    Subscribe: {"type": "subscribe", "data": ["entry", "exit", ...]}
    Messages:  {"type": "<RPCMessageType>", "data": {...}}

Usage from GUI::

    ws = FreqtradeWSClient()
    ws.configure("127.0.0.1", 8080, "iPjGQDlNpi4m_HuSlVkuutU9BEEmhz8Vcw")

    # Register callbacks (should be pyqtSignal.emit in practice)
    ws.on_message = self._ws_message_signal.emit
    ws.on_state_change = self._ws_state_signal.emit

    ws.connect()    # non-blocking — spawns background thread
    ws.disconnect() # clean shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

import websockets
import websockets.asyncio.client

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Freqtrade RPC Message Types ─────────────────────────────────────────────
# Mirrors freqtrade/enums/rpcmessagetype.py

class MessageType(str, Enum):
    """Known Freqtrade WebSocket message types."""
    STATUS                   = "status"
    WARNING                  = "warning"
    EXCEPTION                = "exception"
    STARTUP                  = "startup"
    ENTRY                    = "entry"
    ENTRY_FILL               = "entry_fill"
    ENTRY_CANCEL             = "entry_cancel"
    EXIT                     = "exit"
    EXIT_FILL                = "exit_fill"
    EXIT_CANCEL              = "exit_cancel"
    PROTECTION_TRIGGER       = "protection_trigger"
    PROTECTION_TRIGGER_GLOBAL = "protection_trigger_global"
    STRATEGY_MSG             = "strategy_msg"
    WHITELIST                = "whitelist"
    ANALYZED_DF              = "analyzed_df"
    NEW_CANDLE               = "new_candle"

    @classmethod
    def from_str(cls, value: str) -> "MessageType | None":
        """Safely parse a string to MessageType, returning None if unknown."""
        try:
            return cls(value)
        except ValueError:
            return None


# Default subscriptions — trade events + status, excluding heavy dataframes
DEFAULT_SUBSCRIPTIONS: list[str] = [
    MessageType.STATUS.value,
    MessageType.WARNING.value,
    MessageType.STARTUP.value,
    MessageType.ENTRY.value,
    MessageType.ENTRY_FILL.value,
    MessageType.ENTRY_CANCEL.value,
    MessageType.EXIT.value,
    MessageType.EXIT_FILL.value,
    MessageType.EXIT_CANCEL.value,
    MessageType.PROTECTION_TRIGGER.value,
    MessageType.PROTECTION_TRIGGER_GLOBAL.value,
    MessageType.STRATEGY_MSG.value,
    MessageType.WHITELIST.value,
    MessageType.NEW_CANDLE.value,
]


# ── Connection State ─────────────────────────────────────────────────────────

class WSState(Enum):
    """WebSocket connection state.  GUI binds to this for status indicators."""
    DISCONNECTED = auto()
    CONNECTING   = auto()
    CONNECTED    = auto()
    RECONNECTING = auto()
    ERROR        = auto()


# ── Parsed Message ───────────────────────────────────────────────────────────

@dataclass
class WSMessage:
    """
    Parsed WebSocket message delivered to callbacks.

    Attributes:
        msg_type:  The ``MessageType`` enum (or raw string if unknown).
        data:      The ``data`` payload from the message (dict, list, etc.).
        raw:       The original unparsed JSON dict.
    """
    msg_type: MessageType | str
    data: Any = None
    raw: dict = field(default_factory=dict)

    @property
    def is_trade_event(self) -> bool:
        """True for entry/exit related messages."""
        return self.msg_type in (
            MessageType.ENTRY, MessageType.ENTRY_FILL, MessageType.ENTRY_CANCEL,
            MessageType.EXIT, MessageType.EXIT_FILL, MessageType.EXIT_CANCEL,
        )

    @property
    def is_entry(self) -> bool:
        return self.msg_type in (
            MessageType.ENTRY, MessageType.ENTRY_FILL,
        )

    @property
    def is_exit(self) -> bool:
        return self.msg_type in (
            MessageType.EXIT, MessageType.EXIT_FILL,
        )


# ── Reconnect Config ────────────────────────────────────────────────────────

@dataclass
class ReconnectConfig:
    """Exponential backoff parameters for auto-reconnect."""
    enabled: bool = True
    initial_delay: float = 1.0        # seconds
    max_delay: float = 30.0           # cap
    backoff_factor: float = 2.0
    max_retries: int = 0              # 0 = unlimited


# ── FreqtradeWSClient ───────────────────────────────────────────────────────

class FreqtradeWSClient:
    """
    Persistent WebSocket client for real-time Freqtrade RPC messages.

    Runs its own asyncio event loop in a background thread so it never
    blocks the PyQt6 GUI thread.  All callbacks are fired from the
    background thread — in a PyQt6 app they should be ``pyqtSignal.emit``.
    """

    def __init__(self) -> None:
        # ── Connection config ─────────────────────────────────────────────
        self._host: str = "127.0.0.1"
        self._port: int = 8080
        self._ws_token: str = ""
        self._subscriptions: list[str] = list(DEFAULT_SUBSCRIPTIONS)
        self._reconnect_cfg = ReconnectConfig()

        # ── State (observable) ────────────────────────────────────────────
        self.state: WSState = WSState.DISCONNECTED
        self._retry_count: int = 0

        # ── Callbacks (set by the GUI layer) ──────────────────────────────
        self.on_message: Optional[Callable[[WSMessage], None]] = None
        self.on_state_change: Optional[Callable[[WSState], None]] = None

        # ── Internal threading primitives ─────────────────────────────────
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ws: Optional[websockets.asyncio.client.ClientConnection] = None

        logger.info("FreqtradeWSClient initialized (disconnected)")

    # ══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════

    def configure(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        ws_token: str = "",
        subscriptions: list[str] | None = None,
        reconnect: ReconnectConfig | None = None,
    ) -> None:
        """
        Set connection parameters.  Call before ``connect()``.

        Args:
            host:           Bot API host.
            port:           Bot API port.
            ws_token:       ``ws_token`` value from the Freqtrade config.
            subscriptions:  List of RPCMessageType strings to subscribe to.
            reconnect:      Reconnection behaviour config.
        """
        self._host = host.strip()
        self._port = port
        self._ws_token = ws_token
        if subscriptions is not None:
            self._subscriptions = subscriptions
        if reconnect is not None:
            self._reconnect_cfg = reconnect
        logger.info("WS configured → ws://%s:%d  token=%s…",
                     self._host, self._port, self._ws_token[:8] if self._ws_token else "NONE")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "FreqtradeWSClient":
        """
        Build from a Freqtrade JSON config file.
        Reads ``api_server.listen_ip_address``, ``listen_port``, ``ws_token``.
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
            ws_token=api_cfg.get("ws_token", ""),
        )
        return instance

    @property
    def ws_url(self) -> str:
        """Full WebSocket URL with token query parameter."""
        return f"ws://{self._host}:{self._port}/api/v1/message/ws?token={self._ws_token}"

    @property
    def is_connected(self) -> bool:
        return self.state == WSState.CONNECTED

    # ══════════════════════════════════════════════════════════════════════
    # CONNECT / DISCONNECT (Public API)
    # ══════════════════════════════════════════════════════════════════════

    def connect(self) -> None:
        """
        Start the WebSocket connection in a background thread.

        Non-blocking — returns immediately.  Connection progress and
        incoming messages are delivered via callbacks.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("WS already running — call disconnect() first")
            return

        if not self._ws_token:
            logger.error("Cannot connect: ws_token is not configured")
            self._set_state(WSState.ERROR)
            return

        self._stop_event.clear()
        self._retry_count = 0

        self._thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="FreqtradeWS",
        )
        self._thread.start()
        logger.info("WS background thread started")

    def disconnect(self) -> None:
        """
        Gracefully shut down the WebSocket and its background thread.

        Safe to call from the GUI thread (e.g. on application close).
        """
        self._stop_event.set()

        # Schedule the close inside the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._set_state(WSState.DISCONNECTED)
        logger.info("WS disconnected")

    def update_subscriptions(self, subscriptions: list[str]) -> None:
        """
        Update subscriptions on the fly.
        If already connected, sends a new subscribe message immediately.
        """
        self._subscriptions = subscriptions
        if self.state == WSState.CONNECTED and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_subscribe(), self._loop)

    # ══════════════════════════════════════════════════════════════════════
    # BACKGROUND THREAD — Event Loop
    # ══════════════════════════════════════════════════════════════════════

    def _run_event_loop(self) -> None:
        """
        Entry point for the background thread.
        Creates a fresh asyncio event loop and runs the connection loop.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connection_loop())
        except Exception as exc:
            logger.exception("WS event loop crashed: %s", exc)
            self._set_state(WSState.ERROR)
        finally:
            self._loop.close()
            self._loop = None

    async def _connection_loop(self) -> None:
        """
        Outer loop that handles reconnection with exponential backoff.
        Each iteration attempts a full connect → subscribe → listen cycle.
        """
        while not self._stop_event.is_set():
            try:
                await self._connect_and_listen()
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning("WS connection lost: %s", exc)

            # ── Should we reconnect? ─────────────────────────────────────
            if self._stop_event.is_set():
                break

            if not self._reconnect_cfg.enabled:
                self._set_state(WSState.ERROR)
                break

            self._retry_count += 1
            max_retries = self._reconnect_cfg.max_retries
            if max_retries > 0 and self._retry_count > max_retries:
                logger.error("WS max retries (%d) exceeded", max_retries)
                self._set_state(WSState.ERROR)
                break

            delay = min(
                self._reconnect_cfg.initial_delay * (
                    self._reconnect_cfg.backoff_factor ** (self._retry_count - 1)
                ),
                self._reconnect_cfg.max_delay,
            )
            self._set_state(WSState.RECONNECTING)
            logger.info("WS reconnecting in %.1fs (attempt %d)…", delay, self._retry_count)

            # Interruptible sleep — check every 0.5s if stop was requested
            waited = 0.0
            while waited < delay and not self._stop_event.is_set():
                await asyncio.sleep(min(0.5, delay - waited))
                waited += 0.5

    async def _connect_and_listen(self) -> None:
        """
        Single connection lifecycle: connect → subscribe → listen for messages.
        Raises on any connection/protocol error so the outer loop can reconnect.
        """
        self._set_state(WSState.CONNECTING)
        logger.info("WS connecting to %s…", f"ws://{self._host}:{self._port}/api/v1/message/ws")

        async with websockets.asyncio.client.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
            max_size=2 ** 24,          # 16 MB — analyzed_df can be large
        ) as ws:
            self._ws = ws
            self._retry_count = 0       # reset on successful connect
            self._set_state(WSState.CONNECTED)
            logger.info("WS connected — sending subscriptions")

            await self._send_subscribe()
            await self._listen(ws)

    async def _send_subscribe(self) -> None:
        """Send the subscription message to the Freqtrade WS endpoint."""
        if self._ws is None:
            return

        payload = json.dumps({
            "type": "subscribe",
            "data": self._subscriptions,
        })
        await self._ws.send(payload)
        logger.debug("WS subscribed to: %s", self._subscriptions)

    async def _listen(self, ws: websockets.asyncio.client.ClientConnection) -> None:
        """
        Inner receive loop.  Runs until the connection drops or stop is requested.
        """
        async for raw_msg in ws:
            if self._stop_event.is_set():
                break

            try:
                parsed = self._parse_message(raw_msg)
                if parsed is not None:
                    self._dispatch(parsed)
            except Exception as exc:
                logger.warning("WS message parse error: %s — raw: %s",
                               exc, raw_msg[:200] if isinstance(raw_msg, str) else "<bytes>")

    # ══════════════════════════════════════════════════════════════════════
    # MESSAGE PARSING & DISPATCH
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_message(raw: str | bytes) -> WSMessage | None:
        """
        Parse a raw WebSocket frame into a ``WSMessage``.
        Returns None for messages we can't decode (logged as warning).
        """
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")

        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.debug("WS non-dict message ignored: %s", type(data))
            return None

        raw_type = data.get("type", "")
        msg_type = MessageType.from_str(raw_type)

        return WSMessage(
            msg_type=msg_type if msg_type is not None else raw_type,
            data=data.get("data"),
            raw=data,
        )

    def _dispatch(self, message: WSMessage) -> None:
        """
        Deliver the parsed message to the registered callback.

        In a PyQt6 app ``self.on_message`` should be a ``pyqtSignal.emit``
        so that the actual processing slot runs on the GUI thread.
        """
        logger.debug("WS ← %s", message.msg_type)

        if self.on_message is not None:
            try:
                self.on_message(message)
            except Exception as exc:
                logger.error("WS on_message callback error: %s", exc)

    # ══════════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def _set_state(self, new_state: WSState) -> None:
        if self.state != new_state:
            old = self.state
            self.state = new_state
            logger.debug("WS state: %s → %s", old.name, new_state.name)
            if self.on_state_change is not None:
                try:
                    self.on_state_change(new_state)
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """Alias for ``disconnect()``.  Matches ``FreqtradeClient.close()``."""
        self.disconnect()

    def __del__(self) -> None:
        try:
            self._stop_event.set()
        except Exception:
            pass
