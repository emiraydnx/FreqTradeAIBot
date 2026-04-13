"""
exchange_manager.py — Bot Process Lifecycle Manager
====================================================

Controls starting, stopping, and monitoring the Freqtrade bot process.
Supports two launch modes:

    1. **Docker mode** (recommended):
       Uses ``docker compose`` to manage the container defined in
       ``docker-compose.yml``.  The container exposes the REST API on
       ``127.0.0.1:8080``.

    2. **Subprocess mode** (native / WSL):
       Spawns ``freqtrade trade …`` as a child process directly.
       Useful when Docker is not available or during development.

Architecture (GUI-safe):
────────────────────────
All blocking operations (docker compose, subprocess spawn, health polling)
run in background threads.  State transitions and results are delivered via
callbacks that should be ``pyqtSignal.emit()`` in a PyQt6 app.

State Machine::

    STOPPED  ──start()──►  STARTING  ──health OK──►  RUNNING
       ▲                                                 │
       └─────── STOPPING ◄──── stop() ──────────────────┘
                    │
                  ERROR  (if process crashes or health fails)

Usage from GUI::

    from core.exchange_manager import BotProcessManager, LaunchMode

    mgr = BotProcessManager()
    mgr.configure(
        launch_mode=LaunchMode.DOCKER,
        config_path="config/config_binance.json",
        strategy="TrendMLStrategy",
    )
    mgr.on_state_change = self._bot_state_signal.emit
    mgr.on_log          = self._bot_log_signal.emit

    mgr.start()       # non-blocking
    mgr.stop()        # non-blocking
    mgr.is_running    # True / False
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

from API.rest_client import FreqtradeClient, ApiResult

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────
HEALTH_POLL_INTERVAL = 3.0      # seconds between /ping checks during startup
HEALTH_POLL_TIMEOUT  = 120.0    # max seconds to wait for the bot to respond
PROCESS_STOP_TIMEOUT = 30.0     # seconds to wait for graceful shutdown
DEFAULT_CONTAINER    = "freqtrade_bot"


# ── Launch Mode ──────────────────────────────────────────────────────────────

class LaunchMode(Enum):
    """How the Freqtrade process is managed."""
    DOCKER     = auto()    # docker compose up/down
    SUBPROCESS = auto()    # direct freqtrade CLI as child process


# ── Process State ────────────────────────────────────────────────────────────

class BotState(Enum):
    """Observable bot process state.  GUI binds to this for status indicators."""
    STOPPED  = auto()
    STARTING = auto()
    RUNNING  = auto()
    STOPPING = auto()
    ERROR    = auto()


# ── Log Event ────────────────────────────────────────────────────────────────

@dataclass
class BotLogEvent:
    """A single log line emitted by the process manager."""
    level: str            # "info" | "warning" | "error"
    message: str
    source: str = ""      # "docker" | "subprocess" | "health"


# ── Launch Configuration ─────────────────────────────────────────────────────

@dataclass
class LaunchConfig:
    """
    Everything needed to start the Freqtrade bot process.
    Built from user selections in the Settings Panel.
    """
    launch_mode: LaunchMode = LaunchMode.DOCKER

    # Config file path (relative to project root)
    config_path: str = "config/config_binance.json"

    # Strategy & model
    strategy: str = "TrendMLStrategy"
    freqai_model: str = "LightGBMRegressor"

    # Docker-specific
    compose_file: str = "docker-compose.yml"
    container_name: str = DEFAULT_CONTAINER

    # Subprocess-specific
    freqtrade_cmd: str = "freqtrade"      # executable name or full path
    userdir: str = "user_data"
    logfile: str = "user_data/logs/freqtrade.log"
    db_url: str = "sqlite:///user_data/tradesv3.sqlite"

    # API connection (for health polling after launch)
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    api_username: str = ""
    api_password: str = ""

    @classmethod
    def from_config(cls, config_path: str | Path) -> "LaunchConfig":
        """
        Populate a LaunchConfig from a Freqtrade JSON config file.
        Reads ``api_server`` block for host/port/credentials.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            cfg: dict = json.load(f)

        api_cfg = cfg.get("api_server", {})
        raw_host = api_cfg.get("listen_ip_address", "127.0.0.1")
        host = "127.0.0.1" if raw_host in ("0.0.0.0", "::", "") else raw_host
        return cls(
            config_path=str(path),
            api_host=host,
            api_port=api_cfg.get("listen_port", 8080),
            api_username=api_cfg.get("username", ""),
            api_password=api_cfg.get("password", ""),
        )


# ── BotProcessManager ───────────────────────────────────────────────────────

class BotProcessManager:
    """
    Manages the Freqtrade bot process lifecycle (start / stop / monitor).

    All heavy operations run in background threads.  Callbacks
    (``on_state_change``, ``on_log``) fire from worker threads and should
    be ``pyqtSignal.emit()`` in a PyQt6 context.
    """

    def __init__(self) -> None:
        # ── Configuration ─────────────────────────────────────────────────
        self._config = LaunchConfig()

        # ── State (observable) ────────────────────────────────────────────
        self.state: BotState = BotState.STOPPED

        # ── Callbacks ─────────────────────────────────────────────────────
        self.on_state_change: Optional[Callable[[BotState], None]] = None
        self.on_log: Optional[Callable[[BotLogEvent], None]] = None

        # ── Internal ──────────────────────────────────────────────────────
        self._subprocess: Optional[subprocess.Popen] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._rest_client: Optional[FreqtradeClient] = None

        # ── Project root (for docker compose working dir) ─────────────────
        self._project_root = Path(__file__).resolve().parent.parent

        logger.info("BotProcessManager initialized (stopped)")

    # ══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════

    def configure(
        self,
        launch_mode: LaunchMode | None = None,
        config_path: str | None = None,
        strategy: str | None = None,
        freqai_model: str | None = None,
        compose_file: str | None = None,
        container_name: str | None = None,
        freqtrade_cmd: str | None = None,
        api_host: str | None = None,
        api_port: int | None = None,
        api_username: str | None = None,
        api_password: str | None = None,
    ) -> None:
        """Update launch configuration.  Only provided fields are changed."""
        if launch_mode is not None:
            self._config.launch_mode = launch_mode
        if config_path is not None:
            self._config.config_path = config_path
        if strategy is not None:
            self._config.strategy = strategy
        if freqai_model is not None:
            self._config.freqai_model = freqai_model
        if compose_file is not None:
            self._config.compose_file = compose_file
        if container_name is not None:
            self._config.container_name = container_name
        if freqtrade_cmd is not None:
            self._config.freqtrade_cmd = freqtrade_cmd
        if api_host is not None:
            normalized = "127.0.0.1" if api_host in ("0.0.0.0", "::", "") else api_host
            self._config.api_host = normalized
        if api_port is not None:
            self._config.api_port = api_port
        if api_username is not None:
            self._config.api_username = api_username
        if api_password is not None:
            self._config.api_password = api_password

        # Reset cached REST client so it's rebuilt with updated credentials
        if any(p is not None for p in (api_host, api_port, api_username, api_password)):
            self._rest_client = None

        logger.info("BotProcessManager configured: mode=%s strategy=%s",
                     self._config.launch_mode.name, self._config.strategy)

    def configure_from_config(self, config_path: str | Path) -> None:
        """Load API credentials from a Freqtrade config JSON file."""
        self._config = LaunchConfig.from_config(config_path)
        logger.info("Configured from %s", config_path)

    @property
    def config(self) -> LaunchConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self.state == BotState.RUNNING

    @property
    def is_busy(self) -> bool:
        """True when starting or stopping (GUI should disable controls)."""
        return self.state in (BotState.STARTING, BotState.STOPPING)

    # ══════════════════════════════════════════════════════════════════════
    # START / STOP (Public API — non-blocking)
    # ══════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """
        Start the Freqtrade bot process in the background.

        Non-blocking — spawns a worker thread that:
          1. Launches the process (Docker or subprocess)
          2. Polls ``/api/v1/ping`` until the API responds
          3. Transitions state to RUNNING (or ERROR on timeout)
        """
        if self.state in (BotState.STARTING, BotState.RUNNING):
            self._emit_log("warning", "Bot is already running or starting.")
            return
        if self.state == BotState.STOPPING:
            self._emit_log("warning", "Bot is shutting down — wait before restarting.")
            return

        self._stop_event.clear()
        self._set_state(BotState.STARTING)

        self._worker_thread = threading.Thread(
            target=self._start_worker,
            daemon=True,
            name="BotStart",
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """
        Stop the Freqtrade bot process gracefully.

        Non-blocking — spawns a worker thread that:
          1. Sends stop via REST API (graceful)
          2. If that fails, stops the Docker container / kills the subprocess
          3. Transitions state to STOPPED
        """
        if self.state in (BotState.STOPPED, BotState.STOPPING):
            self._emit_log("warning", "Bot is already stopped or stopping.")
            return

        self._stop_event.set()
        self._set_state(BotState.STOPPING)

        self._worker_thread = threading.Thread(
            target=self._stop_worker,
            daemon=True,
            name="BotStop",
        )
        self._worker_thread.start()

    def restart(self) -> None:
        """Stop then start.  Non-blocking."""
        def _restart_worker() -> None:
            self._stop_worker()
            time.sleep(2.0)       # brief pause between stop and start
            self._stop_event.clear()
            self._start_worker()

        if self.state == BotState.STOPPING:
            self._emit_log("warning", "Already stopping — cannot restart now.")
            return

        self._stop_event.set()
        self._set_state(BotState.STOPPING)

        self._worker_thread = threading.Thread(
            target=_restart_worker,
            daemon=True,
            name="BotRestart",
        )
        self._worker_thread.start()

    def check_health(self) -> ApiResult:
        """
        Synchronous health check via ``/api/v1/ping``.
        Returns the ApiResult — the GUI can call this from a QTimer
        to keep the dashboard status indicator up to date.
        """
        client = self._get_rest_client()
        return client.ping()

    def check_health_async(
        self,
        on_done: Optional[Callable[[ApiResult], None]] = None,
    ) -> None:
        """Non-blocking health check."""
        def _worker() -> None:
            result = self.check_health()
            if on_done:
                on_done(result)

        threading.Thread(target=_worker, daemon=True, name="HealthCheck").start()

    # ══════════════════════════════════════════════════════════════════════
    # DOCKER MODE
    # ══════════════════════════════════════════════════════════════════════

    def _docker_start(self) -> bool:
        """
        Run ``docker compose up -d``.  Returns True on success.
        """
        compose_path = self._project_root / self._config.compose_file
        if not compose_path.exists():
            self._emit_log("error", f"docker-compose file not found: {compose_path}")
            return False

        docker_cmd = self._find_docker_compose()
        if docker_cmd is None:
            self._emit_log("error",
                           "Docker not found. Install Docker Desktop or use subprocess mode.")
            return False

        # Pass user-selected config/strategy/model as env vars
        # so docker-compose.yml ${VAR:-default} substitution picks them up
        env = os.environ.copy()
        env["CONFIG_FILE"] = Path(self._config.config_path).name
        env["STRATEGY"] = self._config.strategy
        env["FREQAI_MODEL"] = self._config.freqai_model or "LightGBMRegressor"

        cmd = [*docker_cmd, "-f", str(compose_path), "up", "-d"]
        self._emit_log("info", f"Starting Docker: {' '.join(cmd)}")
        self._emit_log("info",
                       f"Config={env['CONFIG_FILE']}  Strategy={env['STRATEGY']}  "
                       f"Model={env['FREQAI_MODEL']}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            self._emit_log("error", "Docker compose up timed out after 60s.")
            return False
        except FileNotFoundError:
            self._emit_log("error", "Docker executable not found on PATH.")
            return False

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            self._emit_log("error", f"Docker compose failed: {stderr}")
            return False

        self._emit_log("info", "Docker container started successfully.")
        return True

    def _docker_stop(self) -> bool:
        """Run ``docker compose down``.  Returns True on success."""
        compose_path = self._project_root / self._config.compose_file

        docker_cmd = self._find_docker_compose()
        if docker_cmd is None:
            self._emit_log("error", "Docker not found — cannot stop container.")
            return False

        cmd = [*docker_cmd, "-f", str(compose_path), "down"]
        self._emit_log("info", f"Stopping Docker: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=PROCESS_STOP_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            self._emit_log("error", "Docker compose down timed out.")
            return False

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            self._emit_log("warning", f"Docker compose down warning: {stderr}")

        self._emit_log("info", "Docker container stopped.")
        return True

    def _docker_is_container_running(self) -> bool:
        """Check if the container is currently running."""
        docker_cmd = self._find_docker_compose()
        if docker_cmd is None:
            return False

        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}",
                 self._config.container_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip().lower() == "true"
        except Exception:
            return False

    @staticmethod
    def _find_docker_compose() -> list[str] | None:
        """
        Find the docker compose command.
        Modern Docker uses ``docker compose`` (plugin);
        older installs use ``docker-compose`` (standalone).
        """
        # Try modern "docker compose" first
        if shutil.which("docker"):
            try:
                result = subprocess.run(
                    ["docker", "compose", "version"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    return ["docker", "compose"]
            except Exception:
                pass

        # Fall back to standalone docker-compose
        if shutil.which("docker-compose"):
            return ["docker-compose"]

        return None

    # ══════════════════════════════════════════════════════════════════════
    # SUBPROCESS MODE
    # ══════════════════════════════════════════════════════════════════════

    def _subprocess_start(self) -> bool:
        """
        Spawn ``freqtrade trade`` as a child process.  Returns True on success.
        """
        cmd = self._build_freqtrade_command()
        self._emit_log("info", f"Starting subprocess: {' '.join(cmd)}")

        try:
            self._subprocess = subprocess.Popen(
                cmd,
                cwd=str(self._project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            self._emit_log("error",
                           f"'{self._config.freqtrade_cmd}' not found. "
                           "Is Freqtrade installed and on PATH?")
            return False
        except Exception as exc:
            self._emit_log("error", f"Failed to start subprocess: {exc}")
            return False

        self._emit_log("info", f"Subprocess started (PID {self._subprocess.pid})")

        # Start a daemon thread to stream stdout → on_log
        threading.Thread(
            target=self._stream_subprocess_output,
            daemon=True,
            name="SubprocOutput",
        ).start()

        return True

    def _subprocess_stop(self) -> bool:
        """Terminate the child process gracefully."""
        if self._subprocess is None:
            return True

        pid = self._subprocess.pid
        self._emit_log("info", f"Stopping subprocess (PID {pid})…")

        # Try graceful termination first
        self._subprocess.terminate()
        try:
            self._subprocess.wait(timeout=PROCESS_STOP_TIMEOUT)
            self._emit_log("info", f"Subprocess (PID {pid}) terminated gracefully.")
        except subprocess.TimeoutExpired:
            self._emit_log("warning", f"Subprocess (PID {pid}) didn't stop — killing.")
            self._subprocess.kill()
            self._subprocess.wait(timeout=5)

        self._subprocess = None
        return True

    def _subprocess_is_running(self) -> bool:
        """Check if the child process is still alive."""
        if self._subprocess is None:
            return False
        return self._subprocess.poll() is None

    def _stream_subprocess_output(self) -> None:
        """Read subprocess stdout line-by-line and emit log events."""
        proc = self._subprocess
        if proc is None or proc.stdout is None:
            return

        try:
            for line in proc.stdout:
                stripped = line.rstrip("\n\r")
                if stripped:
                    self._emit_log("info", stripped, source="subprocess")
                # Stop streaming if process ended
                if proc.poll() is not None:
                    break
        except Exception:
            pass

    def _build_freqtrade_command(self) -> list[str]:
        """Assemble the ``freqtrade trade …`` CLI arguments."""
        cmd = [
            self._config.freqtrade_cmd,
            "trade",
            "--config", self._config.config_path,
            "--strategy", self._config.strategy,
            "--userdir", self._config.userdir,
        ]
        if self._config.freqai_model:
            cmd.extend(["--freqaimodel", self._config.freqai_model])
        if self._config.logfile:
            cmd.extend(["--logfile", self._config.logfile])
        if self._config.db_url:
            cmd.extend(["--db-url", self._config.db_url])
        return cmd

    # ══════════════════════════════════════════════════════════════════════
    # WORKER THREADS
    # ══════════════════════════════════════════════════════════════════════

    def _start_worker(self) -> None:
        """Background thread: launch process → poll health → set RUNNING."""
        self._emit_log("info", f"Starting bot ({self._config.launch_mode.name} mode)…")

        # ── 1. Launch the process ────────────────────────────────────────
        if self._config.launch_mode == LaunchMode.DOCKER:
            ok = self._docker_start()
        else:
            ok = self._subprocess_start()

        if not ok:
            self._set_state(BotState.ERROR)
            return

        # ── 2. Wait for the REST API to become reachable ─────────────────
        self._emit_log("info", "Waiting for bot API to respond…")
        if self._poll_health():
            self._set_state(BotState.RUNNING)
            self._emit_log("info", "Bot is RUNNING and API is reachable.")
        else:
            self._set_state(BotState.ERROR)
            self._emit_log("error",
                           f"Bot API did not respond within {HEALTH_POLL_TIMEOUT}s. "
                           "Check bot logs for errors.")

    def _stop_worker(self) -> None:
        """Background thread: graceful stop → verify stopped → set STOPPED."""
        self._emit_log("info", "Stopping bot…")

        # ── 1. Try graceful stop via REST API first ──────────────────────
        try:
            client = self._get_rest_client()
            result = client.stop()
            if result.success:
                self._emit_log("info", "Bot stop command sent via API.")
                # Give it a moment to wind down
                time.sleep(3.0)
        except Exception:
            pass

        # ── 2. Stop the process / container ──────────────────────────────
        if self._config.launch_mode == LaunchMode.DOCKER:
            self._docker_stop()
        else:
            self._subprocess_stop()

        self._set_state(BotState.STOPPED)
        self._emit_log("info", "Bot stopped.")

    def _poll_health(self) -> bool:
        """
        Poll ``/api/v1/ping`` until the bot responds or timeout expires.
        Returns True if the bot became reachable.
        """
        client = self._get_rest_client()
        start_time = time.monotonic()
        attempt = 0

        while (time.monotonic() - start_time) < HEALTH_POLL_TIMEOUT:
            if self._stop_event.is_set():
                return False

            attempt += 1
            result = client.ping()
            if result.success:
                self._emit_log("info",
                               f"Bot API responded after {attempt} attempt(s) "
                               f"({time.monotonic() - start_time:.1f}s).")
                return True

            time.sleep(HEALTH_POLL_INTERVAL)

        return False

    # ══════════════════════════════════════════════════════════════════════
    # STATUS QUERIES
    # ══════════════════════════════════════════════════════════════════════

    def is_docker_available(self) -> bool:
        """Check if Docker is installed and accessible."""
        return self._find_docker_compose() is not None

    def is_freqtrade_available(self) -> bool:
        """Check if the freqtrade CLI is on PATH."""
        return shutil.which(self._config.freqtrade_cmd) is not None

    def get_available_modes(self) -> list[LaunchMode]:
        """Return which launch modes are available on this system."""
        modes = []
        if self.is_docker_available():
            modes.append(LaunchMode.DOCKER)
        if self.is_freqtrade_available():
            modes.append(LaunchMode.SUBPROCESS)
        return modes

    def get_process_info(self) -> dict[str, Any]:
        """
        Return diagnostic info about the current process.
        Useful for the Settings Panel status display.
        """
        info: dict[str, Any] = {
            "state": self.state.name,
            "launch_mode": self._config.launch_mode.name,
            "config": self._config.config_path,
            "strategy": self._config.strategy,
            "model": self._config.freqai_model,
            "api_url": f"http://{self._config.api_host}:{self._config.api_port}",
        }

        if self._config.launch_mode == LaunchMode.DOCKER:
            info["container"] = self._config.container_name
            info["container_running"] = self._docker_is_container_running()
        else:
            info["pid"] = self._subprocess.pid if self._subprocess else None
            info["process_alive"] = self._subprocess_is_running()

        return info

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _get_rest_client(self) -> FreqtradeClient:
        """Lazily create a FreqtradeClient configured for health checks."""
        if self._rest_client is None:
            self._rest_client = FreqtradeClient()
            self._rest_client.configure(
                host=self._config.api_host,
                port=self._config.api_port,
                username=self._config.api_username,
                password=self._config.api_password,
            )
        return self._rest_client

    def _set_state(self, new_state: BotState) -> None:
        if self.state != new_state:
            old = self.state
            self.state = new_state
            logger.debug("BotState: %s → %s", old.name, new_state.name)
            if self.on_state_change is not None:
                try:
                    self.on_state_change(new_state)
                except Exception:
                    pass

    def _emit_log(self, level: str, message: str, source: str = "") -> None:
        """Emit a log event to both Python logging and the GUI callback."""
        getattr(logger, level, logger.info)(message)
        if self.on_log is not None:
            try:
                self.on_log(BotLogEvent(
                    level=level,
                    message=message,
                    source=source or self._config.launch_mode.name.lower(),
                ))
            except Exception:
                pass

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Clean up resources.  Does NOT stop the bot process."""
        if self._rest_client:
            self._rest_client.close()
            self._rest_client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
