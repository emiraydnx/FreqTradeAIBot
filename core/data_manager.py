"""
data_manager.py — FreqTradeAIBot Veri Katmanı
=============================================

TASARIM PRENSİPLERİ (GUI Entegrasyonu İçin):
─────────────────────────────────────────────
1. Non-blocking: Tüm ağ çağrıları ayrı thread'de çalışır, GUI donar.
2. Progress callbacks: Yükleme ilerlemesi GUI'ye anlık iletilir.
3. Config-driven: Borsa, pair listesi config'den okunur, GUI değiştirebilir.
4. Kullanıcı dostu hatalar: Her teknik hata, UI'da gösterilebilir mesaja sahip.
5. Observable: DataManager durumu (yükleniyor, hazır, hata) GUI'nin dinleyebileceği
   bir state ile takip edilir.

GUI KULLANIM ÖRNEĞİ (ilerideki gui/dashboard.py için):
───────────────────────────────────────────────────────
    dm = DataManager.from_config("config/config_binance.json")

    def on_progress(event: ProgressEvent):
        self.progress_bar.setValue(event.percent)
        self.status_label.setText(event.message)

    def on_done(result: DataResult):
        if result.success:
            self.chart.load(result.dataframe)
        else:
            QMessageBox.warning(self, "Veri Hatası", result.user_message)

    dm.load_async("BTC/USDT", "1h", days=365,
                  on_progress=on_progress,
                  on_done=on_done)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import ccxt
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────
OHLCV_COLUMNS       = ["date", "open", "high", "low", "close", "volume"]
PAIR_SEP            = "_"
CCXT_PAGE_LIMIT     = 1000   # Binance max bar/istek
DEFAULT_TIMEOUT_MS  = 30_000

SUPPORTED_TIMEFRAMES: dict[str, int] = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# Kullanıcıya gösterilecek timeframe etiketleri (GUI dropdown için)
TIMEFRAME_LABELS: dict[str, str] = {
    "1m":  "1 Dakika",
    "5m":  "5 Dakika",
    "15m": "15 Dakika",
    "30m": "30 Dakika",
    "1h":  "1 Saat",
    "4h":  "4 Saat",
    "1d":  "1 Gün",
}


# ── Durum & Sonuç Nesneleri ───────────────────────────────────────────────────

class LoadState(Enum):
    """DataManager'ın şu anki durumu. GUI bu değeri okuyarak UI'ı günceller."""
    IDLE       = auto()   # Bekliyor
    LOADING    = auto()   # Veri çekiliyor
    READY      = auto()   # Veri hazır
    ERROR      = auto()   # Hata oluştu


@dataclass
class ProgressEvent:
    """
    Yükleme sırasında GUI'ye gönderilen ilerleme bilgisi.
    GUI'de progress bar ve status label için kullanılır.
    """
    percent: int           # 0-100
    message: str           # "BTC/USDT verisi çekiliyor: %43"
    bars_loaded: int = 0   # Şimdiye kadar yüklenen bar sayısı
    bars_total: int = 0    # Tahmini toplam bar sayısı


@dataclass
class DataResult:
    """
    load_async() tamamlandığında on_done callback'ine iletilen sonuç.
    GUI bu nesneyi kontrol ederek ya grafiği doldurur ya hata gösterir.
    """
    success: bool
    pair: str
    timeframe: str

    # Başarı durumunda dolu
    dataframe: Optional[pd.DataFrame] = None

    # Hata durumunda dolu
    error_code: str = ""          # Teknik kod (loglama için)
    user_message: str = ""        # Kullanıcıya gösterilecek Türkçe mesaj

    # Meta bilgi (her zaman dolu)
    source: str = ""              # "cache" | "ccxt"
    duration_sec: float = 0.0    # Kaç saniyede yüklendi
    bars: int = 0                 # Toplam bar sayısı


@dataclass
class ExchangeConfig:
    """
    Bir borsanın bağlantı konfigürasyonu.
    GUI'deki "Borsa Ayarları" ekranından doldurulur.
    """
    exchange_id: str          = "binance"
    api_key: str              = ""          # .env'den okunmalı, burada boş kalır
    api_secret: str           = ""          # .env'den okunmalı, burada boş kalır
    timeout_ms: int           = DEFAULT_TIMEOUT_MS
    sandbox_mode: bool        = False       # True -> testnet kullan
    default_pairs: list[str]  = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    default_timeframe: str    = "1h"


# ── Özel Hatalar ─────────────────────────────────────────────────────────────

class DataFetchError(Exception):
    """
    Veri çekimi başarısız olduğunda fırlatılır.
    user_message: GUI'de doğrudan gösterilebilecek Türkçe açıklama.
    """
    def __init__(self, technical_msg: str, user_message: str = ""):
        super().__init__(technical_msg)
        self.user_message = user_message or "Veri yüklenirken bir hata oluştu. Bağlantınızı kontrol edin."


class UnsupportedTimeframeError(ValueError):
    pass


# ── DataManager ───────────────────────────────────────────────────────────────

class DataManager:
    """
    FreqTradeAIBot için tek veri erişim noktası.

    Doğrudan başlatma:
        dm = DataManager(userdata_path="user_data/data", exchange_id="binance")

    Config dosyasından başlatma (önerilen GUI kullanımı):
        dm = DataManager.from_config("config/config_binance.json")

    Senkron kullanım (script / backtesting):
        df = dm.get_data("BTC/USDT", "1h", days=365)

    Asenkron kullanım (GUI):
        dm.load_async("BTC/USDT", "1h", days=365,
                      on_progress=my_progress_fn,
                      on_done=my_done_fn)
    """

    def __init__(
        self,
        userdata_path: Optional[str | Path] = None,
        exchange_config: Optional[ExchangeConfig] = None,
    ) -> None:
        self.config = exchange_config or ExchangeConfig()

        # user_data/data dizini
        if userdata_path is None:
            repo_root = Path(__file__).resolve().parent
            userdata_path = repo_root / "user_data" / "data"
        self.data_dir = Path(userdata_path)

        # GUI'nin dinleyebileceği durum
        self.state: LoadState = LoadState.IDLE

        # Aktif CCXT bağlantısı (lazy init, thread-safe lock ile)
        self._exchange: Optional[ccxt.Exchange] = None
        self._exchange_lock = threading.Lock()

        # Çalışan thread referansı
        self._active_thread: Optional[threading.Thread] = None

        logger.info(
            "DataManager hazır | data_dir=%s | exchange=%s",
            self.data_dir,
            self.config.exchange_id,
        )

    # ── Factory Methods ───────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DataManager":
        """
        Freqtrade config JSON dosyasından DataManager oluşturur.
        GUI'nin config yükleme ekranı bu metodu kullanır.

        Args:
            config_path: config/config_binance.json gibi bir yol

        Returns:
            DataManager: Konfigüre edilmiş instance
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config dosyası bulunamadı: {path}")

        with path.open("r", encoding="utf-8") as f:
            cfg: dict = json.load(f)

        exchange_name = cfg.get("exchange", {}).get("name", "binance")
        pairs = cfg.get("exchange", {}).get("pair_whitelist", ["BTC/USDT"])
        timeframe = cfg.get("timeframe", "1h")
        userdata = cfg.get("user_data_dir", "user_data")

        exchange_config = ExchangeConfig(
            exchange_id=exchange_name,
            default_pairs=pairs,
            default_timeframe=timeframe,
        )

        data_path = Path(userdata) / "data"
        instance = cls(userdata_path=data_path, exchange_config=exchange_config)
        logger.info("Config'den yüklendi | %s | %d pair", path.name, len(pairs))
        return instance

    # ── Public API: Senkron (Script / Backtesting) ────────────────────────────

    def get_data(
        self,
        pair: str,
        timeframe: str = "1h",
        days: int = 365,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Veri çeker ve DataFrame döner. GUI thread'inde ÇAĞIRMA — donar.
        Script, backtesting ve test ortamları için kullanın.

        Raises:
            UnsupportedTimeframeError
            DataFetchError
        """
        self._validate_timeframe(timeframe)
        pair = self._normalize_pair(pair)

        # 1. Freqtrade JSON cache
        if not force_download:
            df = self._load_from_cache(pair, timeframe)
            if df is not None:
                df = self._filter_by_days(df, days)
                logger.info("Cache | %s | %d bar", pair, len(df))
                return df

        # 2. CCXT fallback
        logger.info("CCXT çekimi | %s %s %d gün", pair, timeframe, days)
        return self._fetch_from_ccxt(pair, timeframe, days)

    def get_multiple(
        self,
        pairs: list[str],
        timeframe: str = "1h",
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Birden fazla pair için senkron veri çekimi."""
        results: dict[str, pd.DataFrame] = {}
        for pair in pairs:
            try:
                results[pair] = self.get_data(pair, timeframe, days)
            except DataFetchError as exc:
                logger.error("Pair atlandı: %s — %s", pair, exc)
        return results

    # ── Public API: Asenkron (GUI) ────────────────────────────────────────────

    def load_async(
        self,
        pair: str,
        timeframe: str = "1h",
        days: int = 365,
        force_download: bool = False,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        on_done: Optional[Callable[[DataResult], None]] = None,
    ) -> None:
        """
        Veriyi arka planda thread'de çeker. GUI thread'i donmaz.

        Args:
            pair:            Trading pair. Örn: "BTC/USDT"
            timeframe:       OHLCV zaman dilimi
            days:            Kaç günlük veri
            force_download:  Cache'i atla
            on_progress:     İlerleme callback'i — her sayfa sonrası çağrılır
            on_done:         Tamamlanma callback'i — başarı veya hata

        NOT: on_progress ve on_done GUI thread'inden değil,
             worker thread'inden çağrılır.
             PyQt6 kullanıyorsanız signal/slot üzerinden bağlayın:
             dm.load_async(..., on_done=self.data_ready_signal.emit)
        """
        if self.state == LoadState.LOADING:
            logger.warning("Zaten bir yükleme devam ediyor.")
            return

        self.state = LoadState.LOADING

        thread = threading.Thread(
            target=self._load_worker,
            args=(pair, timeframe, days, force_download, on_progress, on_done),
            daemon=True,
            name=f"DataLoader-{pair.replace('/', '')}-{timeframe}",
        )
        self._active_thread = thread
        thread.start()

    def cancel_loading(self) -> None:
        """
        Devam eden asenkron yüklemeyi iptal eder.
        GUI'deki "İptal" butonu bu metodu çağırır.
        """
        with self._exchange_lock:
            if self._exchange:
                try:
                    self._exchange.close()
                except Exception:
                    pass
                self._exchange = None
        self.state = LoadState.IDLE
        logger.info("Veri yükleme iptal edildi.")

    # ── Cache Bilgileri (GUI için) ─────────────────────────────────────────────

    def list_cached_pairs(self, timeframe: str = "1h") -> list[str]:
        """
        Cache'de mevcut pairleri döner.
        GUI'deki "Pair Seç" dropdown'ını doldurmak için kullanılır.
        """
        if not self.data_dir.exists():
            return []

        found = set()
        for subdir in [self.data_dir, self.data_dir / self.config.exchange_id]:
            if not subdir.exists():
                continue
            for p in subdir.glob(f"*-{timeframe}.json"):
                stem = p.stem.replace(f"-{timeframe}", "")
                pair = stem.replace(PAIR_SEP, "/", 1)
                found.add(pair)
        return sorted(found)

    def get_cache_info(self, pair: str, timeframe: str = "1h") -> dict:
        """
        Bir pair'in cache bilgisi.
        GUI'deki "Veri Durumu" paneli için kullanılır.

        Returns:
            {
                "exists": bool,
                "bars": int,
                "from": date | None,
                "to": date | None,
                "size_mb": float,
                "label": str   <- Kullanıcıya gösterilecek özet metin
            }
        """
        path = self._cache_path(pair, timeframe)
        if not path.exists():
            return {
                "exists": False,
                "label": "Cache bulunamadı — veri indirilmedi",
            }

        df = self._load_from_cache(pair, timeframe)
        size_mb = round(path.stat().st_size / 1_048_576, 2)
        bars = len(df) if df is not None else 0
        date_from = df.index[0].date() if df is not None else None
        date_to = df.index[-1].date() if df is not None else None

        return {
            "exists": True,
            "bars": bars,
            "from": date_from,
            "to": date_to,
            "size_mb": size_mb,
            "label": f"{bars} bar | {date_from} -> {date_to} | {size_mb} MB",
        }

    @staticmethod
    def get_available_timeframes() -> dict[str, str]:
        """
        GUI dropdown için timeframe seçenekleri.

        Returns:
            {"1h": "1 Saat", "4h": "4 Saat", ...}
        """
        return TIMEFRAME_LABELS.copy()

    @staticmethod
    def get_popular_pairs() -> list[str]:
        """
        GUI'de önerilen pair listesi (kullanıcı henüz config yüklemediyse).
        """
        return [
            "BTC/USDT", "ETH/USDT", "SOL/USDT",
            "BNB/USDT", "XRP/USDT", "ADA/USDT",
        ]

    # ── Worker (Thread İçinde Çalışır) ────────────────────────────────────────

    def _load_worker(
        self,
        pair: str,
        timeframe: str,
        days: int,
        force_download: bool,
        on_progress: Optional[Callable[[ProgressEvent], None]],
        on_done: Optional[Callable[[DataResult], None]],
    ) -> None:
        """Arka plan thread'inde çalışan yükleme mantığı."""
        start = datetime.now()
        pair = self._normalize_pair(pair)
        source = "cache"

        def _emit_progress(percent: int, message: str, bars_loaded: int = 0, bars_total: int = 0) -> None:
            if on_progress:
                on_progress(ProgressEvent(
                    percent=percent,
                    message=message,
                    bars_loaded=bars_loaded,
                    bars_total=bars_total,
                ))

        try:
            self._validate_timeframe(timeframe)
            _emit_progress(5, f"{pair} için cache kontrol ediliyor...")

            df = None
            if not force_download:
                df = self._load_from_cache(pair, timeframe)
                if df is not None:
                    df = self._filter_by_days(df, days)
                    _emit_progress(100, f"{pair} cache'den yüklendi | {len(df)} bar")

            if df is None:
                source = "ccxt"
                _emit_progress(10, f"{pair} borsadan indiriliyor...")
                df = self._fetch_from_ccxt(
                    pair, timeframe, days,
                    progress_callback=lambda p, msg, bl, bt: _emit_progress(
                        10 + int(p * 0.88), msg, bl, bt
                    ),
                )
                _emit_progress(100, f"{pair} indirildi | {len(df)} bar")

            duration = (datetime.now() - start).total_seconds()
            self.state = LoadState.READY

            result = DataResult(
                success=True,
                pair=pair,
                timeframe=timeframe,
                dataframe=df,
                source=source,
                duration_sec=round(duration, 2),
                bars=len(df),
            )

        except UnsupportedTimeframeError as exc:
            self.state = LoadState.ERROR
            result = DataResult(
                success=False,
                pair=pair,
                timeframe=timeframe,
                error_code="UNSUPPORTED_TIMEFRAME",
                user_message=f"'{timeframe}' geçerli bir zaman dilimi değil. Lütfen listeden seçin.",
            )
            logger.error("Timeframe hatası: %s", exc)

        except DataFetchError as exc:
            self.state = LoadState.ERROR
            result = DataResult(
                success=False,
                pair=pair,
                timeframe=timeframe,
                error_code="FETCH_FAILED",
                user_message=exc.user_message,
            )
            logger.error("Veri hatası: %s", exc)

        except Exception as exc:
            self.state = LoadState.ERROR
            result = DataResult(
                success=False,
                pair=pair,
                timeframe=timeframe,
                error_code="UNKNOWN",
                user_message="Beklenmedik bir hata oluştu. Lütfen tekrar deneyin.",
            )
            logger.exception("Beklenmedik hata: %s", exc)

        if on_done:
            on_done(result)

    # ── Freqtrade Cache ───────────────────────────────────────────────────────

    def _load_from_cache(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(pair, timeframe)
        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                raw: list = json.load(f)
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
            return self._clean_dataframe(df)
        except Exception as exc:
            logger.warning("Cache okunamadı: %s — %s", path, exc)
            return None

    def _cache_path(self, pair: str, timeframe: str) -> Path:
        filename = f"{self._pair_to_filename(pair)}-{timeframe}.json"
        exchange_path = self.data_dir / self.config.exchange_id / filename
        if exchange_path.exists():
            return exchange_path
        return self.data_dir / filename

    # ── CCXT ─────────────────────────────────────────────────────────────────

    def _fetch_from_ccxt(
        self,
        pair: str,
        timeframe: str,
        days: int,
        progress_callback: Optional[Callable] = None,
    ) -> pd.DataFrame:
        exchange = self._get_exchange()
        since_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        tf_ms = SUPPORTED_TIMEFRAMES[timeframe]
        estimated_bars = int((days * 86_400_000) / tf_ms)

        all_candles: list = []
        current_since = since_ms

        try:
            while True:
                candles = exchange.fetch_ohlcv(
                    pair,
                    timeframe=timeframe,
                    since=current_since,
                    limit=CCXT_PAGE_LIMIT,
                )

                if not candles:
                    break

                all_candles.extend(candles)
                loaded = len(all_candles)
                percent = min(100, int((loaded / max(estimated_bars, 1)) * 100))

                if progress_callback:
                    progress_callback(
                        percent / 100,
                        f"{pair} indiriliyor... {loaded:,} / ~{estimated_bars:,} bar",
                        loaded,
                        estimated_bars,
                    )

                if len(candles) < CCXT_PAGE_LIMIT:
                    break

                current_since = candles[-1][0] + tf_ms

        except ccxt.NetworkError as exc:
            raise DataFetchError(
                str(exc),
                user_message="İnternet bağlantısı kurulamadı. Ağ ayarlarınızı kontrol edin."
            ) from exc
        except ccxt.ExchangeError as exc:
            raise DataFetchError(
                str(exc),
                user_message=f"Borsa bağlantı hatası: {pair} için veri alınamadı. Pair adını kontrol edin."
            ) from exc

        if not all_candles:
            raise DataFetchError(
                f"Boş yanıt: {pair} {timeframe}",
                user_message=f"{pair} için veri bulunamadı. Pair adı doğru mu?"
            )

        df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
        return self._clean_dataframe(df)

    def _get_exchange(self) -> ccxt.Exchange:
        with self._exchange_lock:
            if self._exchange is None:
                exchange_class = getattr(ccxt, self.config.exchange_id, None)
                if exchange_class is None:
                    raise DataFetchError(
                        f"Bilinmeyen borsa: {self.config.exchange_id}",
                        user_message=f"'{self.config.exchange_id}' desteklenmiyor. Config dosyasını kontrol edin."
                    )
                self._exchange = exchange_class({
                    "timeout": self.config.timeout_ms,
                    "enableRateLimit": True,
                    "sandbox": self.config.sandbox_mode,
                })
        return self._exchange

    # ── DataFrame Temizliği ───────────────────────────────────────────────────

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        df = df.set_index("date")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df[(df["close"] > 0) & (df["open"] > 0)]
        df = df.dropna()
        return df

    @staticmethod
    def _filter_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        return df[df.index >= cutoff]

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        return pair.upper().strip()

    @staticmethod
    def _pair_to_filename(pair: str) -> str:
        return pair.replace("/", PAIR_SEP).upper()

    @staticmethod
    def _validate_timeframe(timeframe: str) -> None:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise UnsupportedTimeframeError(
                f"Geçersiz timeframe: '{timeframe}'. "
                f"Desteklenenler: {sorted(SUPPORTED_TIMEFRAMES)}"
            )


# ── CLI / Smoke Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("\n" + "="*55)
    print("  FreqTradeAIBot — DataManager Smoke Test")
    print("="*55)

    # Config'den başlat
    config_path = Path("config/config_binance.json")
    if config_path.exists():
        dm = DataManager.from_config(config_path)
        print(f"\n  Config yüklendi: {config_path}")
    else:
        dm = DataManager()
        print("\n  Config bulunamadı, varsayılan ayarlarla başlatıldı.")

    # Cache bilgisi
    print(f"\n  Cache'deki pairler (1h): {dm.list_cached_pairs('1h') or 'Bulunamadı'}")
    info = dm.get_cache_info("BTC/USDT", "1h")
    print(f"  BTC/USDT cache: {info.get('label', 'Yok')}")

    # Asenkron yükleme testi
    print("\n  Asenkron yükleme testi (BTC/USDT 1h)...")
    done_event = threading.Event()

    def on_progress(evt: ProgressEvent):
        bar = "#" * (evt.percent // 5) + "." * (20 - evt.percent // 5)
        print(f"  [{bar}] %{evt.percent:3d} — {evt.message}", end="\r")

    def on_done(result: DataResult):
        print()
        if result.success:
            print(f"\n  Yuklendi | {result.bars} bar | kaynak={result.source} | sure={result.duration_sec}s")
            print(f"  Ilk: {result.dataframe.index[0]}  |  Son: {result.dataframe.index[-1]}")
            print(f"  NaN: {result.dataframe.isnull().sum().sum()}")
        else:
            print(f"\n  Hata [{result.error_code}]: {result.user_message}")
        done_event.set()

    dm.load_async("BTC/USDT", "1h", days=365, on_progress=on_progress, on_done=on_done)
    done_event.wait(timeout=120)

    print("\n" + "="*55)
    print("  Test tamamlandi.")
    print("="*55 + "\n")