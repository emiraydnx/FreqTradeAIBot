"""
features/technical.py — Teknik İndikatör Katmanı
=================================================

TASARIM PRENSİPLERİ (GUI Entegrasyonu İçin):
─────────────────────────────────────────────
1. Her indikatör grubu bağımsız fonksiyon → GUI "hangi indikatörler aktif?"
   sorusunu sorabilir, kullanıcı istediğini açıp kapatabilir.
2. IndicatorConfig dataclass → GUI'deki "İndikatör Ayarları" paneli bu
   nesneyi doldurur, period'ları slider ile değiştirir.
3. IndicatorResult → hesaplama sonrası GUI'ye hem DataFrame hem meta bilgi
   (kaç kolon eklendi, warmup kaç bar) döner.
4. Tüm hatalar kullanıcı dostu mesajlarla sarılır.
5. Her fonksiyon bağımsız test edilebilir (unit test dostu).

GUI KULLANIM ÖRNEĞİ (ilerideki gui/settings_panel.py için):
────────────────────────────────────────────────────────────
    # Kullanıcı ayar panelinden RSI period'unu 14'ten 21'e çekti
    config = IndicatorConfig(rsi_period=21, ema_periods=[20, 50, 200])

    result = TechnicalIndicators.compute(df, config)

    if result.success:
        self.chart.update_dataframe(result.dataframe)
        self.info_label.setText(result.summary)
    else:
        QMessageBox.warning(self, "İndikatör Hatası", result.user_message)

EKLENEN İNDİKATÖRLER:
──────────────────────
    Momentum   : rsi, stoch_rsi_k, stoch_rsi_d
    Trend      : ema_20/50/200, ema_cross_20_50, ema_cross_50_200,
                 price_vs_ema200, macd, macd_signal, macd_hist
    Volatilite : atr, bb_upper, bb_mid, bb_lower, bb_width, bb_percent
    Hacim      : volume_sma, volume_ratio, obv, obv_sma
    Mum        : candle_body, candle_upper_wick, candle_lower_wick, is_bullish
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# ── Sabitler ─────────────────────────────────────────────────────────────────

# Minimum bar sayısı — warmup period'u geçebilmek için
# EMA(200) en uzun warmup'ı gerektiriyor
MIN_BARS_REQUIRED = 250

# GUI'de gösterilecek indikatör kategori etiketleri
INDICATOR_CATEGORIES = {
    "momentum":   "Momentum İndikatörleri",
    "trend":      "Trend İndikatörleri",
    "volatility": "Volatilite İndikatörleri",
    "volume":     "Hacim İndikatörleri",
    "candle":     "Mum Örüntüsü Özellikleri",
}

# Her indikatörün hangi kolonları ürettiği — GUI "aktif kolonlar" listesi için
INDICATOR_COLUMNS: dict[str, list[str]] = {
    "rsi":        ["rsi"],
    "stoch_rsi":  ["stoch_rsi_k", "stoch_rsi_d"],
    "ema":        ["ema_20", "ema_50", "ema_200",
                   "ema_cross_20_50", "ema_cross_50_200", "price_vs_ema200"],
    "macd":       ["macd", "macd_signal", "macd_hist"],
    "atr":        ["atr"],
    "bbands":     ["bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_percent"],
    "volume":     ["volume_sma", "volume_ratio", "obv", "obv_sma"],
    "candle":     ["candle_body", "candle_upper_wick",
                   "candle_lower_wick", "is_bullish"],
}


# ── Config & Sonuç Nesneleri ──────────────────────────────────────────────────

@dataclass
class IndicatorConfig:
    """
    Hangi indikatörlerin hesaplanacağını ve period'larını tanımlar.
    GUI'deki "İndikatör Ayarları" paneli bu nesneyi oluşturur.

    Tüm 'enabled_*' flag'leri True → her şeyi hesapla (varsayılan).
    Kullanıcı GUI'den bir indikatörü kapatırsa ilgili flag False yapılır.
    """
    # Period ayarları (GUI slider'larıyla değiştirilir)
    rsi_period: int        = 14
    stoch_rsi_period: int  = 14
    ema_periods: list[int] = field(default_factory=lambda: [20, 50, 200])
    macd_fast: int         = 12
    macd_slow: int         = 26
    macd_signal: int       = 9
    atr_period: int        = 14
    bb_period: int         = 20
    bb_std: float          = 2.0
    volume_sma_period: int = 20
    obv_sma_period: int    = 20

    # Aktif/pasif toggle'lar (GUI checkbox'ları)
    enabled_rsi: bool       = True
    enabled_stoch_rsi: bool = True
    enabled_ema: bool       = True
    enabled_macd: bool      = True
    enabled_atr: bool       = True
    enabled_bbands: bool    = True
    enabled_volume: bool    = True
    enabled_candle: bool    = True

    def active_indicators(self) -> list[str]:
        """GUI'nin 'aktif indikatörler' listesi için."""
        active = []
        if self.enabled_rsi:       active.append("rsi")
        if self.enabled_stoch_rsi: active.append("stoch_rsi")
        if self.enabled_ema:       active.append("ema")
        if self.enabled_macd:      active.append("macd")
        if self.enabled_atr:       active.append("atr")
        if self.enabled_bbands:    active.append("bbands")
        if self.enabled_volume:    active.append("volume")
        if self.enabled_candle:    active.append("candle")
        return active

    def expected_columns(self) -> list[str]:
        """Bu config ile üretilecek tüm kolon adları."""
        cols = []
        for ind in self.active_indicators():
            cols.extend(INDICATOR_COLUMNS.get(ind, []))
        return cols


@dataclass
class IndicatorResult:
    """
    TechnicalIndicators.compute() metodunun döndürdüğü sonuç.
    GUI bu nesneyi kontrol ederek grafiği veya hata mesajını gösterir.
    """
    success: bool
    dataframe: Optional[pd.DataFrame] = None

    # Başarı durumunda
    added_columns: list[str]  = field(default_factory=list)
    warmup_bars: int          = 0      # Bu kadar bar warmup için harcandı
    bars_remaining: int       = 0      # Kullanılabilir bar sayısı
    summary: str              = ""     # GUI info label için özet

    # Hata durumunda
    error_code: str           = ""
    user_message: str         = ""


# ── Özel Hatalar ─────────────────────────────────────────────────────────────

class InsufficientDataError(ValueError):
    """Yeterli bar yoksa fırlatılır."""
    def __init__(self, have: int, need: int):
        self.have = have
        self.need = need
        self.user_message = (
            f"Yeterli veri yok: {have} bar mevcut, "
            f"en az {need} bar gerekli. "
            f"Daha fazla geçmiş veri indirin."
        )
        super().__init__(self.user_message)


class IndicatorComputeError(RuntimeError):
    """İndikatör hesaplama hatası."""
    def __init__(self, indicator: str, detail: str):
        self.indicator = indicator
        self.user_message = (
            f"'{indicator}' hesaplanırken hata oluştu. "
            f"Veri formatını kontrol edin."
        )
        super().__init__(f"{indicator}: {detail}")


# ── TechnicalIndicators ───────────────────────────────────────────────────────

class TechnicalIndicators:
    """
    Teknik indikatörleri hesaplayan ana sınıf.

    Kullanım — tam hesaplama (GUI ve pipeline için):
        result = TechnicalIndicators.compute(df, config)

    Kullanım — tek indikatör (grafik overlay için):
        df = TechnicalIndicators.add_rsi(df, period=14)
        df = TechnicalIndicators.add_ema(df, periods=[20, 50, 200])
    """

    # ── Ana Giriş Noktası ─────────────────────────────────────────────────────

    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        config: Optional[IndicatorConfig] = None,
        drop_warmup: bool = True,
    ) -> IndicatorResult:
        """
        Config'e göre tüm indikatörleri hesaplar.

        Args:
            df:          OHLCV DataFrame (data_manager.py'den gelen)
            config:      Hangi indikatörler, hangi period'larla hesaplansın
            drop_warmup: True → warmup barlarını düşür (ML için önerilir)
                         False → NaN içeren satırları koru (grafik için)

        Returns:
            IndicatorResult: success=True ise dataframe dolu, False ise
                             user_message GUI'de gösterilir.
        """
        if config is None:
            config = IndicatorConfig()

        try:
            cls._validate_input(df)
        except InsufficientDataError as exc:
            return IndicatorResult(
                success=False,
                error_code="INSUFFICIENT_DATA",
                user_message=exc.user_message,
            )

        result_df = df.copy()
        added: list[str] = []
        bars_before = len(result_df)

        # Her indikatör grubunu sırayla hesapla
        compute_steps = [
            (config.enabled_rsi,       lambda d: cls.add_rsi(d, config.rsi_period),       "rsi"),
            (config.enabled_stoch_rsi, lambda d: cls.add_stoch_rsi(d, config.stoch_rsi_period), "stoch_rsi"),
            (config.enabled_ema,       lambda d: cls.add_ema(d, config.ema_periods),       "ema"),
            (config.enabled_macd,      lambda d: cls.add_macd(d, config.macd_fast, config.macd_slow, config.macd_signal), "macd"),
            (config.enabled_atr,       lambda d: cls.add_atr(d, config.atr_period),        "atr"),
            (config.enabled_bbands,    lambda d: cls.add_bbands(d, config.bb_period, config.bb_std), "bbands"),
            (config.enabled_volume,    lambda d: cls.add_volume(d, config.volume_sma_period, config.obv_sma_period), "volume"),
            (config.enabled_candle,    lambda d: cls.add_candle_features(d),               "candle"),
        ]

        for enabled, fn, name in compute_steps:
            if not enabled:
                logger.debug("Atlandı: %s (devre dışı)", name)
                continue
            try:
                result_df = fn(result_df)
                new_cols = INDICATOR_COLUMNS.get(name, [])
                added.extend([c for c in new_cols if c in result_df.columns])
                logger.debug("Hesaplandı: %s | %d kolon", name, len(new_cols))
            except Exception as exc:
                logger.error("İndikatör hatası: %s — %s", name, exc)
                return IndicatorResult(
                    success=False,
                    error_code="COMPUTE_ERROR",
                    user_message=f"'{name}' hesaplanırken hata oluştu. Veri formatını kontrol edin.",
                )

        # Warmup satırlarını düşür
        warmup = 0
        if drop_warmup:
            before = len(result_df)
            result_df = result_df.dropna(subset=added) if added else result_df.dropna()
            warmup = before - len(result_df)

        bars_remaining = len(result_df)
        summary = (
            f"{len(added)} kolon eklendi | "
            f"{warmup} warmup bar düşürüldü | "
            f"{bars_remaining} bar kullanılabilir"
        )

        logger.info("Teknik indikatörler hazır | %s", summary)

        return IndicatorResult(
            success=True,
            dataframe=result_df,
            added_columns=added,
            warmup_bars=warmup,
            bars_remaining=bars_remaining,
            summary=summary,
        )

    # ── Momentum ─────────────────────────────────────────────────────────────

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        RSI (Relative Strength Index)

        Üretilen kolonlar:
            rsi  — 0-100 arası değer. >70 aşırı alım, <30 aşırı satım.

        GUI kullanımı: Grafik alt panelinde RSI çizgisi olarak gösterilir.
        """
        df = df.copy()
        rsi = ta.rsi(df["close"], length=period)
        df["rsi"] = rsi
        return df

    @staticmethod
    def add_stoch_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Stochastic RSI — RSI'nın hassaslaştırılmış versiyonu.

        Üretilen kolonlar:
            stoch_rsi_k  — Hızlı sinyal (0-100)
            stoch_rsi_d  — Yavaş sinyal / K'nın SMA'sı (0-100)

        GUI kullanımı: RSI panelinde K/D çizgisi olarak gösterilir.
        """
        df = df.copy()
        stoch = ta.stochrsi(df["close"], length=period, rsi_length=period, k=3, d=3)
        if stoch is not None and not stoch.empty:
            # pandas_ta kolon adları dinamik gelir, normalize ediyoruz
            cols = stoch.columns.tolist()
            if len(cols) >= 2:
                df["stoch_rsi_k"] = stoch.iloc[:, 0]
                df["stoch_rsi_d"] = stoch.iloc[:, 1]
        return df

    # ── Trend ─────────────────────────────────────────────────────────────────

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
        """
        Exponential Moving Average — çoklu period.

        Üretilen kolonlar:
            ema_20, ema_50, ema_200   — EMA değerleri
            ema_cross_20_50           — EMA20 > EMA50 ise 1, değilse 0
                                        (golden/death cross sinyali)
            ema_cross_50_200          — EMA50 > EMA200 ise 1, değilse 0
            price_vs_ema200           — (close - ema200) / ema200
                                        Fiyatın uzun vadeli trendden sapması

        GUI kullanımı: Grafik üzerinde overlay olarak gösterilir.
        """
        if periods is None:
            periods = [20, 50, 200]

        df = df.copy()
        for period in sorted(periods):
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        # Cross sinyalleri — sadece istenen periodlar varsa hesapla
        if 20 in periods and 50 in periods:
            df["ema_cross_20_50"] = (df["ema_20"] > df["ema_50"]).astype(int)

        if 50 in periods and 200 in periods:
            df["ema_cross_50_200"] = (df["ema_50"] > df["ema_200"]).astype(int)

        if 200 in periods:
            df["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)

        Üretilen kolonlar:
            macd        — MACD çizgisi (hızlı EMA - yavaş EMA)
            macd_signal — Signal çizgisi (MACD'nin EMA'sı)
            macd_hist   — Histogram (macd - signal), momentum değişimi

        GUI kullanımı: Grafik alt panelinde histogram + çizgi olarak gösterilir.
        """
        df = df.copy()
        macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if macd_df is not None and not macd_df.empty:
            df["macd"]        = macd_df.iloc[:, 0]   # MACD_fast_slow_signal
            df["macd_signal"] = macd_df.iloc[:, 2]   # MACDs
            df["macd_hist"]   = macd_df.iloc[:, 1]   # MACDh
        return df

    # ── Volatilite ────────────────────────────────────────────────────────────

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ATR (Average True Range) — Volatilite ölçütü.

        Üretilen kolonlar:
            atr  — Ortalama gerçek aralık (fiyat birimi)

        Kullanım alanları:
            - Stop-loss mesafesi (risk yönetimi)
            - Pozisyon büyüklüğü hesabı
            - market_regime.py'deki volatilite tespiti

        GUI kullanımı: Risk yönetimi panelinde "güncel ATR" değeri olarak gösterilir.
        """
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=period)
        return df

    @staticmethod
    def add_bbands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Bollinger Bands — Volatilite bandı.

        Üretilen kolonlar:
            bb_upper    — Üst band (orta + std * std_sapma)
            bb_mid      — Orta band (SMA)
            bb_lower    — Alt band (orta - std * std_sapma)
            bb_width    — Band genişliği: (upper - lower) / mid
                          Yüksek → yüksek volatilite
            bb_percent  — Fiyatın band içindeki konumu: (close - lower) / (upper - lower)
                          0 = alt banda değiyor, 1 = üst banda değiyor

        GUI kullanımı: Grafik üzerinde şeffaf alan olarak gösterilir.
        """
        df = df.copy()
        bb = ta.bbands(df["close"], length=period, std=std)
        if bb is not None and not bb.empty:
            df["bb_lower"]   = bb.iloc[:, 0]   # BBL
            df["bb_mid"]     = bb.iloc[:, 1]   # BBM
            df["bb_upper"]   = bb.iloc[:, 2]   # BBU
            df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
            df["bb_percent"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            ).replace(0, np.nan)   # sıfıra bölünmeyi önle
        return df

    # ── Hacim ─────────────────────────────────────────────────────────────────

    @staticmethod
    def add_volume(
        df: pd.DataFrame,
        sma_period: int = 20,
        obv_sma_period: int = 20,
    ) -> pd.DataFrame:
        """
        Hacim tabanlı özellikler.

        Üretilen kolonlar:
            volume_sma    — Hacim hareketli ortalaması
            volume_ratio  — volume / volume_sma
                            >1.5 → anormal yüksek hacim (kırılma sinyali)
                            <0.5 → sessiz piyasa
            obv           — On-Balance Volume: hacim yönünü fiyatla ilişkilendirir
            obv_sma       — OBV'nin hareketli ortalaması (trend onayı için)

        GUI kullanımı: Grafik alt panelinde hacim çubukları + ratio çizgisi.
        """
        df = df.copy()
        df["volume_sma"]   = ta.sma(df["volume"], length=sma_period)
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
        df["obv"]          = ta.obv(df["close"], df["volume"])
        df["obv_sma"]      = ta.sma(df["obv"], length=obv_sma_period)
        return df

    # ── Mum Özellikleri ───────────────────────────────────────────────────────

    @staticmethod
    def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ham fiyat verisinden türetilen mum örüntüsü özellikleri.
        pandas_ta gerektirmez, saf pandas hesabı.

        Üretilen kolonlar:
            candle_body        — |close - open| / open  (gövde büyüklüğü, %)
            candle_upper_wick  — (high - max(open,close)) / open  (üst fitil, %)
            candle_lower_wick  — (min(open,close) - low) / open   (alt fitil, %)
            is_bullish         — close > open ise 1, değilse 0

        GUI kullanımı: Mum grafiği üzerinde örüntü işaretçileri.
        ML kullanımı: Doji, hammer, shooting star gibi örüntüleri ML modeline öğretir.
        """
        df = df.copy()

        body_top    = df[["open", "close"]].max(axis=1)
        body_bottom = df[["open", "close"]].min(axis=1)
        open_safe   = df["open"].replace(0, np.nan)

        df["candle_body"]        = (body_top - body_bottom) / open_safe
        df["candle_upper_wick"]  = (df["high"] - body_top) / open_safe
        df["candle_lower_wick"]  = (body_bottom - df["low"]) / open_safe
        df["is_bullish"]         = (df["close"] > df["open"]).astype(int)

        return df

    # ── Yardımcı Metodlar ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """
        DataFrame'in geçerli olduğunu doğrular.
        Hatalı veri ile GUI'de sessiz çöküş yerine anlamlı hata gösterir.
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Eksik kolonlar: {missing}")

        if len(df) < MIN_BARS_REQUIRED:
            raise InsufficientDataError(have=len(df), need=MIN_BARS_REQUIRED)

    @staticmethod
    def get_indicator_info() -> dict[str, dict]:
        """
        GUI'deki "İndikatör Bilgisi" tooltip'leri için meta veri.

        Returns:
            {
                "rsi": {
                    "name": "RSI",
                    "category": "Momentum İndikatörleri",
                    "columns": ["rsi"],
                    "description": "..."
                },
                ...
            }
        """
        return {
            "rsi": {
                "name": "RSI (Göreceli Güç Endeksi)",
                "category": INDICATOR_CATEGORIES["momentum"],
                "columns": INDICATOR_COLUMNS["rsi"],
                "description": (
                    "0-100 arası momentum ölçütü. "
                    "70 üzeri aşırı alım, 30 altı aşırı satım bölgesidir."
                ),
            },
            "stoch_rsi": {
                "name": "Stochastic RSI",
                "category": INDICATOR_CATEGORIES["momentum"],
                "columns": INDICATOR_COLUMNS["stoch_rsi"],
                "description": (
                    "RSI'nın Stochastic uygulaması. "
                    "K ve D çizgilerinin kesişimi alım/satım sinyali verir."
                ),
            },
            "ema": {
                "name": "EMA (Üstel Hareketli Ortalama)",
                "category": INDICATOR_CATEGORIES["trend"],
                "columns": INDICATOR_COLUMNS["ema"],
                "description": (
                    "Trend yönünü belirler. "
                    "EMA20 > EMA50 yükseliş trendi (golden cross), "
                    "EMA20 < EMA50 düşüş trendi (death cross) işareti."
                ),
            },
            "macd": {
                "name": "MACD",
                "category": INDICATOR_CATEGORIES["trend"],
                "columns": INDICATOR_COLUMNS["macd"],
                "description": (
                    "Momentum değişimini ölçer. "
                    "Histogram sıfır çizgisini yukarı kesiyor → alım sinyali."
                ),
            },
            "atr": {
                "name": "ATR (Ortalama Gerçek Aralık)",
                "category": INDICATOR_CATEGORIES["volatility"],
                "columns": INDICATOR_COLUMNS["atr"],
                "description": (
                    "Piyasa volatilitesini ölçer. "
                    "Stop-loss ve pozisyon büyüklüğü hesabında kullanılır."
                ),
            },
            "bbands": {
                "name": "Bollinger Bantları",
                "category": INDICATOR_CATEGORIES["volatility"],
                "columns": INDICATOR_COLUMNS["bbands"],
                "description": (
                    "Fiyatın istatistiksel sınırlarını gösterir. "
                    "Band daralması büyük hareket öncesi görülür."
                ),
            },
            "volume": {
                "name": "Hacim İndikatörleri",
                "category": INDICATOR_CATEGORIES["volume"],
                "columns": INDICATOR_COLUMNS["volume"],
                "description": (
                    "Fiyat hareketinin güvenilirliğini ölçer. "
                    "Yüksek hacimli kırılmalar daha güçlü sinyal verir."
                ),
            },
            "candle": {
                "name": "Mum Özellikleri",
                "category": INDICATOR_CATEGORIES["candle"],
                "columns": INDICATOR_COLUMNS["candle"],
                "description": (
                    "Ham fiyat verisinden türetilen özellikler. "
                    "Doji, hammer gibi örüntüleri ML modeline öğretir."
                ),
            },
        }


# ── CLI / Smoke Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("\n" + "="*60)
    print("  FreqTradeAIBot — TechnicalIndicators Smoke Test")
    print("="*60)

    # Örnek OHLCV verisi oluştur (data_manager olmadan test)
    print("\n  Sentetik OHLCV verisi oluşturuluyor (500 bar)...")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 40000 + np.cumsum(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    high  = np.maximum(close, open_) + abs(np.random.randn(n) * 150)
    low   = np.minimum(close, open_) - abs(np.random.randn(n) * 150)
    vol   = abs(np.random.randn(n) * 1000 + 5000)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )
    df.index.name = "date"
    print(f"  Girdi: {df.shape[0]} bar x {df.shape[1]} kolon")

    # Varsayılan config ile hesapla
    print("\n  Tüm indikatörler hesaplanıyor...")
    config = IndicatorConfig()
    result = TechnicalIndicators.compute(df, config, drop_warmup=True)

    if result.success:
        print(f"\n  Sonuc: {result.summary}")
        print(f"\n  Eklenen kolonlar ({len(result.added_columns)}):")
        for col in sorted(result.added_columns):
            sample = result.dataframe[col].dropna().iloc[-1]
            print(f"    {col:<22} son deger: {sample:.4f}")

        print(f"\n  NaN kontrolu:")
        nan_counts = result.dataframe[result.added_columns].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if nan_cols.empty:
            print("    Tum kolonlar temiz (NaN yok)")
        else:
            print(f"    NaN iceren kolonlar:\n{nan_cols}")

        print(f"\n  Kullanilabilir veri: {result.bars_remaining} bar")
    else:
        print(f"\n  HATA [{result.error_code}]: {result.user_message}")

    # Bireysel indikatör testi
    print("\n  Bireysel indikatör testi (sadece RSI + EMA):")
    config2 = IndicatorConfig(
        enabled_rsi=True,
        enabled_ema=True,
        enabled_stoch_rsi=False,
        enabled_macd=False,
        enabled_atr=False,
        enabled_bbands=False,
        enabled_volume=False,
        enabled_candle=False,
    )
    result2 = TechnicalIndicators.compute(df, config2)
    if result2.success:
        print(f"  Eklenen: {result2.added_columns}")

    print("\n" + "="*60)
    print("  Test tamamlandi.")
    print("="*60 + "\n")