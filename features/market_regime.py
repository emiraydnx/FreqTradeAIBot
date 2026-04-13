"""
features/market_regime.py — Piyasa Rejimi Tespiti
==================================================

AMAÇ:
─────
ML modeline "bu bar hangi piyasa koşulunda oluştu?" sorusunun cevabını
feature olarak vermek. LightGBM, trending ve ranging dönemleri için
farklı ağırlıklar öğrenebilir — bu dosya olmadan model bu iki durumu
kör bir şekilde karıştırır.

TASARIM PRENSİPLERİ (GUI Entegrasyonu İçin):
─────────────────────────────────────────────
1. RegimeConfig → GUI'deki "Rejim Ayarları" paneli eşik değerlerini
   slider ile ayarlayabilir (adx_trend_threshold, volatility_window vb.)
2. RegimeResult → GUI'ye hem DataFrame hem görsel açıklama döner.
3. get_regime_summary() → Dashboard'da "Mevcut Piyasa Durumu" kartı için.
4. Her tespit metodu bağımsız — GUI tek bir rejim tipini de sorgulayabilir.
5. Kullanıcı dostu hata mesajları.

EKLENEN ÖZELLİKLER:
────────────────────
    Trend Rejimi   : adx, regime_trending, regime_strength (zayıf/orta/güçlü)
    Volatilite     : volatility_ratio, regime_volatility (düşük/normal/yüksek)
    Hacim Rejimi   : volume_regime (sessiz/normal/aktif)
    Kompozit Skor  : market_regime_score (0-100, GUI gauge widget için)
    Özet Etiketi   : market_regime_label ("Güçlü Yükseliş Trendi" vb.)

GUI KULLANIM ÖRNEĞİ (ilerideki gui/dashboard.py için):
───────────────────────────────────────────────────────
    # Mevcut piyasa durumu kartı
    result = MarketRegime.compute(df, config)
    if result.success:
        latest = result.current_regime   # En son bar'ın rejim bilgisi
        self.regime_label.setText(latest["label"])
        self.regime_gauge.setValue(latest["score"])
        self.regime_badge.setColor(latest["color"])  # yeşil/sarı/kırmızı

PIPELINE KULLANIM ÖRNEĞİ (feature_pipeline.py için):
──────────────────────────────────────────────────────
    df = TechnicalIndicators.compute(df, tech_config).dataframe
    result = MarketRegime.compute(df, regime_config)
    df = result.dataframe   # artık rejim kolonları da eklenmiş
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

MIN_BARS_REQUIRED = 100   # ADX(14) + volatilite penceresi için minimum

# Rejim etiket renk kodları — GUI badge/icon rengi için
REGIME_COLORS = {
    "Güçlü Yükseliş Trendi":   "#10b981",   # yeşil
    "Yükseliş Trendi":          "#34d399",   # açık yeşil
    "Güçlü Düşüş Trendi":      "#ef4444",   # kırmızı
    "Düşüş Trendi":             "#f87171",   # açık kırmızı
    "Yatay Piyasa":             "#f59e0b",   # sarı
    "Yüksek Volatilite":        "#8b5cf6",   # mor
    "Belirsiz":                 "#64748b",   # gri
}

# Üretilen tüm kolon adları — feature_pipeline.py bu listeyi kullanır
REGIME_COLUMNS = [
    # Trend
    "adx",
    "regime_trending",       # 1=trend, 0=ranging
    "regime_strength",       # 0=zayıf, 1=orta, 2=güçlü
    # Volatilite
    "volatility_ratio",      # atr / atr_rolling_mean
    "regime_volatility",     # 0=düşük, 1=normal, 2=yüksek
    # Hacim
    "volume_regime",         # 0=sessiz, 1=normal, 2=aktif
    # Kompozit
    "market_regime_score",   # 0-100 (GUI gauge için)
]


# ── Config & Sonuç Nesneleri ──────────────────────────────────────────────────

@dataclass
class RegimeConfig:
    """
    Rejim tespiti parametreleri.
    GUI'deki "Rejim Ayarları" paneli bu nesneyi oluşturur.

    Eşik değerleri slider ile ayarlanabilir:
        adx_trend_threshold    : Bu değerin üzeri = trend var
        adx_strong_threshold   : Bu değerin üzeri = güçlü trend
        volatility_window      : Volatilite ort. hesabı için pencere
        volatility_high_mult   : ATR / ATR_ortalama > bu oran = yüksek volatilite
        volatility_low_mult    : ATR / ATR_ortalama < bu oran = düşük volatilite
        volume_high_mult       : Hacim / Hacim_ortalama > bu oran = aktif
        volume_low_mult        : Hacim / Hacim_ortalama < bu oran = sessiz
    """
    adx_period: int              = 14
    adx_trend_threshold: float   = 25.0    # klasik ADX eşiği
    adx_strong_threshold: float  = 40.0    # güçlü trend eşiği

    volatility_window: int       = 50      # ATR ortalaması için
    volatility_high_mult: float  = 1.5     # 1.5x ortalama ATR = yüksek vol
    volatility_low_mult: float   = 0.7     # 0.7x ortalama ATR = düşük vol

    volume_window: int           = 20
    volume_high_mult: float      = 1.5
    volume_low_mult: float       = 0.7

    # Kompozit skor ağırlıkları (toplam 1.0 olmalı)
    score_weight_trend: float      = 0.50
    score_weight_volatility: float = 0.30
    score_weight_volume: float     = 0.20


@dataclass
class CurrentRegime:
    """
    En son bar'ın rejim bilgisi.
    GUI'deki "Mevcut Piyasa Durumu" kartı bu nesneyi gösterir.
    """
    label: str           # "Güçlü Yükseliş Trendi"
    score: int           # 0-100
    color: str           # "#10b981"
    adx: float           # Sayısal ADX değeri
    is_trending: bool
    volatility_level: str   # "Düşük" / "Normal" / "Yüksek"
    volume_level: str       # "Sessiz" / "Normal" / "Aktif"
    description: str        # GUI tooltip için uzun açıklama


@dataclass
class RegimeResult:
    """
    MarketRegime.compute() metodunun döndürdüğü sonuç.
    GUI bu nesneyi kontrol eder.
    """
    success: bool
    dataframe: Optional[pd.DataFrame]     = None
    current_regime: Optional[CurrentRegime] = None

    # Başarı istatistikleri
    trending_pct: float   = 0.0    # Verinin kaçında trend var (%)
    high_vol_pct: float   = 0.0    # Verinin kaçında yüksek volatilite (%)
    summary: str          = ""

    # Hata
    error_code: str       = ""
    user_message: str     = ""


# ── Özel Hata ────────────────────────────────────────────────────────────────

class RegimeComputeError(RuntimeError):
    def __init__(self, detail: str, user_message: str = ""):
        self.user_message = user_message or "Rejim analizi hesaplanamadı. Teknik indikatörlerin hesaplanmış olduğundan emin olun."
        super().__init__(detail)


# ── MarketRegime ──────────────────────────────────────────────────────────────

class MarketRegime:
    """
    Piyasa rejimini tespit eden ana sınıf.

    ÖNEMLI: Bu sınıf TechnicalIndicators'tan SONRA çalışır.
    DataFrame'de 'atr' ve 'volume_sma' kolonlarının var olması beklenir.
    Yoksa bu değerleri kendisi hesaplar.

    Kullanım — tam hesaplama:
        result = MarketRegime.compute(df, config)

    Kullanım — sadece mevcut rejim (GUI dashboard kartı):
        regime = MarketRegime.get_current(df)
        self.label.setText(regime.label)
    """

    # ── Ana Giriş Noktası ─────────────────────────────────────────────────────

    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        config: Optional[RegimeConfig] = None,
    ) -> RegimeResult:
        """
        Tüm rejim özelliklerini hesaplar ve DataFrame'e ekler.

        Args:
            df:      TechnicalIndicators.compute() çıktısı (OHLCV + teknik ind.)
            config:  Rejim parametreleri

        Returns:
            RegimeResult: success=True ise dataframe + current_regime dolu.
        """
        if config is None:
            config = RegimeConfig()

        if len(df) < MIN_BARS_REQUIRED:
            return RegimeResult(
                success=False,
                error_code="INSUFFICIENT_DATA",
                user_message=(
                    f"Rejim analizi için en az {MIN_BARS_REQUIRED} bar gerekli, "
                    f"{len(df)} bar mevcut."
                ),
            )

        result_df = df.copy()

        try:
            # 1. Bağımlı kolonları hazırla (teknik indikatörlerden gelmesi beklenir)
            result_df = cls._ensure_dependencies(result_df, config)

            # 2. Trend rejimi
            result_df = cls._add_trend_regime(result_df, config)

            # 3. Volatilite rejimi
            result_df = cls._add_volatility_regime(result_df, config)

            # 4. Hacim rejimi
            result_df = cls._add_volume_regime(result_df, config)

            # 5. Kompozit skor
            result_df = cls._add_composite_score(result_df, config)

            # 6. NaN temizliği (yalnızca rejim kolonlarında)
            result_df = result_df.dropna(subset=REGIME_COLUMNS)

        except Exception as exc:
            logger.exception("Rejim hesaplama hatası: %s", exc)
            return RegimeResult(
                success=False,
                error_code="COMPUTE_ERROR",
                user_message="Rejim analizi sırasında beklenmedik hata oluştu.",
            )

        # İstatistikler
        trending_pct = round(result_df["regime_trending"].mean() * 100, 1)
        high_vol_pct = round((result_df["regime_volatility"] == 2).mean() * 100, 1)

        summary = (
            f"Trend oranı: %{trending_pct} | "
            f"Yüksek volatilite: %{high_vol_pct} | "
            f"{len(result_df)} bar analiz edildi"
        )

        # Mevcut rejim (son bar)
        current = cls._build_current_regime(result_df, config)

        logger.info("Rejim analizi tamamlandı | %s", summary)

        return RegimeResult(
            success=True,
            dataframe=result_df,
            current_regime=current,
            trending_pct=trending_pct,
            high_vol_pct=high_vol_pct,
            summary=summary,
        )

    # ── Mevcut Rejim (GUI Dashboard İçin) ─────────────────────────────────────

    @classmethod
    def get_current(
        cls,
        df: pd.DataFrame,
        config: Optional[RegimeConfig] = None,
    ) -> CurrentRegime:
        """
        Sadece son bar'ın rejim bilgisini döner.
        GUI'deki "Anlık Piyasa Durumu" kartı için optimize edilmiş hızlı yol.

        Tüm DataFrame'i hesaplamak yerine sadece son N bar'ı işler.
        """
        if config is None:
            config = RegimeConfig()

        # Yeterli warmup için son (window + period) bar'ı al
        needed = max(config.volatility_window, config.adx_period * 3) + 10
        slice_df = df.tail(needed).copy()

        result = cls.compute(slice_df, config)
        if result.success and result.current_regime:
            return result.current_regime

        # Hesaplama başarısız olursa varsayılan döndür
        return CurrentRegime(
            label="Belirsiz",
            score=50,
            color=REGIME_COLORS["Belirsiz"],
            adx=0.0,
            is_trending=False,
            volatility_level="Normal",
            volume_level="Normal",
            description="Yeterli veri yok veya hesaplama yapılamadı.",
        )

    @staticmethod
    def get_regime_colors() -> dict[str, str]:
        """
        GUI badge renkleri için renk haritası.
        gui/dashboard.py bu metodu çağırarak renk tanımlarını alır.
        """
        return REGIME_COLORS.copy()

    # ── Bağımlılık Kontrolü ───────────────────────────────────────────────────

    @classmethod
    def _ensure_dependencies(
        cls, df: pd.DataFrame, config: RegimeConfig
    ) -> pd.DataFrame:
        """
        ATR ve volume_sma teknik indikatörlerden gelmesi beklenir.
        Eğer yoksa burada hesaplar — pipeline sırası değişse bile çalışır.
        """
        if "atr" not in df.columns:
            logger.warning("'atr' kolonu bulunamadı, hesaplanıyor...")
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=config.adx_period)

        if "volume_sma" not in df.columns:
            logger.warning("'volume_sma' kolonu bulunamadı, hesaplanıyor...")
            df["volume_sma"] = ta.sma(df["volume"], length=config.volume_window)

        return df

    # ── Trend Rejimi ──────────────────────────────────────────────────────────

    @classmethod
    def _add_trend_regime(
        cls, df: pd.DataFrame, config: RegimeConfig
    ) -> pd.DataFrame:
        """
        ADX (Average Directional Index) ile trend gücü tespiti.

        ADX < 20        → ranging (yatay)     regime_trending = 0
        20 ≤ ADX < 25   → zayıf trend         regime_trending = 1, strength = 0
        25 ≤ ADX < 40   → orta trend          regime_trending = 1, strength = 1
        ADX ≥ 40        → güçlü trend         regime_trending = 1, strength = 2

        NOT: ADX trend YÖNÜNÜ söylemez, sadece GÜCÜNÜ söyler.
        Yön için EMA cross sinyalleri (technical.py) kullanılır.
        """
        adx_result = ta.adx(
            df["high"], df["low"], df["close"],
            length=config.adx_period
        )

        # pandas_ta ADX kolonunu dinamik isimle döndürür: ADX_14
        if adx_result is not None and not adx_result.empty:
            adx_col = [c for c in adx_result.columns if c.startswith("ADX_")]
            if adx_col:
                df["adx"] = adx_result[adx_col[0]]
            else:
                df["adx"] = adx_result.iloc[:, 0]
        else:
            logger.warning("ADX hesaplanamadı, sıfır ile dolduruldu.")
            df["adx"] = 0.0

        # Trending flag
        df["regime_trending"] = (
            df["adx"] >= config.adx_trend_threshold
        ).astype(int)

        # Trend gücü kategorisi
        conditions = [
            df["adx"] < config.adx_trend_threshold,
            (df["adx"] >= config.adx_trend_threshold) & (df["adx"] < config.adx_strong_threshold),
            df["adx"] >= config.adx_strong_threshold,
        ]
        df["regime_strength"] = np.select(conditions, [0, 1, 2], default=0)

        return df

    # ── Volatilite Rejimi ─────────────────────────────────────────────────────

    @classmethod
    def _add_volatility_regime(
        cls, df: pd.DataFrame, config: RegimeConfig
    ) -> pd.DataFrame:
        """
        ATR'nin kendi geçmiş ortalamasına oranıyla volatilite seviyesi tespiti.

        volatility_ratio = atr / atr_rolling_mean(window)

        ratio < 0.7   → düşük volatilite   regime_volatility = 0
        0.7 - 1.5     → normal             regime_volatility = 1
        ratio > 1.5   → yüksek volatilite  regime_volatility = 2

        Bu oran mutlak fiyata bağımlı değil — BTC 30k'da da 60k'da da çalışır.
        """
        atr_mean = df["atr"].rolling(window=config.volatility_window, min_periods=10).mean()

        # Sıfıra bölünmeyi önle
        df["volatility_ratio"] = df["atr"] / atr_mean.replace(0, np.nan)

        conditions = [
            df["volatility_ratio"] < config.volatility_low_mult,
            df["volatility_ratio"] <= config.volatility_high_mult,
            df["volatility_ratio"] > config.volatility_high_mult,
        ]
        df["regime_volatility"] = np.select(conditions, [0, 1, 2], default=1)

        return df

    # ── Hacim Rejimi ──────────────────────────────────────────────────────────

    @classmethod
    def _add_volume_regime(
        cls, df: pd.DataFrame, config: RegimeConfig
    ) -> pd.DataFrame:
        """
        Anlık hacmin hareketli ortalamaya oranıyla hacim aktivitesi tespiti.

        volume_ratio < 0.7   → sessiz piyasa    volume_regime = 0
        0.7 - 1.5            → normal           volume_regime = 1
        volume_ratio > 1.5   → aktif piyasa     volume_regime = 2

        volume_sma teknik indikatörlerden gelmeli (ya da _ensure_dependencies'te hesaplanır).
        """
        volume_ratio = df["volume"] / df["volume_sma"].replace(0, np.nan)

        conditions = [
            volume_ratio < config.volume_low_mult,
            volume_ratio <= config.volume_high_mult,
            volume_ratio > config.volume_high_mult,
        ]
        df["volume_regime"] = np.select(conditions, [0, 1, 2], default=1)

        return df

    # ── Kompozit Skor ─────────────────────────────────────────────────────────

    @classmethod
    def _add_composite_score(
        cls, df: pd.DataFrame, config: RegimeConfig
    ) -> pd.DataFrame:
        """
        Tüm rejim bileşenlerini 0-100 arası tek bir skora indirger.
        GUI'deki gauge/speedometer widget'ı bu skoru gösterir.

        Skor hesabı:
            Trend bileşeni    (ağırlık: 0.50): ADX normalize → 0-100
            Volatilite bileşeni (ağırlık: 0.30): volatility_ratio → 0-100
            Hacim bileşeni    (ağırlık: 0.20): volume_regime → 0-100

        Düşük skor  (0-30)  → durgun, yatay piyasa
        Orta skor   (30-70) → karışık sinyal
        Yüksek skor (70-100)→ güçlü trend + aktif piyasa
        """
        # Trend bileşeni: ADX 0-60 → 0-100 normalize
        trend_score = (df["adx"].clip(0, 60) / 60 * 100)

        # Volatilite bileşeni: ratio 0-3 → 0-100
        vol_score = (df["volatility_ratio"].clip(0, 3) / 3 * 100).fillna(50)

        # Hacim bileşeni: 0/1/2 → 0/50/100
        vol_regime_score = df["volume_regime"] * 50.0

        # Ağırlıklı ortalama
        df["market_regime_score"] = (
            trend_score      * config.score_weight_trend
            + vol_score      * config.score_weight_volatility
            + vol_regime_score * config.score_weight_volume
        ).clip(0, 100).round(1)

        return df

    # ── Mevcut Rejim Nesnesi ──────────────────────────────────────────────────

    @classmethod
    def _build_current_regime(
        cls, df: pd.DataFrame, config: RegimeConfig
    ) -> CurrentRegime:
        """
        Son bar'ın verilerinden CurrentRegime nesnesi oluşturur.
        GUI dashboard kartı için tüm bilgiyi tek nesnede toplar.
        """
        last = df.iloc[-1]

        adx_val        = float(last.get("adx", 0))
        is_trending    = bool(last.get("regime_trending", 0))
        strength       = int(last.get("regime_strength", 0))
        vol_regime     = int(last.get("regime_volatility", 1))
        volume_regime  = int(last.get("volume_regime", 1))
        score          = int(last.get("market_regime_score", 50))

        # Trend yönü için EMA cross sinyalleri (varsa)
        ema_cross = last.get("ema_cross_20_50", None)
        is_uptrend = ema_cross == 1 if ema_cross is not None else None

        # Etiket ve renk belirleme
        label, color = cls._determine_label(
            is_trending, strength, vol_regime, is_uptrend
        )

        # Seviye açıklamaları
        vol_labels   = {0: "Düşük", 1: "Normal", 2: "Yüksek"}
        vol_rej_lbls = {0: "Sessiz", 1: "Normal", 2: "Aktif"}

        vol_level    = vol_labels.get(vol_regime, "Normal")
        volume_level = vol_rej_lbls.get(volume_regime, "Normal")

        # Uzun açıklama (GUI tooltip)
        direction_txt = ""
        if is_uptrend is True:
            direction_txt = "Yükseliş yönlü (EMA20 > EMA50). "
        elif is_uptrend is False:
            direction_txt = "Düşüş yönlü (EMA20 < EMA50). "

        strength_txt = {0: "Zayıf", 1: "Orta", 2: "Güçlü"}.get(strength, "")

        description = (
            f"ADX: {adx_val:.1f} ({strength_txt} {'trend' if is_trending else 'yatay'}). "
            f"{direction_txt}"
            f"Volatilite: {vol_level}. "
            f"Hacim aktivitesi: {volume_level}. "
            f"Kompozit skor: {score}/100."
        )

        return CurrentRegime(
            label=label,
            score=score,
            color=color,
            adx=adx_val,
            is_trending=is_trending,
            volatility_level=vol_level,
            volume_level=volume_level,
            description=description,
        )

    @staticmethod
    def _determine_label(
        is_trending: bool,
        strength: int,
        vol_regime: int,
        is_uptrend: Optional[bool],
    ) -> tuple[str, str]:
        """
        Rejim parametrelerinden insan okunabilir etiket ve renk üretir.
        GUI badge ve ikon rengi için kullanılır.
        """
        if vol_regime == 2 and not is_trending:
            label = "Yüksek Volatilite"
            return label, REGIME_COLORS[label]

        if not is_trending:
            label = "Yatay Piyasa"
            return label, REGIME_COLORS[label]

        # Trend var — yön ve güce göre etiket
        if is_uptrend is True:
            label = "Güçlü Yükseliş Trendi" if strength == 2 else "Yükseliş Trendi"
        elif is_uptrend is False:
            label = "Güçlü Düşüş Trendi" if strength == 2 else "Düşüş Trendi"
        else:
            # EMA cross bilgisi yoksa sadece güce göre etiketle
            label = "Güçlü Yükseliş Trendi" if strength == 2 else "Yükseliş Trendi"

        return label, REGIME_COLORS.get(label, REGIME_COLORS["Belirsiz"])


# ── CLI / Smoke Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from features.technical import TechnicalIndicators, IndicatorConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("\n" + "="*60)
    print("  FreqTradeAIBot — MarketRegime Smoke Test")
    print("="*60)

    # Sentetik veri
    print("\n  Sentetik OHLCV verisi olusturuluyor (600 bar)...")
    np.random.seed(7)
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 40000 + np.cumsum(np.random.randn(n) * 300)
    open_ = close + np.random.randn(n) * 80
    high  = np.maximum(close, open_) + abs(np.random.randn(n) * 200)
    low   = np.minimum(close, open_) - abs(np.random.randn(n) * 200)
    vol   = abs(np.random.randn(n) * 1500 + 6000)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )
    df.index.name = "date"

    # Önce teknik indikatörler
    print("  Teknik indikatörler hesaplaniyor...")
    tech_result = TechnicalIndicators.compute(df, IndicatorConfig())
    assert tech_result.success, f"Teknik hata: {tech_result.user_message}"
    df_tech = tech_result.dataframe

    # Rejim hesaplama
    print("  Rejim analizi yapiliyor...")
    config = RegimeConfig()
    result = MarketRegime.compute(df_tech, config)

    if result.success:
        print(f"\n  Ozet: {result.summary}")
        print(f"\n  Rejim kolonlari (son 5 deger):")
        cols_to_show = [c for c in REGIME_COLUMNS if c in result.dataframe.columns]
        print(result.dataframe[cols_to_show].tail(5).to_string())

        print(f"\n  Mevcut Piyasa Durumu:")
        cr = result.current_regime
        print(f"    Etiket     : {cr.label}")
        print(f"    Skor       : {cr.score}/100")
        print(f"    Renk kodu  : {cr.color}")
        print(f"    ADX        : {cr.adx:.2f}")
        print(f"    Trend mi?  : {'Evet' if cr.is_trending else 'Hayir'}")
        print(f"    Volatilite : {cr.volatility_level}")
        print(f"    Hacim      : {cr.volume_level}")
        print(f"    Aciklama   : {cr.description}")

        print(f"\n  NaN kontrolu:")
        nan_counts = result.dataframe[cols_to_show].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if nan_cols.empty:
            print("    Tum rejim kolonlari temiz (NaN yok)")
        else:
            print(f"    NaN iceren kolonlar:\n{nan_cols}")
    else:
        print(f"\n  HATA [{result.error_code}]: {result.user_message}")

    print("\n" + "="*60)
    print("  Test tamamlandi.")
    print("="*60 + "\n")