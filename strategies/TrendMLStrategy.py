# =============================================================================
# FREQTRADE AI BOT — TrendMLStrategy.py
# Faz 1: LightGBM tabanlı Trend Following stratejisi
# =============================================================================

import logging
from functools import reduce
from typing import Optional

import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class TrendMLStrategy(IStrategy):
    """
    FreqAI destekli Trend Following stratejisi.
    Model: LightGBMRegressor (Faz 1)
    Timeframe: 1h ana, 4h trend filtresi
    """

    # ── Strateji metadata ──────────────────────────────────────────────────
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # ── FreqAI ────────────────────────────────────────────────────────────
    freqai_info: dict = {}

    # ── Risk yönetimi ──────────────────────────────────────────────────────
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.1,
        "60": 0.05,
        "120": 0.02,
        "240": 0
    }

    # ── Optimize edilebilir parametreler (Hyperopt için) ───────────────────
    buy_rsi_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    sell_rsi_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    trend_ema_period = IntParameter(50, 200, default=100, space="buy", optimize=True)
    ai_threshold_long = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)

    # ── Süreç ayarları ─────────────────────────────────────────────────────
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 200

    # ==========================================================================
    # FEATURE ENGINEERING — FreqAI bu metodları çağırır
    # ==========================================================================

    def feature_engineering_expand_all(
        self, dataframe: pd.DataFrame, period: int, metadata: dict, **kwargs
    ) -> pd.DataFrame:
        """
        FreqAI her period için bu özellikleri otomatik genişletir.
        include_timeframes ve indicator_periods_candles config'den okunur.
        """
        # Momentum
        dataframe["%-rsi"] = pta.rsi(dataframe["close"], length=period)
        dataframe["%-mfi"] = pta.mfi(
            dataframe["high"], dataframe["low"],
            dataframe["close"], dataframe["volume"], length=period
        )

        # Trend
        dataframe["%-ema"] = pta.ema(dataframe["close"], length=period)
        dataframe["%-sma"] = pta.sma(dataframe["close"], length=period)
        dataframe["%-adx"] = pta.adx(
            dataframe["high"], dataframe["low"],
            dataframe["close"], length=period
        )["ADX_" + str(period)]

        # Volatilite
        dataframe["%-atr"] = pta.atr(
            dataframe["high"], dataframe["low"],
            dataframe["close"], length=period
        )
        bb = pta.bbands(dataframe["close"], length=period)
        dataframe["%-bb_width"] = (bb[f"BBU_{period}_2.0"] - bb[f"BBL_{period}_2.0"]) / bb[f"BBM_{period}_2.0"]
        dataframe["%-bb_position"] = (dataframe["close"] - bb[f"BBL_{period}_2.0"]) / (
            bb[f"BBU_{period}_2.0"] - bb[f"BBL_{period}_2.0"]
        )

        # Volume
        dataframe["%-volume_mean"] = dataframe["volume"] / dataframe["volume"].rolling(period).mean()

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: pd.DataFrame, metadata: dict, **kwargs
    ) -> pd.DataFrame:
        """
        Tüm timeframe'lerde bir kez hesaplanan temel özellikler.
        """
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        # MACD
        macd = pta.macd(dataframe["close"])
        dataframe["%-macd"] = macd["MACD_12_26_9"]
        dataframe["%-macd_signal"] = macd["MACDs_12_26_9"]
        dataframe["%-macd_hist"] = macd["MACDh_12_26_9"]

        return dataframe

    def feature_engineering_standard(
        self, dataframe: pd.DataFrame, metadata: dict, **kwargs
    ) -> pd.DataFrame:
        """
        Ham fiyat bilgisi — FreqAI'ın ihtiyaç duyduğu standart özellikler.
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(
        self, dataframe: pd.DataFrame, metadata: dict, **kwargs
    ) -> pd.DataFrame:
        """
        FreqAI'ın tahmin edeceği hedef değer.
        label_period_candles sonraki mumun fiyat değişimi.
        """
        dataframe["&-price_change"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .div(dataframe["close"])
            .sub(1)
        )
        return dataframe

    # ==========================================================================
    # GİRİŞ / ÇIKIŞ SİNYALLERİ
    # ==========================================================================

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        FreqAI modelini çalıştır, tahminleri dataframe'e ekle.
        """
        dataframe = self.freqai.start(dataframe, metadata, self)

        # Ek teknik göstergeler (sinyal filtresi için)
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)
        dataframe["ema_trend"] = pta.ema(dataframe["close"], length=self.trend_ema_period.value)

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        Long giriş koşulları:
        1. FreqAI tahmini pozitif (fiyat yükselecek)
        2. RSI aşırı satım bölgesinde değil
        3. Fiyat trend EMA'nın üzerinde
        """
        conditions = [
            # FreqAI sinyali — model yeterince yükseliş tahmin ediyor
            dataframe["&-price_change"] > self.ai_threshold_long.value,

            # Trend filtresi — fiyat EMA'nın üzerinde
            dataframe["close"] > dataframe["ema_trend"],

            # RSI filtresi — aşırı alım bölgesinde değil
            dataframe["rsi"] < self.sell_rsi_threshold.value,

            # Geçerli mum verisi
            dataframe["volume"] > 0,
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        Long çıkış koşulları:
        1. FreqAI tahmini negatife döndü
        2. RSI aşırı alım bölgesine girdi
        """
        conditions = [
            # FreqAI sinyali tersine döndü
            dataframe["&-price_change"] < 0,

            # RSI aşırı alım
            dataframe["rsi"] > self.sell_rsi_threshold.value,

            dataframe["volume"] > 0,
        ]

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            "exit_long"
        ] = 1

        return dataframe

    # ==========================================================================
    # İSTEĞE BAĞLI — Pozisyon yönetimi
    # ==========================================================================

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """
        ATR tabanlı dinamik stoploss — ileride geliştirilebilir.
        Şimdilik config'deki sabit stoploss kullanılıyor.
        """
        return self.stoploss