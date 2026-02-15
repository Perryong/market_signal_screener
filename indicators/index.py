import pandas as pd

from indicators.oscillators import Oscillators
from indicators.overlays import Overlays


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare technical indicators for a given DataFrame."""
    df = df.copy()

    # RSI
    df["rsi"] = Oscillators.rsi(df["Close"], 14)

    # MACD
    m_line, m_signal, m_hist = Oscillators.macd(df["Close"])
    df["macd"] = m_line
    df["macd_signal"] = m_signal
    df["macd_hist"] = m_hist

    # ATR
    df["atr"] = Oscillators.atr(df, 14)

    # ADX with +DI/-DI
    adx_val, plus_di, minus_di = Oscillators.adx_with_di(df, 14)
    df["adx"] = adx_val
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # OBV slope (vectorized)
    df["obv_slope"] = Oscillators.on_balance_volume_slope(df)

    # Divergence detection
    df["rsi_divergence"] = Oscillators.rsi_divergence(df)
    df["macd_divergence"] = Oscillators.macd_divergence(df)

    # Moving averages
    df["ma50"] = Overlays.sma(df["Close"], 50)
    df["ma200"] = Overlays.sma(df["Close"], 200)

    # Bollinger Bands
    upper, mid, lower = Overlays.bollinger_bands(df["Close"], 20, 2)
    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower

    # Volume SMA
    df["volume_sma20"] = Overlays.sma(df["Volume"], 20)

    # Support/Resistance with type
    snr_level, snr_type = Overlays.winshift_snr_with_type(df[["High", "Low", "Open", "Close"]])
    df["snr"] = snr_level
    df["snr_type"] = snr_type

    return df
