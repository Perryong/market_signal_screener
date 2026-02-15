import numpy as np
import pandas as pd
from typing import Tuple

from candle_sticks.helper import get_mid_body, get_price_range


class Overlays:

    @staticmethod
    def sma(series: pd.Series, length: int = 1) -> pd.Series:
        """Simple Moving Average (SMA)."""
        return series.rolling(window=length, min_periods=length).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average (SMA)."""
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def ma_cross(series: pd.Series, fast: int = 9, slow: int = 21) -> pd.Series:
        """
        Moving Average Cross detection.
        Returns:
            +1  when fast MA crosses above slow MA (bullish cross)
            -1  when fast MA crosses below slow MA (bearish cross)
            0  otherwise
        """
        fast_ma = Overlays.sma(series, fast)
        slow_ma = Overlays.sma(series, slow)

        cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        signal = pd.Series(0, index=series.index)
        signal[cross_up] = 1
        signal[cross_down] = -1
        return signal

    @staticmethod
    def bollinger_bands(
        series: pd.Series, length: int = 20, stds: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (upper, middle, lower)."""
        ma = series.rolling(window=length, min_periods=length).mean()
        std = series.rolling(window=length, min_periods=length).std()
        upper = ma + stds * std
        lower = ma - stds * std
        return upper, ma, lower

    @staticmethod
    def recent_swing_low(df: pd.DataFrame, lookback: int = 20) -> float:
        return df["Low"].tail(lookback).min()

    @staticmethod
    def recent_swing_high(df: pd.DataFrame, lookback: int = 20) -> float:
        return df["High"].tail(lookback).max()

    @staticmethod
    def winshift_snr(df: pd.DataFrame, dur: int = 15) -> pd.Series:
        """
        Support/Resistance detection using Window Shifting method.
        Returns a Series indexed like df with detected S/R levels (NaN otherwise).
        """
        level, _ = Overlays.winshift_snr_with_type(df, dur)
        return level

    @staticmethod
    def winshift_snr_with_type(df: pd.DataFrame, dur: int = 15) -> Tuple[pd.Series, pd.Series]:
        """
        Support/Resistance detection using Window Shifting method.

        Returns:
            Tuple of (level, type) Series where:
                - level: the S/R level value (NaN if none)
                - type: "support", "resistance", or NaN
        """
        avg_range = float(np.mean(get_price_range(df)))
        res_buf, sup_buf = [], []
        levels = []  # List of (value, type) tuples

        def is_new_level(value: float) -> bool:
            return all(abs(value - lvl[0]) >= avg_range for lvl in levels)

        for i in range(dur, len(df) - dur):
            # Resistance
            high_window = df["High"][i - dur : i + dur + 1]
            current_max = float(high_window.max())
            res_buf = (
                res_buf + [current_max]
                if (not res_buf or res_buf[-1] == current_max)
                else [current_max]
            )
            if len(res_buf) >= dur and is_new_level(current_max):
                levels.append((current_max, "resistance"))
                res_buf = []

            # Support
            low_window = df["Low"][i - dur : i + dur + 1]
            current_min = float(low_window.min())
            sup_buf = (
                sup_buf + [current_min]
                if (not sup_buf or sup_buf[-1] == current_min)
                else [current_min]
            )
            if len(sup_buf) >= dur and is_new_level(current_min):
                levels.append((current_min, "support"))
                sup_buf = []

        mid_body = get_mid_body(df)
        matched_levels = []
        matched_types = []

        for price in mid_body:
            # find all levels within tolerance
            close_levels = [(lvl, typ) for lvl, typ in levels if abs(price - lvl) <= price * 0.05]

            if close_levels:
                nearest = min(close_levels, key=lambda x: abs(price - x[0]))
                matched_levels.append(nearest[0])
                matched_types.append(nearest[1])
            else:
                matched_levels.append(np.nan)
                matched_types.append(np.nan)

        return pd.Series(matched_levels, index=df.index), pd.Series(matched_types, index=df.index)
