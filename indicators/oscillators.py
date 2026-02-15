import numpy as np
import pandas as pd
from typing import Tuple


class Oscillators:

    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Relative Strength Index (RSI) using exponential smoothing."""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        ma_up = up.ewm(alpha=1 / length, adjust=False).mean()
        ma_down = down.ewm(alpha=1 / length, adjust=False).mean()

        rs = ma_up / ma_down.replace(0, np.nan)  # avoid div by zero
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val.fillna(50)  # default to neutral

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9):
        """MACD line, Signal line, Histogram."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def on_balance_volume_slope(df: pd.DataFrame) -> pd.Series:
        """
        On Balance Volume Slope: Who's in control (Buyers vs Sellers).
        Vectorized implementation for performance.
        """
        close = df["Close"]
        volume = df["Volume"]

        # Vectorized OBV calculation
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()

        # Return the slope (3-period rolling mean of diff)
        return obv.diff().rolling(3).mean().fillna(0)

    @staticmethod
    def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """Average True Range (ATR)."""
        prev_close = df["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - prev_close).abs()
        tr3 = (df["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / length, adjust=False).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX).
        Returns only the ADX value. Use adx_with_di for +DI/-DI values.
        """
        adx_val, _, _ = Oscillators.adx_with_di(df, period)
        return adx_val

    @staticmethod
    def adx_with_di(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (ADX) with +DI and -DI.

        Returns:
            Tuple of (ADX, +DI, -DI) Series
        """
        # Calculate directional movements
        plus_diff = df["High"].diff()
        minus_diff = df["Low"].diff().abs()

        plus_dm = np.where((plus_diff > minus_diff) & (plus_diff > 0), plus_diff, 0.0)
        minus_dm = np.where(
            (minus_diff > plus_diff) & (df["Low"].shift() - df["Low"] > 0),
            df["Low"].shift() - df["Low"],
            0.0,
        )

        atr_val = Oscillators.atr(df)

        # Calculate +DI and -DI
        plus_di = 100 * (
            pd.Series(plus_dm, index=df.index).rolling(window=period).sum() / atr_val
        )
        minus_di = 100 * (
            pd.Series(minus_dm, index=df.index).rolling(window=period).sum() / atr_val
        )

        # Calculate DX with safe division (replace 0 with NaN to avoid division by zero)
        di_sum = (plus_di + minus_di).abs()
        di_diff = (plus_di - minus_di).abs()
        dx = (di_diff / di_sum.replace(0, np.nan)) * 100

        # ADX (smoothed DX)
        adx_val = dx.rolling(window=period).mean()

        return adx_val, plus_di, minus_di

    @staticmethod
    def rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """
        Detect RSI divergence.

        Returns:
            Series with values:
                1 = bullish divergence (price lower low, RSI higher low)
               -1 = bearish divergence (price higher high, RSI lower high)
                0 = no divergence
        """
        close = df["Close"]
        rsi = Oscillators.rsi(close, lookback)

        divergence = pd.Series(0, index=df.index)

        for i in range(lookback * 2, len(df)):
            window_start = i - lookback

            # Get price and RSI values in window
            price_window = close.iloc[window_start:i + 1]
            rsi_window = rsi.iloc[window_start:i + 1]

            # Find local minima and maxima
            current_price = price_window.iloc[-1]
            current_rsi = rsi_window.iloc[-1]

            min_price_idx = price_window.idxmin()
            max_price_idx = price_window.idxmax()
            min_price = price_window.loc[min_price_idx]
            max_price = price_window.loc[max_price_idx]

            # Bullish divergence: price makes lower low, RSI makes higher low
            if min_price_idx != price_window.index[-1]:
                prev_rsi_at_min = rsi.loc[min_price_idx]
                if current_price < min_price and current_rsi > prev_rsi_at_min:
                    divergence.iloc[i] = 1

            # Bearish divergence: price makes higher high, RSI makes lower high
            if max_price_idx != price_window.index[-1]:
                prev_rsi_at_max = rsi.loc[max_price_idx]
                if current_price > max_price and current_rsi < prev_rsi_at_max:
                    divergence.iloc[i] = -1

        return divergence

    @staticmethod
    def macd_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """
        Detect MACD histogram divergence.

        Returns:
            Series with values:
                1 = bullish divergence (price lower low, MACD histogram higher low)
               -1 = bearish divergence (price higher high, MACD histogram lower high)
                0 = no divergence
        """
        close = df["Close"]
        _, _, hist = Oscillators.macd(close)

        divergence = pd.Series(0, index=df.index)

        for i in range(lookback * 2, len(df)):
            window_start = i - lookback

            # Get price and MACD histogram values in window
            price_window = close.iloc[window_start:i + 1]
            hist_window = hist.iloc[window_start:i + 1]

            # Find local minima and maxima
            current_price = price_window.iloc[-1]
            current_hist = hist_window.iloc[-1]

            min_price_idx = price_window.idxmin()
            max_price_idx = price_window.idxmax()
            min_price = price_window.loc[min_price_idx]
            max_price = price_window.loc[max_price_idx]

            # Bullish divergence: price makes lower low, MACD hist makes higher low
            if min_price_idx != price_window.index[-1]:
                prev_hist_at_min = hist.loc[min_price_idx]
                if current_price < min_price and current_hist > prev_hist_at_min:
                    divergence.iloc[i] = 1

            # Bearish divergence: price makes higher high, MACD hist makes lower high
            if max_price_idx != price_window.index[-1]:
                prev_hist_at_max = hist.loc[max_price_idx]
                if current_price > max_price and current_hist < prev_hist_at_max:
                    divergence.iloc[i] = -1

        return divergence
