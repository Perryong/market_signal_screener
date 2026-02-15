import pandas as pd

from candle_sticks.helper import get_candle_body, get_price_range, safe_divide


def is_body_longer(df: pd.DataFrame, body_percentage: float = 0.6) -> pd.Series:
    return safe_divide(get_candle_body(df), get_price_range(df)) > body_percentage


def is_downtrend(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    return df["Close"] < df["Close"].shift(lookback)


def is_uptrend(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    return df["Close"] > df["Close"].shift(lookback)


def is_bullish(df: pd.DataFrame) -> pd.Series:
    return df["Close"] > df["Open"]


def is_bearish(df: pd.DataFrame) -> pd.Series:
    return df["Close"] < df["Open"]


def current_bull_prev_bear(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return is_bullish(df) & is_bearish(prev)


def current_bear_prev_bull(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return is_bearish(df) & is_bullish(prev)


def opens_below_prev_close(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Open"] < prev["Close"]


def opens_above_prev_close(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Open"] > prev["Close"]


def opens_below_prev_open(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Open"] < prev["Open"]


def opens_above_prev_open(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Open"] > prev["Open"]


def closes_above_prev_open(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Close"] > prev["Open"]


def closes_below_prev_open(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Close"] < prev["Open"]


def closes_above_prev_close(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Close"] > prev["Close"]


def closes_below_prev_close(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return df["Close"] < prev["Close"]
