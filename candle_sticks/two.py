import pandas as pd

from candle_sticks.conditions import (
    closes_above_prev_open,
    closes_below_prev_open,
    current_bear_prev_bull,
    current_bull_prev_bear,
    is_body_longer,
    is_downtrend,
    is_uptrend,
    opens_above_prev_close,
    opens_below_prev_close,
)
from candle_sticks.helper import get_candle_body, get_mid_body


def _is_bullish_trend(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return is_downtrend(df) & current_bull_prev_bear(df, prev)


def _is_bearish_trend(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return is_uptrend(df) & current_bear_prev_bull(df, prev)


def _is_body_longer(df: pd.DataFrame, prev: pd.DataFrame) -> pd.Series:
    return is_body_longer(df) & is_body_longer(prev)


def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish Engulfing."""
    prev = df.shift(1)
    condition = (
        _is_bullish_trend(df, prev)
        & closes_above_prev_open(df, prev)
        & opens_below_prev_close(df, prev)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = 0.5
    rating.loc[condition & _is_body_longer(df, prev)] = 1.0

    return rating


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish Engulfing."""
    prev = df.shift(1)
    condition = (
        _is_bearish_trend(df, prev)
        & closes_below_prev_open(df, prev)
        & opens_above_prev_close(df, prev)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = -0.5
    rating.loc[condition & _is_body_longer(df, prev)] = -1.0

    return rating


def is_bullish_harami(df: pd.DataFrame) -> pd.Series:
    """Bullish Harami."""
    prev = df.shift(1)
    condition = (
        _is_bullish_trend(df, prev)
        & closes_below_prev_open(df, prev)
        & opens_above_prev_close(df, prev)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = 0.5
    rating.loc[condition & _is_body_longer(df, prev)] = 1.0

    return rating


def is_bearish_harami(df: pd.DataFrame) -> pd.Series:
    """Bearish Harami."""
    prev = df.shift(1)
    condition = (
        _is_bearish_trend(df, prev)
        & closes_above_prev_open(df, prev)
        & opens_below_prev_close(df, prev)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = -0.5
    rating.loc[condition & _is_body_longer(df, prev)] = -1.0

    return rating


def is_tweezer_bottom(df: pd.DataFrame, tol: float = 0.005) -> pd.Series:
    """Tweezer Bottom."""
    prev = df.shift(1)
    condition = (
        _is_bullish_trend(df, prev)
        & _is_body_longer(df, prev)
        & (abs(df["Low"] - prev["Low"]) <= df["Low"] * tol)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = 0.5
    rating.loc[condition & get_candle_body(df) > get_candle_body(prev)] = 1.0

    return rating


def is_tweezer_top(df: pd.DataFrame, tol: float = 0.005) -> pd.Series:
    """Tweezer Top."""
    prev = df.shift(1)
    condition = (
        _is_bearish_trend(df, prev)
        & _is_body_longer(df, prev)
        & (abs(df["High"] - prev["High"]) <= df["High"] * tol)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = -0.5
    rating.loc[condition & get_candle_body(df) > get_candle_body(prev)] = -1.0

    return rating


def is_rising_sun(df: pd.DataFrame) -> pd.Series:
    """Rising Sun."""
    prev = df.shift(1)
    mid_body = get_mid_body(prev)
    condition = (
        _is_bullish_trend(df, prev)
        & closes_below_prev_open(df, prev)
        & (df["Open"] < prev["High"])
        & (df["Close"] > mid_body)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = 1.0
    rating.loc[condition & _is_body_longer(df, prev)] = 1.0

    return rating


def is_dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """Dark Cloud Cover."""
    prev = df.shift(1)
    mid_body = get_mid_body(prev)
    condition = (
        _is_bearish_trend(df, prev)
        & closes_above_prev_open(df, prev)
        & (df["Open"] > prev["High"])
        & (df["Close"] < mid_body)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = -0.5
    rating.loc[condition & _is_body_longer(df, prev)] = -1.0

    return rating
