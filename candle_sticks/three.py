import pandas as pd

from candle_sticks.conditions import (
    closes_above_prev_close,
    closes_below_prev_close,
    is_bearish,
    is_body_longer,
    is_bullish,
    is_downtrend,
    is_uptrend,
)
from candle_sticks.helper import get_mid_body
from candle_sticks.single import SingleCandlePatterns


def is_morning_star(df: pd.DataFrame) -> pd.Series:
    """Morning Star."""
    prev = df.shift(1)
    prev_2 = df.shift(2)
    mid_body = get_mid_body(prev_2)
    condition = (
        is_downtrend(df)
        & is_bullish(df)
        & is_bearish(prev_2)
        & (prev["High"] < mid_body)
        & (df["Close"] > mid_body)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition & SingleCandlePatterns.is_spinning_top(prev)] = 1.0
    rating.loc[condition & SingleCandlePatterns.is_doji(prev)] = 1.5
    rating.loc[condition & SingleCandlePatterns.is_inverted_hammer(prev)] = 2.0
    rating.loc[condition & SingleCandlePatterns.is_hammer(prev)] = 2.5
    rating.loc[condition & SingleCandlePatterns.is_dragonfly_doji(prev)] = 3.0

    return rating


def is_evening_star(df: pd.DataFrame) -> pd.Series:
    """Evening Star."""
    prev = df.shift(1)
    prev_2 = df.shift(2)
    mid_body = get_mid_body(prev_2)
    condition = (
        is_uptrend(df)
        & is_bearish(df)
        & is_bullish(prev_2)
        & (prev["Low"] > mid_body)
        & (df["Close"] < mid_body)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition & SingleCandlePatterns.is_spinning_top(prev)] = -1.0
    rating.loc[condition & SingleCandlePatterns.is_doji(prev)] = -1.5
    rating.loc[condition & SingleCandlePatterns.is_hanging_man(prev)] = -2.0
    rating.loc[condition & SingleCandlePatterns.is_shooting_star(prev)] = -2.5
    rating.loc[condition & SingleCandlePatterns.is_gravestone_doji(prev)] = -3.0

    return rating


def is_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Three White Soldiers."""
    prev = df.shift(1)
    prev_2 = df.shift(2)
    condition = (
        is_downtrend(df)
        & is_bullish(df)
        & is_bullish(prev)
        & is_bullish(prev_2)
        & closes_above_prev_close(df, prev)
        & closes_above_prev_close(prev, prev_2)
    )
    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = 1.0
    rating.loc[
        condition & is_body_longer(df) & is_body_longer(prev) & is_body_longer(prev_2)
    ] = 1.5
    rating.loc[
        condition
        & is_body_longer(df, 0.9)
        & is_body_longer(prev, 0.9)
        & is_body_longer(prev_2, 0.9)
    ] = 2.0

    return rating


def is_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Three Black Crows."""
    prev = df.shift(1)
    prev_2 = df.shift(2)
    condition = (
        is_uptrend(df)
        & is_bearish(df)
        & is_bearish(prev)
        & is_bearish(prev_2)
        & closes_below_prev_close(df, prev)
        & closes_below_prev_close(prev, prev_2)
    )

    rating = pd.Series(0.0, index=df.index)
    rating.loc[condition] = -1.0
    rating.loc[
        condition & is_body_longer(df) & is_body_longer(prev) & is_body_longer(prev_2)
    ] = -1.5
    rating.loc[
        condition
        & is_body_longer(df, 0.9)
        & is_body_longer(prev, 0.9)
        & is_body_longer(prev_2, 0.9)
    ] = -2.0

    return rating
