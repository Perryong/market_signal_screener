import pandas as pd
from candle_sticks.conditions import is_downtrend, is_uptrend
from candle_sticks.single import SingleCandlePatterns
from candle_sticks.three import (
    is_evening_star,
    is_morning_star,
    is_three_black_crows,
    is_three_white_soldiers,
)
from candle_sticks.two import (
    is_bearish_engulfing,
    is_bearish_harami,
    is_bullish_engulfing,
    is_bullish_harami,
    is_dark_cloud_cover,
    is_rising_sun,
    is_tweezer_bottom,
    is_tweezer_top,
)


def prepare_candle_sticks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # Single Candlestick Patterns
    bullish_marubozu = SingleCandlePatterns.is_bullish_marubozu(df)
    bull_marubozu = pd.Series(0.0, index=df.index)
    bull_marubozu.loc[bullish_marubozu] = 0.5
    bull_marubozu.loc[bullish_marubozu & is_downtrend(df)] = 1.0
    df["bull_marubozu"] = bull_marubozu

    bearish_marubozu = SingleCandlePatterns.is_bearish_marubozu(df)
    bear_marubozu = pd.Series(0.0, index=df.index)
    bear_marubozu.loc[bearish_marubozu] = -0.5
    bear_marubozu.loc[bearish_marubozu & is_uptrend(df)] = -1.0
    df["bear_marubozu"] = bear_marubozu

    # -----------------------------
    # Two Candlestick Patterns
    df["bull_engulfing"] = is_bullish_engulfing(df)
    df["bear_engulfing"] = is_bearish_engulfing(df)
    df["bull_harami"] = is_bullish_harami(df)
    df["bear_harami"] = is_bearish_harami(df)
    df["bull_tweezer"] = is_tweezer_bottom(df)
    df["bear_tweezer"] = is_tweezer_top(df)
    df["bull_rising_sun"] = is_rising_sun(df)
    df["bear_dark_cloud"] = is_dark_cloud_cover(df)

    # -----------------------------
    # Three Candlestick Patterns
    df["bull_morning_star"] = is_morning_star(df)
    df["bear_evening_star"] = is_evening_star(df)
    df["bull_3_white"] = is_three_white_soldiers(df)
    df["bear_3_black"] = is_three_black_crows(df)

    df.attrs["pattern_types"] = {
        "bull_marubozu": "one",
        "bear_marubozu": "one",
        "bull_engulfing": "two",
        "bear_engulfing": "two",
        "bull_harami": "two",
        "bear_harami": "two",
        "bull_tweezer": "two",
        "bear_tweezer": "two",
        "bull_rising_sun": "two",
        "bear_dark_cloud": "two",
        "bull_morning_star": "three",
        "bear_evening_star": "three",
        "bull_3_white": "three",
        "bear_3_black": "three",
    }

    return df
