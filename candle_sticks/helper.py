import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two series, replacing division by zero with fill_value.

    Args:
        numerator: The numerator series
        denominator: The denominator series
        fill_value: Value to use when denominator is zero (default: 0.0)

    Returns:
        Series with safe division results
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(fill_value)


def get_price_range(df: pd.DataFrame) -> pd.Series:
    return df["High"] - df["Low"]


def get_candle_body(df: pd.DataFrame) -> pd.Series:
    return abs(df["Close"] - df["Open"])


def get_mid_body(df: pd.DataFrame) -> pd.Series:
    return (df["Open"] + df["Close"]) / 2


def get_lower_wick(df: pd.DataFrame) -> pd.Series:
    return np.minimum(df["Open"], df["Close"]) - df["Low"]


def get_upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["High"] - np.maximum(df["Open"], df["Close"])
