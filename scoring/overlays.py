import numpy as np
import pandas as pd


def _check_nan(value, name: str) -> tuple[bool, tuple[float, str]]:
    """Check if value is NaN and return neutral score if so."""
    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
        return True, (0.5, f"{name}: Data unavailable")
    return False, (0.5, "")


def get_ma_cross_score(
    current: pd.DataFrame, previous: pd.DataFrame
) -> tuple[float, str]:

    ma50 = current["ma50"]
    ma50_prev = previous["ma50"]
    ma200 = current["ma200"]
    ma200_prev = previous["ma200"]

    # Check for NaN values
    for val, name in [(ma50, "MA50"), (ma200, "MA200")]:
        is_nan, result = _check_nan(val, name)
        if is_nan:
            return result

    # indicators
    prev_diff = ma50_prev - ma200_prev
    curr_diff = ma50 - ma200
    diff_delta = curr_diff - prev_diff

    crossed_up = (prev_diff <= 0) and (curr_diff > 0)
    crossed_down = (prev_diff >= 0) and (curr_diff < 0)

    if crossed_up:
        score, label = 1.0, "Strong Bullish (golden cross)"
    elif ma50 > ma200:
        # bullish region; stronger if separation increasing
        score, label = (0.9, "Bullish") if diff_delta > 0 else (0.75, "Mild Bullish ")
    elif crossed_down:
        score, label = 0.0, "Strong Bearish (death cross)"
    else:  # ma50 < ma200
        # bearish region; stronger if separation reducing
        score, label = (0.1, "Bearish") if diff_delta < 0 else (0.3, "Mild Bearish")

    reason = f"MA50 {ma50:.2f} vs MA200 {ma200:.2f} : {label}"
    return score, reason


def get_bb_score(current: pd.DataFrame, previous: pd.DataFrame) -> tuple[float, str]:

    close = current["Close"]
    close_prev = previous["Close"]

    upper = current["bb_upper"]
    middle = current["bb_mid"]
    lower = current["bb_lower"]

    # Check for NaN values
    for val, name in [(upper, "BB Upper"), (middle, "BB Mid"), (lower, "BB Lower")]:
        is_nan, result = _check_nan(val, name)
        if is_nan:
            return result

    # location classification
    if close < lower:
        if close > close_prev:
            score, label = 1.0, "Strong Bullish"
        else:
            score, label = 0.9, "Bullish (<lower)"
    elif lower <= close < middle:
        if close > close_prev:
            score, label = 0.85, "Bullish (lower<->middle, rising)"
        else:
            score, label = 0.7, "Bullish (lower<->middle)"
    elif middle <= close <= upper:
        # interpolate neutrality: closer to middle => 0.55, closer to upper => 0.4
        ratio = (close - middle) / (upper - middle) if upper != middle else 0.5
        score = 0.55 - 0.15 * ratio
        label = "Neutral (middle<->upper)"
    else:  # close > upper
        if close < close_prev:
            score, label = 0.3, "Bearish (>upper)"
        else:
            score, label = 0.0, "Strong Bearish"

    reason = f"BB (L:{lower:.2f}, M:{middle:.2f}, U:{upper:.2f} : {label}"
    return score, reason


def get_volume_sma20_score(
    current: pd.DataFrame, previous: pd.DataFrame
) -> tuple[float, str]:

    vol = current["Volume"]
    vol_prev = previous["Volume"]
    vol_sma20 = current["volume_sma20"]

    # Check for NaN values
    for val, name in [(vol, "Volume"), (vol_sma20, "Volume SMA20")]:
        is_nan, result = _check_nan(val, name)
        if is_nan:
            return result

    upper_thresh = vol_sma20 * 1.2
    lower_thresh = vol_sma20 * 0.8

    if vol >= upper_thresh:
        score, label = 1.0, "Strong Bullish"
    elif vol > vol_sma20:
        score, label = (0.85, "Bullish(+)") if vol > vol_prev else (0.7, "Bullish(-)")
    elif lower_thresh <= vol <= vol_sma20:
        score, label = (0.55, "Mild Bullish") if vol > vol_prev else (0.5, "Neutral")
    else:  # volume < lower_thresh
        if vol > vol_prev:
            score, label = 0.4, "Mild Bearish"
        else:
            score, label = 0.0, "Strong Bearish"

    reason = f"Volume {vol:.0f} vs Volume_SMA20 {vol_sma20:.0f} : {label}"
    return score, reason
