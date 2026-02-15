import numpy as np
import pandas as pd


def _check_nan(value, name: str) -> tuple[bool, tuple[float, str]]:
    """Check if value is NaN and return neutral score if so."""
    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
        return True, (0.5, f"{name}: Data unavailable")
    return False, (0.5, "")


def get_rsi_score(current: pd.DataFrame, previous: pd.DataFrame) -> tuple[float, str]:
    rsi = current["rsi"]
    rsi_prev = previous["rsi"]

    # Check for NaN values
    is_nan, result = _check_nan(rsi, "RSI")
    if is_nan:
        return result

    # trend
    rising = rsi_prev < rsi
    falling = rsi_prev > rsi

    if rsi < 30:
        score, label = (1.0, "Strong Bullish") if rising else (0.8, "Bullish")
    elif rsi < 40:
        score, label = (0.7, "Bullish") if rising else (0.5, "Neutral")
    elif rsi < 60:
        score, label = (0.5, "Neutral")
    elif rsi < 70:
        score, label = (0.3, "Bearish") if falling else (0.5, "Neutral")
    else:
        score, label = (0.2, "Bearish") if falling else (0.0, "Strong Bearish")

    reason = f"RSI {rsi:.1f} : {label}"
    return score, reason


def get_macd_score(current: pd.DataFrame, previous: pd.DataFrame) -> tuple[float, str]:

    macd = current["macd"]
    macd_prev = previous["macd"]
    signal = current["macd_signal"]
    signal_prev = previous["macd_signal"]
    hist = current["macd_hist"]
    hist_prev = previous["macd_hist"]

    # Check for NaN values
    for val, name in [(macd, "MACD"), (signal, "MACD Signal"), (hist, "MACD Histogram")]:
        is_nan, result = _check_nan(val, name)
        if is_nan:
            return result

    # indicators
    diff = macd - signal
    diff_prev = macd_prev - signal_prev
    diff_delta = diff - diff_prev

    hist_delta = hist - hist_prev

    # primary bullish / bearish signals
    macd_above = diff > 0
    macd_cross_up = (diff_prev <= 0) and (diff > 0)
    macd_cross_down = (diff_prev >= 0) and (diff < 0)

    # scoring logic (simple rule set)
    if macd_cross_up and hist > 0 and hist_delta > 0:
        score, label = 1.0, "Strong Bullish (cross + ^hist)"
    elif macd_above and diff_delta > 0 and hist > 0:
        score, label = 0.8, "Bullish (MACD > Signal)"
    elif macd_above and hist > 0:
        score, label = 0.65, "Mild Bullish (MACD > Signal, hist+)"
    elif macd_cross_down and hist < 0 and hist_delta < 0:
        score, label = 0.0, "Strong Bearish (cross + vhist)"
    elif (not macd_above) and diff_delta < 0 and hist < 0:
        score, label = 0.2, "Bearish (MACD < Signal)"
    else:  # fallback neutral-ish grade influenced by histogram sign
        if hist > 0:
            score, label = 0.55, "Neutral Bullish"
        elif hist < 0:
            score, label = 0.45, "Neutral Bearish"
        else:
            score, label = 0.5, "Neutral"

    reason = f"prev_diff={diff_prev:.4f}, curr_diff={diff:.4f}, hist_delta={hist_delta:.4f} : {label}"
    return score, reason


def get_obv_slope_score(
    current: pd.DataFrame, previous: pd.DataFrame
) -> tuple[float, str]:

    obv = current["obv_slope"]
    obv_prev = previous["obv_slope"]

    # Check for NaN values
    is_nan, result = _check_nan(obv, "OBV Slope")
    if is_nan:
        return result

    # small tolerance to treat near-zero as zero
    eps = 1e-9
    if abs(obv) <= eps:
        score, label = 0.5, "Neutral"
    elif obv > 0:  # Positive slope = accumulation (buyers)
        score, label = (1.0, "Strong Bullish") if obv > obv_prev else (0.8, "Bullish")
    else:  # obv < 0: # Negative slope = distribution (sellers)
        score, label = (0.0, "Strong Bearish") if obv < obv_prev else (0.2, "Bearish")

    reason = f"OBV_slope {obv:.3f} : {label}"
    return score, reason
