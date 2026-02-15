import numpy as np
import pandas as pd

from config import (
    INDICATOR_WEIGHTS,
    SIGNAL_THRESHOLDS,
    ADX_THRESHOLDS,
    ADX_FACTORS,
    SNR_FACTORS,
    ENTRY_RANGE_MULTIPLIERS,
    CANDLESTICK_CONFIG,
)
from data_types.signal import Signal
from scoring.candlesticks import get_candlestick_score
from scoring.oscillators import get_macd_score, get_obv_slope_score, get_rsi_score
from scoring.overlays import get_bb_score, get_ma_cross_score, get_volume_sma20_score


def generate_signal(df: pd.DataFrame, buy_threshold: float = None, sell_threshold: float = None) -> Signal:
    """
    Generate trading signal from technical indicators.

    Args:
        df: DataFrame with OHLCV data and computed indicators
        buy_threshold: Override for buy threshold (default from config)
        sell_threshold: Override for sell threshold (default from config)

    Returns:
        Signal object with trading recommendation
    """
    reasons = []

    # Use config defaults if not specified
    buy_threshold = buy_threshold if buy_threshold is not None else SIGNAL_THRESHOLDS["buy"]
    sell_threshold = sell_threshold if sell_threshold is not None else SIGNAL_THRESHOLDS["sell"]

    # Validate DataFrame has enough data
    if df.empty or len(df) < 2:
        raise ValueError(f"DataFrame is empty or has insufficient data (length: {len(df)})")

    # Validate required columns exist
    required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    current = df.iloc[-1]
    previous = df.iloc[-2]

    # -----------------------------
    # Oscillators
    rsi_score, rsi_reason = get_rsi_score(current, previous)
    macd_score, macd_reason = get_macd_score(current, previous)
    obv_score, obv_reason = get_obv_slope_score(current, previous)

    # -----------------------------
    # Overlays
    ma_cross_score, ma_cross_reason = get_ma_cross_score(current, previous)
    bollinger_score, bollinger_reason = get_bb_score(current, previous)
    vol_sma20_score, vol_sma20_reason = get_volume_sma20_score(current, previous)

    # -----------------------------
    # Candlesticks patterns
    candle_score, candle_reason = get_candlestick_score(df)

    # -----------------------------
    # ATR-based expected move (with improved entry range)
    atr_val = float(current["atr"])
    last_close = float(current["Close"])
    # More patient pullback, less chase
    entry_low = last_close - ENTRY_RANGE_MULTIPLIERS["pullback"] * atr_val
    entry_high = last_close + ENTRY_RANGE_MULTIPLIERS["chase"] * atr_val

    # -----------------------------
    # ADX adjustment (0–100 → 0–1 strength)
    adx_val = float(current["adx"]) if not np.isnan(current["adx"]) else 25.0
    adx_strength = adx_val / 100

    # Get +DI/-DI for directional confirmation
    plus_di = float(current.get("plus_di", 0)) if "plus_di" in current and not np.isnan(current.get("plus_di", np.nan)) else 0
    minus_di = float(current.get("minus_di", 0)) if "minus_di" in current and not np.isnan(current.get("minus_di", np.nan)) else 0
    trend_bullish = plus_di > minus_di
    trend_bearish = minus_di > plus_di

    # -----------------------------
    # Winshift SNR adjustment
    # Initialize snr_strength before the if block to avoid undefined variable
    snr_strength = 0

    near_snr = current["snr"]
    snr_type = current.get("snr_type", np.nan)  # "support" or "resistance"

    if not np.isnan(near_snr):
        tolerance = last_close * 0.05
        dist = abs(last_close - near_snr)
        snr_strength = max(0, 1 - dist / tolerance)

    # -----------------------------
    # Divergence bonuses
    rsi_div = current.get("rsi_divergence", 0)
    macd_div = current.get("macd_divergence", 0)
    divergence_bonus = 0

    if rsi_div == 1 or macd_div == 1:  # Bullish divergence
        divergence_bonus = 0.1
        reasons.append("Bullish divergence detected")
    elif rsi_div == -1 or macd_div == -1:  # Bearish divergence
        divergence_bonus = -0.1
        reasons.append("Bearish divergence detected")

    # -----------------------------
    # Calculate base score with updated weights
    base_score = (
        rsi_score * INDICATOR_WEIGHTS["rsi"]
        + macd_score * INDICATOR_WEIGHTS["macd"]
        + obv_score * INDICATOR_WEIGHTS["obv"]
        + ma_cross_score * INDICATOR_WEIGHTS["ma_cross"]
        + bollinger_score * INDICATOR_WEIGHTS["bollinger"]
        + vol_sma20_score * INDICATOR_WEIGHTS["volume"]
        + candle_score * INDICATOR_WEIGHTS["candlestick"]
    )

    # Add divergence bonus
    base_score += divergence_bonus

    # Apply ADX multiplier
    if adx_val < ADX_THRESHOLDS["weak"]:
        adx_factor = ADX_FACTORS["weak_trend"]
    elif adx_val > ADX_THRESHOLDS["strong"]:
        adx_factor = ADX_FACTORS["strong_trend"]
    else:
        adx_factor = ADX_FACTORS["normal_trend"]

    # Apply SNR multiplier with type awareness
    if np.isnan(near_snr):
        snr_factor = SNR_FACTORS["no_level"]
    elif snr_strength > 0.7:
        # Boost BUY near support, SELL near resistance
        if snr_type == "support":
            snr_factor = SNR_FACTORS["near_level"]  # Good for buying
            if base_score > 0.5:  # Bullish bias
                snr_factor = 1.3  # Extra boost
        elif snr_type == "resistance":
            snr_factor = SNR_FACTORS["near_level"]  # Could be reversal point
            if base_score < 0.5:  # Bearish bias
                snr_factor = 1.3  # Extra boost for sell
        else:
            snr_factor = SNR_FACTORS["near_level"]
    else:
        snr_factor = SNR_FACTORS["far_from_level"]

    # Apply +DI/-DI directional filter
    di_factor = 1.0
    if adx_val > ADX_THRESHOLDS["weak"]:  # Only apply in trending conditions
        if trend_bullish and base_score > 0.5:
            di_factor = 1.1  # Confirm bullish
        elif trend_bearish and base_score < 0.5:
            di_factor = 1.1  # Confirm bearish
        elif trend_bullish and base_score < 0.5:
            di_factor = 0.9  # Counter-trend signal
        elif trend_bearish and base_score > 0.5:
            di_factor = 0.9  # Counter-trend signal

    adjusted_score = base_score * adx_factor * snr_factor * di_factor

    # Clamp final score to 0–1
    bullish_score = min(1.0, max(0.0, adjusted_score))

    # -----------------------------
    # Collect reasons
    reasons.extend(
        [
            rsi_reason,
            macd_reason,
            obv_reason,
            ma_cross_reason,
            bollinger_reason,
            vol_sma20_reason,
            candle_reason,
        ]
    )
    if adx_strength > 0.6:
        reasons.append(f"Strong trend (ADX={adx_val:.1f})")
    if not np.isnan(near_snr):
        snr_type_str = f" ({snr_type})" if snr_type and not pd.isna(snr_type) else ""
        reasons.append(f"Near S/R level at {near_snr:.2f}{snr_type_str}")
    if plus_di > 0 and minus_di > 0:
        trend_dir = "bullish" if trend_bullish else "bearish"
        reasons.append(f"Trend direction: {trend_dir} (+DI={plus_di:.1f}, -DI={minus_di:.1f})")

    # -----------------------------
    # Final decision based on updated thresholds
    if bullish_score >= buy_threshold:
        signal = "BUY"
    elif bullish_score <= sell_threshold:
        signal = "SELL"
    else:
        if candle_score > CANDLESTICK_CONFIG["strong_threshold"]:
            signal = "BUY"
        elif candle_score < -CANDLESTICK_CONFIG["strong_threshold"]:
            signal = "SELL"
        else:
            signal = "HOLD"

    return Signal(
        ticker=df.attrs.get("TICKER", "UNK"),
        signal=signal,
        reasons=reasons,
        entry_range=(round(entry_low, 4), round(entry_high, 4)),
        last_close=round(last_close, 4),
        atr=round(atr_val, 6),
    )
