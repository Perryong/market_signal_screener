import numpy as np
import pandas as pd


def get_candlestick_score(df: pd.DataFrame) -> tuple[float, str]:
    lookback_days = 30
    decay_factor = 0.95  # each day back is multiplied by decay_factor^(days_ago)

    candle_score = 0.0
    pattern_types = df.attrs.get("pattern_types", {})
    single_pattern = ""
    double_pattern = ""
    triple_pattern = ""

    for i in range(1, lookback_days + 1):
        if i > len(df):
            break
        row = df.iloc[-i]
        days_ago = i - 1
        weight_multiplier = decay_factor**days_ago

        for col in df.columns:
            if col in pattern_types:
                signal_value = row[col]
                if signal_value == 0:
                    continue  # skip no-signal candles

                if pattern_types[col] == "one":
                    single_pattern = col
                elif pattern_types[col] == "two":
                    double_pattern = col
                elif pattern_types[col] == "three":
                    triple_pattern = col

                # Determine polarity based on column name
                if "bull" in col:
                    candle_score += signal_value * weight_multiplier
                elif "bear" in col:
                    candle_score -= signal_value * weight_multiplier

    score = np.clip(candle_score / 6, -1.0, 1.0)

    if score > 0.7:
        trend = "Strong Bullish"
    elif score < -0.7:
        trend = "Strong Bearish"
    else:
        trend = "Neutral"

    label = (
        triple_pattern
        if triple_pattern
        else double_pattern if double_pattern else single_pattern
    )

    reason = f"{trend} Candlesticks {score:.2f} : {label}"
    return score, reason
