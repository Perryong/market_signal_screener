import numpy as np
import pandas as pd
import yfinance as yf
import os

from candle_sticks.index import prepare_candle_sticks
from engine import generate_signal
from indicators.index import prepare_indicators
from visualizations import plot_comprehensive_analysis


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame with indicators and candlestick patterns."""
    df = df.reset_index()
    df = prepare_indicators(df)
    return prepare_candle_sticks(df)


def generate_signals_backtest(
    df: pd.DataFrame, buy_threshold: float, sell_threshold: float
) -> pd.DataFrame:
    """
    Generate bullish_score and signal for each row in the dataframe.

    Uses the engine.generate_signal() function with configurable thresholds
    for grid search optimization.

    Args:
        df: DataFrame with OHLCV data and computed indicators
        buy_threshold: Threshold for BUY signal (0-1)
        sell_threshold: Threshold for SELL signal (0-1)

    Returns:
        DataFrame with 'bullish_score' and 'signal' columns added
    """
    df = df.copy()
    df["bullish_score"] = np.nan
    df["signal"] = "HOLD"

    # We need at least 2 rows for signal generation
    if len(df) < 2:
        return df

    # Generate signals for each row starting from index 1
    for i in range(1, len(df)):
        try:
            # Create a slice of the DataFrame up to and including current row
            df_slice = df.iloc[:i + 1].copy()

            # Generate signal using engine with custom thresholds
            signal_result = generate_signal(
                df_slice,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold
            )

            # Extract the signal
            df.at[df.index[i], "signal"] = signal_result.signal

            # Calculate bullish_score for visualization
            from config import (
                INDICATOR_WEIGHTS,
                ADX_THRESHOLDS,
                ADX_FACTORS,
                SNR_FACTORS,
            )
            from scoring.candlesticks import get_candlestick_score
            from scoring.oscillators import get_macd_score, get_obv_slope_score, get_rsi_score
            from scoring.overlays import get_bb_score, get_ma_cross_score, get_volume_sma20_score
            
            current = df_slice.iloc[-1]
            previous = df_slice.iloc[-2]
            
            # Get individual scores
            rsi_score, _ = get_rsi_score(current, previous)
            macd_score, _ = get_macd_score(current, previous)
            obv_score, _ = get_obv_slope_score(current, previous)
            ma_cross_score, _ = get_ma_cross_score(current, previous)
            bollinger_score, _ = get_bb_score(current, previous)
            vol_sma20_score, _ = get_volume_sma20_score(current, previous)
            candle_score, _ = get_candlestick_score(df_slice)
            
            # Divergence bonuses
            rsi_div = current.get("rsi_divergence", 0)
            macd_div = current.get("macd_divergence", 0)
            divergence_bonus = 0
            if rsi_div == 1 or macd_div == 1:
                divergence_bonus = 0.1
            elif rsi_div == -1 or macd_div == -1:
                divergence_bonus = -0.1
            
            # Calculate base score
            base_score = (
                rsi_score * INDICATOR_WEIGHTS["rsi"]
                + macd_score * INDICATOR_WEIGHTS["macd"]
                + obv_score * INDICATOR_WEIGHTS["obv"]
                + ma_cross_score * INDICATOR_WEIGHTS["ma_cross"]
                + bollinger_score * INDICATOR_WEIGHTS["bollinger"]
                + vol_sma20_score * INDICATOR_WEIGHTS["volume"]
                + candle_score * INDICATOR_WEIGHTS["candlestick"]
            )
            base_score += divergence_bonus
            
            # Apply ADX multiplier
            adx_val = float(current["adx"]) if not np.isnan(current["adx"]) else 25.0
            if adx_val < ADX_THRESHOLDS["weak"]:
                adx_factor = ADX_FACTORS["weak_trend"]
            elif adx_val > ADX_THRESHOLDS["strong"]:
                adx_factor = ADX_FACTORS["strong_trend"]
            else:
                adx_factor = ADX_FACTORS["normal_trend"]
            
            # Apply SNR multiplier
            near_snr = current["snr"]
            last_close = float(current["Close"])
            snr_strength = 0
            if not np.isnan(near_snr):
                tolerance = last_close * 0.05
                dist = abs(last_close - near_snr)
                snr_strength = max(0, 1 - dist / tolerance)
            
            if np.isnan(near_snr):
                snr_factor = SNR_FACTORS["no_level"]
            elif snr_strength > 0.7:
                snr_factor = SNR_FACTORS["near_level"]
            else:
                snr_factor = SNR_FACTORS["far_from_level"]
            
            # Apply +DI/-DI factor
            plus_di = float(current.get("plus_di", 0)) if "plus_di" in current and not np.isnan(current.get("plus_di", np.nan)) else 0
            minus_di = float(current.get("minus_di", 0)) if "minus_di" in current and not np.isnan(current.get("minus_di", np.nan)) else 0
            trend_bullish = plus_di > minus_di
            trend_bearish = minus_di > plus_di
            
            di_factor = 1.0
            if adx_val > ADX_THRESHOLDS["weak"]:
                if trend_bullish and base_score > 0.5:
                    di_factor = 1.1
                elif trend_bearish and base_score < 0.5:
                    di_factor = 1.1
                elif trend_bullish and base_score < 0.5:
                    di_factor = 0.9
                elif trend_bearish and base_score > 0.5:
                    di_factor = 0.9
            
            adjusted_score = base_score * adx_factor * snr_factor * di_factor
            bullish_score = min(1.0, max(0.0, adjusted_score))
            
            df.at[df.index[i], "bullish_score"] = bullish_score

        except (ValueError, KeyError) as e:
            # Skip rows with insufficient data or missing columns
            continue

    return df


sg_tickers = [
    "CRPU.SI",
    "J69U.SI",
    "BUOU.SI",
    "M44U.SI",
    "ME8U.SI",
    "JYEU.SI",
    "AJBU.SI",
    "DCRU.SI",
    "U11.SI",
    "C6L.SI",
    "CJLU.SI",
    "O39.SI",
]

us_tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "AMZN",
    "BABA",
    "V",
    "MA",
    "JPM",
    "JNJ",
    "META",
]

tickers = us_tickers

df_daily = yf.download(
    tickers, period="2y", interval="1d", group_by="ticker", auto_adjust=True
)
df_weekly = yf.download(
    tickers, period="5y", interval="1wk", group_by="ticker", auto_adjust=True
)

average_buy_range = 0
average_sell_range = 0

for ticker in tickers:
    df = prepare_df(df_daily[ticker])

    best_result = {
        "buy_range": 0.0,
        "sell_range": 1.0,
        "max equity": 0,
        "latest equity": 0,
    }

    for buy in np.linspace(0.6, 1.0, 9):
        for sell in np.linspace(0.4, 0.0, 9):
            if sell >= buy:
                continue

            df = generate_signals_backtest(df, buy, sell)

            capital = 100000
            position = 0
            equity_curve = []

            for i, row in df.iterrows():
                if row["signal"] == "BUY" and position == 0:
                    entry_price = row["Close"]
                    position = capital / entry_price
                elif row["signal"] == "SELL" and position > 0:
                    capital = position * row["Close"]
                    position = 0
                equity_curve.append(
                    capital if position == 0 else position * row["Close"]
                )

            df["equity"] = equity_curve
            max_equity = df["equity"].max()

            if best_result["max equity"] < max_equity:
                best_result = {
                    "buy_range": buy,
                    "sell_range": sell,
                    "max equity": max_equity,
                    "latest equity": df["equity"].iloc[-1],
                }
            elif best_result["max equity"] == max_equity:
                best_result = {
                    "buy_range": max(buy, best_result.get("buy_range", buy)),
                    "sell_range": min(sell, best_result.get("sell_range", sell)),
                    "max equity": max_equity,
                    "latest equity": df["equity"].iloc[-1],
                }
    
    # Generate final signals with best parameters for visualization
    df_final = prepare_df(df_daily[ticker].copy())
    df_final = generate_signals_backtest(
        df_final, 
        best_result["buy_range"], 
        best_result["sell_range"]
    )
    
    # Calculate equity curve for visualization
    capital = 100000
    position = 0
    equity_curve = []
    for i, row in df_final.iterrows():
        if row["signal"] == "BUY" and position == 0:
            entry_price = row["Close"]
            position = capital / entry_price
        elif row["signal"] == "SELL" and position > 0:
            capital = position * row["Close"]
            position = 0
        equity_curve.append(
            capital if position == 0 else position * row["Close"]
        )
    df_final["equity"] = equity_curve
    
    # Ensure Date column exists for visualization
    if 'Date' not in df_final.columns and df_final.index.name == 'Date':
        df_final = df_final.reset_index()
    elif 'Date' not in df_final.columns:
        # Try to get date from index if it's a DatetimeIndex
        if isinstance(df_final.index, pd.DatetimeIndex):
            df_final['Date'] = df_final.index
        else:
            print(f"Warning: Could not determine Date column for {ticker}")
    
    # Create visualizations
    save_dir = os.path.join("charts", ticker)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nGenerating visualizations for {ticker}...")
    try:
        plot_comprehensive_analysis(
            df_final,
            ticker,
            title_suffix=f" (Buy: {best_result['buy_range']:.2f}, Sell: {best_result['sell_range']:.2f})",
            save_dir=save_dir,
            show=False
        )
    except Exception as e:
        print(f"Error generating visualizations for {ticker}: {e}")
    
    average_buy_range += best_result["buy_range"]
    average_sell_range += best_result["sell_range"]
    print(f"Ticker: {ticker} {best_result}")

print("")
print(
    f"Buy range: {average_buy_range / len(tickers)}, Sell range: {average_sell_range / len(tickers)}"
)
