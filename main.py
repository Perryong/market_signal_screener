import os
import time
import pickle
import hashlib
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from candle_sticks.index import prepare_candle_sticks
from data_sources.alphavantage import AlphaVantageProvider
from data_types.signal import Signal
from engine import generate_signal
from indicators.index import prepare_indicators
from previews.verbose import print_signals_multi_tf
from visualizations import plot_comprehensive_analysis
from trading_agents_integration import (
    initialize_trading_agents,
    analyze_with_trading_agents,
    format_trading_agents_signal
)
from config import TRADING_AGENTS_CONFIG

sg_tickers = [

    "BUOU.SI",

]

us_tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "AMZN",
    "META",
]


def prepare_df(df: pd.DataFrame):
    df = df.reset_index()
    df = prepare_indicators(df)
    return prepare_candle_sticks(df)


def _get_yfinance_cache_path(ticker, period=None, interval=None, start=None, end=None, cache_dir="cache/yfinance"):
    """
    Generate cache file path for yfinance data.
    
    Args:
        ticker: Ticker symbol or list of tickers
        period: Period string (e.g., "2y", "60d")
        interval: Interval string (e.g., "1d", "4h", "1h")
        start: Start date
        end: End date
        cache_dir: Cache directory path
        
    Returns:
        Path to cache file
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the parameters to create a safe filename
    if isinstance(ticker, list):
        ticker_str = "_".join(sorted(ticker))
    else:
        ticker_str = str(ticker)
    
    if period and interval:
        cache_key = f"{ticker_str}_{period}_{interval}"
    elif start and end:
        cache_key = f"{ticker_str}_{start}_{end}"
    else:
        cache_key = f"{ticker_str}_unknown"
    
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    filename = f"{cache_hash}.pkl"
    return os.path.join(cache_dir, filename)


def _get_yfinance_cache_expiration_hours(interval: str) -> int:
    """
    Get cache expiration time in hours based on interval.
    
    Args:
        interval: Interval string (e.g., "1d", "4h", "1h")
        
    Returns:
        Number of hours before cache expires
    """
    # 1h data: cache for 1 hour (intraday data changes frequently)
    # 4h data: cache for 4 hours (less frequent updates)
    # Daily data: cache for 24 hours
    # Weekly data: cache for 168 hours (1 week)
    if interval == '1h':
        return 1  # 1 hour
    elif interval == '4h':
        return 4  # 4 hours
    elif interval == '1d':
        return 24  # 1 day
    elif interval == '1wk':
        return 168  # 1 week
    else:
        return 1  # Default to 1 hour for unknown intervals


def _is_yfinance_cache_valid(cache_path: str, interval: str) -> bool:
    """
    Check if yfinance cache file exists and is still valid.
    
    Args:
        cache_path: Path to cache file
        interval: Interval string
        
    Returns:
        True if cache is valid, False otherwise
    """
    if not os.path.exists(cache_path):
        return False
    
    # Check file modification time
    file_mtime = os.path.getmtime(cache_path)
    file_age_hours = (time.time() - file_mtime) / 3600
    expiration_hours = _get_yfinance_cache_expiration_hours(interval)
    
    return file_age_hours < expiration_hours


def _load_yfinance_from_cache(cache_path: str) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from yfinance cache file.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded yfinance data from cache: {cache_path}")
            return data
    except Exception as e:
        print(f"Warning: Failed to load yfinance cache from {cache_path}: {e}")
        return None


def _save_yfinance_to_cache(df: pd.DataFrame, cache_path: str):
    """
    Save DataFrame to yfinance cache file.
    
    Args:
        df: DataFrame to cache
        cache_path: Path to cache file
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Saved yfinance data to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save yfinance cache to {cache_path}: {e}")


def safe_yfinance_download(ticker, period=None, interval=None, start=None, end=None, 
                           max_retries=3, retry_delay=2, backoff_factor=2, use_cache=True):
    """
    Safely download data from yfinance with rate limit handling, retry logic, and caching.
    
    Args:
        ticker: Ticker symbol or list of tickers
        period: Period string (e.g., "2y", "60d")
        interval: Interval string (e.g., "1d", "4h", "1h")
        start: Start date
        end: End date
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        use_cache: Whether to use caching (default: True)
        
    Returns:
        DataFrame with downloaded data, or empty DataFrame if all retries fail
    """
    # Check cache first (only for intraday data: 1h and 4h)
    if use_cache and interval in ['1h', '4h']:
        cache_path = _get_yfinance_cache_path(ticker, period=period, interval=interval, start=start, end=end)
        if _is_yfinance_cache_valid(cache_path, interval):
            cached_df = _load_yfinance_from_cache(cache_path)
            if cached_df is not None and not cached_df.empty:
                return cached_df
    
    # If not in cache or cache expired, download from yfinance
    for attempt in range(max_retries):
        try:
            # Add delay between requests to avoid rate limiting
            if attempt > 0:
                delay = retry_delay * (backoff_factor ** (attempt - 1))
                print(f"Retrying yfinance download for {ticker} (attempt {attempt + 1}/{max_retries}) after {delay:.1f}s delay...")
                time.sleep(delay)
            else:
                # Small delay even on first attempt to avoid rate limits
                time.sleep(0.5)
            
            if period and interval:
                df = yf.download(
                    ticker, period=period, interval=interval, 
                    auto_adjust=True, progress=False, group_by="ticker"
                )
            elif start and end:
                df = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False, group_by="ticker"
                )
            else:
                raise ValueError("Must provide either (period, interval) or (start, end)")
            
            # Save to cache if successful and it's intraday data
            if use_cache and not df.empty and interval in ['1h', '4h']:
                cache_path = _get_yfinance_cache_path(ticker, period=period, interval=interval, start=start, end=end)
                _save_yfinance_to_cache(df, cache_path)
            
            return df
            
        except Exception as e:
            # Check if it's a rate limit error (yfinance raises various exceptions)
            error_str = str(e)
            is_rate_limit = (
                "Rate limit" in error_str or 
                "YFRateLimitError" in error_str or
                "Too Many Requests" in error_str or
                "429" in error_str
            )
            
            if is_rate_limit:
                if attempt < max_retries - 1:
                    delay = retry_delay * (backoff_factor ** attempt)
                    print(f"Rate limit hit for {ticker}. Waiting {delay:.1f}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(delay)
                else:
                    print(f"Error: Rate limit exceeded for {ticker} after {max_retries} attempts. Skipping...")
                    return pd.DataFrame()
            else:
                if attempt < max_retries - 1:
                    print(f"Error downloading {ticker}: {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error: Failed to download {ticker} after {max_retries} attempts: {e}")
                    return pd.DataFrame()
    
    return pd.DataFrame()


def generate_historical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate signals and bullish scores for each row in the dataframe for visualization purposes.
    
    Args:
        df: DataFrame with OHLCV data and computed indicators
        
    Returns:
        DataFrame with 'signal' and 'bullish_score' columns added
    """
    from engine import generate_signal
    from config import (
        INDICATOR_WEIGHTS,
        ADX_THRESHOLDS,
        ADX_FACTORS,
        SNR_FACTORS,
    )
    from scoring.candlesticks import get_candlestick_score
    from scoring.oscillators import get_macd_score, get_obv_slope_score, get_rsi_score
    from scoring.overlays import get_bb_score, get_ma_cross_score, get_volume_sma20_score
    import numpy as np
    
    df = df.copy()
    df["signal"] = "HOLD"
    df["bullish_score"] = np.nan
    
    # We need at least 2 rows for signal generation
    if len(df) < 2:
        return df
    
    # Generate signals for each row starting from index 1
    for i in range(1, len(df)):
        try:
            # Create a slice of the DataFrame up to and including current row
            df_slice = df.iloc[:i + 1].copy()
            
            # Generate signal using engine
            signal_result = generate_signal(df_slice)
            
            # Extract the signal
            df.at[df.index[i], "signal"] = signal_result.signal
            
            # Calculate bullish_score for visualization
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


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Initialize Alpha Vantage provider for US market
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY environment variable is not set. "
                        "Please create a .env file with your API key.")
    av_provider = AlphaVantageProvider(api_key=api_key)

    # Initialize TradingAgentsGraph if enabled
    ta_graph = None
    if TRADING_AGENTS_CONFIG.get("enabled", True):
        print("\n" + "="*60)
        print("Initializing TradingAgents LLM framework...")
        print("="*60)
        try:
            ta_config = {
                "deep_think_llm": TRADING_AGENTS_CONFIG.get("deep_think_llm", "gpt-4o-mini"),
                "quick_think_llm": TRADING_AGENTS_CONFIG.get("quick_think_llm", "gpt-4o-mini"),
                "max_debate_rounds": TRADING_AGENTS_CONFIG.get("max_debate_rounds", 1),
                "data_vendors": TRADING_AGENTS_CONFIG.get("data_vendors", {
                    "core_stock_apis": "yfinance",
                    "technical_indicators": "yfinance",
                    "fundamental_data": "alpha_vantage",
                    "news_data": "alpha_vantage",
                }),
            }
            print(f"Configuration: {TRADING_AGENTS_CONFIG.get('selected_analysts', [])} analysts")
            ta_graph = initialize_trading_agents(
                config=ta_config,
                debug=TRADING_AGENTS_CONFIG.get("debug", False),
                selected_analysts=TRADING_AGENTS_CONFIG.get("selected_analysts", ["market", "social", "news", "fundamentals"])
            )
            if ta_graph:
                print("✓ TradingAgents initialized successfully.")
                print("="*60 + "\n")
            else:
                print("✗ Warning: TradingAgents initialization failed. Continuing with technical analysis only.")
                print("="*60 + "\n")
        except Exception as e:
            print(f"✗ Error during TradingAgents initialization: {e}")
            print("Continuing with technical analysis only.")
            print("="*60 + "\n")
            ta_graph = None
    else:
        print("TradingAgents is disabled in config. Running technical analysis only.\n")

    for tickers in [sg_tickers, us_tickers]:
        # Use Alpha Vantage for US tickers, yfinance for Singapore tickers
        use_alphavantage = tickers == us_tickers

        if use_alphavantage:
            # Fetch data using Alpha Vantage
            df_daily_dict = {}
            df_weekly_dict = {}
            df_4h_dict = {}
            df_1h_dict = {}
            
            for ticker in tickers:
                try:
                    # Fetch daily data (2 years = ~730 days)
                    # Note: Alpha Vantage free tier 'compact' gives ~100 data points
                    # So we request 100 days to get maximum available data
                    df_daily = av_provider.fetch_data(ticker, '1d', days=100)
                    if df_daily.empty:
                        print(f"Warning: No daily data returned for {ticker}")
                        continue
                    df_daily = av_provider.normalize_to_yfinance_format(df_daily)
                    if df_daily.empty:
                        print(f"Warning: Empty DataFrame after normalization for {ticker}")
                        continue
                    df_daily_dict[ticker] = df_daily
                    
                    # Fetch weekly data (aggregated from daily)
                    # Request same amount of days, will be aggregated to weekly
                    df_weekly = av_provider.fetch_data(ticker, '1wk', days=100)
                    if df_weekly.empty:
                        print(f"Warning: No weekly data returned for {ticker}")
                        continue
                    df_weekly = av_provider.normalize_to_yfinance_format(df_weekly)
                    if df_weekly.empty:
                        print(f"Warning: Empty DataFrame after normalization for {ticker} weekly")
                        continue
                    df_weekly_dict[ticker] = df_weekly
                    
                    # Fetch 4h data (60 days for intraday)
                    # Try Alpha Vantage first, fallback to yfinance if premium required
                    df_4h = av_provider.fetch_data(ticker, '4h', days=60)
                    if df_4h.empty:
                        print(f"Warning: Alpha Vantage 4h data unavailable for {ticker} (premium required). Trying yfinance...")
                        df_4h_yf = safe_yfinance_download(ticker, period="60d", interval="4h")
                        if not df_4h_yf.empty and len(df_4h_yf) > 0:
                            # Handle MultiIndex columns from yfinance
                            if isinstance(df_4h_yf.columns, pd.MultiIndex):
                                df_4h = df_4h_yf.xs(ticker, level=0, axis=1) if ticker in df_4h_yf.columns.levels[0] else pd.DataFrame()
                            else:
                                df_4h = df_4h_yf
                            if not df_4h.empty:
                                print(f"Successfully fetched 4h data for {ticker} from yfinance")
                                df_4h_dict[ticker] = df_4h
                        else:
                            print(f"Warning: Could not fetch 4h data for {ticker} from yfinance")
                    else:
                        df_4h = av_provider.normalize_to_yfinance_format(df_4h)
                        if not df_4h.empty:
                            df_4h_dict[ticker] = df_4h
                    
                    # Fetch 1h data (30 days for intraday)
                    # Try Alpha Vantage first, fallback to yfinance if premium required
                    df_1h = av_provider.fetch_data(ticker, '1h', days=30)
                    if df_1h.empty:
                        print(f"Warning: Alpha Vantage 1h data unavailable for {ticker} (premium required). Trying yfinance...")
                        df_1h_yf = safe_yfinance_download(ticker, period="30d", interval="1h")
                        if not df_1h_yf.empty and len(df_1h_yf) > 0:
                            # Handle MultiIndex columns from yfinance
                            if isinstance(df_1h_yf.columns, pd.MultiIndex):
                                df_1h = df_1h_yf.xs(ticker, level=0, axis=1) if ticker in df_1h_yf.columns.levels[0] else pd.DataFrame()
                            else:
                                df_1h = df_1h_yf
                            if not df_1h.empty:
                                print(f"Successfully fetched 1h data for {ticker} from yfinance")
                                df_1h_dict[ticker] = df_1h
                        else:
                            print(f"Warning: Could not fetch 1h data for {ticker} from yfinance")
                    else:
                        df_1h = av_provider.normalize_to_yfinance_format(df_1h)
                        if not df_1h.empty:
                            df_1h_dict[ticker] = df_1h
                except Exception as e:
                    print(f"Error fetching {ticker} from Alpha Vantage: {e}")
                    continue
        else:
            # Use yfinance for Singapore tickers with rate limiting
            print("Fetching daily data from yfinance...")
            df_daily = safe_yfinance_download(tickers, period="2y", interval="1d")
            
            print("Fetching weekly data from yfinance...")
            df_weekly = safe_yfinance_download(tickers, period="5y", interval="1wk")
            
            # Fetch 4h data (60 days) - may not be available for all tickers
            print("Fetching 4h data from yfinance...")
            df_4h = safe_yfinance_download(tickers, period="60d", interval="4h")
            if df_4h.empty:
                print("Warning: No 4h data available from yfinance")
            
            # Fetch 1h data (30 days) - may not be available for all tickers
            print("Fetching 1h data from yfinance...")
            df_1h = safe_yfinance_download(tickers, period="30d", interval="1h")
            if df_1h.empty:
                print("Warning: No 1h data available from yfinance")

        results = []
        for ticker in tickers:
            try:
                if use_alphavantage:
                    if ticker not in df_daily_dict or ticker not in df_weekly_dict:
                        continue
                    prep_daily_df = prepare_df(df_daily_dict[ticker])
                    prep_weekly_df = prepare_df(df_weekly_dict[ticker])
                    prep_4h_df = prepare_df(df_4h_dict[ticker]) if ticker in df_4h_dict else None
                    prep_1h_df = prepare_df(df_1h_dict[ticker]) if ticker in df_1h_dict else None
                    short_name = ticker  # Use ticker as short_name for Alpha Vantage
                else:
                    prep_daily_df = prepare_df(df_daily[ticker])
                    prep_weekly_df = prepare_df(df_weekly[ticker])
                    # Handle 4h and 1h data (may be empty if not available)
                    prep_4h_df = None
                    prep_1h_df = None
                    if not df_4h.empty:
                        if isinstance(df_4h.columns, pd.MultiIndex) and ticker in df_4h.columns.levels[0]:
                            prep_4h_df = prepare_df(df_4h[ticker])
                        elif ticker in df_4h.columns:
                            prep_4h_df = prepare_df(df_4h[ticker])
                    if not df_1h.empty:
                        if isinstance(df_1h.columns, pd.MultiIndex) and ticker in df_1h.columns.levels[0]:
                            prep_1h_df = prepare_df(df_1h[ticker])
                        elif ticker in df_1h.columns:
                            prep_1h_df = prepare_df(df_1h[ticker])
                    short_name = yf.Ticker(ticker).info.get("shortName", ticker)

                # Validate DataFrames before generating signals
                if prep_daily_df.empty or len(prep_daily_df) < 2:
                    print(f"Warning: Insufficient daily data for {ticker} (length: {len(prep_daily_df)})")
                    continue
                if prep_weekly_df.empty or len(prep_weekly_df) < 2:
                    print(f"Warning: Insufficient weekly data for {ticker} (length: {len(prep_weekly_df)})")
                    continue

                sig_daily = generate_signal(prep_daily_df)
                sig_weekly = generate_signal(prep_weekly_df)
                sig_4h = generate_signal(prep_4h_df) if prep_4h_df is not None and len(prep_4h_df) >= 2 else None
                sig_1h = generate_signal(prep_1h_df) if prep_1h_df is not None and len(prep_1h_df) >= 2 else None

                # Combine signals from all timeframes
                # Count BUY and SELL signals
                buy_count = sum([
                    sig_weekly.signal == "BUY",
                    sig_daily.signal == "BUY",
                    sig_4h.signal == "BUY" if sig_4h else False,
                    sig_1h.signal == "BUY" if sig_1h else False,
                ])
                sell_count = sum([
                    sig_weekly.signal == "SELL",
                    sig_daily.signal == "SELL",
                    sig_4h.signal == "SELL" if sig_4h else False,
                    sig_1h.signal == "SELL" if sig_1h else False,
                ])
                
                # Determine final signal based on majority
                if buy_count >= 3:  # Strong BUY if 3+ timeframes agree
                    final_signal = "BUY"
                elif sell_count >= 3:  # Strong SELL if 3+ timeframes agree
                    final_signal = "SELL"
                elif buy_count > sell_count:
                    final_signal = "BUY"
                elif sell_count > buy_count:
                    final_signal = "SELL"
                else:
                    final_signal = "HOLD"

                reasons = [
                    f"Weekly: {sig_weekly.signal}",
                    f"Daily: {sig_daily.signal}",
                ]
                if sig_4h:
                    reasons.append(f"4h: {sig_4h.signal}")
                if sig_1h:
                    reasons.append(f"1h: {sig_1h.signal}")

                combined = Signal(
                    short_name=short_name,
                    ticker=ticker,
                    signal=final_signal,
                    reasons=reasons,
                    entry_range=sig_weekly.entry_range,
                    last_close=sig_daily.last_close,
                    atr=sig_weekly.atr,
                )

                # Run TradingAgents LLM analysis if enabled
                ta_signal = None
                if ta_graph is not None:
                    try:
                        # Get latest date from daily data
                        # Handle both DatetimeIndex and Date column cases
                        if isinstance(prep_daily_df.index, pd.DatetimeIndex):
                            latest_date = prep_daily_df.index[-1].strftime("%Y-%m-%d")
                        elif 'Date' in prep_daily_df.columns:
                            # Date is in a column (after reset_index)
                            latest_date = pd.to_datetime(prep_daily_df['Date'].iloc[-1]).strftime("%Y-%m-%d")
                        else:
                            # Try to get from the original dataframe before prepare_df
                            # Fallback: use today's date minus 1 day (to ensure it's a past date)
                            from datetime import datetime, timedelta
                            latest_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                            print(f"Warning: Could not extract date from DataFrame for {ticker}, using {latest_date}")
                        
                        print(f"Running TradingAgents analysis for {ticker} on {latest_date}...")
                        
                        # Run TradingAgents analysis
                        save_trace = TRADING_AGENTS_CONFIG.get("save_full_trace", True)
                        ta_state, ta_decision = analyze_with_trading_agents(
                            ticker, latest_date, ta_graph, save_full_trace=save_trace
                        )
                        
                        print(f"TradingAgents decision for {ticker}: {ta_decision}")
                        
                        # Format TradingAgents signal
                        ta_signal = format_trading_agents_signal(
                            ticker, ta_decision, ta_state, short_name
                        )
                        
                        # Fill in missing data from technical analysis
                        ta_signal.entry_range = combined.entry_range
                        ta_signal.last_close = combined.last_close
                        ta_signal.atr = combined.atr
                        
                    except Exception as e:
                        print(f"Warning: TradingAgents analysis failed for {ticker}: {e}")
                        import traceback
                        traceback.print_exc()
                        ta_signal = None
                else:
                    if TRADING_AGENTS_CONFIG.get("enabled", True):
                        print(f"Warning: TradingAgents graph not initialized, skipping LLM analysis for {ticker}")

                results.append({
                    "daily": sig_daily,
                    "weekly": sig_weekly,
                    "4h": sig_4h,
                    "1h": sig_1h,
                    "combined": combined,
                    "trading_agents": ta_signal
                })
                
                # Generate visualizations for daily timeframe
                try:
                    # Generate historical signals for visualization
                    prep_daily_df_viz = generate_historical_signals(prep_daily_df.copy())
                    
                    # Ensure Date column exists
                    if 'Date' not in prep_daily_df_viz.columns:
                        if isinstance(prep_daily_df_viz.index, pd.DatetimeIndex):
                            prep_daily_df_viz['Date'] = prep_daily_df_viz.index
                        elif 'Date' in prep_daily_df_viz.index.names:
                            prep_daily_df_viz = prep_daily_df_viz.reset_index()
                    
                    # Create visualizations
                    save_dir = os.path.join("charts", ticker, "daily")
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Generating daily visualizations for {ticker}...")
                    plot_comprehensive_analysis(
                        prep_daily_df_viz,
                        f"{ticker} (Daily)",
                        title_suffix=" - Daily Timeframe",
                        save_dir=save_dir,
                        show=False
                    )
                except Exception as e:
                    print(f"Warning: Could not generate visualizations for {ticker} daily: {e}")
                
                # Generate visualizations for weekly timeframe
                try:
                    # Generate historical signals for visualization
                    prep_weekly_df_viz = generate_historical_signals(prep_weekly_df.copy())
                    
                    # Ensure Date column exists
                    if 'Date' not in prep_weekly_df_viz.columns:
                        if isinstance(prep_weekly_df_viz.index, pd.DatetimeIndex):
                            prep_weekly_df_viz['Date'] = prep_weekly_df_viz.index
                        elif 'Date' in prep_weekly_df_viz.index.names:
                            prep_weekly_df_viz = prep_weekly_df_viz.reset_index()
                    
                    # Create visualizations
                    save_dir = os.path.join("charts", ticker, "weekly")
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Generating weekly visualizations for {ticker}...")
                    plot_comprehensive_analysis(
                        prep_weekly_df_viz,
                        f"{ticker} (Weekly)",
                        title_suffix=" - Weekly Timeframe",
                        save_dir=save_dir,
                        show=False
                    )
                except Exception as e:
                    print(f"Warning: Could not generate visualizations for {ticker} weekly: {e}")
                
                # Generate visualizations for 4h timeframe
                if prep_4h_df is not None and len(prep_4h_df) >= 2:
                    try:
                        # Generate historical signals for visualization
                        prep_4h_df_viz = generate_historical_signals(prep_4h_df.copy())
                        
                        # Ensure Date column exists
                        if 'Date' not in prep_4h_df_viz.columns:
                            if isinstance(prep_4h_df_viz.index, pd.DatetimeIndex):
                                prep_4h_df_viz['Date'] = prep_4h_df_viz.index
                            elif 'Date' in prep_4h_df_viz.index.names:
                                prep_4h_df_viz = prep_4h_df_viz.reset_index()
                        
                        # Create visualizations
                        save_dir = os.path.join("charts", ticker, "4h")
                        os.makedirs(save_dir, exist_ok=True)
                        print(f"Generating 4h visualizations for {ticker}...")
                        plot_comprehensive_analysis(
                            prep_4h_df_viz,
                            f"{ticker} (4h)",
                            title_suffix=" - 4h Timeframe",
                            save_dir=save_dir,
                            show=False
                        )
                    except Exception as e:
                        print(f"Warning: Could not generate visualizations for {ticker} 4h: {e}")
                
                # Generate visualizations for 1h timeframe
                if prep_1h_df is not None and len(prep_1h_df) >= 2:
                    try:
                        # Generate historical signals for visualization
                        prep_1h_df_viz = generate_historical_signals(prep_1h_df.copy())
                        
                        # Ensure Date column exists
                        if 'Date' not in prep_1h_df_viz.columns:
                            if isinstance(prep_1h_df_viz.index, pd.DatetimeIndex):
                                prep_1h_df_viz['Date'] = prep_1h_df_viz.index
                            elif 'Date' in prep_1h_df_viz.index.names:
                                prep_1h_df_viz = prep_1h_df_viz.reset_index()
                        
                        # Create visualizations
                        save_dir = os.path.join("charts", ticker, "1h")
                        os.makedirs(save_dir, exist_ok=True)
                        print(f"Generating 1h visualizations for {ticker}...")
                        plot_comprehensive_analysis(
                            prep_1h_df_viz,
                            f"{ticker} (1h)",
                            title_suffix=" - 1h Timeframe",
                            save_dir=save_dir,
                            show=False
                        )
                    except Exception as e:
                        print(f"Warning: Could not generate visualizations for {ticker} 1h: {e}")
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        # Print results with market label
        market_label = "US Market" if use_alphavantage else "Singapore Market"
        if results:
            print(f"\n{'='*60}")
            print(f"{market_label} Analysis Results")
            print(f"{'='*60}")
            print_signals_multi_tf(results, scores_only=True)
        else:
            print(f"\n{market_label}: No results to display")


if __name__ == "__main__":
    main()
