"""
Centralized configuration for MarketSignals.
All magic numbers, thresholds, and weights are defined here.
"""

# Indicator weights for signal scoring
INDICATOR_WEIGHTS = {
    "rsi": 0.15,        # momentum (increased from 0.10)
    "macd": 0.20,       # trend/momentum (decreased from 0.25)
    "obv": 0.10,        # volume flow
    "ma_cross": 0.15,   # trend (decreased from 0.25)
    "bollinger": 0.15,  # volatility/overbought (increased from 0.10)
    "volume": 0.10,     # raw volume
    "candlestick": 0.15,  # candlesticks (increased from 0.10)
}

# Signal decision thresholds
SIGNAL_THRESHOLDS = {
    "buy": 0.60,        # relaxed from 0.65 (more opportunities)
    "sell": 0.25,       # raised from 0.15 (better capital preservation)
    "strong_buy": 0.75,
    "strong_sell": 0.10,
}

# ADX trend strength thresholds
ADX_THRESHOLDS = {
    "weak": 20,         # below this = weak trend
    "strong": 40,       # above this = strong trend
}

# ADX adjustment factors
ADX_FACTORS = {
    "weak_trend": 0.8,   # multiplier when ADX < weak threshold
    "strong_trend": 1.2,  # multiplier when ADX > strong threshold
    "normal_trend": 1.0,  # multiplier when ADX between thresholds
}

# SNR (Support/Resistance) proximity factors
SNR_FACTORS = {
    "near_level": 1.2,   # multiplier when close to S/R (strength > 0.7)
    "far_from_level": 0.8,  # multiplier when far from S/R
    "no_level": 1.0,     # multiplier when no S/R detected
}

# Entry range multipliers for ATR-based expected move
ENTRY_RANGE_MULTIPLIERS = {
    "pullback": 0.8,    # changed from 0.5 (more patient pullback)
    "chase": 0.3,       # changed from 0.8 (less chase)
}

# Candlestick pattern scoring
CANDLESTICK_CONFIG = {
    "lookback_days": 30,
    "decay_factor": 0.95,
    "score_divisor": 6,
    "strong_threshold": 0.7,
}

# S/R detection parameters
SNR_CONFIG = {
    "tolerance_percent": 0.05,  # 5% tolerance for S/R proximity
    "window_duration": 15,      # lookback for S/R detection
}

# RSI thresholds
RSI_THRESHOLDS = {
    "oversold": 30,
    "oversold_zone": 40,
    "overbought_zone": 60,
    "overbought": 70,
}

# Volume thresholds
VOLUME_THRESHOLDS = {
    "high_multiplier": 1.2,
    "low_multiplier": 0.8,
}

# Divergence detection parameters
DIVERGENCE_CONFIG = {
    "lookback_period": 14,
    "min_peaks": 2,
}

# TradingAgents LLM-based analysis configuration
TRADING_AGENTS_CONFIG = {
    "enabled": False,  # Set to False to disable TradingAgents analysis, True to enable
    "debug": False,  # Enable debug mode for TradingAgents
    "save_full_trace": True,  # Save full agent conversation trace to JSON files
    "selected_analysts": ["market", "social", "news", "fundamentals"],  # Analyst types to include
    "max_debate_rounds": 1,  # Number of debate rounds between bull/bear researchers
    "deep_think_llm": "gpt-4o-mini",  # LLM for deep thinking (research manager, risk manager)
    "quick_think_llm": "gpt-4o-mini",  # LLM for quick thinking (analysts, researchers)
    "data_vendors": {
        "core_stock_apis": "yfinance",  # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "alpha_vantage",  # Options: openai, alpha_vantage, local
        "news_data": "alpha_vantage",  # Options: openai, alpha_vantage, google, local
    },
}