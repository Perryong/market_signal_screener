# MarketSignal v4

A comprehensive technical analysis and trading signal generation system that combines multiple indicators, candlestick patterns, and market analysis to provide actionable trading signals.

## Features

### 🎯 Multi-Timeframe Analysis

- **Multi-Timeframe Support**: Analyzes weekly, daily, 4-hour, and 1-hour timeframes for comprehensive market view
- **Intelligent Signal Aggregation**: Combines signals from all available timeframes using majority voting for stronger conviction
- **Timeframe Priority**: Strong BUY/SELL signals when 3+ timeframes agree, otherwise uses majority rule

### 📊 Technical Indicators

- **Oscillators**: RSI, MACD, OBV (On-Balance Volume)
- **Overlays**: Moving Average Crossovers, Bollinger Bands, Volume SMA
- **Trend Analysis**: ADX (Average Directional Index) with +DI/-DI for trend strength and direction
- **Support/Resistance**: Automatic detection and proximity analysis
- **Divergence Detection**: RSI and MACD divergence analysis for early trend reversal signals

### 🕯️ Candlestick Pattern Recognition

- **Single Candle Patterns**: Doji, Hammer, Shooting Star, etc.
- **Two Candle Patterns**: Engulfing, Harami, etc.
- **Three Candle Patterns**: Morning Star, Evening Star, Three White Soldiers, etc.

### 🎲 Scoring System

- **Weighted Scoring**: Each indicator contributes with carefully tuned weights
- **ADX Adjustment**: Trend strength modifies signal confidence
- **S/R Proximity**: Distance from support/resistance levels affects signal strength
- **Threshold-Based Decisions**: Clear BUY/SELL/HOLD signals based on score thresholds

### 📈 Backtesting Capabilities

- **Historical Analysis**: Test strategies on historical data
- **Performance Metrics**: Track signal accuracy and profitability
- **Parameter Optimization**: Fine-tune buy/sell thresholds

## Project Structure

```
MarketSignal v4/
├── main.py                 # Main execution script with data fetching
├── engine.py              # Core signal generation logic
├── backtest.py            # Backtesting functionality
├── config.py              # Centralized configuration (weights, thresholds)
├── data_types/
│   └── signal.py          # Signal data structure
├── data_sources/
│   └── alphavantage.py    # Alpha Vantage API provider for US stocks
├── indicators/
│   ├── index.py           # Indicator preparation
│   ├── oscillators.py     # RSI, MACD, OBV calculations
│   └── overlays.py        # MA, Bollinger Bands, Volume
├── candle_sticks/
│   ├── index.py           # Candlestick pattern detection
│   ├── single.py          # Single candle patterns
│   ├── two.py             # Two candle patterns
│   ├── three.py           # Three candle patterns
│   ├── conditions.py      # Pattern conditions
│   └── helper.py          # Utility functions
├── scoring/
│   ├── candlesticks.py    # Candlestick pattern scoring
│   ├── oscillators.py     # Oscillator scoring
│   └── overlays.py        # Overlay indicator scoring
├── previews/
│   ├── charts.py          # Chart visualization
│   └── verbose.py         # Detailed signal output
├── visualizations/
│   └── time_series.py     # Comprehensive time-series charting
├── utils/
│   ├── debug.py           # Debug utilities
│   └── helper.py          # General helper functions
├── cache/                 # Cached market data (yfinance, alphavantage)
└── charts/                # Generated chart visualizations
```

## Usage

### Prerequisites

1. **Python 3.8+** installed
2. **Alpha Vantage API Key** (for US market data):
   - Sign up at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Create a `.env` file in the project root:
     ```
     ALPHAVANTAGE_API_KEY=your_api_key_here
     ```

### Basic Usage

```bash
python main.py
```

The script will:
1. Fetch market data for configured tickers
2. Generate signals for all timeframes (weekly, daily, 4h, 1h)
3. Combine signals using majority voting
4. Generate comprehensive visualizations
5. Display results in the console

### Supported Markets

- **Singapore Stocks**: REITs and major companies (uses yfinance)
  - Example: `BUOU.SI`
- **US Stocks**: Major tech and blue-chip stocks (uses Alpha Vantage API)
  - Example: `AAPL`, `MSFT`, `GOOGL`, `NVDA`, `AMZN`, `META`

### Data Sources

- **US Stocks**: Alpha Vantage API (requires free API key)
- **Singapore Stocks**: yfinance (free, no API key required)
- **Caching**: Intraday data (1h, 4h) is cached to reduce API calls
  - 1h data: cached for 1 hour
  - 4h data: cached for 4 hours

### Signal Types

- **BUY**: Strong bullish signal (score ≥ 0.60)
- **SELL**: Strong bearish signal (score ≤ 0.25)
- **HOLD**: Neutral or weak signal
- **Strong BUY**: Very strong bullish signal (score ≥ 0.75)
- **Strong SELL**: Very strong bearish signal (score ≤ 0.10)

### Signal Components

Each signal includes:

- **Ticker**: Stock symbol
- **Signal**: BUY/SELL/HOLD
- **Reasons**: Detailed explanation of signal components
- **Entry Range**: Suggested entry price range based on ATR
- **Last Close**: Most recent closing price
- **ATR**: Average True Range for volatility assessment

## Technical Details

### Scoring Weights

- **RSI**: 15% (momentum)
- **MACD**: 20% (trend/momentum)
- **OBV**: 10% (volume flow)
- **MA Cross**: 15% (trend)
- **Bollinger Bands**: 15% (volatility/overbought)
- **Volume SMA**: 10% (volume confirmation)
- **Candlesticks**: 15% (pattern recognition)

### Adjustments

- **ADX Factor** (trend strength):
  - < 20: 0.8x (weak trend)
  - 20-40: 1.0x (normal)
  - > 40: 1.2x (strong trend)
- **S/R Factor** (support/resistance proximity):
  - Near S/R (within 5%): 1.2x (stronger signal)
  - Far from S/R: 0.8x (weaker signal)
  - No S/R detected: 1.0x (neutral)
- **+DI/-DI Factor** (trend direction):
  - Aligned with trend: 1.1x (bullish trend + bullish score, or bearish trend + bearish score)
  - Contrarian: 0.9x (bullish trend + bearish score, or bearish trend + bullish score)
- **Divergence Bonus**:
  - Bullish divergence: +0.1 to score
  - Bearish divergence: -0.1 to score

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.0
- matplotlib >= 3.7.0
- requests >= 2.31.0
- python-dotenv >= 1.0.0

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd market_signal_screener
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Alpha Vantage API key:
     ```
     ALPHAVANTAGE_API_KEY=your_api_key_here
     ```

4. Configure tickers (optional):
   - Edit `main.py` to modify `sg_tickers` and `us_tickers` lists

5. Run the main script:
   ```bash
   python main.py
   ```

## Backtesting

Use the backtesting module to test strategies on historical data:

```bash
python backtest.py
```

## Visualizations

The system automatically generates comprehensive charts for each ticker and timeframe:

- **Price Action**: Candlestick charts with support/resistance levels
- **Indicators**: RSI, MACD, OBV, Bollinger Bands, Moving Averages
- **Signals**: Historical BUY/SELL signals overlaid on price charts
- **Bullish Score**: Time-series of signal strength scores

Charts are saved to `charts/<ticker>/<timeframe>/` directory.

## Configuration

All parameters can be customized in `config.py`:

- **Indicator Weights**: Adjust contribution of each indicator
- **Signal Thresholds**: Modify BUY/SELL decision boundaries
- **ADX Settings**: Configure trend strength thresholds
- **S/R Detection**: Adjust support/resistance detection parameters
- **Candlestick Config**: Pattern recognition settings

## Features in Detail

### Data Fetching
- **Rate Limiting**: Automatic retry logic with exponential backoff
- **Caching**: Intelligent caching for intraday data to minimize API calls
- **Error Handling**: Graceful handling of API failures and missing data
- **Multi-Source**: Seamless integration of Alpha Vantage and yfinance

### Signal Generation
- **Weighted Scoring**: Combines multiple indicators with configurable weights
- **Multi-Factor Analysis**: Considers trend strength, momentum, volume, and patterns
- **Context-Aware**: Adjusts signals based on proximity to support/resistance
- **Divergence Detection**: Identifies potential trend reversals early

### Visualization
- **Multi-Timeframe Charts**: Separate visualizations for each timeframe
- **Comprehensive Analysis**: Price, volume, indicators, and signals in one view
- **Historical Context**: Shows signal performance over time
- **Export Ready**: High-quality charts saved for analysis

## Contributing

This is a personal trading signal system. Feel free to fork and modify for your own use.

## License

This project is for personal use. Modify and distribute as needed.

## Disclaimer

⚠️ **IMPORTANT**: This software is for educational and research purposes only. Trading involves substantial risk, and past performance does not guarantee future results. The signals generated by this system are not financial advice. Always:

- Do your own research (DYOR)
- Understand the risks involved
- Consider consulting with a licensed financial advisor
- Never invest more than you can afford to lose
- Test strategies thoroughly before live trading

The authors and contributors are not responsible for any financial losses incurred from using this software.
