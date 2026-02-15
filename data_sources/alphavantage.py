"""
Alpha Vantage data provider for TraderXO framework.

Fetches market data from Alpha Vantage API.
Handles symbol mapping, timeframe conversion, and data normalization.
"""

from typing import Optional, Dict, List
from datetime import datetime, timedelta
import pandas as pd
import requests
import time
import logging
import os
import pickle
import hashlib

# Try to use loguru, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


class AlphaVantageProvider:
    """
    Fetches market data from Alpha Vantage API.
    
    Handles:
    - Symbol format conversion (stocks, crypto, forex)
    - Timeframe mapping (1d, 4h, 1h)
    - Data normalization to TraderXO format
    - Rate limiting (5 calls/minute free tier)
    - Error handling and retries
    """

    BASE_URL = "https://www.alphavantage.co/query"
    
    # Symbol mapping for Alpha Vantage formats
    # Note: Alpha Vantage uses different endpoints for crypto and forex
    CRYPTO_SYMBOLS = {
        'BTC/USD': 'BTC',
        'ETH/USD': 'ETH',
    }
    
    FOREX_SYMBOLS = {
        # Note: XAU (Gold) is not supported in Alpha Vantage FX endpoints
        # Common forex pairs: EUR, GBP, JPY, AUD, CAD, CHF, etc.
        # 'XAU/USD': 'XAU',  # Commented out - XAU not supported in FX endpoints
    }
    
    # Timeframe mapping: TraderXO -> Alpha Vantage function
    TIMEFRAME_MAP = {
        '1d': 'TIME_SERIES_DAILY',
        '4h': 'TIME_SERIES_INTRADAY',  # Will use 60min interval
        '1h': 'TIME_SERIES_INTRADAY',  # Will use 60min interval
        '1m': 'TIME_SERIES_INTRADAY',
        '5m': 'TIME_SERIES_INTRADAY',
        '15m': 'TIME_SERIES_INTRADAY',
        '30m': 'TIME_SERIES_INTRADAY',
    }
    
    # Interval mapping for intraday
    INTERVAL_MAP = {
        '1h': '60min',
        '4h': '60min',  # Alpha Vantage doesn't support 4h directly, we'll aggregate
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
    }

    def __init__(self, api_key: str, cache_dir: str = "cache/alphavantage", enable_cache: bool = True):
        """
        Initialize Alpha Vantage data provider.
        
        Args:
            api_key: Alpha Vantage API key
            cache_dir: Directory to store cached data (default: "cache/alphavantage")
            enable_cache: Whether to enable caching (default: True)
        """
        self.api_key = api_key
        self.last_call_time = 0
        self.call_count = 0
        self.calls_per_minute = 60  # Free tier: 1 request per second = 60 per minute
        self.min_seconds_between_calls = 1.0  # Free tier: 1 request per second
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Alpha Vantage data provider initialized with cache: {self.cache_dir}")
        else:
            logger.info("Alpha Vantage data provider initialized (cache disabled)")

    def _rate_limit(self):
        """Enforce rate limiting (1 call per second for free tier)."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_seconds_between_calls:
            sleep_time = self.min_seconds_between_calls - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
        self.call_count += 1
        
        # Log warning if approaching daily limit (25 requests for free tier)
        if self.call_count % 20 == 0:
            logger.warning(f"API call count: {self.call_count}. Free tier limit: 25 requests per day")

    def normalize_symbol(self, symbol: str) -> tuple:
        """
        Convert symbol to Alpha Vantage format and determine endpoint type.
        
        Args:
            symbol: Symbol in any format (NVDA, BTC/USD, etc.)
            
        Returns:
            Tuple of (normalized_symbol, endpoint_type)
            endpoint_type: 'stock', 'crypto', or 'forex'
        """
        # Check if crypto
        if symbol in self.CRYPTO_SYMBOLS:
            return (self.CRYPTO_SYMBOLS[symbol], 'crypto')
        
        # Check if forex
        if symbol in self.FOREX_SYMBOLS:
            return (self.FOREX_SYMBOLS[symbol], 'forex')
        
        # Stocks are typically direct (NVDA, TSLA, GOOGL)
        return (symbol, 'stock')

    def _get_cache_path(self, symbol: str, timeframe: str, days: int) -> str:
        """
        Generate cache file path for a given symbol, timeframe, and days.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            days: Number of days
            
        Returns:
            Path to cache file
        """
        # Create a hash of the parameters to create a safe filename
        cache_key = f"{symbol}_{timeframe}_{days}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        filename = f"{cache_hash}.pkl"
        return os.path.join(self.cache_dir, filename)

    def _get_cache_expiration_hours(self, timeframe: str) -> int:
        """
        Get cache expiration time in hours based on timeframe.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Number of hours before cache expires
        """
        # Daily data: cache for 1 day (24 hours)
        # Weekly data: cache for 1 week (168 hours)
        # Intraday data: cache for 1 hour
        if timeframe == '1wk':
            return 168  # 1 week
        elif timeframe == '1d':
            return 24  # 1 day
        else:
            return 1  # 1 hour for intraday

    def _is_cache_valid(self, cache_path: str, timeframe: str) -> bool:
        """
        Check if cache file exists and is still valid.
        
        Args:
            cache_path: Path to cache file
            timeframe: Timeframe string
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not self.enable_cache:
            return False
        
        if not os.path.exists(cache_path):
            return False
        
        # Check file modification time
        file_mtime = os.path.getmtime(cache_path)
        file_age_hours = (time.time() - file_mtime) / 3600
        expiration_hours = self._get_cache_expiration_hours(timeframe)
        
        return file_age_hours < expiration_hours

    def _load_from_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded data from cache: {cache_path}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: str):
        """
        Save DataFrame to cache file.
        
        Args:
            df: DataFrame to cache
            cache_path: Path to cache file
        """
        if not self.enable_cache:
            return
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Saved data to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")

    def fetch_data(
        self,
        symbol: str,
        timeframe: str,
        days: int = 90,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch market data from Alpha Vantage.
        
        Args:
            symbol: Trading symbol (NVDA, TSLA, BTC/USD, etc.)
            timeframe: Timeframe (1d, 4h, 1h, 1wk)
            days: Number of days to fetch (default: 90)
            start_date: Start date (optional, overrides days)
            end_date: End date (optional, defaults to now)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, symbol
            
        Raises:
            ValueError: If symbol or timeframe is invalid
            Exception: If data fetch fails
        """
        # Check cache first
        cache_path = self._get_cache_path(symbol, timeframe, days)
        if self._is_cache_valid(cache_path, timeframe):
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None and not cached_df.empty:
                # Filter by date range if specified
                if start_date is not None or end_date is not None:
                    if end_date is None:
                        end_date = datetime.now()
                    if start_date is None:
                        start_date = end_date - timedelta(days=days)
                    
                    # Ensure dates are naive datetime objects
                    if isinstance(start_date, pd.Timestamp):
                        start_naive = start_date.to_pydatetime()
                        if start_naive.tzinfo:
                            start_naive = start_naive.replace(tzinfo=None)
                    elif hasattr(start_date, 'tzinfo') and start_date.tzinfo:
                        start_naive = start_date.replace(tzinfo=None)
                    else:
                        start_naive = start_date
                        
                    if isinstance(end_date, pd.Timestamp):
                        end_naive = end_date.to_pydatetime()
                        if end_naive.tzinfo:
                            end_naive = end_naive.replace(tzinfo=None)
                    elif hasattr(end_date, 'tzinfo') and end_date.tzinfo:
                        end_naive = end_date.replace(tzinfo=None)
                    else:
                        end_naive = end_date
                    
                    # Filter cached data by date range
                    if 'timestamp' in cached_df.columns:
                        # Handle both timezone-aware and timezone-naive timestamps
                        if cached_df['timestamp'].dt.tz is not None:
                            df_timestamps_naive = cached_df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                        else:
                            df_timestamps_naive = cached_df['timestamp']
                        mask = (df_timestamps_naive >= start_naive) & (df_timestamps_naive <= end_naive)
                        cached_df = cached_df[mask]
                
                logger.info(f"Using cached data for {symbol} {timeframe} ({len(cached_df)} records)")
                return cached_df
        
        # Handle weekly timeframe by aggregating daily data
        # Note: For weekly, we fetch daily data with the same days parameter
        # (not days*7) since we'll aggregate it to weekly
        if timeframe == '1wk':
            # Check cache for weekly data first
            weekly_cache_path = self._get_cache_path(symbol, '1wk', days)
            if self._is_cache_valid(weekly_cache_path, '1wk'):
                cached_weekly = self._load_from_cache(weekly_cache_path)
                if cached_weekly is not None and not cached_weekly.empty:
                    logger.info(f"Using cached weekly data for {symbol} ({len(cached_weekly)} records)")
                    return cached_weekly
            
            # Fetch daily data (use same days, we'll aggregate to weekly)
            df = self.fetch_data(symbol, '1d', days=days, start_date=start_date, end_date=end_date)
            if not df.empty:
                df = self._aggregate_to_weekly(df)
                # Save aggregated weekly data to cache
                if self.enable_cache:
                    self._save_to_cache(df, weekly_cache_path)
            return df
        
        # Normalize symbol and get endpoint type
        av_symbol, endpoint_type = self.normalize_symbol(symbol)
        
        # Map timeframe based on endpoint type
        if endpoint_type == 'crypto':
            # Crypto endpoints
            if timeframe == '1d':
                function = 'DIGITAL_CURRENCY_DAILY'
            elif timeframe in ['1h', '4h']:
                function = 'CRYPTO_INTRADAY'  # Alpha Vantage crypto intraday
            else:
                function = 'CRYPTO_INTRADAY'
        elif endpoint_type == 'forex':
            # Forex endpoints
            if timeframe == '1d':
                function = 'FX_DAILY'
            elif timeframe in ['1h', '4h']:
                function = 'FX_INTRADAY'
            else:
                function = 'FX_INTRADAY'
        else:
            # Stock endpoints
            if timeframe not in self.TIMEFRAME_MAP:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            function = self.TIMEFRAME_MAP[timeframe]
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=days)
        
        # Ensure dates are naive datetime objects for comparison
        # (Alpha Vantage returns UTC timestamps, we'll convert them to naive for comparison)
        if isinstance(start_date, pd.Timestamp):
            start_naive = start_date.to_pydatetime()
            if start_naive.tzinfo:
                start_naive = start_naive.replace(tzinfo=None)
        elif hasattr(start_date, 'tzinfo') and start_date.tzinfo:
            start_naive = start_date.replace(tzinfo=None)
        else:
            start_naive = start_date
            
        if isinstance(end_date, pd.Timestamp):
            end_naive = end_date.to_pydatetime()
            if end_naive.tzinfo:
                end_naive = end_naive.replace(tzinfo=None)
        elif hasattr(end_date, 'tzinfo') and end_date.tzinfo:
            end_naive = end_date.replace(tzinfo=None)
        else:
            end_naive = end_date
        
        logger.info(f"Fetching {av_symbol} data: {timeframe} from {start_naive.date()} to {end_naive.date()}")
        
        # Enforce rate limiting
        self._rate_limit()
        
        try:
            # Build API request based on endpoint type
            params = {
                'apikey': self.api_key,
                'datatype': 'json'
            }
            
            if endpoint_type == 'crypto':
                if function == 'DIGITAL_CURRENCY_DAILY':
                    params['function'] = 'DIGITAL_CURRENCY_DAILY'
                    params['symbol'] = av_symbol
                    params['market'] = 'USD'
                elif function == 'CRYPTO_INTRADAY':
                    params['function'] = 'CRYPTO_INTRADAY'
                    params['symbol'] = av_symbol
                    params['market'] = 'USD'
                    if timeframe in self.INTERVAL_MAP:
                        params['interval'] = self.INTERVAL_MAP[timeframe]
                    else:
                        params['interval'] = '60min'
            elif endpoint_type == 'forex':
                if function == 'FX_DAILY':
                    params['function'] = 'FX_DAILY'
                    params['from_symbol'] = av_symbol
                    params['to_symbol'] = 'USD'
                elif function == 'FX_INTRADAY':
                    params['function'] = 'FX_INTRADAY'
                    params['from_symbol'] = av_symbol
                    params['to_symbol'] = 'USD'
                    if timeframe in self.INTERVAL_MAP:
                        params['interval'] = self.INTERVAL_MAP[timeframe]
                    else:
                        params['interval'] = '60min'
            else:
                # Stock endpoints
                params['function'] = function
                params['symbol'] = av_symbol
                # Free tier: use 'compact' (last ~100 data points) instead of 'full' (premium feature)
                # For daily data, compact gives ~100 days, which is usually sufficient
                params['outputsize'] = 'compact'
                
                # Add interval for intraday data
                if function == 'TIME_SERIES_INTRADAY':
                    if timeframe in self.INTERVAL_MAP:
                        params['interval'] = self.INTERVAL_MAP[timeframe]
                    else:
                        params['interval'] = '60min'  # Default to hourly
            
            # Make API request
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            
            # Check for rate limit information (but don't fail if we have data)
            if 'Information' in data:
                info_msg = data['Information']
                logger.info(f"Alpha Vantage Info: {info_msg[:100]}...")  # Log first 100 chars
            
            if 'Note' in data:
                raise ValueError(f"Alpha Vantage API Note: {data['Note']} (Rate limit exceeded)")
            
            # Extract time series data (different keys for different endpoints)
            time_series_key = None
            
            # Check all possible time series keys
            for key in data.keys():
                if 'Time Series' in key or 'Digital Currency' in key or 'FX' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                # If no time series data and we have Information field, check what it says
                if 'Information' in data:
                    info_msg = data['Information']
                    if 'rate limit' in info_msg.lower() or '25 requests' in info_msg.lower():
                        raise ValueError(f"Alpha Vantage Rate Limit: {info_msg}")
                    elif 'premium' in info_msg.lower() or 'outputsize=full' in info_msg.lower():
                        # This shouldn't happen now that we use 'compact', but handle it gracefully
                        logger.warning(f"Premium feature required for {av_symbol} {timeframe}: {info_msg[:100]}")
                        return pd.DataFrame()
                    else:
                        logger.warning(f"No time series data for {av_symbol} {timeframe}: {info_msg[:100]}")
                        return pd.DataFrame()
                
                logger.warning(f"No time series data found for {av_symbol} {timeframe}")
                logger.debug(f"Available keys: {list(data.keys())}")
                if 'Error Message' in data:
                    logger.debug(f"Error: {data['Error Message']}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = self._parse_time_series(time_series, symbol, timeframe)
            
            # Filter by date range
            if not df.empty:
                # Convert timezone-aware timestamps to naive UTC for comparison
                # Handle both timezone-aware and timezone-naive timestamps
                if df['timestamp'].dt.tz is not None:
                    df_timestamps_naive = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                else:
                    df_timestamps_naive = df['timestamp']
                # Ensure start_naive and end_naive are naive datetime objects
                if isinstance(start_naive, pd.Timestamp):
                    start_naive = start_naive.to_pydatetime()
                if isinstance(end_naive, pd.Timestamp):
                    end_naive = end_naive.to_pydatetime()
                # Filter using naive timestamps
                mask = (df_timestamps_naive >= start_naive) & (df_timestamps_naive <= end_naive)
                df = df[mask]
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} records for {symbol} {timeframe}")
            
            # Handle 4h timeframe by aggregating 1h data
            if timeframe == '4h' and not df.empty:
                df = self._aggregate_to_4h(df)
            
            # Save to cache after successful fetch
            if not df.empty:
                self._save_to_cache(df, cache_path)
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching data for {av_symbol} {timeframe}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching data for {av_symbol} {timeframe}: {e}")
            raise

    def _parse_time_series(
        self,
        time_series: Dict,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Parse Alpha Vantage time series response to DataFrame.
        
        Args:
            time_series: Dictionary with timestamp keys and OHLCV values
            symbol: Original symbol
            timeframe: Timeframe string
            
        Returns:
            Normalized DataFrame
        """
        records = []
        
        for timestamp_str, values in time_series.items():
            # Parse timestamp (Alpha Vantage uses different formats)
            try:
                # Try ISO format first
                timestamp = pd.to_datetime(timestamp_str)
            except:
                # Try other common formats
                try:
                    timestamp = pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S')
                except:
                    timestamp = pd.to_datetime(timestamp_str, format='%Y-%m-%d')
            
            # Extract OHLCV values (Alpha Vantage uses different key names for different endpoints)
            # Stock format: '1. open', '2. high', etc.
            # Crypto format: '1a. open (USD)', '2a. high (USD)', etc. or '1. open', '2. high'
            # Forex format: '1. open', '2. high', etc.
            open_price = float(values.get('1. open', values.get('1a. open (USD)', values.get('open', 0))))
            high_price = float(values.get('2. high', values.get('2a. high (USD)', values.get('high', 0))))
            low_price = float(values.get('3. low', values.get('3a. low (USD)', values.get('low', 0))))
            close_price = float(values.get('4. close', values.get('4a. close (USD)', values.get('close', 0))))
            volume = float(values.get('5. volume', values.get('5. volume (USD)', values.get('volume', 0))))
            
            record = {
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'symbol': symbol
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Ensure timestamp is datetime (timezone-naive UTC)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Convert to timezone-naive UTC for consistency
            # (All timestamps in the system are stored as naive UTC)
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove any rows with NaN in required columns
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        return df

    def _aggregate_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate hourly data to 4-hourly.
        
        Args:
            df: DataFrame with hourly data
            
        Returns:
            DataFrame with 4-hourly aggregated data
        """
        if df.empty:
            return df
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Resample to 4 hours
        resampled = df_indexed.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'symbol': 'first'
        })
        
        # Reset index
        resampled = resampled.reset_index()
        
        # Remove any rows with NaN
        resampled = resampled.dropna()
        
        return resampled

    def _aggregate_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily data to weekly.
        
        Args:
            df: DataFrame with daily data
            
        Returns:
            DataFrame with weekly aggregated data
        """
        if df.empty:
            return df
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Resample to weekly (week starts on Monday)
        resampled = df_indexed.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'symbol': 'first'
        })
        
        # Reset index
        resampled = resampled.reset_index()
        
        # Remove any rows with NaN
        resampled = resampled.dropna()
        
        return resampled

    def fetch_multiple(
        self,
        symbols: List[str],
        timeframe: str,
        days: int = 90
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch
            timeframe: Timeframe string
            days: Number of days to fetch
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, timeframe, days=days)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} {timeframe}: {e}")
                continue
        
        return results

    def normalize_to_yfinance_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Alpha Vantage DataFrame format to yfinance format.
        
        Alpha Vantage returns: timestamp, open, high, low, close, volume, symbol
        yfinance returns: Date index, Open, High, Low, Close, Volume (capitalized)
        
        Args:
            df: DataFrame from Alpha Vantage
            
        Returns:
            DataFrame in yfinance format
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Set timestamp as index (like yfinance uses Date)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
            df.index.name = 'Date'
        
        # Rename columns to match yfinance format (capitalized)
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Keep only OHLCV columns (remove symbol if present)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
