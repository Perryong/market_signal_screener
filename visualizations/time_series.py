"""
Time-series visualization module for market signals analysis.
Creates comprehensive charts showing price, indicators, signals, and performance over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List
import numpy as np


def plot_price_with_signals(
    df: pd.DataFrame,
    ticker: str,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot price chart with BUY/SELL signals marked.
    
    Args:
        df: DataFrame with Date, Open, High, Low, Close, Volume columns
        ticker: Stock ticker symbol
        title_suffix: Additional text for title
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    # Ensure Date is datetime and set as index
    if 'Date' in df.columns:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Plot price with candlesticks
    dates = df.index
    colors = ['green' if df.loc[date, 'Close'] >= df.loc[date, 'Open'] else 'red' 
              for date in dates]
    
    # Plot candlesticks
    for date in dates:
        row = df.loc[date]
        high = row['High']
        low = row['Low']
        open_price = row['Open']
        close = row['Close']
        
        # Draw the wick
        ax1.plot([date, date], [low, high], color='black', linewidth=0.5)
        # Draw the body
        body_color = 'green' if close >= open_price else 'red'
        body_bottom = min(open_price, close)
        body_top = max(open_price, close)
        ax1.bar(date, body_top - body_bottom, bottom=body_bottom, 
                color=body_color, alpha=0.8, width=0.8)
    
    # Plot moving averages if available
    if 'ma50' in df.columns:
        ax1.plot(dates, df['ma50'], label='MA50', color='blue', linewidth=1.5, alpha=0.7)
    if 'ma200' in df.columns:
        ax1.plot(dates, df['ma200'], label='MA200', color='orange', linewidth=1.5, alpha=0.7)
    
    # Plot Bollinger Bands if available
    if all(col in df.columns for col in ['bb_upper', 'bb_mid', 'bb_lower']):
        ax1.plot(dates, df['bb_upper'], label='BB Upper', color='gray', linewidth=1, alpha=0.5, linestyle='--')
        ax1.plot(dates, df['bb_mid'], label='BB Mid', color='gray', linewidth=1, alpha=0.5, linestyle=':')
        ax1.plot(dates, df['bb_lower'], label='BB Lower', color='gray', linewidth=1, alpha=0.5, linestyle='--')
        ax1.fill_between(dates, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
    
    # Plot Support/Resistance levels if available
    if 'snr' in df.columns:
        for date in dates:
            snr_val = df.loc[date, 'snr']
            if not pd.isna(snr_val):
                snr_type = df.loc[date, 'snr_type'] if 'snr_type' in df.columns else None
                color = 'green' if snr_type == 'support' else 'red' if snr_type == 'resistance' else 'blue'
                ax1.axhline(y=snr_val, color=color, alpha=0.3, linewidth=0.5)
    
    # Mark BUY signals
    if 'signal' in df.columns:
        buy_dates = df[df['signal'] == 'BUY'].index
        buy_prices = df.loc[buy_dates, 'Close']
        ax1.scatter(buy_dates, buy_prices, color='green', marker='^', 
                   s=150, label='BUY', zorder=5, edgecolors='darkgreen', linewidths=1.5)
        
        # Mark SELL signals
        sell_dates = df[df['signal'] == 'SELL'].index
        sell_prices = df.loc[sell_dates, 'Close']
        ax1.scatter(sell_dates, sell_prices, color='red', marker='v', 
                   s=150, label='SELL', zorder=5, edgecolors='darkred', linewidths=1.5)
    
    ax1.set_title(f'{ticker} - Price Chart with Signals{title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot volume
    if 'Volume' in df.columns:
        volume_colors = ['green' if df.loc[date, 'Close'] >= df.loc[date, 'Open'] else 'red' 
                        for date in dates]
        ax2.bar(dates, df['Volume'], color=volume_colors, alpha=0.6)
        if 'volume_sma20' in df.columns:
            ax2.plot(dates, df['volume_sma20'], label='Volume SMA20', color='blue', linewidth=1.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_indicators(
    df: pd.DataFrame,
    ticker: str,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot technical indicators over time.
    
    Args:
        df: DataFrame with Date and indicator columns
        ticker: Stock ticker symbol
        title_suffix: Additional text for title
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    # Ensure Date is datetime and set as index
    if 'Date' in df.columns:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    dates = df.index
    
    # Create subplots for different indicators
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # RSI
    ax1 = fig.add_subplot(gs[0, 0])
    if 'rsi' in df.columns:
        ax1.plot(dates, df['rsi'], label='RSI', color='purple', linewidth=2)
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax1.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax1.fill_between(dates, 70, 100, alpha=0.1, color='red')
        ax1.fill_between(dates, 0, 30, alpha=0.1, color='green')
        ax1.set_ylabel('RSI', fontsize=11)
        ax1.set_title('RSI (14)', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # MACD
    ax2 = fig.add_subplot(gs[0, 1])
    if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist']):
        ax2.plot(dates, df['macd'], label='MACD', color='blue', linewidth=2)
        ax2.plot(dates, df['macd_signal'], label='Signal', color='red', linewidth=2)
        colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
        ax2.bar(dates, df['macd_hist'], label='Histogram', color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('MACD', fontsize=11)
        ax2.set_title('MACD', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ADX with +DI/-DI
    ax3 = fig.add_subplot(gs[1, 0])
    if all(col in df.columns for col in ['adx', 'plus_di', 'minus_di']):
        ax3.plot(dates, df['adx'], label='ADX', color='black', linewidth=2)
        ax3.plot(dates, df['plus_di'], label='+DI', color='green', linewidth=1.5)
        ax3.plot(dates, df['minus_di'], label='-DI', color='red', linewidth=1.5)
        ax3.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='Weak Trend (25)')
        ax3.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='Strong Trend (40)')
        ax3.set_ylabel('ADX / DI', fontsize=11)
        ax3.set_title('ADX with +DI/-DI', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # OBV Slope
    ax4 = fig.add_subplot(gs[1, 1])
    if 'obv_slope' in df.columns:
        colors = ['green' if val >= 0 else 'red' for val in df['obv_slope']]
        ax4.bar(dates, df['obv_slope'], color=colors, alpha=0.6)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('OBV Slope', fontsize=11)
        ax4.set_title('On-Balance Volume Slope', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Divergence indicators
    ax5 = fig.add_subplot(gs[2, 0])
    if 'rsi_divergence' in df.columns:
        rsi_div = df['rsi_divergence'].fillna(0)
        colors = ['green' if val == 1 else 'red' if val == -1 else 'gray' for val in rsi_div]
        ax5.scatter(dates, rsi_div, c=colors, s=50, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_ylabel('RSI Divergence', fontsize=11)
        ax5.set_title('RSI Divergence (1=Bullish, -1=Bearish)', fontsize=12, fontweight='bold')
        ax5.set_ylim(-1.5, 1.5)
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    if 'macd_divergence' in df.columns:
        ax6 = fig.add_subplot(gs[2, 1])
        macd_div = df['macd_divergence'].fillna(0)
        colors = ['green' if val == 1 else 'red' if val == -1 else 'gray' for val in macd_div]
        ax6.scatter(dates, macd_div, c=colors, s=50, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.set_ylabel('MACD Divergence', fontsize=11)
        ax6.set_title('MACD Divergence (1=Bullish, -1=Bearish)', fontsize=12, fontweight='bold')
        ax6.set_ylim(-1.5, 1.5)
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Bullish Score (if available)
    if 'bullish_score' in df.columns:
        ax7 = fig.add_subplot(gs[3, :])
        ax7.plot(dates, df['bullish_score'], label='Bullish Score', color='blue', linewidth=2)
        ax7.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='BUY Threshold (0.6)')
        ax7.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='SELL Threshold (0.25)')
        ax7.fill_between(dates, 0.6, 1.0, alpha=0.1, color='green', label='BUY Zone')
        ax7.fill_between(dates, 0.0, 0.25, alpha=0.1, color='red', label='SELL Zone')
        ax7.set_ylabel('Bullish Score', fontsize=11)
        ax7.set_xlabel('Date', fontsize=12)
        ax7.set_title('Bullish Score Over Time', fontsize=12, fontweight='bold')
        ax7.set_ylim(0, 1)
        ax7.legend(loc='best', fontsize=9)
        ax7.grid(True, alpha=0.3)
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle(f'{ticker} - Technical Indicators{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_equity_curve(
    df: pd.DataFrame,
    ticker: str,
    initial_capital: float = 100000,
    title_suffix: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot equity curve from backtest results.
    
    Args:
        df: DataFrame with Date, equity, signal columns
        ticker: Stock ticker symbol
        initial_capital: Initial capital for reference
        title_suffix: Additional text for title
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    # Ensure Date is datetime and set as index
    if 'Date' in df.columns:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    if 'equity' not in df.columns:
        print("Warning: 'equity' column not found in DataFrame")
        return
    
    dates = df.index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Plot equity curve
    ax1.plot(dates, df['equity'], label='Equity Curve', color='green', linewidth=2)
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label=f'Initial Capital (${initial_capital:,.0f})')
    
    # Mark BUY signals
    if 'signal' in df.columns:
        buy_dates = df[df['signal'] == 'BUY'].index
        if len(buy_dates) > 0:
            buy_equity = df.loc[buy_dates, 'equity']
            ax1.scatter(buy_dates, buy_equity, color='green', marker='^', 
                       s=150, label='BUY', zorder=5, edgecolors='darkgreen', linewidths=1.5)
        
        # Mark SELL signals
        sell_dates = df[df['signal'] == 'SELL'].index
        if len(sell_dates) > 0:
            sell_equity = df.loc[sell_dates, 'equity']
            ax1.scatter(sell_dates, sell_equity, color='red', marker='v', 
                       s=150, label='SELL', zorder=5, edgecolors='darkred', linewidths=1.5)
    
    # Calculate and display performance metrics
    final_equity = df['equity'].iloc[-1]
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    max_equity = df['equity'].max()
    max_drawdown = ((max_equity - df['equity'].min()) / max_equity) * 100
    
    ax1.set_title(f'{ticker} - Equity Curve{title_suffix}\n'
                  f'Total Return: {total_return:.2f}% | Max Drawdown: {max_drawdown:.2f}%',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot drawdown
    rolling_max = df['equity'].expanding().max()
    drawdown = ((df['equity'] - rolling_max) / rolling_max) * 100
    ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    ax2.plot(dates, drawdown, color='red', linewidth=1.5)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comprehensive_analysis(
    df: pd.DataFrame,
    ticker: str,
    title_suffix: str = "",
    save_dir: Optional[str] = None,
    show: bool = True
):
    """
    Create all visualizations for a comprehensive analysis.
    
    Args:
        df: DataFrame with all analysis data
        ticker: Stock ticker symbol
        title_suffix: Additional text for titles
        save_dir: Optional directory to save all charts
        show: Whether to display the plots
    """
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        price_path = os.path.join(save_dir, f'{ticker}_price_signals.png')
        indicators_path = os.path.join(save_dir, f'{ticker}_indicators.png')
    else:
        price_path = None
        indicators_path = None
    
    # Plot price with signals
    plot_price_with_signals(df, ticker, title_suffix, price_path, show)
    
    # Plot indicators
    plot_indicators(df, ticker, title_suffix, indicators_path, show)
    
    # Plot equity curve if available
    if 'equity' in df.columns:
        if save_dir:
            equity_path = os.path.join(save_dir, f'{ticker}_equity_curve.png')
        else:
            equity_path = None
        plot_equity_curve(df, ticker, title_suffix=title_suffix, 
                         save_path=equity_path, show=show)
