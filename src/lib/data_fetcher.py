"""
Data Fetching and Preprocessing Module
Handles stock data retrieval and preprocessing operations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import streamlit as st


class StockDataFetcher:
    """Fetch and preprocess stock market data"""
    
    def __init__(self):
        self.cache = {}
    
    @st.cache_data(ttl=3600)
    def fetch_stock_data(_self, ticker: str, period: str = '2y', 
                         interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical stock data for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Clean data
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            
            return df
        
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    
    @st.cache_data(ttl=3600)
    def fetch_multiple_stocks(_self, tickers: List[str], period: str = '2y',
                              interval: str = '1d') -> dict:
        """
        Fetch data for multiple stocks
        
        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        data = {}
        for ticker in tickers:
            try:
                data[ticker] = _self.fetch_stock_data(ticker, period, interval)
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
        
        return data
    
    @st.cache_data(ttl=3600)
    def get_stock_info(_self, ticker: str) -> dict:
        """
        Get stock information and metadata
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
                'description': info.get('longBusinessSummary', 'N/A')
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            Series with daily returns
        """
        return df['Close'].pct_change().dropna()
    
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate logarithmic returns
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            Series with log returns
        """
        return np.log(df['Close'] / df['Close'].shift(1)).dropna()
    
    def normalize_data(self, data: pd.Series) -> np.ndarray:
        """
        Normalize data using min-max scaling
        
        Args:
            data: Series to normalize
        
        Returns:
            Normalized numpy array
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized = scaler.fit_transform(data.values.reshape(-1, 1))
        
        return normalized, scaler
    
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Input data array
            sequence_length: Length of each sequence
        
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def get_price_changes(self, df: pd.DataFrame) -> dict:
        """
        Calculate various price change metrics
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            Dictionary with price change metrics
        """
        current_price = df['Close'].iloc[-1]
        
        changes = {
            '1d': self._calculate_change(df, 1),
            '1w': self._calculate_change(df, 7),
            '1m': self._calculate_change(df, 30),
            '3m': self._calculate_change(df, 90),
            '1y': self._calculate_change(df, 252),
            'ytd': self._calculate_ytd_change(df),
            'current_price': current_price
        }
        
        return changes
    
    def _calculate_change(self, df: pd.DataFrame, days: int) -> float:
        """Calculate percentage change over specified days"""
        if len(df) < days:
            return 0.0
        
        current = df['Close'].iloc[-1]
        previous = df['Close'].iloc[-days]
        
        return ((current - previous) / previous) * 100
    
    def _calculate_ytd_change(self, df: pd.DataFrame) -> float:
        """Calculate year-to-date change"""
        current_year = datetime.now().year
        ytd_data = df[df.index.year == current_year]
        
        if len(ytd_data) < 2:
            return 0.0
        
        start_price = ytd_data['Close'].iloc[0]
        current_price = ytd_data['Close'].iloc[-1]
        
        return ((current_price - start_price) / start_price) * 100
    
    def get_market_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for market data
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            Dictionary with summary statistics
        """
        returns = self.calculate_returns(df)
        
        summary = {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'min_price': df['Close'].min(),
            'max_price': df['Close'].max(),
            'avg_volume': df['Volume'].mean(),
            'total_trading_days': len(df)
        }
        
        return summary
