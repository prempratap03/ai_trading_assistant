"""
Technical Indicators Module
Implements various technical analysis indicators and trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class TechnicalIndicators:
    """Calculate technical indicators and generate trading signals"""
    
    def __init__(self):
        pass
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Price series
            period: Period for moving average
        
        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price series
            period: Period for EMA
        
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Price series
            period: RSI period
        
        Returns:
            RSI series
        """
        delta = data.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, 
                                  num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price series
            period: Period for moving average
            num_std: Number of standard deviations
        
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle_band = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
        
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for ATR
        
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume
        
        Args:
            close: Close price series
            volume: Volume series
        
        Returns:
            OBV series
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for ADX
        
        Returns:
            ADX series
        """
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Calculate ATR
        atr = self.calculate_atr(high, low, close, period)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for CCI
        
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci
    
    def generate_rsi_signals(self, rsi: pd.Series, oversold: float = 30, 
                            overbought: float = 70) -> pd.Series:
        """
        Generate trading signals based on RSI
        
        Args:
            rsi: RSI series
            oversold: Oversold threshold
            overbought: Overbought threshold
        
        Returns:
            Signal series (1=Buy, -1=Sell, 0=Hold)
        """
        signals = pd.Series(0, index=rsi.index)
        signals[rsi < oversold] = 1  # Buy signal
        signals[rsi > overbought] = -1  # Sell signal
        
        return signals
    
    def generate_macd_signals(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """
        Generate trading signals based on MACD
        
        Args:
            macd: MACD line
            signal: Signal line
        
        Returns:
            Signal series (1=Buy, -1=Sell, 0=Hold)
        """
        signals = pd.Series(0, index=macd.index)
        
        # Buy when MACD crosses above signal
        signals[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 1
        
        # Sell when MACD crosses below signal
        signals[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = -1
        
        return signals
    
    def generate_bollinger_signals(self, close: pd.Series, upper: pd.Series, 
                                   lower: pd.Series) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            close: Close price series
            upper: Upper band
            lower: Lower band
        
        Returns:
            Signal series (1=Buy, -1=Sell, 0=Hold)
        """
        signals = pd.Series(0, index=close.index)
        
        # Buy when price touches lower band
        signals[close <= lower] = 1
        
        # Sell when price touches upper band
        signals[close >= upper] = -1
        
        return signals
    
    def generate_combined_signals(self, df: pd.DataFrame, 
                                 rsi_period: int = 14,
                                 macd_fast: int = 12,
                                 macd_slow: int = 26,
                                 macd_signal: int = 9,
                                 bb_period: int = 20) -> pd.DataFrame:
        """
        Generate combined trading signals from multiple indicators
        
        Args:
            df: DataFrame with OHLCV data
            rsi_period: RSI period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            bb_period: Bollinger Bands period
        
        Returns:
            DataFrame with all indicators and signals
        """
        result = df.copy()
        
        # Calculate indicators
        result['RSI'] = self.calculate_rsi(df['Close'], rsi_period)
        
        macd, signal, hist = self.calculate_macd(df['Close'], macd_fast, macd_slow, macd_signal)
        result['MACD'] = macd
        result['MACD_Signal'] = signal
        result['MACD_Hist'] = hist
        
        upper, middle, lower = self.calculate_bollinger_bands(df['Close'], bb_period)
        result['BB_Upper'] = upper
        result['BB_Middle'] = middle
        result['BB_Lower'] = lower
        
        # Generate signals
        result['RSI_Signal'] = self.generate_rsi_signals(result['RSI'])
        result['MACD_Signal_Line'] = self.generate_macd_signals(result['MACD'], result['MACD_Signal'])
        result['BB_Signal'] = self.generate_bollinger_signals(df['Close'], upper, lower)
        
        # Combined signal (majority vote)
        result['Combined_Signal'] = (
            result['RSI_Signal'] + 
            result['MACD_Signal_Line'] + 
            result['BB_Signal']
        )
        
        # Normalize combined signal to -1, 0, 1
        result['Combined_Signal'] = result['Combined_Signal'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        
        return result
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate support and resistance levels
        
        Args:
            df: DataFrame with OHLCV data
            window: Window for calculation
        
        Returns:
            Dictionary with support and resistance levels
        """
        recent_data = df.tail(window)
        
        support_levels = []
        resistance_levels = []
        
        for i in range(1, len(recent_data) - 1):
            # Support: local minimum
            if (recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-1] and 
                recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+1]):
                support_levels.append(recent_data['Low'].iloc[i])
            
            # Resistance: local maximum
            if (recent_data['High'].iloc[i] > recent_data['High'].iloc[i-1] and 
                recent_data['High'].iloc[i] > recent_data['High'].iloc[i+1]):
                resistance_levels.append(recent_data['High'].iloc[i])
        
        return {
            'support': np.mean(support_levels) if support_levels else df['Low'].min(),
            'resistance': np.mean(resistance_levels) if resistance_levels else df['High'].max(),
            'current_price': df['Close'].iloc[-1]
        }
    
    def backtest_strategy(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Backtest trading strategy based on combined signals
        
        Args:
            df: DataFrame with signals
            initial_capital: Initial capital for backtesting
        
        Returns:
            Dictionary with backtest results
        """
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(len(df)):
            signal = df['Combined_Signal'].iloc[i]
            price = df['Close'].iloc[i]
            
            if signal == 1 and position == 0:  # Buy signal
                shares = capital / price
                position = shares
                capital = 0
                trades.append({'type': 'buy', 'price': price, 'date': df.index[i]})
            
            elif signal == -1 and position > 0:  # Sell signal
                capital = position * price
                position = 0
                trades.append({'type': 'sell', 'price': price, 'date': df.index[i]})
        
        # Close any open position
        if position > 0:
            capital = position * df['Close'].iloc[-1]
            position = 0
        
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        return {
            'final_capital': capital,
            'total_return': total_return,
            'num_trades': len(trades),
            'trades': trades
        }
