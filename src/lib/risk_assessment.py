"""
Risk Assessment Module
Implements various risk metrics and analysis tools
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


class RiskAnalyzer:
    """Comprehensive risk analysis for stocks and portfolios"""
    
    def __init__(self, confidence_level: float = 0.95, trading_days: int = 252):
        self.confidence_level = confidence_level
        self.trading_days = trading_days
        self.risk_free_rate = 0.02
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns"""
        return prices.pct_change().dropna()
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize the volatility
        
        Returns:
            Volatility value
        """
        vol = returns.std()
        
        if annualize:
            vol *= np.sqrt(self.trading_days)
        
        return vol
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (defaults to class attribute)
        
        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns.mean() * self.trading_days - risk_free_rate
        volatility = self.calculate_volatility(returns, annualize=True)
        
        return excess_returns / volatility if volatility != 0 else 0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only considers downside volatility)
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
        
        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = returns.mean() * self.trading_days - risk_free_rate
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.trading_days)
        
        return excess_returns / downside_std if downside_std != 0 else 0
    
    def calculate_var_historical(self, returns: pd.Series) -> float:
        """
        Calculate Value at Risk using historical method
        
        Args:
            returns: Series of returns
        
        Returns:
            VaR value (positive number representing potential loss)
        """
        var = np.percentile(returns, (1 - self.confidence_level) * 100)
        return abs(var)
    
    def calculate_var_parametric(self, returns: pd.Series) -> float:
        """
        Calculate Value at Risk using parametric method
        
        Args:
            returns: Series of returns
        
        Returns:
            VaR value
        """
        mean = returns.mean()
        std = returns.std()
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        var = abs(mean + z_score * std)
        return var
    
    def calculate_cvar(self, returns: pd.Series) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Series of returns
        
        Returns:
            CVaR value
        """
        var = self.calculate_var_historical(returns)
        cvar = abs(returns[returns <= -var].mean())
        return cvar
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict:
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            prices: Series of prices
        
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate cumulative returns
        cumulative = (1 + self.calculate_returns(prices)).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_date = cumulative[:max_dd_date].idxmax()
        
        # Calculate recovery
        recovery_date = None
        if max_dd_date != drawdown.index[-1]:
            recovery_mask = (drawdown.loc[max_dd_date:] >= 0)
            if recovery_mask.any():
                recovery_date = drawdown.loc[max_dd_date:][recovery_mask].index[0]
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'peak_date': peak_date,
            'trough_date': max_dd_date,
            'recovery_date': recovery_date,
            'drawdown_duration': (max_dd_date - peak_date).days if peak_date else None,
            'recovery_duration': (recovery_date - max_dd_date).days if recovery_date else None
        }
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta (systematic risk relative to market)
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns
        
        Returns:
            Beta value
        """
        # Align the series
        aligned_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        covariance = aligned_data['stock'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        beta = covariance / market_variance if market_variance != 0 else 0
        return beta
    
    def calculate_alpha(self, stock_returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate alpha (excess return over expected return given beta)
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns
            risk_free_rate: Risk-free rate
        
        Returns:
            Alpha value (annualized)
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        beta = self.calculate_beta(stock_returns, market_returns)
        
        stock_return = stock_returns.mean() * self.trading_days
        market_return = market_returns.mean() * self.trading_days
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        alpha = stock_return - expected_return
        
        return alpha
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
        
        Returns:
            Information ratio
        """
        aligned_data = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        
        ir = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self.trading_days)
        return ir
    
    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """
        Calculate Calmar ratio (return over max drawdown)
        
        Args:
            returns: Series of returns
            prices: Series of prices
        
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * self.trading_days
        max_dd = abs(self.calculate_maximum_drawdown(prices)['max_drawdown'])
        
        return annual_return / max_dd if max_dd != 0 else 0
    
    def calculate_ulcer_index(self, prices: pd.Series) -> float:
        """
        Calculate Ulcer Index (measure of downside risk)
        
        Args:
            prices: Series of prices
        
        Returns:
            Ulcer Index
        """
        cumulative = (1 + self.calculate_returns(prices)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        
        ulcer = np.sqrt(np.mean(drawdown ** 2))
        return ulcer
    
    def generate_risk_report(self, prices: pd.Series, 
                           market_prices: Optional[pd.Series] = None) -> Dict:
        """
        Generate comprehensive risk report
        
        Args:
            prices: Series of stock prices
            market_prices: Series of market prices (optional)
        
        Returns:
            Dictionary with all risk metrics
        """
        returns = self.calculate_returns(prices)
        
        report = {
            'volatility': self.calculate_volatility(returns),
            'annualized_return': returns.mean() * self.trading_days,
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'var_historical': self.calculate_var_historical(returns),
            'var_parametric': self.calculate_var_parametric(returns),
            'cvar': self.calculate_cvar(returns),
            'max_drawdown': self.calculate_maximum_drawdown(prices),
            'calmar_ratio': self.calculate_calmar_ratio(returns, prices),
            'ulcer_index': self.calculate_ulcer_index(prices)
        }
        
        # Add market-relative metrics if market data provided
        if market_prices is not None:
            market_returns = self.calculate_returns(market_prices)
            report['beta'] = self.calculate_beta(returns, market_returns)
            report['alpha'] = self.calculate_alpha(returns, market_returns)
            report['information_ratio'] = self.calculate_information_ratio(returns, market_returns)
        
        return report
    
    def calculate_rolling_volatility(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculate rolling volatility
        
        Args:
            returns: Series of returns
            window: Rolling window size
        
        Returns:
            Series of rolling volatility
        """
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(self.trading_days)
        return rolling_vol
    
    def calculate_rolling_sharpe(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculate rolling Sharpe ratio
        
        Args:
            returns: Series of returns
            window: Rolling window size
        
        Returns:
            Series of rolling Sharpe ratios
        """
        rolling_mean = returns.rolling(window=window).mean() * self.trading_days
        rolling_std = returns.rolling(window=window).std() * np.sqrt(self.trading_days)
        
        rolling_sharpe = (rolling_mean - self.risk_free_rate) / rolling_std
        return rolling_sharpe
    
    def stress_test(self, returns: pd.Series, scenarios: Dict[str, float]) -> Dict:
        """
        Perform stress testing with different scenarios
        
        Args:
            returns: Series of returns
            scenarios: Dictionary of scenario name -> return shock
        
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        for scenario_name, shock in scenarios.items():
            shocked_returns = returns + shock
            results[scenario_name] = {
                'return': shocked_returns.mean() * self.trading_days,
                'volatility': shocked_returns.std() * np.sqrt(self.trading_days),
                'var': self.calculate_var_historical(shocked_returns)
            }
        
        return results
    
    def calculate_downside_risk(self, returns: pd.Series, mar: float = 0) -> float:
        """
        Calculate downside risk relative to minimum acceptable return
        
        Args:
            returns: Series of returns
            mar: Minimum acceptable return
        
        Returns:
            Downside risk
        """
        downside_returns = returns[returns < mar] - mar
        downside_risk = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(self.trading_days)
        
        return downside_risk
