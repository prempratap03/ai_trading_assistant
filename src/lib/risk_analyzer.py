"""
Risk Analysis Module
Implements various risk metrics and analysis tools.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """
    Comprehensive risk analysis for stocks and portfolios.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, data: pd.DataFrame, column: str = 'Close') -> pd.Series:
        """
        Calculate daily returns.
        
        Args:
            data: DataFrame with price data
            column: Column name for prices
        
        Returns:
            Series with daily returns
        """
        return data[column].pct_change().dropna()
    
    def calculate_volatility(
        self, 
        returns: pd.Series, 
        annualize: bool = True
    ) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize volatility
        
        Returns:
            Volatility value
        """
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Assuming 252 trading days
        return vol
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize the ratio
        
        Returns:
            Sharpe ratio
        """
        mean_return = returns.mean()
        std_return = returns.std()
        
        if annualize:
            mean_return *= 252
            std_return *= np.sqrt(252)
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        return sharpe
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        annualize: bool = True
    ) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: Series of returns
            annualize: Whether to annualize the ratio
        
        Returns:
            Sortino ratio
        """
        mean_return = returns.mean()
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if annualize:
            mean_return *= 252
            downside_std *= np.sqrt(252)
        
        sortino = (mean_return - self.risk_free_rate) / downside_std
        return sortino
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
        
        Returns:
            VaR value (as a positive number representing loss)
        """
        if method == 'historical':
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            var = stats.norm.ppf(1 - confidence_level, mean, std)
        else:
            raise ValueError("Method must be 'historical' or 'parametric'")
        
        return abs(var)
    
    def calculate_cvar(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence_level, method='historical')
        cvar = returns[returns <= -var].mean()
        return abs(cvar)
    
    def calculate_beta(
        self, 
        stock_returns: pd.Series, 
        market_returns: pd.Series
    ) -> float:
        """
        Calculate Beta (systematic risk relative to market).
        
        Args:
            stock_returns: Returns of the stock
            market_returns: Returns of the market index
        
        Returns:
            Beta value
        """
        # Align the series
        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
        
        if combined.empty or len(combined) < 2:
            return np.nan
        
        covariance = combined.cov().iloc[0, 1]
        market_variance = combined.iloc[:, 1].var()
        
        beta = covariance / market_variance
        return beta
    
    def calculate_alpha(
        self, 
        stock_returns: pd.Series, 
        market_returns: pd.Series,
        beta: Optional[float] = None
    ) -> float:
        """
        Calculate Alpha (excess return over market).
        
        Args:
            stock_returns: Returns of the stock
            market_returns: Returns of the market index
            beta: Beta value (calculated if not provided)
        
        Returns:
            Alpha value (annualized)
        """
        if beta is None:
            beta = self.calculate_beta(stock_returns, market_returns)
        
        stock_return = stock_returns.mean() * 252
        market_return = market_returns.mean() * 252
        
        alpha = stock_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
        return alpha
    
    def calculate_max_drawdown(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Calculate maximum drawdown.
        
        Args:
            data: DataFrame with price data
            column: Column name for prices
        
        Returns:
            Dictionary with drawdown information
        """
        prices = data[column]
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find the peak before max drawdown
        peak_date = running_max[:max_dd_date].idxmax()
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'peak_date': peak_date,
            'drawdown_series': drawdown
        }
    
    def calculate_calmar_ratio(
        self, 
        returns: pd.Series, 
        max_drawdown: float
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Args:
            returns: Series of returns
            max_drawdown: Maximum drawdown value
        
        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * 252
        calmar = annual_return / abs(max_drawdown)
        return calmar
    
    def calculate_information_ratio(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information ratio.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
        
        Returns:
            Information ratio
        """
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return np.nan
        
        ir = (active_returns.mean() * 252) / tracking_error
        return ir
    
    def calculate_downside_risk(
        self, 
        returns: pd.Series, 
        target_return: float = 0.0
    ) -> float:
        """
        Calculate downside risk (semi-deviation).
        
        Args:
            returns: Series of returns
            target_return: Target return threshold
        
        Returns:
            Downside risk (annualized)
        """
        downside_returns = returns[returns < target_return]
        downside_risk = downside_returns.std() * np.sqrt(252)
        return downside_risk
    
    def calculate_treynor_ratio(
        self, 
        stock_returns: pd.Series, 
        market_returns: pd.Series
    ) -> float:
        """
        Calculate Treynor ratio (return per unit of systematic risk).
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns
        
        Returns:
            Treynor ratio
        """
        beta = self.calculate_beta(stock_returns, market_returns)
        annual_return = stock_returns.mean() * 252
        
        treynor = (annual_return - self.risk_free_rate) / beta
        return treynor
    
    def comprehensive_risk_report(
        self, 
        data: pd.DataFrame, 
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate comprehensive risk analysis report.
        
        Args:
            data: DataFrame with stock data
            market_data: Optional DataFrame with market index data
        
        Returns:
            Dictionary with all risk metrics
        """
        returns = self.calculate_returns(data)
        
        report = {
            'volatility': self.calculate_volatility(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'var_99': self.calculate_var(returns, 0.99),
            'cvar_99': self.calculate_cvar(returns, 0.99),
        }
        
        # Maximum drawdown
        dd_info = self.calculate_max_drawdown(data)
        report['max_drawdown'] = dd_info['max_drawdown']
        report['max_drawdown_date'] = dd_info['max_drawdown_date']
        report['calmar_ratio'] = self.calculate_calmar_ratio(returns, dd_info['max_drawdown'])
        
        # Market-relative metrics (if market data provided)
        if market_data is not None:
            market_returns = self.calculate_returns(market_data)
            report['beta'] = self.calculate_beta(returns, market_returns)
            report['alpha'] = self.calculate_alpha(returns, market_returns, report['beta'])
            report['treynor_ratio'] = self.calculate_treynor_ratio(returns, market_returns)
            report['information_ratio'] = self.calculate_information_ratio(returns, market_returns)
        
        return report
    
    def calculate_portfolio_var(
        self, 
        weights: np.ndarray, 
        returns: pd.DataFrame, 
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate portfolio Value at Risk.
        
        Args:
            weights: Portfolio weights
            returns: DataFrame with asset returns
            confidence_level: Confidence level
        
        Returns:
            Portfolio VaR
        """
        portfolio_returns = (returns * weights).sum(axis=1)
        var = self.calculate_var(portfolio_returns, confidence_level)
        return var


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.
    
    Args:
        returns: DataFrame with returns for multiple assets
    
    Returns:
        Correlation matrix
    """
    return returns.corr()


def calculate_rolling_volatility(
    returns: pd.Series, 
    window: int = 30
) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Series of returns
        window: Rolling window size
    
    Returns:
        Series with rolling volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(252)
