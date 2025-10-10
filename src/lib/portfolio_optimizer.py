"""
Portfolio Optimization Module
Implements Modern Portfolio Theory and portfolio optimization strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import cvxpy as cp
from scipy.optimize import minimize
import streamlit as st


class PortfolioOptimizer:
    """Modern Portfolio Theory based portfolio optimization"""
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
    
    def calculate_returns(self, prices_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate daily returns for multiple stocks
        
        Args:
            prices_dict: Dictionary of ticker -> DataFrame with prices
        
        Returns:
            DataFrame with daily returns for each stock
        """
        returns_data = {}
        
        for ticker, df in prices_dict.items():
            returns_data[ticker] = df['Close'].pct_change().dropna()
        
        self.returns = pd.DataFrame(returns_data)
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        return self.returns
    
    def portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Tuple of (return, volatility, sharpe_ratio)
        """
        # Annualized return
        portfolio_return = np.sum(self.mean_returns * weights) * self.trading_days
        
        # Annualized volatility
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix * self.trading_days, weights))
        )
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def generate_random_portfolios(self, num_portfolios: int = 10000) -> pd.DataFrame:
        """
        Generate random portfolios for efficient frontier
        
        Args:
            num_portfolios: Number of random portfolios to generate
        
        Returns:
            DataFrame with portfolio metrics
        """
        num_assets = len(self.mean_returns)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            # Random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Calculate metrics
            ret, vol, sharpe = self.portfolio_metrics(weights)
            results[0, i] = ret
            results[1, i] = vol
            results[2, i] = sharpe
        
        portfolios = pd.DataFrame({
            'Return': results[0],
            'Volatility': results[1],
            'Sharpe': results[2]
        })
        
        # Add weights
        for i, ticker in enumerate(self.mean_returns.index):
            portfolios[ticker] = [w[i] for w in weights_record]
        
        return portfolios
    
    def optimize_max_sharpe(self) -> Dict:
        """
        Optimize portfolio for maximum Sharpe ratio
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        num_assets = len(self.mean_returns)
        
        def neg_sharpe(weights):
            ret, vol, sharpe = self.portfolio_metrics(weights)
            return -sharpe
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        init_guess = num_assets * [1. / num_assets]
        
        # Optimize
        result = minimize(
            neg_sharpe,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_metrics(optimal_weights)
        
        return {
            'weights': dict(zip(self.mean_returns.index, optimal_weights)),
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        }
    
    def optimize_min_volatility(self) -> Dict:
        """
        Optimize portfolio for minimum volatility
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        num_assets = len(self.mean_returns)
        
        def portfolio_volatility(weights):
            return np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * self.trading_days, weights))
            )
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        init_guess = num_assets * [1. / num_assets]
        
        # Optimize
        result = minimize(
            portfolio_volatility,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_metrics(optimal_weights)
        
        return {
            'weights': dict(zip(self.mean_returns.index, optimal_weights)),
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        }
    
    def optimize_target_return(self, target_return: float) -> Dict:
        """
        Optimize portfolio for target return with minimum risk
        
        Args:
            target_return: Target annual return
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        num_assets = len(self.mean_returns)
        
        def portfolio_volatility(weights):
            return np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * self.trading_days, weights))
            )
        
        # Constraints: weights sum to 1 and achieve target return
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) * self.trading_days - target_return}
        )
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        init_guess = num_assets * [1. / num_assets]
        
        # Optimize
        result = minimize(
            portfolio_volatility,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_metrics(optimal_weights)
            
            return {
                'weights': dict(zip(self.mean_returns.index, optimal_weights)),
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe
            }
        else:
            return None
    
    def efficient_frontier(self, num_points: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            num_points: Number of points on the frontier
        
        Returns:
            DataFrame with efficient frontier points
        """
        # Get min and max returns
        min_ret = self.mean_returns.min() * self.trading_days
        max_ret = self.mean_returns.max() * self.trading_days
        
        target_returns = np.linspace(min_ret, max_ret, num_points)
        
        frontier_volatility = []
        frontier_returns = []
        frontier_sharpe = []
        
        for target in target_returns:
            result = self.optimize_target_return(target)
            if result is not None:
                frontier_returns.append(result['return'])
                frontier_volatility.append(result['volatility'])
                frontier_sharpe.append(result['sharpe'])
        
        return pd.DataFrame({
            'Return': frontier_returns,
            'Volatility': frontier_volatility,
            'Sharpe': frontier_sharpe
        })
    
    def equal_weight_portfolio(self) -> Dict:
        """
        Calculate equal-weight portfolio
        
        Returns:
            Dictionary with equal weights and metrics
        """
        num_assets = len(self.mean_returns)
        weights = np.array([1. / num_assets] * num_assets)
        
        ret, vol, sharpe = self.portfolio_metrics(weights)
        
        return {
            'weights': dict(zip(self.mean_returns.index, weights)),
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        }
    
    def market_cap_weighted_portfolio(self, market_caps: Dict[str, float]) -> Dict:
        """
        Calculate market cap weighted portfolio
        
        Args:
            market_caps: Dictionary of ticker -> market cap
        
        Returns:
            Dictionary with market cap weights and metrics
        """
        total_cap = sum(market_caps.values())
        weights = np.array([market_caps[ticker] / total_cap for ticker in self.mean_returns.index])
        
        ret, vol, sharpe = self.portfolio_metrics(weights)
        
        return {
            'weights': dict(zip(self.mean_returns.index, weights)),
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        }
    
    def risk_parity_portfolio(self) -> Dict:
        """
        Calculate risk parity portfolio (equal risk contribution)
        
        Returns:
            Dictionary with risk parity weights and metrics
        """
        num_assets = len(self.mean_returns)
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * self.trading_days, weights))
            )
            
            # Marginal risk contributions
            marginal_contrib = np.dot(self.cov_matrix * self.trading_days, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Objective: minimize variance of risk contributions
            return np.var(risk_contrib)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        init_guess = num_assets * [1. / num_assets]
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_metrics(optimal_weights)
        
        return {
            'weights': dict(zip(self.mean_returns.index, optimal_weights)),
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        }
    
    def compare_strategies(self, market_caps: Dict[str, float] = None) -> pd.DataFrame:
        """
        Compare different portfolio strategies
        
        Args:
            market_caps: Dictionary of market caps (optional)
        
        Returns:
            DataFrame comparing different strategies
        """
        strategies = {}
        
        # Equal weight
        strategies['Equal Weight'] = self.equal_weight_portfolio()
        
        # Max Sharpe
        strategies['Max Sharpe'] = self.optimize_max_sharpe()
        
        # Min Volatility
        strategies['Min Volatility'] = self.optimize_min_volatility()
        
        # Risk Parity
        strategies['Risk Parity'] = self.risk_parity_portfolio()
        
        # Market Cap (if provided)
        if market_caps is not None:
            strategies['Market Cap'] = self.market_cap_weighted_portfolio(market_caps)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            strategy: {
                'Return (%)': data['return'] * 100,
                'Volatility (%)': data['volatility'] * 100,
                'Sharpe Ratio': data['sharpe']
            }
            for strategy, data in strategies.items()
        }).T
        
        return comparison
