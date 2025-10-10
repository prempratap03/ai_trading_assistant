"""
AI Trading Assistant Library
Contains core modules for trading analysis and prediction
"""

__version__ = '1.0.0'
__author__ = 'Prem Pratap, Punit Chetwani, Zaheer Khan'

from .data_fetcher import StockDataFetcher
from .ml_predictor import LSTMPredictor, ProphetPredictor, EnsemblePredictor
from .portfolio_optimizer import PortfolioOptimizer
from .risk_assessment import RiskAnalyzer
from .technical_indicators import TechnicalIndicators
from .sentiment_analyzer import SentimentAnalyzer
from .visualizations import Visualizer

__all__ = [
    'StockDataFetcher',
    'LSTMPredictor',
    'ProphetPredictor',
    'EnsemblePredictor',
    'PortfolioOptimizer',
    'RiskAnalyzer',
    'TechnicalIndicators',
    'SentimentAnalyzer',
    'Visualizer'
]
