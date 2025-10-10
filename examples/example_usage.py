"""
Example Usage Scripts for AI Trading Assistant

This file demonstrates how to use individual modules programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lib.data_fetcher import StockDataFetcher
from lib.ml_predictor import LSTMPredictor, ProphetPredictor, EnsemblePredictor
from lib.portfolio_optimizer import PortfolioOptimizer
from lib.risk_assessment import RiskAnalyzer
from lib.technical_indicators import TechnicalIndicators
from lib.sentiment_analyzer import SentimentAnalyzer
from lib.visualizations import Visualizer


def example_1_fetch_data():
    """Example 1: Fetch stock data"""
    print("\n=== Example 1: Fetching Stock Data ===")
    
    fetcher = StockDataFetcher()
    
    # Fetch single stock
    df = fetcher.fetch_stock_data('AAPL', period='1y')
    print(f"Fetched {len(df)} days of data for AAPL")
    print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
    
    # Fetch multiple stocks
    stocks = ['AAPL', 'GOOGL', 'MSFT']
    data_dict = fetcher.fetch_multiple_stocks(stocks, period='1y')
    print(f"\nFetched data for {len(data_dict)} stocks")
    
    # Get stock info
    info = fetcher.get_stock_info('AAPL')
    print(f"\nCompany: {info['name']}")
    print(f"Sector: {info['sector']}")
    print(f"Market Cap: ${info['market_cap']:,.0f}")


def example_2_price_prediction():
    """Example 2: Price prediction with LSTM and Prophet"""
    print("\n=== Example 2: Price Prediction ===")
    
    # Fetch data
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data('AAPL', period='2y')
    prices = df['Close']
    
    # Initialize predictor
    ensemble = EnsemblePredictor()
    
    # Train models
    print("Training models...")
    train_results = ensemble.train_ensemble(prices, df)
    
    for model, result in train_results.items():
        print(f"{model.upper()}: {result['status']}")
    
    # Make predictions
    predictions = ensemble.predict_ensemble(prices, days=30)
    
    print(f"\nCurrent price: ${prices.iloc[-1]:.2f}")
    if predictions.get('ensemble') is not None:
        print(f"30-day forecast: ${predictions['ensemble'][-1]:.2f}")


def example_3_portfolio_optimization():
    """Example 3: Portfolio optimization"""
    print("\n=== Example 3: Portfolio Optimization ===")
    
    # Fetch data
    fetcher = StockDataFetcher()
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data_dict = fetcher.fetch_multiple_stocks(tickers, period='2y')
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Calculate returns
    returns = optimizer.calculate_returns(data_dict)
    
    # Optimize for max Sharpe ratio
    max_sharpe = optimizer.optimize_max_sharpe()
    print("Max Sharpe Portfolio:")
    print(f"  Expected Return: {max_sharpe['return']*100:.2f}%")
    print(f"  Volatility: {max_sharpe['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {max_sharpe['sharpe']:.2f}")
    print("\n  Allocation:")
    for ticker, weight in max_sharpe['weights'].items():
        print(f"    {ticker}: {weight*100:.2f}%")
    
    # Compare strategies
    comparison = optimizer.compare_strategies()
    print("\nStrategy Comparison:")
    print(comparison)


def example_4_risk_assessment():
    """Example 4: Risk assessment"""
    print("\n=== Example 4: Risk Assessment ===")
    
    # Fetch data
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data('AAPL', period='2y')
    prices = df['Close']
    
    # Initialize risk analyzer
    analyzer = RiskAnalyzer()
    
    # Generate risk report
    report = analyzer.generate_risk_report(prices)
    
    print("Risk Metrics:")
    print(f"  Volatility: {report['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {report['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {report['sortino_ratio']:.2f}")
    print(f"  VaR (95%): {report['var_historical']*100:.2f}%")
    print(f"  CVaR: {report['cvar']*100:.2f}%")
    print(f"  Max Drawdown: {report['max_drawdown']['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio: {report['calmar_ratio']:.2f}")


def example_5_technical_analysis():
    """Example 5: Technical analysis"""
    print("\n=== Example 5: Technical Analysis ===")
    
    # Fetch data
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data('AAPL', period='6mo')
    
    # Initialize technical analyzer
    technical = TechnicalIndicators()
    
    # Generate signals
    df_signals = technical.generate_combined_signals(df)
    
    # Get latest signals
    latest = df_signals.iloc[-1]
    
    print("Current Technical Indicators:")
    print(f"  RSI: {latest['RSI']:.2f}")
    print(f"  MACD: {latest['MACD']:.2f}")
    print(f"  MACD Signal: {latest['MACD_Signal']:.2f}")
    print(f"  Bollinger Upper: ${latest['BB_Upper']:.2f}")
    print(f"  Bollinger Lower: ${latest['BB_Lower']:.2f}")
    
    print("\nTrading Signals:")
    print(f"  RSI Signal: {latest['RSI_Signal']}")
    print(f"  MACD Signal: {latest['MACD_Signal_Line']}")
    print(f"  BB Signal: {latest['BB_Signal']}")
    print(f"  Combined Signal: {latest['Combined_Signal']}")
    
    # Backtest
    backtest = technical.backtest_strategy(df_signals)
    print(f"\nBacktest Results:")
    print(f"  Total Return: {backtest['total_return']:.2f}%")
    print(f"  Number of Trades: {backtest['num_trades']}")


def example_6_sentiment_analysis():
    """Example 6: Sentiment analysis"""
    print("\n=== Example 6: Sentiment Analysis ===")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze stock sentiment
    analysis = analyzer.analyze_stock_sentiment('AAPL', use_transformer=False)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    aggregate = analysis['aggregate']
    
    print("Sentiment Analysis:")
    print(f"  Overall Sentiment: {aggregate['overall_sentiment'].upper()}")
    print(f"  Sentiment Score: {aggregate['sentiment_score']:.1f}/100")
    print(f"  Positive Ratio: {aggregate['positive_ratio']*100:.1f}%")
    print(f"  Negative Ratio: {aggregate['negative_ratio']*100:.1f}%")
    print(f"  Total Articles: {aggregate['total_articles']}")
    
    print("\nTop Keywords:")
    for keyword, freq in analysis['keywords'][:5]:
        print(f"  {keyword}: {freq}")
    
    signal = analysis['signal']
    signal_text = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")
    print(f"\nSentiment-Based Signal: {signal_text}")


def example_7_visualizations():
    """Example 7: Creating visualizations"""
    print("\n=== Example 7: Visualizations ===")
    
    # Fetch data
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data('AAPL', period='1y')
    
    # Initialize visualizer
    viz = Visualizer()
    
    # Create candlestick chart
    fig = viz.plot_candlestick(df, title="AAPL Price History")
    
    # Save to file
    fig.write_html('/home/ubuntu/ai_trading_assistant/examples/candlestick_chart.html')
    print("Candlestick chart saved to examples/candlestick_chart.html")
    
    # Create technical analysis chart
    technical = TechnicalIndicators()
    df_indicators = technical.generate_combined_signals(df)
    
    fig = viz.plot_price_with_indicators(df_indicators, title="AAPL Technical Analysis")
    fig.write_html('/home/ubuntu/ai_trading_assistant/examples/technical_chart.html')
    print("Technical analysis chart saved to examples/technical_chart.html")


def example_8_complete_analysis():
    """Example 8: Complete analysis workflow"""
    print("\n=== Example 8: Complete Analysis Workflow ===")
    
    ticker = 'AAPL'
    print(f"\nAnalyzing {ticker}...")
    
    # 1. Fetch data
    fetcher = StockDataFetcher()
    df = fetcher.fetch_stock_data(ticker, period='2y')
    prices = df['Close']
    
    # 2. Technical analysis
    technical = TechnicalIndicators()
    df_signals = technical.generate_combined_signals(df)
    latest_signal = df_signals['Combined_Signal'].iloc[-1]
    technical_signal = "BUY" if latest_signal > 0 else ("SELL" if latest_signal < 0 else "HOLD")
    
    # 3. Risk assessment
    analyzer = RiskAnalyzer()
    risk_report = analyzer.generate_risk_report(prices)
    
    # 4. Sentiment analysis
    sentiment_analyzer = SentimentAnalyzer()
    sentiment = sentiment_analyzer.analyze_stock_sentiment(ticker, use_transformer=False)
    
    # 5. Print summary
    print("\n" + "="*50)
    print(f"ANALYSIS SUMMARY FOR {ticker}")
    print("="*50)
    
    print(f"\nCurrent Price: ${prices.iloc[-1]:.2f}")
    
    print("\nRisk Metrics:")
    print(f"  Sharpe Ratio: {risk_report['sharpe_ratio']:.2f}")
    print(f"  Volatility: {risk_report['volatility']*100:.2f}%")
    print(f"  Max Drawdown: {risk_report['max_drawdown']['max_drawdown_pct']:.2f}%")
    
    print("\nSignals:")
    print(f"  Technical: {technical_signal}")
    
    if 'error' not in sentiment:
        sentiment_signal = "BUY" if sentiment['signal'] == 1 else ("SELL" if sentiment['signal'] == -1 else "HOLD")
        print(f"  Sentiment: {sentiment_signal}")
        print(f"  Sentiment Score: {sentiment['aggregate']['sentiment_score']:.1f}/100")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    print("AI Trading Assistant - Example Usage")
    print("="*50)
    
    # Run examples (comment out ones you don't want to run)
    
    example_1_fetch_data()
    # example_2_price_prediction()  # Takes 2-5 minutes
    example_3_portfolio_optimization()
    example_4_risk_assessment()
    example_5_technical_analysis()
    # example_6_sentiment_analysis()  # Requires internet
    # example_7_visualizations()
    example_8_complete_analysis()
    
    print("\n" + "="*50)
    print("Examples completed!")
