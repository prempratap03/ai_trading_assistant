"""
AI-Powered Trading/Investment Assistant
Main Streamlit Application

Team Members:
- Prem Pratap (22070126078)
- Punit Chetwani (22070126079)
- Zaheer Khan (22070126066)
"""
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.data_fetcher import StockDataFetcher
from lib.ml_predictor import LSTMPredictor, ProphetPredictor, EnsemblePredictor
from lib.portfolio_optimizer import PortfolioOptimizer
from lib.risk_assessment import RiskAnalyzer
from lib.technical_indicators import TechnicalIndicators
from lib.sentiment_analyzer import SentimentAnalyzer
from lib.visualizations import Visualizer

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))
import config

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = StockDataFetcher()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'technical' not in st.session_state:
    st.session_state.technical = TechnicalIndicators()
if 'risk_analyzer' not in st.session_state:
    st.session_state.risk_analyzer = RiskAnalyzer()
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📈 AI Trading Assistant</div>', unsafe_allow_html=True)
    st.markdown("**Team:** Prem Pratap (22070126078), Punit Chetwani (22070126079), Zaheer Khan (22070126066)")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1753/1753732.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select a feature:",
            [
                "📊 Dashboard",
                "🔮 Price Prediction",
                "💼 Portfolio Optimization",
                "⚠️ Risk Assessment",
                "📉 Technical Analysis",
                "📰 Sentiment Analysis",
                "ℹ️ About"
            ]
        )
        
        st.markdown("---")
        
        # Stock selection
        st.subheader("Stock Selection")
        ticker = st.text_input("Enter Ticker Symbol", "AAPL").upper()
        
        # Period selection
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=4
        )
    
    # Route to pages
    if page == "📊 Dashboard":
        dashboard_page(ticker, period)
    elif page == "🔮 Price Prediction":
        prediction_page(ticker, period)
    elif page == "💼 Portfolio Optimization":
        portfolio_page()
    elif page == "⚠️ Risk Assessment":
        risk_page(ticker, period)
    elif page == "📉 Technical Analysis":
        technical_page(ticker, period)
    elif page == "📰 Sentiment Analysis":
        sentiment_page(ticker)
    elif page == "ℹ️ About":
        about_page()


def dashboard_page(ticker: str, period: str):
    """Main dashboard with overview"""
    
    st.markdown('<div class="sub-header">📊 Stock Dashboard</div>', unsafe_allow_html=True)
    
    try:
        # Fetch data
        with st.spinner(f"Fetching data for {ticker}..."):
            df = st.session_state.data_fetcher.fetch_stock_data(ticker, period)
            info = st.session_state.data_fetcher.get_stock_info(ticker)
        
        # Display stock info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        
        with col2:
            change = st.session_state.data_fetcher.get_price_changes(df)
            st.metric("1D Change", f"{change['1d']:.2f}%", delta=f"{change['1d']:.2f}%")
        
        with col3:
            st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
        
        with col4:
            st.metric("52W High", f"${info.get('52w_high', 0):.2f}")
        
        # Stock information
        st.markdown("### Company Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {info.get('name', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        
        with col2:
            st.write(f"**Market Cap:** ${info.get('market_cap', 0):,.0f}")
            st.write(f"**P/E Ratio:** {info.get('pe_ratio', 0):.2f}")
            st.write(f"**Beta:** {info.get('beta', 0):.2f}")
        
        # Price chart
        st.markdown("### Price History")
        fig = st.session_state.visualizer.plot_candlestick(df, f"{ticker} Price History")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent performance
        st.markdown("### Recent Performance")
        changes = st.session_state.data_fetcher.get_price_changes(df)
        
        perf_df = pd.DataFrame({
            'Period': ['1 Day', '1 Week', '1 Month', '3 Months', '1 Year', 'YTD'],
            'Change (%)': [
                changes['1d'],
                changes['1w'],
                changes['1m'],
                changes['3m'],
                changes['1y'],
                changes['ytd']
            ]
        })
        
        st.dataframe(perf_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


def prediction_page(ticker: str, period: str):
    """Price prediction page"""
    
    st.markdown('<div class="sub-header">🔮 Price Prediction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        forecast_days = st.slider("Forecast Days", 7, 90, 30)
        use_ensemble = st.checkbox("Use Ensemble Model", value=True)
        
        run_prediction = st.button("Run Prediction", type="primary")
    
    if run_prediction:
        try:
            with st.spinner("Fetching data and training models..."):
                df = st.session_state.data_fetcher.fetch_stock_data(ticker, period)
                
                # Prepare data
                prices = df['Close']
                
                # Train models
                ensemble = EnsemblePredictor()
                
                st.info("Training LSTM and Prophet models... This may take a few minutes.")
                train_results = ensemble.train_ensemble(prices, df)
                
                # Show training status
                for model, result in train_results.items():
                    if result['status'] == 'success':
                        st.success(f"✓ {model.upper()} trained successfully")
                    else:
                        st.warning(f"✗ {model.upper()} training failed: {result.get('error', 'Unknown error')}")
                
                # Generate predictions
                predictions = ensemble.predict_ensemble(prices, forecast_days)
                
                # Plot predictions
                st.markdown("### Predictions")
                fig = st.session_state.visualizer.plot_predictions(
                    prices,
                    predictions,
                    f"{ticker} Price Predictions ({forecast_days} days)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary
                st.markdown("### Prediction Summary")
                
                cols = st.columns(3)
                
                if predictions.get('lstm') is not None:
                    with cols[0]:
                        st.metric("LSTM Final Price", f"${predictions['lstm'][-1]:.2f}")
                
                if predictions.get('prophet') is not None:
                    with cols[1]:
                        st.metric("Prophet Final Price", f"${predictions['prophet'][-1]:.2f}")
                
                if predictions.get('ensemble') is not None:
                    with cols[2]:
                        st.metric("Ensemble Final Price", f"${predictions['ensemble'][-1]:.2f}")
                
                # Show predictions table
                if predictions.get('ensemble') is not None:
                    pred_df = pd.DataFrame({
                        'Day': range(1, forecast_days + 1),
                        'Predicted Price': predictions['ensemble']
                    })
                    
                    with st.expander("View Detailed Predictions"):
                        st.dataframe(pred_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.info("Try selecting a different time period or stock.")


def portfolio_page():
    """Portfolio optimization page"""
    
    st.markdown('<div class="sub-header">💼 Portfolio Optimization</div>', unsafe_allow_html=True)
    
    # Stock selection
    st.markdown("### Select Stocks for Portfolio")
    
    default_stocks = config.DEFAULT_STOCKS
    selected_stocks = st.multiselect(
        "Choose stocks (minimum 2)",
        default_stocks,
        default=default_stocks[:5]
    )
    
    period = st.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1)
    
    if len(selected_stocks) < 2:
        st.warning("Please select at least 2 stocks for portfolio optimization.")
        return
    
    if st.button("Optimize Portfolio", type="primary"):
        try:
            with st.spinner("Fetching data and optimizing portfolio..."):
                # Fetch data
                data_dict = st.session_state.data_fetcher.fetch_multiple_stocks(
                    selected_stocks, period
                )
                
                if len(data_dict) < 2:
                    st.error("Could not fetch data for enough stocks. Please try different tickers.")
                    return
                
                # Initialize optimizer
                optimizer = PortfolioOptimizer()
                
                # Calculate returns
                returns = optimizer.calculate_returns(data_dict)
                
                # Generate random portfolios for efficient frontier
                st.info("Generating efficient frontier...")
                portfolios = optimizer.generate_random_portfolios(num_portfolios=5000)
                
                # Calculate optimal portfolios
                max_sharpe = optimizer.optimize_max_sharpe()
                min_vol = optimizer.optimize_min_volatility()
                equal_weight = optimizer.equal_weight_portfolio()
                risk_parity = optimizer.risk_parity_portfolio()
                
                optimal_points = {
                    'Max Sharpe': max_sharpe,
                    'Min Volatility': min_vol,
                    'Equal Weight': equal_weight,
                    'Risk Parity': risk_parity
                }
                
                # Display efficient frontier
                st.markdown("### Efficient Frontier")
                fig = st.session_state.visualizer.plot_efficient_frontier(portfolios, optimal_points)
                st.plotly_chart(fig, use_container_width=True)
                
                # Strategy comparison
                st.markdown("### Strategy Comparison")
                comparison = optimizer.compare_strategies()
                st.dataframe(comparison, use_container_width=True)
                
                # Display allocations
                st.markdown("### Portfolio Allocations")
                
                tabs = st.tabs(["Max Sharpe", "Min Volatility", "Equal Weight", "Risk Parity"])
                
                strategies = [max_sharpe, min_vol, equal_weight, risk_parity]
                
                for tab, strategy in zip(tabs, strategies):
                    with tab:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = st.session_state.visualizer.plot_portfolio_allocation(
                                strategy['weights']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.metric("Expected Return", f"{strategy['return']*100:.2f}%")
                            st.metric("Volatility", f"{strategy['volatility']*100:.2f}%")
                            st.metric("Sharpe Ratio", f"{strategy['sharpe']:.2f}")
                            
                            # Show weights
                            st.markdown("**Weights:**")
                            for ticker, weight in strategy['weights'].items():
                                st.write(f"{ticker}: {weight*100:.2f}%")
                
        except Exception as e:
            st.error(f"Error in portfolio optimization: {str(e)}")


def risk_page(ticker: str, period: str):
    """Risk assessment page"""
    
    st.markdown('<div class="sub-header">⚠️ Risk Assessment</div>', unsafe_allow_html=True)
    
    if st.button("Run Risk Analysis", type="primary"):
        try:
            with st.spinner("Analyzing risk metrics..."):
                # Fetch data
                df = st.session_state.data_fetcher.fetch_stock_data(ticker, period)
                prices = df['Close']
                
                # Generate risk report
                risk_report = st.session_state.risk_analyzer.generate_risk_report(prices)
                
                # Display metrics
                st.markdown("### Risk Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Volatility", f"{risk_report['volatility']*100:.2f}%")
                    st.metric("VaR (95%)", f"{risk_report['var_historical']*100:.2f}%")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{risk_report['sharpe_ratio']:.2f}")
                    st.metric("Sortino Ratio", f"{risk_report['sortino_ratio']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{risk_report['max_drawdown']['max_drawdown_pct']:.2f}%")
                    st.metric("Calmar Ratio", f"{risk_report['calmar_ratio']:.2f}")
                
                with col4:
                    st.metric("CVaR", f"{risk_report['cvar']*100:.2f}%")
                    st.metric("Ulcer Index", f"{risk_report['ulcer_index']:.2f}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Risk-Adjusted Returns")
                    fig = st.session_state.visualizer.plot_risk_metrics(risk_report)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### Drawdown Analysis")
                    fig = st.session_state.visualizer.plot_drawdown(prices)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown details
                st.markdown("### Maximum Drawdown Details")
                dd = risk_report['max_drawdown']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Peak Date:** {dd['peak_date']}")
                with col2:
                    st.write(f"**Trough Date:** {dd['trough_date']}")
                with col3:
                    if dd['recovery_date']:
                        st.write(f"**Recovery Date:** {dd['recovery_date']}")
                    else:
                        st.write("**Status:** Not recovered")
                
        except Exception as e:
            st.error(f"Error in risk analysis: {str(e)}")


def technical_page(ticker: str, period: str):
    """Technical analysis page"""
    
    st.markdown('<div class="sub-header">📉 Technical Analysis</div>', unsafe_allow_html=True)
    
    if st.button("Run Technical Analysis", type="primary"):
        try:
            with st.spinner("Calculating indicators..."):
                # Fetch data
                df = st.session_state.data_fetcher.fetch_stock_data(ticker, period)
                
                # Calculate indicators
                df_with_indicators = st.session_state.technical.generate_combined_signals(df)
                
                # Display chart
                st.markdown("### Technical Indicators")
                fig = st.session_state.visualizer.plot_price_with_indicators(
                    df_with_indicators,
                    f"{ticker} Technical Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Current signals
                st.markdown("### Current Trading Signals")
                
                latest = df_with_indicators.iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi_val = latest['RSI']
                    rsi_signal = "BUY" if rsi_val < 30 else ("SELL" if rsi_val > 70 else "HOLD")
                    st.metric("RSI", f"{rsi_val:.2f}", rsi_signal)
                
                with col2:
                    macd_signal = "BUY" if latest['MACD'] > latest['MACD_Signal'] else "SELL"
                    st.metric("MACD", f"{latest['MACD']:.2f}", macd_signal)
                
                with col3:
                    bb_signal = "BUY" if latest['Close'] < latest['BB_Lower'] else (
                        "SELL" if latest['Close'] > latest['BB_Upper'] else "HOLD"
                    )
                    st.metric("Bollinger Bands", "N/A", bb_signal)
                
                with col4:
                    combined = latest['Combined_Signal']
                    combined_signal = "BUY" if combined > 0 else ("SELL" if combined < 0 else "HOLD")
                    st.metric("Combined Signal", combined_signal, 
                             delta=f"Strength: {abs(combined)}")
                
                # Support and resistance
                st.markdown("### Support & Resistance Levels")
                sr_levels = st.session_state.technical.calculate_support_resistance(df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Support Level", f"${sr_levels['support']:.2f}")
                with col2:
                    st.metric("Current Price", f"${sr_levels['current_price']:.2f}")
                with col3:
                    st.metric("Resistance Level", f"${sr_levels['resistance']:.2f}")
                
                # Backtest
                st.markdown("### Strategy Backtest")
                backtest_results = st.session_state.technical.backtest_strategy(df_with_indicators)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{backtest_results['total_return']:.2f}%")
                with col2:
                    st.metric("Final Capital", f"${backtest_results['final_capital']:.2f}")
                with col3:
                    st.metric("Number of Trades", backtest_results['num_trades'])
                
        except Exception as e:
            st.error(f"Error in technical analysis: {str(e)}")


def sentiment_page(ticker: str):
    """Sentiment analysis page"""
    
    st.markdown('<div class="sub-header">📰 Sentiment Analysis</div>', unsafe_allow_html=True)
    
    use_transformer = st.checkbox("Use Advanced AI Model (slower but more accurate)")
    
    if st.button("Analyze Sentiment", type="primary"):
        try:
            with st.spinner("Fetching and analyzing news..."):
                # Analyze sentiment
                analysis = st.session_state.sentiment_analyzer.analyze_stock_sentiment(
                    ticker, use_transformer
                )
                
                if 'error' in analysis:
                    st.warning(f"Could not fetch news: {analysis['error']}")
                    return
                
                # Display aggregate sentiment
                st.markdown("### Overall Sentiment")
                
                aggregate = analysis['aggregate']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sentiment_emoji = "🟢" if aggregate['overall_sentiment'] == 'positive' else (
                        "🔴" if aggregate['overall_sentiment'] == 'negative' else "🟡"
                    )
                    st.metric("Sentiment", f"{sentiment_emoji} {aggregate['overall_sentiment'].upper()}")
                
                with col2:
                    st.metric("Sentiment Score", f"{aggregate['sentiment_score']:.1f}/100")
                
                with col3:
                    st.metric("Positive News", f"{aggregate['positive_ratio']*100:.1f}%")
                
                with col4:
                    st.metric("Total Articles", aggregate['total_articles'])
                
                # Trading signal
                signal = analysis['signal']
                signal_text = "BUY" if signal == 1 else ("SELL" if signal == -1 else "HOLD")
                signal_color = "green" if signal == 1 else ("red" if signal == -1 else "orange")
                
                st.markdown(f"### Sentiment-Based Signal: <span style='color:{signal_color}; font-size:2rem; font-weight:bold'>{signal_text}</span>", 
                           unsafe_allow_html=True)
                
                # Keywords
                st.markdown("### Trending Keywords")
                keywords_df = pd.DataFrame(analysis['keywords'], columns=['Keyword', 'Frequency'])
                st.bar_chart(keywords_df.set_index('Keyword'))
                
                # Recent articles
                st.markdown("### Recent News Articles")
                
                articles_df = pd.DataFrame(analysis['articles'])
                
                for _, article in articles_df.head(10).iterrows():
                    sentiment_color = "green" if article['sentiment'] == 'positive' else (
                        "red" if article['sentiment'] == 'negative' else "gray"
                    )
                    
                    st.markdown(f"**{article['title']}**")
                    st.markdown(f"<span style='color:{sentiment_color}'>● {article['sentiment'].upper()}</span> | "
                               f"Source: {article['source']} | "
                               f"Polarity: {article['polarity']:.2f}",
                               unsafe_allow_html=True)
                    st.markdown("---")
                
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")


def about_page():
    """About page"""
    
    st.markdown('<div class="sub-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## AI-Powered Trading/Investment Assistant
    
    This application provides comprehensive tools for stock market analysis and investment decision-making.
    
    ### 🎓 Team Members
    - **Prem Pratap** (22070126078)
    - **Punit Chetwani** (22070126079)
    - **Zaheer Khan** (22070126066)
    
    ### 🚀 Features
    
    1. **Dashboard** - Real-time stock data and company information
    2. **Price Prediction** - LSTM and Prophet ML models for forecasting
    3. **Portfolio Optimization** - Modern Portfolio Theory implementation
    4. **Risk Assessment** - Comprehensive risk metrics (VaR, Sharpe, etc.)
    5. **Technical Analysis** - RSI, MACD, Bollinger Bands, and trading signals
    6. **Sentiment Analysis** - News sentiment analysis with AI
    
    ### 🛠️ Technologies Used
    
    - **Frontend:** Streamlit
    - **Data:** yfinance
    - **ML:** TensorFlow, Prophet, scikit-learn
    - **Visualization:** Plotly
    - **NLP:** Transformers, TextBlob
    - **Optimization:** SciPy, cvxpy
    
    ### 📚 Documentation
    
    For detailed documentation, please refer to the `docs/` directory in the repository.
    
    ### ⚠️ Disclaimer
    
    This application is for educational purposes only. The predictions and recommendations 
    provided should not be considered as financial advice. Always do your own research 
    and consult with a qualified financial advisor before making investment decisions.
    
    ### 📧 Contact
    
    For questions or feedback, please contact the team members.
    """)


if __name__ == "__main__":
    main()
