"""
Configuration settings for AI Trading Assistant
"""

# Stock Data Settings
DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'JNJ']
DEFAULT_PERIOD = '2y'
DEFAULT_INTERVAL = '1d'

# ML Model Settings
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_SEQUENCE_LENGTH = 60
TRAIN_TEST_SPLIT = 0.8

# Prophet Settings
PROPHET_PERIODS = 90  # Days to forecast
PROPHET_CONFIDENCE = 0.95

# Portfolio Optimization Settings
RISK_FREE_RATE = 0.02  # 2% risk-free rate
NUM_PORTFOLIOS = 10000  # Number of random portfolios to generate
TRADING_DAYS = 252

# Risk Assessment Settings
CONFIDENCE_LEVEL = 0.95  # For VaR calculation
VAR_METHOD = 'historical'  # 'historical' or 'parametric'

# Technical Indicators Settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Sentiment Analysis Settings
NEWS_API_KEY = None  # Set your NewsAPI key here or use environment variable
MAX_NEWS_ARTICLES = 100
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'

# UI Settings
PAGE_TITLE = "AI Trading Assistant"
PAGE_ICON = "📈"
LAYOUT = "wide"

# Cache Settings
CACHE_TTL = 3600  # 1 hour in seconds

# Color Schemes
COLOR_PROFIT = '#00ff00'
COLOR_LOSS = '#ff0000'
COLOR_NEUTRAL = '#ffa500'
