# AI-Powered Trading/Investment Assistant 📈

A comprehensive FinTech application that leverages Machine Learning, Modern Portfolio Theory, and Advanced Analytics to provide intelligent stock market insights and investment recommendations.

## 🎓 Team Members

- **Prem Pratap** (22070126078)
- **Punit Chetwani** (22070126079)
- **Ayush Anand** (22070126025)
- **Paarth Chauhan** (22070126069)

## 🌟 Features

### 1. Interactive Dashboard
- Real-time stock data visualization
- Company information and key metrics
- Historical price charts with candlestick patterns
- Performance tracking across multiple timeframes

### 2. ML-Powered Price Prediction
- **LSTM Neural Networks** for time-series forecasting
- **Facebook Prophet** for robust trend analysis
- **Ensemble Models** combining multiple algorithms
- Configurable forecast periods (7-90 days)

### 3. Portfolio Optimization
- **Modern Portfolio Theory (MPT)** implementation
- Efficient Frontier calculation
- Multiple optimization strategies:
  - Maximum Sharpe Ratio
  - Minimum Volatility
  - Risk Parity
  - Equal Weight
- Interactive portfolio allocation visualization

### 4. Comprehensive Risk Assessment
- Value at Risk (VaR) - Historical & Parametric
- Conditional Value at Risk (CVaR)
- Sharpe, Sortino, and Calmar Ratios
- Maximum Drawdown Analysis
- Beta and Alpha calculations
- Rolling volatility and risk metrics

### 5. Technical Analysis
- **Technical Indicators:**
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - ATR, ADX, CCI
- Automated trading signals
- Support & Resistance levels
- Strategy backtesting

### 6. Sentiment Analysis
- News sentiment analysis using AI
- DistilBERT transformer model for accuracy
- Sentiment-based trading signals
- Trending keywords extraction
- Time-series sentiment tracking

## 🛠️ Technology Stack

### Frontend
- **Streamlit** - Interactive web interface
- **Plotly** - Interactive visualizations
- Custom CSS for enhanced UI/UX

### Data & APIs
- **yfinance** - Stock market data
- **NewsAPI** - Financial news (optional)

### Machine Learning
- **TensorFlow/Keras** - LSTM neural networks
- **Prophet** - Time-series forecasting
- **scikit-learn** - Data preprocessing and metrics
- **Transformers** - NLP sentiment analysis

### Analytics & Optimization
- **NumPy & Pandas** - Data manipulation
- **SciPy** - Statistical analysis
- **cvxpy** - Convex optimization

## 📁 Project Structure

```
ai_trading_assistant/
├── src/
│   ├── main.py                 # Main Streamlit application
│   └── lib/
│       ├── data_fetcher.py     # Data retrieval and preprocessing
│       ├── ml_predictor.py     # ML models (LSTM, Prophet, Ensemble)
│       ├── portfolio_optimizer.py  # MPT and portfolio optimization
│       ├── risk_assessment.py  # Risk metrics and analysis
│       ├── technical_indicators.py # Technical analysis tools
│       ├── sentiment_analyzer.py   # News sentiment analysis
│       └── visualizations.py   # Plotly visualization components
├── config/
│   └── config.py               # Configuration settings
├── docs/
│   ├── USER_GUIDE.md          # Comprehensive user guide
│   └── DEPLOYMENT.md          # Deployment instructions
├── examples/
│   └── example_usage.py       # Example code snippets
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ai_trading_assistant
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run src/main.py
```

5. **Access the application:**
Open your browser and navigate to `http://localhost:8501`

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t ai-trading-assistant .
```

### Run Container
```bash
docker run -p 8501:8501 ai-trading-assistant
```

## ☁️ Streamlit Cloud Deployment

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `src/main.py` as the main file
5. Deploy!

## 📖 Usage Guide

### 1. Dashboard
- Select a stock ticker (e.g., AAPL, GOOGL, MSFT)
- Choose time period
- View real-time data and charts

### 2. Price Prediction
- Select forecast horizon (7-90 days)
- Choose ensemble model for best results
- Compare LSTM vs Prophet predictions

### 3. Portfolio Optimization
- Select 2+ stocks for your portfolio
- View efficient frontier
- Compare optimization strategies
- Get optimal allocation recommendations

### 4. Risk Assessment
- Analyze comprehensive risk metrics
- View drawdown charts
- Understand risk-adjusted returns

### 5. Technical Analysis
- View multiple technical indicators
- Get automated trading signals
- Backtest strategies

### 6. Sentiment Analysis
- Analyze news sentiment
- Get sentiment-based trading signals
- Track trending keywords

## 🔧 Configuration

Edit `config/config.py` to customize:
- Default stock tickers
- ML model parameters
- Risk-free rate
- Technical indicator periods
- UI settings

## 📊 Example Code

### Fetch Stock Data
```python
from lib.data_fetcher import StockDataFetcher

fetcher = StockDataFetcher()
df = fetcher.fetch_stock_data('AAPL', period='1y')
```

### Run Price Prediction
```python
from lib.ml_predictor import EnsemblePredictor

predictor = EnsemblePredictor()
predictions = predictor.predict_ensemble(prices, days=30)
```

### Optimize Portfolio
```python
from lib.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
optimal = optimizer.optimize_max_sharpe()
```

## 📚 Documentation

- [User Guide](docs/USER_GUIDE.md) - Detailed feature documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [API Reference](docs/API.md) - Module and function documentation

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/
```

## 🤝 Contributing

This is an academic project. For contributions or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is created for educational purposes as part of a FinTech assignment.

## ⚠️ Disclaimer

**IMPORTANT:** This application is for educational and research purposes only. 

- The predictions and recommendations provided are not financial advice
- Past performance does not guarantee future results
- Always conduct your own research
- Consult with qualified financial advisors before making investment decisions
- The developers are not responsible for any financial losses

## 🐛 Known Issues

- ML model training can be slow on CPU-only systems
- News sentiment requires internet connection
- Some stocks may have limited historical data

## 🔮 Future Enhancements

- [ ] Real-time streaming data
- [ ] Crypto currency support
- [ ] Advanced options analysis
- [ ] Social media sentiment integration
- [ ] Mobile app version
- [ ] Multi-language support


