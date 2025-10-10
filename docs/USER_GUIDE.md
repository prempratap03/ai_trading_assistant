# AI Trading Assistant - User Guide 📖

## Table of Contents
1. [Getting Started](#getting-started)
2. [Dashboard](#dashboard)
3. [Price Prediction](#price-prediction)
4. [Portfolio Optimization](#portfolio-optimization)
5. [Risk Assessment](#risk-assessment)
6. [Technical Analysis](#technical-analysis)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Tips & Best Practices](#tips--best-practices)

---

## Getting Started

### First Launch

1. **Run the application:**
   ```bash
   streamlit run src/main.py
   ```

2. **Navigate the sidebar:**
   - Use the radio buttons to switch between features
   - Enter stock ticker symbols (e.g., AAPL, GOOGL, MSFT)
   - Select time periods for analysis

3. **Stock Ticker Format:**
   - US stocks: Use standard ticker (AAPL, TSLA)
   - Case insensitive (aapl = AAPL)

---

## Dashboard

### Overview
The Dashboard provides a comprehensive overview of selected stocks with real-time data and visualizations.

### Features

#### 1. Key Metrics Display
- **Current Price:** Latest closing price
- **Daily Change:** Percentage change from previous day
- **Volume:** Trading volume
- **52-Week High:** Highest price in last year

#### 2. Company Information
- Name, Sector, and Industry
- Market Capitalization
- P/E Ratio (Price-to-Earnings)
- Beta (Market correlation)

#### 3. Interactive Price Chart
- Candlestick visualization
- Zoom and pan capabilities
- Hover for detailed information
- Date range selection

#### 4. Performance Metrics
- 1 Day, 1 Week, 1 Month changes
- 3 Month, 1 Year, YTD performance
- Color-coded gains/losses

### Usage Tips
- Compare multiple periods to identify trends
- Check volume for liquidity assessment
- Use beta to understand market correlation

---

## Price Prediction

### Overview
Uses advanced Machine Learning models to forecast future stock prices.

### Models

#### 1. LSTM (Long Short-Term Memory)
- Deep learning neural network
- Captures complex temporal patterns
- Best for short to medium-term predictions

#### 2. Prophet
- Developed by Facebook
- Handles trends and seasonality
- Robust to missing data

#### 3. Ensemble
- Combines LSTM and Prophet
- Reduces individual model bias
- **Recommended for most accurate predictions**

### Step-by-Step Usage

1. **Select Stock:** Enter ticker in sidebar
2. **Choose Period:** Select historical data period
3. **Set Forecast Days:** Use slider (7-90 days)
4. **Enable Ensemble:** Check box for best results
5. **Run Prediction:** Click "Run Prediction" button

### Understanding Results

#### Prediction Chart
- **Blue Line:** Historical prices
- **Red Dashed:** LSTM predictions
- **Green Dashed:** Prophet predictions
- **Orange Dotted:** Ensemble predictions

#### Metrics
- **Final Price:** Predicted price at end of forecast
- **Confidence:** Model training success rate

### Best Practices
- Use 2+ years of historical data for training
- Ensemble predictions are most reliable
- Consider external factors (news, earnings)
- Use predictions as one input, not sole decision maker

### Limitations
- Cannot predict black swan events
- Market manipulation not captured
- Training time: 2-5 minutes

---

## Portfolio Optimization

### Overview
Implements Modern Portfolio Theory to find optimal asset allocation.

### Theory Background

**Modern Portfolio Theory (MPT):**
- Maximize return for given risk level
- Minimize risk for target return
- Diversification reduces portfolio risk

**Key Concepts:**
- **Expected Return:** Anticipated portfolio gain
- **Volatility:** Risk measure (standard deviation)
- **Sharpe Ratio:** Risk-adjusted return

### Optimization Strategies

#### 1. Maximum Sharpe Ratio
- **Goal:** Best risk-adjusted returns
- **Best For:** Balanced investors
- **Characteristics:** Moderate risk, good returns

#### 2. Minimum Volatility
- **Goal:** Lowest possible risk
- **Best For:** Conservative investors
- **Characteristics:** Lower returns, stable

#### 3. Equal Weight
- **Goal:** Simple diversification
- **Best For:** Beginners
- **Characteristics:** 1/N allocation

#### 4. Risk Parity
- **Goal:** Equal risk contribution
- **Best For:** Advanced investors
- **Characteristics:** Risk-balanced allocation

### Step-by-Step Usage

1. **Select Stocks:** Choose 2+ stocks
2. **Set Period:** Historical data for calculations
3. **Run Optimization:** Click "Optimize Portfolio"
4. **Review Results:**
   - Efficient Frontier graph
   - Strategy comparison table
   - Individual allocations

### Interpreting Results

#### Efficient Frontier
- **X-axis:** Volatility (Risk)
- **Y-axis:** Return
- **Color:** Sharpe Ratio (darker = better)
- **Stars:** Optimal portfolios

#### Portfolio Allocation
- Pie chart showing asset distribution
- Percentage weights for each stock
- Expected return and risk metrics

### Best Practices
- Include 5-10 stocks for proper diversification
- Use different sectors to reduce correlation
- Rebalance quarterly based on new data
- Consider transaction costs in real implementation

---

## Risk Assessment

### Overview
Comprehensive risk analysis using industry-standard metrics.

### Risk Metrics Explained

#### 1. Volatility
- **Definition:** Standard deviation of returns
- **Interpretation:** Higher = riskier
- **Typical Range:** 15-30% annually for stocks

#### 2. Value at Risk (VaR)
- **Definition:** Maximum expected loss at confidence level
- **95% VaR = 5%:** 5% chance of losing more than VaR
- **Use Case:** Risk budgeting

#### 3. Conditional VaR (CVaR)
- **Definition:** Expected loss when VaR is exceeded
- **Also Called:** Expected Shortfall
- **Use Case:** Worst-case scenario planning

#### 4. Sharpe Ratio
- **Formula:** (Return - Risk_free) / Volatility
- **Interpretation:** 
  - \> 1: Good
  - \> 2: Very Good
  - \> 3: Excellent

#### 5. Sortino Ratio
- **Similar to Sharpe** but only considers downside risk
- **Better for:** Asymmetric return distributions

#### 6. Maximum Drawdown
- **Definition:** Largest peak-to-trough decline
- **Components:**
  - Peak date
  - Trough date
  - Recovery date
  - Duration

#### 7. Calmar Ratio
- **Formula:** Return / Max Drawdown
- **Higher is better**

#### 8. Beta
- **Definition:** Correlation with market
- **Interpretation:**
  - β = 1: Moves with market
  - β > 1: More volatile than market
  - β < 1: Less volatile than market

#### 9. Alpha
- **Definition:** Excess return vs. expected return
- **Positive α:** Outperformance

### Usage Steps

1. **Select Stock and Period**
2. **Run Risk Analysis**
3. **Review Metrics**
4. **Analyze Drawdown Chart**
5. **Compare with Benchmarks**

### Risk Profile Interpretation

| Sharpe Ratio | Risk Profile |
|--------------|--------------|
| < 0 | Poor - Loss |
| 0 - 1 | Sub-optimal |
| 1 - 2 | Good |
| 2 - 3 | Very Good |
| > 3 | Excellent |

---

## Technical Analysis

### Overview
Automated technical analysis with multiple indicators and trading signals.

### Technical Indicators

#### 1. RSI (Relative Strength Index)
- **Range:** 0-100
- **Signals:**
  - < 30: Oversold (Buy signal)
  - \> 70: Overbought (Sell signal)
- **Best Use:** Mean reversion strategies

#### 2. MACD (Moving Average Convergence Divergence)
- **Components:**
  - MACD Line
  - Signal Line
  - Histogram
- **Signals:**
  - MACD crosses above Signal: Buy
  - MACD crosses below Signal: Sell

#### 3. Bollinger Bands
- **Components:**
  - Upper Band (MA + 2σ)
  - Middle Band (20-day MA)
  - Lower Band (MA - 2σ)
- **Signals:**
  - Price touches lower band: Buy
  - Price touches upper band: Sell

#### 4. Support & Resistance
- **Support:** Price floor (buy interest)
- **Resistance:** Price ceiling (sell pressure)
- **Breakouts:** Strong signals when levels broken

### Trading Signals

#### Signal Interpretation
- **BUY:** All or majority of indicators bullish
- **SELL:** All or majority of indicators bearish
- **HOLD:** Mixed or neutral signals

#### Combined Signal
- Aggregates all indicators
- Majority vote system
- Strength indicator shows conviction

### Backtesting

The application includes strategy backtesting:
- Initial capital: $10,000
- Follows combined signals
- Reports:
  - Final capital
  - Total return %
  - Number of trades

### Best Practices
- Don't rely on single indicator
- Confirm signals across multiple indicators
- Consider volume for confirmation
- Use with fundamental analysis
- Set stop-losses in real trading

---

## Sentiment Analysis

### Overview
Analyzes news sentiment using Natural Language Processing and AI.

### How It Works

1. **News Collection:** Fetches recent articles about stock
2. **Sentiment Analysis:** AI models analyze tone
3. **Aggregation:** Combines individual sentiments
4. **Signal Generation:** Creates trading recommendation

### Analysis Methods

#### 1. Basic (TextBlob)
- Fast processing
- Good for quick overview
- Uses rule-based approach

#### 2. Advanced (DistilBERT)
- Deep learning transformer model
- More accurate
- Slower processing
- **Recommended for important decisions**

### Metrics

#### Sentiment Score (0-100)
- 0-40: Negative sentiment
- 40-60: Neutral sentiment
- 60-100: Positive sentiment

#### Polarity (-1 to +1)
- -1: Very negative
- 0: Neutral
- +1: Very positive

#### Ratios
- Positive ratio: % positive articles
- Negative ratio: % negative articles
- Neutral ratio: % neutral articles

### Trading Signals

- **BUY:** Sentiment score > 60
- **HOLD:** Sentiment score 40-60
- **SELL:** Sentiment score < 40

### Trending Keywords
Identifies most discussed topics:
- Product launches
- Earnings reports
- Legal issues
- Market trends

### Usage Tips

1. **Cross-reference with price:** 
   - Does sentiment match price movement?
   - Divergence might signal opportunity

2. **Consider recency:**
   - More recent news is more relevant
   - Old news already priced in

3. **Volume matters:**
   - More articles = higher conviction
   - Single article = less reliable

4. **Context is key:**
   - Read actual headlines
   - Understand what's driving sentiment

### Limitations
- Cannot predict future news
- May lag market reaction
- Subject to media bias
- Limited to English articles

---

## Tips & Best Practices

### General Usage

1. **Start Simple**
   - Begin with dashboard
   - Understand one stock thoroughly
   - Graduate to complex features

2. **Cross-Validation**
   - Don't rely on single feature
   - Combine technical + fundamental + sentiment
   - Verify signals across timeframes

3. **Risk Management**
   - Always check risk metrics
   - Understand maximum drawdown
   - Set stop-loss levels
   - Never invest more than you can afford to lose

### Feature-Specific Tips

#### Price Prediction
- Use ensemble models
- Longer history = better predictions
- Validate against technical analysis
- Update predictions weekly

#### Portfolio Optimization
- Rebalance quarterly
- Monitor correlation changes
- Consider transaction costs
- Tax implications matter

#### Technical Analysis
- Multiple timeframes for confirmation
- Volume confirms signals
- False signals are common
- Combine with fundamentals

#### Sentiment Analysis
- Recent news more important
- Large volume = higher confidence
- Read actual articles
- Context matters

### Advanced Strategies

1. **Mean Reversion**
   - Use RSI + Bollinger Bands
   - Buy oversold, sell overbought
   - Works in ranging markets

2. **Trend Following**
   - Use MACD + Moving Averages
   - Follow the trend
   - Works in trending markets

3. **Risk Parity Portfolio**
   - Equal risk contribution
   - Better diversification
   - More stable returns

### Common Mistakes to Avoid

1. ❌ **Over-trading:** Too many signals
2. ❌ **Ignoring risk:** Focus only on returns
3. ❌ **Single indicator:** Need confirmation
4. ❌ **Emotional decisions:** Stick to analysis
5. ❌ **Ignoring costs:** Fees eat returns
6. ❌ **No stop-loss:** Protect capital
7. ❌ **Past performance:** Not future guarantee

### Performance Monitoring

Track these metrics:
- Monthly returns
- Sharpe ratio
- Maximum drawdown
- Win rate
- Average gain/loss

### When to Seek Help

Consult financial advisor if:
- Large investment amounts
- Complex tax situations
- Retirement planning
- Estate planning
- Uncertain about risk tolerance

---

## Troubleshooting

### Common Issues

**Data Not Loading:**
- Check internet connection
- Verify ticker symbol is correct
- Try different time period

**Slow Performance:**
- ML models take 2-5 minutes
- Use smaller date ranges
- Close other applications

**No News Found:**
- Stock might not have recent news
- Try larger companies
- Check ticker symbol

**Prediction Errors:**
- Insufficient historical data
- Try longer time period
- Some stocks unpredictable

---

## Keyboard Shortcuts

- `r` - Rerun application
- `c` - Clear cache
- `s` - Settings
- `?` - Help

---

## Getting Support

For issues or questions:
1. Check this user guide
2. Review example code
3. Contact development team

---

**Remember: This tool is for educational purposes. Always do your own research and consult professionals for investment decisions.**
