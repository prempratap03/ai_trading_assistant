"""
Sentiment Analysis Module
Analyzes market news and social media sentiment
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Analyze sentiment from news and text data"""
    
    def __init__(self):
        self.transformer_model = None
    
    def analyze_text_basic(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob (basic approach)
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'confidence': abs(polarity)
            }
        except Exception as e:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': 'neutral',
                'confidence': 0,
                'error': str(e)
            }
    
    def analyze_text_transformer(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model (DistilBERT)
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment scores
        """
        try:
            from transformers import pipeline
            
            # Initialize model if not already done
            if self.transformer_model is None:
                self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            
            result = self.transformer_model(text[:512])[0]  # Limit text length
            
            # Convert to standard format
            sentiment = result['label'].lower()
            confidence = result['score']
            
            # Convert to polarity scale
            polarity = confidence if sentiment == 'positive' else -confidence
            
            return {
                'polarity': polarity,
                'sentiment': sentiment,
                'confidence': confidence,
                'model': 'transformer'
            }
        
        except Exception as e:
            # Fallback to basic method
            return self.analyze_text_basic(text)
    
    def analyze_news_batch(self, news_list: List[Dict], use_transformer: bool = False) -> pd.DataFrame:
        """
        Analyze sentiment for multiple news articles
        
        Args:
            news_list: List of dictionaries with 'title' and 'description'
            use_transformer: Whether to use transformer model
        
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for news in news_list:
            text = f"{news.get('title', '')} {news.get('description', '')}"
            
            if use_transformer:
                sentiment = self.analyze_text_transformer(text)
            else:
                sentiment = self.analyze_text_basic(text)
            
            results.append({
                'title': news.get('title', 'N/A'),
                'date': news.get('publishedAt', 'N/A'),
                'source': news.get('source', {}).get('name', 'N/A'),
                'polarity': sentiment['polarity'],
                'sentiment': sentiment['sentiment'],
                'confidence': sentiment['confidence']
            })
        
        return pd.DataFrame(results)
    
    def calculate_aggregate_sentiment(self, sentiment_df: pd.DataFrame) -> Dict:
        """
        Calculate aggregate sentiment metrics
        
        Args:
            sentiment_df: DataFrame with sentiment analysis
        
        Returns:
            Dictionary with aggregate metrics
        """
        if sentiment_df.empty:
            return {
                'overall_sentiment': 'neutral',
                'avg_polarity': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0,
                'sentiment_score': 0
            }
        
        total = len(sentiment_df)
        positive = len(sentiment_df[sentiment_df['sentiment'] == 'positive'])
        negative = len(sentiment_df[sentiment_df['sentiment'] == 'negative'])
        neutral = len(sentiment_df[sentiment_df['sentiment'] == 'neutral'])
        
        avg_polarity = sentiment_df['polarity'].mean()
        
        # Calculate overall sentiment
        if avg_polarity > 0.1:
            overall = 'positive'
        elif avg_polarity < -0.1:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        # Sentiment score (0-100)
        sentiment_score = ((avg_polarity + 1) / 2) * 100
        
        return {
            'overall_sentiment': overall,
            'avg_polarity': avg_polarity,
            'positive_ratio': positive / total,
            'negative_ratio': negative / total,
            'neutral_ratio': neutral / total,
            'sentiment_score': sentiment_score,
            'total_articles': total
        }
    
    def get_trending_keywords(self, news_list: List[Dict], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extract trending keywords from news
        
        Args:
            news_list: List of news articles
            top_n: Number of top keywords to return
        
        Returns:
            List of (keyword, frequency) tuples
        """
        from collections import Counter
        import re
        
        # Common words to exclude
        stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ])
        
        words = []
        for news in news_list:
            text = f"{news.get('title', '')} {news.get('description', '')}"
            # Extract words
            text_words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            words.extend([w for w in text_words if w not in stop_words])
        
        # Count frequencies
        word_freq = Counter(words)
        
        return word_freq.most_common(top_n)
    
    def sentiment_time_series(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment time series
        
        Args:
            sentiment_df: DataFrame with sentiment and dates
        
        Returns:
            DataFrame with daily sentiment aggregates
        """
        try:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            
            daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
                'polarity': 'mean',
                'confidence': 'mean',
                'sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
            }).reset_index()
            
            return daily_sentiment
        
        except Exception as e:
            return pd.DataFrame()
    
    def generate_sentiment_signal(self, sentiment_score: float, threshold_buy: float = 60,
                                 threshold_sell: float = 40) -> int:
        """
        Generate trading signal based on sentiment
        
        Args:
            sentiment_score: Sentiment score (0-100)
            threshold_buy: Buy threshold
            threshold_sell: Sell threshold
        
        Returns:
            Signal (1=Buy, -1=Sell, 0=Hold)
        """
        if sentiment_score > threshold_buy:
            return 1  # Buy
        elif sentiment_score < threshold_sell:
            return -1  # Sell
        else:
            return 0  # Hold
    
    def fetch_news_yfinance(self, ticker: str) -> List[Dict]:
        """
        Fetch news for a ticker using yfinance
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            List of news articles
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            news = stock.news
            
            return news if news else []
        
        except Exception as e:
            return []
    
    def analyze_stock_sentiment(self, ticker: str, use_transformer: bool = False) -> Dict:
        """
        Complete sentiment analysis for a stock
        
        Args:
            ticker: Stock ticker symbol
            use_transformer: Whether to use transformer model
        
        Returns:
            Dictionary with complete sentiment analysis
        """
        # Fetch news
        news = self.fetch_news_yfinance(ticker)
        
        if not news:
            return {
                'ticker': ticker,
                'sentiment': 'neutral',
                'error': 'No news found'
            }
        
        # Analyze sentiment
        sentiment_df = self.analyze_news_batch(news, use_transformer)
        
        # Calculate aggregates
        aggregate = self.calculate_aggregate_sentiment(sentiment_df)
        
        # Get keywords
        keywords = self.get_trending_keywords(news, top_n=10)
        
        # Generate signal
        signal = self.generate_sentiment_signal(aggregate['sentiment_score'])
        
        return {
            'ticker': ticker,
            'aggregate': aggregate,
            'keywords': keywords,
            'signal': signal,
            'articles': sentiment_df.to_dict('records'),
            'num_articles': len(news)
        }
    
    def compare_stock_sentiments(self, tickers: List[str], use_transformer: bool = False) -> pd.DataFrame:
        """
        Compare sentiment across multiple stocks
        
        Args:
            tickers: List of ticker symbols
            use_transformer: Whether to use transformer model
        
        Returns:
            DataFrame comparing sentiments
        """
        results = []
        
        for ticker in tickers:
            analysis = self.analyze_stock_sentiment(ticker, use_transformer)
            
            if 'error' not in analysis:
                results.append({
                    'ticker': ticker,
                    'sentiment': analysis['aggregate']['overall_sentiment'],
                    'sentiment_score': analysis['aggregate']['sentiment_score'],
                    'avg_polarity': analysis['aggregate']['avg_polarity'],
                    'positive_ratio': analysis['aggregate']['positive_ratio'],
                    'num_articles': analysis['num_articles']
                })
        
        return pd.DataFrame(results)
    
    def sentiment_correlation_analysis(self, ticker: str, prices: pd.Series) -> Dict:
        """
        Analyze correlation between sentiment and price movements
        
        Args:
            ticker: Stock ticker symbol
            prices: Price series
        
        Returns:
            Dictionary with correlation analysis
        """
        # Fetch and analyze news
        news = self.fetch_news_yfinance(ticker)
        sentiment_df = self.analyze_news_batch(news)
        
        if sentiment_df.empty:
            return {'error': 'No sentiment data available'}
        
        # Create time series
        sentiment_ts = self.sentiment_time_series(sentiment_df)
        
        if sentiment_ts.empty:
            return {'error': 'Could not create time series'}
        
        # Calculate price returns
        returns = prices.pct_change().dropna()
        
        # Try to align sentiment with returns
        try:
            sentiment_ts['date'] = pd.to_datetime(sentiment_ts['date'])
            sentiment_ts = sentiment_ts.set_index('date')
            
            # Merge data
            merged = pd.merge(
                returns.to_frame('return'),
                sentiment_ts[['polarity']],
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            if len(merged) < 2:
                return {'error': 'Insufficient data for correlation'}
            
            correlation = merged['return'].corr(merged['polarity'])
            
            return {
                'correlation': correlation,
                'sample_size': len(merged),
                'significance': 'significant' if abs(correlation) > 0.3 else 'weak'
            }
        
        except Exception as e:
            return {'error': str(e)}
