"""
Visualization Module
Creates interactive charts and visualizations using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List


class Visualizer:
    """Create interactive visualizations for trading data"""
    
    def __init__(self):
        self.template = 'plotly_dark'
    
    def plot_candlestick(self, df: pd.DataFrame, title: str = "Stock Price") -> go.Figure:
        """
        Create candlestick chart
        
        Args:
            df: DataFrame with OHLCV data
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template=self.template,
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_price_with_indicators(self, df: pd.DataFrame, title: str = "Technical Analysis") -> go.Figure:
        """
        Create price chart with technical indicators
        
        Args:
            df: DataFrame with price and indicators
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=('Price with Bollinger Bands', 'Volume', 'RSI', 'MACD')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        if 'BB_Upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                          line=dict(color='rgba(250, 0, 0, 0.5)', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle',
                          line=dict(color='rgba(0, 0, 250, 0.5)', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                          line=dict(color='rgba(0, 250, 0, 0.5)', width=1)),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                 for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                  marker_color=colors),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                          line=dict(color='blue', width=2)),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                          line=dict(color='orange', width=2)),
                row=4, col=1
            )
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
                      marker_color='gray'),
                row=4, col=1
            )
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        return fig
    
    def plot_predictions(self, historical: pd.Series, predictions: Dict, 
                        title: str = "Price Predictions") -> go.Figure:
        """
        Plot historical prices with predictions
        
        Args:
            historical: Historical price series
            predictions: Dictionary with prediction arrays
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Create future dates
        last_date = historical.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(predictions.get('lstm', [])) + 1)[1:]
        
        # LSTM predictions
        if 'lstm' in predictions and predictions['lstm'] is not None:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions['lstm'],
                name='LSTM Prediction',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Prophet predictions
        if 'prophet' in predictions and predictions['prophet'] is not None:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions['prophet'],
                name='Prophet Prediction',
                line=dict(color='green', width=2, dash='dash')
            ))
        
        # Ensemble predictions
        if 'ensemble' in predictions and predictions['ensemble'] is not None:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions['ensemble'],
                name='Ensemble Prediction',
                line=dict(color='orange', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template=self.template,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_efficient_frontier(self, portfolios: pd.DataFrame, optimal_points: Dict) -> go.Figure:
        """
        Plot efficient frontier
        
        Args:
            portfolios: DataFrame with random portfolios
            optimal_points: Dictionary with optimal portfolio points
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Random portfolios
        fig.add_trace(go.Scatter(
            x=portfolios['Volatility'],
            y=portfolios['Return'],
            mode='markers',
            marker=dict(
                size=5,
                color=portfolios['Sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Random Portfolios',
            text=[f"Sharpe: {s:.2f}" for s in portfolios['Sharpe']],
            hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}<extra></extra>'
        ))
        
        # Optimal points
        for name, data in optimal_points.items():
            fig.add_trace(go.Scatter(
                x=[data['volatility']],
                y=[data['return']],
                mode='markers',
                marker=dict(size=15, symbol='star'),
                name=name,
                hovertemplate=f'{name}<br>Volatility: %{{x:.2%}}<br>Return: %{{y:.2%}}<br>Sharpe: {data["sharpe"]:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            template=self.template,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_portfolio_allocation(self, weights: Dict, title: str = "Portfolio Allocation") -> go.Figure:
        """
        Plot portfolio allocation pie chart
        
        Args:
            weights: Dictionary of ticker -> weight
            title: Chart title
        
        Returns:
            Plotly figure
        """
        tickers = list(weights.keys())
        values = [weights[t] * 100 for t in tickers]
        
        fig = go.Figure(data=[go.Pie(
            labels=tickers,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=500
        )
        
        return fig
    
    def plot_risk_metrics(self, risk_report: Dict) -> go.Figure:
        """
        Plot risk metrics comparison
        
        Args:
            risk_report: Dictionary with risk metrics
        
        Returns:
            Plotly figure
        """
        metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        values = [
            risk_report.get('sharpe_ratio', 0),
            risk_report.get('sortino_ratio', 0),
            risk_report.get('calmar_ratio', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        
        fig.update_layout(
            title='Risk-Adjusted Return Metrics',
            yaxis_title='Ratio',
            template=self.template,
            height=400
        )
        
        return fig
    
    def plot_drawdown(self, prices: pd.Series) -> go.Figure:
        """
        Plot drawdown chart
        
        Args:
            prices: Price series
        
        Returns:
            Plotly figure
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Drawdown Over Time',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template=self.template,
            height=400
        )
        
        return fig
    
    def plot_sentiment_analysis(self, sentiment_df: pd.DataFrame) -> go.Figure:
        """
        Plot sentiment analysis results
        
        Args:
            sentiment_df: DataFrame with sentiment scores
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sentiment Distribution', 'Sentiment Over Time'),
            specs=[[{"type": "bar"}], [{"type": "scatter"}]]
        )
        
        # Sentiment distribution
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                  marker_color=['green', 'gray', 'red']),
            row=1, col=1
        )
        
        # Sentiment over time
        if 'date' in sentiment_df.columns:
            fig.add_trace(
                go.Scatter(x=sentiment_df['date'], y=sentiment_df['polarity'],
                          mode='lines+markers', name='Polarity'),
                row=2, col=1
            )
        
        fig.update_layout(
            template=self.template,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_matrix(self, returns_df: pd.DataFrame) -> go.Figure:
        """
        Plot correlation matrix heatmap
        
        Args:
            returns_df: DataFrame with returns for multiple stocks
        
        Returns:
            Plotly figure
        """
        corr = returns_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Stock Correlation Matrix',
            template=self.template,
            height=600,
            width=700
        )
        
        return fig
    
    def plot_performance_comparison(self, comparison_df: pd.DataFrame) -> go.Figure:
        """
        Plot strategy performance comparison
        
        Args:
            comparison_df: DataFrame with strategy comparisons
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        metrics = comparison_df.columns
        
        for strategy in comparison_df.index:
            fig.add_trace(go.Bar(
                name=strategy,
                x=metrics,
                y=comparison_df.loc[strategy],
                text=comparison_df.loc[strategy].round(2),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Portfolio Strategy Comparison',
            barmode='group',
            template=self.template,
            height=500,
            yaxis_title='Value'
        )
        
        return fig
