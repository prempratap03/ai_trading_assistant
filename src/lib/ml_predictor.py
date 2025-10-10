"""
Machine Learning Prediction Module
Implements LSTM and Prophet models for stock price prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class LSTMPredictor:
    """LSTM-based stock price prediction"""
    
    def __init__(self, sequence_length: int = 60, epochs: int = 50, batch_size: int = 32):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def prepare_data(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: Price series
        
        Returns:
            Tuple of (X_train, y_train, scaled_data)
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_data
    
    def build_model(self, input_shape: Tuple) -> None:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.1) -> dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_split: Validation data proportion
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            verbose=0
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input data
        
        Returns:
            Predictions (inverse transformed)
        """
        predictions = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def forecast_future(self, data: pd.Series, days: int = 30) -> np.ndarray:
        """
        Forecast future prices
        
        Args:
            data: Historical price data
            days: Number of days to forecast
        
        Returns:
            Array of forecasted prices
        """
        # Prepare last sequence
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            current_input = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            next_pred = self.model.predict(current_input, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred, axis=0)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()


class ProphetPredictor:
    """Prophet-based stock price prediction"""
    
    def __init__(self, confidence_interval: float = 0.95):
        self.confidence_interval = confidence_interval
        self.model = None
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet
        
        Args:
            df: DataFrame with stock data
        
        Returns:
            DataFrame formatted for Prophet
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df['Close'].values
        })
        
        return prophet_df
    
    def train(self, df: pd.DataFrame, **kwargs) -> None:
        """
        Train Prophet model
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            **kwargs: Additional Prophet parameters
        """
        from prophet import Prophet
        
        # Default parameters optimized for stock data
        default_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'changepoint_prior_scale': 0.05,
            'interval_width': self.confidence_interval
        }
        
        # Override with user parameters
        default_params.update(kwargs)
        
        self.model = Prophet(**default_params)
        self.model.fit(df)
    
    def predict(self, periods: int = 90, freq: str = 'D') -> pd.DataFrame:
        """
        Make future predictions
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions ('D' for daily)
        
        Returns:
            DataFrame with predictions
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast
    
    def get_forecast_summary(self, forecast: pd.DataFrame, last_n: int = 30) -> dict:
        """
        Get summary of forecast
        
        Args:
            forecast: Prophet forecast DataFrame
            last_n: Number of last predictions to summarize
        
        Returns:
            Dictionary with forecast summary
        """
        last_forecast = forecast.tail(last_n)
        
        summary = {
            'mean_prediction': last_forecast['yhat'].mean(),
            'std_prediction': last_forecast['yhat'].std(),
            'lower_bound': last_forecast['yhat_lower'].mean(),
            'upper_bound': last_forecast['yhat_upper'].mean(),
            'trend': 'upward' if last_forecast['trend'].iloc[-1] > last_forecast['trend'].iloc[0] else 'downward'
        }
        
        return summary


class EnsemblePredictor:
    """Ensemble of multiple prediction models"""
    
    def __init__(self):
        self.lstm = None
        self.prophet = None
        self.weights = {'lstm': 0.5, 'prophet': 0.5}
    
    def train_ensemble(self, data: pd.Series, df: pd.DataFrame) -> dict:
        """
        Train both LSTM and Prophet models
        
        Args:
            data: Price series for LSTM
            df: Full DataFrame for Prophet
        
        Returns:
            Dictionary with training results
        """
        results = {}
        
        # Train LSTM
        try:
            self.lstm = LSTMPredictor()
            X, y, _ = self.lstm.prepare_data(data)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            
            lstm_history = self.lstm.train(X_train, y_train)
            results['lstm'] = {'status': 'success', 'history': lstm_history}
        except Exception as e:
            results['lstm'] = {'status': 'failed', 'error': str(e)}
        
        # Train Prophet
        try:
            self.prophet = ProphetPredictor()
            prophet_df = self.prophet.prepare_data(df)
            self.prophet.train(prophet_df)
            results['prophet'] = {'status': 'success'}
        except Exception as e:
            results['prophet'] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def predict_ensemble(self, data: pd.Series, days: int = 30) -> dict:
        """
        Generate ensemble predictions
        
        Args:
            data: Historical price data
            days: Days to forecast
        
        Returns:
            Dictionary with predictions from both models
        """
        predictions = {}
        
        # LSTM predictions
        if self.lstm is not None:
            try:
                lstm_pred = self.lstm.forecast_future(data, days)
                predictions['lstm'] = lstm_pred
            except Exception as e:
                predictions['lstm'] = None
                st.warning(f"LSTM prediction failed: {str(e)}")
        
        # Prophet predictions
        if self.prophet is not None:
            try:
                prophet_forecast = self.prophet.predict(periods=days)
                predictions['prophet'] = prophet_forecast['yhat'].tail(days).values
            except Exception as e:
                predictions['prophet'] = None
                st.warning(f"Prophet prediction failed: {str(e)}")
        
        # Ensemble prediction (weighted average)
        if predictions.get('lstm') is not None and predictions.get('prophet') is not None:
            predictions['ensemble'] = (
                self.weights['lstm'] * predictions['lstm'] +
                self.weights['prophet'] * predictions['prophet']
            )
        
        return predictions
