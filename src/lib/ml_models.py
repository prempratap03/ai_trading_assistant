"""
Machine Learning Models Module
Implements LSTM neural networks and Prophet for stock price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM model will not work.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Prophet model will not work.")


class LSTMPredictor:
    """
    LSTM-based stock price prediction model.
    """
    
    def __init__(self, lookback: int = 60, units: int = 50, dropout: float = 0.2):
        """
        Initialize LSTM predictor.
        
        Args:
            lookback: Number of previous days to use for prediction
            units: Number of LSTM units
            dropout: Dropout rate for regularization
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
    
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            data: DataFrame with stock data
            target_col: Column to predict
        
        Returns:
            Tuple of (X_train, y_train, scaled_data)
        """
        # Extract target column
        dataset = data[target_col].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_data
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(units=self.units),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(
        self, 
        data: pd.DataFrame, 
        epochs: int = 50, 
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            data: DataFrame with stock data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Preparing data for LSTM training...")
        X, y, scaled_data = self.prepare_data(data)
        
        # Build model
        self.model = self.build_model((X.shape[1], 1))
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train model
        logger.info("Training LSTM model...")
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Make predictions on training data
        predictions = self.model.predict(X, verbose=0)
        predictions = self.scaler.inverse_transform(predictions)
        y_actual = self.scaler.inverse_transform(y.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(y_actual, predictions)
        mae = mean_absolute_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, predictions)
        
        logger.info(f"Training completed. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'history': self.history.history
        }
    
    def predict(self, data: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Predict future stock prices.
        
        Args:
            data: DataFrame with historical stock data
            days: Number of days to predict
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare last sequence
        dataset = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(dataset)
        
        # Get last lookback days
        last_sequence = scaled_data[-self.lookback:]
        predictions = []
        
        # Predict future values
        current_sequence = last_sequence.copy()
        for _ in range(days):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.lookback, 1)
            
            # Make prediction
            pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], [[pred]], axis=0)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create prediction DataFrame
        last_date = data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=days + 1, freq='D')[1:]
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions.flatten()
        })
        
        return pred_df


class ProphetPredictor:
    """
    Prophet-based stock price prediction model.
    """
    
    def __init__(
        self, 
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0
    ):
        """
        Initialize Prophet predictor.
        
        Args:
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Strength of seasonality
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for this model")
        
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
    
    def train(self, data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """
        Train the Prophet model.
        
        Args:
            data: DataFrame with stock data (must have 'Date' column)
            target_col: Column to predict
        
        Returns:
            Dictionary with training information
        """
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        df = data[['Date', target_col]].copy()
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Initialize and train model
        logger.info("Training Prophet model...")
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        self.model.fit(df)
        logger.info("Prophet training completed")
        
        return {'status': 'success', 'trained_on': len(df)}
    
    def predict(self, days: int = 30) -> pd.DataFrame:
        """
        Predict future stock prices.
        
        Args:
            days: Number of days to predict
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=days)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return relevant columns
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
        result.columns = ['Date', 'Predicted_Close', 'Lower_Bound', 'Upper_Bound']
        result.reset_index(drop=True, inplace=True)
        
        return result
    
    def get_components(self) -> pd.DataFrame:
        """
        Get trend and seasonality components.
        
        Returns:
            DataFrame with model components
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        return forecast


class SimpleMovingAveragePredictor:
    """
    Simple Moving Average based prediction (baseline model).
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize SMA predictor.
        
        Args:
            window: Moving average window size
        """
        self.window = window
    
    def predict(self, data: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """
        Predict using simple moving average.
        
        Args:
            data: DataFrame with historical stock data
            days: Number of days to predict
        
        Returns:
            DataFrame with predictions
        """
        # Calculate moving average
        sma = data['Close'].rolling(window=self.window).mean()
        last_sma = sma.iloc[-1]
        
        # Use last SMA as prediction for all future days (naive approach)
        last_date = data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=days + 1, freq='D')[1:]
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': [last_sma] * days
        })
        
        return pred_df


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """
    Evaluate prediction performance.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        Dictionary with evaluation metrics
    """
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
