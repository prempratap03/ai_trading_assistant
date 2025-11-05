import os
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime


class StockDataFetcher:
    """Fetch and preprocess stock market data with RapidAPI fallback."""

    @st.cache_data(ttl=3600)
    def fetch_stock_data(_self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        
        # --- Fallback: RapidAPI Yahoo Finance 166 ---
        try:
            api_key = os.getenv("RAPIDAPI_KEY")
            if not api_key:
                raise Exception("RAPIDAPI_KEY not found in environment")

            url = "https://yahoo-finance166.p.rapidapi.com/api/stock/get-chart"
            params = {"symbol": ticker, "interval": "1d", "range": "1y"}
            headers = {
                "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
                "x-rapidapi-key": api_key
            }

            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code != 200:
                raise Exception(f"RapidAPI error ({response.status_code}): {response.text}")

            data = response.json()
            chart_data = data.get("chart", {}).get("result", [])
            if not chart_data or not isinstance(chart_data, list):
                raise Exception("Unexpected response format from Yahoo Finance 166 API")

            result = chart_data[0]
            timestamps = result.get("timestamp", [])
            quotes = result.get("indicators", {}).get("quote", [{}])[0]

            if not timestamps or not quotes:
                raise Exception("No valid price data returned")

            df = pd.DataFrame({
                "Date": pd.to_datetime(timestamps, unit="s"),
                "Open": quotes.get("open", []),
                "High": quotes.get("high", []),
                "Low": quotes.get("low", []),
                "Close": quotes.get("close", []),
                "Volume": quotes.get("volume", [])
            })

            df.set_index("Date", inplace=True)
            df.dropna(inplace=True)

            st.sidebar.info(f"✅ Data fetched from RapidAPI (Yahoo Finance 166) for {ticker}")
            return df

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")
    def get_price_changes(self, df: pd.DataFrame) -> dict:
        """
        Calculate percentage price changes over different timeframes.
        """
        if df.empty or 'Close' not in df.columns:
            raise ValueError("Invalid DataFrame: missing 'Close' column")

        def pct_change(days: int):
            if len(df) < days + 1:
                return 0.0
            return ((df['Close'].iloc[-1] - df['Close'].iloc[-(days + 1)]) / df['Close'].iloc[-(days + 1)]) * 100

        # Calculate daily, weekly, monthly, etc. changes
        changes = {
            '1d': pct_change(1),
            '1w': pct_change(5),
            '1m': pct_change(22),   # roughly 1 month
            '3m': pct_change(66),
            '1y': pct_change(252),
            'ytd': ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100 if len(df) > 1 else 0.0
        }

        return changes
    def fetch_multiple_stocks(self, tickers, period='2y', interval='1d'):
        """
        Fetch data for multiple tickers using the same method as fetch_stock_data.
        Returns a dict of {ticker: DataFrame}
        """
        data = {}
        for ticker in tickers:
            try:
                df = self.fetch_stock_data(ticker, period, interval)
                if not df.empty:
                    data[ticker] = df
                else:
                    st.warning(f"No data found for {ticker}")
            except Exception as e:
                st.warning(f"Could not fetch data for {ticker}: {str(e)}")
        return data
    def get_price_changes(self, df: pd.DataFrame) -> dict:
        """
        Calculate percentage price changes over different timeframes.
        """
        if df.empty or 'Close' not in df.columns:
            raise ValueError("Invalid DataFrame: missing 'Close' column")

        def pct_change(days: int):
            if len(df) < days + 1:
                return 0.0
            return ((df['Close'].iloc[-1] - df['Close'].iloc[-(days + 1)]) /
                    df['Close'].iloc[-(days + 1)]) * 100

        # Calculate daily, weekly, monthly, etc. changes
        changes = {
            '1d': pct_change(1),
            '1w': pct_change(5),
            '1m': pct_change(22),   # ~1 month
            '3m': pct_change(66),
            '1y': pct_change(252),
            'ytd': ((df['Close'].iloc[-1] - df['Close'].iloc[0]) /
                    df['Close'].iloc[0]) * 100 if len(df) > 1 else 0.0
        }

        return changes
    def get_stock_info(self, ticker):
        """
        Fetch stock metadata (name, 52w high/low, market cap, etc.)
        Compatible with Yahoo Finance 166 /chart API response.
        """
        try:
            api_key = os.getenv("RAPIDAPI_KEY")
            if not api_key:
                raise Exception("RAPIDAPI_KEY not found in environment")

            url = "https://yahoo-finance166.p.rapidapi.com/api/stock/get-chart"
            params = {"symbol": ticker, "region": "US", "interval": "1d", "range": "1y"}
            headers = {
                "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
                "x-rapidapi-key": api_key
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                raise Exception(f"RapidAPI error ({response.status_code})")

            data = response.json()

            # Validate structure
            result_list = data.get("chart", {}).get("result", [])
            if not result_list or not isinstance(result_list, list):
                raise Exception("Unexpected API structure (missing chart.result)")

            meta = result_list[0].get("meta", {})
            if not meta:
                raise Exception("Missing 'meta' field in response")

            # Helper to safely extract numbers
            def safe_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            return {
                "name": meta.get("longName", ticker),
                "sector": meta.get("exchangeName", "N/A"),  # not directly available here
                "industry": meta.get("instrumentType", "N/A"),
                "market_cap": 0.0,  # not present in this endpoint
                "pe_ratio": 0.0,    # not present either
                "beta": 0.0,        # not present either
                "52w_high": safe_float(meta.get("fiftyTwoWeekHigh")),
                "52w_low": safe_float(meta.get("fiftyTwoWeekLow")),
                "current_price": safe_float(meta.get("regularMarketPrice")),
                "volume": int(meta.get("regularMarketVolume", 0)),
                "exchange": meta.get("fullExchangeName", "N/A"),
                "currency": meta.get("currency", "USD")
            }

        except Exception as e:
            st.warning(f"Could not fetch info for {ticker}: {e}")
            return {
                "name": ticker,
                "sector": "N/A",
                "industry": "N/A",
                "market_cap": 0.0,
                "pe_ratio": 0.0,
                "beta": 0.0,
                "52w_high": 0.0,
                "52w_low": 0.0,
                "current_price": 0.0,
                "volume": 0,
                "exchange": "N/A",
                "currency": "USD"
            }
