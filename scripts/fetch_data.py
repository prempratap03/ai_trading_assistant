import requests
import pandas as pd
import argparse
import os
from datetime import datetime, timedelta


def fetch_stock_data(ticker, rapidapi_key, out_path, region="US"):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-chart"
    params = {"interval": "1d", "symbol": ticker, "range": "1mo", "region": region}
    headers = {
        "x-rapidapi-host": "apidojo-yahoo-finance-v1.p.rapidapi.com",
        "x-rapidapi-key": rapidapi_key,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code} - {response.text}")

    data = response.json()
    timestamps = data["chart"]["result"][0]["timestamp"]
    prices = data["chart"]["result"][0]["indicators"]["quote"][0]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices["open"],
        "high": prices["high"],
        "low": prices["low"],
        "close": prices["close"],
        "volume": prices["volume"]
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.to_csv(out_path, index=False)
    print(f"✅ Data saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch stock data using RapidAPI Yahoo Finance")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g., AAPL")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    parser.add_argument("--region", default="US", help="Region (default: US)")
    parser.add_argument("--rapidapi_key", default=None, help="RapidAPI key")
    args = parser.parse_args()

    rapidapi_key = args.rapidapi_key or os.getenv("RAPIDAPI_KEY")

    if not rapidapi_key:
        raise RuntimeError("RapidAPI key missing. Pass --rapidapi_key or set RAPIDAPI_KEY env var.")

    fetch_stock_data(args.ticker, rapidapi_key, args.out, args.region)


if __name__ == "__main__":
    main()
