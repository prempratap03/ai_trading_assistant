"""
Fetch stock data using Yahoo Finance (yfinance) and save it as CSV.
"""

import argparse
import pandas as pd
from src.lib.data_fetcher import StockDataFetcher


def main():
    parser = argparse.ArgumentParser(description="Fetch stock data using Yahoo Finance")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--period", type=str, default="2y", help="Data period (e.g., 1y, 2y, 5y, max)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (e.g., 1d, 1wk, 1mo)")
    parser.add_argument("--out", type=str, required=True, help="Output CSV file path")

    args = parser.parse_args()

    fetcher = StockDataFetcher()

    print(f"[INFO] Fetching data for {args.ticker} from Yahoo Finance...")
    df = fetcher.fetch_stock_data(args.ticker, period=args.period, interval=args.interval)

    if df.empty:
        raise ValueError(f"No data found for ticker {args.ticker}")

    df.to_csv(args.out, index=True)
    print(f"[INFO] Data saved successfully at: {args.out}")


if __name__ == "__main__":
    main()
