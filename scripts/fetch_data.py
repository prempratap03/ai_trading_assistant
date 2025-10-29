import argparse, os, json, time
from datetime import datetime, timedelta, timezone
from dateutil import parser as du
import requests
import pandas as pd

RAPIDAPI_HOST = "yh-finance.p.rapidapi.com"
CHART_URL = f"https://{RAPIDAPI_HOST}/stock/v3/get-chart"

def parse_date(dstr, default):
    if not dstr:
        return default
    return du.parse(dstr).replace(tzinfo=timezone.utc)

def to_unix(dt):
    return int(dt.timestamp())

def fetch_chart(symbol, start, end, interval, rapidapi_key):
    """
    Fetch historical data using RapidAPI Yahoo Finance chart endpoint.
    interval: 1d, 1wk, 1mo, etc.
    """
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    params = {
        "symbol": symbol,
        "region": "US",
        "interval": interval,
        "period1": to_unix(start),
        "period2": to_unix(end),
        "events": "div,splits",
    }
    r = requests.get(CHART_URL, headers=headers, params=params, timeout=30)
    # Handle rate limits gracefully
    if r.status_code == 429:
        raise RuntimeError("RapidAPI rate-limited (429). Try again later or reduce frequency.")
    r.raise_for_status()
    data = r.json()
    result = (data or {}).get("chart", {}).get("result", [])
    if not result:
        err = (data or {}).get("chart", {}).get("error", {})
        raise RuntimeError(f"No result for {symbol}. Error: {err}")
    res = result[0]
    ts = res.get("timestamp", [])
    ind = res.get("indicators", {})
    quote = (ind.get("quote") or [{}])[0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    vols = quote.get("volume", [])
    adjclose = (ind.get("adjclose") or [{}])[0].get("adjclose", closes)

    if not ts or not closes:
        raise RuntimeError(f"No time series for {symbol} between {start} and {end}")

    # Build DataFrame
    df = pd.DataFrame({
        "Date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Adj Close": adjclose,
        "Volume": vols,
    })

    # Drop any rows with missing Close
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df

def safe_defaults(start_arg, end_arg):
    today = datetime.utcnow().date()
    # end yesterday to avoid partial day issues
    end = parse_date(end_arg, datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=1))
    start = parse_date(start_arg, end - timedelta(days=730))
    # ensure start < end
    if start >= end:
        start = end - timedelta(days=365)
    return start, end

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    ap.add_argument("--start", help="Start date (YYYY-MM-DD). Defaults to end-730d")
    ap.add_argument("--end", help="End date (YYYY-MM-DD). Defaults to yesterday (UTC)")
    ap.add_argument("--interval", default="1d", help="Interval: 1d, 1wk, 1mo")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--rapidapi_key", help="RapidAPI key (or set RAPIDAPI_KEY env var)")

    args = ap.parse_args()
    rapidapi_key = args.rapidapi_key or os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        raise RuntimeError("RapidAPI key missing. Pass --rapidapi_key or set RAPIDAPI_KEY env var.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start_dt, end_dt = safe_defaults(args.start, args.end)

    df = fetch_chart(args.ticker, start_dt, end_dt, args.interval, rapidapi_key)
    if df.empty:
        raise RuntimeError(f"Empty data for {args.ticker} {start_dt.date()}..{end_dt.date()}")

    # Keep only required columns and sort
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].sort_values("Date").reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(json.dumps({
        "rows": int(len(df)),
        "path": args.out,
        "ticker": args.ticker,
        "start": str(start_dt.date()),
        "end": str(end_dt.date()),
        "interval": args.interval
    }))
if __name__ == "__main__":
    main()
