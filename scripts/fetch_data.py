import argparse, os, json, sys, time
from datetime import datetime, timedelta

try:
    import yfinance as yf
except Exception:
    yf = None

def safe_dates(start_arg: str | None, end_arg: str | None):
    # Avoid current UTC day; end yesterday to prevent partial-day issues
    today = datetime.utcnow().date()
    end = end_arg or (today - timedelta(days=1)).strftime("%Y-%m-%d")
    # 2 years back so we have enough rows after indicators dropna
    start = start_arg or (today - timedelta(days=730)).strftime("%Y-%m-%d")
    return start, end

def fetch_with_retries(ticker: str, start: str, end: str, retries: int = 6, pause: int = 3):
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(pause)
    raise RuntimeError(f"Failed to fetch after {retries} retries: {ticker} {start}..{end}; last_err={last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start, end = safe_dates(args.start, args.end)
    df = fetch_with_retries(args.ticker, start, end, retries=8, pause=2).reset_index()

    # Normalize date column
    if 'Date' not in df.columns and 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
    if 'Date' not in df.columns and df.index.name:
        df['Date'] = df.index

    if df.empty:
        raise RuntimeError(f"No data fetched for {args.ticker} {start}..{end}")

    df.to_csv(args.out, index=False)
    print(json.dumps({"rows": int(len(df)), "path": args.out, "ticker": args.ticker, "start": start, "end": end}))

if __name__ == "__main__":
    main()
