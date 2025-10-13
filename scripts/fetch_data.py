import argparse, os, json
import pandas as pd
from datetime import datetime, timedelta
# If you have your own fetcher, import it; fallback uses yfinance
try:
    import yfinance as yf
except Exception:
    yf = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", required=False)
    ap.add_argument("--end", required=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start = args.start or (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    if yf is None:
        raise RuntimeError("yfinance not installed. Add it to requirements.txt or implement your data_fetcher.")

    df = yf.download(args.ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data fetched for {args.ticker} {start}..{end}")
    df.reset_index().to_csv(args.out, index=False)
    print(json.dumps({"rows": len(df), "path": args.out}))

if __name__ == "__main__":
    main()
