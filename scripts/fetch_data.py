import argparse, os, json, time
from datetime import datetime, timedelta

def safe_dates(start_arg, end_arg):
    today = datetime.utcnow().date()
    end = end_arg or (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start = start_arg or (today - timedelta(days=730)).strftime("%Y-%m-%d")
    return start, end

def try_yfinance(ticker, start, end, retries=5, pause=2):
    try:
        import yfinance as yf
    except Exception:
        return None
    last_err=None
    for _ in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                return df.reset_index()
        except Exception as e:
            last_err=e
        time.sleep(pause)
    return None

def try_stooq(ticker, start, end):
    try:
        import pandas_datareader.data as web
        df = web.DataReader(ticker, 'stooq', start, end)
        if df is not None and not df.empty:
            df = df.sort_index().reset_index()
            df.rename(columns={'Date':'Date','Open':'Open','High':'High','Low':'Low','Close':'Close','Volume':'Volume'}, inplace=True)
            return df
    except Exception:
        return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start, end = safe_dates(args.start, args.end)

    df = try_yfinance(args.ticker, start, end)
    if df is None:
        df = try_stooq(args.ticker, start, end)

    if df is None or df.empty:
        raise RuntimeError(f"Failed to fetch data for {args.ticker} {start}..{end} from both yfinance and stooq")

    if 'Date' not in df.columns:
        df['Date'] = df.index

    df.to_csv(args.out, index=False)
    print(json.dumps({"rows": int(len(df)), "path": args.out, "ticker": args.ticker, "start": start, "end": end}))

if __name__ == "__main__":
    main()
