import argparse, os, json
import pandas as pd
import numpy as np

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def sma(series, window=20):
    return series.rolling(window).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.inp, parse_dates=["Date"])
    df = df.sort_values("Date")
    df["SMA_20"] = sma(df["Close"], 20)
    df["RSI_14"] = rsi(df["Close"], 14)
    df["Return_1d"] = df["Close"].pct_change().shift(-1)  # next-day return (label)
    df.dropna().to_csv(args.out, index=False)
    print(json.dumps({"rows": int(df.dropna().shape[0]), "path": args.out}))

if __name__ == "__main__":
    main()
