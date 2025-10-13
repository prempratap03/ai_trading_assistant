import argparse, os, json, joblib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.inp).dropna()
    features = ["SMA_20", "RSI_14", "Close", "Volume"]
    X = df[features]

    model = joblib.load(args.model)
    preds = model.predict(X)
    out = df.copy()
    out["pred_return_1d"] = preds
    out.to_csv(args.out, index=False)
    print(json.dumps({"rows": len(out), "path": args.out}))

if __name__ == "__main__":
    main()
