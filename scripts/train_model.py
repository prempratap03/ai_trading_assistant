import argparse, os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--metrics_out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    df = pd.read_csv(args.inp)
    features = ["SMA_20", "RSI_14", "Close", "Volume"]
    df = df.dropna()
    X = df[features]
    y = df["Return_1d"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    joblib.dump(model, args.model_out)
    with open(args.metrics_out, "w") as f:
        f.write(json.dumps({"r2": score}, indent=2))

    print(json.dumps({"r2": score, "model": args.model_out, "metrics": args.metrics_out}))

if __name__ == "__main__":
    main()
