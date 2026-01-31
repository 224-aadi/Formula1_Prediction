import argparse
import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from src.f1outcome.models.dnf import train_dnf, is_dnf_status
from src.f1outcome.models.ranker import FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train_max_season", type=int, default=2023)
    ap.add_argument("--cal_season", type=int, default=2024)
    ap.add_argument("--cal_max_round", type=int, default=12)
    ap.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--out", default="artifacts/dnf_calibrated.joblib")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)

    train_df = df[df["season"] <= args.train_max_season].copy()
    cal_df = df[(df["season"] == args.cal_season) & (df["round"] <= args.cal_max_round)].copy()
    test_df = df[(df["season"] == args.cal_season) & (df["round"] > args.cal_max_round)].copy()

    if cal_df.empty:
        raise ValueError("Calibration split is empty. Check cal_season/cal_max_round.")
    if test_df.empty:
        print("Warning: test split empty; set cal_max_round smaller to leave some 2024 rounds for testing.")

    # Train base DNF classifier on 2019–2023
    base = train_dnf(train_df)

    # Calibrate on early 2024
    X_cal = cal_df[FEATURES].fillna(-1)
    y_cal = cal_df["status"].apply(is_dnf_status).astype(int)

    calibrator = CalibratedClassifierCV(estimator=base, cv="prefit", method=args.method)
    calibrator.fit(X_cal, y_cal)

    # Quick sanity metrics on the held-out part of 2024 (rounds > cal_max_round)
    if not test_df.empty:
        X_test = test_df[FEATURES].fillna(-1)
        y_test = test_df["status"].apply(is_dnf_status).astype(int)
        p = calibrator.predict_proba(X_test)[:, 1]
        print(f"[Calibrated DNF] method={args.method}")
        print("  Brier:", brier_score_loss(y_test, p))
        print("  LogLoss:", log_loss(y_test, p))

    joblib.dump(calibrator, args.out)
    print("Saved ->", args.out)

if __name__ == "__main__":
    main()
