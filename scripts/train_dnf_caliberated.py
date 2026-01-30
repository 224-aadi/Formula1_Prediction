import argparse
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV  # :contentReference[oaicite:4]{index=4}

from f1outcome.models.dnf import train_dnf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_train", required=True)  # e.g., 2019-2023
    ap.add_argument("--data_cal", required=True)    # e.g., 2024 (or subset)
    ap.add_argument("--out", default="artifacts/dnf_calibrated.joblib")
    ap.add_argument("--method", default="sigmoid", choices=["sigmoid", "isotonic"])
    args = ap.parse_args()

    train_df = pd.read_parquet(args.data_train)
    cal_df = pd.read_parquet(args.data_cal)

    base = train_dnf(train_df)  # trains LGBMClassifier

    X_cal = cal_df[base.feature_name_].fillna(-1) if hasattr(base, "feature_name_") else cal_df
    # safer: reuse FEATURES
    from f1outcome.models.ranker import FEATURES
    X_cal = cal_df[FEATURES].fillna(-1)

    # Build y_cal from status
    from f1outcome.models.dnf import is_dnf_status
    y_cal = cal_df["status"].apply(is_dnf_status).astype(int)

    calibrator = CalibratedClassifierCV(base_estimator=base, method=args.method, cv="prefit")
    calibrator.fit(X_cal, y_cal)

    joblib.dump(calibrator, args.out)
    print("Saved ->", args.out)

if __name__ == "__main__":
    main()
