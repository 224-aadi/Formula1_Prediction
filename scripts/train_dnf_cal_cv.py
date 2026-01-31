import argparse
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from src.f1outcome.models.dnf import is_dnf_status
from src.f1outcome.models.ranker import FEATURES
from lightgbm import LGBMClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_season", type=int, default=2024)
    ap.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--out", default="artifacts/dnf_cal_cv.joblib")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    df = df[df["season"] <= args.max_season].copy()

    y = df["status"].apply(is_dnf_status).astype(int)
    X = df[FEATURES].fillna(-1)

    base = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    cal = CalibratedClassifierCV(estimator=base, method=args.method, cv=args.cv)
    cal.fit(X, y)

    p = cal.predict_proba(X)[:, 1]
    print("Train-set Brier (for sanity):", brier_score_loss(y, p))
    print("Train-set LogLoss (for sanity):", log_loss(y, p))

    joblib.dump(cal, args.out)
    print("Saved ->", args.out)

if __name__ == "__main__":
    main()
