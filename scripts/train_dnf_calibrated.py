import argparse
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path
from f1outcome.config import SETTINGS
from f1outcome.models.ranker import FEATURES
from f1outcome.models.dnf import is_dnf_status

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(SETTINGS.artifacts_dir / "dnf_calibrated.joblib"))
    ap.add_argument("--max_season", type=int, default=2023, help="Train on seasons <= max_season")
    ap.add_argument("--test_season", type=int, default=2024, help="Evaluate on this season")
    args = ap.parse_args()

    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # --- Fix: Normalize types for filtering consistency ---
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    if "constructorId" in df.columns:
        df["constructorId"] = df["constructorId"].astype(str).str.strip().str.lower()
    # ----------------------------------------------------

    # 1. Prepare Train/Test Split
    train_df = df[df["season"] <= args.max_season].copy()
    test_df = df[df["season"] == args.test_season].copy()

    print(f"Train: {len(train_df)} rows (<= {args.max_season})")
    print(f"Test:  {len(test_df)} rows (== {args.test_season})")
    
    # Prepare labels
    train_df["dnf"] = train_df["status"].apply(is_dnf_status).astype(int)
    test_df["dnf"] = test_df["status"].apply(is_dnf_status).astype(int)

    X_train = train_df[FEATURES]
    y_train = train_df["dnf"]
    
    X_test = test_df[FEATURES]
    y_test = test_df["dnf"]

    # 2. Define Base Model (Same params as uncalibrated)
    base_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    # 3. Define Calibrated Model (CV=5, Isotonic)
    print("Training CalibratedClassifierCV (method='isotonic', cv=5)...")
    calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    
    calibrated_model.fit(X_train, y_train)
    
    # 4. Save
    print(f"Saving calibrated model to {args.out}...")
    joblib.dump(calibrated_model, args.out)

    # 5. Evaluate on 2024 (Test)
    # Filter for valid status rows only (just in case)
    mask = test_df["status"].notna()
    X_test_eval = X_test[mask]
    y_test_eval = y_test[mask]
    
    probs = calibrated_model.predict_proba(X_test_eval)[:, 1]
    
    auc = roc_auc_score(y_test_eval, probs)
    pr_auc = average_precision_score(y_test_eval, probs)
    brier = brier_score_loss(y_test_eval, probs)
    
    print("-" * 30)
    print(f"EVALUATION RESULTS ({args.test_season})")
    print("-" * 30)
    print(f"ROC AUC: {auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print(f"Brier:   {brier:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
