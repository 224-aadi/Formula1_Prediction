import pandas as pd
from f1outcome.config import SETTINGS
from f1outcome.models.dnf import train_dnf, save as save_dnf
from f1outcome.models.ranker import train_ranker, save as save_ranker, FEATURES
from sklearn.metrics import roc_auc_score

def retrain():
    df = pd.read_parquet('data/processed/final_hybrid_dataset.parquet')
    
    train_df = df[df["season"] <= 2023]
    test_df = df[df["season"] == 2024]
    
    print(f"\nTraining Hybrid Ranker (Train: {len(train_df)} rows)...")
    model_ranker = train_ranker(train_df)
    save_ranker(model_ranker, "artifacts/final/ranker.joblib")
    
    importances = model_ranker.booster_.feature_importance()
    pairs = list(zip(FEATURES, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    print("Hybrid Ranker Feature Importances:")
    for name, val in pairs[:5]:
        print(f"  {name}: {val}")

    print(f"\nTraining Hybrid DNF (Train: {len(train_df)} rows)...")
    model_dnf = train_dnf(train_df)
    save_dnf(model_dnf, "artifacts/final/dnf_raw.joblib")
    
    try:
        y_true = test_df["status"].apply(lambda s: 0 if str(s).strip() in {"Finished", "Lapped"} or (str(s).startswith("+") and "Lap" in str(s)) else 1)
        y_pred = model_dnf.predict_proba(test_df[FEATURES])[:, 1]
        auc = roc_auc_score(y_true, y_pred)
        print(f"\n>>> TRUE Hybrid DNF ROC AUC on 2024: {auc:.4f} <<<")
    except Exception as e:
         print(f"Could not calculate AUC: {e}")
         
if __name__ == "__main__":
    retrain()
