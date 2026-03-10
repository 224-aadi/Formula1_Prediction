import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Loading datasets and models...")
    df_base = pd.read_parquet('data/processed/dataset_seasons_2019_2020_2021_2022_2023_2024.parquet')
    df_f1 = pd.read_parquet('data/processed/dataset_seasons_2019_2020_2021_2022_2023_2024_fastf1.parquet')
    
    # Simulating a live fetch of 2024_1
    df_base = df_base[(df_base.season == 2024) & (df_base["round"] == 1)].copy()
    
    f1_cols = ["season", "round", "fullName_norm", "q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap"]
    for c in f1_cols:
        if c not in df_f1.columns:
            df_f1[c] = float('nan')
            
    df_f1_subset = df_f1[f1_cols].drop_duplicates(subset=["season", "round", "fullName_norm"])
    df_base = df_base.drop(columns=["q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap"], errors='ignore')
    df = df_base.merge(df_f1_subset, on=["season", "round", "fullName_norm"], how="left")
    
    # Load Models and features
    ranker = joblib.load('artifacts/final/ranker.joblib')
    dnf_model = joblib.load('artifacts/final/dnf.joblib')
    with open('artifacts/final/features.json') as f:
        features = json.load(f)
        
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
            
    X = df[features]
    
    # Predict
    df['rank_score'] = ranker.predict(X)
    df['dnf_prob'] = dnf_model.predict_proba(X)[:, 1]
    
    df = df.sort_values(by="rank_score", ascending=False).reset_index(drop=True)
    
    print("\n========= SMOKE TEST =========")
    print(f"Loaded records for 2024_1: {len(df)}")
    assert len(df) == 20, "Expected exactly 20 drivers for 2024_1!"
    
    print("Top 3 Predictions by Ranker Match:")
    for i, row in df.head(3).iterrows():
        print(f" {i+1}. {row.get('givenName', '')} {row.get('familyName', '')} - score: {row['rank_score']:.4f}  (DNF Risk: {row['dnf_prob']:.1%})")
        
    # Check monotonicity
    scores = df['rank_score'].tolist()
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores are not monotonically decreasing!"
    print("Assertion Passed: Ordering is monotonic.")
    print("Assertion Passed: Output length is 20.")
    print("Smoke Test OK!")

if __name__ == "__main__":
    main()
