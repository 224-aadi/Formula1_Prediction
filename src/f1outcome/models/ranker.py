from __future__ import annotations
import joblib
import pandas as pd
from lightgbm import LGBMRanker

FEATURES = [
    "grid",
    "qualiPos",
    "qualiGapMs",
    "qualiGapPct",
    "driverFormAvgFinish",
    "teamFormAvgFinish",
    "driverFormAvgPoints",
    "teamFormAvgPoints",
    "teammateFormFinishDelta",
    "grid_bucket",
    "q_s1_gap",
    "q_s2_gap",
    "q_s3_gap",
    "fp_longrun_gap",
    # Reliability
    "driver_dnf_rate",
    "driver_mech_dnf_rate",
    "team_dnf_rate",
    "team_mech_dnf_rate",
    # Context
    "is_street_circuit",
]



def make_group(df: pd.DataFrame) -> list[int]:
    return df.groupby("raceId").size().tolist()

def train_ranker(df: pd.DataFrame) -> LGBMRanker:
    df = df.dropna(subset=["relevance"]).copy()
    df = df.sort_values(["season", "round", "raceId"])

    X = df[FEATURES]  # LightGBM handles NaNs natively; do not fill with -1
    y = df["relevance"].astype(int)
    group = make_group(df)

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y, group=group)
    return model

def save(model: LGBMRanker, path: str):
    joblib.dump(model, path)

def load(path: str) -> LGBMRanker:
    return joblib.load(path)
