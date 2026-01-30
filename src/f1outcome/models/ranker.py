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
]


def make_group(df: pd.DataFrame) -> list[int]:
    return df.groupby("raceId").size().tolist()

def train_ranker(df: pd.DataFrame) -> LGBMRanker:
    df = df.dropna(subset=["relevance"]).copy()
    df = df.sort_values(["season", "round", "raceId"])

    X = df[FEATURES].fillna(-1)
    y = df["relevance"].astype(int)
    group = make_group(df)

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y, group=group)
    return model

def save(model: LGBMRanker, path: str):
    joblib.dump(model, path)

def load(path: str) -> LGBMRanker:
    return joblib.load(path)
