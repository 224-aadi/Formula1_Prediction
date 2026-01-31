from __future__ import annotations
import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from src.f1outcome.models.ranker import FEATURES  # reuse same features list

def is_dnf_status(status: str | None) -> int:
    """
    Return 1 if the driver did NOT finish (DNF/DNS/DSQ), else 0.

    In Ergast/Jolpica, finishers can have statuses like:
    - "Finished"
    - "Lapped"
    - "+1 Lap", "+2 Laps", etc.
    """
    if status is None:
        return 0

    s = str(status).strip()

    # Finishers
    if s in {"Finished", "Lapped"}:
        return 0
    if s.startswith("+") and "Lap" in s:
        return 0

    # Everything else treated as non-finish (Retired, Did not start, Disqualified, etc.)
    return 1

def train_dnf(df: pd.DataFrame) -> LGBMClassifier:
    if "status" not in df.columns:
        raise ValueError("DNF training requires a 'status' column in the dataset.")

    df = df.copy()
    df["dnf"] = df["status"].apply(is_dnf_status).astype(int)

    X = df[FEATURES].fillna(-1)
    y = df["dnf"]

    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y)
    return model

def save(model: LGBMClassifier, path: str):
    joblib.dump(model, path)

def load(path: str) -> LGBMClassifier:
    return joblib.load(path)
