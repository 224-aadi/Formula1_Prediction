from __future__ import annotations
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier

from f1outcome.models.ranker import FEATURES  # reuse same features list

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
    from lightgbm import LGBMClassifier

    if "status" not in df.columns:
        raise ValueError("DNF training requires a 'status' column in the dataset.")

    df = df.copy()
    df["dnf"] = df["status"].apply(is_dnf_status).astype(int)

    X = df[FEATURES]  # LightGBM handles NaNs natively; do not fill with -1
    y = df["dnf"]

    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model

def save(model: LGBMClassifier, path: str):
    joblib.dump(model, path)

def load(path: str) -> Any:
    return joblib.load(path)
