from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
from src.f1outcome.models.ranker import load as load_model, FEATURES

from src.f1outcome.models.dnf import load as load_dnf

RANKER_PATH = Path("artifacts") / "ranker_2019_2024.joblib"
DNF_PATH    = Path("artifacts") / "dnf_2019_2024.joblib"
ALPHA = 10.0
P_DNF_CAP = 0.30
COMBINE = "subtract_cap"


app = FastAPI(title="F1 Outcome Lab")

# ---- Paths (repo root relative) ----
MODEL_PATH = RANKER_PATH
DATASET_PATH = Path("data") / "processed" / "dataset_seasons_2023.parquet"  # your file

# ---- Response/request models (better OpenAPI docs) ----
class ScoredDriver(BaseModel):
    driverId: str
    score_rank: float
    p_dnf: float
    score_adj: float

class HomeResponse(BaseModel):
    message: str
    docs: str
    predict: str

class PredictResponse(BaseModel):
    raceId: str
    order: list[ScoredDriver]

class RaceInfo(BaseModel):
    season: int
    round: int
    raceId: str
    raceName: str | None = None

# ---- Cached loaders (load once per process) ----
@lru_cache(maxsize=1)
def get_model():
    if not RANKER_PATH.exists():
        raise FileNotFoundError(f"Ranker model not found at: {RANKER_PATH}")
    return load_model(str(RANKER_PATH))

@lru_cache(maxsize=1)
def get_dnf_model():
    if not DNF_PATH.exists():
        raise FileNotFoundError(f"DNF model not found at: {DNF_PATH}")
    return load_dnf(str(DNF_PATH))

@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    # Read only what we need (pandas supports columns= for parquet) :contentReference[oaicite:3]{index=3}
    needed_cols = ["season", "round", "raceId", "raceName", "driverId"] + FEATURES
    # Some older parquet versions may not have raceName; handle gracefully
    df = pd.read_parquet(DATASET_PATH)
    cols = [c for c in needed_cols if c in df.columns]
    df = df[cols].copy()

    # Ensure raceId exists (our builder created it; but just in case)
    if "raceId" not in df.columns:
        df["raceId"] = df["season"].astype(str) + "_" + df["round"].astype(str)

    return df

@app.get("/", response_model=HomeResponse)
def home():
    return {"message": "F1 Outcome Lab is running", "docs": "/docs", "predict": "/predict/from_parquet", "Using": f"Ranker model at {RANKER_PATH}, DNF model at {DNF_PATH}, Dataset at {DATASET_PATH}"}

@app.get("/races", response_model=list[RaceInfo])
def list_races(season: int | None = Query(default=None)):
    df = get_dataset()
    if season is not None:
        df = df[df["season"] == season]
    races = (
        df[["season", "round", "raceId"] + (["raceName"] if "raceName" in df.columns else [])]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )
    if "raceName" not in races.columns:
        races["raceName"] = None
    return races.to_dict(orient="records")

@app.get("/meta")
def meta():
    return {
        "ranker_path": str(RANKER_PATH),
        "dnf_path": str(DNF_PATH),
        "alpha": ALPHA,
        "dataset": str(DATASET_PATH),
    }

@app.get("/predict/from_parquet", response_model=PredictResponse)
def predict_from_parquet(
    season: int = Query(..., ge=1950),
    round: int = Query(..., ge=1),
):
    df = get_dataset()
    race_df = df[(df["season"] == season) & (df["round"] == round)].copy()

    if race_df.empty:
        raise HTTPException(status_code=404, detail=f"No rows found for season={season}, round={round}")

    ranker = get_model()
    dnf_model = get_dnf_model()

    X = race_df[FEATURES].fillna(-1)
    race_df["score_rank"] = ranker.predict(X)
    race_df["p_dnf"] = dnf_model.predict_proba(X)[:, 1]
    p = np.minimum(race_df["p_dnf"].to_numpy(), P_DNF_CAP)
    race_df["score_adj"] = race_df["score_rank"].to_numpy() - ALPHA * p

    race_df = race_df.sort_values("score_adj", ascending=False)

    race_id = str(race_df["raceId"].iloc[0])
    order = race_df[["driverId", "score_rank", "p_dnf", "score_adj"]].to_dict(orient="records")
    return {"raceId": race_id, "order": order, "alpha": ALPHA, "p_dnf_cap": P_DNF_CAP}
