from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, List
import traceback

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from f1outcome.models.ranker import load as load_ranker, FEATURES
from f1outcome.models.dnf import load as load_dnf
from f1outcome.data.live_builder import LiveBuilder
from f1outcome.api.frontend import PRODUCT_HTML

from fastapi.middleware.cors import CORSMiddleware





import os

# ----------------------------
# Artifact paths / constants
# ----------------------------
RANKER_PATH = Path(os.environ.get("F1_RANKER_PATH", "artifacts/final/ranker.joblib"))

# Use RAW (uncalibrated) for scoring signal
DNF_RAW_PATH = Path(os.environ.get("F1_DNF_RAW_PATH", "artifacts/final/dnf_raw.joblib"))

# Use CAL (calibrated) for user-facing probability
DNF_CAL_PATH = Path(os.environ.get("F1_DNF_CAL_PATH", "artifacts/final/dnf_cal.joblib"))

# Dataset for API predictions
DATASET_PATH = Path(os.environ.get("F1_DATASET_PATH", "data/processed/final_hybrid_dataset.parquet"))

# Scoring knobs
ALPHA: float = 5.0
P_DNF_CAP: float = 0.30  # cap penalty influence
MODE: str = "subtract_cap"  # default combine rule: "subtract" or "subtract_cap"


app = FastAPI(title="F1 Outcome Lab")

PUBLIC_DIR = Path("frontend/public")
if PUBLIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(PUBLIC_DIR)), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Pydantic models (OpenAPI)
# ----------------------------
class HomeResponse(BaseModel):
    message: str
    docs: str
    predict: str
    predict_live: str


class MetaResponse(BaseModel):
    ranker_path: str
    dnf_raw_path: str
    dnf_cal_path: str
    alpha: float
    p_dnf_cap: float
    mode: str
    dataset: str


class RaceInfo(BaseModel):
    season: int
    round: int
    raceId: str
    raceName: Optional[str] = None


class ScoredDriver(BaseModel):
    driverId: str
    score_rank: float
    p_dnf_raw: float  # used in scoring (uncalibrated)
    p_dnf: float      # displayed to user (calibrated)
    score_adj: float  # score_rank - alpha * clip(p_dnf_raw)


class PredictResponse(BaseModel):
    raceId: str
    order: List[ScoredDriver]
    alpha: float
    p_dnf_cap: float
    mode: str

class PredictLiveResponse(BaseModel):
    raceId: str
    order: List[ScoredDriver]
    alpha: float
    p_dnf_cap: float
    mode: str
    sources: dict
    warnings: list
    form_cutoff_raceId: str


# ----------------------------
# Cached loaders (fast)
# ----------------------------
@lru_cache(maxsize=1)
def get_ranker():
    if not RANKER_PATH.exists():
        raise FileNotFoundError(f"Ranker model not found at: {RANKER_PATH}")
    return load_ranker(str(RANKER_PATH))


@lru_cache(maxsize=1)
def get_dnf_raw():
    if not DNF_RAW_PATH.exists():
        raise FileNotFoundError(f"DNF RAW model not found at: {DNF_RAW_PATH}")
    return load_dnf(str(DNF_RAW_PATH))


@lru_cache(maxsize=1)
def get_dnf_cal():
    if not DNF_CAL_PATH.exists():
        raise FileNotFoundError(f"DNF CAL model not found at: {DNF_CAL_PATH}")
    return load_dnf(str(DNF_CAL_PATH))


@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df = pd.read_parquet(DATASET_PATH)

    if "raceId" not in df.columns:
        df["raceId"] = df["season"].astype(str) + "_" + df["round"].astype(str)

    return df


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return PRODUCT_HTML


@app.get("/health", response_model=HomeResponse)
def health():
    return {
        "message": "F1 Outcome Lab is running",
        "docs": "/docs",
        "predict": "/predict/from_parquet",
        "predict_live": "/predict/live",
    }


@app.get("/meta", response_model=MetaResponse)
def meta():
    return {
        "ranker_path": str(RANKER_PATH),
        "dnf_raw_path": str(DNF_RAW_PATH),
        "dnf_cal_path": str(DNF_CAL_PATH),
        "alpha": ALPHA,
        "p_dnf_cap": P_DNF_CAP,
        "mode": MODE,
        "dataset": str(DATASET_PATH),
    }


@app.get("/races", response_model=List[RaceInfo])
def list_races(season: int | None = Query(default=None)):
    df = get_dataset()

    if season is not None:
        df = df[df["season"] == season].copy()

    cols = ["season", "round", "raceId"]
    if "raceName" in df.columns:
        cols.append("raceName")

    races = df[cols].drop_duplicates().sort_values(["season", "round"])

    if "raceName" not in races.columns:
        races["raceName"] = None

    return races.to_dict(orient="records")


@app.get("/predict/from_parquet", response_model=PredictResponse)
def predict_from_parquet(
    season: int = Query(..., ge=1950),
    round: int = Query(..., ge=1),
    mode: str = Query(default=MODE, pattern="^(subtract|subtract_cap)$"),
):
    df = get_dataset()
    race_df = df[(df["season"] == season) & (df["round"] == round)].copy()

    if race_df.empty:
        raise HTTPException(status_code=404, detail=f"No rows found for season={season}, round={round}")

    missing = [c for c in FEATURES if c not in race_df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Dataset missing required feature columns: {missing}")

    try:
        ranker = get_ranker()
        dnf_raw = get_dnf_raw()
        dnf_cal = get_dnf_cal()
    except Exception as exc:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model load failed: {type(exc).__name__}: {exc}")
    
    # Debug Telemetry as requested by User
    f1_cols = [c for c in ["q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap"] if c in race_df.columns]
    missing_rate = race_df[f1_cols].isna().mean().mean() if f1_cols else 1.0
    print("\n--- API DEBUG TRACE ---")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Models: {RANKER_PATH.name}, {DNF_RAW_PATH.name}, {DNF_CAL_PATH.name}")
    print(f"Race: {season}_{round} (rows={len(race_df)})")
    print(f"FastF1 missing rate: {missing_rate:.2%}")
    print("------------------------\n")

    X = race_df[FEATURES]  # LightGBM handles NaNs natively. Do not fillna!

    race_df["score_rank"] = ranker.predict(X)

    p_raw = dnf_raw.predict_proba(X)[:, 1]
    p_cal = dnf_cal.predict_proba(X)[:, 1]

    race_df["p_dnf_raw"] = p_raw
    race_df["p_dnf"] = p_cal

    if mode == "subtract":
        p_used = race_df["p_dnf_raw"].to_numpy()
    else:
        p_used = np.minimum(race_df["p_dnf_raw"].to_numpy(), P_DNF_CAP)

    race_df["score_adj"] = race_df["score_rank"].to_numpy() - ALPHA * p_used

    race_df = race_df.sort_values("score_adj", ascending=False)

    race_id = str(race_df["raceId"].iloc[0])
    order = race_df[["driverId", "score_rank", "p_dnf_raw", "p_dnf", "score_adj"]].to_dict(orient="records")

    return {
        "raceId": race_id,
        "order": order,
        "alpha": ALPHA,
        "p_dnf_cap": P_DNF_CAP,
        "mode": mode,
    }

@app.get("/predict/live", response_model=PredictLiveResponse)
def predict_live(
    season: int = Query(..., ge=2024),
    round: int = Query(..., ge=1),
    mode: str = Query(default=MODE, pattern="^(subtract|subtract_cap)$"),
):
    builder = LiveBuilder(DATASET_PATH)
    try:
        race_df, sources, warnings, cutoff_id = builder.build_upcoming_race(season, round)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if race_df.empty:
        raise HTTPException(status_code=404, detail=f"No drivers found for {season} round {round} from Ergast Qualifying.")
        
    # --- Strict Verification Logging (Test 1, 2, 4) ---
    print("\n" + "="*50)
    print(f"LIVE PREDICTION DIAGNOSTICS: {season} Round {round}")
    print(f"Drivers Merged: {len(race_df)}")
    if len(race_df) < 20:
        warnings.append(f"Grid has only {len(race_df)} drivers.")
        print(f"WARNING: Expected ~20 drivers, got {len(race_df)}. (Could be DNS/DNQ).")
        
    print(f"Form Cutoff Race ID (No Leakage): {cutoff_id}")
    print(f"Sources Used: {sources}")
    
    missing_rates = {c: f"{race_df[c].isna().mean():.1%}" for c in FEATURES if c in race_df.columns}
    fastf1_cols = ["q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap"]
    fastf1_coverage = {c: f"{1.0 - race_df[c].isna().mean():.1%}" for c in fastf1_cols if c in race_df.columns}
    
    print("--- Missing Rates (Core Features) ---")
    print(missing_rates)
    print("--- FastF1 Non-Null Coverage ---")
    print(fastf1_coverage)
    
    if warnings:
        print("--- Sanity Check Warnings ---")
        for w in warnings:
            print(f"   ! {w}")
    print("="*50 + "\n")
    # ------------------------------------------------
    
    missing = [c for c in FEATURES if c not in race_df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Dataset missing required feature columns: {missing}")
        
    try:
        ranker = get_ranker()
        dnf_raw = get_dnf_raw()
        dnf_cal = get_dnf_cal()
    except Exception as exc:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model load failed: {type(exc).__name__}: {exc}")
    
    X = race_df[FEATURES]

    race_df["score_rank"] = ranker.predict(X)

    p_raw = dnf_raw.predict_proba(X)[:, 1]
    p_cal = dnf_cal.predict_proba(X)[:, 1]

    race_df["p_dnf_raw"] = p_raw
    race_df["p_dnf"] = p_cal

    if mode == "subtract":
        p_used = race_df["p_dnf_raw"].to_numpy()
    else:
        p_used = np.minimum(race_df["p_dnf_raw"].to_numpy(), P_DNF_CAP)

    race_df["score_adj"] = race_df["score_rank"].to_numpy() - ALPHA * p_used

    race_df = race_df.sort_values("score_adj", ascending=False)
    
    race_id = f"{season}_{round}"
    order = race_df[["driverId", "score_rank", "p_dnf_raw", "p_dnf", "score_adj"]].to_dict(orient="records")

    return {
        "raceId": race_id,
        "order": order,
        "alpha": ALPHA,
        "p_dnf_cap": P_DNF_CAP,
        "mode": mode,
        "sources": sources,
        "warnings": warnings,
        "form_cutoff_raceId": cutoff_id
    }


# ----------------------------
# Backtest — Historical Accuracy
# ----------------------------
class RaceAccuracy(BaseModel):
    season: int
    round: int
    raceName: Optional[str] = None
    top3_hits: int
    top3_possible: int
    top10_precision: float
    kendall_tau: float

class BacktestResponse(BaseModel):
    season: int
    races: List[RaceAccuracy]
    avg_top3_rate: float
    avg_top10_precision: float
    avg_kendall_tau: float


@app.get("/predict/backtest", response_model=BacktestResponse)
def backtest(season: int = Query(..., ge=2019, le=2024)):
    from scipy.stats import kendalltau

    df = get_dataset()
    season_df = df[df["season"] == season].copy()

    if season_df.empty:
        raise HTTPException(status_code=404, detail=f"No data for season {season}")

    ranker = get_ranker()
    dnf_raw = get_dnf_raw()

    rounds = sorted(season_df["round"].unique())
    race_results: list[dict] = []

    for rnd in rounds:
        race_df = season_df[season_df["round"] == rnd].copy()
        if race_df.empty or "finishPosition" not in race_df.columns:
            continue

        X = race_df[FEATURES]
        race_df["score_rank"] = ranker.predict(X)
        p_raw = dnf_raw.predict_proba(X)[:, 1]
        p_used = np.minimum(p_raw, P_DNF_CAP)
        race_df["score_adj"] = race_df["score_rank"].to_numpy() - ALPHA * p_used

        predicted = race_df.sort_values("score_adj", ascending=False)["driverId"].tolist()
        actual = race_df.sort_values("finishPosition")["driverId"].tolist()

        # Top-3 hits
        pred_top3 = set(predicted[:3])
        actual_top3 = set(actual[:3])
        top3_hits = len(pred_top3 & actual_top3)

        # Top-10 precision
        pred_top10 = set(predicted[:10])
        actual_top10 = set(actual[:10])
        top10_prec = len(pred_top10 & actual_top10) / 10.0

        # Kendall Tau
        pred_ranks = {d: i for i, d in enumerate(predicted)}
        actual_ranks = {d: i for i, d in enumerate(actual)}
        common = sorted(set(predicted) & set(actual))
        if len(common) >= 2:
            p_r = [pred_ranks[d] for d in common]
            a_r = [actual_ranks[d] for d in common]
            tau, _ = kendalltau(p_r, a_r)
        else:
            tau = 0.0

        race_name = race_df["raceName"].iloc[0] if "raceName" in race_df.columns else None
        race_results.append({
            "season": int(season),
            "round": int(rnd),
            "raceName": race_name,
            "top3_hits": top3_hits,
            "top3_possible": 3,
            "top10_precision": round(top10_prec, 3),
            "kendall_tau": round(tau, 3),
        })

    if not race_results:
        raise HTTPException(status_code=404, detail=f"No races with results for {season}")

    avg_t3 = sum(r["top3_hits"] for r in race_results) / (3 * len(race_results))
    avg_t10 = sum(r["top10_precision"] for r in race_results) / len(race_results)
    avg_tau = sum(r["kendall_tau"] for r in race_results) / len(race_results)

    return {
        "season": season,
        "races": race_results,
        "avg_top3_rate": round(avg_t3, 3),
        "avg_top10_precision": round(avg_t10, 3),
        "avg_kendall_tau": round(avg_tau, 3),
    }
