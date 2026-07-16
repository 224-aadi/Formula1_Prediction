#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from f1outcome.config import SETTINGS
from f1outcome.data.build_dataset import DatasetBuilder
from f1outcome.data.jolpica import JolpicaClient
from f1outcome.models.dnf import is_dnf_status, save as save_dnf, train_dnf
from f1outcome.models.ranker import FEATURES, save as save_ranker, train_ranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild production data and model artifacts.")
    parser.add_argument("--start-season", type=int, default=2019)
    parser.add_argument("--end-season", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--form-window", type=int, default=5)
    parser.add_argument("--use-fastf1", action="store_true")
    parser.add_argument("--fastf1-cache", default=str(SETTINGS.raw_dir / "fastf1_cache"))
    return parser.parse_args()


def train_calibrated_dnf(df: pd.DataFrame):
    train_df = df.copy()
    train_df["dnf"] = train_df["status"].apply(is_dnf_status).astype(int)

    base_model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(train_df[FEATURES], train_df["dnf"])
    return model


def main() -> None:
    args = parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))

    SETTINGS.raw_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.processed_dir.mkdir(parents=True, exist_ok=True)
    (SETTINGS.artifacts_dir / "final").mkdir(parents=True, exist_ok=True)

    client = JolpicaClient(
        base_url=SETTINGS.jolpica_base,
        cache_dir=SETTINGS.raw_dir / "jolpica_cache",
        min_interval_s=1.10,
    )
    builder = DatasetBuilder(client)
    df = builder.build(
        seasons,
        form_window=args.form_window,
        use_fastf1=args.use_fastf1,
        fastf1_cache_dir=Path(args.fastf1_cache),
    )
    if df.empty:
        raise SystemExit("No completed race data was fetched; refusing to overwrite production artifacts.")

    df = df.sort_values(["season", "round", "finishPosition"], na_position="last").reset_index(drop=True)
    dataset_path = SETTINGS.processed_dir / "final_hybrid_dataset.parquet"
    df.to_parquet(dataset_path, index=False)

    print(f"Training ranker on {len(df):,} rows...")
    ranker = train_ranker(df)
    save_ranker(ranker, "artifacts/final/ranker.joblib")

    print("Training raw DNF model...")
    dnf_raw = train_dnf(df)
    save_dnf(dnf_raw, "artifacts/final/dnf_raw.joblib")
    save_dnf(dnf_raw, "artifacts/final/dnf.joblib")

    print("Training calibrated DNF model...")
    dnf_cal = train_calibrated_dnf(df)
    joblib.dump(dnf_cal, "artifacts/final/dnf_cal.joblib")

    races_by_season = (
        df[["season", "round"]]
        .drop_duplicates()
        .groupby("season")["round"]
        .agg(["min", "max", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "rows": int(len(df)),
        "seasons": [int(s) for s in sorted(df["season"].dropna().unique())],
        "latest_season": int(df["season"].max()),
        "latest_round": int(df[df["season"] == df["season"].max()]["round"].max()),
        "races_by_season": races_by_season,
        "features": FEATURES,
        "use_fastf1": bool(args.use_fastf1),
    }
    Path("artifacts/final/data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
