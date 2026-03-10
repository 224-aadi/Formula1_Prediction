import argparse
import pandas as pd
from f1outcome.config import SETTINGS
from f1outcome.models.dnf import train_dnf, save

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(SETTINGS.artifacts_dir / "dnf_2019_2024.joblib"))
    ap.add_argument("--max_season", type=int, default=None, help="Train only on seasons <= max_season")
    args = ap.parse_args()

    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    
    # --- Fix: Normalize types for filtering consistency ---
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    if "constructorId" in df.columns:
        df["constructorId"] = df["constructorId"].astype(str).str.strip().str.lower()
    # ----------------------------------------------------

    if args.max_season is not None:
        print(f"Training strict time split: season <= {args.max_season} only.")
        df = df[df["season"] <= args.max_season].copy()

    model = train_dnf(df)
    save(model, args.out)
    print(f"Saved DNF model -> {args.out} (rows={len(df):,})")

if __name__ == "__main__":
    main()
