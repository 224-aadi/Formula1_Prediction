import argparse
import pandas as pd
from f1outcome.config import SETTINGS
from f1outcome.models.ranker import train_ranker, save, FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default=str(SETTINGS.artifacts_dir / "ranker.joblib"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default=str(SETTINGS.artifacts_dir / "ranker.joblib"))
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--max_season", type=int, default=None, help="Train only on seasons <= max_season")
    ap.add_argument("--max_round", type=int, default=None, help="If set, train only on rounds <= max_round")
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

    if args.season is not None:
        df = df[df["season"] == args.season].copy()
    if args.max_round is not None:
        df = df[df["round"] <= args.max_round].copy()

    model = train_ranker(df)
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None and hasattr(model, "booster_"):
            importances = model.booster_.feature_importance()
        if importances is not None:
            pairs = list(zip(FEATURES, importances))
            pairs.sort(key=lambda x: x[1], reverse=True)
            print("Feature importances (desc):")
            for name, val in pairs:
                print(f"  {name}: {val}")
        else:
            print("Feature importances not available on model.")
    except Exception as e:
        print(f"Failed to read feature importances: {type(e).__name__}: {e}")
    save(model, args.out)
    print(f"Saved model -> {args.out} (rows={len(df):,})")

if __name__ == "__main__":
    main()
