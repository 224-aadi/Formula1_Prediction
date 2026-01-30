import argparse
import pandas as pd
from src.f1outcome.config import SETTINGS
from src.f1outcome.models.dnf import train_dnf, save

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(SETTINGS.artifacts_dir / "dnf_2019_2024.joblib"))
    args = ap.parse_args()

    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    model = train_dnf(df)
    save(model, args.out)
    print(f"Saved DNF model -> {args.out} (rows={len(df):,})")

if __name__ == "__main__":
    main()
