from pathlib import Path
import argparse

from src.f1outcome.config import SETTINGS
from src.f1outcome.data.jolpica import JolpicaClient
from src.f1outcome.data.build_dataset import DatasetBuilder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023])
    ap.add_argument("--form_window", type=int, default=5)

    # NEW: FastF1 feature augmentation
    ap.add_argument("--use_fastf1", action="store_true", help="Augment dataset with FastF1 timing-derived features (best for 2018+).")
    ap.add_argument("--fastf1_cache", type=str, default=str(SETTINGS.raw_dir / "fastf1_cache"))

    args = ap.parse_args()

    SETTINGS.raw_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.processed_dir.mkdir(parents=True, exist_ok=True)

    client = JolpicaClient(
        base_url=SETTINGS.jolpica_base,
        cache_dir=SETTINGS.raw_dir / "jolpica_cache",
        min_interval_s=0.40,  # slightly safer to avoid 429 bursts
    )

    builder = DatasetBuilder(client)

    df = builder.build(
        args.seasons,
        form_window=args.form_window,
        use_fastf1=args.use_fastf1,
        fastf1_cache_dir=Path(args.fastf1_cache),
    )

    suffix = "_fastf1" if args.use_fastf1 else ""
    out = SETTINGS.processed_dir / f"dataset_seasons_{'_'.join(map(str, args.seasons))}{suffix}.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows")

if __name__ == "__main__":
    main()
