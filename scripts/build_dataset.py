from pathlib import Path
import argparse
from f1outcome.config import SETTINGS
from f1outcome.data.jolpica import JolpicaClient
from f1outcome.data.build_dataset import DatasetBuilder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023])
    ap.add_argument("--form_window", type=int, default=5)
    args = ap.parse_args()

    SETTINGS.raw_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.processed_dir.mkdir(parents=True, exist_ok=True)

    client = JolpicaClient(
        base_url=SETTINGS.jolpica_base,
        cache_dir=SETTINGS.raw_dir / "jolpica_cache",
        min_interval_s=0.30,
    )

    builder = DatasetBuilder(client)
    df = builder.build(args.seasons, form_window=args.form_window)

    out = SETTINGS.processed_dir / f"dataset_seasons_{'_'.join(map(str,args.seasons))}.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows")

if __name__ == "__main__":
    main()
