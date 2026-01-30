import pandas as pd
from pathlib import Path

a = Path("data/processed/dataset_seasons_2019_2020_2021.parquet")
b = Path("data/processed/dataset_seasons_2022_2023_2024.parquet")
out = Path("data/processed/dataset_seasons_2019_2024.parquet")

df = pd.concat([pd.read_parquet(a), pd.read_parquet(b)], ignore_index=True)
df.to_parquet(out, index=False)
print("Wrote:", out, "rows:", len(df))
