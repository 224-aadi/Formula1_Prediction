import pandas as pd
from src.f1outcome.models.dnf import is_dnf_status

df = pd.read_parquet("data/processed/dataset_seasons_2019_2024.parquet")

# overall rate (per driver-race row)
df["dnf"] = df["status"].apply(is_dnf_status).astype(int)
print("Overall DNF rate:", df["dnf"].mean())

# 2024 overall + early rounds (your calibration slice)
df24 = df[df["season"] == 2024].copy()
print("2024 DNF rate:", df24["dnf"].mean())

df24_early = df24[df24["round"] <= 12]
print("2024 rounds 1-12 DNF rate:", df24_early["dnf"].mean())
