import pandas as pd
from src.f1outcome.models.dnf import is_dnf_status

df = pd.read_parquet("data/processed/dataset_seasons_2019_2024.parquet")
df24 = df[df["season"] == 2024].copy()

# show counts of status strings
print("\nTop statuses in 2024:")
print(df24["status"].value_counts().head(30))

# show what your current label marks as DNF
df24["dnf"] = df24["status"].apply(is_dnf_status).astype(int)
print("\n2024 DNF rate (current label):", df24["dnf"].mean())

print("\nStatuses counted as DNF (top 30):")
dnf_statuses = df24[df24["dnf"] == 1]["status"].value_counts().head(30)
print(dnf_statuses)
