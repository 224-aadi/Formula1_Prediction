from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    jolpica_base: str = "https://api.jolpi.ca/ergast/f1"  # Ergast-compatible API :contentReference[oaicite:2]{index=2}
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")

SETTINGS = Settings()
