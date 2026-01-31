from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from anyio import Path
import pandas as pd
from .jolpica import JolpicaClient
from src.f1outcome.data.fastf1_features import fastf1_driver_features, FastF1FeatureConfig


def _to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _parse_time_to_ms(t: str):
    """
    Jolpica qualifying times often like "1:30.123".
    Return ms or None.
    """
    if not t or not isinstance(t, str):
        return None
    try:
        mins, rest = t.split(":")
        secs, ms = rest.split(".")
        return (int(mins) * 60 + int(secs)) * 1000 + int(ms.ljust(3, "0")[:3])
    except Exception:
        return None

@dataclass
class DatasetBuilder:
    client: JolpicaClient

    def get_season_rounds(self, season: int) -> List[int]:
        js = self.client.get_json(f"{season}.json")
        races = js["MRData"]["RaceTable"]["Races"]
        return [_to_int(r["round"]) for r in races if _to_int(r.get("round")) is not None]

    def fetch_round_results(self, season: int, rnd: int) -> pd.DataFrame:
        js = self.client.get_json(f"{season}/{rnd}/results.json")
        races = js["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        results = races[0]["Results"]
        rows = []
        for r in results:
            rows.append({
                "season": season,
                "round": rnd,
                "raceName": races[0].get("raceName"),
                "driverId": r["Driver"]["driverId"],
                "givenName": r["Driver"].get("givenName"),
                "familyName": r["Driver"].get("familyName"),
                "constructorName": r["Constructor"].get("name"),
                "constructorId": r["Constructor"]["constructorId"],
                "grid": _to_int(r.get("grid")),
                "finishPosition": _to_int(r.get("position")),
                "status": r.get("status"),
            })
        return pd.DataFrame(rows)

    def fetch_round_qualifying(self, season: int, rnd: int) -> pd.DataFrame:
        js = self.client.get_json(f"{season}/{rnd}/qualifying.json")
        races = js["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        q = races[0].get("QualifyingResults", [])
        rows = []
        for r in q:
            q1 = _parse_time_to_ms(r.get("Q1"))
            q2 = _parse_time_to_ms(r.get("Q2"))
            q3 = _parse_time_to_ms(r.get("Q3"))
            best = min([x for x in [q1, q2, q3] if x is not None], default=None)
            rows.append({
                "season": season,
                "round": rnd,
                "driverId": r["Driver"]["driverId"],
                "qualiBestMs": best,
                "qualiPos": _to_int(r.get("position")),
            })
        return pd.DataFrame(rows)

    def build(self, seasons: List[int], form_window: int = 5, use_fastf1: bool = False, fastf1_cache_dir: Path | None = None) -> pd.DataFrame:
        all_rows = []
        for season in seasons:
            rounds = self.get_season_rounds(season)
            for rnd in rounds:
                res = self.fetch_round_results(season, rnd)
                if res.empty:
                    continue
                qua = self.fetch_round_qualifying(season, rnd)
                df = res.merge(qua, on=["season", "round", "driverId"], how="left")
                if "qualiBestMs" in df.columns:
                    pole = df.groupby(["season", "round"])["qualiBestMs"].transform("min")
                    df["qualiGapMs"] = df["qualiBestMs"] - pole
                    df["qualiGapPct"] = (df["qualiBestMs"] - pole) / pole
                all_rows.append(df)

        data = pd.concat(all_rows, ignore_index=True)

        # Rolling “form” features (previous races only => no leakage)
        data = data.sort_values(["season", "round"])
        data["raceId"] = data["season"].astype(str) + "_" + data["round"].astype(str)

        def add_rolling(group, col, out):
            group = group.sort_values(["season", "round"])
            group[out] = group[col].shift(1).rolling(form_window, min_periods=1).mean()
            return group

        data = data.groupby("driverId", group_keys=False).apply(
            lambda g: add_rolling(g, "finishPosition", "driverFormAvgFinish")
        )
        data = data.groupby("constructorId", group_keys=False).apply(
            lambda g: add_rolling(g, "finishPosition", "teamFormAvgFinish")
        )

        # Target relevance for ranking (winner highest)
        # If finishPosition is 1..20, convert to relevance: 21 - position
        data["relevance"] = data["finishPosition"].apply(lambda p: (21 - p) if isinstance(p, int) else None)
        data["fullName_norm"] = (data["givenName"].fillna("") + data["familyName"].fillna("")).apply(lambda s: "".join(ch.lower() for ch in str(s) if ch.isalnum()))

        if use_fastf1:
            cfg = FastF1FeatureConfig(cache_dir=fastf1_cache_dir or Path("data/raw/fastf1_cache"))

            # For each (season, round, raceName) get features and merge
            feat_rows = []
            for (season, rnd), g in data.groupby(["season", "round"]):
                race_name = g["raceName"].iloc[0]
                # FastF1 works best 2018+; skip older automatically
                if int(season) < 2018:
                    continue
                try:
                    f = fastf1_driver_features(int(season), int(rnd), cfg)
                    if not f.empty:
                        f["season"] = int(season)
                        f["round"] = int(rnd)
                        feat_rows.append(f)
                except Exception:
                    # Don't fail dataset build because one event didn't load
                    continue

            if feat_rows:
                feats = pd.concat(feat_rows, ignore_index=True)
                data = data.merge(feats, on=["season", "round", "fullName_norm"], how="left")

        return data
