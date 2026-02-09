from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Tuple
 # NOTE: avoid shadowing pathlib.Path with anyio.Path here.
import pandas as pd
from tqdm import tqdm
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
        fastf1_ok = 0
        fastf1_fail = 0
        last_heartbeat = time.time()

        for season in seasons:
            use_fastf1_for_season = use_fastf1 and int(season) >= 2018
            consec_fail = 0
            rounds = self.get_season_rounds(season)
            pbar = tqdm(rounds, desc=f"Season {season}", unit="round")

            for rnd in pbar:
                # heartbeat every ~15s even if things are slow
                now = time.time()
                if now - last_heartbeat > 15:
                    tqdm.write(f"[heartbeat] still working... season={season} round={rnd}")
                    last_heartbeat = now

                res = self.fetch_round_results(season, rnd)
                if res.empty:
                    pbar.set_postfix_str("results=empty")
                    continue

                qua = self.fetch_round_qualifying(season, rnd)
                df = res.merge(qua, on=["season", "round", "driverId"], how="left")

                # ---- optional: merge FastF1 features for this race ----
                if use_fastf1_for_season:
                    try:
                        cfg = FastF1FeatureConfig(cache_dir=fastf1_cache_dir)

                        f = fastf1_driver_features(int(season), int(rnd), cfg)
                        if f is None or f.empty:
                            fastf1_fail += 1
                            consec_fail += 1
                            pbar.set_postfix_str(f"FastF1=EMPTY ({fastf1_ok} ok/{fastf1_fail} fail)")
                        else:
                            # IMPORTANT: prevent _x/_y duplicates
                            for c in ["q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap"]:
                                if c in df.columns:
                                    df.drop(columns=[c], inplace=True)

                            # join key must exist
                            if "fullName_norm" not in df.columns:
                                df["fullName_norm"] = (df["givenName"].fillna("") + df["familyName"].fillna("")).apply(
                                    lambda s: "".join(ch.lower() for ch in str(s) if ch.isalnum())
                                )

                            df = df.merge(f, on="fullName_norm", how="left")

                            fastf1_ok += 1
                            consec_fail = 0
                            pbar.set_postfix_str(f"FastF1=OK ({fastf1_ok} ok/{fastf1_fail} fail)")

                    except Exception as e:
                        fastf1_fail += 1
                        consec_fail += 1
                        pbar.set_postfix_str(f"FastF1=FAIL ({fastf1_ok} ok/{fastf1_fail} fail)")

                        # Log the actual error so you can inspect it after the run
                        try:
                            from pathlib import Path
                            Path("data/raw").mkdir(parents=True, exist_ok=True)
                            with open("data/raw/fastf1_failures.log", "a", encoding="utf-8") as fp:
                                fp.write(f"{season},{rnd} :: {type(e).__name__}: {e}\n")
                        except Exception:
                            pass

                    # If FastF1 is failing repeatedly, stop trying for the rest of this season
                    if consec_fail >= 5:
                        tqdm.write(
                            f"[fastf1] disabling FastF1 for season {season} after {consec_fail} consecutive failures"
                        )
                        use_fastf1_for_season = False

                all_rows.append(df)

        all_rows = [df for df in all_rows if df is not None and not df.empty]
        if not all_rows:
            return pd.DataFrame()
        print(f"[post] collected {len(all_rows)} non-empty race frames; concatenating...")
        data = pd.concat(all_rows, ignore_index=True)
        print(f"[post] concatenated rows={len(data):,}; computing rolling form features...")

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
        # Team form = rolling mean of team-average finish over prior races only
        team_race = (
            data.groupby(["constructorId", "season", "round"], as_index=False)
            .agg(teamRaceAvgFinish=("finishPosition", "mean"))
            .sort_values(["constructorId", "season", "round"])
        )
        g = team_race.groupby("constructorId", sort=False)
        team_race["teamFormAvgFinish"] = (
            g["teamRaceAvgFinish"]
            .rolling(window=form_window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        team_race["teamFormAvgFinish"] = g["teamFormAvgFinish"].shift(1)
        data = data.merge(
            team_race[["constructorId", "season", "round", "teamFormAvgFinish"]],
            on=["constructorId", "season", "round"],
            how="left",
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
