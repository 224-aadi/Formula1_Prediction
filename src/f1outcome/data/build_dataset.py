from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
from .jolpica import JolpicaClient
from .fastf1_features import fastf1_driver_features, FastF1FeatureConfig


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
                "points": float(r.get("points", 0.0)),
                "status": r.get("status"),
                "circuitId": races[0]["Circuit"]["circuitId"],
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

                # --- NEW: Chaos & Weather Features ---
                sc_heavy_circuits = {"monaco", "marina_bay", "baku", "jeddah", "albert_park", "miami", "las_vegas", "zandvoort"}
                sc_med_circuits = {"interlagos", "montreal", "suzuka", "spa", "imola", "nurburgring"}
                
                def get_sc_prob(cid):
                    c = str(cid).lower()
                    if c in sc_heavy_circuits: return 0.85
                    if c in sc_med_circuits: return 0.50
                    return 0.20
                    
                if "circuitId" in df.columns:
                    df["track_sc_prob"] = df["circuitId"].apply(get_sc_prob)
                else:
                    df["track_sc_prob"] = 0.20
                    
                df["is_wet_race"] = 0 # Placeholder: Could scrape weather from FastF1 if needed
                # -------------------------------------

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
                        if "RateLimitError" in str(type(e)):
                            # Global killswitch for FastF1 if rate limit hit
                            tqdm.write(f"[fastf1] RateLimitError hit in Season {season}. Disabling FastF1 entirely to protect API.")
                            use_fastf1_for_season = False
                            use_fastf1 = False # Stop for all subsequent seasons too
                            
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
        data = data.groupby("driverId", group_keys=False).apply(
            lambda g: add_rolling(g, "points", "driverFormAvgPoints")
        )
        
        # Team form = rolling mean of team-average finish/points over prior races only
        team_race = (
            data.groupby(["constructorId", "season", "round"], as_index=False)
            .agg(
                teamRaceAvgFinish=("finishPosition", "mean"),
                teamRaceAvgPoints=("points", "mean")
            )
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
        
        team_race["teamFormAvgPoints"] = (
            g["teamRaceAvgPoints"]
            .rolling(window=form_window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        team_race["teamFormAvgPoints"] = g["teamFormAvgPoints"].shift(1)
        
        data = data.merge(
            team_race[["constructorId", "season", "round", "teamFormAvgFinish", "teamFormAvgPoints"]],
            on=["constructorId", "season", "round"],
            how="left",
        )
        
        # Teammate Delta (Driver Form Finish - Teammate Form Finish)
        # Note: If team has >2 drivers in a race (rare but possible), this averages all others.
        def calc_teammate_delta(group):
            tdeltas = []
            for _, row in group.iterrows():
                teammates = group[(group["driverId"] != row["driverId"])]
                if not teammates.empty:
                    teammate_form = teammates["driverFormAvgFinish"].mean()
                    tdeltas.append(row["driverFormAvgFinish"] - teammate_form)
                else:
                    tdeltas.append(0.0) # No teammate data
            group["teammateFormFinishDelta"] = tdeltas
            return group
            
        data = data.groupby(["season", "round", "constructorId"], group_keys=False).apply(calc_teammate_delta)
        

        # --- Reliability Features (DNF Rates) ---
        def is_dnf(status):
            s = str(status).strip()
            if s in {"Finished", "Lapped"} or (s.startswith("+") and "Lap" in s):
                return 0
            return 1
        
        def is_mech_dnf(status):
            s = str(status).strip().lower()
            # Common mechanical failures in Ergast
            mech = {
                "engine", "gearbox", "transmission", "clutch", "hydraulics", 
                "electrical", "radiator", "suspension", "brakes", "differential", 
                "overheating", "mechanical", "power unit", "battery", "electronics",
                "drivetrain", "cooling system", "oil leak", "water leak", "exhaust"
            }
            if any(m in s for m in mech):
                return 1
            return 0
            
        data["is_dnf"] = data["status"].apply(is_dnf)
        data["is_mech_dnf"] = data["status"].apply(is_mech_dnf)
        
        # Helper for rolling stats: shift(1) to avoid leakage
        def add_rolling_rate(group, target_col, out_col, window):
            group = group.sort_values(["season", "round"])
            # rolling mean of binary 0/1 is the rate
            group[out_col] = group[target_col].shift(1).rolling(window, min_periods=1).mean()
            return group

        # Driver Reliability (Last 20 races)
        data = data.groupby("driverId", group_keys=False).apply(
            lambda g: add_rolling_rate(g, "is_dnf", "driver_dnf_rate", 20)
        )
        data = data.groupby("driverId", group_keys=False).apply(
            lambda g: add_rolling_rate(g, "is_mech_dnf", "driver_mech_dnf_rate", 20)
        )
        
        # Team Reliability (Last 20 races)
        data = data.groupby("constructorId", group_keys=False).apply(
            lambda g: add_rolling_rate(g, "is_dnf", "team_dnf_rate", 20)
        )
        data = data.groupby("constructorId", group_keys=False).apply(
            lambda g: add_rolling_rate(g, "is_mech_dnf", "team_mech_dnf_rate", 20)
        )
        
        # --- Track Context & Grid Features ---
        street_circuits = {
            "monaco", "marina_bay", "baku", "jeddah", "miami", "las_vegas", 
            "albert_park", "villeneuve", "sochi", "valencia", "adler", "detroit", "dallas"
        }
        
        if "circuitId" in data.columns:
            data["is_street_circuit"] = data["circuitId"].apply(lambda c: 1 if str(c) in street_circuits else 0)
            data["trackId"] = data["circuitId"].astype("category").cat.codes
        else:
            pass
            
        def get_grid_bucket(grid):
            if pd.isna(grid) or grid <= 0: return 4 # Pit lane / default
            if grid <= 3: return 1  # P1-P3
            if grid <= 10: return 2 # P4-P10
            if grid <= 20: return 3 # P11-P20
            return 4 # P21+
            
        data["grid_bucket"] = data["grid"].apply(get_grid_bucket)
        
        # --- Qualifying Gaps ---
        # Calculate gap to pole for each race
        def calc_quali_gaps(g):
            if "qualiBestMs" not in g.columns or g["qualiBestMs"].dropna().empty:
                g["qualiGapMs"] = None
                g["qualiGapPct"] = None
                return g
            
            pole = g["qualiBestMs"].min()
            if pd.isna(pole) or pole <= 0:
                g["qualiGapMs"] = None
                g["qualiGapPct"] = None
                return g

            g["qualiGapMs"] = g["qualiBestMs"] - pole
            g["qualiGapPct"] = g["qualiGapMs"] / pole
            return g

        data = data.groupby(["season", "round"], group_keys=False).apply(calc_quali_gaps)
        
        # --- Ensure All Features Exist (fill missing with NaN) ---
        # This handles cases where FastF1 failed entirely or some features weren't computed
        expected_features = [
            "grid", "qualiPos", "qualiGapMs", "qualiGapPct", 
            "driverFormAvgFinish", "teamFormAvgFinish", 
            "driverFormAvgPoints", "teamFormAvgPoints", "teammateFormFinishDelta", "grid_bucket", # New
            "q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap",
            "driver_dnf_rate", "driver_mech_dnf_rate",
            "team_dnf_rate", "team_mech_dnf_rate",
            "is_street_circuit", "trackId"
        ]
        
        for c in expected_features:
            if c not in data.columns:
                data[c] = float("nan")

        # ----------------------------------------

        # Target relevance for ranking (winner highest)
        # If finishPosition is 1..20, convert to relevance: 21 - position
        data["relevance"] = data["finishPosition"].apply(
            lambda p: max(0, 21 - int(p)) if pd.notna(p) else None
        )
        data["fullName_norm"] = (data["givenName"].fillna("") + data["familyName"].fillna("")).apply(lambda s: "".join(ch.lower() for ch in str(s) if ch.isalnum()))

        return data
