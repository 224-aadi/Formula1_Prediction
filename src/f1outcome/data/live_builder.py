import pandas as pd
import numpy as np
from pathlib import Path
from f1outcome.data.jolpica import JolpicaClient
from f1outcome.data.fastf1_features import fastf1_driver_features, FastF1FeatureConfig, get_weather_flag
from f1outcome.data.build_dataset import _parse_time_to_ms, _to_int
from f1outcome.config import SETTINGS

class LiveBuilder:
    """Builds a feature matrix for an emerging/upcoming race weekend."""
    def __init__(self, historical_parquet: str | Path):
        self.hist_df = pd.read_parquet(historical_parquet)
        self.client = JolpicaClient(
            base_url=SETTINGS.jolpica_base,
            cache_dir=Path("data/raw/jolpica_cache"),
            min_interval_s=0.2
        )
        
    def _get_last_known_form(self, target_season: int, target_round: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Extracts the most recent rolling form values for every driver and team strictly BEFORE the target race."""
        # Enforce strict no-leakage (Test 2)
        # Filter strictly before the target date
        hist = self.hist_df.copy()
        
        # We need races where (season < target_season) OR (season == target_season AND round < target_round)
        mask = (hist["season"] < target_season) | ((hist["season"] == target_season) & (hist["round"] < target_round))
        hist = hist[mask].sort_values(["season", "round"])
        
        if hist.empty:
            raise ValueError(f"No historical races found prior to {target_season} Round {target_round} for form calculation!")
            
        last_race_id = str(hist["raceId"].iloc[-1])
        
        # Get the VERY LAST row for each driver
        last_driver = hist.drop_duplicates(subset=["driverId"], keep="last").copy()
        driver_form = last_driver[[
            "driverId", "driverFormAvgFinish", "driverFormAvgPoints", 
            "driver_dnf_rate", "driver_mech_dnf_rate"
        ]]
        
        # Get the VERY LAST row for each constructor
        last_team = hist.drop_duplicates(subset=["constructorId"], keep="last").copy()
        team_form = last_team[[
            "constructorId", "teamFormAvgFinish", "teamFormAvgPoints",
            "team_dnf_rate", "team_mech_dnf_rate"
        ]]
        
        return driver_form, team_form, last_race_id
        
    def fetch_live_qualifying(self, season: int, rnd: int) -> pd.DataFrame:
        """Fetches the actual qualifying results for the upcoming race."""
        js = self.client.get_json(f"{season}/{rnd}/qualifying.json")
        races = js["MRData"]["RaceTable"]["Races"]
        if not races:
            raise ValueError(f"No qualifying data found for {season} Round {rnd}. Qualifying must be finished to predict the race!")
            
        q = races[0].get("QualifyingResults", [])
        circuitId = races[0]["Circuit"]["circuitId"]
        
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
                "givenName": r["Driver"].get("givenName"),
                "familyName": r["Driver"].get("familyName"),
                "constructorId": r["Constructor"]["constructorId"],
                "qualiBestMs": best,
                "qualiPos": _to_int(r.get("position")),
                "grid": _to_int(r.get("position")), # Assume grid = qualiPos before penalties
                "circuitId": circuitId
            })
            
        df = pd.DataFrame(rows)
        
        # Calculate gaps
        pole = df["qualiBestMs"].min()
        if pd.isna(pole) or pole <= 0:
            df["qualiGapMs"] = None
            df["qualiGapPct"] = None
        else:
            df["qualiGapMs"] = df["qualiBestMs"] - pole
            df["qualiGapPct"] = df["qualiGapMs"] / pole
            
        # Grid bucket
        def get_grid_bucket(grid):
            if pd.isna(grid) or grid <= 0: return 4
            if grid <= 3: return 1  
            if grid <= 10: return 2 
            if grid <= 20: return 3 
            return 4
        df["grid_bucket"] = df["grid"].apply(get_grid_bucket)
        
        # Street circuit
        street_circuits = {
            "monaco", "marina_bay", "baku", "jeddah", "miami", "las_vegas", 
            "albert_park", "villeneuve", "sochi", "valencia", "adler", "detroit", "dallas"
        }
        df["is_street_circuit"] = df["circuitId"].apply(lambda c: 1 if str(c) in street_circuits else 0)
        
        # --- NEW: Chaos & Weather Features ---
        sc_heavy_circuits = {"monaco", "marina_bay", "baku", "jeddah", "albert_park", "miami", "las_vegas", "zandvoort"}
        sc_med_circuits = {"interlagos", "montreal", "suzuka", "spa", "imola", "nurburgring"}
        
        def get_sc_prob(cid):
            c = str(cid).lower()
            if c in sc_heavy_circuits: return 0.85
            if c in sc_med_circuits: return 0.50
            return 0.20
            
        df["track_sc_prob"] = df["circuitId"].apply(get_sc_prob)
        # Live weather from FastF1 (falls back to 0 for future/unavailable races)
        try:
            wet_flag = get_weather_flag(season, rnd, FastF1FeatureConfig(cache_dir=Path("data/raw/fastf1_cache")))
        except Exception:
            wet_flag = 0
        df["is_wet_race"] = wet_flag
        # -------------------------------------
        
        df["trackId"] = 0 # Not used by model anymore
        
        # Normalze name for FastF1
        df["fullName_norm"] = (df["givenName"].fillna("") + df["familyName"].fillna("")).apply(
            lambda s: "".join(ch.lower() for ch in str(s) if ch.isalnum())
        )
        
        return df

    def build_upcoming_race(self, season: int, rnd: int) -> tuple[pd.DataFrame, dict, list, str]:
        """Assembles the complete feature matrix for X_pred. Returns (df, sources, warnings, cutoff_id)."""
        warnings_list = []
        sources = {"ergast": False, "fastf1": False, "dataset_form": True}
        
        # 1. Fetch live quali
        try:
            df = self.fetch_live_qualifying(season, rnd)
            sources["ergast"] = True
            
            # Sanity checks (Test 4)
            if not df.empty:
                max_grid = df["grid"].max()
                if max_grid > 20 or max_grid < 1:
                    warnings_list.append(f"Grid position out of bounds: max {max_grid}")
                
                max_gap = df["qualiGapPct"].max()
                if max_gap is not None and max_gap > 0.1:
                    warnings_list.append(f"Warning: Huge quali gap detected: {max_gap:.2%} for some drivers.")
                    
        except Exception as e:
            raise ValueError(f"Ergast Qualifying required to predict. Failed: {e}")
        
        # 2. Get historical rolling form
        driver_form, team_form, cutoff_id = self._get_last_known_form(season, rnd)
        
        df = df.merge(driver_form, on="driverId", how="left")
        df = df.merge(team_form, on="constructorId", how="left")
        
        # 3. Teammate Delta
        def calc_teammate_delta(group):
            tdeltas = []
            for _, row in group.iterrows():
                teammates = group[(group["driverId"] != row["driverId"])]
                if not teammates.empty:
                    teammate_form = teammates["driverFormAvgFinish"].mean()
                    tdeltas.append(row["driverFormAvgFinish"] - teammate_form)
                else:
                    tdeltas.append(0.0)
            group["teammateFormFinishDelta"] = tdeltas
            return group
            
        df = df.groupby(["season", "round", "constructorId"], group_keys=False).apply(calc_teammate_delta)
        
        # 4. Fetch Live FastF1
        cfg = FastF1FeatureConfig(cache_dir=Path("data/raw/fastf1_cache"))
        try:
            f1 = fastf1_driver_features(season, rnd, cfg)
            if not f1.empty:
                df = df.merge(f1, on="fullName_norm", how="left")
                sources["fastf1"] = True
                
                # Sanity checks for FastF1
                for c in ["q_s1_gap", "q_s2_gap", "q_s3_gap"]:
                    if c in df.columns:
                        negatives = df[df[c] < -1.0] # Massive negative gap
                        if not negatives.empty:
                            warnings_list.append(f"Suspicious FastF1 negative gap in {c} for {negatives['driverId'].tolist()}")
        except Exception as e:
            warnings_list.append(f"FastF1 fetch failed (fallback to Ergast used): {e}")
            
        # Ensure FastF1 empty columns exist
        for c in ["q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap"]:
            if c not in df.columns:
                df[c] = float('nan')
                
        # Fill missing numeric forms with defaults if a true rookie appears
        df.fillna({
            "driverFormAvgFinish": 15.0,
            "driverFormAvgPoints": 0.0,
            "teamFormAvgFinish": 15.0,
            "teamFormAvgPoints": 0.0,
            "driver_dnf_rate": 0.15,
            "driver_mech_dnf_rate": 0.05,
            "team_dnf_rate": 0.15,
            "team_mech_dnf_rate": 0.05,
            "teammateFormFinishDelta": 0.0
        }, inplace=True)
        
        return df, sources, warnings_list, cutoff_id
