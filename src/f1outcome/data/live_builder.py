import pandas as pd
import numpy as np
import os
from pathlib import Path
from f1outcome.data.jolpica import JolpicaClient
from f1outcome.data.fastf1_features import fastf1_driver_features, FastF1FeatureConfig, get_weather_flag
from f1outcome.data.build_dataset import _parse_time_to_ms, _to_int
from f1outcome.config import SETTINGS

FALLBACK_RACE_METADATA = {
    (2026, 1): {"raceName": "Australian Grand Prix", "circuitId": "albert_park", "circuitName": "Albert Park Grand Prix Circuit", "date": "2026-03-08"},
    (2026, 2): {"raceName": "Chinese Grand Prix", "circuitId": "shanghai", "circuitName": "Shanghai International Circuit", "date": "2026-03-15"},
    (2026, 3): {"raceName": "Japanese Grand Prix", "circuitId": "suzuka", "circuitName": "Suzuka Circuit", "date": "2026-03-29"},
    (2026, 4): {"raceName": "Miami Grand Prix", "circuitId": "miami", "circuitName": "Miami International Autodrome", "date": "2026-05-03"},
    (2026, 5): {"raceName": "Canadian Grand Prix", "circuitId": "villeneuve", "circuitName": "Circuit Gilles Villeneuve", "date": "2026-05-24"},
    (2026, 6): {"raceName": "Monaco Grand Prix", "circuitId": "monaco", "circuitName": "Circuit de Monaco", "date": "2026-06-07"},
    (2026, 7): {"raceName": "Barcelona Grand Prix", "circuitId": "catalunya", "circuitName": "Circuit de Barcelona-Catalunya", "date": "2026-06-14"},
    (2026, 8): {"raceName": "Austrian Grand Prix", "circuitId": "red_bull_ring", "circuitName": "Red Bull Ring", "date": "2026-06-28"},
    (2026, 9): {"raceName": "British Grand Prix", "circuitId": "silverstone", "circuitName": "Silverstone Circuit", "date": "2026-07-05"},
    (2026, 10): {"raceName": "Belgian Grand Prix", "circuitId": "spa", "circuitName": "Circuit de Spa-Francorchamps", "date": "2026-07-19"},
    (2026, 11): {"raceName": "Hungarian Grand Prix", "circuitId": "hungaroring", "circuitName": "Hungaroring", "date": "2026-07-26"},
    (2026, 12): {"raceName": "Dutch Grand Prix", "circuitId": "zandvoort", "circuitName": "Circuit Zandvoort", "date": "2026-08-23"},
    (2026, 13): {"raceName": "Italian Grand Prix", "circuitId": "monza", "circuitName": "Autodromo Nazionale di Monza", "date": "2026-09-06"},
    (2026, 14): {"raceName": "Spanish Grand Prix", "circuitId": "madrid", "circuitName": "Madring", "date": "2026-09-13"},
    (2026, 15): {"raceName": "Azerbaijan Grand Prix", "circuitId": "baku", "circuitName": "Baku City Circuit", "date": "2026-09-26"},
    (2026, 16): {"raceName": "Singapore Grand Prix", "circuitId": "marina_bay", "circuitName": "Marina Bay Street Circuit", "date": "2026-10-11"},
    (2026, 17): {"raceName": "United States Grand Prix", "circuitId": "americas", "circuitName": "Circuit of the Americas", "date": "2026-10-25"},
    (2026, 18): {"raceName": "Mexico City Grand Prix", "circuitId": "rodriguez", "circuitName": "Autodromo Hermanos Rodriguez", "date": "2026-11-01"},
    (2026, 19): {"raceName": "Sao Paulo Grand Prix", "circuitId": "interlagos", "circuitName": "Autodromo Jose Carlos Pace", "date": "2026-11-08"},
    (2026, 20): {"raceName": "Las Vegas Grand Prix", "circuitId": "las_vegas", "circuitName": "Las Vegas Strip Circuit", "date": "2026-11-21"},
    (2026, 21): {"raceName": "Qatar Grand Prix", "circuitId": "losail", "circuitName": "Lusail International Circuit", "date": "2026-11-29"},
    (2026, 22): {"raceName": "Abu Dhabi Grand Prix", "circuitId": "yas_marina", "circuitName": "Yas Marina Circuit", "date": "2026-12-06"},
}

class LiveBuilder:
    """Builds a feature matrix for an emerging/upcoming race weekend."""
    def __init__(self, historical_parquet: str | Path):
        self.hist_df = pd.read_parquet(historical_parquet)
        cache_root = Path(os.environ.get("F1_CACHE_DIR", "/tmp/f1outcome_cache"))
        self.client = JolpicaClient(
            base_url=SETTINGS.jolpica_base,
            cache_dir=cache_root / "jolpica",
            min_interval_s=0.2
        )

    @staticmethod
    def _grid_bucket(grid):
        if pd.isna(grid) or grid <= 0:
            return 4
        if grid <= 3:
            return 1
        if grid <= 10:
            return 2
        if grid <= 20:
            return 3
        return 4

    @staticmethod
    def _is_street_circuit(circuit_id: str | None) -> int:
        street_circuits = {
            "monaco", "marina_bay", "baku", "jeddah", "miami", "las_vegas",
            "albert_park", "villeneuve", "sochi", "valencia", "adler", "detroit", "dallas"
        }
        return 1 if str(circuit_id).lower() in street_circuits else 0

    @staticmethod
    def _safety_car_probability(circuit_id: str | None) -> float:
        sc_heavy_circuits = {"monaco", "marina_bay", "baku", "jeddah", "albert_park", "miami", "las_vegas", "zandvoort"}
        sc_med_circuits = {"interlagos", "montreal", "suzuka", "spa", "imola", "nurburgring"}
        circuit = str(circuit_id).lower()
        if circuit in sc_heavy_circuits:
            return 0.85
        if circuit in sc_med_circuits:
            return 0.50
        return 0.20

    @staticmethod
    def _add_teammate_delta(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["teammateFormFinishDelta"] = 0.0

        for _, group_idx in df.groupby(["season", "round", "constructorId"], dropna=False).groups.items():
            group = df.loc[group_idx]
            for idx, row in group.iterrows():
                teammates = group[group["driverId"] != row["driverId"]]
                if not teammates.empty:
                    teammate_form = teammates["driverFormAvgFinish"].mean()
                    df.at[idx, "teammateFormFinishDelta"] = row["driverFormAvgFinish"] - teammate_form

        return df

    def fetch_race_metadata(self, season: int, rnd: int) -> dict:
        """Fetch scheduled race metadata. Works before results or qualifying exist."""
        fallback = FALLBACK_RACE_METADATA.get((season, rnd))
        try:
            js = self.client.get_json(f"{season}/{rnd}.json")
        except Exception as exc:
            if fallback:
                return {**fallback, "source": "fallback", "error": str(exc)}
            raise

        races = js["MRData"]["RaceTable"]["Races"]
        if not races:
            return {**fallback, "source": "fallback"} if fallback else {
                "raceName": None,
                "circuitId": None,
                "circuitName": None,
                "date": None,
                "source": "empty",
            }

        race = races[0]
        circuit = race.get("Circuit") or {}
        return {
            "raceName": race.get("raceName"),
            "circuitId": circuit.get("circuitId"),
            "circuitName": circuit.get("circuitName"),
            "date": race.get("date"),
            "source": "jolpica",
        }
        
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
            
        df["grid_bucket"] = df["grid"].apply(self._grid_bucket)
        
        # Street circuit
        df["is_street_circuit"] = df["circuitId"].apply(self._is_street_circuit)
        df["track_sc_prob"] = df["circuitId"].apply(self._safety_car_probability)
        # Live weather from FastF1 (falls back to 0 for future/unavailable races)
        try:
            wet_flag = get_weather_flag(season, rnd, FastF1FeatureConfig(cache_dir=Path(os.environ.get("F1_CACHE_DIR", "/tmp/f1outcome_cache")) / "fastf1"))
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

    def build_prequalifying_forecast(self, season: int, rnd: int) -> tuple[pd.DataFrame, dict, list, str]:
        """Build a forecast before qualifying exists.

        This intentionally does not invent qualifying times. It uses the latest
        completed race roster, rolling form, reliability, neutral grid inputs,
        and scheduled track context so the model can produce a next-race view
        before the weekend sessions are available.
        """
        metadata = self.fetch_race_metadata(season, rnd)
        if metadata.get("source") == "empty" and not metadata.get("raceName"):
            raise ValueError(f"No scheduled race found for {season} Round {rnd}.")

        warnings_list = [
            "Pre-qualifying forecast: qualifying is not available yet, so grid and qualifying features are neutral.",
        ]
        sources = {
            "ergast": metadata.get("source") == "jolpica",
            "fastf1": False,
            "dataset_form": True,
            "pre_qualifying_forecast": True,
            "schedule_fallback": metadata.get("source") == "fallback",
        }
        if metadata.get("source") == "fallback":
            warnings_list.append("Race schedule metadata came from the built-in calendar fallback.")

        prior = self.hist_df[
            (self.hist_df["season"] < season) |
            ((self.hist_df["season"] == season) & (self.hist_df["round"] < rnd))
        ].copy()
        if prior.empty:
            raise ValueError(f"No historical races found prior to {season} Round {rnd} for forecast.")

        latest_keys = prior[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).iloc[-1]
        latest_season = int(latest_keys["season"])
        latest_round = int(latest_keys["round"])
        latest_race_id = f"{latest_season}_{latest_round}"
        roster = prior[(prior["season"] == latest_season) & (prior["round"] == latest_round)].copy()
        roster = roster.drop_duplicates(subset=["driverId"], keep="last")
        if roster.empty:
            raise ValueError(f"No provisional roster available from {latest_race_id}.")

        warnings_list.append(f"Provisional entry list copied from latest completed race {latest_race_id}.")
        if len(roster) < 18:
            warnings_list.append(f"Provisional roster has only {len(roster)} drivers.")

        neutral_grid = 11
        circuit_id = metadata.get("circuitId")
        if circuit_id is None and "circuitId" in roster.columns and not roster["circuitId"].dropna().empty:
            circuit_id = roster["circuitId"].dropna().iloc[-1]

        base_cols = [
            "driverId", "givenName", "familyName", "constructorId", "constructorName", "fullName_norm"
        ]
        for col in base_cols:
            if col not in roster.columns:
                roster[col] = None

        df = roster[base_cols].copy()
        df["season"] = season
        df["round"] = rnd
        df["raceName"] = metadata.get("raceName")
        df["raceId"] = f"{season}_{rnd}"
        df["circuitId"] = circuit_id
        df["grid"] = neutral_grid
        df["qualiPos"] = neutral_grid
        df["qualiBestMs"] = np.nan
        df["qualiGapMs"] = np.nan
        df["qualiGapPct"] = np.nan
        df["grid_bucket"] = self._grid_bucket(neutral_grid)
        df["is_street_circuit"] = self._is_street_circuit(circuit_id)
        df["track_sc_prob"] = self._safety_car_probability(circuit_id)
        df["is_wet_race"] = 0
        df["trackId"] = 0

        missing_name = df["fullName_norm"].isna() | (df["fullName_norm"].astype(str) == "")
        df.loc[missing_name, "fullName_norm"] = (
            df.loc[missing_name, "givenName"].fillna("") + df.loc[missing_name, "familyName"].fillna("")
        ).apply(lambda s: "".join(ch.lower() for ch in str(s) if ch.isalnum()))

        driver_form, team_form, cutoff_id = self._get_last_known_form(season, rnd)
        df = df.merge(driver_form, on="driverId", how="left")
        df = df.merge(team_form, on="constructorId", how="left")
        df = self._add_teammate_delta(df)

        for c in ["q_s1_gap", "q_s2_gap", "q_s3_gap", "fp_longrun_gap"]:
            df[c] = float("nan")

        df.fillna({
            "driverFormAvgFinish": 15.0,
            "driverFormAvgPoints": 0.0,
            "teamFormAvgFinish": 15.0,
            "teamFormAvgPoints": 0.0,
            "driver_dnf_rate": 0.15,
            "driver_mech_dnf_rate": 0.05,
            "team_dnf_rate": 0.15,
            "team_mech_dnf_rate": 0.05,
            "teammateFormFinishDelta": 0.0,
        }, inplace=True)

        return df, sources, warnings_list, cutoff_id

    def build_upcoming_race(self, season: int, rnd: int, allow_prequalifying: bool = True) -> tuple[pd.DataFrame, dict, list, str]:
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
            if not allow_prequalifying:
                raise ValueError(f"Ergast Qualifying required to predict. Failed: {e}")
            df, sources, warnings_list, cutoff_id = self.build_prequalifying_forecast(season, rnd)
            warnings_list.insert(0, f"Qualifying fetch failed; using pre-qualifying forecast. Original error: {e}")
            return df, sources, warnings_list, cutoff_id
        
        # 2. Get historical rolling form
        driver_form, team_form, cutoff_id = self._get_last_known_form(season, rnd)
        
        df = df.merge(driver_form, on="driverId", how="left")
        df = df.merge(team_form, on="constructorId", how="left")
        
        # 3. Teammate Delta
        df = self._add_teammate_delta(df)
        
        # 4. Fetch Live FastF1
        cfg = FastF1FeatureConfig(cache_dir=Path(os.environ.get("F1_CACHE_DIR", "/tmp/f1outcome_cache")) / "fastf1")
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
