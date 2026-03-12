from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd


def _norm_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


@dataclass
class FastF1FeatureConfig:
    cache_dir: Path
    min_stint_laps: int = 5
    retries: int = 4
    base_sleep_s: float = 0.8
    load_timeout_s: float = 60.0
    failure_log_path: Optional[Path] = None



def _setup_fastf1(cfg: FastF1FeatureConfig) -> None:
    import fastf1
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cfg.cache_dir))
    fastf1.set_log_level("ERROR")  # reduce spam :contentReference[oaicite:4]{index=4}


import time
import concurrent.futures as cf


def _log_failure(cfg: FastF1FeatureConfig, msg: str) -> None:
    try:
        if cfg.failure_log_path is None:
            cfg.failure_log_path = Path("data/raw/fastf1_failures.log")
        cfg.failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.failure_log_path, "a", encoding="utf-8") as fp:
            fp.write(msg.rstrip() + "\n")
    except Exception:
        pass

def _load_session(year: int, rnd: int, code: str, cfg):
    import fastf1

    last_err = None
    for attempt in range(cfg.retries):
        try:
            sess = fastf1.get_session(year, rnd, code)

            def _do_load():
                # only what we need
                sess.load(laps=True, telemetry=False, weather=False, messages=False)

            with cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_load)
                fut.result(timeout=cfg.load_timeout_s)  # <-- hard timeout

            if sess.laps is None or sess.laps.empty:
                raise RuntimeError("Session loaded but laps is empty")

            time.sleep(cfg.base_sleep_s)  # small throttle
            return sess

        except cf.TimeoutError as e:
            last_err = e
            # backoff then retry
            time.sleep(cfg.base_sleep_s * (2 ** attempt))
            continue

        except Exception as e:
            last_err = e
            time.sleep(cfg.base_sleep_s * (2 ** attempt))
            continue

    if last_err is not None:
        _log_failure(
            cfg,
            f"{year},{rnd},{code} :: {type(last_err).__name__}: {last_err}",
        )
    else:
        _log_failure(cfg, f"{year},{rnd},{code} :: UNKNOWN: session load failed")
    return None



def _quali_sector_gaps(year: int, rnd: int, cfg: FastF1FeatureConfig) -> pd.DataFrame:
    sess = _load_session(year, rnd, "Q", cfg)
    if sess is None:
        _log_failure(cfg, f"{year},{rnd},Q :: EMPTY: session=None")
        return pd.DataFrame()

    laps = sess.laps
    if laps is None or laps.empty:
        _log_failure(cfg, f"{year},{rnd},Q :: EMPTY: laps empty")
        return pd.DataFrame()

    rows = []
    for drv in laps["Driver"].dropna().unique():
        dlaps = laps.pick_drivers([drv])

        if "IsAccurate" in dlaps.columns:
            dlaps = dlaps[dlaps["IsAccurate"] == True]
        if dlaps.empty:
            continue

        try:
            fl = dlaps.pick_fastest()
        except Exception:
            continue

        def sec(x):
            if pd.isna(x):
                return np.nan
            return x.total_seconds()

        s1 = sec(fl.get("Sector1Time"))
        s2 = sec(fl.get("Sector2Time"))
        s3 = sec(fl.get("Sector3Time"))

        # Use FullName from results if available
        full_name = None
        try:
            res = sess.results
            if res is not None and not res.empty and "Abbreviation" in res.columns and "FullName" in res.columns:
                m = res[res["Abbreviation"] == drv]
                if not m.empty:
                    full_name = m["FullName"].iloc[0]
        except Exception:
            pass

        full_name = full_name or str(drv)
        rows.append({"fullName_norm": _norm_name(full_name), "q_s1": s1, "q_s2": s2, "q_s3": s3})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["q_s1", "q_s2", "q_s3"]:
        best = df[c].min(skipna=True)
        df[c + "_gap"] = df[c] - best

    return df[["fullName_norm", "q_s1_gap", "q_s2_gap", "q_s3_gap"]]


def _practice_longrun_gap(year: int, rnd: int, cfg: FastF1FeatureConfig) -> pd.DataFrame:
    """
    Try FP2 long-run proxy; fallback to FP1 if FP2 unavailable.
    """
    sess = _load_session(year, rnd, "FP2", cfg)
    if sess is None:
        sess = _load_session(year, rnd, "FP1", cfg)
    if sess is None:
        _log_failure(cfg, f"{year},{rnd},FP2/FP1 :: EMPTY: session=None")
        return pd.DataFrame()

    laps = sess.laps
    if laps is None or laps.empty or "LapTime" not in laps.columns:
        _log_failure(cfg, f"{year},{rnd},FP2/FP1 :: EMPTY: laps empty or LapTime missing")
        return pd.DataFrame()

    laps = laps.loc[laps["LapTime"].notna()].copy()

    # Exclude pit in/out laps when possible
    for col in ["PitInTime", "PitOutTime"]:
        if col in laps.columns:
            laps = laps.loc[laps[col].isna()].copy()

    def to_sec(td):
        if pd.isna(td):
            return np.nan
        return td.total_seconds()

    laps.loc[:, "lap_s"] = laps["LapTime"].apply(to_sec)
    laps = laps.loc[laps["lap_s"].notna()].copy()

    rows = []
    for drv in laps["Driver"].dropna().unique():
        dlaps = laps.pick_drivers([drv])
        if "IsAccurate" in dlaps.columns:
            dlaps = dlaps[dlaps["IsAccurate"] == True]
        if dlaps.empty:
            continue

        best_longrun = np.nan

        if "Stint" in dlaps.columns and dlaps["Stint"].notna().any():
            for _, stint_df in dlaps.groupby("Stint"):
                if len(stint_df) < cfg.min_stint_laps:
                    continue
                med = float(np.nanmedian(stint_df["lap_s"].values))
                if np.isnan(best_longrun) or med < best_longrun:
                    best_longrun = med
        else:
            arr = dlaps["lap_s"].values
            if len(arr) >= cfg.min_stint_laps:
                for i in range(0, len(arr) - cfg.min_stint_laps + 1):
                    med = float(np.nanmedian(arr[i:i+cfg.min_stint_laps]))
                    if np.isnan(best_longrun) or med < best_longrun:
                        best_longrun = med

        if np.isnan(best_longrun):
            continue

        full_name = None
        try:
            res = sess.results
            if res is not None and not res.empty and "Abbreviation" in res.columns and "FullName" in res.columns:
                m = res[res["Abbreviation"] == drv]
                if not m.empty:
                    full_name = m["FullName"].iloc[0]
        except Exception:
            pass

        full_name = full_name or str(drv)
        rows.append({"fullName_norm": _norm_name(full_name), "fp_longrun_s": best_longrun})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    best = df["fp_longrun_s"].min(skipna=True)
    df["fp_longrun_gap"] = df["fp_longrun_s"] - best
    return df[["fullName_norm", "fp_longrun_gap"]]


def fastf1_driver_features(season: int, rnd: int, cfg: FastF1FeatureConfig) -> pd.DataFrame:
    """
    Per-driver features keyed by fullName_norm for a season+round.
    """
    try:
        _setup_fastf1(cfg)

        if season < 2018:
            return pd.DataFrame()

        q = _quali_sector_gaps(season, rnd, cfg)
        fp = _practice_longrun_gap(season, rnd, cfg)

        if q.empty and fp.empty:
            _log_failure(cfg, f"{season},{rnd},ALL :: EMPTY: q and fp empty")
            return pd.DataFrame()

        if not q.empty and not fp.empty:
            return q.merge(fp, on="fullName_norm", how="outer")
        return q if not q.empty else fp

    except Exception as e:
        _log_failure(cfg, f"{season},{rnd},ALL :: EXC: {type(e).__name__}: {e}")
        # Never crash the dataset build
        return pd.DataFrame()


def get_weather_flag(season: int, rnd: int, cfg: FastF1FeatureConfig) -> int:
    """Return 1 if the race had rainfall, 0 otherwise. Falls back to 0 on any error."""
    try:
        import fastf1
        _setup_fastf1(cfg)
        sess = fastf1.get_session(season, rnd, "R")
        sess.load(weather=True, laps=False, telemetry=False, messages=False)
        
        wd = sess.weather_data
        if wd is not None and "Rainfall" in wd.columns:
            if wd["Rainfall"].any():
                return 1
        return 0
    except Exception:
        return 0
