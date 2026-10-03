"""One live calendar for race selection and presentation, independent of training data."""
from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile

from f1outcome.config import SETTINGS
from f1outcome.data.jolpica import JolpicaClient


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_schedule(season: int) -> list[dict]:
    cache_root = Path(os.environ.get("F1_CACHE_DIR", str(Path(tempfile.gettempdir()) / "f1outcome_cache")))
    client = JolpicaClient(
        SETTINGS.jolpica_base, cache_root / "schedule", cache_ttl_s=300, max_retries=1,
    )
    payload = client.get_json(f"{season}.json", params={"limit": 100})
    races = []
    for race in payload["MRData"]["RaceTable"]["Races"]:
        circuit = race.get("Circuit") or {}
        # Unknown start times are treated conservatively: never select a race
        # from an earlier date or silently invent a start time on race day.
        date = race["date"]
        start = datetime.fromisoformat(f"{date}T{race.get('time') or '00:00:00Z'}".replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        rnd = int(race["round"])
        races.append({
            "season": season, "round": rnd, "raceId": f"{season}_{rnd}",
            "raceName": race["raceName"], "date": date, "startTime": start.isoformat(),
            "circuitId": circuit.get("circuitId"), "circuitName": circuit.get("circuitName"),
            "country": (circuit.get("Location") or {}).get("country"),
        })
    return sorted(races, key=lambda race: race["startTime"])


def next_race(races: list[dict], now: datetime) -> dict | None:
    return next((race for race in races if datetime.fromisoformat(race["startTime"]) > now), None)
