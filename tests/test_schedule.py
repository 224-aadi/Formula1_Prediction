from datetime import datetime, timezone
import json
import os
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from f1outcome.api import app as app_module
from f1outcome.data import schedule
from f1outcome.data.jolpica import JolpicaClient
from f1outcome.data.live_builder import LiveBuilder


def race(rnd, date, time, name):
    return {"round": str(rnd), "date": date, "time": time, "raceName": name,
            "Circuit": {"circuitId": "sepang", "circuitName": "Sepang International Circuit"}}


@pytest.fixture
def calendar(monkeypatch):
    payload = {"MRData": {"RaceTable": {"Races": [
        race(15, "2026-09-26", "11:00:00Z", "Azerbaijan Grand Prix"),
        race(16, "2026-10-04", "07:00:00Z", "Bahrain Grand Prix in Malaysia"),
        race(17, "2026-10-11", "12:00:00Z", "Singapore Grand Prix"),
    ]}}}
    monkeypatch.setattr(JolpicaClient, "get_json", lambda self, path, params=None: payload)
    return schedule.get_schedule(2026)


@pytest.mark.parametrize("now, expected", [
    ("2026-09-26T10:59:59+00:00", 15),
    ("2026-09-26T11:00:00+00:00", 16),
    ("2026-10-03T23:00:00+00:00", 16),
    ("2026-10-04T06:59:59+00:00", 16),
    ("2026-10-04T07:00:00+00:00", 17),
    ("2026-12-31T12:00:00+00:00", None),
])
def test_next_race_uses_utc_start_time(calendar, now, expected):
    result = schedule.next_race(calendar, datetime.fromisoformat(now))
    assert (result["round"] if result else None) == expected


def test_default_season_uses_clock_not_dataset(monkeypatch, calendar):
    monkeypatch.setattr(app_module, "utc_now", lambda: datetime(2026, 10, 3, tzinfo=timezone.utc))
    def get_calendar(season):
        assert season == 2026
        return calendar
    monkeypatch.setattr(app_module, "get_schedule", get_calendar)
    monkeypatch.setattr(app_module, "get_dataset", Mock(side_effect=AssertionError("Must not select from dataset")))
    monkeypatch.setattr(app_module, "predict_live", lambda **kwargs: {
        "raceId": f"{kwargs['season']}_{kwargs['round']}", "order": [], "alpha": 5,
        "p_dnf_cap": .3, "mode": kwargs["mode"], "prediction_type": "live_qualifying",
        "sources": {}, "warnings": [], "form_cutoff_raceId": "2026_15",
    })
    response = TestClient(app_module.app).get("/predict/next")
    assert response.status_code == 200
    assert response.json()["raceId"] == "2026_16"
    assert response.json()["warnings"] == []


def test_no_next_race_after_season_end(monkeypatch, calendar):
    monkeypatch.setattr(app_module, "get_schedule", lambda season: calendar)
    monkeypatch.setattr(app_module, "utc_now", lambda: datetime(2026, 12, 31, tzinfo=timezone.utc))
    response = TestClient(app_module.app).get("/predict/next?season=2026")
    assert response.status_code == 404
    assert "No upcoming race" in response.json()["detail"]


def test_calendar_outage_does_not_predict_an_old_race(monkeypatch):
    monkeypatch.setattr(app_module, "get_schedule", Mock(side_effect=RuntimeError("Provider unavailable")))
    client = TestClient(app_module.app)
    assert client.get("/schedule?season=2026").status_code == 503
    assert client.get("/predict/next?season=2026").status_code == 503


def test_schedule_endpoint_includes_updated_venue(monkeypatch, calendar):
    monkeypatch.setattr(app_module, "get_schedule", lambda season: calendar)
    response = TestClient(app_module.app).get("/schedule?season=2026")
    assert response.status_code == 200
    assert response.json()[1]["circuitName"] == "Sepang International Circuit"
    assert response.json()[1]["raceName"] == "Bahrain Grand Prix in Malaysia"


def test_schedule_accepts_historical_seasons_shown_in_ui(monkeypatch):
    monkeypatch.setattr(app_module, "get_schedule", lambda season: [])
    assert TestClient(app_module.app).get("/schedule?season=2019").status_code == 200


def test_expired_empty_qualifying_cache_is_refetched(tmp_path):
    client = JolpicaClient("https://example.test", tmp_path, cache_ttl_s=300)
    cached = tmp_path / "2026_16_qualifying.json"
    cached.write_text(json.dumps({"old": True}))
    os.utime(cached, (1, 1))
    response = Mock(status_code=200)
    response.json.return_value = {"fresh": True}
    client.session.get = Mock(return_value=response)
    assert client.get_json("2026/16/qualifying.json") == {"fresh": True}
    assert client.get_json("2026/16/qualifying.json") == {"fresh": True}
    client.session.get.assert_called_once()


def test_qualifying_keeps_race_name_and_22_driver_grid(monkeypatch):
    from f1outcome.data import live_builder
    builder = LiveBuilder.__new__(LiveBuilder)
    metadata = race(16, "2026-10-04", "07:00:00Z", "Bahrain Grand Prix in Malaysia")
    metadata["QualifyingResults"] = [
        {"Driver": {"driverId": f"driver_{i}", "givenName": "Driver", "familyName": str(i)},
         "Constructor": {"constructorId": f"team_{i // 2}"}, "position": str(i), "Q1": "1:30.000"}
        for i in range(1, 23)
    ]
    builder.client = Mock()
    builder.client.get_json.return_value = {"MRData": {"RaceTable": {"Races": [metadata]}}}
    monkeypatch.setattr(live_builder, "get_weather_flag", lambda *args: 0)
    frame = builder.fetch_live_qualifying(2026, 16)
    assert len(frame) == 22
    assert frame["raceName"].unique().tolist() == ["Bahrain Grand Prix in Malaysia"]
