import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
from f1outcome.api.app import app
from f1outcome.data.live_builder import LiveBuilder

client = TestClient(app)

def test_meta_endpoint():
    """Verify the API logic is functioning and environmental paths bind correctly."""
    response = client.get("/meta")
    assert response.status_code == 200
    data = response.json()
    assert "ranker_path" in data
    assert "dataset" in data

@pytest.mark.skipif(
    not Path("artifacts/final/ranker.joblib").exists() or not Path("data/processed/final_hybrid_dataset.parquet").exists(), 
    reason="Frozen Models or Dataset Parquet not found locally. Skipping offline parity check in CI."
)
def test_offline_parity_baseline():
    """
    Test a known historical race to ensure model ranking parity without making
    any external API calls to FastF1 or Ergast (strict offline check).
    """
    # Test 2024 Round 1 directly from locally cached parquet
    response = client.get("/predict/from_parquet?season=2024&round=1")
    assert response.status_code == 200
    data = response.json()
    
    drivers = [d["driverId"] for d in data["order"]]
    
    # Assert sanity of returned output
    assert len(drivers) == 20
    assert drivers[0] == "max_verstappen", "Model parity failed! Max should be predicted P1 for Bahrain 2024 offline baseline."


@pytest.mark.skipif(
    not Path("artifacts/final/ranker.joblib").exists() or not Path("data/processed/final_hybrid_dataset.parquet").exists(),
    reason="Frozen models or dataset parquet not found locally. Skipping live forecast check in CI.",
)
def test_live_endpoint_falls_back_to_prequalifying_forecast(monkeypatch):
    def no_qualifying(self, season, rnd):
        raise ValueError("No qualifying data found")

    def belgian_metadata(self, season, rnd):
        return {
            "raceName": "Belgian Grand Prix",
            "circuitId": "spa",
            "circuitName": "Circuit de Spa-Francorchamps",
            "date": "2026-07-19",
        }

    monkeypatch.setattr(LiveBuilder, "fetch_live_qualifying", no_qualifying)
    monkeypatch.setattr(LiveBuilder, "fetch_race_metadata", belgian_metadata)

    response = client.get("/predict/live?season=2026&round=10")
    assert response.status_code == 200
    data = response.json()

    assert data["raceId"] == "2026_10"
    assert data["raceName"] == "Belgian Grand Prix"
    assert data["prediction_type"] == "pre_qualifying_forecast"
    assert data["sources"]["dataset_form"] is True
    assert data["sources"]["pre_qualifying_forecast"] is True
    assert data["form_cutoff_raceId"] == "2026_9"
    assert len(data["order"]) >= 20
    assert any("Pre-qualifying forecast" in warning for warning in data["warnings"])


@pytest.mark.skipif(
    not Path("artifacts/final/ranker.joblib").exists() or not Path("data/processed/final_hybrid_dataset.parquet").exists(),
    reason="Frozen models or dataset parquet not found locally. Skipping next race check in CI.",
)
def test_predict_next_targets_upcoming_round(monkeypatch):
    def no_qualifying(self, season, rnd):
        raise ValueError("No qualifying data found")

    def belgian_metadata(self, season, rnd):
        assert (season, rnd) == (2026, 10)
        return {
            "raceName": "Belgian Grand Prix",
            "circuitId": "spa",
            "circuitName": "Circuit de Spa-Francorchamps",
            "date": "2026-07-19",
            "source": "jolpica",
        }

    monkeypatch.setattr(LiveBuilder, "fetch_live_qualifying", no_qualifying)
    monkeypatch.setattr(LiveBuilder, "fetch_race_metadata", belgian_metadata)

    response = client.get("/predict/next?season=2026")
    assert response.status_code == 200
    data = response.json()

    assert data["raceId"] == "2026_10"
    assert data["raceName"] == "Belgian Grand Prix"
    assert data["prediction_type"] == "pre_qualifying_forecast"
    assert data["sources"]["ergast"] is True
    assert data["form_cutoff_raceId"] == "2026_9"


@pytest.mark.skipif(
    not Path("artifacts/final/ranker.joblib").exists() or not Path("data/processed/final_hybrid_dataset.parquet").exists(),
    reason="Frozen models or dataset parquet not found locally. Skipping strict live check in CI.",
)
def test_live_endpoint_can_still_require_qualifying(monkeypatch):
    def no_qualifying(self, season, rnd):
        raise ValueError("No qualifying data found")

    monkeypatch.setattr(LiveBuilder, "fetch_live_qualifying", no_qualifying)

    response = client.get("/predict/live?season=2026&round=10&allow_prequalifying=false")
    assert response.status_code == 400
    assert "Ergast Qualifying required" in response.json()["detail"]
