import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
from f1outcome.api.app import app

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
