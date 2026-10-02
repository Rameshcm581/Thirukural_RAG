import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app
from backend.resources.explain import generate_explanation, _fallback_explanation
from backend.configs import BaseConfig


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ThirukuralService"
    assert data["status"] == "running"


def test_health_endpoint(client):
    response = client.get("/health")
    # Status code will be 200 or 503 depending on whether local Qdrant is currently running
    assert response.status_code in (200, 503)
    data = response.json()
    assert data["service"] == "ThirukuralService"
    assert "status" in data
    assert "embedding_model" in data
    assert "qdrant" in data


def test_search_empty_scenario(client):
    response = client.post("/search", json={"scenario": ""})
    # Fails Pydantic validation (min_length=1) with 422
    assert response.status_code == 422


def test_search_whitespace_scenario(client):
    response = client.post("/search", json={"scenario": "   "})
    # Handled by controller whitespace strip check with 400
    assert response.status_code == 400
    assert "Scenario cannot be empty" in response.json()["detail"]


def test_search_oversized_scenario(client):
    response = client.post("/search", json={"scenario": "a" * 1001})
    assert response.status_code == 422


def test_generate_explanation_fallback():
    dummy_kural = {
        "kural_number": 999,
        "english_couplet": "Dummy couplet",
        "english_explanation": "Dummy explanation",
    }
    with patch("backend.resources.explain._get_client", return_value=None):
        result = generate_explanation("I am stressed", dummy_kural)
        assert result == _fallback_explanation(dummy_kural)
        assert "Dummy explanation" in result


def test_search_mocked(client):
    sample_raw = [
        {
            "kural_number": 1,
            "book": "Virtue",
            "adikaram": "The Praise of God",
            "tamil_couplet": ["அகர முதல", "எழுத்தெல்லாம்"],
            "transliteration": "Akara Mudhala",
            "english_couplet": "A is starting letter",
            "english_explanation": "God is first",
            "matched_tag": "beginning of life",
            "similarity_score": 0.88,
        }
    ]
    with patch("resources.searchController.retrieve_kurals", return_value=sample_raw), \
         patch("resources.searchController.generate_explanation", return_value="Wise teaching"):
        response = client.post("/search", json={"scenario": "starting a new chapter in life"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["kural_number"] == 1
        assert data["results"][0]["explanation"] == "Wise teaching"
        assert data["results"][0]["matched_tag"] == "beginning of life"
