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


def test_chat_lifecycle_with_postgres(client):
    test_session = "test-session-12345"
    # Send first user message
    with patch("resources.chat.retrieve_kurals", return_value=[]):
        resp = client.post("/chat", json={"session_id": test_session, "message": "How do I deal with anger?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == test_session
        assert "reply" in data
        assert data["mode"] in ("answered", "clarifying_question", "multiple_options")

    # Verify session appears in session list
    resp_list = client.get("/chat/sessions")
    assert resp_list.status_code == 200
    sessions = resp_list.json()["sessions"]
    matching = [s for s in sessions if s["session_id"] == test_session]
    assert len(matching) == 1
    assert matching[0]["message_count"] >= 2  # user msg + assistant msg

    # Verify session history
    resp_hist = client.get(f"/chat/sessions/{test_session}")
    assert resp_hist.status_code == 200
    messages = resp_hist.json()["messages"]
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "How do I deal with anger?"
    assert messages[1]["role"] == "assistant"

    # Delete session
    resp_del = client.delete(f"/chat/sessions/{test_session}")
    assert resp_del.status_code == 200
    assert resp_del.json()["deleted"] is True
