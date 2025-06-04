# tests/test_api.py
"""
Smoke-tests the FastAPI endpoints in api_server.py without hitting
LLM or SerpApi.  We monkey-patch the pre-built chains so the HTTP
layer can be exercised quickly and deterministically.
"""

from fastapi.testclient import TestClient
import api_server


# ------------------------------------------------------------------ #
# Helpers: stub chains                                                #
# ------------------------------------------------------------------ #
def _fake_eval_chain(inputs):
    return {
        "result": "stub-recommendations",
        "context": "stub-context",
        "snippet": "stub-snippet",
        "docs": ["doc-1", "doc-2", "doc-3"],
    }


def _fake_qa_chain(inputs):
    return {
        "answer": "stub answer no more than two hundred words.",
        "context": "stub snippet 1\n\nstub snippet 2\n\nstub snippet 3",
    }


# ------------------------------------------------------------------ #
# Tests                                                              #
# ------------------------------------------------------------------ #
def test_evaluate_endpoint(monkeypatch):
    """POST /evaluate returns JSON with expected keys."""
    monkeypatch.setattr(api_server, "EVAL_CHAIN", _fake_eval_chain)
    client = TestClient(api_server.app)

    resp = client.post("/evaluate", json={"summary": "Test startup idea"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["result"] == "stub-recommendations"
    assert data["context"] == "stub-context"
    assert data["snippet"] == "stub-snippet"
    assert len(data["docs"]) == 3


def test_qa_endpoint(monkeypatch):
    """POST /qa returns JSON with expected keys."""
    monkeypatch.setattr(api_server, "QA_CHAIN", _fake_qa_chain)
    client = TestClient(api_server.app)

    resp = client.post("/qa", json={"question": "What is CRISPR?"})
    assert resp.status_code == 200

    data = resp.json()
    assert data["answer"].startswith("stub answer")
    assert "stub snippet" in data["context"]
