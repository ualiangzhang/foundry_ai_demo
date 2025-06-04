# tests/test_chains_basic.py
"""
Unit-test the public build_chain API in src.rag.chains without
requiring external services.  Heavy network/model calls are stubbed.
"""

from typing import Dict, Any
import importlib
import types

import pytest
from src.rag import chains as chains_mod


# ------------------------------------------------------------------ #
# Stubs for SerpApi and OpenAI                                       #
# ------------------------------------------------------------------ #
class _StubSerpApi:
    @staticmethod
    def search(_params):
        # Minimal payload resembling SerpApi's structure
        return {
            "organic_results": [
                {"snippet": "stub snippet one " * 5},
                {"snippet": "stub snippet two " * 5},
                {"snippet": "stub snippet three " * 5},
            ]
        }


class _StubChatCompletions:
    @staticmethod
    def create(*_args, **_kwargs):
        stub = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="stub answer"))]
        )
        return stub


class _StubOpenAI:
    api_key = "test"
    chat = types.SimpleNamespace(completions=_StubChatCompletions)


# ------------------------------------------------------------------ #
# Test build_chain(kind="qa")                                         #
# ------------------------------------------------------------------ #
@pytest.fixture(autouse=True)
def patch_external(monkeypatch):
    # Patch serpapi module
    monkeypatch.setitem(sys.modules, "serpapi", _StubSerpApi)
    # Patch openai module
    monkeypatch.setitem(sys.modules, "openai", _StubOpenAI)
    # Reload the chains module so it picks up our stubs
    importlib.reload(chains_mod)
    yield
    # No explicit teardown needed; Pytest will discard monkeypatch


def test_qa_chain_output():
    qa_chain = chains_mod.build_chain(kind="qa")
    out: Dict[str, Any] = qa_chain({"question": "What is AI?"})
    assert "answer" in out and "context" in out
    # context should contain our stubbed snippets joined by double-newline
    assert "stub snippet one" in out["context"]
    assert len(out["answer"].split()) <= 200
