# tests/test_chains_basic.py
"""
Unit-test the public build_chain API in src/rag/chains without
requiring external services. Heavy network/model calls are stubbed.
"""

import importlib
import sys
import types
from typing import Any, Dict

import pytest
from src.rag import chains as chains_mod


# ------------------------------------------------------------------ #
# Stubs for SerpApi and OpenAI                                       #
# ------------------------------------------------------------------ #
class _StubSerpApi:
    """Stub implementation of SerpApi.search to return fixed snippets."""

    @staticmethod
    def search(_params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a minimal payload resembling SerpApi's structure.

        Args:
            _params: The parameters for the SerpApi search (ignored).

        Returns:
            A dict with an 'organic_results' list containing three stub snippets.
        """
        return {
            "organic_results": [
                {"snippet": "stub snippet one " * 5},
                {"snippet": "stub snippet two " * 5},
                {"snippet": "stub snippet three " * 5},
            ]
        }


class _StubChatCompletions:
    """Stub for OpenAI ChatCompletion.create() returning a fixed answer."""

    @staticmethod
    def create(*_args: Any, **_kwargs: Any) -> types.SimpleNamespace:
        """Return a SimpleNamespace mimicking a ChatCompletion with choices.

        Args:
            *_args: Positional arguments passed to the create method (ignored).
            **_kwargs: Keyword arguments passed to the create method (ignored).

        Returns:
            A SimpleNamespace with a 'choices' list containing one message with 'content'.
        """
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="stub answer")
            )]
        )


class _StubOpenAI:
    """Stub for the openai module exposing a 'chat.completions' interface."""

    api_key: str = "test"
    chat = types.SimpleNamespace(completions=_StubChatCompletions)


# ------------------------------------------------------------------ #
# Test build_chain(kind="qa")                                         #
# ------------------------------------------------------------------ #
@pytest.fixture(autouse=True)
def patch_external(monkeypatch: Any) -> None:
    """Automatically patch external dependencies before each test.

    This fixture replaces the 'serpapi' and 'openai' modules in sys.modules
    with stub implementations, and reloads the chains_mod so it picks up the stubs.

    Args:
        monkeypatch: The pytest monkeypatch fixture for modifying sys.modules.
    """
    # Patch serpapi module
    monkeypatch.setitem(sys.modules, "serpapi", _StubSerpApi)
    # Patch openai module
    monkeypatch.setitem(sys.modules, "openai", _StubOpenAI)
    # Reload the chains module so it picks up our stubs
    importlib.reload(chains_mod)
    yield
    # No explicit teardown needed; pytest will discard monkeypatch changes


def test_qa_chain_output() -> None:
    """Verify that build_chain(kind='qa') returns expected keys and content.

    This test constructs the QA chain, invokes it with a question,
    and asserts that the output contains both 'answer' and 'context'.
    It also checks that the context includes the stub snippet text.

    Raises:
        AssertionError: If the chain output does not meet expectations.
    """
    qa_chain = chains_mod.build_chain(kind="qa")
    out: Dict[str, Any] = qa_chain({"question": "What is AI?"})

    # Ensure both 'answer' and 'context' keys are present
    assert "answer" in out and "context" in out

    # The context should contain the stubbed snippets joined by double newlines
    assert "stub snippet one" in out["context"]

    # The answer should not exceed 250 words (stub will be short)
    assert len(out["answer"].split()) <= 250
