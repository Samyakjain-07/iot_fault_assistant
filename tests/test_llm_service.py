"""Tests for the Ollama wrapper. No real server is contacted."""

from __future__ import annotations

import dataclasses
import io
import json
import urllib.error

import pytest

from src import llm_service
from src.config import get_config
from src.llm_service import LLMError, generate


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Replace urlopen; ``calls`` records (request, timeout) per call."""
    state = {"calls": [], "body": b'{"response": "  It is the bearing.  "}'}

    def urlopen(request, timeout=None):
        state["calls"].append((request, timeout))
        if isinstance(state["body"], Exception):
            raise state["body"]
        return _FakeResponse(state["body"])

    monkeypatch.setattr(llm_service.urllib.request, "urlopen", urlopen)
    return state


def test_generate_posts_prompt_to_ollama_and_returns_stripped_text(fake_urlopen):
    config = get_config()

    assert generate("why is it hot?") == "It is the bearing."

    request, timeout = fake_urlopen["calls"][0]
    assert request.full_url == f"{config.ollama_host.rstrip('/')}/api/generate"
    assert timeout == config.data["llm"]["timeout_seconds"]
    body = json.loads(request.data)
    assert body["model"] == config.ollama_model
    assert body["prompt"] == "why is it hot?"
    assert body["stream"] is False


def test_unreachable_server_raises_llm_error(fake_urlopen):
    fake_urlopen["body"] = urllib.error.URLError("connection refused")

    with pytest.raises(LLMError, match="ollama serve"):
        generate("hi")


def test_timeout_raises_llm_error(fake_urlopen):
    fake_urlopen["body"] = TimeoutError("timed out")

    with pytest.raises(LLMError):
        generate("hi")


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[]", b"{}", b'{"response": ""}', b'{"response": "   "}'],
)
def test_unusable_response_raises_llm_error(fake_urlopen, body):
    fake_urlopen["body"] = body

    with pytest.raises(LLMError):
        generate("hi")


def test_unsupported_backend_rejected(fake_urlopen):
    config = dataclasses.replace(get_config(), llm_backend="bedrock")

    with pytest.raises(ValueError, match="bedrock"):
        generate("hi", config)
    assert fake_urlopen["calls"] == []
