"""LLM backend wrapper: ``generate(prompt) -> str``.

The only implemented backend is a local Ollama server (``LLM_BACKEND=ollama``),
reached over its REST API. Callers depend only on :func:`generate`, so adding
another backend later (see the ADR in ``CLAUDE.md``) is a change in this file.
"""

from __future__ import annotations

import json
import urllib.request

from src.config import Config, get_config

OLLAMA_BACKEND = "ollama"


class LLMError(RuntimeError):
    """The LLM backend is unreachable, timed out, or returned something unusable."""


def _generate_ollama(prompt: str, config: Config) -> str:
    llm = config.data.get("llm", {})
    body = json.dumps(
        {
            "model": config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": llm.get("temperature", 0.1)},
        }
    ).encode()
    request = urllib.request.Request(
        f"{config.ollama_host.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=llm.get("timeout_seconds", 120)
        ) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as exc:
        # OSError covers URLError, HTTP errors, refused connections and timeouts.
        raise LLMError(
            f"Ollama request to {config.ollama_host} failed "
            f"(is `ollama serve` running and `{config.ollama_model}` pulled?): {exc}"
        ) from exc

    answer = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise LLMError(f"Ollama returned no text: {payload!r}")
    return answer.strip()


def generate(prompt: str, config: Config | None = None) -> str:
    """Send ``prompt`` to the configured LLM backend and return its text answer.

    Raises :class:`LLMError` if the backend cannot be reached or returns no
    text, and ``ValueError`` if ``LLM_BACKEND`` names an unimplemented backend.
    """
    config = config or get_config()
    if config.llm_backend != OLLAMA_BACKEND:
        raise ValueError(
            f"Unsupported LLM_BACKEND {config.llm_backend!r}; "
            f"only {OLLAMA_BACKEND!r} is implemented."
        )
    return _generate_ollama(prompt, config)
