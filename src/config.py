"""Central configuration loader.

Non-secret values come from ``config/project.yaml``; secrets come from ``.env``
(or the real process environment, which wins over ``.env``). Nothing else in
the project should read ``os.environ`` directly -- go through ``get_config()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML_PATH = PROJECT_ROOT / "config" / "project.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Config:
    """Merged view of ``project.yaml`` (``data``) and the ``.env`` secrets."""

    data: dict[str, Any]

    mqtt_broker_host: str
    mqtt_broker_port: int
    mqtt_username: str
    mqtt_password: str
    mqtt_use_tls: bool

    aws_region: str
    llm_backend: str
    ollama_model: str
    ollama_host: str
    log_level: str

    @property
    def db_path(self) -> Path:
        """SQLite file location, resolved against the project root."""
        return resolve_path(self.data["database"]["path"])

    @property
    def topics(self) -> dict[str, str]:
        """MQTT topic names keyed by purpose (telemetry, status, ...)."""
        return self.data["mqtt"]["topics"]


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path; relative paths are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(
    yaml_path: str | Path = DEFAULT_YAML_PATH,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> Config:
    """Read the YAML file and ``.env`` and return a :class:`Config`.

    A missing ``.env`` is fine (e.g. in CI); secrets then fall back to
    whatever is in the process environment, or to empty/default values.
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    load_dotenv(env_path, override=False)
    env = os.environ

    return Config(
        data=data,
        mqtt_broker_host=env.get("MQTT_BROKER_HOST", ""),
        mqtt_broker_port=int(env.get("MQTT_BROKER_PORT", "8883")),
        mqtt_username=env.get("MQTT_USERNAME", ""),
        mqtt_password=env.get("MQTT_PASSWORD", ""),
        mqtt_use_tls=_as_bool(env.get("MQTT_USE_TLS", "true")),
        aws_region=env.get("AWS_REGION", ""),
        llm_backend=env.get("LLM_BACKEND", "ollama"),
        ollama_model=env.get("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_host=env.get("OLLAMA_HOST", "http://localhost:11434"),
        log_level=env.get("LOG_LEVEL", "INFO"),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide default config (loaded once)."""
    return load_config()
