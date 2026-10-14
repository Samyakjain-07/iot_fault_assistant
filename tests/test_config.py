from pathlib import Path

from src.config import PROJECT_ROOT, load_config, resolve_path

YAML_TEXT = """
project:
  device_id: "TEST-001"
mqtt:
  topics:
    telemetry: "t/telemetry"
database:
  path: "database/test.db"
"""

ENV_KEYS = [
    "MQTT_BROKER_HOST",
    "MQTT_BROKER_PORT",
    "MQTT_USERNAME",
    "MQTT_PASSWORD",
    "MQTT_USE_TLS",
    "LLM_BACKEND",
    "OLLAMA_MODEL",
]


def _clear_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_loads_yaml_and_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    yaml_file = tmp_path / "project.yaml"
    yaml_file.write_text(YAML_TEXT, encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MQTT_BROKER_HOST=broker.example.com\n"
        "MQTT_BROKER_PORT=1234\n"
        "MQTT_USE_TLS=false\n",
        encoding="utf-8",
    )

    cfg = load_config(yaml_file, env_file)

    assert cfg.data["project"]["device_id"] == "TEST-001"
    assert cfg.topics["telemetry"] == "t/telemetry"
    assert cfg.mqtt_broker_host == "broker.example.com"
    assert cfg.mqtt_broker_port == 1234
    assert cfg.mqtt_use_tls is False
    assert cfg.db_path == PROJECT_ROOT / "database" / "test.db"


def test_missing_env_file_uses_defaults(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    yaml_file = tmp_path / "project.yaml"
    yaml_file.write_text(YAML_TEXT, encoding="utf-8")

    cfg = load_config(yaml_file, tmp_path / "does-not-exist.env")

    assert cfg.mqtt_broker_host == ""
    assert cfg.mqtt_broker_port == 8883
    assert cfg.mqtt_use_tls is True
    assert cfg.llm_backend == "ollama"
    assert cfg.ollama_model == "llama3.2:3b"


def test_real_project_yaml_loads():
    cfg = load_config()
    assert cfg.topics["telemetry"] == "factory/iot/telemetry"
    assert cfg.db_path.name == "app.db"


def test_resolve_path_relative_and_absolute(tmp_path):
    assert resolve_path("a/b.db") == PROJECT_ROOT / "a" / "b.db"
    absolute = Path(tmp_path) / "x.db"
    assert resolve_path(absolute) == absolute
