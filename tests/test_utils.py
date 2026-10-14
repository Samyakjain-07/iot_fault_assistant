from src.config import Config
from src.utils import create_mqtt_client


def _config(username="user", use_tls=True):
    return Config(
        data={},
        mqtt_broker_host="broker.example",
        mqtt_broker_port=8883,
        mqtt_username=username,
        mqtt_password="secret",
        mqtt_use_tls=use_tls,
        aws_region="",
        llm_backend="ollama",
        ollama_model="m",
        ollama_host="h",
        log_level="INFO",
    )


def test_client_gets_credentials_tls_and_id():
    client = create_mqtt_client(_config(), client_id="test-client")

    assert client._client_id == b"test-client"
    assert client._username == b"user"
    assert client._password == b"secret"
    assert client._ssl_context is not None


def test_tls_disabled_and_no_username():
    client = create_mqtt_client(_config(username="", use_tls=False), client_id="c")

    assert client._ssl_context is None
    assert client._username is None
