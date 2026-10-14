"""Small helpers shared across modules."""

from __future__ import annotations

import ssl

import paho.mqtt.client as mqtt

from src.config import Config


def create_mqtt_client(config: Config, client_id: str) -> mqtt.Client:
    """Build a paho client configured for the EMQX Cloud broker.

    Applies the credentials from ``.env`` and, when ``MQTT_USE_TLS`` is on,
    TLS with certificate verification against the system CA store. Does not
    connect -- the caller decides when.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    if config.mqtt_username:
        client.username_pw_set(config.mqtt_username, config.mqtt_password)
    if config.mqtt_use_tls:
        client.tls_set_context(ssl.create_default_context())
    return client
