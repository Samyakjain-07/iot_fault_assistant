"""MQTT subscriber: ingests telemetry from EMQX Cloud into SQLite.

Phase 2 scope: subscribe to the telemetry topic, validate each message, and
write it to the ``telemetry`` table. Later phases hook anomaly detection and
diagnosis into :func:`handle_message`.

Run with ``python -m src.mqtt_subscriber``.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Config, get_config
from src.database import TELEMETRY_COLUMNS, TELEMETRY_FIELDS, init_db, insert_telemetry
from src.utils import create_mqtt_client

logger = logging.getLogger(__name__)


def validate_telemetry(payload: bytes | str) -> dict[str, Any]:
    """Parse and validate a telemetry message.

    Returns a dict holding exactly the ``telemetry`` columns (extra keys in
    the message are dropped). Raises ``ValueError`` if the payload is not a
    JSON object, a field is missing, ``device_id``/``timestamp`` are not
    usable, or a sensor value is not a finite number.
    """
    try:
        message = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ValueError("payload must be a JSON object")  # noqa: TRY004

    missing = [name for name in TELEMETRY_COLUMNS if name not in message]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")

    device_id = message["device_id"]
    if not isinstance(device_id, str) or not device_id.strip():
        raise ValueError("device_id must be a non-empty string")

    timestamp = message["timestamp"]
    try:
        datetime.fromisoformat(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timestamp is not ISO 8601: {timestamp!r}") from exc

    for name in TELEMETRY_FIELDS:
        value = message[name]
        # bool is an int subclass; reject it so `true` isn't stored as 1.0.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a number, got {value!r}")  # noqa: TRY004
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    return {name: message[name] for name in TELEMETRY_COLUMNS}


def handle_message(
    payload: bytes | str, db_path: str | Path | None = None
) -> int | None:
    """Validate one telemetry message and store it.

    Returns the new ``telemetry`` row id, or ``None`` if the message was
    invalid or could not be written. Never raises for bad input -- one bad
    message must not stop the subscriber.
    """
    try:
        reading = validate_telemetry(payload)
    except ValueError as exc:
        logger.warning("dropping invalid telemetry: %s", exc)
        return None
    try:
        row_id = insert_telemetry(reading, db_path)
    except sqlite3.Error:
        logger.exception("failed to store telemetry")
        return None
    logger.info("stored telemetry id=%s: %s", row_id, reading)
    return row_id


def run_subscriber(
    config: Config | None = None, db_path: str | Path | None = None
) -> None:
    """Connect to EMQX Cloud and store telemetry until interrupted (Ctrl-C).

    Creates the database if needed. Raises ``ConnectionError`` if the broker
    is unreachable or rejects the credentials.
    """
    config = config or get_config()
    init_db(db_path)
    topic = config.topics["telemetry"]
    device_id = config.data["project"]["device_id"]
    refused: list[str] = []

    client = create_mqtt_client(config, client_id=f"{device_id}-subscriber")

    def on_connect(client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            refused.append(str(reason_code))
            client.disconnect()  # ends loop_forever
            return
        # Subscribing here (not once at startup) restores the subscription
        # after an automatic reconnect.
        client.subscribe(topic, qos=1)
        logger.info("subscribed to %s", topic)

    def on_message(_client, _userdata, message):
        handle_message(message.payload, db_path)

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(config.mqtt_broker_host, config.mqtt_broker_port, keepalive=60)
    except OSError as exc:
        raise ConnectionError(
            f"Could not reach {config.mqtt_broker_host}:"
            f"{config.mqtt_broker_port}: {exc}"
        ) from exc
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")
    finally:
        client.disconnect()
    if refused:
        raise ConnectionError(f"Broker refused the connection: {refused[0]}")


def main() -> None:
    """Entry point for ``python -m src.mqtt_subscriber``."""
    config = get_config()
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_subscriber(config)


if __name__ == "__main__":
    main()
