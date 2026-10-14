"""MQTT subscriber: ingests telemetry from EMQX Cloud into SQLite.

Subscribes to the telemetry topic, validates each message, writes it to the
``telemetry`` table, and scores it with the anomaly detector; readings flagged
ANOMALY are written to ``anomaly_events``, and each one is then diagnosed
(RAG + LLM) in a background thread, with the answer logged to
``diagnosis_log``.

Run with ``python -m src.mqtt_subscriber``.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import Config, get_config
from src.database import (
    TELEMETRY_COLUMNS,
    TELEMETRY_FIELDS,
    init_db,
    insert_anomaly_event,
    insert_telemetry,
)
from src.diagnosis_service import diagnose
from src.prediction_service import PredictionService
from src.utils import create_mqtt_client

logger = logging.getLogger(__name__)

# Called with (reading, anomaly_event_id) for every reading flagged ANOMALY.
Diagnoser = Callable[[dict[str, Any], int], object]


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


def run_diagnosis(
    reading: dict[str, Any], anomaly_event_id: int, db_path: str | Path | None = None
) -> None:
    """Diagnose one flagged reading, logging (never raising) any failure.

    A failed diagnosis (Ollama down, vector store not built, ...) writes no
    ``diagnosis_log`` row: there is no grounded answer to record.
    """
    try:
        diagnosis = diagnose(reading, anomaly_event_id, db_path)
    except Exception:  # noqa: BLE001 - one failed diagnosis must not stop ingestion
        logger.exception("diagnosis failed for anomaly_event %s", anomaly_event_id)
        return
    logger.info(
        "diagnosis id=%s for anomaly_event %s (%s, cited: %s)",
        diagnosis.diagnosis_id,
        anomaly_event_id,
        diagnosis.confidence_flag,
        ", ".join(diagnosis.cited_sections) or "none",
    )


def detect_anomaly(
    reading: dict[str, Any],
    telemetry_id: int,
    predictor: PredictionService,
    db_path: str | Path | None = None,
    diagnoser: Diagnoser | None = None,
) -> bool:
    """Score a stored reading; if it is an ANOMALY, record it in ``anomaly_events``.

    Only flagged readings are written. Returns whether the reading was
    flagged. A database failure is logged and the reading is reported as not
    flagged, so it never interrupts ingestion. If a ``diagnoser`` is given it
    is called with ``(reading, anomaly_event_id)`` for each flagged reading.
    """
    prediction = predictor.predict(reading)
    if not prediction.is_anomaly:
        return False
    try:
        event_id = insert_anomaly_event(
            telemetry_id,
            reading["device_id"],
            reading["timestamp"],
            prediction.label,
            prediction.score,
            db_path,
        )
    except sqlite3.Error:
        logger.exception("failed to store anomaly event for telemetry %s", telemetry_id)
        return False
    logger.warning(
        "ANOMALY anomaly_event id=%s telemetry id=%s score=%.3f: %s",
        event_id,
        telemetry_id,
        prediction.score,
        reading,
    )
    if diagnoser is not None:
        try:
            diagnoser(reading, event_id)
        except Exception:  # noqa: BLE001 - never let diagnosis interrupt ingestion
            logger.exception("could not start diagnosis for anomaly_event %s", event_id)
    return True


def handle_message(
    payload: bytes | str,
    db_path: str | Path | None = None,
    predictor: PredictionService | None = None,
    diagnoser: Diagnoser | None = None,
) -> int | None:
    """Validate one telemetry message, store it, and run anomaly detection.

    Detection runs only when a ``predictor`` is given; flagged readings are
    written to ``anomaly_events`` and, if a ``diagnoser`` is given, handed to
    it. Returns the new ``telemetry`` row id, or ``None`` if the message was
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
    if predictor is not None:
        detect_anomaly(reading, row_id, predictor, db_path, diagnoser)
    return row_id


def load_predictor() -> PredictionService | None:
    """Load the trained model, or return ``None`` (and warn) if there isn't one.

    A missing model degrades the subscriber to ingest-only rather than
    stopping it.
    """
    try:
        return PredictionService.load()
    except FileNotFoundError as exc:
        logger.warning("anomaly detection disabled: %s", exc)
        return None


def run_subscriber(
    config: Config | None = None, db_path: str | Path | None = None
) -> None:
    """Connect to EMQX Cloud and store telemetry until interrupted (Ctrl-C).

    Creates the database if needed. Raises ``ConnectionError`` if the broker
    is unreachable or rejects the credentials.
    """
    config = config or get_config()
    init_db(db_path)
    predictor = load_predictor()
    # One worker: LLM calls take seconds, so they run off the MQTT network
    # thread, one at a time so a fault burst doesn't flood a local model.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diagnosis")

    def diagnoser(reading: dict[str, Any], anomaly_event_id: int) -> None:
        executor.submit(run_diagnosis, reading, anomaly_event_id, db_path)

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
        handle_message(message.payload, db_path, predictor, diagnoser)

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
        executor.shutdown(wait=False, cancel_futures=True)
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
