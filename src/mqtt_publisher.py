"""Synthetic telemetry generator that publishes to EMQX Cloud.

Simulates the induction motor + inline energy meter described in
``data/manuals/motor_fault_reference.md`` (400V / 15A / 7.5kW rated). Normal
readings hover around a healthy operating point; ``FAULT_SCENARIOS`` shift the
relevant signals into the patterns that the fault reference describes, so the
anomaly detector and the RAG layer have something real to find.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import paho.mqtt.client as mqtt

from src.config import Config, get_config
from src.utils import create_mqtt_client

logger = logging.getLogger(__name__)

# Fault name -> the section of the fault reference it exercises.
FAULT_SCENARIOS = (
    "high_voltage",
    "low_voltage",
    "high_current",
    "high_temperature",
    "low_power_factor",
)

CONNECT_TIMEOUT_SECONDS = 15
PUBLISH_TIMEOUT_SECONDS = 10


def _normal_signals(rng: random.Random) -> dict[str, float]:
    """Healthy operating point: ~12A draw, ~0.93 PF, housing around 52C."""
    return {
        "voltage": rng.gauss(400, 3),
        "current": rng.gauss(12, 0.4),
        "power_factor": rng.gauss(0.93, 0.01),
        "temperature": rng.gauss(52, 2),
    }


def _apply_fault(signals: dict[str, float], fault: str, rng: random.Random) -> None:
    """Overwrite the signals a given fault affects, leaving the rest healthy."""
    if fault == "high_voltage":
        # Upstream supply problem: voltage up, current normal or slightly down.
        signals["voltage"] = rng.uniform(430, 450)
        signals["current"] = rng.gauss(11.5, 0.4)
    elif fault == "low_voltage":
        # Sag: voltage down, current up and power factor down together.
        signals["voltage"] = rng.uniform(340, 370)
        signals["current"] = rng.uniform(14, 16)
        signals["power_factor"] = rng.uniform(0.78, 0.85)
    elif fault == "high_current":
        # Mechanical load: current well over the 15A rating, temp still normal.
        signals["current"] = rng.uniform(17, 20)
        signals["temperature"] = rng.gauss(60, 1.5)
    elif fault == "high_temperature":
        # Ventilation/bearing problem: temperature up with current normal.
        signals["temperature"] = rng.uniform(70, 85)
    elif fault == "low_power_factor":
        # Lightly loaded or winding fault: PF well below 0.90, voltage normal.
        signals["power_factor"] = rng.uniform(0.55, 0.75)
        signals["current"] = rng.gauss(11, 0.5)
    else:
        raise ValueError(
            f"Unknown fault {fault!r}; expected one of {', '.join(FAULT_SCENARIOS)}"
        )


def generate_reading(
    device_id: str,
    rng: random.Random | None = None,
    fault: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Produce one telemetry reading, optionally with a fault injected.

    ``power_kw`` is derived from the other signals (three-phase power), so the
    five values stay physically consistent. The returned dict is exactly the
    MQTT payload: it carries no fault label, as the real device wouldn't.
    """
    rng = rng or random.Random()
    signals = _normal_signals(rng)
    if fault is not None:
        _apply_fault(signals, fault, rng)

    power_kw = (
        math.sqrt(3)
        * signals["voltage"]
        * signals["current"]
        * signals["power_factor"]
        / 1000
    )
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "device_id": device_id,
        "timestamp": timestamp.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "voltage": round(signals["voltage"], 1),
        "current": round(signals["current"], 2),
        "power_kw": round(power_kw, 3),
        "power_factor": round(signals["power_factor"], 3),
        "temperature": round(signals["temperature"], 1),
    }


def fault_for_reading(
    index: int,
    fault: str | None,
    fault_after: int = 0,
    fault_count: int | None = None,
) -> str | None:
    """Return the fault to inject into the ``index``-th reading (0-based).

    The fault starts after ``fault_after`` healthy readings and lasts for
    ``fault_count`` readings (``None`` = until the run ends).
    """
    if fault is None or index < fault_after:
        return None
    if fault_count is not None and index >= fault_after + fault_count:
        return None
    return fault


def publish_loop(
    client: mqtt.Client,
    topic: str,
    device_id: str,
    interval: float,
    count: int | None = None,
    fault: str | None = None,
    fault_after: int = 0,
    fault_count: int | None = None,
    rng: random.Random | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Publish readings on an already-connected client; return how many were sent.

    Runs ``count`` readings, or forever when ``count`` is ``None``. Interrupt
    with Ctrl-C.
    """
    if fault is not None and fault not in FAULT_SCENARIOS:
        raise ValueError(
            f"Unknown fault {fault!r}; expected one of {', '.join(FAULT_SCENARIOS)}"
        )
    rng = rng or random.Random()
    sent = 0
    while count is None or sent < count:
        active_fault = fault_for_reading(sent, fault, fault_after, fault_count)
        reading = generate_reading(device_id, rng, active_fault)
        info = client.publish(topic, json.dumps(reading), qos=1)
        info.wait_for_publish(timeout=PUBLISH_TIMEOUT_SECONDS)
        if info.is_published():
            logger.info(
                "published%s: %s", f" [{active_fault}]" if active_fault else "", reading
            )
        else:
            logger.warning("publish not confirmed by broker: %s", reading)
        sent += 1
        if count is None or sent < count:
            sleep(interval)
    return sent


def connect_and_wait(client: mqtt.Client, config: Config) -> None:
    """Connect to the broker and block until CONNACK, or raise ``ConnectionError``.

    Starts the client's network loop thread, which the caller stops with
    ``client.loop_stop()``.
    """
    connected = threading.Event()
    failure: list[str] = []

    def on_connect(_client, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            failure.append(str(reason_code))
        connected.set()

    client.on_connect = on_connect
    try:
        client.connect(config.mqtt_broker_host, config.mqtt_broker_port, keepalive=60)
    except OSError as exc:
        raise ConnectionError(
            f"Could not reach {config.mqtt_broker_host}:"
            f"{config.mqtt_broker_port}: {exc}"
        ) from exc
    client.loop_start()
    if not connected.wait(CONNECT_TIMEOUT_SECONDS):
        client.loop_stop()
        raise ConnectionError(
            "Timed out waiting for the broker to accept the connection"
        )
    if failure:
        client.loop_stop()
        raise ConnectionError(f"Broker refused the connection: {failure[0]}")


def run_publisher(
    config: Config | None = None,
    *,
    interval: float | None = None,
    count: int | None = None,
    fault: str | None = None,
    fault_after: int = 0,
    fault_count: int | None = None,
    seed: int | None = None,
) -> int:
    """Connect to EMQX Cloud and publish telemetry; return the number sent.

    ``interval`` defaults to ``project.telemetry_interval_seconds`` from the
    YAML config. The telemetry topic and device id also come from config.
    """
    config = config or get_config()
    project = config.data["project"]
    if interval is None:
        interval = project["telemetry_interval_seconds"]

    client = create_mqtt_client(config, client_id=f"{project['device_id']}-publisher")
    connect_and_wait(client, config)
    logger.info("connected to %s:%s", config.mqtt_broker_host, config.mqtt_broker_port)
    try:
        return publish_loop(
            client,
            config.topics["telemetry"],
            project["device_id"],
            interval,
            count=count,
            fault=fault,
            fault_after=fault_after,
            fault_count=fault_count,
            rng=random.Random(seed),
        )
    finally:
        client.disconnect()
        client.loop_stop()
