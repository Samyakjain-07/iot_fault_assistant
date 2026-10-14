import json
import random
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src import mqtt_publisher
from src.config import Config
from src.mqtt_publisher import (
    FAULT_SCENARIOS,
    connect_and_wait,
    fault_for_reading,
    generate_reading,
    publish_loop,
)

SENSOR_FIELDS = ("voltage", "current", "power_kw", "power_factor", "temperature")


def _readings(fault=None, n=200, seed=1):
    rng = random.Random(seed)
    return [generate_reading("MOTOR-001", rng, fault) for _ in range(n)]


# --- generate_reading ----------------------------------------------------


def test_reading_has_exactly_the_payload_fields():
    reading = generate_reading("MOTOR-001", random.Random(0))

    assert set(reading) == {"device_id", "timestamp", *SENSOR_FIELDS}
    assert reading["device_id"] == "MOTOR-001"


def test_timestamp_is_iso_utc():
    now = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)

    reading = generate_reading("MOTOR-001", random.Random(0), now=now)

    assert reading["timestamp"] == "2026-03-04T05:06:07Z"


def test_same_seed_gives_same_readings():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = generate_reading("M", random.Random(7), "high_current", now)
    b = generate_reading("M", random.Random(7), "high_current", now)

    assert a == b


def test_normal_readings_stay_inside_healthy_ranges():
    for r in _readings():
        assert 380 < r["voltage"] < 420
        assert 10 < r["current"] < 15
        assert 0.90 <= r["power_factor"] <= 0.96
        assert 40 < r["temperature"] < 65


def test_power_is_consistent_with_other_signals():
    for r in _readings(fault="low_voltage", n=20):
        expected = 3**0.5 * r["voltage"] * r["current"] * r["power_factor"] / 1000
        assert r["power_kw"] == pytest.approx(expected, rel=0.01)


# --- fault scenarios -----------------------------------------------------
# Each scenario must show the signature described in the fault reference.


def test_high_voltage_fault():
    for r in _readings("high_voltage"):
        assert r["voltage"] > 420
        assert r["current"] < 15


def test_low_voltage_fault_moves_voltage_current_and_pf_together():
    for r in _readings("low_voltage"):
        assert r["voltage"] < 380
        assert r["current"] > 13
        assert r["power_factor"] < 0.88


def test_high_current_fault_with_normal_temperature():
    for r in _readings("high_current"):
        assert r["current"] > 15
        assert r["temperature"] < 65
        assert 380 < r["voltage"] < 420


def test_high_temperature_fault_with_normal_current():
    for r in _readings("high_temperature"):
        assert r["temperature"] > 65
        assert r["current"] < 15


def test_low_power_factor_fault_with_normal_voltage():
    for r in _readings("low_power_factor"):
        assert r["power_factor"] < 0.80
        assert 380 < r["voltage"] < 420


def test_every_scenario_is_covered_by_a_generator_branch():
    for fault in FAULT_SCENARIOS:
        assert generate_reading("M", random.Random(0), fault)


def test_unknown_fault_rejected():
    with pytest.raises(ValueError, match="Unknown fault"):
        generate_reading("M", random.Random(0), "bogus")


# --- fault_for_reading ---------------------------------------------------


def test_no_fault_requested_means_never_faulty():
    assert all(fault_for_reading(i, None) is None for i in range(5))


def test_fault_applies_from_start_by_default():
    assert fault_for_reading(0, "high_current") == "high_current"


def test_fault_window_with_delay_and_count():
    window = [
        fault_for_reading(i, "high_current", fault_after=2, fault_count=3)
        for i in range(7)
    ]

    assert window == [
        None,
        None,
        "high_current",
        "high_current",
        "high_current",
        None,
        None,
    ]


def test_fault_without_count_lasts_to_the_end():
    assert fault_for_reading(1000, "high_current", fault_after=2) == "high_current"


# --- publish_loop --------------------------------------------------------


class FakeClient:
    def __init__(self, published=True):
        self.messages = []
        self._published = published

    def publish(self, topic, payload, qos=0):
        self.messages.append((topic, json.loads(payload), qos))
        return SimpleNamespace(
            wait_for_publish=lambda timeout=None: None,
            is_published=lambda: self._published,
        )


def test_publish_loop_sends_count_readings_on_topic_with_qos1():
    client = FakeClient()
    sleeps = []

    sent = publish_loop(
        client, "factory/iot/telemetry", "MOTOR-001", 5, count=3, sleep=sleeps.append
    )

    assert sent == 3
    assert [(t, q) for t, _, q in client.messages] == [("factory/iot/telemetry", 1)] * 3
    assert all(m["device_id"] == "MOTOR-001" for _, m, _ in client.messages)
    # Sleeps between readings, not after the last one.
    assert sleeps == [5, 5]


def test_publish_loop_injects_fault_window():
    client = FakeClient()

    publish_loop(
        client,
        "t",
        "MOTOR-001",
        0,
        count=6,
        fault="high_temperature",
        fault_after=2,
        fault_count=2,
        rng=random.Random(3),
        sleep=lambda _: None,
    )

    hot = [m["temperature"] > 65 for _, m, _ in client.messages]
    assert hot == [False, False, True, True, False, False]


def test_publish_loop_rejects_unknown_fault_before_publishing():
    client = FakeClient()

    with pytest.raises(ValueError, match="Unknown fault"):
        publish_loop(client, "t", "M", 0, count=1, fault="bogus")

    assert client.messages == []


def test_publish_loop_continues_when_broker_does_not_confirm():
    client = FakeClient(published=False)

    sent = publish_loop(client, "t", "M", 0, count=2, sleep=lambda _: None)

    assert sent == 2


# --- connect_and_wait ----------------------------------------------------


def _config():
    return Config(
        data={},
        mqtt_broker_host="broker.example",
        mqtt_broker_port=8883,
        mqtt_username="u",
        mqtt_password="p",
        mqtt_use_tls=True,
        aws_region="",
        llm_backend="ollama",
        ollama_model="m",
        ollama_host="h",
        log_level="INFO",
    )


class FakeConnectClient:
    """Fires on_connect from loop_start, like paho's network thread would."""

    def __init__(self, failed=None, connect_error=None):
        self.on_connect = None
        self.failed = failed
        self.connect_error = connect_error
        self.loop_running = False

    def connect(self, host, port, keepalive=60):
        if self.connect_error:
            raise self.connect_error

    def loop_start(self):
        self.loop_running = True
        if self.failed is not None:
            self.on_connect(self, None, None, _Reason(self.failed), None)

    def loop_stop(self):
        self.loop_running = False


class _Reason:
    def __init__(self, is_failure):
        self.is_failure = is_failure

    def __str__(self):
        return "Not authorized" if self.is_failure else "Success"


def test_connect_and_wait_success_leaves_loop_running():
    client = FakeConnectClient(failed=False)

    connect_and_wait(client, _config())

    assert client.loop_running


def test_connect_and_wait_bad_credentials():
    client = FakeConnectClient(failed=True)

    with pytest.raises(ConnectionError, match="Not authorized"):
        connect_and_wait(client, _config())

    assert not client.loop_running


def test_connect_and_wait_unreachable_broker():
    client = FakeConnectClient(connect_error=OSError("no route"))

    with pytest.raises(ConnectionError, match="broker.example:8883"):
        connect_and_wait(client, _config())


def test_connect_and_wait_times_out(monkeypatch):
    monkeypatch.setattr(mqtt_publisher, "CONNECT_TIMEOUT_SECONDS", 0.01)
    client = FakeConnectClient(failed=None)  # never calls on_connect

    with pytest.raises(ConnectionError, match="Timed out"):
        connect_and_wait(client, _config())

    assert not client.loop_running
