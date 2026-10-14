import json
import random

import pytest

from src.database import count_telemetry, get_telemetry, init_db
from src.mqtt_publisher import generate_reading
from src.mqtt_subscriber import handle_message, validate_telemetry


def _payload(**overrides):
    reading = {
        "device_id": "MOTOR-001",
        "timestamp": "2026-01-01T00:00:00Z",
        "voltage": 400.0,
        "current": 12.0,
        "power_kw": 7.7,
        "power_factor": 0.93,
        "temperature": 52.0,
    }
    reading.update(overrides)
    return reading


@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "app.db"
    init_db(path)
    return path


# --- validate_telemetry --------------------------------------------------


def test_valid_payload_accepted_as_str_or_bytes():
    raw = json.dumps(_payload())

    assert validate_telemetry(raw) == _payload()
    assert validate_telemetry(raw.encode()) == _payload()


def test_integer_sensor_values_accepted():
    assert validate_telemetry(json.dumps(_payload(voltage=400)))["voltage"] == 400


def test_extra_keys_dropped():
    result = validate_telemetry(json.dumps(_payload(note="hi")))

    assert "note" not in result


@pytest.mark.parametrize("raw", ["not json", b"\xff\xfe", "", "[1, 2]", "42", "null"])
def test_non_object_or_unparseable_payload_rejected(raw):
    with pytest.raises(ValueError):
        validate_telemetry(raw)


@pytest.mark.parametrize(
    "field",
    [
        "device_id",
        "timestamp",
        "voltage",
        "current",
        "power_kw",
        "power_factor",
        "temperature",
    ],
)
def test_missing_field_rejected(field):
    payload = _payload()
    del payload[field]

    with pytest.raises(ValueError, match=f"missing fields: {field}"):
        validate_telemetry(json.dumps(payload))


@pytest.mark.parametrize("device_id", ["", "   ", None, 5])
def test_bad_device_id_rejected(device_id):
    with pytest.raises(ValueError, match="device_id"):
        validate_telemetry(json.dumps(_payload(device_id=device_id)))


@pytest.mark.parametrize("timestamp", ["yesterday", "", None, 12345])
def test_bad_timestamp_rejected(timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        validate_telemetry(json.dumps(_payload(timestamp=timestamp)))


@pytest.mark.parametrize("value", ["400", None, True, [400], {"v": 1}])
def test_non_numeric_sensor_value_rejected(value):
    with pytest.raises(ValueError, match="voltage must be a number"):
        validate_telemetry(json.dumps(_payload(voltage=value)))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_sensor_value_rejected(literal):
    # json.dumps would emit these literals; json.loads accepts them.
    raw = json.dumps(_payload()).replace("400.0", literal)

    with pytest.raises(ValueError, match="voltage must be finite"):
        validate_telemetry(raw)


# --- handle_message ------------------------------------------------------


def test_valid_message_is_stored(db_file):
    row_id = handle_message(json.dumps(_payload()), db_file)

    assert row_id == 1
    assert get_telemetry(db_path=db_file) == [{"id": 1, **_payload()}]


def test_invalid_message_is_dropped_and_not_stored(db_file, caplog):
    assert handle_message("garbage", db_file) is None
    assert handle_message(json.dumps(_payload(voltage="x")), db_file) is None

    assert count_telemetry(db_file) == 0
    assert "dropping invalid telemetry" in caplog.text


def test_database_error_does_not_raise(tmp_path):
    uninitialised = tmp_path / "empty.db"  # no tables

    assert handle_message(json.dumps(_payload()), uninitialised) is None


def test_messages_accumulate_in_order(db_file):
    for i in range(5):
        handle_message(json.dumps(_payload(voltage=400 + i)), db_file)

    assert count_telemetry(db_file) == 5
    assert [r["voltage"] for r in get_telemetry(db_path=db_file)] == [
        404,
        403,
        402,
        401,
        400,
    ]


def test_publisher_output_passes_subscriber_validation(db_file):
    """The generator's payload (healthy and every fault) must always be ingestible."""
    from src.mqtt_publisher import FAULT_SCENARIOS

    rng = random.Random(0)
    for fault in (None, *FAULT_SCENARIOS):
        payload = json.dumps(generate_reading("MOTOR-001", rng, fault))
        assert handle_message(payload, db_file) is not None

    assert count_telemetry(db_file) == 1 + len(FAULT_SCENARIOS)
