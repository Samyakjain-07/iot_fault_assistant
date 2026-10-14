import json
import random

import pytest

from src.anomaly_detector import train
from src.database import (
    count_telemetry,
    get_anomaly_events,
    get_diagnoses,
    get_telemetry,
    init_db,
)
from src.diagnosis_service import diagnose
from src.feature_engineering import build_feature_matrix
from src.llm_service import LLMError
from src.mqtt_publisher import generate_reading
from src.mqtt_subscriber import (
    handle_message,
    load_predictor,
    run_diagnosis,
    validate_telemetry,
)
from src.prediction_service import Prediction, PredictionService
from src.rag.retriever import retrieve


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


# --- anomaly detection wiring --------------------------------------------


@pytest.fixture(scope="module")
def predictor():
    rng = random.Random(0)
    readings = [generate_reading("MOTOR-001", rng) for _ in range(800)]
    model = train(build_feature_matrix(readings), contamination=0.05, random_state=0)
    return PredictionService(model)


class _AlwaysAnomaly:
    def predict(self, reading):
        return Prediction("ANOMALY", -0.9)


def test_anomalous_reading_is_written_to_anomaly_events(db_file, predictor):
    faulty = _payload(temperature=150.0, current=30.0, voltage=520.0)

    telemetry_id = handle_message(json.dumps(faulty), db_file, predictor)

    events = get_anomaly_events(db_path=db_file)
    assert len(events) == 1
    assert events[0]["telemetry_id"] == telemetry_id
    assert events[0]["device_id"] == "MOTOR-001"
    assert events[0]["timestamp"] == faulty["timestamp"]
    assert events[0]["prediction"] == "ANOMALY"
    assert events[0]["anomaly_score"] < 0


def test_normal_reading_is_stored_but_not_flagged(db_file, predictor):
    handle_message(json.dumps(_payload(power_kw=7.73)), db_file, predictor)

    assert count_telemetry(db_file) == 1
    assert get_anomaly_events(db_path=db_file) == []


def test_no_predictor_means_no_detection(db_file):
    faulty = _payload(temperature=150.0, current=30.0, voltage=520.0)

    assert handle_message(json.dumps(faulty), db_file) is not None
    assert get_anomaly_events(db_path=db_file) == []


def test_invalid_message_is_not_scored(db_file):
    assert handle_message("garbage", db_file, _AlwaysAnomaly()) is None
    assert get_anomaly_events(db_path=db_file) == []


def test_anomaly_event_links_to_its_own_telemetry_row(db_file):
    handle_message(json.dumps(_payload()), db_file)  # row 1, no detection
    second = handle_message(json.dumps(_payload()), db_file, _AlwaysAnomaly())

    assert second == 2
    assert get_anomaly_events(db_path=db_file)[0]["telemetry_id"] == 2


def test_injected_faults_are_flagged_end_to_end(db_file, predictor):
    """Generator output -> subscriber -> anomaly_events, for each fault type."""
    from src.mqtt_publisher import FAULT_SCENARIOS

    rng = random.Random(1)
    for fault in FAULT_SCENARIOS:
        for _ in range(10):
            payload = json.dumps(generate_reading("MOTOR-001", rng, fault))
            handle_message(payload, db_file, predictor)

    flagged = len(get_anomaly_events(limit=1000, db_path=db_file))
    assert count_telemetry(db_file) == 10 * len(FAULT_SCENARIOS)
    assert flagged >= 0.7 * count_telemetry(db_file)


# --- diagnosis wiring ----------------------------------------------------


def test_anomaly_triggers_diagnoser_with_reading_and_event_id(db_file):
    calls = []

    handle_message(
        json.dumps(_payload()),
        db_file,
        _AlwaysAnomaly(),
        lambda reading, event_id: calls.append((reading, event_id)),
    )

    event_id = get_anomaly_events(db_path=db_file)[0]["id"]
    assert calls == [(_payload(), event_id)]


def test_normal_reading_does_not_trigger_diagnoser(db_file, predictor):
    calls = []

    handle_message(
        json.dumps(_payload(power_kw=7.73)),
        db_file,
        predictor,
        lambda *args: calls.append(args),
    )

    assert calls == []


def test_diagnoser_failure_does_not_stop_ingestion(db_file, caplog):
    def broken(_reading, _event_id):
        raise RuntimeError("boom")

    row_id = handle_message(json.dumps(_payload()), db_file, _AlwaysAnomaly(), broken)

    assert row_id == 1
    assert len(get_anomaly_events(db_path=db_file)) == 1
    assert "could not start diagnosis" in caplog.text


def test_injected_high_temperature_fault_produces_a_grounded_diagnosis_row(
    db_file, predictor, manual_store
):
    """Generator -> detector -> retriever -> (stubbed) LLM -> diagnosis_log."""
    # Seed 1 is a high-temperature reading the fixture predictor flags (the
    # detector catches only some of them, so the seed is chosen deliberately).
    reading = generate_reading("MOTOR-001", random.Random(1), "high_temperature")
    prompts = []

    def stub_llm(prompt):
        prompts.append(prompt)
        return (
            "Temperature is high while current is normal: check ventilation.\n"
            "Source: motor_fault_reference.md § High Temperature"
        )

    def diagnoser(flagged, event_id):
        diagnose(
            flagged,
            event_id,
            db_file,
            store_path=manual_store,
            generate_fn=stub_llm,
        )

    handle_message(json.dumps(reading), db_file, predictor, diagnoser)

    [event] = get_anomaly_events(db_path=db_file)
    [row] = get_diagnoses(db_path=db_file)
    returned = retrieve(row["query_text"], store_path=manual_store)
    assert row["anomaly_event_id"] == event["id"]
    assert row["retrieved_chunk_ids"] == [c.chunk_id for c in returned]
    assert "High Temperature" in row["llm_answer"]
    assert row["confidence_flag"] == "grounded"
    assert len(prompts) == 1


def test_run_diagnosis_logs_failures_instead_of_raising(monkeypatch, caplog):
    def failing(*_args, **_kwargs):
        raise LLMError("ollama down")

    monkeypatch.setattr("src.mqtt_subscriber.diagnose", failing)

    run_diagnosis(_payload(), 7)

    assert "diagnosis failed for anomaly_event 7" in caplog.text


def test_load_predictor_returns_none_without_a_model(monkeypatch, caplog):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(PredictionService, "load", missing)

    assert load_predictor() is None
    assert "anomaly detection disabled" in caplog.text
