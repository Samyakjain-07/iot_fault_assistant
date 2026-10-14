import sqlite3
from contextlib import closing

import pytest

from src.database import (
    count_telemetry,
    get_anomaly_events,
    get_connection,
    get_diagnoses,
    get_telemetry,
    get_telemetry_by_id,
    init_db,
    insert_anomaly_event,
    insert_diagnosis,
    insert_telemetry,
)

EXPECTED_COLUMNS = {
    "telemetry": [
        "id",
        "device_id",
        "timestamp",
        "voltage",
        "current",
        "power_kw",
        "power_factor",
        "temperature",
    ],
    "anomaly_events": [
        "id",
        "telemetry_id",
        "device_id",
        "timestamp",
        "prediction",
        "anomaly_score",
    ],
    "diagnosis_log": [
        "id",
        "anomaly_event_id",
        "timestamp",
        "query_text",
        "retrieved_chunk_ids",
        "llm_answer",
        "confidence_flag",
    ],
}


def _columns(conn, table):
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_init_db_creates_file_and_all_tables(tmp_path):
    db_file = tmp_path / "nested" / "app.db"

    result = init_db(db_file)

    assert result == db_file
    assert db_file.exists()
    with closing(get_connection(db_file)) as conn:
        for table, columns in EXPECTED_COLUMNS.items():
            assert _columns(conn, table) == columns


def test_init_db_is_idempotent(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    with closing(get_connection(db_file)) as conn, conn:
        conn.execute(
            "INSERT INTO telemetry (device_id, timestamp) VALUES (?, ?)",
            ("MOTOR-001", "2026-01-01T00:00:00Z"),
        )

    init_db(db_file)

    with closing(get_connection(db_file)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
    assert count == 1


def test_foreign_keys_enforced(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)

    with (
        closing(get_connection(db_file)) as conn,
        pytest.raises(sqlite3.IntegrityError),
    ):
        conn.execute(
            "INSERT INTO anomaly_events "
            "(telemetry_id, device_id, timestamp, prediction) "
            "VALUES (?, ?, ?, ?)",
            (999, "MOTOR-001", "2026-01-01T00:00:00Z", "ANOMALY"),
        )


def test_prediction_check_constraint(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)

    with closing(get_connection(db_file)) as conn, conn:
        conn.execute(
            "INSERT INTO telemetry (device_id, timestamp) VALUES (?, ?)",
            ("MOTOR-001", "2026-01-01T00:00:00Z"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO anomaly_events "
                "(telemetry_id, device_id, timestamp, prediction) "
                "VALUES (1, 'MOTOR-001', '2026-01-01T00:00:00Z', 'MAYBE')"
            )


def _reading(**overrides):
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


def test_insert_telemetry_returns_id_and_round_trips(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)

    row_id = insert_telemetry(_reading(), db_file)

    assert row_id == 1
    assert get_telemetry(db_path=db_file) == [{"id": 1, **_reading()}]


def test_insert_telemetry_ignores_extra_keys(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)

    insert_telemetry(_reading(unexpected="x"), db_file)

    assert "unexpected" not in get_telemetry(db_path=db_file)[0]


def test_get_telemetry_newest_first_with_limit_and_device_filter(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    insert_telemetry(_reading(timestamp="2026-01-01T00:00:00Z"), db_file)
    insert_telemetry(_reading(device_id="MOTOR-002"), db_file)
    insert_telemetry(_reading(timestamp="2026-01-01T00:00:10Z"), db_file)

    newest_two = get_telemetry(limit=2, db_path=db_file)
    motor_1 = get_telemetry(device_id="MOTOR-001", db_path=db_file)

    assert [row["id"] for row in newest_two] == [3, 2]
    assert [row["timestamp"] for row in motor_1] == [
        "2026-01-01T00:00:10Z",
        "2026-01-01T00:00:00Z",
    ]


def test_count_telemetry(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    assert count_telemetry(db_file) == 0

    insert_telemetry(_reading(), db_file)
    insert_telemetry(_reading(), db_file)

    assert count_telemetry(db_file) == 2


def test_insert_telemetry_requires_all_fields(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    incomplete = _reading()
    del incomplete["voltage"]

    with pytest.raises(KeyError):
        insert_telemetry(incomplete, db_file)


def test_insert_anomaly_event_round_trips_newest_first(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    first = insert_telemetry(_reading(), db_file)
    second = insert_telemetry(_reading(timestamp="2026-01-01T00:00:05Z"), db_file)

    insert_anomaly_event(
        first, "MOTOR-001", "2026-01-01T00:00:00Z", "ANOMALY", -0.7, db_file
    )
    event_id = insert_anomaly_event(
        second, "MOTOR-001", "2026-01-01T00:00:05Z", "ANOMALY", -0.6, db_file
    )

    events = get_anomaly_events(db_path=db_file)
    assert event_id == 2
    assert [e["telemetry_id"] for e in events] == [second, first]
    assert events[0] == {
        "id": 2,
        "telemetry_id": second,
        "device_id": "MOTOR-001",
        "timestamp": "2026-01-01T00:00:05Z",
        "prediction": "ANOMALY",
        "anomaly_score": -0.6,
    }
    assert len(get_anomaly_events(limit=1, db_path=db_file)) == 1


def test_insert_anomaly_event_rejects_unknown_telemetry_id(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)

    with pytest.raises(sqlite3.IntegrityError):
        insert_anomaly_event(
            99, "MOTOR-001", "2026-01-01T00:00:00Z", "ANOMALY", -0.5, db_file
        )


def _event(db_file):
    telemetry_id = insert_telemetry(_reading(), db_file)
    return insert_anomaly_event(
        telemetry_id, "MOTOR-001", "2026-01-01T00:00:00Z", "ANOMALY", -0.7, db_file
    )


def test_insert_diagnosis_round_trips_chunk_ids_as_list(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    event_id = _event(db_file)

    diagnosis_id = insert_diagnosis(
        event_id,
        "2026-01-01T00:00:01Z",
        "motor running hot",
        ["manual::high-temperature::0", "manual::high-current::0"],
        "Check airflow.",
        "grounded",
        db_file,
    )

    assert diagnosis_id == 1
    assert get_diagnoses(db_path=db_file) == [
        {
            "id": 1,
            "anomaly_event_id": event_id,
            "timestamp": "2026-01-01T00:00:01Z",
            "query_text": "motor running hot",
            "retrieved_chunk_ids": [
                "manual::high-temperature::0",
                "manual::high-current::0",
            ],
            "llm_answer": "Check airflow.",
            "confidence_flag": "grounded",
        }
    ]


def test_get_diagnoses_newest_first_with_event_filter(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    first, second = _event(db_file), _event(db_file)
    for event_id in (first, second, first):
        insert_diagnosis(event_id, "t", "q", [], "a", "low_confidence", db_file)

    assert [d["id"] for d in get_diagnoses(db_path=db_file)] == [3, 2, 1]
    for_first = get_diagnoses(anomaly_event_id=first, db_path=db_file)
    assert [d["id"] for d in for_first] == [3, 1]
    assert len(get_diagnoses(limit=1, db_path=db_file)) == 1


def test_insert_diagnosis_enforces_foreign_key_and_confidence_flag(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    event_id = _event(db_file)

    with pytest.raises(sqlite3.IntegrityError):
        insert_diagnosis(99, "t", "q", [], "a", "grounded", db_file)
    with pytest.raises(sqlite3.IntegrityError):
        insert_diagnosis(event_id, "t", "q", [], "a", "sure", db_file)


def test_get_telemetry_by_id_returns_the_row_or_none(tmp_path):
    db_file = tmp_path / "app.db"
    init_db(db_file)
    row_id = insert_telemetry(_reading(voltage=431.5), db_file)

    row = get_telemetry_by_id(row_id, db_file)

    assert row["id"] == row_id
    assert row["voltage"] == 431.5
    assert get_telemetry_by_id(row_id + 1, db_file) is None
