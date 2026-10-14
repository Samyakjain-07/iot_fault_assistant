import sqlite3
from contextlib import closing

import pytest

from src.database import (
    count_telemetry,
    get_connection,
    get_telemetry,
    init_db,
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
