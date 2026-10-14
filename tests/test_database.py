import sqlite3
from contextlib import closing

import pytest

from src.database import get_connection, init_db

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
