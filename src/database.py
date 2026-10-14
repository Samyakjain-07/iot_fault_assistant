"""SQLite access layer.

All database access goes through this module so the backing store can be
swapped later without touching callers. Schema follows
``docs/05_DATA_SCHEMA.md``.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from src.config import get_config, resolve_path

# The five sensor fields that feed the ML feature set (docs/05_DATA_SCHEMA.md).
TELEMETRY_FIELDS = ("voltage", "current", "power_kw", "power_factor", "temperature")
TELEMETRY_COLUMNS = ("device_id", "timestamp", *TELEMETRY_FIELDS)

SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id    TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    voltage      REAL,
    current      REAL,
    power_kw     REAL,
    power_factor REAL,
    temperature  REAL
);

CREATE TABLE IF NOT EXISTS anomaly_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    telemetry_id  INTEGER NOT NULL REFERENCES telemetry(id),
    device_id     TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    prediction    TEXT NOT NULL CHECK (prediction IN ('NORMAL', 'ANOMALY')),
    anomaly_score REAL
);

CREATE TABLE IF NOT EXISTS diagnosis_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    anomaly_event_id    INTEGER NOT NULL REFERENCES anomaly_events(id),
    timestamp           TEXT NOT NULL,
    query_text          TEXT,
    retrieved_chunk_ids TEXT,
    llm_answer          TEXT,
    confidence_flag     TEXT CHECK (confidence_flag IN ('grounded', 'low_confidence'))
);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection to the app database with foreign keys enforced.

    Defaults to the path in ``config/project.yaml``. The caller is
    responsible for closing the connection.
    """
    path = resolve_path(db_path) if db_path else get_config().db_path
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path | None = None) -> Path:
    """Create the database file and all three tables if they don't exist.

    Safe to call repeatedly. Returns the path of the database file.
    """
    path = resolve_path(db_path) if db_path else get_config().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(get_connection(path)) as conn, conn:
        conn.executescript(SCHEMA)
    return path


def insert_telemetry(
    reading: dict[str, Any], db_path: str | Path | None = None
) -> int:
    """Insert one telemetry reading and return its new row id.

    ``reading`` must have ``device_id`` and ``timestamp`` plus the five sensor
    fields in ``TELEMETRY_FIELDS``; any extra keys are ignored.
    """
    values = [reading[column] for column in TELEMETRY_COLUMNS]
    placeholders = ", ".join("?" for _ in TELEMETRY_COLUMNS)
    sql = (
        f"INSERT INTO telemetry ({', '.join(TELEMETRY_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    with closing(get_connection(db_path)) as conn, conn:
        cursor = conn.execute(sql, values)
        return cursor.lastrowid


def get_telemetry(
    limit: int = 100,
    device_id: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent telemetry rows, newest first.

    Optionally restricted to one ``device_id``.
    """
    sql = "SELECT * FROM telemetry"
    params: list[Any] = []
    if device_id is not None:
        sql += " WHERE device_id = ?"
        params.append(device_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with closing(get_connection(db_path)) as conn:
        return [dict(row) for row in conn.execute(sql, params)]


def count_telemetry(db_path: str | Path | None = None) -> int:
    """Return the number of rows in the ``telemetry`` table."""
    with closing(get_connection(db_path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
