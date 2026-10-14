"""SQLite access layer.

All database access goes through this module so the backing store can be
swapped later without touching callers. Schema follows
``docs/05_DATA_SCHEMA.md``.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from src.config import get_config, resolve_path

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
