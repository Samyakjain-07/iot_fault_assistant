"""Tests for the Streamlit dashboard, driven through ``streamlit.testing``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import get_config
from src.database import (
    get_diagnoses,
    init_db,
    insert_anomaly_event,
    insert_diagnosis,
    insert_telemetry,
)

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
HOT_ANSWER = (
    "The motor is overheating.\n\nSource: motor_fault_reference.md § High Temperature"
)
PASSAGES = {
    "c1": {
        "chunk_id": "c1",
        "text": "High Temperature: check the cooling fan.",
        "source": "motor_fault_reference.md",
        "section": "High Temperature",
    },
    "c2": {
        "chunk_id": "c2",
        "text": "Low Voltage: look for sags.",
        "source": "motor_fault_reference.md",
        "section": "Low Voltage",
    },
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    """An empty app database that the dashboard (via src.database) will use."""
    config = get_config()
    data = {**config.data, "database": {"path": str(tmp_path / "app.db")}}
    monkeypatch.setattr(
        "src.database.get_config", lambda: dataclasses.replace(config, data=data)
    )
    return init_db()


def _reading(i: int, **overrides) -> dict:
    reading = {
        "device_id": "MOTOR-001",
        "timestamp": f"2026-10-14T10:00:{i:02d}Z",
        "voltage": 400.0 + i,
        "current": 12.0,
        "power_kw": 7.7,
        "power_factor": 0.93,
        "temperature": 52.0,
    }
    return {**reading, **overrides}


def _seed(db, readings: int = 10, anomaly: bool = True) -> int | None:
    """Store telemetry; if ``anomaly``, flag the last reading. Returns event id."""
    for i in range(readings - 1):
        insert_telemetry(_reading(i), db)
    last = insert_telemetry(_reading(readings - 1, temperature=80.0), db)
    if not anomaly:
        return None
    return insert_anomaly_event(
        last, "MOTOR-001", "2026-10-14T10:00:09Z", "ANOMALY", -0.21, db
    )


def _log_diagnosis(db, event_id, answer=HOT_ANSWER, flag="grounded") -> int:
    return insert_diagnosis(
        event_id, "2026-10-14T10:01:00Z", "motor running hot", ["c1", "c2"],
        answer, flag, db,
    )  # fmt: skip


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_empty_database_shows_hint_not_an_error(db):
    at = _run()

    assert not at.exception
    assert any("No telemetry yet" in i.value for i in at.info)
    assert not at.selectbox  # nothing to diagnose


def test_live_view_shows_latest_values_and_anomaly_list(db):
    event_id = _seed(db)

    at = _run()

    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Voltage (V)"] == "409.0"
    assert metrics["Temperature (°C)"] == "80.0"
    assert len(metrics) == 5
    table = at.dataframe[0].value
    assert list(table["event"]) == [event_id]
    assert list(table["diagnosis"]) == ["pending"]


def test_without_anomalies_chat_says_nothing_to_diagnose(db):
    _seed(db, anomaly=False)

    at = _run()

    assert not at.exception
    assert not at.dataframe
    assert not at.selectbox
    assert any("no anomalies" in m.value.lower() for m in at.markdown)


def test_logged_diagnosis_shows_answer_and_cited_passage(db, monkeypatch):
    monkeypatch.setattr(
        "src.rag.retriever.get_chunks",
        lambda ids: [PASSAGES[i] for i in ids],
    )
    event_id = _seed(db)
    _log_diagnosis(db, event_id)

    at = _run()

    assert not at.exception
    assert list(at.dataframe[0].value["diagnosis"]) == ["grounded"]
    assert any("The motor is overheating." in m.value for m in at.markdown)
    cited = next(e for e in at.expander if e.label.endswith("(cited)"))
    assert cited.label == "motor_fault_reference.md § High Temperature (cited)"
    assert cited.proto.expanded
    assert any("check the cooling fan" in m.value for m in cited.markdown)
    other = next(e for e in at.expander if "Low Voltage" in e.label)
    assert not other.proto.expanded
    assert at.button[0].label == "Diagnose again"


def test_missing_vector_store_degrades_to_a_warning(db, monkeypatch):
    def no_store(_ids):
        raise RuntimeError("Vector store is empty")

    monkeypatch.setattr("src.rag.retriever.get_chunks", no_store)
    _log_diagnosis(db, _seed(db))

    at = _run()

    assert not at.exception
    assert any("Vector store is empty" in w.value for w in at.warning)
    assert any("The motor is overheating." in m.value for m in at.markdown)


def test_diagnose_button_runs_diagnosis_and_shows_result(db, monkeypatch):
    monkeypatch.setattr(
        "src.rag.retriever.get_chunks", lambda ids: [PASSAGES[i] for i in ids]
    )
    event_id = _seed(db)
    calls = []

    def fake_diagnose(reading, anomaly_event_id):
        calls.append((reading, anomaly_event_id))
        _log_diagnosis(db, anomaly_event_id)

    monkeypatch.setattr("src.diagnosis_service.diagnose", fake_diagnose)
    at = _run()
    assert at.button[0].label == "Diagnose this anomaly"

    at.button[0].click().run()

    assert not at.exception
    assert [(c[0]["temperature"], c[1]) for c in calls] == [(80.0, event_id)]
    assert len(get_diagnoses(db_path=db)) == 1
    assert any("The motor is overheating." in m.value for m in at.markdown)
    assert at.button[0].label == "Diagnose again"


def test_diagnose_failure_is_shown_not_raised(db, monkeypatch):
    def broken(_reading, _event_id):
        raise RuntimeError("Ollama request failed")

    monkeypatch.setattr("src.diagnosis_service.diagnose", broken)
    _seed(db)
    at = _run()

    at.button[0].click().run()

    assert not at.exception
    assert any("Ollama request failed" in e.value for e in at.error)
    assert get_diagnoses(db_path=db) == []
