"""Tests for the diagnosis service. The LLM is always stubbed; retrieval is real."""

from __future__ import annotations

import random

import pytest

from src.database import get_diagnoses, init_db, insert_anomaly_event, insert_telemetry
from src.diagnosis_service import (
    GROUNDED,
    LOW_CONFIDENCE,
    NO_CONTEXT_ANSWER,
    build_prompt,
    build_query,
    cited_sections,
    diagnose,
)
from src.llm_service import LLMError
from src.mqtt_publisher import FAULT_SCENARIOS, generate_reading
from src.rag.retriever import RetrievedChunk, retrieve

# Fault injected by the publisher -> manual section that should explain it.
EXPECTED_SECTION = {
    "high_voltage": "High Voltage",
    "low_voltage": "Low Voltage",
    "high_current": "High Current",
    "high_temperature": "High Temperature",
    "low_power_factor": "Low Power Factor",
}


@pytest.fixture
def db_file(tmp_path):
    path = tmp_path / "app.db"
    init_db(path)
    return path


def _flagged(db_file, fault="high_temperature", seed=0):
    """Store a faulty reading and its anomaly event; return (reading, event_id)."""
    reading = generate_reading("MOTOR-001", random.Random(seed), fault)
    telemetry_id = insert_telemetry(reading, db_file)
    event_id = insert_anomaly_event(
        telemetry_id, "MOTOR-001", reading["timestamp"], "ANOMALY", -0.8, db_file
    )
    return reading, event_id


class _StubLLM:
    """Records prompts and returns a fixed answer."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.answer


def _chunk(section="High Temperature", score=0.8, chunk_id="m::high-temperature::0"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"{section}\nsome text",
        source="motor_fault_reference.md",
        section=section,
        score=score,
    )


# --- build_query ---------------------------------------------------------


def test_healthy_reading_is_not_described_as_a_symptom():
    reading = generate_reading("MOTOR-001", random.Random(0))

    assert "Motor fault" not in build_query(reading)


@pytest.mark.parametrize("fault", FAULT_SCENARIOS)
def test_query_for_each_injected_fault_retrieves_its_section(manual_store, fault):
    reading = generate_reading("MOTOR-001", random.Random(3), fault)

    hits = retrieve(build_query(reading), top_k=3, store_path=manual_store)

    assert any(h.section.startswith(EXPECTED_SECTION[fault]) for h in hits)


# --- prompt --------------------------------------------------------------


def test_prompt_restricts_model_to_context_and_demands_citation():
    reading = generate_reading("MOTOR-001", random.Random(0), "high_temperature")

    prompt = build_prompt(reading, [_chunk()])

    assert "ONLY from the REFERENCE PASSAGES" in prompt
    assert "does not cover this" in prompt
    assert "Source: <file> § <section>" in prompt
    assert "[1] Source: motor_fault_reference.md § High Temperature" in prompt
    assert "some text" in prompt
    assert f"temperature: {reading['temperature']}" in prompt


# --- cited_sections ------------------------------------------------------


def test_cited_sections_matches_only_retrieved_sections_case_insensitively():
    chunks = [_chunk("High Temperature"), _chunk("High Current", chunk_id="x")]

    answer = "Likely ventilation. Source: motor_fault_reference.md § high temperature"

    assert cited_sections(answer, chunks) == ["High Temperature"]
    assert cited_sections("Source: § Low Voltage", chunks) == []


# --- diagnose ------------------------------------------------------------


def test_high_temperature_fault_is_logged_with_retrieved_chunk_ids(
    db_file, manual_store
):
    reading, event_id = _flagged(db_file, "high_temperature")
    llm = _StubLLM(
        "Temperature is high with normal current, so check ventilation first.\n"
        "Source: motor_fault_reference.md § High Temperature"
    )

    result = diagnose(
        reading, event_id, db_file, store_path=manual_store, generate_fn=llm
    )

    [row] = get_diagnoses(db_path=db_file)
    expected = retrieve(row["query_text"], store_path=manual_store)
    assert row["id"] == result.diagnosis_id
    assert row["anomaly_event_id"] == event_id
    assert row["retrieved_chunk_ids"] == [c.chunk_id for c in expected]
    assert "High Temperature" in [c.section for c in expected]
    assert "High Temperature" in row["llm_answer"]
    assert row["confidence_flag"] == GROUNDED
    assert result.cited_sections == ["High Temperature"]
    # The model saw exactly the retrieved passages, nothing else.
    assert all(c.text in llm.prompts[0] for c in expected)


def test_answer_without_citation_is_flagged_low_confidence(db_file, manual_store):
    reading, event_id = _flagged(db_file)

    result = diagnose(
        reading,
        event_id,
        db_file,
        store_path=manual_store,
        generate_fn=_StubLLM("Probably a bad bearing."),
    )

    [row] = get_diagnoses(db_path=db_file)
    assert row["confidence_flag"] == LOW_CONFIDENCE
    assert row["llm_answer"].startswith("LOW CONFIDENCE:")
    assert "does not cite" in row["llm_answer"]
    assert row["llm_answer"].endswith("Probably a bad bearing.")
    assert result.cited_sections == []


def test_weak_retrieval_is_flagged_low_confidence(db_file):
    reading, event_id = _flagged(db_file)
    weak = [_chunk(score=0.05)]

    result = diagnose(
        reading,
        event_id,
        db_file,
        generate_fn=_StubLLM("Check airflow. Source: x § High Temperature"),
        retrieve_fn=lambda query, **_: weak,
    )

    assert result.confidence_flag == LOW_CONFIDENCE
    assert "matched weakly" in result.answer
    assert get_diagnoses(db_path=db_file)[0]["confidence_flag"] == LOW_CONFIDENCE


def test_no_retrieved_chunks_skips_the_llm(db_file):
    reading, event_id = _flagged(db_file)

    def must_not_be_called(_prompt):
        raise AssertionError("LLM called without any context")

    result = diagnose(
        reading,
        event_id,
        db_file,
        generate_fn=must_not_be_called,
        retrieve_fn=lambda query, **_: [],
    )

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.confidence_flag == LOW_CONFIDENCE
    assert get_diagnoses(db_path=db_file)[0]["retrieved_chunk_ids"] == []


def test_llm_failure_propagates_and_logs_nothing(db_file, manual_store):
    reading, event_id = _flagged(db_file)

    def broken(_prompt):
        raise LLMError("ollama down")

    with pytest.raises(LLMError):
        diagnose(
            reading, event_id, db_file, store_path=manual_store, generate_fn=broken
        )

    assert get_diagnoses(db_path=db_file) == []


def test_missing_vector_store_propagates_and_logs_nothing(db_file, tmp_path):
    reading, event_id = _flagged(db_file)

    with pytest.raises(RuntimeError, match="ingest_manuals"):
        diagnose(
            reading,
            event_id,
            db_file,
            store_path=tmp_path / "empty_store",
            generate_fn=_StubLLM("unused"),
        )

    assert get_diagnoses(db_path=db_file) == []
