"""Tests for similarity search over the ingested motor fault reference."""

from __future__ import annotations

import pytest

from src.config import get_config
from src.rag.retriever import RetrievedChunk, retrieve

MANUAL = "motor_fault_reference.md"


def test_motor_running_hot_retrieves_high_temperature_in_top_3(manual_store):
    hits = retrieve("motor running hot", top_k=3, store_path=manual_store)
    assert len(hits) == 3
    assert "High Temperature" in [h.section for h in hits]


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("motor running hot", "High Temperature"),
        ("supply voltage above 400V, suspect the grid or transformer", "High Voltage"),
        ("voltage sag when other loads start on the circuit", "Low Voltage"),
        ("current above rating, bearing seized or conveyor jammed", "High Current"),
        ("power factor dropped, motor lightly loaded", "Low Power Factor"),
    ],
)
def test_known_query_retrieves_expected_section_in_top_3(
    manual_store, query, expected_section
):
    hits = retrieve(query, top_k=3, store_path=manual_store)
    assert any(h.section.startswith(expected_section) for h in hits)


def test_results_carry_chunk_metadata_and_are_ranked(manual_store):
    hits = retrieve("motor running hot", top_k=3, store_path=manual_store)
    assert all(isinstance(h, RetrievedChunk) for h in hits)
    assert all(h.source == MANUAL for h in hits)
    assert all(h.chunk_id.startswith("motor_fault_reference::") for h in hits)
    assert all(h.text.startswith(h.section) for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_defaults_to_config(manual_store):
    hits = retrieve("motor running hot", store_path=manual_store)
    assert len(hits) == get_config().data["rag"]["retrieval_top_k"]


def test_retrieve_raises_if_store_not_ingested(tmp_path):
    with pytest.raises(RuntimeError, match="ingest_manuals"):
        retrieve("motor running hot", store_path=tmp_path)
