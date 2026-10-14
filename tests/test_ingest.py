"""Tests for chunking and ingestion of the markdown manuals."""

from __future__ import annotations

import chromadb
import pytest

from scripts import ingest_manuals as ingest_script
from src.config import get_config, resolve_path
from src.rag.ingest import (
    chunk_markdown,
    ingest_manuals,
    load_chunks,
    split_sections,
    split_text,
)

SAMPLE = """# Pump Guide

Intro paragraph.

## Low Flow

Flow is below nominal. Check the intake strainer.

## Overheating

Housing is hot.
Check the coolant level.
"""


def test_split_text_respects_chunk_size():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = split_text(text, chunk_size=60, chunk_overlap=15)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)


def test_split_text_overlaps_and_covers_all_words():
    words = [f"w{i}" for i in range(50)]
    chunks = split_text(" ".join(words), chunk_size=40, chunk_overlap=10)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.split()[-1] in nxt.split()  # shared trailing word
    assert set(" ".join(chunks).split()) == set(words)


def test_split_text_short_text_is_one_chunk():
    assert split_text("just a few words", 400, 50) == ["just a few words"]


def test_split_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("a b c", chunk_size=10, chunk_overlap=10)


def test_split_sections_uses_headings_and_title_for_preamble():
    sections = dict(split_sections(SAMPLE))
    assert list(sections) == ["Pump Guide", "Low Flow", "Overheating"]
    assert "Intro paragraph" in sections["Pump Guide"]
    assert "intake strainer" in sections["Low Flow"]


def test_chunk_markdown_ids_and_metadata():
    chunks = chunk_markdown(SAMPLE, "pump_guide.md", 400, 50)
    by_section = {c.section: c for c in chunks}
    assert by_section["Overheating"].chunk_id == "pump_guide::overheating::0"
    assert by_section["Overheating"].source == "pump_guide.md"
    # The heading is part of the embedded text so each chunk names its fault.
    assert by_section["Overheating"].text.startswith("Overheating\n")
    assert "coolant level" in by_section["Overheating"].text


def test_chunk_ids_are_unique_and_deterministic_for_real_manual():
    first = load_chunks()
    second = load_chunks()
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in second]


def test_real_manual_has_every_fault_section():
    sections = {c.section for c in load_chunks()}
    assert {
        "High Voltage",
        "Low Voltage",
        "High Current",
        "High Temperature",
        "Low Power Factor / High Reactive Load",
    } <= sections


def test_ingest_manuals_stores_every_chunk(manual_store):
    collection = chromadb.PersistentClient(path=str(manual_store)).get_collection(
        "manuals"
    )
    assert collection.count() == len(load_chunks())


def test_ingest_is_rerunnable_without_duplicates(tmp_path):
    expected = len(load_chunks())
    assert ingest_manuals(store_path=tmp_path) == expected
    assert ingest_manuals(store_path=tmp_path) == expected

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection("manuals")
    assert collection.count() == expected


def test_ingest_with_no_manuals_returns_zero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ingest_manuals(manuals_dir=empty, store_path=tmp_path / "store") == 0


def test_config_points_at_existing_manuals_dir():
    manuals_dir = resolve_path(get_config().data["rag"]["manuals_dir"])
    assert (manuals_dir / "motor_fault_reference.md").is_file()


def test_script_main_ingests_and_prints_query_results(tmp_path, capsys):
    code = ingest_script.main(
        ["--store-path", str(tmp_path), "--query", "motor running hot"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Ingested" in out
    assert "High Temperature" in out


def test_script_main_fails_when_no_manuals(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    code = ingest_script.main(
        ["--manuals-dir", str(empty), "--store-path", str(tmp_path / "store")]
    )
    assert code == 1
    assert "nothing ingested" in capsys.readouterr().err
