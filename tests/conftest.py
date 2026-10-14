"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.rag.ingest import ingest_manuals


@pytest.fixture(scope="session")
def manual_store(tmp_path_factory: pytest.TempPathFactory):
    """A Chroma store built once from the real ``data/manuals`` content."""
    store_path = tmp_path_factory.mktemp("rag_store")
    ingest_manuals(store_path=store_path)
    return store_path
