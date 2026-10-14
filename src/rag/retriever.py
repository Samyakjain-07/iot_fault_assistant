"""Similarity search over the manuals vector store built by ``rag/ingest.py``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import get_config
from src.rag.ingest import COLLECTION_NAME, get_client, get_embedding_function


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by :func:`retrieve`, best match first.

    ``score`` is cosine similarity (higher is better, 1.0 = identical).
    """

    chunk_id: str
    text: str
    source: str
    section: str
    score: float


def get_chunks(
    chunk_ids: list[str], store_path: Path | None = None
) -> list[dict[str, str]]:
    """Look chunks up by id (e.g. those recorded in ``diagnosis_log``).

    Returns ``chunk_id`` / ``text`` / ``source`` / ``section`` dicts in the
    order of ``chunk_ids``; ids no longer in the store are skipped. Raises
    ``RuntimeError`` if the store has not been built yet.
    """
    if not chunk_ids:
        return []
    client = get_client(store_path)
    if COLLECTION_NAME not in {c.name for c in client.list_collections()}:
        raise RuntimeError(
            "Vector store is empty; run `python scripts/ingest_manuals.py` first."
        )
    # No embedding function needed: a lookup by id never embeds anything.
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(ids=chunk_ids)
    found = {
        chunk_id: {
            "chunk_id": chunk_id,
            "text": text,
            "source": meta["source"],
            "section": meta["section"],
        }
        for chunk_id, text, meta in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
    }
    return [found[i] for i in chunk_ids if i in found]


def retrieve(
    query: str,
    top_k: int | None = None,
    store_path: Path | None = None,
    model_name: str | None = None,
) -> list[RetrievedChunk]:
    """Return the ``top_k`` chunks most similar to ``query``.

    ``top_k`` defaults to ``rag.retrieval_top_k``. Raises ``RuntimeError`` if
    the store has not been built yet (run ``scripts/ingest_manuals.py``).
    """
    rag = get_config().data["rag"]
    top_k = top_k or rag["retrieval_top_k"]
    model_name = model_name or rag["embedding_model"]

    client = get_client(store_path)
    if COLLECTION_NAME not in {c.name for c in client.list_collections()}:
        raise RuntimeError(
            "Vector store is empty; run `python scripts/ingest_manuals.py` first."
        )
    collection = client.get_collection(
        COLLECTION_NAME, embedding_function=get_embedding_function(model_name)
    )

    result = collection.query(query_texts=[query], n_results=top_k)
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            source=meta["source"],
            section=meta["section"],
            score=1.0 - distance,
        )
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
    ]
