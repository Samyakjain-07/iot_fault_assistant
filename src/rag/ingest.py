"""Chunk the markdown manuals and embed them into a persistent Chroma store.

Each ``## `` section of a manual is split into chunks of roughly
``rag.chunk_size`` characters (with ``rag.chunk_overlap`` characters of overlap
between neighbours). Every chunk is prefixed with its section heading so it
still says which fault it describes when retrieved on its own, and carries
``source`` / ``section`` metadata so answers can cite where they came from.

``ingest_manuals`` rebuilds the collection from scratch, so re-running it after
editing a manual never leaves stale chunks behind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import get_config, resolve_path

COLLECTION_NAME = "manuals"

_SECTION_RE = re.compile(r"^## +(.+?) *$", re.MULTILINE)
_TITLE_RE = re.compile(r"^# +(.+?) *$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One embeddable piece of a manual."""

    chunk_id: str
    text: str
    source: str
    section: str


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split ``text`` into word-aligned chunks of at most ``chunk_size`` chars.

    Consecutive chunks share roughly ``chunk_overlap`` characters of trailing
    words. A single word longer than ``chunk_size`` becomes its own chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + 1
        length = len(words[start])
        while end < len(words) and length + 1 + len(words[end]) <= chunk_size:
            length += 1 + len(words[end])
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

        # Step back from `end` to carry ~chunk_overlap chars into the next chunk.
        next_start = end
        overlap = 0
        while (
            next_start > start + 1
            and overlap + 1 + len(words[next_start - 1]) <= chunk_overlap
        ):
            next_start -= 1
            overlap += 1 + len(words[next_start])
        start = next_start
    return chunks


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a manual into ``(section title, body)`` pairs on ``## `` headings.

    Text before the first ``## `` heading (the document title and intro) is
    returned as its own section named after the ``# `` title.
    """
    matches = list(_SECTION_RE.finditer(markdown))
    preamble_end = matches[0].start() if matches else len(markdown)
    preamble = markdown[:preamble_end]

    title_match = _TITLE_RE.search(preamble)
    title = title_match.group(1) if title_match else "Overview"
    intro = _TITLE_RE.sub("", preamble, count=1)
    sections = [(title, intro)]

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((match.group(1), markdown[match.end() : end]))
    return [(name, body) for name, body in sections if body.strip()]


def chunk_markdown(
    markdown: str, source: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Turn one manual into chunks with deterministic ids.

    Ids look like ``<source stem>::<section slug>::<n>`` so the same manual
    always yields the same ids (useful for citing chunks in ``diagnosis_log``).
    """
    stem = Path(source).stem
    chunks: list[Chunk] = []
    for section, body in split_sections(markdown):
        pieces = split_text(body, chunk_size, chunk_overlap)
        for n, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}::{_slug(section)}::{n}",
                    text=f"{section}\n{piece}",
                    source=Path(source).name,
                    section=section,
                )
            )
    return chunks


def load_chunks(
    manuals_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk every ``*.md`` file in the manuals directory (default: config)."""
    rag = get_config().data["rag"]
    manuals_dir = manuals_dir or resolve_path(rag["manuals_dir"])
    chunk_size = chunk_size or rag["chunk_size"]
    chunk_overlap = rag["chunk_overlap"] if chunk_overlap is None else chunk_overlap

    chunks: list[Chunk] = []
    for path in sorted(Path(manuals_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, path.name, chunk_size, chunk_overlap))
    return chunks


@lru_cache(maxsize=2)
def get_embedding_function(model_name: str) -> SentenceTransformerEmbeddingFunction:
    """Load (once per process) the sentence-transformers embedding model."""
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_client(store_path: Path | None = None) -> ClientAPI:
    """Open the persistent Chroma store (default: ``rag.vector_store_path``)."""
    if store_path is None:
        store_path = resolve_path(get_config().data["rag"]["vector_store_path"])
    return chromadb.PersistentClient(path=str(store_path))


def get_collection(client: ClientAPI, model_name: str | None = None) -> Collection:
    """Get the manuals collection, creating it (cosine space) if missing."""
    model_name = model_name or get_config().data["rag"]["embedding_model"]
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(model_name),
        configuration={"hnsw": {"space": "cosine"}},
    )


def ingest_manuals(
    manuals_dir: Path | None = None,
    store_path: Path | None = None,
    model_name: str | None = None,
) -> int:
    """(Re)build the vector store from the manuals; returns the chunk count."""
    chunks = load_chunks(manuals_dir)
    client = get_client(store_path)
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    if not chunks:
        return 0

    collection = get_collection(client, model_name)
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "section": c.section} for c in chunks],
    )
    return len(chunks)
