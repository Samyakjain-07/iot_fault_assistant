"""Chunk and embed the fault manuals into the Chroma vector store.

Reads ``data/manuals/*.md`` and (re)builds the store at ``rag.vector_store_path``
from ``config/project.yaml``. Safe to re-run after editing a manual: the
collection is rebuilt from scratch. The first run downloads the embedding model
unless it is already cached.

Examples:
    python scripts/ingest_manuals.py
    python scripts/ingest_manuals.py --query "motor running hot"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/ingest_manuals.py` (not just `python -m scripts...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.ingest import ingest_manuals  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manuals-dir",
        type=Path,
        help="folder of markdown manuals (default: rag.manuals_dir)",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        help="Chroma store folder (default: rag.vector_store_path)",
    )
    parser.add_argument(
        "--query",
        help="after ingesting, run this query and print the top results",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the vector store; returns a process exit code."""
    args = parse_args(argv)

    count = ingest_manuals(manuals_dir=args.manuals_dir, store_path=args.store_path)
    if count == 0:
        print("No manual chunks found; nothing ingested.", file=sys.stderr)
        return 1
    print(f"Ingested {count} chunks into the vector store")

    if args.query:
        print(f"\nTop results for {args.query!r}:")
        for hit in retrieve(args.query, store_path=args.store_path):
            print(f"  {hit.score:.3f}  {hit.source} > {hit.section}  [{hit.chunk_id}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
