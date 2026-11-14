"""Build (or rebuild) the vector index.

    python -m rag.ingest                                   # policies and SOPs in rag/docs
    python -m rag.ingest --tickets data/processed/kb_tickets.csv --reset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.config import get_settings
from rag.pipeline import RAGPipeline


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Index knowledge-base documents into Chroma")
    ap.add_argument("--docs", nargs="*", default=["rag/docs"], help="files or folders (.md, .txt, .pdf)")
    ap.add_argument("--tickets", default=None, help="CSV of past resolved tickets (subject, body, answer, queue)")
    ap.add_argument("--index-dir", default=s.rag_index_dir)
    ap.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = ap.parse_args()

    rag = RAGPipeline(index_dir=args.index_dir, score_threshold=s.rag_score_threshold, llm=object(),
                      collection_name=s.rag_collection, embedding_model_id=s.embedding_model_id, device=s.device)
    if args.reset:
        rag.reset()
    paths = list(args.docs) + ([args.tickets] if args.tickets else [])
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Not found: {missing}")
    t0 = time.time()
    stats = rag.ingest(paths)
    stats.update({"total_chunks_in_index": rag.count(), "seconds": round(time.time() - t0, 1),
                  "embedding_model": s.embedding_model_id})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
