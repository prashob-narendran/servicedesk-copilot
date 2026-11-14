#!/bin/sh
# Builds the RAG index on first start if the mounted index folder is empty, then runs the given command.
set -e
if [ "${SKIP_INGEST:-false}" != "true" ] && [ ! -f "${RAG_INDEX_DIR:-rag/index}/chroma.sqlite3" ]; then
  echo "Building the knowledge-base index ..."
  if [ -f data/processed/kb_tickets.csv ]; then
    python -m rag.ingest --tickets data/processed/kb_tickets.csv
  else
    python -m rag.ingest
  fi
fi
exec "$@"
