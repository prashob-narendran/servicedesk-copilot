"""Part III: retrieval-augmented generation over the TechServ knowledge base.

Hand-over interface from the API contract:

    rag = RAGPipeline(index_dir="rag/index")
    rag.ingest(["rag/docs"])                       -> {"documents": n, "chunks": m}
    rag.answer("What is the SLA for a P1?")        -> {"answer": str, "sources": [...], "grounded": bool}

Design choices
  * Chunking: markdown headers first (keeps sections intact), then a recursive splitter of ~1,600 characters
    (about 400 tokens) with 200 characters of overlap. Each past ticket is one chunk.
  * Embeddings: BAAI/bge-small-en-v1.5, normalised; queries get bge's retrieval instruction prefix.
  * Vector store: Chroma, persisted to disk, cosine similarity.
  * Guardrail: if no chunk scores above the similarity threshold, the bot says the knowledge base does not
    cover the question and the LLM is never called, so it cannot answer from general knowledge.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

NOT_COVERED = ("The knowledge base doesn't cover this question. Please escalate to a senior agent "
               "or check with the relevant team.")
_NOT_COVERED_TOKEN = "NOT_COVERED"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_CHAT_SYSTEM = (
    "You are the TechServ service-desk assistant for support agents. Answer ONLY from the numbered context "
    "passages. Cite every fact with its passage number in square brackets, e.g. [1] or [2][3]. Be concise and "
    "practical: steps as a short numbered list when the question asks how to do something. If the context does "
    f"not contain the answer, reply with exactly {_NOT_COVERED_TOKEN}. Keep masks such as [PERSON_1] as written."
)
_REPLY_SYSTEM = (
    "You are a TechServ support agent drafting a reply to a customer ticket. Use ONLY the numbered context "
    "passages (company policies, procedures and similar past tickets) for facts, steps, timelines and "
    "commitments, and cite them with [n]. Write a polite, specific reply of at most 150 words: acknowledge the "
    "issue, give the next steps or the resolution, and set expectations (for example the SLA). Address the "
    "customer with the mask from the ticket if there is one (e.g. [PERSON_1]); never invent names, numbers or "
    f"promises. If the context gives nothing useful for this ticket, reply with exactly {_NOT_COVERED_TOKEN}."
)


# ---------------------------------------------------------------- embeddings
class SentenceTransformerEmbedder:
    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)
        self.query_prefix = _BGE_QUERY_PREFIX if "bge" in model_id.lower() else ""

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------- loading documents
def _parse_front_matter(text: str) -> tuple[dict, str]:
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4 :].lstrip()
    return meta, text


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "doc"


def load_documents(paths: list[str]) -> list[dict]:
    """Return documents as dicts: {doc_id, title, url, text, kind}. Supports .md, .txt, .pdf and ticket .csv."""
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p])
    docs: list[dict] = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix in {".md", ".txt"}:
            meta, body = _parse_front_matter(f.read_text(encoding="utf-8"))
            title = meta.get("title") or next(
                (ln.lstrip("# ").strip() for ln in body.splitlines() if ln.startswith("# ")), f.stem)
            docs.append({"doc_id": meta.get("doc_id", f.stem), "title": title, "url": meta.get("source_url"),
                         "text": body, "kind": "markdown"})
        elif suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(f))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            docs.append({"doc_id": f.stem, "title": f.stem.replace("_", " ").replace("-", " "), "url": None,
                         "text": text, "kind": "text"})
        elif suffix == ".csv":
            with f.open(encoding="utf-8") as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    answer = (row.get("answer") or "").strip()
                    body = (row.get("body") or "").strip()
                    if not answer or not body:
                        continue
                    subject = (row.get("subject") or "").strip() or "Support ticket"
                    queue = (row.get("queue") or "").strip()
                    tid = row.get("ticket_id") or f"{i:05d}"
                    text = (f"Past ticket ({queue})\nSubject: {subject}\nCustomer: {body}\n"
                            f"Resolution given by agent: {answer}")
                    docs.append({"doc_id": f"ticket-{tid}", "title": f"Past ticket: {subject[:80]}", "url": None,
                                 "text": text, "kind": "ticket"})
    return docs


def chunk_document(doc: dict, chunk_size: int = 1600, overlap: int = 200) -> list[dict]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    if doc["kind"] == "ticket":
        pieces = [(doc["text"], "")]
    else:
        pieces = []
        if doc["kind"] == "markdown":
            header_split = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
            sections = header_split.split_text(doc["text"])
        else:
            sections = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        if sections:
            for sec in sections:
                section_name = sec.metadata.get("h3") or sec.metadata.get("h2") or sec.metadata.get("h1") or ""
                for piece in splitter.split_text(sec.page_content):
                    pieces.append((piece, section_name))
        else:
            pieces = [(p, "") for p in splitter.split_text(doc["text"])]
    chunks = []
    for i, (text, section) in enumerate(pieces):
        if not text.strip():
            continue
        chunks.append({
            "id": f"{doc['doc_id']}#{i}",
            "text": text.strip(),
            "metadata": {"doc_id": doc["doc_id"], "title": doc["title"], "section": section,
                         "url": doc["url"] or "", "kind": doc["kind"]},
        })
    return chunks


# ---------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(self, index_dir: str = "rag/index", llm_model: str | None = None, score_threshold: float = 0.5,
                 *, embedder=None, llm=None, collection_name: str = "techserv_kb",
                 embedding_model_id: str = "BAAI/bge-small-en-v1.5", device: str = "cpu", default_top_k: int = 4):
        """embedder: object with encode(texts, is_query) -> vectors. llm: object with complete(system, user, max_tokens)."""
        import chromadb

        self.index_dir = index_dir
        self.score_threshold = score_threshold
        self.default_top_k = default_top_k
        self.embedder = embedder or SentenceTransformerEmbedder(embedding_model_id, device=device)
        if llm is None:
            from app.config import get_settings
            from app.llm import LLMClient

            settings = get_settings()
            if llm_model:
                settings = settings.model_copy(update={"llm_model": llm_model})
            llm = LLMClient(settings)
        self.llm = llm
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=index_dir)
        self.collection_name = collection_name
        self.collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    @property
    def model_name(self) -> str:
        emb = getattr(self.embedder, "model_id", "embedder").split("/")[-1]
        return f"rag: {emb} + chroma + {getattr(self.llm, 'model', 'llm')}"

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self.collection = self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    # ---- ingestion ----
    def ingest(self, paths: list[str], batch_size: int = 64) -> dict:
        docs = load_documents(paths)
        chunks = [c for d in docs for c in chunk_document(d)]
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
                embeddings=self.embedder.encode([c["text"] for c in batch]),
            )
        return {"documents": len(docs), "chunks": len(chunks)}

    # ---- retrieval ----
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.default_top_k
        if self.count() == 0:
            return []
        qvec = self.embedder.encode([query], is_query=True)
        res = self.collection.query(query_embeddings=qvec, n_results=min(top_k * 3, self.count()),
                                    include=["documents", "metadatas", "distances"])
        hits, seen = [], set()
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            fingerprint = hashlib.md5(doc[:300].encode()).hexdigest()
            if fingerprint in seen:  # drop near-identical chunks (e.g. duplicated tickets)
                continue
            seen.add(fingerprint)
            hits.append({"doc_id": meta.get("doc_id", ""), "title": meta.get("title", ""), "chunk_id": cid,
                         "score": round(1.0 - float(dist), 4), "text": doc, "url": meta.get("url") or None,
                         "section": meta.get("section", "")})
            if len(hits) == top_k:
                break
        return hits

    # ---- generation ----
    def answer(self, query: str, history: list[dict] | None = None, top_k: int | None = None,
               context: str | None = None) -> dict:
        """query must already be PII-redacted. context (summary + queue) switches to reply-drafting mode."""
        retrieval_query = f"{context}\n{query}" if context else query
        hits = [h for h in self.retrieve(retrieval_query, top_k) if h["score"] >= self.score_threshold]
        if not hits:
            return {"answer": NOT_COVERED, "sources": [], "grounded": False}

        passages = "\n\n".join(
            f"[{i}] {h['title']}{' — ' + h['section'] if h['section'] and h['section'] != h['title'] else ''}\n{h['text']}"
            for i, h in enumerate(hits, 1))
        if context:
            system = _REPLY_SYSTEM
            user = f"Context passages:\n{passages}\n\nTicket triage notes: {context}\n\nCustomer ticket:\n{query}"
            max_tokens = 350
        else:
            system = _CHAT_SYSTEM
            convo = "".join(f"{t['role'].capitalize()}: {t['content']}\n" for t in (history or [])[-6:])
            user = (f"Context passages:\n{passages}\n\n"
                    + (f"Conversation so far:\n{convo}\n" if convo else "")
                    + f"Question: {query}")
            max_tokens = 450
        text = self.llm.complete(system, user, max_tokens=max_tokens).strip()
        if not text or text.startswith(_NOT_COVERED_TOKEN):
            return {"answer": NOT_COVERED, "sources": [], "grounded": False}

        # All retrieved passages are returned in order, so citation [n] always points to sources[n-1].
        sources = [{"doc_id": h["doc_id"], "title": h["title"], "chunk_id": h["chunk_id"], "score": h["score"],
                    "snippet": _snippet(h["text"]), "url": h["url"]} for h in hits]
        return {"answer": text, "sources": sources, "grounded": True}


def _snippet(text: str, n: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"
