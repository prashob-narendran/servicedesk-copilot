"""Evaluate the RAG chatbot on rag/eval/questions.jsonl.

    python -m rag.eval.run_eval                 # full evaluation (needs the LLM key; about 30–45 min on the free Groq tier)
    python -m rag.eval.run_eval --retrieval-only  # free: retrieval hit rate + threshold calibration only

Metrics (definitions follow the RAGAS framework, implemented directly with an LLM judge so the
evaluation does not depend on fast-changing RAGAS / LangChain package versions):
  * faithfulness       share of the answer's claims that are supported by the retrieved context
  * answer_relevancy   mean cosine similarity between the question and 3 questions the judge
                       reverse-generates from the answer (RAGAS method)
  * context_precision  average precision of the retrieved chunks, judged relevant against the ground truth
  * hit_rate@k         the expected document is among the top-k retrieved chunks
  * refusal accuracy   out-of-scope questions get the "not covered" reply; in-scope ones get an answer
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.llm import LLMClient
from rag.pipeline import RAGPipeline

QUESTIONS = Path("rag/eval/questions.jsonl")
OUT_DIR = Path("reports/metrics")

_FAITH = ("You check answers for hallucination. Split the ANSWER into atomic factual claims and decide for "
          "each whether it is directly supported by the CONTEXT. Ignore citation markers like [1]. Reply as JSON: "
          '{"claims": <number of claims>, "supported": <number supported>}')
_REL = ("Write 3 different questions that the following answer would be a good answer to. "
        'Reply as JSON: {"questions": ["...", "...", "..."]}')
_PREC = ("Decide whether the CONTEXT chunk is useful for producing the REFERENCE answer to the QUESTION. "
         'Reply as JSON: {"relevant": true} or {"relevant": false}')


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def average_precision(rels: list[bool]) -> float:
    hits, score = 0, 0.0
    for k, r in enumerate(rels, 1):
        if r:
            hits += 1
            score += hits / k
    return score / hits if hits else 0.0


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--top-k", type=int, default=s.rag_top_k)
    args = ap.parse_args()

    llm = LLMClient(s)
    rag = RAGPipeline(index_dir=s.rag_index_dir, score_threshold=s.rag_score_threshold, llm=llm,
                      collection_name=s.rag_collection, embedding_model_id=s.embedding_model_id, device=s.device)
    if rag.count() == 0:
        raise SystemExit("The index is empty. Run: python -m rag.ingest")
    items = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    rows = []
    for it in items:
        q, in_scope = it["question"], it["expected_doc_id"] is not None
        hits = rag.retrieve(q, args.top_k)
        row = {"id": it["id"], "question": q, "in_scope": in_scope,
               "top1_score": hits[0]["score"] if hits else 0.0,
               "hit": (it["expected_doc_id"] in {h["doc_id"] for h in hits}) if in_scope else None}
        if not args.retrieval_only:
            t0 = time.time()
            res = rag.answer(q, top_k=args.top_k)
            row["latency_ms"] = int((time.time() - t0) * 1000)
            row["grounded"] = res["grounded"]
            row["answer"] = res["answer"]
            row["refusal_correct"] = res["grounded"] == in_scope
            if in_scope and res["grounded"]:
                ctx_hits = [h for h in hits if h["score"] >= rag.score_threshold]
                context = "\n\n".join(h["text"] for h in ctx_hits)
                f = llm.complete_json(_FAITH, f"CONTEXT:\n{context}\n\nANSWER:\n{res['answer']}")
                claims = max(int(f.get("claims", 0) or 0), 1)
                row["faithfulness"] = round(min(int(f.get("supported", 0) or 0), claims) / claims, 3)
                gen = llm.complete_json(_REL, res["answer"]).get("questions", [])[:3]
                if gen:
                    vecs = rag.embedder.encode([q] + gen, is_query=True)
                    row["answer_relevancy"] = round(statistics.mean(cosine(vecs[0], v) for v in vecs[1:]), 3)
                rels = [bool(llm.complete_json(
                    _PREC, f"QUESTION: {q}\nREFERENCE: {it['ground_truth']}\nCONTEXT:\n{h['text']}").get("relevant"))
                    for h in ctx_hits]
                row["context_precision"] = round(average_precision(rels), 3)
        rows.append(row)
        print(f"{it['id']}: top1={row['top1_score']:.3f} hit={row['hit']} grounded={row.get('grounded')}")

    in_rows = [r for r in rows if r["in_scope"]]
    out_rows = [r for r in rows if not r["in_scope"]]

    def mean(key, rs):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    in_min = min(r["top1_score"] for r in in_rows)
    out_max = max(r["top1_score"] for r in out_rows) if out_rows else 0.0
    summary = {
        "questions": len(rows), "in_scope": len(in_rows), "out_of_scope": len(out_rows), "top_k": args.top_k,
        "score_threshold": rag.score_threshold, "embedding_model": s.embedding_model_id, "llm": f"{llm.provider}:{llm.model}",
        f"hit_rate@{args.top_k}": round(sum(bool(r["hit"]) for r in in_rows) / len(in_rows), 3),
        "top1_score_in_scope_min": in_min, "top1_score_out_of_scope_max": out_max,
        "suggested_threshold": round((in_min + out_max) / 2, 3) if in_min > out_max else None,
    }
    if not args.retrieval_only:
        summary.update({
            "faithfulness": mean("faithfulness", in_rows),
            "answer_relevancy": mean("answer_relevancy", in_rows),
            "context_precision": mean("context_precision", in_rows),
            "answered_in_scope": round(sum(bool(r["grounded"]) for r in in_rows) / len(in_rows), 3),
            "refused_out_of_scope": round(sum(not r["grounded"] for r in out_rows) / max(len(out_rows), 1), 3),
            "latency_ms_median": int(statistics.median(r["latency_ms"] for r in rows)),
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rag_eval.json").write_text(json.dumps(summary, indent=2))
    with (OUT_DIR / "rag_eval_details.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))
    if summary["suggested_threshold"] is None:
        print("Note: in-scope and out-of-scope top-1 scores overlap; review rag_eval_details.csv before "
              "changing RAG_SCORE_THRESHOLD.")


if __name__ == "__main__":
    main()
