"""Part IV helper: run a member's workplace tickets and questions through the running API.

    python scripts/run_workplace.py workplace/prashob/tickets.json

Writes workplace/<member>/results.md (Analyse table + answers) and results.json.
Add the member's workplace documents to the index first:
    python -m rag.ingest --docs workplace/prashob/docs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

API = os.getenv("API_URL", "http://localhost:8000/api/v1").rstrip("/")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}


def main() -> None:
    path = Path(sys.argv[1])
    spec = json.loads(path.read_text(encoding="utf-8"))
    out = {"member": spec.get("member"), "tickets": [], "questions": []}
    for t in spec["tickets"]:
        r = requests.post(f"{API}/triage", json={"ticket_id": t["id"], "subject": t["subject"], "body": t["body"]},
                          headers=HEADERS, timeout=300)
        r.raise_for_status()
        d = r.json()
        row = {
            "id": t["id"], "subject": t["subject"], "expected_team": t["expected_team"],
            "expected_priority": t["expected_priority"],
            "queue": (d["queue"] or {}).get("label"), "queue_confidence": (d["queue"] or {}).get("confidence"),
            "needs_review": (d["queue"] or {}).get("needs_human_review"),
            "priority": (d["priority"] or {}).get("label"), "sentiment": (d["sentiment"] or {}).get("label"),
            "masked": [e["type"] for e in (d["entities"] or []) if e.get("masked_as")],
            "summary": d.get("summary"),
            "reply_grounded": (d["draft_reply"] or {}).get("grounded"),
            "reply_sources": [s["title"] for s in (d["draft_reply"] or {}).get("sources", [])],
            "reply": (d["draft_reply"] or {}).get("text"),
            "latency_ms": d["meta"]["latency_ms"],
        }
        out["tickets"].append(row)
        print(f"{t['id']}: {row['queue']} ({row['queue_confidence']}) · {row['priority']} · {row['latency_ms']} ms")
    for q in spec["questions"]:
        r = requests.post(f"{API}/chat", json={"query": q}, headers=HEADERS, timeout=300)
        r.raise_for_status()
        d = r.json()
        out["questions"].append({"question": q, "grounded": d["grounded"], "answer": d["answer"],
                                 "sources": [s["title"] for s in d["sources"]], "latency_ms": d["meta"]["latency_ms"]})
        print(f"Q: {q[:60]} · grounded={d['grounded']}")

    md = ["| Ticket | Should go to | Expected priority | Copilot queue (confidence) | Copilot priority | Sentiment | PII masked |",
          "|---|---|---|---|---|---|---|"]
    for t in out["tickets"]:
        conf = f"{t['queue_confidence']:.0%}" if t["queue_confidence"] is not None else "—"
        flag = " ⚠ review" if t["needs_review"] else ""
        md.append(f"| {t['id']} {t['subject']} | {t['expected_team']} | {t['expected_priority']} | {t['queue']} ({conf}){flag} "
                  f"| {t['priority']} | {t['sentiment']} | {', '.join(t['masked']) or '—'} |")
    md.append("\n## Summaries and draft replies\n")
    for t in out["tickets"]:
        md.append(f"**{t['id']}** summary: {t['summary']}\n\nReply grounded: {t['reply_grounded']} · sources: "
                  f"{'; '.join(t['reply_sources'])}\n")
    md.append("\n## Knowledge-base questions\n")
    for q in out["questions"]:
        md.append(f"**Q:** {q['question']}\n\n**A:** {q['answer']}\n\nSources: {'; '.join(q['sources'])}\n")
    folder = path.parent
    (folder / "results.md").write_text("\n".join(md), encoding="utf-8")
    (folder / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {folder / 'results.md'} and {folder / 'results.json'}")


if __name__ == "__main__":
    main()
