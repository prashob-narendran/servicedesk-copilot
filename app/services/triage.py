"""The /triage pipeline: redact -> classify -> priority + sentiment -> summarise -> draft reply (RAG).

Each step is timed and isolated: if one step fails, its field is null, a warning names it, and the
remaining steps still run, so one broken module never breaks the whole demo.
Local models see the original text; anything sent to the external LLM uses the redacted text.
"""
from __future__ import annotations

import logging
import time

from app.container import Services
from app.schemas import TriageRequest

log = logging.getLogger("copilot")

# How the classifier's labels map to the SLA policy's levels (see rag/docs/02-sla-and-escalation-policy.md)
SLA_LEVEL = {"high": "P1 or P2", "medium": "P3", "low": "P4"}


def run_triage(req: TriageRequest, svc: Services) -> dict:
    out: dict = {"ticket_id": req.ticket_id, "redacted_text": None, "entities": None, "queue": None,
                 "priority": None, "sentiment": None, "summary": None, "draft_reply": None}
    latency: dict[str, int] = {}
    warnings: list[str] = []
    text = req.text()

    def step(name, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as e:
            log.warning("triage step %s failed: %s", name, e)
            warnings.append(f"{name} failed: {getattr(e, 'message', str(e))[:200]}")
            return None
        finally:
            latency[name] = int((time.perf_counter() - t0) * 1000)

    # 1. PII redaction + NER (no LLM may see text before this succeeds)
    red = step("redact", lambda: svc.redactor.redact(text))
    if red:
        out["redacted_text"], out["entities"] = red["redacted_text"], red["entities"]
    safe_text = red["redacted_text"] if red else None  # subject + body, PII masked

    # 2. Queue classification (fine-tuned model)
    preds = step("classify", lambda: svc.classifier.predict([text], top_k=3)[0])
    if preds:
        conf = preds[0]["score"]
        out["queue"] = {"label": preds[0]["label"], "confidence": conf,
                        "needs_human_review": conf < svc.settings.review_threshold}

    # 3. Priority + sentiment
    pr = step("priority", lambda: svc.priority.priority(text, redacted_text=safe_text))
    if pr:
        out["priority"] = pr[0]
    se = step("sentiment", lambda: svc.priority.sentiment(text))
    if se:
        out["sentiment"] = se

    if safe_text is None:
        warnings.append("summarise and draft_reply skipped: text could not be redacted, so it was not sent "
                        "to the external LLM")
        out.update(step_latency_ms=latency, warnings=warnings)
        return out

    # 4. Summary (LLM on redacted text)
    summary = step("summarise", lambda: svc.summariser.summarise("", safe_text, 25))
    out["summary"] = summary

    # 5. Grounded reply draft (RAG on redacted text)
    if req.draft_reply:
        notes = "; ".join(filter(None, [
            f"summary: {summary}" if summary else None,
            f"queue: {out['queue']['label']}" if out["queue"] else None,
            (f"priority: {out['priority']['label']} ({SLA_LEVEL.get(out['priority']['label'], '')} in the SLA policy)"
             if out["priority"] else None),
        ])) or "customer support ticket"
        reply = step("draft_reply", lambda: svc.rag.answer(safe_text, context=notes))
        if reply:
            out["draft_reply"] = {"text": reply["answer"], "sources": reply["sources"], "grounded": reply["grounded"]}

    out.update(step_latency_ms=latency, warnings=warnings)
    return out
