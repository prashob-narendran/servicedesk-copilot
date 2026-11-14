"""Sub-task 3: priority and customer-sentiment detection.

Priority comes from the fine-tuned priority classifier when it exists (trained with the same recipe as the
queue classifier). If it has not been trained, the LLM classifies priority instead, so the app still works.
Sentiment uses a pre-trained RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment-latest).
"""
from __future__ import annotations

PRIORITIES = ["high", "medium", "low"]
SENTIMENTS = ["negative", "neutral", "positive"]
_SENTIMENT_ALIASES = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}

_PRIORITY_SYSTEM = (
    "You triage IT service-desk tickets. Classify the ticket's priority as exactly one of: high, medium, low. "
    "high = outage, security incident, data loss, many users or revenue blocked; "
    "medium = one user or team blocked with a workaround, or a time-bound request; "
    "low = questions, feedback, how-to and routine requests. "
    'Reply as JSON: {"high": p, "medium": p, "low": p} with probabilities that sum to 1.'
)


class PriorityService:
    def __init__(self, priority_clf=None, sentiment_pipe=None, llm=None,
                 priority_name: str = "llm", sentiment_name: str = "sentiment"):
        self.priority_clf = priority_clf
        self.sentiment_pipe = sentiment_pipe
        self.llm = llm
        self.model_name = f"{priority_name} + {sentiment_name}"

    @classmethod
    def from_settings(cls, settings, llm) -> "PriorityService":
        from pathlib import Path

        from transformers import pipeline

        from classifier.predict import TextClassifier

        clf, pname = None, f"llm:{settings.llm_model}"
        pid = settings.priority_model_id
        is_local = pid and pid.startswith(("models/", "./", "/"))
        if pid and (Path(pid).exists() or not is_local):  # existing local folder, or a Hub repo id
            clf, pname = TextClassifier(pid, device=settings.device), pid.rstrip("/").split("/")[-1]
        sent = pipeline("text-classification", model=settings.sentiment_model_id, top_k=None,
                        device=0 if settings.device == "cuda" else -1)
        return cls(clf, sent, llm, pname, settings.sentiment_model_id.split("/")[-1])

    # ---- priority ----
    def priority(self, text: str, redacted_text: str | None = None) -> list[dict]:
        if self.priority_clf is not None:
            scores = self.priority_clf.predict([text], top_k=3)[0]
            return [{"label": s["label"].lower(), "score": s["score"]} for s in scores]
        if self.llm is None:
            raise RuntimeError("no priority model and no LLM configured")
        if redacted_text is None:  # never send unredacted text to the external LLM
            raise RuntimeError("LLM priority fallback needs PII-redacted text")
        data = self.llm.complete_json(_PRIORITY_SYSTEM, f"Ticket:\n{redacted_text}", max_tokens=60)
        raw = {}
        for p in PRIORITIES:
            try:
                raw[p] = max(0.0, float(data.get(p, 0) or 0))
            except (TypeError, ValueError):
                raw[p] = 0.0
        total = sum(raw.values())
        if total == 0:  # unparseable answer: fall back to a neutral default
            raw, total = {"medium": 1.0, "high": 0.0, "low": 0.0}, 1.0
        scores = [{"label": p, "score": round(v / total, 4)} for p, v in raw.items()]
        return sorted(scores, key=lambda s: s["score"], reverse=True)

    # ---- sentiment ----
    def sentiment(self, text: str) -> dict:
        if self.sentiment_pipe is None:
            raise RuntimeError("sentiment model not loaded")
        out = self.sentiment_pipe(text, truncation=True, max_length=512)
        rows = out[0] if out and isinstance(out[0], list) else out
        best = max(rows, key=lambda r: r["score"])
        label = str(best["label"]).lower()
        label = _SENTIMENT_ALIASES.get(label, label)
        return {"label": label if label in SENTIMENTS else "neutral", "score": round(float(best["score"]), 4)}
