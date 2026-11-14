"""Sub-task 1: PII redaction and named-entity recognition.

Hybrid design, as used by tools such as Microsoft Presidio:
  * Regex recognisers catch structured PII (emails, phones, IPs, URLs, account numbers) with high precision.
  * GLiNER, a zero-shot NER model, finds people, organisations and products.
PII entities are replaced with numbered masks such as [PERSON_1]; ORG and PRODUCT are reported but kept,
because the agent needs them to route the ticket.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PII_TYPES = {"PERSON", "EMAIL", "PHONE", "ACCOUNT_NUMBER", "IP_ADDRESS", "URL"}

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+")
_IP = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_ACCOUNT = re.compile(
    r"(?i)\b(?:account|acct|a/c|customer\s+id|client\s+id|policy|membership)\s*"
    r"(?:no\.?|number|num|#|id)?\s*[:#-]?\s*(?P<value>[A-Z0-9][A-Z0-9-]{5,33})\b"
)
_PHONE = re.compile(r"(?<![\w/.])\+?\d[\d\s().-]{7,18}\d(?![\w/])")
_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")

GLINER_LABELS = {"person": "PERSON", "organization": "ORG", "product": "PRODUCT"}


@dataclass
class Span:
    type: str
    start: int
    end: int
    text: str
    score: float


def _regex_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _EMAIL.finditer(text):
        spans.append(Span("EMAIL", m.start(), m.end(), m.group(), 0.99))
    for m in _URL.finditer(text):
        spans.append(Span("URL", m.start(), m.end(), m.group(), 0.98))
    for m in _IP.finditer(text):
        spans.append(Span("IP_ADDRESS", m.start(), m.end(), m.group(), 0.97))
    for m in _IBAN.finditer(text):
        spans.append(Span("ACCOUNT_NUMBER", m.start(), m.end(), m.group(), 0.95))
    for m in _ACCOUNT.finditer(text):
        value = m.group("value")
        if any(ch.isdigit() for ch in value):
            spans.append(Span("ACCOUNT_NUMBER", m.start("value"), m.end("value"), value, 0.9))
    for m in _PHONE.finditer(text):
        raw = m.group().strip()
        digits = re.sub(r"\D", "", raw)
        if 9 <= len(digits) <= 15 and not _DATE.match(raw):
            spans.append(Span("PHONE", m.start(), m.start() + len(raw), raw, 0.9))
    return spans


def _merge(spans: list[Span]) -> list[Span]:
    """Keep the first-found span when two overlap (regex spans are listed before model spans)."""
    kept: list[Span] = []
    for s in spans:
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def _repeat_occurrences(text: str, spans: list[Span]) -> list[Span]:
    """A name or email detected once is masked everywhere it appears, even where the model missed it."""
    extra: list[Span] = []
    for s in spans:
        if s.type not in PII_TYPES or len(s.text) < 3:
            continue
        for m in re.finditer(re.escape(s.text), text):
            if m.start() != s.start:
                extra.append(Span(s.type, m.start(), m.end(), m.group(), s.score))
    return extra


class Redactor:
    def __init__(self, ner_model=None, threshold: float = 0.5, model_name: str = "regex"):
        """ner_model: any object with GLiNER's predict_entities(text, labels, threshold=...) method."""
        self.ner_model = ner_model
        self.threshold = threshold
        self.model_name = model_name

    @classmethod
    def from_settings(cls, settings) -> "Redactor":
        if not settings.ner_enabled:
            return cls(None, model_name="regex")
        from gliner import GLiNER

        model = GLiNER.from_pretrained(settings.ner_model_id)
        return cls(model, threshold=settings.ner_threshold,
                   model_name=f"regex + {settings.ner_model_id.split('/')[-1]}")

    def _model_spans(self, text: str) -> list[Span]:
        if self.ner_model is None:
            return []
        preds = self.ner_model.predict_entities(text, list(GLINER_LABELS), threshold=self.threshold)
        out = []
        for p in preds:
            etype = GLINER_LABELS.get(str(p["label"]).lower())
            if etype:
                out.append(Span(etype, int(p["start"]), int(p["end"]), p["text"], round(float(p["score"]), 3)))
        return out

    def redact(self, text: str) -> dict:
        found = _regex_spans(text) + self._model_spans(text)
        spans = _merge(found + _repeat_occurrences(text, found))
        masks: dict[tuple[str, str], str] = {}
        counters: dict[str, int] = {}
        entities = []
        for s in spans:
            masked_as = None
            if s.type in PII_TYPES:
                key = (s.type, s.text.lower())
                if key not in masks:
                    counters[s.type] = counters.get(s.type, 0) + 1
                    masks[key] = f"[{s.type}_{counters[s.type]}]"
                masked_as = masks[key]
            entities.append({"type": s.type, "text": s.text, "start": s.start, "end": s.end,
                             "score": s.score, "masked_as": masked_as})
        redacted = text
        for e in sorted(entities, key=lambda e: e["start"], reverse=True):
            if e["masked_as"]:
                redacted = redacted[: e["start"]] + e["masked_as"] + redacted[e["end"]:]
        return {"redacted_text": redacted, "entities": entities}
