"""Request and response models. These are the shapes in the API contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal["PERSON", "EMAIL", "PHONE", "ACCOUNT_NUMBER", "IP_ADDRESS", "URL", "ORG", "PRODUCT"]


# ---------- shared ----------
class Meta(BaseModel):
    request_id: str
    model: str
    model_version: str = "v1"
    latency_ms: int


class Ticket(BaseModel):
    ticket_id: str | None = Field(default=None, max_length=64)
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=5000)
    language: Literal["en"] = "en"

    def text(self) -> str:
        """Model input format used in training: subject + blank line + body."""
        return f"{self.subject}\n\n{self.body}".strip()


class Entity(BaseModel):
    type: EntityType
    text: str
    start: int
    end: int
    score: float
    masked_as: str | None = None


class LabelScore(BaseModel):
    label: str
    score: float


class Source(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    score: float
    snippet: str
    url: str | None = None


# ---------- /redact ----------
class RedactRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class RedactResponse(BaseModel):
    redacted_text: str
    entities: list[Entity]
    meta: Meta


# ---------- /classify ----------
class ClassifyRequest(Ticket):
    top_k: int = Field(default=3, ge=1, le=10)


class ClassifyResponse(BaseModel):
    ticket_id: str | None
    queue: str
    confidence: float
    top_k: list[LabelScore]
    needs_human_review: bool
    meta: Meta


# ---------- /priority ----------
class PriorityResponse(BaseModel):
    ticket_id: str | None
    priority: Literal["low", "medium", "high"]
    priority_scores: list[LabelScore]
    sentiment: Literal["negative", "neutral", "positive"]
    sentiment_score: float
    meta: Meta


# ---------- /summarise ----------
class SummariseRequest(BaseModel):
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=5000)
    max_words: int = Field(default=25, ge=10, le=60)


class SummariseResponse(BaseModel):
    summary: str
    meta: Meta


# ---------- /chat ----------
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    query: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)
    top_k: int = Field(default=4, ge=1, le=10)


class ChatResponse(BaseModel):
    session_id: str | None
    answer: str
    sources: list[Source]
    grounded: bool
    meta: Meta


# ---------- /triage ----------
class TriageRequest(Ticket):
    draft_reply: bool = True


class QueueResult(BaseModel):
    label: str
    confidence: float
    needs_human_review: bool


class DraftReply(BaseModel):
    text: str
    sources: list[Source]
    grounded: bool


class TriageResponse(BaseModel):
    ticket_id: str | None
    redacted_text: str | None
    entities: list[Entity] | None
    queue: QueueResult | None
    priority: LabelScore | None
    sentiment: LabelScore | None
    summary: str | None
    draft_reply: DraftReply | None
    step_latency_ms: dict[str, int]
    warnings: list[str] = Field(default_factory=list)
    meta: Meta


# ---------- /health ----------
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, str]
    version: str
