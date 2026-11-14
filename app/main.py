"""AI Service Desk Copilot — FastAPI service.

Run:  uvicorn app.main:app --reload        Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.container import Services
from app.errors import ServiceError
from app.schemas import (ChatRequest, ChatResponse, ClassifyRequest, ClassifyResponse, HealthResponse, Meta,
                         PriorityResponse, RedactRequest, RedactResponse, SummariseRequest, SummariseResponse,
                         Ticket, TriageRequest, TriageResponse)
from app.services.triage import run_triage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("copilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not hasattr(app.state, "services"):  # tests may inject their own
        app.state.services = Services(settings)
    if settings.preload_models:
        log.info("preloading models ...")
        await run_in_threadpool(app.state.services.preload)
    yield


app = FastAPI(
    title="AI Service Desk Copilot API",
    version=get_settings().version,
    description="Ticket triage (PII redaction, queue routing, priority, sentiment, summary) and a "
                "knowledge-base chatbot for customer-support and IT service-desk teams. Group 60, AIMLZG549.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- middleware and errors
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "request_id": rid}},
                        headers={"X-Request-ID": rid})


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    return _error(request, exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errs = exc.errors()
    if any(e.get("type") == "json_invalid" for e in errs):
        return _error(request, 400, "BAD_REQUEST", "Malformed JSON body")
    first = errs[0] if errs else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    message = f"{field}: {first.get('msg', 'invalid value')}" if field else first.get("msg", "invalid request")
    if any(e.get("type") == "string_too_long" for e in errs):
        return _error(request, 413, "PAYLOAD_TOO_LARGE", message)
    return _error(request, 422, "VALIDATION_ERROR", message)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
    return _error(request, exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled error (request %s)", getattr(request.state, "request_id", "-"))
    return _error(request, 500, "INTERNAL_ERROR", "Unexpected server error; see logs for the request ID")


# ---------------------------------------------------------------- dependencies
def services(request: Request) -> Services:
    return request.app.state.services


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise ServiceError(401, "UNAUTHORIZED", "Missing or invalid X-API-Key header")


def meta(request: Request, model: str) -> Meta:
    return Meta(request_id=request.state.request_id, model=model, model_version="v1",
                latency_ms=int((time.perf_counter() - request.state.t0) * 1000))


router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])


# ---------------------------------------------------------------- endpoints
@router.post("/redact", response_model=RedactResponse, tags=["sub-tasks"],
             summary="Detect entities and mask personal data")
def redact(body: RedactRequest, request: Request, svc: Services = Depends(services)):
    red = svc.redactor.redact(body.text)
    return RedactResponse(**red, meta=meta(request, svc.redactor.model_name))


@router.post("/classify", response_model=ClassifyResponse, tags=["sub-tasks"],
             summary="Route a ticket to a support queue (fine-tuned model)")
def classify(body: ClassifyRequest, request: Request, svc: Services = Depends(services)):
    preds = svc.classifier.predict([body.text()], top_k=body.top_k)[0]
    conf = preds[0]["score"]
    return ClassifyResponse(ticket_id=body.ticket_id, queue=preds[0]["label"], confidence=conf, top_k=preds,
                            needs_human_review=conf < svc.settings.review_threshold,
                            meta=meta(request, getattr(svc.classifier, "model_id", "queue-classifier")))


@router.post("/priority", response_model=PriorityResponse, tags=["sub-tasks"],
             summary="Predict priority (low/medium/high) and customer sentiment")
def priority(body: Ticket, request: Request, svc: Services = Depends(services)):
    text = body.text()
    redacted = svc.redactor.redact(text)["redacted_text"]
    scores = svc.priority.priority(text, redacted_text=redacted)
    sent = svc.priority.sentiment(text)
    return PriorityResponse(ticket_id=body.ticket_id, priority=scores[0]["label"], priority_scores=scores,
                            sentiment=sent["label"], sentiment_score=sent["score"],
                            meta=meta(request, svc.priority.model_name))


@router.post("/summarise", response_model=SummariseResponse, tags=["sub-tasks"],
             summary="One-line summary for the agent (text is redacted before the LLM call)")
def summarise(body: SummariseRequest, request: Request, svc: Services = Depends(services)):
    subject = svc.redactor.redact(body.subject)["redacted_text"] if body.subject else ""
    text = svc.redactor.redact(body.body)["redacted_text"]
    summary = svc.summariser.summarise(subject, text, body.max_words)
    return SummariseResponse(summary=summary, meta=meta(request, svc.summariser.model_name))


@router.post("/triage", response_model=TriageResponse, tags=["pipeline"],
             summary="Full pipeline: redact → classify → priority → summarise → draft reply")
def triage(body: TriageRequest, request: Request, svc: Services = Depends(services)):
    result = run_triage(body, svc)
    return TriageResponse(**result, meta=meta(request, "pipeline"))


@router.post("/chat", response_model=ChatResponse, tags=["RAG"],
             summary="Answer an agent's question from the knowledge base, with sources")
def chat(body: ChatRequest, request: Request, svc: Services = Depends(services)):
    query = svc.redactor.redact(body.query)["redacted_text"]
    history = [{"role": t.role, "content": svc.redactor.redact(t.content)["redacted_text"]} for t in body.history]
    res = svc.rag.answer(query, history=history, top_k=body.top_k)
    return ChatResponse(session_id=body.session_id, answer=res["answer"], sources=res["sources"],
                        grounded=res["grounded"], meta=meta(request, svc.rag.model_name))


@router.get("/health", response_model=HealthResponse, tags=["ops"], summary="Status of each component")
def health(svc: Services = Depends(services)):
    comps = svc.status()
    bad = any(v.startswith("error") for v in comps.values()) or comps["llm"].startswith("missing")
    return HealthResponse(status="degraded" if bad else "ok", components=comps, version=svc.settings.version)


app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {"name": "AI Service Desk Copilot API", "docs": "/docs", "health": "/api/v1/health"}
