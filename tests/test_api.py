TICKET = {
    "ticket_id": "T-1042",
    "subject": "VPN not connecting",
    "body": "Hi, this is the finance team. Since the update the VPN drops every 5 minutes for 12 people. "
            "Call +971 50 123 4567 or mail anita.rao@acme.in",
}


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["components"]["classifier"] == "loaded"
    assert body["components"]["vector_index"].startswith("loaded (")
    assert r.headers["X-Request-ID"]


def test_redact(client):
    r = client.post("/api/v1/redact", json={"text": TICKET["body"]})
    assert r.status_code == 200
    body = r.json()
    assert "anita.rao@acme.in" not in body["redacted_text"]
    assert {e["type"] for e in body["entities"]} >= {"EMAIL", "PHONE"}
    assert set(body["meta"]) == {"request_id", "model", "model_version", "latency_ms"}


def test_classify(client):
    r = client.post("/api/v1/classify", json={**TICKET, "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["queue"] == "Technical Support" and len(body["top_k"]) == 2
    assert body["needs_human_review"] is False and body["ticket_id"] == "T-1042"


def test_low_confidence_is_flagged(client, services):
    services.get("classifier").conf = 0.4
    body = client.post("/api/v1/classify", json=TICKET).json()
    assert body["needs_human_review"] is True


def test_priority(client):
    body = client.post("/api/v1/priority", json=TICKET).json()
    assert body["priority"] == "high" and body["sentiment"] == "negative"
    assert abs(sum(s["score"] for s in body["priority_scores"]) - 1) < 0.01


def test_summarise_sends_only_redacted_text(client, fake_llm):
    r = client.post("/api/v1/summarise", json={"subject": TICKET["subject"], "body": TICKET["body"], "max_words": 12})
    assert r.status_code == 200
    assert len(r.json()["summary"].split()) <= 12
    sent = fake_llm.calls[-1][1]
    assert "anita.rao@acme.in" not in sent and "[EMAIL_1]" in sent


def test_triage_end_to_end(client, fake_llm):
    r = client.post("/api/v1/triage", json=TICKET)
    assert r.status_code == 200
    body = r.json()
    assert body["warnings"] == []
    assert body["queue"]["label"] == "Technical Support"
    assert body["priority"]["label"] == "high"
    assert body["sentiment"]["label"] == "negative"
    assert body["summary"] and body["draft_reply"]["grounded"] is True
    assert body["draft_reply"]["sources"][0]["doc_id"] == "sop-vpn"
    assert set(body["step_latency_ms"]) == {"redact", "classify", "priority", "sentiment", "summarise", "draft_reply"}
    assert all("anita.rao@acme.in" not in user for _, user in fake_llm.calls)


def test_triage_survives_a_failing_step(client, services):
    class Broken:
        def predict(self, *a, **k):
            raise RuntimeError("model crashed")
    services._objs["classifier"] = Broken()
    body = client.post("/api/v1/triage", json=TICKET).json()
    assert body["queue"] is None
    assert any(w.startswith("classify failed") for w in body["warnings"])
    assert body["summary"] and body["draft_reply"]


def test_chat_grounded_and_refusal(client, services):
    body = client.post("/api/v1/chat", json={"query": "VPN disconnects every few minutes, adapter power saving"}).json()
    assert body["grounded"] is True and body["sources"]
    services.get("rag").score_threshold = 0.95
    body = client.post("/api/v1/chat", json={"query": "Who won the world cup?"}).json()
    assert body["grounded"] is False and body["sources"] == []


def test_validation_errors(client):
    r = client.post("/api/v1/classify", json={"subject": "x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = client.post("/api/v1/classify", json={"body": "x", "language": "de"})
    assert r.status_code == 422
    r = client.post("/api/v1/redact", json={"text": "a" * 5001})
    assert r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    r = client.post("/api/v1/redact", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_REQUEST"


def test_api_key_enforced(client, services):
    services.settings.api_key = "secret"
    try:
        r = client.get("/api/v1/health")
        assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
        assert client.get("/api/v1/health", headers={"X-API-Key": "secret"}).status_code == 200
    finally:
        services.settings.api_key = None


def test_model_not_ready_returns_503(client, services):
    services._objs.pop("classifier")
    services.settings.classifier_model_id = "models/does-not-exist"
    r = client.post("/api/v1/classify", json=TICKET)
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_READY"
    assert client.get("/api/v1/health").json()["status"] == "degraded"
