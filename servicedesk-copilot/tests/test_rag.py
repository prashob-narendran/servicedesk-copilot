from rag.pipeline import NOT_COVERED, chunk_document, load_documents


def test_docs_load_with_titles_and_ids():
    docs = load_documents(["rag/docs"])
    ids = {d["doc_id"] for d in docs}
    assert {"sla-policy", "sop-vpn", "billing-policy", "service-catalogue"} <= ids
    assert all(d["title"] for d in docs)


def test_chunks_keep_section_metadata():
    doc = next(d for d in load_documents(["rag/docs"]) if d["doc_id"] == "sla-policy")
    chunks = chunk_document(doc)
    assert len(chunks) >= 3
    assert any(c["metadata"]["section"] == "Escalation" for c in chunks)
    assert all(c["id"].startswith("sla-policy#") for c in chunks)


def test_retrieval_finds_the_right_document(rag):
    hits = rag.retrieve("VPN disconnects every few minutes wifi power saving", top_k=3)
    assert hits[0]["doc_id"] == "sop-vpn"
    hits = rag.retrieve("duplicate charge refund payment method business days", top_k=3)
    assert hits[0]["doc_id"] == "billing-policy"


def test_answer_has_sources_and_is_grounded(rag):
    res = rag.answer("VPN disconnects every few minutes, adapter power saving")
    assert res["grounded"] is True
    assert res["sources"] and res["sources"][0]["doc_id"] == "sop-vpn"
    assert len(res["sources"][0]["snippet"]) <= 300


def test_out_of_scope_question_is_refused_without_llm_call(rag, fake_llm):
    rag.score_threshold = 0.9
    n = len(fake_llm.calls)
    res = rag.answer("Who won the football world cup?")
    assert res == {"answer": NOT_COVERED, "sources": [], "grounded": False}
    assert len(fake_llm.calls) == n


def test_ticket_csv_ingestion(tmp_path, rag):
    csv_path = tmp_path / "kb_tickets.csv"
    csv_path.write_text("ticket_id,subject,body,answer,queue\n"
                        "1,Printer offline,The office printer is offline for everyone,Restarted the print spooler,IT Support\n"
                        "2,No answer,Body only,,IT Support\n", encoding="utf-8")
    before = rag.count()
    stats = rag.ingest([str(csv_path)])
    assert stats == {"documents": 1, "chunks": 1}
    assert rag.count() == before + 1
