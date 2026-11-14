"""Streamlit front-end for the AI Service Desk Copilot. It talks only to the FastAPI service.

Run:  streamlit run ui/streamlit_app.py      (API must be running; set API_URL if not on localhost:8000)
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
SAMPLES = json.loads((Path(__file__).parent / "sample_tickets.json").read_text(encoding="utf-8"))
PRIORITY_COLOUR = {"high": "red", "medium": "orange", "low": "green"}
SENTIMENT_COLOUR = {"negative": "red", "neutral": "gray", "positive": "green"}

st.set_page_config(page_title="AI Service Desk Copilot", page_icon="🎧", layout="wide")


def call(method: str, path: str, payload: dict | None = None, timeout: int = 120):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = requests.request(method, f"{API_URL}{path}", json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return None, f"Cannot reach the API at {API_URL}: {e}"
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code >= 400:
        err = data.get("error", {})
        return None, f"{r.status_code} {err.get('code', '')}: {err.get('message', r.text)}"
    return data, None


def highlight_masks(text: str) -> str:
    return re.sub(r"\[([A-Z_]+_\d+)\]", r"**:blue[[\1]]**", text)


def card(col, label: str, value: str, sub: str) -> None:
    with col.container(border=True):
        st.caption(label)
        st.markdown(f"#### {value}")
        st.markdown(sub)


def sources_block(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            link = f" · [open]({s['url']})" if s.get("url") else ""
            st.markdown(f"**[{i}] {s['title']}** · similarity {s['score']:.2f}{link}")
            st.caption(s["snippet"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎧 Service Desk Copilot")
    st.caption("Group 60 · AIMLZG549 Assignment II")
    health, herr = call("GET", "/health", timeout=10)
    if herr:
        st.error(herr)
    else:
        colour = "green" if health["status"] == "ok" else "orange"
        st.markdown(f"**API status:** :{colour}[{health['status']}] · v{health['version']}")
        for comp, state in health["components"].items():
            icon = "🟢" if state.startswith(("loaded", "configured")) else ("🔴" if state.startswith(("error", "missing")) else "⚪")
            st.caption(f"{icon} **{comp}**: {state}")
    st.divider()
    st.markdown(f"[API docs (Swagger)]({API_URL.rsplit('/api/', 1)[0]}/docs)")

tab_triage, tab_chat = st.tabs(["📨 Ticket triage", "💬 Knowledge-base chat"])

# ---------------------------------------------------------------- triage tab
with tab_triage:
    st.subheader("Triage an incoming ticket")
    names = ["✏️ Write my own"] + [s["name"] for s in SAMPLES]
    choice = st.selectbox("Start from a sample ticket", names, index=1)
    sample = next((s for s in SAMPLES if s["name"] == choice), {"subject": "", "body": ""})
    with st.form("ticket"):
        subject = st.text_input("Subject", value=sample["subject"], max_chars=300)
        body = st.text_area("Ticket body", value=sample["body"], height=160, max_chars=5000)
        draft = st.checkbox("Draft a reply from the knowledge base", value=True)
        submitted = st.form_submit_button("Run triage", type="primary")

    if submitted:
        if not body.strip():
            st.warning("Enter the ticket body.")
        else:
            with st.spinner("Redacting, routing, prioritising, summarising and drafting…"):
                res, err = call("POST", "/triage", {"ticket_id": f"T-{uuid.uuid4().hex[:6]}", "subject": subject,
                                                    "body": body, "draft_reply": draft})
            if err:
                st.error(err)
            else:
                for w in res.get("warnings", []):
                    st.warning(w)
                c1, c2, c3 = st.columns(3)
                q = res.get("queue")
                if q:
                    card(c1, "Queue", q["label"], f"{q['confidence']:.0%} confidence" + (
                        " · :orange[⚠ route manually]" if q["needs_human_review"] else ""))
                p = res.get("priority")
                if p:
                    card(c2, "Priority", f":{PRIORITY_COLOUR.get(p['label'], 'gray')}[{p['label'].upper()}]",
                         f"{p['score']:.0%} probability")
                s = res.get("sentiment")
                if s:
                    card(c3, "Customer sentiment",
                         f":{SENTIMENT_COLOUR.get(s['label'], 'gray')}[{s['label'].capitalize()}]",
                         f"{s['score']:.0%} probability")

                if res.get("summary"):
                    st.info(f"**Summary:** {res['summary']}")

                left, right = st.columns(2)
                with left:
                    st.markdown("##### 🔒 Redacted ticket (sent to the LLM)")
                    if res.get("redacted_text"):
                        st.markdown(highlight_masks(res["redacted_text"]))
                    ents = res.get("entities") or []
                    if ents:
                        st.dataframe([{"type": e["type"], "text": e["text"], "masked as": e["masked_as"] or "— kept",
                                       "score": e["score"]} for e in ents], hide_index=True, width="stretch")
                with right:
                    st.markdown("##### ✉️ Draft reply")
                    d = res.get("draft_reply")
                    if d:
                        if d["grounded"]:
                            st.markdown(d["text"])
                        else:
                            st.warning(d["text"])
                        sources_block(d["sources"])
                    elif draft:
                        st.caption("No draft produced.")

                lat = res.get("step_latency_ms") or {}
                if lat:
                    st.markdown("##### ⏱ Step latency (ms)")
                    st.bar_chart(pd.DataFrame({"step": list(lat), "ms": list(lat.values())}),
                                 x="step", y="ms", horizontal=True, sort=False, height=220)
                    st.caption(f"Total: {res['meta']['latency_ms']} ms · request {res['meta']['request_id']}")

# ---------------------------------------------------------------- chat tab
with tab_chat:
    st.subheader("Ask the knowledge base")
    st.caption("Answers come only from TechServ policies, SOPs and past tickets, with sources. "
               "Questions outside the knowledge base get a 'not covered' reply.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
        st.session_state.session_id = uuid.uuid4().hex[:8]
    if st.button("Clear conversation"):
        st.session_state.chat = []

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant" and not turn.get("grounded", True):
                st.warning(turn["content"])
            else:
                st.markdown(turn["content"])
            if turn["role"] == "assistant":
                sources_block(turn.get("sources", []))
                st.caption(f"{turn.get('ms', 0)} ms")

    if st.session_state.get("chat_error"):
        st.error(st.session_state.pop("chat_error"))
    question = st.chat_input("e.g. What is the SLA for a P1 outage?")
    if question:
        history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat][-6:]
        with st.spinner("Searching the knowledge base…"):
            res, err = call("POST", "/chat", {"session_id": st.session_state.session_id, "query": question,
                                              "history": history})
        if err:
            st.session_state.chat_error = err
        else:
            st.session_state.chat += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": res["answer"], "sources": res["sources"],
                 "grounded": res["grounded"], "ms": res["meta"]["latency_ms"]}]
        st.rerun()
