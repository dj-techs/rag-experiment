"""Streamlit demo UI. Talks to the FastAPI gateway over HTTP.

Run:
    # terminal 1
    uv run uvicorn src.api:app --reload
    # terminal 2
    uv run streamlit run ui/app.py
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
FEEDBACK_PATH = Path("data/feedback.jsonl")

st.set_page_config(
    page_title="GenAI Demo - Industry RAG Showcase",
    page_icon=":mag:",
    layout="wide",
)


def _persist_feedback(query: str, data: dict, helpful: bool) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.now().isoformat(),
            "query": query,
            "helpful": helpful,
            "confidence": data["confidence"],
            "intent": data["intent"],
        }) + "\n")


# -- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("Loaded corpus")
    st.markdown(
        "- **Apple (AAPL)** — most recent 10-K\n"
        "- **Microsoft (MSFT)** — most recent 10-K\n"
        "- **NVIDIA (NVDA)** — most recent 10-K\n\n"
        "Source: SEC EDGAR (live download)"
    )
    st.divider()
    arch_path = Path("architecture_diagram.png")
    if arch_path.exists():
        with st.expander("Architecture diagram"):
            st.image(str(arch_path))
    st.divider()
    try:
        h = requests.get(f"{API_URL}/health", timeout=2)
        if h.ok:
            st.success(f"API: {h.json().get('status')}")
        else:
            st.error(f"API down: HTTP {h.status_code}")
    except Exception as e:
        st.error(f"API unreachable at {API_URL}\n\n{e}")


# -- Main --------------------------------------------------------------------

st.title("GenAI Demo - Industry RAG Showcase")
st.caption(
    "LangGraph orchestrator over hybrid retrieval (dense + BM25) on SEC "
    "10-K filings, with strict-grounding prompts + citation validation."
)

st.subheader("Try a sample question")
samples = [
    ("Factual (in-corpus)",
     "What were Apple's main revenue drivers in fiscal 2025?"),
    ("Multi-company synthesis",
     "Compare R&D spending across the three companies"),
    ("Out-of-scope (abstention)",
     "What was the weather in Tokyo last Tuesday?"),
    ("Ambiguous (clarification)",
     "Tell me about their products"),
]
cols = st.columns(len(samples))
for col, (label, q) in zip(cols, samples):
    if col.button(label, use_container_width=True):
        st.session_state["query_text"] = q

query = st.text_input(
    "Ask a question:",
    value=st.session_state.get("query_text", ""),
    key="query_input",
)
ask = st.button("Ask", type="primary", disabled=not query.strip())

if ask and query.strip():
    with st.spinner("Running orchestrator (router -> planner -> executor -> verifier)..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"query": query, "tenant_id": "demo"},
                timeout=120,
            )
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()
        if not resp.ok:
            st.error(f"API error {resp.status_code}: {resp.text}")
            st.stop()
        data = resp.json()

    conf = data["confidence"]
    badge_color = {"high": "green", "medium": "orange", "low": "red"}[conf]
    cols = st.columns([2, 2, 2, 6])
    cols[0].markdown(f"**Confidence:** :{badge_color}[{conf.upper()}]")
    cols[1].markdown(f"**Intent:** `{data['intent']}`")
    cols[2].markdown(f"**Iterations:** {data['iterations']}")

    if conf == "low":
        st.warning(
            "Low confidence -- review recommended. "
            "(In production this is the HITL gate; here it's a UX flag.)"
        )

    st.markdown("### Answer")
    st.markdown(data["answer"])

    if data["citations"]:
        st.markdown("### Citations")
        for i, c in enumerate(data["citations"], 1):
            label = f"[{i}] {c['source_id']}"
            if c.get("page"):
                label += f" -- page {c['page']}"
            label += f"  ({c['chunk_id']})"
            with st.expander(label):
                st.text(c["excerpt"])

    st.caption(
        f"Cost: ${data['cost_usd']:.4f}  |  "
        f"Verifier: {'pass' if data['verifier_passed'] else 'fail'}  |  "
        f"Backend: {API_URL}"
    )

    fb1, fb2, _ = st.columns([1, 1, 8])
    if fb1.button("Helpful", key="fb_yes"):
        _persist_feedback(query, data, True)
        st.toast("Thanks!")
    if fb2.button("Not helpful", key="fb_no"):
        _persist_feedback(query, data, False)
        st.toast("Logged.")
