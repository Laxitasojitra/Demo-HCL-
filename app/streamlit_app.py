"""Streamlit UI for the IT Helpdesk Copilot.  Run:  streamlit run app/streamlit_app.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent import RETRIEVAL_THRESHOLD, active_provider  # noqa: E402
from app.pipeline import run_ticket_verbose  # noqa: E402
from rag import retrieve  # noqa: E402

st.set_page_config(page_title="IT Helpdesk Copilot", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base and embedding model...")
def warm_up():
    retrieve("warm up", 1)  # builds the shared index once per server process
    return True


@st.cache_data
def sample_tickets():
    return json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))


warm_up()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    provider = active_provider()
    if provider:
        st.success(f"LLM provider: **{provider}**")
    else:
        st.warning("No LLM API key found. Running in **offline mode** (keyword classifier, extractive answers).")
    threshold = st.slider("Retrieval threshold", 0.0, 1.0, RETRIEVAL_THRESHOLD, 0.01,
                          help="Escalate when the best KB match scores below this cosine similarity.")
    st.caption("Escalates on: security incident, prompt injection, privileged access or out-of-scope ticket; "
               "top retrieval score < threshold; the KB doesn't answer it; bad citations; or confidence < 0.45.")

# ---------------------------------------------------------------- input
st.title("🛠️ AI Copilot for Enterprise IT Helpdesk")
st.write("Paste an IT support ticket. The copilot classifies it, searches the knowledge base, "
         "answers only from what it finds, and escalates to a human when it is not confident.")

samples = sample_tickets()
choice = st.selectbox("Try a sample ticket", ["(write your own)"] + [t["ticket"] for t in samples])
ticket = st.text_area("Ticket", value="" if choice == "(write your own)" else choice, height=110,
                      placeholder="e.g. My VPN keeps disconnecting when I work from home")

if st.button("Submit ticket", type="primary", disabled=not ticket.strip()):
    with st.spinner("Classifying, retrieving, answering..."):
        run = run_ticket_verbose(ticket, threshold=threshold)
    result, chunks = run["result"], run["chunks"]

    c1, c2, c3, c4 = st.columns([2.2, 1.2, 1, 1])
    c1.metric("Category", result["category"])
    c2.metric("Priority", result["priority"])
    c3.metric("Confidence", f"{result['confidence']:.0%}")
    c4.metric("Top retrieval score", f"{chunks[0]['score']:.2f}" if chunks else "n/a")
    st.progress(result["confidence"])

    if result["escalate"]:
        st.error(f"**Escalated to a human agent** — {run['reason']}.")
        st.subheader("Handoff summary")
        st.text(result["handoff_summary"])
    else:
        st.success("Resolved by the copilot — no escalation needed.")

    st.subheader("Answer")
    if result["answer"] == "NOT_FOUND":
        st.info("NOT_FOUND — the knowledge base does not cover this ticket.")
    else:
        st.markdown(result["answer"])
    st.markdown("**Sources:** " + (", ".join(f"`{s}`" for s in result["sources"]) or "none"))

    with st.expander("Retrieved chunks and scores", expanded=False):
        for c in chunks:
            st.markdown(f"**{c['source']}** — _{c['heading']}_ — score `{c['score']:.3f}`")
            st.text(c["text"])
            st.divider()

    with st.expander("JSON output (contract)"):
        st.json(result)
