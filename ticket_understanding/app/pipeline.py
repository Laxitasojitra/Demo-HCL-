"""Integrated pipeline: one ticket through all three team members' modules.

    ticket
      -> ticket_understanding.classify()   Person 2: guard (PII mask, injection, security)
                                                      + category / priority / confidence
      -> rag.retrieve(clean_text)          Person 1: top-k KB chunks with scores
      -> agent.gate.resolve()              Person 3: grounded answer, confidence gate,
                                                      escalation rules, handoff summary
      -> JSON contract

Only the PII-masked text (`clean_text`) is passed to retrieval and to any LLM call.

Input:  {"ticket": "..."}
Output: {"category", "priority", "answer", "sources", "confidence", "escalate",
         "handoff_summary" (only if escalate)}
"""

from agent.core import NOT_FOUND, RETRIEVAL_THRESHOLD  # noqa: F401  (re-exported for the UI/eval)
from agent.gate import handoff_summary, resolve
from rag import retrieve
from ticket_understanding.classifier import classify

TOP_K = 4
HOLDING_MESSAGE = "I couldn't resolve this automatically. I've passed it to a support engineer with a summary."

# Person 2 uses short labels; the KB, data/tickets.json and the UI use these.
CATEGORY_MAP = {"Access": "Account & Access", "Email": "Email & Collaboration"}


def understand(ticket):
    """Person 2: guard + classification, with labels mapped to the team's label set."""
    c = classify(ticket)
    c["category"] = CATEGORY_MAP.get(c["category"], c["category"])
    c["cls_confidence"] = c["confidence"]  # the name Person 3's gate reads
    return c


def run_ticket_verbose(ticket, threshold=RETRIEVAL_THRESHOLD, k=TOP_K):
    """Run the full pipeline.

    Returns {"result": <contract dict>, "chunks": [...], "reason": str, "classification": {...}}.
    """
    ticket = ticket.strip()
    classification = understand(ticket)          # Person 2
    text = classification["clean_text"]          # PII masked from here on
    chunks = retrieve(text, k)                   # Person 1
    out = resolve(text, chunks, classification, threshold=threshold)  # Person 3
    reasons = list(out["reasons"])

    # Person 2's guard can catch things Person 3's rules don't (e.g. a login from another
    # country). If the guard says escalate, escalate, even when the gate would have answered.
    if classification["must_escalate"] and not out["escalate"]:
        reasons = [f"ticket guard: {r}" for r in classification["escalate_reasons"]]
        decision = {
            "reason": "; ".join(reasons),
            "flags": out["flags"],
            "top_score": round(float(chunks[0]["score"]), 4) if chunks else 0.0,
        }
        out = {
            **out,
            "answer": HOLDING_MESSAGE,
            "sources": [],
            "escalate": True,
            "handoff_summary": handoff_summary(text, classification, decision, chunks),
        }

    result = {
        "category": classification["category"],
        "priority": classification["priority"],
        "answer": out["answer"],
        "sources": out["sources"],
        "confidence": out["confidence"],
        "escalate": out["escalate"],
    }
    if out["escalate"]:
        result["handoff_summary"] = out["handoff_summary"]
    return {"result": result, "chunks": chunks, "reason": "; ".join(reasons), "classification": classification}


def run_ticket(payload, threshold=RETRIEVAL_THRESHOLD):
    """Contract entry point: takes {"ticket": "..."} and returns the output dict."""
    return run_ticket_verbose(payload["ticket"], threshold)["result"]


if __name__ == "__main__":
    import json
    import sys

    text = " ".join(sys.argv[1:]) or "My VPN says gateway unreachable at the hotel"
    print(json.dumps(run_ticket({"ticket": text}), indent=2))
