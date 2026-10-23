"""Wire rag + agent into one call that follows the JSON contract.

Input:  {"ticket": "..."}
Output: {"category", "priority", "answer", "sources", "confidence", "escalate",
         "handoff_summary" (only if escalate)}
"""

from agent import (
    NOT_FOUND,
    RETRIEVAL_THRESHOLD,
    answer_from_context,
    classify_ticket,
    confidence_gate,
    handoff_summary,
)
from rag import retrieve

TOP_K = 4


def run_ticket_verbose(ticket, threshold=RETRIEVAL_THRESHOLD, k=TOP_K):
    """Run the full pipeline. Returns {"result": <contract dict>, "chunks": [...], "reason": str}."""
    ticket = ticket.strip()
    classification = classify_ticket(ticket)
    chunks = retrieve(ticket, k)
    top_score = chunks[0]["score"] if chunks else 0.0

    if top_score >= threshold:
        answer, sources = answer_from_context(ticket, chunks)
    else:
        answer, sources = NOT_FOUND, []  # KB match too weak to ground an answer
    confidence, escalate, reason = confidence_gate(top_score, answer, threshold)

    result = {
        "category": classification["category"],
        "priority": classification["priority"],
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "escalate": escalate,
    }
    if escalate:
        result["handoff_summary"] = handoff_summary(
            ticket, result["category"], result["priority"], reason, chunks
        )
    return {"result": result, "chunks": chunks, "reason": reason}


def run_ticket(payload, threshold=RETRIEVAL_THRESHOLD):
    """Contract entry point: takes {"ticket": "..."} and returns the output dict."""
    return run_ticket_verbose(payload["ticket"], threshold)["result"]


if __name__ == "__main__":
    import json
    import sys

    text = " ".join(sys.argv[1:]) or "My VPN says gateway unreachable at the hotel"
    print(json.dumps(run_ticket({"ticket": text}), indent=2))
