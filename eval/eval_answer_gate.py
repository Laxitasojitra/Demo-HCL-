"""Evaluate Person 3's answer + escalation logic on data/tickets.json.

Uses Person 1's rag.retrieve() and the existing classify_ticket(), then runs
agent.gate.resolve() on each ticket and reports how well escalation matches
the labels. Works offline (no API key) or with an LLM key set in .env.

Usage (from the repo root):
    python -m eval.eval_answer_gate
    python -m eval.eval_answer_gate --threshold 0.35 --min-confidence 0.5
Writes per-ticket rows to eval/answer_gate_results.csv.
"""

import argparse
import csv
import json
from pathlib import Path

from agent import active_provider, classify_ticket
from agent.gate import MIN_CONFIDENCE, RETRIEVAL_THRESHOLD, resolve
from rag import retrieve

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=RETRIEVAL_THRESHOLD)
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    args = parser.parse_args()

    tickets = json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))
    rows = []
    for t in tickets:
        chunks = retrieve(t["ticket"], TOP_K)
        cls = classify_ticket(t["ticket"])
        out = resolve(t["ticket"], chunks, cls, threshold=args.threshold, min_confidence=args.min_confidence)
        retrieved = {c["source"] for c in chunks}
        rows.append({
            "id": t["id"],
            "ticket": t["ticket"],
            "category": cls["category"],
            "top_score": chunks[0]["score"] if chunks else 0.0,
            "confidence": out["confidence"],
            "expected_escalate": t["expected_escalate"],
            "escalate": out["escalate"],
            "escalate_ok": out["escalate"] == t["expected_escalate"],
            "citations_valid": all(s in retrieved for s in out["sources"]),
            "sources": ";".join(out["sources"]),
            "flags": ";".join(out["flags"]),
            "reasons": " | ".join(out["reasons"]),
        })
        print(f"  ran ticket {t['id']:>2}/{len(tickets)}", end="\r")

    n = len(rows)
    should = [r for r in rows if r["expected_escalate"]]
    should_not = [r for r in rows if not r["expected_escalate"]]
    answered = [r for r in rows if not r["escalate"]]

    print(f"\nProvider: {active_provider() or 'offline (no API key)'}   "
          f"threshold: {args.threshold}   min confidence: {args.min_confidence}\n")
    print(f"{'id':>3} {'score':>6} {'conf':>5}  esc exp/got   flags / reasons")
    print("-" * 100)
    for r in rows:
        mark = " " if r["escalate_ok"] else "x"
        why = r["flags"] or r["reasons"]
        print(f"{r['id']:>3} {r['top_score']:>6.2f} {r['confidence']:>5.2f}  "
              f"{str(r['expected_escalate'])[0]}/{str(r['escalate'])[0]} {mark}       {why[:70]}")

    print("\nSummary")
    print(f"  Escalation accuracy         : {sum(r['escalate_ok'] for r in rows) / n:.0%}")
    print(f"  Must-escalate caught        : {sum(r['escalate'] for r in should)}/{len(should)}")
    print(f"  In-scope auto-resolved      : {sum(not r['escalate'] for r in should_not)}/{len(should_not)}")
    print(f"  Escalation rate             : {sum(r['escalate'] for r in rows) / n:.0%}  "
          f"(expected {len(should) / n:.0%})")
    if answered:
        print(f"  Answers with valid citations: {sum(r['citations_valid'] and r['sources'] != '' for r in answered)}/{len(answered)}")

    out_path = ROOT / "eval" / "answer_gate_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-ticket results written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
