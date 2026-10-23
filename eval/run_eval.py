"""Run every sample ticket through the pipeline and report accuracy.

Usage (from the repo root):
    python -m eval.run_eval                 # default threshold
    python -m eval.run_eval --threshold 0.4
Writes per-ticket results to eval/results.csv.
"""

import argparse
import csv
import json
from pathlib import Path

from agent import RETRIEVAL_THRESHOLD, active_provider
from app.pipeline import run_ticket_verbose

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=RETRIEVAL_THRESHOLD)
    args = parser.parse_args()

    tickets = json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))
    rows = []
    for t in tickets:
        run = run_ticket_verbose(t["ticket"], threshold=args.threshold)
        r = run["result"]
        rows.append({
            "id": t["id"],
            "ticket": t["ticket"],
            "expected_category": t["expected_category"],
            "category": r["category"],
            "category_ok": r["category"] == t["expected_category"],
            "priority": r["priority"],
            "top_score": run["chunks"][0]["score"] if run["chunks"] else 0.0,
            "confidence": r["confidence"],
            "expected_escalate": t["expected_escalate"],
            "escalate": r["escalate"],
            "escalate_ok": r["escalate"] == t["expected_escalate"],
            "sources": ";".join(r["sources"]),
        })
        print(f"  ran ticket {t['id']:>2}/{len(tickets)}", end="\r")

    n = len(rows)
    should = [x for x in rows if x["expected_escalate"]]
    should_not = [x for x in rows if not x["expected_escalate"]]

    print(f"\nProvider: {active_provider() or 'offline (no API key)'}   threshold: {args.threshold}\n")
    header = f"{'id':>3}  {'expected':<22}{'predicted':<22}{'prio':<7}{'score':>6}{'conf':>6}  {'esc exp/got':<12} ticket"
    print(header)
    print("-" * len(header))
    for x in rows:
        mark_c = " " if x["category_ok"] else "x"
        mark_e = " " if x["escalate_ok"] else "x"
        print(f"{x['id']:>3}  {x['expected_category']:<22}{x['category'][:20]:<20}{mark_c} {x['priority']:<7}"
              f"{x['top_score']:>6.2f}{x['confidence']:>6.2f}  {str(x['expected_escalate'])[0]}/{str(x['escalate'])[0]} {mark_e:<7} "
              f"{x['ticket'][:55]}")

    print("\nSummary")
    print(f"  Category accuracy      : {sum(x['category_ok'] for x in rows) / n:.0%}")
    print(f"  Escalation accuracy    : {sum(x['escalate_ok'] for x in rows) / n:.0%}")
    print(f"  Escalation rate        : {sum(x['escalate'] for x in rows) / n:.0%}  "
          f"(expected {len(should) / n:.0%})")
    print(f"  Out-of-scope caught    : {sum(x['escalate'] for x in should)}/{len(should)}")
    print(f"  In-scope auto-resolved : {sum(not x['escalate'] for x in should_not)}/{len(should_not)}")

    out = ROOT / "eval" / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-ticket results written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
