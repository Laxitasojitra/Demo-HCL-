"""
eval_classifier.py - measure category / priority accuracy and escalation on data/test_tickets.csv.

    python eval_classifier.py                 # LLM classifier (falls back to keywords if LLM fails)
    python eval_classifier.py --keyword-only  # keyword fallback only, no API key needed
"""
import argparse
import csv
import os

from classifier import PRIORITIES, classify

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(HERE, "data", "test_tickets.csv"))
    ap.add_argument("--keyword-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "classifier_results.csv"))
    args = ap.parse_args()

    with open(args.file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results, cat_ok, pri_ok, pri_near = [], 0, 0, 0
    tp = fp = fn = 0
    methods = {}
    for r in rows:
        out = classify(r["ticket"], use_llm=not args.keyword_only)
        c = out["category"] == r["expected_category"]
        p = out["priority"] == r["expected_priority"]
        near = abs(PRIORITIES.index(out["priority"]) - PRIORITIES.index(r["expected_priority"])) <= 1
        exp_esc = r["should_escalate"].strip().lower() == "true"
        cat_ok += c
        pri_ok += p
        pri_near += near
        tp += exp_esc and out["must_escalate"]
        fp += (not exp_esc) and out["must_escalate"]
        fn += exp_esc and not out["must_escalate"]
        methods[out["method"]] = methods.get(out["method"], 0) + 1
        results.append({
            "id": r["id"], "type": r["type"],
            "expected_category": r["expected_category"], "got_category": out["category"], "category_ok": c,
            "expected_priority": r["expected_priority"], "got_priority": out["priority"], "priority_ok": p,
            "should_escalate": exp_esc, "got_escalate": out["must_escalate"],
            "confidence": out["confidence"], "method": out["method"],
            "masked": "|".join(out["masked"]), "reason": out["reason"], "llm_error": out["llm_error"] or "",
        })

    n = len(rows)
    print(f"\nTickets: {n}   methods used: {methods}")
    print(f"Category accuracy : {cat_ok}/{n} = {cat_ok / n:.0%}   (target >= 80%)")
    print(f"Priority accuracy : {pri_ok}/{n} = {pri_ok / n:.0%}   (within one level: {pri_near / n:.0%})")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"Escalation        : recall {rec:.0%}, precision {prec:.0%}  (TP={tp} FP={fp} FN={fn})")

    for t in ("easy", "tricky", "escalate"):
        sub = [x for x in results if x["type"] == t]
        if sub:
            print(f"  {t:9s} category {sum(x['category_ok'] for x in sub)}/{len(sub)}, "
                  f"priority {sum(x['priority_ok'] for x in sub)}/{len(sub)}")

    misses = [x for x in results if not (x["category_ok"] and x["priority_ok"])]
    if misses:
        print("\nMisses:")
        for x in misses:
            print(f"  {x['id']}: category {x['expected_category']} -> {x['got_category']}, "
                  f"priority {x['expected_priority']} -> {x['got_priority']}  [{x['method']}]")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nPer-ticket results written to {args.out}")


if __name__ == "__main__":
    main()
