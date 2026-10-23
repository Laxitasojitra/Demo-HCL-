# Ticket understanding (Person 2)

Classification + guard: what the ticket is, how urgent it is, and whether it is safe to process.

| File | What it does |
|---|---|
| `llm.py` | One wrapper for Groq / Gemini / OpenAI (OpenAI-compatible endpoints). Temperature 0, strict JSON, one retry on bad JSON. `LLM_PROVIDER=mock` forces fallbacks. |
| `guard.py` | Masks emails, phone numbers, passwords. Detects prompt injection and security incidents (phishing, malware, ransomware, account compromise). |
| `classifier.py` | `classify(ticket)` -> category, priority, confidence, `must_escalate`. LLM first, keyword fallback if the LLM fails or the ticket is an injection attempt. |
| `eval_classifier.py` | Category / priority accuracy and escalation recall on `data/test_tickets.csv`. |
| `data/test_tickets.csv` | 20 labelled tickets: 12 easy, 4 tricky, 4 must-escalate. |

## Run

```bash
cd ticket_understanding
pip install openai
export LLM_PROVIDER=groq GROQ_API_KEY=...      # or gemini / openai
python classifier.py "My VPN keeps disconnecting every 5 min"
python eval_classifier.py                      # LLM (falls back to keywords on failure)
python eval_classifier.py --keyword-only       # no API key needed
```

Keyword-only results: 20/20 category on the test set (keywords were tuned on it, so optimistic), 9/10 on unseen tickets.

## Integration notes for the team

Labels differ from `agent/core.py`; agree on one set before wiring this in:

| Here | `agent/core.py` |
|---|---|
| Access | Account & Access |
| Email | Email & Collaboration |
| Priority adds **Critical** | Low / Medium / High only |

`classify()` returns `must_escalate` + `escalate_reasons` (injection, security incident, critical priority). The confidence gate should escalate whenever `must_escalate` is true. Always send `clean_text` (PII masked) to any LLM, never the raw ticket.
