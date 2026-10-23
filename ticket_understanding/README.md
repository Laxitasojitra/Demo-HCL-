# AI Copilot for Enterprise IT Helpdesk

Hackathon prototype. Takes an IT support ticket in plain English and:

1. **Classifies** it (category + priority) with a few-shot, strict-JSON prompt.
2. **Retrieves** the most relevant chunks from a small markdown knowledge base (MiniLM embeddings + in-memory FAISS).
3. **Answers** only from the retrieved context, citing source files, or returns `NOT_FOUND`.
4. **Escalates** to a human with an auto-written handoff summary when confidence is low.

```
ticket ──► ticket_understanding.classify()   Person 2: guard (mask PII, injection, security)
             │                                         + category / priority / confidence
             ▼  clean_text (PII masked)
           rag.retrieve(k=4)                  Person 1: top-k KB chunks with scores
             │
             ▼
           agent.gate.resolve()               Person 3: grounded answer, citation check,
             │                                         confidence gate, handoff summary
             ▼
           JSON  (answer, or escalate + handoff_summary)
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # then add ONE API key (optional, see below)
streamlit run app/streamlit_app.py
```

Run the evaluation over the 25 sample tickets:

```bash
python -m eval.run_eval
python -m eval.run_eval --threshold 0.4   # try another threshold
```

Run a single ticket from the command line:

```bash
python -m app.pipeline "My VPN says gateway unreachable at the hotel"
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). Everything runs on CPU.

### LLM provider

All LLM calls go through one function, `llm(prompt)` in [agent/llm.py](agent/llm.py). The provider is picked from whichever key is set
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`), or forced with `LLM_PROVIDER`. Override the model with `LLM_MODEL`.
To add another provider, add one branch to `llm()`.

**Offline mode:** with no key set, the app still runs end to end. It uses a keyword classifier, returns the best-matching KB section as the answer, and writes a template handoff note. This is handy for demos without network access, and as a baseline to compare the LLM against.

## JSON contract

Input: `{"ticket": "..."}`

Output:

```json
{
  "category": "Network",
  "priority": "Low|Medium|High",
  "answer": "1. ... [vpn_setup.md]",
  "sources": ["vpn_setup.md"],
  "confidence": 0.97,
  "escalate": false,
  "handoff_summary": "... (only present if escalate is true)"
}
```

Entry point: `app.pipeline.run_ticket({"ticket": "..."})`.

## Escalation rule

Escalate if **the top retrieval score is below the threshold** (default `0.30` cosine similarity, `RETRIEVAL_THRESHOLD` in [agent/core.py](agent/core.py), adjustable with a slider in the UI) **or the model returns `NOT_FOUND`**.
When retrieval is below the threshold the answer step is skipped, because there is nothing relevant to ground an answer in.

`confidence` maps the top cosine score onto 0–1 (0.15 → 0, 0.65 → 1) and is capped at 0.3 when the answer is `NOT_FOUND`.

## Project layout (one module per teammate)

| Path | Owner | What it does |
|---|---|---|
| `rag/store.py` | RAG | Load `data/kb/*.md`, split by `##` section, embed, FAISS search. `retrieve(query, k)` returns `[{text, source, heading, score}]`. |
| `agent/llm.py` | Agent | `llm(prompt)` provider wrapper. |
| `agent/core.py` | Agent | Original `classify_ticket()`, `answer_from_context()`, `confidence_gate()`, `handoff_summary()`. Superseded in the pipeline by the modules below; kept for `eval/eval_answer_gate.py` and comparison. |
| `ticket_understanding/` | Person 2 | Guard (PII masking, prompt-injection and security detection) + classifier (LLM with keyword fallback). See its README. |
| `agent/answerer.py`, `agent/gate.py` | Person 3 | Grounded answer with citation checks, escalation rules, confidence, handoff summary. |
| `app/pipeline.py` | App | Wires Person 2 → Person 1 → Person 3 into the JSON contract. |
| `app/streamlit_app.py` | App | UI: ticket input, category/priority/confidence, answer, sources, retrieved chunks, escalation and handoff. |
| `data/kb/` | – | 12 IT FAQ/SOP docs (VPN, password reset, lockout, MFA, software, Wi-Fi, Outlook, Teams, printing, laptop, hardware requests, shared drives). |
| `data/tickets.json` | – | 25 labelled tickets: 20 in scope, 5 out of scope (payroll, facilities, security incident, production access, legal). |
| `eval/run_eval.py` | – | Runs all tickets, prints category accuracy, escalation accuracy, escalation rate, and a results table; writes `eval/results.csv`. |

Categories: `Network`, `Account & Access`, `Software`, `Email & Collaboration`, `Hardware`, `Other`.

## Baseline results (original pipeline, before integration; offline mode, threshold 0.30)

Run `python -m eval.run_eval` with an API key to get the LLM numbers.

| Metric | Offline |
|---|---|
| Category accuracy | 92% (23/25) |
| Escalation accuracy | 96% (24/25) |
| Escalation rate | 16% (expected 20%) |

The hard case is ticket 23 (credentials typed into a phishing page). It retrieves well-matching KB text about phishing and passwords, so the retrieval gate doesn't fire. Only the LLM's `NOT_FOUND` judgement can catch it. Note that the 0.30 threshold was picked on these same 25 tickets, so treat the numbers as optimistic.

## Person 3: answer generation and escalation (`agent/answerer.py`, `agent/gate.py`)

A stricter answer + escalation layer that sits on top of Person 1's `retrieve()` output. It does not change `agent/core.py` or `app/pipeline.py`; wiring it in is a separate step.

| Function | What it does |
|---|---|
| `generate_answer(ticket, chunks)` | Strict-JSON prompt: answer only from the chunks, cite `[file.md]` after each step, return `answerable`. Retries once on bad JSON; offline mode returns the top chunk. Every cited file is checked against the retrieved files, and made-up ones go to `invalid_citations`. |
| `decide(ticket, chunks, answer_result, classification)` | Code decides answer vs escalate. Returns `confidence`, `escalate`, `reasons`, `flags`. |
| `handoff_summary(ticket, classification, decision, chunks)` | 3-line note: `Issue:` / `Triage:` / `Reason escalated: ... Next step: ...`. LLM when available, template otherwise. |
| `resolve(ticket, chunks, classification)` | One call: answer, gate, handoff. Returns `answer, sources, confidence, escalate, handoff_summary, reasons, flags`. |

**Escalate if any of these is true:**

1. Hard rule on the ticket: security incident (phishing, typed password on a fake page, malware), prompt injection, privileged/production access, out-of-scope topic (payroll, facilities, legal), or category `Other` / priority `Critical`. Security and injection tickets are never sent to the answer prompt.
2. The LLM failed or returned invalid JSON twice.
3. Top retrieval score below `0.30`.
4. The model says the context does not answer the ticket.
5. The answer cites a file that was not retrieved, or cites nothing.
6. Confidence below `0.45`.

**Confidence** = `0.6 × retrieval + 0.2 × classifier confidence + 0.2 × answerable`, where retrieval maps the cosine score onto 0–1 (0.15 → 0, 0.65 → 1). If the classifier returns no confidence, it becomes `0.75 × retrieval + 0.25 × answerable`. Not-answerable results are capped at 0.3.

```python
from rag import retrieve
from agent import classify_ticket
from agent.gate import resolve

ticket = "My VPN says gateway unreachable at the hotel"
out = resolve(ticket, retrieve(ticket, 4), classify_ticket(ticket))
```

Tests (no API key or embedding model needed) and evaluation on the 25 labelled tickets:

```bash
python -m pytest tests -q
python -m eval.eval_answer_gate
```

## Integrated pipeline (all three modules)

`app/pipeline.py` now runs one ticket through all three modules, so the UI and `eval/run_eval.py` use them automatically:

1. **Person 2** `ticket_understanding.classify()` masks emails, phone numbers and passwords, detects prompt injection and security incidents, and returns category, priority (`Low/Medium/High/Critical`) and confidence. Its short labels are mapped to the team's (`Access` → `Account & Access`, `Email` → `Email & Collaboration`).
2. **Person 1** `rag.retrieve()` searches the KB using the masked text.
3. **Person 3** `agent.gate.resolve()` answers from the chunks, checks citations and decides answer vs escalate. Security and injection tickets never reach the answer prompt.
4. If Person 2's guard says `must_escalate` (e.g. a login from another country) but the gate would have answered, the pipeline escalates anyway and writes the handoff note.

Only the PII-masked text is sent to retrieval and to any LLM.

**One LLM provider for everything.** `ticket_understanding/llm.py` defers to `agent/llm.py` unless `LLM_PROVIDER` is set to one of its own providers, so the classifier and the answerer use the same model. `agent/llm.py` now also supports Groq (`GROQ_API_KEY`, free tier).

Offline check on `data/tickets.json`: category accuracy 23/25, same as the original classifier. The two differences are labelling choices to agree on as a team: ticket 23 (phishing) and 24 (production DB access) are labelled `Other`, while the new classifier says `Email & Collaboration` and `Account & Access`. Both still escalate. Re-run `python -m eval.run_eval` with the real retriever and an API key for the final numbers.

## Ideas for next steps

- Also escalate on `category == "Other"` or on security keywords.
- Hybrid retrieval (BM25 + embeddings) for product names that MiniLM matches weakly (e.g. "Tableau").
- Create a ticket in ServiceNow/Jira with the handoff summary.
