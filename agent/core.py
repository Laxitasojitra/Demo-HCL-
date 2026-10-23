"""Agent steps: classify, answer from context, confidence gate, handoff summary.

Each step calls llm() when a key is configured. Without one, each step uses a
small offline fallback so the pipeline can still be demoed end to end.
"""

import json
import re

from agent.llm import LLMUnavailable, llm

CATEGORIES = ["Network", "Account & Access", "Software", "Email & Collaboration", "Hardware", "Other"]
PRIORITIES = ["Low", "Medium", "High"]
NOT_FOUND = "NOT_FOUND"
RETRIEVAL_THRESHOLD = 0.30  # cosine score below which we don't trust the KB match

# ---------------------------------------------------------------- classify

CLASSIFY_PROMPT = """You are an IT helpdesk triage assistant. Classify the ticket.

Allowed categories: {categories}
Use "Other" for anything that is not an IT support issue covered by these categories
(HR, payroll, facilities, legal, security incidents, privileged/production access).

Priority rules:
- High: user cannot work at all, many users affected, or possible security incident.
- Medium: user is blocked on one task or the issue is recurring.
- Low: how-to questions and requests that can wait.

Reply with strict JSON only, no prose, no code fences:
{{"category": "<one allowed category>", "priority": "Low|Medium|High"}}

Examples:
Ticket: My VPN won't connect from home and I have a client call in 10 minutes.
{{"category": "Network", "priority": "High"}}

Ticket: How do I set up my email on my new iPhone?
{{"category": "Email & Collaboration", "priority": "Low"}}

Ticket: I'm locked out of my account and can't log in to my laptop.
{{"category": "Account & Access", "priority": "High"}}

Ticket: Can I get a new wireless mouse?
{{"category": "Hardware", "priority": "Low"}}

Ticket: Adobe Reader keeps crashing when I open large PDFs.
{{"category": "Software", "priority": "Medium"}}

Ticket: The coffee machine in the kitchen is leaking.
{{"category": "Other", "priority": "Low"}}

Ticket: {ticket}
"""

KEYWORDS = {
    "Network": ["vpn", "wi-fi", "wifi", "globalprotect", "internet", "network", "gateway"],
    "Account & Access": ["password", "locked", "lockout", "mfa", "authenticator", "access", "login", "log in", "sharepoint", "folder"],
    "Software": ["install", "software", "portal", "app ", "vs code", "python", "tableau", "zoom", "update"],
    "Email & Collaboration": ["outlook", "email", "mailbox", "teams", "meeting", "microphone", "camera"],
    "Hardware": ["laptop", "printer", "print", "monitor", "dock", "battery", "bitlocker", "keyboard", "mouse", "screen"],
}


def _keyword_classify(ticket):
    text = ticket.lower()
    hits = {cat: sum(w in text for w in words) for cat, words in KEYWORDS.items()}
    category = max(hits, key=hits.get) if max(hits.values()) > 0 else "Other"
    if any(w in text for w in ["urgent", "can't work", "cannot work", "everyone", "outage", "breach", "exposed", "can't log in"]):
        priority = "High"
    elif any(w in text for w in ["how do i", "how can i", "can i get", "request"]):
        priority = "Low"
    else:
        priority = "Medium"
    return {"category": category, "priority": priority}


def _parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def classify_ticket(ticket):
    """Return {"category": ..., "priority": ...} using few-shot JSON classification."""
    try:
        raw = llm(CLASSIFY_PROMPT.format(categories=", ".join(CATEGORIES), ticket=ticket))
        data = _parse_json(raw)
    except LLMUnavailable:
        return _keyword_classify(ticket)
    except (json.JSONDecodeError, ValueError):
        data = {}

    fallback = _keyword_classify(ticket)
    category = data.get("category") if data.get("category") in CATEGORIES else fallback["category"]
    priority = data.get("priority") if data.get("priority") in PRIORITIES else fallback["priority"]
    return {"category": category, "priority": priority}


# ---------------------------------------------------------------- answer

ANSWER_PROMPT = """You are an IT helpdesk assistant. Answer the employee's ticket using ONLY the context below.

Rules:
- Use only facts from the context. Do not use outside knowledge.
- If the context does not contain the answer, reply with exactly: NOT_FOUND
- Give short, numbered steps where possible.
- Cite the source file for each step in square brackets, e.g. [vpn_setup.md].

Context:
{context}

Ticket: {ticket}

Answer:"""


def format_context(chunks):
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)


def answer_from_context(ticket, chunks):
    """Return (answer_text, sources). answer_text is NOT_FOUND if the context lacks the answer."""
    if not chunks:
        return NOT_FOUND, []
    try:
        answer = llm(ANSWER_PROMPT.format(context=format_context(chunks), ticket=ticket))
    except LLMUnavailable:
        # Offline fallback: return the best-matching KB section verbatim.
        top = chunks[0]
        body = top["text"].split("\n", 1)[-1]  # drop the "Title - Heading" line
        return f"From the knowledge base ({top['heading']}):\n\n{body} [{top['source']}]", [top["source"]]

    if not answer or NOT_FOUND in answer.upper():
        return NOT_FOUND, []

    retrieved = [c["source"] for c in chunks]
    cited = [s for s in dict.fromkeys(re.findall(r"\[([\w\-.]+\.md)\]", answer)) if s in retrieved]
    return answer, cited or [retrieved[0]]


# ---------------------------------------------------------------- gate

def confidence_gate(top_score, answer, threshold=RETRIEVAL_THRESHOLD):
    """Return (confidence 0-1, escalate bool, reason).

    Escalate if the best retrieval score is below the threshold OR the model said NOT_FOUND.
    Confidence maps the cosine score onto 0-1 (0.15 -> 0.0, 0.65 -> 1.0) and is capped
    low when the model could not answer.
    """
    confidence = min(1.0, max(0.0, (top_score - 0.15) / 0.5))
    reasons = []
    if top_score < threshold:
        reasons.append(f"top retrieval score {top_score:.2f} is below threshold {threshold:.2f}")
    elif answer == NOT_FOUND:
        reasons.append("the model found no answer in the retrieved knowledge base articles")
    if answer == NOT_FOUND:
        confidence = min(confidence, 0.3)
    return round(confidence, 2), bool(reasons), "; ".join(reasons)


# ---------------------------------------------------------------- handoff

HANDOFF_PROMPT = """Write a short handoff note for a human IT engineer who is taking over this ticket.
Use 3-5 bullet points: the user's problem, category and priority, what the assistant found
(or that the knowledge base had no answer), why it was escalated, and a suggested next step.
Keep it under 90 words. Plain text only.

Ticket: {ticket}
Category: {category}
Priority: {priority}
Escalation reason: {reason}
Closest knowledge base sources: {sources}
"""


def handoff_summary(ticket, category, priority, reason, chunks):
    """Return a short summary a human agent can pick up."""
    sources = ", ".join(dict.fromkeys(c["source"] for c in chunks)) or "none"
    try:
        summary = llm(HANDOFF_PROMPT.format(ticket=ticket, category=category, priority=priority,
                                            reason=reason, sources=sources))
        if summary:
            return summary
    except LLMUnavailable:
        pass
    return (
        f"- Issue: {ticket}\n"
        f"- Category / priority: {category} / {priority}\n"
        f"- Escalated because {reason}.\n"
        f"- Closest KB articles: {sources}\n"
        f"- Next step: contact the user and investigate manually."
    )
