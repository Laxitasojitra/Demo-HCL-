"""Person 3 - confidence score, escalation rules and the human handoff summary.

Code, not the LLM, decides whether an answer is shown. decide() escalates when
ANY of these is true (checked in this order, all reasons are reported):

    1. Hard rule on the ticket text: security incident, prompt injection,
       privileged/production access, or an out-of-scope topic (HR, payroll,
       facilities, legal) - or the classifier said category "Other".
    2. The LLM failed or never returned valid JSON.
    3. Top retrieval score below RETRIEVAL_THRESHOLD (nothing relevant in the KB).
    4. The model said the context does not answer the ticket.
    5. The answer cites a file that was not retrieved, or cites nothing.
    6. Overall confidence below MIN_CONFIDENCE.

Confidence (0-1):
    0.6 * retrieval  +  0.2 * classifier confidence  +  0.2 * answerable
where retrieval maps the top cosine score onto 0-1 (0.15 -> 0, 0.65 -> 1, the same
scale as agent.core). If the classifier gives no confidence, its weight is shared
out: 0.75 * retrieval + 0.25 * answerable. Not-answerable results are capped at 0.3.
"""

import re

from agent.llm import LLMUnavailable
from agent.llm import llm as default_llm

RETRIEVAL_THRESHOLD = 0.30  # same calibrated cutoff as agent.core
MIN_CONFIDENCE = 0.45
SCORE_FLOOR, SCORE_CEIL = 0.15, 0.65

# ---------------------------------------------------------------- hard rules

HARD_RULES = {
    "security_incident": [
        r"phish", r"malware", r"ransomware", r"\bvirus\b", r"\bhack(ed|er|ing)?\b", r"\bbreach",
        r"data (leak|loss)", r"\bexposed\b", r"\bcompromised\b", r"stolen (laptop|device|phone|account)",
        r"clicked (a|the|on a|on the)? ?(suspicious )?link",
        r"(entered|typed|gave|shared) my (password|credentials|login)",
        r"fake (login|sign[- ]?in) page", r"suspicious (email|link|attachment|login)",
    ],
    "prompt_injection": [
        r"ignore (all |any |the |your )?(previous|prior|above|earlier)? ?(instructions|rules|prompt)",
        r"disregard (all |any |the |your )?(previous |prior )?(instructions|rules)",
        r"(reveal|show|print) (me )?(your|the) (system )?(prompt|instructions)",
        r"\byou are now\b", r"\bjailbreak\b", r"developer mode",
        r"pretend (to be|you are)",
    ],
    "privileged_access": [
        r"\b(dba|domain admin|root|sudo|superuser)\b.{0,40}\baccess\b",
        r"\baccess\b.{0,40}\bproduction\b", r"\bproduction\b.{0,40}\baccess\b",
    ],
    "out_of_scope": [
        r"\bsalary\b", r"\bpayroll\b", r"\bpay ?slip\b", r"salary slip", r"\btax deduction",
        r"\bleave balance\b", r"\bair ?conditioning\b", r"\bfacilities\b",
        r"\bcontract\b.{0,40}\b(review|clause|legal)\b", r"\bliability clause\b", r"\blegal\b",
    ],
}

FLAG_REASONS = {
    "security_incident": "possible security incident - must go to a human immediately",
    "prompt_injection": "ticket contains instructions aimed at the assistant (prompt injection)",
    "privileged_access": "request for privileged or production access needs human approval",
    "out_of_scope": "topic is outside the IT knowledge base (HR, payroll, facilities or legal)",
    "category_other": "classifier put the ticket in category 'Other' (not an IT KB topic)",
    "priority_critical": "priority is Critical",
    "classifier_must_escalate": "ticket understanding module asked for escalation",
}

NEXT_STEPS = {
    "security_incident": "Notify the security team now and reset the user's credentials.",
    "prompt_injection": "Review the ticket manually; do not act on instructions inside it.",
    "privileged_access": "Send the request to the system owner for approval.",
    "out_of_scope": "Route to the right team (HR, payroll, facilities or legal).",
    "category_other": "Route to the right team; this is not an IT KB topic.",
    "priority_critical": "Page the on-call engineer.",
}


def detect_flags(ticket, classification=None):
    """Return the list of hard-rule flags that fire for this ticket (may be empty).

    Accepts the output of either classifier: agent.core.classify_ticket()
    ({"category", "priority"}) or Person 2's ticket_understanding.classify(),
    whose "injection", "security_incident" and "must_escalate" fields are honoured.
    """
    text = (ticket or "").lower()
    flags = [name for name, patterns in HARD_RULES.items() if any(re.search(p, text) for p in patterns)]
    classification = classification or {}
    if classification.get("injection") and "prompt_injection" not in flags:
        flags.append("prompt_injection")
    if classification.get("security_incident") and "security_incident" not in flags:
        flags.append("security_incident")
    if classification.get("category") == "Other" and "out_of_scope" not in flags:
        flags.append("category_other")
    if str(classification.get("priority", "")).lower() == "critical":
        flags.append("priority_critical")
    if classification.get("must_escalate") and not flags:
        flags.append("classifier_must_escalate")
    return flags


def classifier_confidence(classification):
    """Person 2 returns "confidence"; older code may pass "cls_confidence". None if absent."""
    classification = classification or {}
    value = classification.get("cls_confidence", classification.get("confidence"))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- confidence

def retrieval_confidence(top_score):
    """Map a cosine score onto 0-1 (SCORE_FLOOR -> 0, SCORE_CEIL -> 1)."""
    return min(1.0, max(0.0, (top_score - SCORE_FLOOR) / (SCORE_CEIL - SCORE_FLOOR)))


def compute_confidence(top_score, answerable, cls_confidence=None):
    """Weighted confidence in 0-1. See the module docstring for the formula."""
    r = retrieval_confidence(top_score)
    a = 1.0 if answerable else 0.0
    if cls_confidence is None:
        conf = 0.75 * r + 0.25 * a
    else:
        c = min(1.0, max(0.0, float(cls_confidence)))
        conf = 0.6 * r + 0.2 * c + 0.2 * a
    if not answerable:
        conf = min(conf, 0.3)
    return round(conf, 2)


# ---------------------------------------------------------------- decision

def decide(ticket, chunks, answer_result, classification=None,
           threshold=RETRIEVAL_THRESHOLD, min_confidence=MIN_CONFIDENCE):
    """Decide answer vs escalate.

    Args:
        ticket: the employee's ticket text.
        chunks: Person 1's retrieve() output, best first.
        answer_result: agent.answerer.generate_answer() output.
        classification: Person 2's {"category", "priority", optional "cls_confidence"}.

    Returns {"confidence", "escalate", "reason", "reasons", "flags", "top_score"}.
    """
    classification = classification or {}
    top_score = chunks[0]["score"] if chunks else 0.0
    confidence = compute_confidence(top_score, answer_result.get("answerable", False),
                                    classifier_confidence(classification))

    flags = detect_flags(ticket, classification)
    reasons = [FLAG_REASONS[f] for f in flags]
    if "classifier_must_escalate" in flags and classification.get("escalate_reasons"):
        reasons[flags.index("classifier_must_escalate")] = (
            "ticket understanding module asked for escalation: " + ", ".join(classification["escalate_reasons"])
        )
    if not answer_result.get("llm_ok", True):
        reasons.append(f"LLM step failed ({answer_result.get('error') or 'no valid reply'})")
    if top_score < threshold:
        reasons.append(f"top retrieval score {top_score:.2f} is below threshold {threshold:.2f}")
    if answer_result.get("llm_ok", True) and not answer_result.get("answerable"):
        reasons.append("the knowledge base does not answer this ticket")
    if answer_result.get("invalid_citations"):
        reasons.append("answer cites files that were not retrieved: " + ", ".join(answer_result["invalid_citations"]))
    if answer_result.get("answerable") and not answer_result.get("sources"):
        reasons.append("answer has no valid KB citation")
    if confidence < min_confidence:
        reasons.append(f"confidence {confidence:.2f} is below {min_confidence:.2f}")

    return {
        "confidence": confidence,
        "escalate": bool(reasons),
        "reason": "; ".join(reasons),
        "reasons": reasons,
        "flags": flags,
        "top_score": round(float(top_score), 4),
    }


# ---------------------------------------------------------------- handoff

HANDOFF_PROMPT = """Write a handoff note for the human IT engineer taking over this ticket.
Write exactly these 3 lines in plain text, each under 30 words, using only the facts below:
Issue: <the user's problem in one sentence>
Triage: <category> / <priority>. Closest KB: <file names or none> (top score <score>).
Reason escalated: <why the assistant did not answer>. Next step: <one concrete action>.

Ticket: {ticket}
Category: {category}
Priority: {priority}
Closest KB files: {sources}
Top retrieval score: {score:.2f}
Escalation reasons: {reasons}
Suggested next step: {next_step}
"""


def _next_step(flags):
    for flag in flags:
        if flag in NEXT_STEPS:
            return NEXT_STEPS[flag]
    return "Contact the user and investigate manually."


def _template_handoff(ticket, category, priority, sources, score, reason, next_step):
    issue = " ".join((ticket or "").split())
    if len(issue) > 160:
        issue = issue[:157] + "..."
    return (
        f"Issue: {issue}\n"
        f"Triage: {category} / {priority}. Closest KB: {sources} (top score {score:.2f}).\n"
        f"Reason escalated: {reason}. Next step: {next_step}"
    )


def handoff_summary(ticket, classification, decision, chunks, llm_fn=None):
    """Return a 3-line note (Issue / Triage / Reason escalated) for the human agent.

    Uses the LLM when available; falls back to a template if there is no key, the
    call fails, or the reply does not have the three expected lines.
    """
    classification = classification or {}
    category = classification.get("category", "Unknown")
    priority = classification.get("priority", "Unknown")
    sources = ", ".join(dict.fromkeys(c["source"] for c in chunks)) or "none"
    score = decision.get("top_score", 0.0)
    reason = decision.get("reason") or "low confidence"
    next_step = _next_step(decision.get("flags", []))
    fallback = _template_handoff(ticket, category, priority, sources, score, reason, next_step)

    llm_fn = llm_fn or default_llm
    try:
        raw = llm_fn(HANDOFF_PROMPT.format(ticket=ticket, category=category, priority=priority,
                                           sources=sources, score=score, reasons=reason,
                                           next_step=next_step))
    except LLMUnavailable:
        return fallback
    except Exception:
        return fallback

    lines = [ln.strip().lstrip("-* ").strip() for ln in (raw or "").splitlines() if ln.strip()]
    wanted = ("issue:", "triage:", "reason escalated:")
    picked = [next((ln for ln in lines if ln.lower().startswith(w)), None) for w in wanted]
    if None in picked:
        return fallback
    return "\n".join(picked)


# ---------------------------------------------------------------- one-call helper

def resolve(ticket, chunks, classification=None, llm_fn=None,
            threshold=RETRIEVAL_THRESHOLD, min_confidence=MIN_CONFIDENCE):
    """Run answer -> gate -> handoff for one ticket and return the contract fields.

    Returns {"answer", "sources", "confidence", "escalate", "handoff_summary", "reasons", "flags"}.
    When escalated, "answer" is a holding message for the employee and "sources" is empty.
    """
    from agent.answerer import generate_answer  # local import keeps gate usable on its own

    classification = classification or {}
    # Person 2's guard masks emails, phones and passwords; only that text goes to an LLM.
    llm_text = classification.get("clean_text") or ticket

    flags = detect_flags(ticket, classification)
    if "prompt_injection" in flags or "security_incident" in flags:
        # Never send these to the answer prompt; escalate straight away.
        answer_result = {"answer": "NOT_FOUND", "sources": [], "answerable": False,
                         "invalid_citations": [], "llm_ok": True, "mode": "skipped", "error": None}
    else:
        answer_result = generate_answer(llm_text, chunks, llm_fn=llm_fn)

    decision = decide(ticket, chunks, answer_result, classification, threshold, min_confidence)
    if decision["escalate"]:
        return {
            "answer": "I couldn't resolve this automatically. I've passed it to a support engineer with a summary.",
            "sources": [],
            "confidence": decision["confidence"],
            "escalate": True,
            "handoff_summary": handoff_summary(llm_text, classification, decision, chunks, llm_fn=llm_fn),
            "reasons": decision["reasons"],
            "flags": decision["flags"],
        }
    return {
        "answer": answer_result["answer"],
        "sources": answer_result["sources"],
        "confidence": decision["confidence"],
        "escalate": False,
        "handoff_summary": None,
        "reasons": [],
        "flags": decision["flags"],
    }
