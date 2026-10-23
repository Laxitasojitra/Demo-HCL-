"""
llm.py - one LLM wrapper for the whole team (Groq / Gemini / OpenAI).

All three providers expose an OpenAI-compatible chat endpoint, so we use the
`openai` SDK with a different base_url per provider. One code path, three providers.

Config (environment variables):
    LLM_PROVIDER   groq | gemini | openai | mock
                   If unset (or set to a provider not listed here, e.g. anthropic) and
                   this runs inside the full app, calls go through agent/llm.py so the
                   whole pipeline uses one provider. Standalone, the default is groq.
    LLM_MODEL      optional model override
    GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY
    LLM_TIMEOUT    seconds, default 20

    LLM_PROVIDER=mock makes every call raise LLMError, which forces all callers
    onto their fallbacks. Use it to test offline or to rehearse an API outage.

Usage:
    from llm import chat, chat_json, LLMError
    data = chat_json(system_prompt, user_prompt, required_keys=("category", "priority"))
    text = chat(system_prompt, user_prompt)
"""
import json
import os
import re

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}


class LLMError(Exception):
    """Raised when the LLM is unavailable or keeps returning unusable output.
    Callers should catch this and use their fallback."""


_client = None


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "").strip().lower()


def model_name() -> str:
    return os.getenv("LLM_MODEL") or PROVIDERS.get(provider() or "groq", {}).get("model", "")


def _own_provider_ready() -> bool:
    """True when LLM_PROVIDER names one of our providers and its key is set."""
    p = provider()
    return p in PROVIDERS and bool(os.getenv(PROVIDERS[p]["key_env"]))


def _team_llm():
    """The repo-wide wrapper agent/llm.py, if we are running inside the full app."""
    try:
        from agent.llm import llm as team_llm
    except ImportError:
        return None
    return team_llm


def _get_client():
    global _client
    if _client is not None:
        return _client
    p = provider() or "groq"
    if p not in PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{p}'. Use one of: {', '.join(PROVIDERS)} or mock.")
    cfg = PROVIDERS[p]
    key = os.getenv(cfg["key_env"])
    if not key:
        raise LLMError(f"{cfg['key_env']} is not set.")
    try:
        from openai import OpenAI  # imported lazily so offline/mock mode needs no SDK
    except ImportError as e:
        raise LLMError("pip install openai") from e
    _client = OpenAI(
        api_key=key,
        base_url=cfg["base_url"],
        timeout=float(os.getenv("LLM_TIMEOUT", "20")),
        max_retries=1,  # one network-level retry; JSON retry is handled below
    )
    return _client


def chat(system: str, user: str, *, json_mode: bool = False, max_tokens: int = 800) -> str:
    """Single chat completion at temperature 0. Returns the raw text."""
    if provider() == "mock":
        raise LLMError("mock provider: LLM disabled")
    if not _own_provider_ready():
        # Inside the full app: use the same provider/model as the rest of the pipeline.
        team_llm = _team_llm()
        if team_llm is not None:
            try:
                return (team_llm(f"{system}\n\n{user}") or "").strip()
            except Exception as e:  # LLMUnavailable (no key), network errors, ...
                raise LLMError(f"{type(e).__name__}: {e}") from e
    client = _get_client()
    kwargs = dict(
        model=model_name(),
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Some models/providers reject response_format; try once more without it.
        if json_mode and "response_format" in str(e).lower():
            kwargs.pop("response_format", None)
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e2:
                raise LLMError(f"{type(e2).__name__}: {e2}") from e2
        else:
            raise LLMError(f"{type(e).__name__}: {e}") from e
    return (resp.choices[0].message.content or "").strip()


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or stray prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("no JSON object found")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e.msg}") from e
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


def chat_json(system: str, user: str, *, required_keys=(), max_tokens: int = 800) -> dict:
    """Chat call that must return a JSON object containing `required_keys`.
    Retries once with a correction message on bad JSON; raises LLMError after that."""
    prompt = user
    last_error = None
    for _ in range(2):  # first try + one retry
        raw = chat(system, prompt, json_mode=True, max_tokens=max_tokens)
        try:
            data = parse_json(raw)
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"missing keys: {missing}")
            return data
        except ValueError as e:
            last_error = e
            prompt = (
                f"{user}\n\nYour previous reply was invalid ({e}). "
                "Reply again with ONLY one valid JSON object and no other text."
            )
    raise LLMError(f"Bad JSON after retry: {last_error}")


if __name__ == "__main__":
    # Quick smoke test: python llm.py
    try:
        print(chat_json("You reply in JSON.", 'Return JSON {"ok": true}', required_keys=("ok",)))
    except LLMError as e:
        print("LLMError:", e)
