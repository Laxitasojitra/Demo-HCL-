"""One small wrapper around the LLM so the provider can be swapped.

Pick the provider with LLM_PROVIDER (anthropic | openai | gemini) or let it
be auto-detected from whichever API key is set. Override the model with
LLM_MODEL. If no key is configured, llm() raises LLMUnavailable and the
agent falls back to simple offline heuristics so the demo still runs.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}
KEY_VARS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


class LLMUnavailable(RuntimeError):
    pass


def active_provider():
    """Return the provider name to use, or None if no key is configured."""
    chosen = os.getenv("LLM_PROVIDER", "").strip().lower()
    if chosen:
        return chosen if chosen in KEY_VARS else None
    for name, keys in KEY_VARS.items():
        if any(os.getenv(k) for k in keys):
            return name
    return None


def llm(prompt: str) -> str:
    """Send one prompt, return the model's text reply."""
    provider = active_provider()
    if provider is None:
        raise LLMUnavailable("No LLM API key found; set ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY.")
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            # Re-run on a fallback model if a safety classifier declines the request.
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default", "output_config": {"effort": "low"}},
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()

    if provider == "openai":
        from openai import OpenAI

        response = OpenAI().chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()

    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        response = genai.Client(api_key=api_key).models.generate_content(model=model, contents=prompt)
        return (response.text or "").strip()

    raise LLMUnavailable(f"Unknown LLM_PROVIDER: {provider}")
