"""Optional LLM rewrite of the top-3 errors into a short coaching note.

Rules decide the errors; the LLM may only rephrase them, never add any. Disabled unless
FEEDBACK_LLM=1 and Anthropic credentials are available. Any failure falls back to None.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

MODEL = os.environ.get("FEEDBACK_MODEL", "claude-opus-5-5")

SYSTEM = (
    "You are a friendly guitar teacher. You receive a JSON list of the top problems a rules engine "
    "found in a student's practice take. Write a short coaching note (max 90 words, plain text, no "
    "markdown headings) that explains what to fix first and one concrete drill for it. Only mention "
    "problems that are in the list. Do not invent new problems, numbers, chords, strings or fingers."
)


def coaching_note(exercise_title: str, score: float, top: list[dict]) -> str | None:
    if os.environ.get("FEEDBACK_LLM") != "1" or not top:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        payload = {"exercise": exercise_title, "score": score,
                   "problems": [{k: t.get(k) for k in ("type", "chord", "string", "finger", "count", "message")}
                                for t in top]}
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        if resp.stop_reason == "refusal":
            return None
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
    except Exception:
        log.exception("coaching note generation failed")
        return None
