"""Intent router for the Telegram NL-fallback path (Step 10a).

The Telegram bot has two parsing paths: deterministic slash commands
(`/drill NVDA`) and free-text natural-language messages ("what's NVDA
looking like"). This module handles the second path: a single cheap
Haiku call classifies the message into one of seven slash-command intents,
extracts the relevant args, and reports a confidence score.

The bot dispatches when `confidence >= ROUTER_CONFIDENCE_THRESHOLD` (0.7)
AND `intent != "unknown"`. Below that, the bot asks the user to clarify
rather than risk running the wrong action.

Public API:
  - `await classify(text: str) -> RouterDecision`   # the LLM call
  - `should_dispatch(decision) -> bool`             # threshold check
  - `ROUTER_CONFIDENCE_THRESHOLD` (constant, 0.7)

The threshold lives here as a module constant (rather than .env) because
it's a behaviour-shaping decision tied to prompt + schema, not a deployment
knob. Move to env if and when we want per-environment tuning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from utils import logger
from utils.models import MODEL_ROUTER
from utils.openrouter import get_client
from utils.schemas import RouterDecision

# --- Constants -------------------------------------------------------------

ROUTER_CONFIDENCE_THRESHOLD = 0.7
"""Below this, dispatch is replaced with a clarification reply. Tune via
docs/POSTPONED.md §2 trigger if low-confidence false-negatives become
common."""

LLM_MAX_TOKENS = 200
"""Router output is a small JSON object — 200 tokens is plenty and caps
the worst-case bill at ~$0.0003 per call (Haiku rates at 2026-04-26)."""

_PROMPT_PATH = Path(__file__).parent / "prompts" / "router.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


# --- Helpers ---------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite explicit instruction
    not to. Strip both single and triple backtick variants."""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text.strip()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_router_response(raw: str) -> dict:
    """Parse the LLM's JSON output. Liberal in what we accept — strip code
    fences, fall back to regex-extracting a JSON object when the LLM dropped
    extra prose (defensive against router-prompt drift)."""
    cleaned = _strip_code_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fallback: find the first JSON object inside the response. Useful when
    # the model emits "Here you go: {...}" despite the system prompt.
    m = _JSON_OBJECT_RE.search(cleaned)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _coerce_decision(raw_json: dict, fallback_text: str) -> RouterDecision:
    """Turn an LLM raw JSON dict into a validated `RouterDecision`. Any
    parse / validation failure resolves to `unknown` at confidence 0.0 so the
    bot replies with a clarification prompt. Never raises."""
    try:
        intent = raw_json.get("intent", "unknown")
        args_raw = raw_json.get("args") or {}
        confidence = float(raw_json.get("confidence", 0.0))
        # Coerce all arg values to str — the bot dispatchers expect strings.
        args = {
            str(k): str(v).strip()
            for k, v in args_raw.items()
            if v not in (None, "")
        }
        return RouterDecision(intent=intent, args=args, confidence=confidence)
    except Exception as e:
        logger.warning(
            f"[router] could not coerce LLM output to RouterDecision: {e}; "
            f"raw={raw_json!r}; falling back to unknown for input={fallback_text!r}"
        )
        return RouterDecision(intent="unknown", args={}, confidence=0.0)


# --- Public API ------------------------------------------------------------


async def classify(text: str) -> RouterDecision:
    """Classify a free-text Telegram message into one of seven slash-command
    intents. Single LLM call. Always returns a `RouterDecision` — never
    raises (network errors / parse failures resolve to `unknown` so the bot
    can at least ask for clarification)."""
    text = (text or "").strip()
    if not text:
        return RouterDecision(intent="unknown", args={}, confidence=0.0)

    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL_ROUTER,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=LLM_MAX_TOKENS,
        )
    except Exception as e:
        logger.warning(f"[router] OpenRouter call failed: {e}")
        return RouterDecision(intent="unknown", args={}, confidence=0.0)

    raw = (resp.choices[0].message.content or "").strip()
    parsed = _parse_router_response(raw)
    if not parsed:
        logger.warning(
            f"[router] LLM returned non-JSON response: {raw[:200]!r} "
            f"for input={text!r}"
        )
        return RouterDecision(intent="unknown", args={}, confidence=0.0)
    return _coerce_decision(parsed, text)


def should_dispatch(decision: RouterDecision) -> bool:
    """Whether the bot should dispatch this decision to a command handler
    or ask the user to clarify. Single source of truth for the threshold."""
    return (
        decision.intent != "unknown"
        and decision.confidence >= ROUTER_CONFIDENCE_THRESHOLD
    )
