"""Shared backtest-mode as-of context utility (Step B2).

Each agent that talks to an LLM in backtest mode prepends the rendered
`as_of_context.md` block to its system prompt. The block tells the model
"you are operating as of {date}; do not reference post-date events." This
module renders the block and provides a one-line helper for the
prepend-if-set pattern.
"""

from __future__ import annotations

from pathlib import Path

# Load once at import — tiny file, no reason to re-read per call.
_PROMPT_PATH = Path(__file__).parents[1] / "agents" / "prompts" / "as_of_context.md"
_AS_OF_TEMPLATE = _PROMPT_PATH.read_text()


def render_as_of_block(as_of_date: str) -> str:
    """Substitute `{as_of_date}` into the shared as-of context template.

    Caller is responsible for normalising `as_of_date` to a YYYY-MM-DD
    string before invoking this; we don't validate here because the
    template is intentionally tolerant — any string that reads as a date
    in prose is fine.
    """
    return _AS_OF_TEMPLATE.replace("{as_of_date}", as_of_date)


def maybe_inject_as_of(system_prompt: str, as_of_date: str | None) -> str:
    """Prepend the as-of context block to `system_prompt` when in backtest
    mode, leave unchanged otherwise. Production code path (`as_of_date is
    None`) returns the original prompt unmodified.
    """
    if not as_of_date:
        return system_prompt
    block = render_as_of_block(as_of_date)
    return f"{block}\n\n---\n\n{system_prompt}"
