"""`utils/__init__.py` logging hygiene."""

from __future__ import annotations

import logging


def test_httpx_request_lines_are_not_logged_at_info():
    """httpx logs full request URLs at INFO; the Telegram Bot API embeds
    the bot token in the URL path, so the heartbeat log was persisting
    the secret on every cycle. Importing `utils` must raise httpx's
    logger to WARNING."""
    import utils  # noqa: F401  (import side effect under test)

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
