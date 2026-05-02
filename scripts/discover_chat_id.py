"""Discover your Telegram chat_id.

Telegram bots only know about chat_ids that have already messaged them.
This script polls `getUpdates` once and prints the chat_id of every sender
in the bot's recent message buffer — paste the one matching your account
into `.env` as `TELEGRAM_CHAT_ID`.

Prereqs (Step 10 setup, Telegram side):
  1. Created a bot via @BotFather and copied the token
  2. Pasted the token into `.env` as `TELEGRAM_BOT_TOKEN=...`
  3. Sent at least one message to your bot (e.g. "hi") from the account
     you want allowlisted

Usage:
  python -m scripts.discover_chat_id

Notes:
- Telegram only retains updates for ~24h via long-poll. If too long has
  passed since you messaged the bot, send a fresh message and retry.
- If a webhook is configured for the bot, `getUpdates` returns 409 and you
  must remove the webhook first (`/deleteWebhook` from BotFather isn't a
  thing — POST to `https://api.telegram.org/bot<TOKEN>/deleteWebhook`).
"""

from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env", file=sys.stderr)
        print(
            "Paste the token BotFather gave you into .env as "
            "`TELEGRAM_BOT_TOKEN=7891234567:AAFa...`",
            file=sys.stderr,
        )
        return 1

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        resp = httpx.get(url, timeout=10.0)
    except Exception as e:
        print(f"ERROR: could not reach Telegram API: {e}", file=sys.stderr)
        return 1

    if resp.status_code == 401:
        print(
            "ERROR: Telegram rejected the token. Double-check you copied it "
            "exactly from BotFather (no leading/trailing whitespace).",
            file=sys.stderr,
        )
        return 1
    if resp.status_code == 409:
        print(
            "ERROR: a webhook is currently set on this bot — getUpdates is "
            "blocked. Remove it with:\n"
            f"  curl -X POST https://api.telegram.org/bot{token[:8]}.../deleteWebhook",
            file=sys.stderr,
        )
        return 1
    if resp.status_code != 200:
        print(
            f"ERROR: Telegram returned {resp.status_code}: {resp.text}",
            file=sys.stderr,
        )
        return 1

    data = resp.json()
    if not data.get("ok"):
        print(
            f"ERROR: Telegram error: {data.get('description', 'unknown')}",
            file=sys.stderr,
        )
        return 1

    updates = data.get("result", [])
    if not updates:
        print("No recent messages found.")
        print()
        print(
            "Open Telegram, find your bot by its @username, and send any "
            "message to it (e.g. 'hi'). Then re-run this script."
        )
        return 1

    # Distinct (chat_id, label) pairs across all updates we can see. Most
    # users will only have one — themselves — but the loop is defensive in
    # case you tested from a second account.
    seen: dict[int, str] = {}
    for update in updates:
        msg = update.get("message") or update.get("edited_message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None or chat_id in seen:
            continue
        label = (
            chat.get("username")
            or chat.get("first_name")
            or chat.get("title")
            or "?"
        )
        seen[chat_id] = label

    if not seen:
        print(
            "Updates found but none carried chat info. Send a fresh message "
            "to the bot and retry."
        )
        return 1

    print("Senders that have messaged your bot recently:")
    print()
    for chat_id, label in seen.items():
        print(f"  {label}: {chat_id}")
    print()
    if len(seen) == 1:
        only_id = next(iter(seen))
        print("Paste this into .env (replacing any existing TELEGRAM_CHAT_ID):")
        print()
        print(f"  TELEGRAM_CHAT_ID={only_id}")
    else:
        print(
            "Multiple chats found above — paste the one matching your "
            "personal account into .env as TELEGRAM_CHAT_ID."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
