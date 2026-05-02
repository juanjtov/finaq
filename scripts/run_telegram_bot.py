"""Long-running Telegram bot entrypoint.

Run with:  python -m scripts.run_telegram_bot

Reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from `.env`. Long-poll
mode means the script holds an open connection to Telegram's servers and
processes incoming messages as they arrive — no public URL is required
for the bot itself.

Stop with Ctrl-C. In Step 12 this becomes a systemd unit on the droplet.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    from data.telegram import run as run_bot

    run_bot()


if __name__ == "__main__":
    main()
