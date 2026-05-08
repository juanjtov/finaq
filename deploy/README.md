# FINAQ deployment notes

This folder holds the platform-specific service definitions FINAQ
needs to run as a daemonised personal tool.

## CIO heartbeat — macOS (launchd)

The CIO planner runs twice a day on the user's Mac via launchd. The
plist at `launchd/com.finaq.cio.plist` is the canonical schedule:

  - **5am + 1pm PT** (= 8am + 4pm ET — US market open + close).
  - **`RunAtLoad=true`** — fires one cycle on boot / login / `launchctl load`.
    Combined with the dispatcher's freshness check
    (`cio.dispatcher._resolve_auto_mode`), this means a missed slot
    (lid was closed) triggers a `catchup` cycle on next wake — *not* a
    duplicate `heartbeat`.

### One-time install

```sh
cd /Users/jujot/developer/finaq

# 1. Make sure .venv is built and dependencies are installed.
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Backfill the synthesis_reports collection so the CIO has prior
#    drill-ins to RAG over. (Idempotent; safe to re-run.)
.venv/bin/python -m scripts.index_existing_reports

# 3. Sanity-check the dispatcher in dry mode (run on-demand for one ticker
#    that's already drilled, so the planner sees evidence). This prints
#    Telegram + Notion soft-fails if those env vars aren't set yet.
.venv/bin/python -m cio.dispatcher --mode on_demand --ticker NVDA --thesis ai_cake

# 4. Copy the plist into your LaunchAgents folder + load it.
cp deploy/launchd/com.finaq.cio.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.finaq.cio.plist
```

### Verifying the schedule

```sh
launchctl list | grep finaq
# → ID/PID/Status row; PID is the most recent run, "-" if none.

# Force one cycle manually (it'll choose heartbeat or catchup based on
# the freshness gate):
launchctl start com.finaq.cio

# Tail the logs:
tail -f data_cache/cio.log     # stdout
tail -f data_cache/cio.err.log # stderr
```

The dispatcher logs each step to stderr (`logger.info`), so the err log
is the canonical place to grep for "what did the cycle decide?".

### Editing the schedule

If you change the cadence in the plist, **unload + reload** for it to
take effect (launchd caches the parsed config in memory):

```sh
launchctl unload ~/Library/LaunchAgents/com.finaq.cio.plist
launchctl load   ~/Library/LaunchAgents/com.finaq.cio.plist
```

### Uninstall

```sh
launchctl unload ~/Library/LaunchAgents/com.finaq.cio.plist
rm ~/Library/LaunchAgents/com.finaq.cio.plist
```

The state.db rows (`cio_runs`, `cio_actions`) stay on disk — Mission
Control will keep showing the historical cycles until you wipe
`data_cache/state.db`.

## Catch-up behaviour explainer

Three signals interact at boot / wake:

  1. `RunAtLoad=true` fires one launchd invocation.
  2. `cio.dispatcher --mode auto` reads `state.db.cio_runs` for the most
     recent successful cycle's `ended_at`.
  3. If `now - ended_at > 8h` (the `CATCHUP_THRESHOLD_HOURS` constant in
     `cio/dispatcher.py`), the dispatcher chooses `catchup`; otherwise
     `heartbeat`.

This gives "fire one catch-up if we missed a slot" without stacking
N missed cycles.

## Linux (systemd) — Step 12

When the system migrates to a DigitalOcean droplet, the equivalent unit
files will land in `deploy/systemd/`:

  - `finaq-cio.service` — single shot of `python -m cio.dispatcher --mode auto`.
  - `finaq-cio.timer` — schedule (twice daily, plus `OnBootSec=` for catch-up).

That's deferred to Step 12 of the build plan.
