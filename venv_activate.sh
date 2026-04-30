#!/bin/bash
# venv_activate.sh — activate .venv, load .env, install deps via uv when stale.
# Source this; do not execute it.
#   source venv_activate.sh             # activate; reinstall only if requirements.txt is newer than stamp
#   source venv_activate.sh --install   # force reinstall regardless of stamp

if [ ! -f .venv/bin/activate ]; then
  echo "aenv: no .venv/bin/activate in $(pwd)" >&2
  return 1 2>/dev/null || exit 1
fi

source .venv/bin/activate

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

stamp=".venv/.requirements.stamp"
if [ -f requirements.txt ]; then
  if [ "$1" = "--install" ] || [ ! -f "$stamp" ] || [ requirements.txt -nt "$stamp" ]; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "aenv: uv not found on PATH — install with: brew install uv" >&2
      return 1 2>/dev/null || exit 1
    fi
    echo "aenv: installing requirements via uv"
    uv pip install -r requirements.txt && touch "$stamp"
  fi
fi

echo "aenv: ready ($(python --version 2>&1), $(pwd))"
