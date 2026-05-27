#!/bin/bash

set -euo pipefail

REPO_ROOT="/Users/xiandong/projects/EfficientPaper"
PYTHON_BIN="/Users/xiandong/miniconda3/bin/python"
SCRIPT_PATH="$REPO_ROOT/scripts/search_arxiv.py"
LOG_TS_FORMAT="+%Y-%m-%d %H:%M:%S %Z"

export HOME="/Users/xiandong"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/xiandong/.local/bin"

cd "$REPO_ROOT"

# Cron does not load interactive shell config. Pull only the API key we need.
if [[ -z "${MIMO_API_KEY:-}" && -f "$HOME/.zshrc" ]]; then
  key_line="$(grep -m1 '^export MIMO_API_KEY=' "$HOME/.zshrc" || true)"
  if [[ -n "$key_line" ]]; then
    MIMO_API_KEY="${key_line#export MIMO_API_KEY=}"
    MIMO_API_KEY="${MIMO_API_KEY#\"}"
    MIMO_API_KEY="${MIMO_API_KEY%\"}"
    export MIMO_API_KEY
  fi
fi

echo "[$(date "$LOG_TS_FORMAT")] weekly_paper.sh start"
echo "[$(date "$LOG_TS_FORMAT")] cwd=$PWD"

"$PYTHON_BIN" "$SCRIPT_PATH"

echo "[$(date "$LOG_TS_FORMAT")] weekly_paper.sh done"
