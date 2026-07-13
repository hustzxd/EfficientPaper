#!/bin/bash

set -euo pipefail

REPO_ROOT="/Users/xiandong/projects/EfficientPaper"
PYTHON_BIN="/Users/xiandong/miniconda3/bin/python"
SCRIPT_PATH="$REPO_ROOT/scripts/search_arxiv.py"
LOG_TS_FORMAT="+%Y-%m-%d %H:%M:%S %Z"

export HOME="/Users/xiandong"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/xiandong/.local/bin"

cd "$REPO_ROOT"

# Keep the arXiv API footprint small; set env vars before running to override.
: "${WEEKLY_PAPER_USE_LLM:=1}"
: "${WEEKLY_PAPER_MAX_RESULTS:=100}"
: "${WEEKLY_PAPER_ARXIV_DELAY_SECONDS:=15}"
: "${WEEKLY_PAPER_ARXIV_PAGE_SIZE:=100}"
: "${WEEKLY_PAPER_ARXIV_RETRIES:=2}"
: "${WEEKLY_PAPER_ARXIV_RETRY_BASE_DELAY:=300}"
export WEEKLY_PAPER_USE_LLM
export WEEKLY_PAPER_MAX_RESULTS
export WEEKLY_PAPER_ARXIV_DELAY_SECONDS
export WEEKLY_PAPER_ARXIV_PAGE_SIZE
export WEEKLY_PAPER_ARXIV_RETRIES
export WEEKLY_PAPER_ARXIV_RETRY_BASE_DELAY

case "$WEEKLY_PAPER_USE_LLM" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    load_mimo_key=0
    ;;
  *)
    load_mimo_key=1
    ;;
esac

# Cron does not load interactive shell config. Pull the API key only when LLM is enabled.
if [[ "$load_mimo_key" -eq 1 && -z "${MIMO_API_KEY:-}" && -f "$HOME/.zshrc" ]]; then
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
