#!/bin/bash
# One owner for MkDocs, the editor API, and automatic refresh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${PYTHON_BIN:-python}" -u scripts/editor_supervisor.py
