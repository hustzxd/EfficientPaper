#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CURRENT_DIR="$PWD"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${CURRENT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

protoc -I="${CURRENT_DIR}/proto" --python_out="${CURRENT_DIR}/proto" "${CURRENT_DIR}/proto/efficient_paper.proto"
"$PYTHON_BIN" proto/gene_template.py

"$PYTHON_BIN" scripts/ddl.py
"$PYTHON_BIN" scripts/split_by_year.py
"$PYTHON_BIN" scripts/generate_baseline_methods_graph.py
"$PYTHON_BIN" scripts/generate_search_data.py
"$PYTHON_BIN" scripts/generate_readme_pages.py

if [ -z "${1:-}" ]; then
    echo "Refresh done"
else
    echo "Upload changes to github"
    git add .
    git commit -m "$1"
    git push
    # Build site with MkDocs
    mkdocs build
    # Use custom build script that copies notes/ and meta/ directories
    ./build_and_deploy.sh
fi

