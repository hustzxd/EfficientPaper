if [ -n "$1" ]; then
    python scripts/add_paper.py --arxiv_id "$1"
else
    python scripts/add_paper.py
fi
