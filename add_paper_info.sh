if [ -n "$1" ]; then
    python scripts/add_paper.py -f "$1"
else
    python scripts/add_paper.py
fi