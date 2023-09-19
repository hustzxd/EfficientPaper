
if [ -z "$1" ]; then
    echo "Use random name"
    python scripts/add_paper.py
else
    python scripts/add_paper.py --name $1
fi
