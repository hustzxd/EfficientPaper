#!/bin/bash
# Start both MkDocs and Paper Editor API server

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $MKDOCS_PID 2>/dev/null
    kill $API_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

bash refresh_readme.sh

echo -e "${BLUE}Starting Paper Editor Services...${NC}"

# Start MkDocs server
echo -e "${GREEN}Starting MkDocs on http://localhost:8000${NC}"
mkdocs serve > /tmp/mkdocs.log 2>&1 &
MKDOCS_PID=$!

# Wait a bit for MkDocs to start
sleep 2

# Start Paper Editor API server
echo -e "${GREEN}Starting Paper Editor API on http://localhost:8001${NC}"
python scripts/paper_editor_server.py > /tmp/paper_editor_api.log 2>&1 &
API_PID=$!

# Wait a bit for API to start
sleep 1

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Both servers are running!${NC}"
echo ""
echo "MkDocs (Search & Static):  http://localhost:8000"
echo "Paper Editor API:          http://localhost:8001"
echo ""
echo "Usage:"
echo "  1. Open http://localhost:8000/search/ to view papers"
echo "  2. Click 'Edit' on any paper to open the editor"
echo "  3. Make changes and save"
echo ""
echo "Logs:"
echo "  MkDocs:     tail -f /tmp/mkdocs.log"
echo "  API:        tail -f /tmp/paper_editor_api.log"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop both servers${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Wait for background processes
wait
