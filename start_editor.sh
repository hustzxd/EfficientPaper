#!/bin/bash
# Start both MkDocs and Paper Editor API server with auto-reload

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# PID file for MkDocs
MKDOCS_PID_FILE="/tmp/mkdocs_editor.pid"

# Function to start MkDocs
start_mkdocs() {
    echo -e "${GREEN}Starting MkDocs on http://localhost:8000${NC}"
    mkdocs serve > /tmp/mkdocs.log 2>&1 &
    MKDOCS_PID=$!
    echo $MKDOCS_PID > $MKDOCS_PID_FILE
    sleep 2
}

# Function to restart MkDocs
restart_mkdocs() {
    echo -e "${YELLOW}Restarting MkDocs...${NC}"
    if [ -f $MKDOCS_PID_FILE ]; then
        OLD_PID=$(cat $MKDOCS_PID_FILE)
        kill $OLD_PID 2>/dev/null
        sleep 1
    fi
    start_mkdocs
    echo -e "${GREEN}MkDocs restarted!${NC}"
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers and file watcher..."
    if [ -f $MKDOCS_PID_FILE ]; then
        kill $(cat $MKDOCS_PID_FILE) 2>/dev/null
        rm -f $MKDOCS_PID_FILE
    fi
    kill $API_PID 2>/dev/null
    kill $WATCHER_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Initial generation
bash refresh_and_upload.sh

echo -e "${BLUE}Starting Paper Editor Services...${NC}"

# Start MkDocs server
start_mkdocs

# Start Paper Editor API server
echo -e "${GREEN}Starting Paper Editor API on http://localhost:8001${NC}"
python scripts/paper_editor_server.py > /tmp/paper_editor_api.log 2>&1 &
API_PID=$!

# Wait a bit for API to start
sleep 1

# Start file watcher for auto-reload (using Python watchdog)
echo -e "${GREEN}Starting file watcher for auto-reload...${NC}"
python3 -c '
import time
import subprocess
import sys
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog for file monitoring...")
    subprocess.run([sys.executable, "-m", "pip", "install", "watchdog"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = time.time()
        self.debounce_seconds = 2

    def on_modified(self, event):
        if event.is_directory:
            return

        # Only watch .prototxt and .md files
        if not (event.src_path.endswith(".prototxt") or event.src_path.endswith(".md")):
            return

        # Debounce rapid changes
        current_time = time.time()
        if (current_time - self.last_modified) < self.debounce_seconds:
            return

        self.last_modified = current_time
        print(f"\033[1;33mFile change detected: {event.src_path}\033[0m")
        print("\033[1;33mRegenerating...\033[0m")
        subprocess.run(["bash", "refresh_and_upload.sh"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("\033[0;32mRegeneration complete!\033[0m")

        # Restart MkDocs to reload changes
        print("\033[1;33mRestarting MkDocs...\033[0m")
        try:
            with open("/tmp/mkdocs_editor.pid", "r") as f:
                old_pid = int(f.read().strip())
                subprocess.run(["kill", str(old_pid)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)
        except:
            pass

        # Start new MkDocs instance
        proc = subprocess.Popen(["mkdocs", "serve"],
                               stdout=open("/tmp/mkdocs.log", "w"),
                               stderr=subprocess.STDOUT)
        with open("/tmp/mkdocs_editor.pid", "w") as f:
            f.write(str(proc.pid))
        time.sleep(2)
        print("\033[0;32mMkDocs restarted!\033[0m")

# Set up observer
observer = Observer()
handler = ChangeHandler()

# Watch meta and notes directories
meta_path = Path("meta")
notes_path = Path("notes")

if meta_path.exists():
    observer.schedule(handler, str(meta_path), recursive=True)
if notes_path.exists():
    observer.schedule(handler, str(notes_path), recursive=True)

observer.start()
print("\033[0;32mFile watcher started for meta/ and notes/\033[0m")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
' &
WATCHER_PID=$!

sleep 1

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All services are running!${NC}"
echo ""
echo "MkDocs (Search & Static):  http://localhost:8000"
echo "Paper Editor API:          http://localhost:8001"
echo -e "${GREEN}Auto-reload:               Enabled ✓${NC}"
echo ""
echo "Usage:"
echo "  1. Open http://localhost:8000 to view papers"
echo "  2. Click 'Edit Note' on any paper note page"
echo "  3. Make changes and save - auto-reload will regenerate"
echo ""
echo "Watched directories:"
echo "  - meta/    (prototxt files)"
echo "  - notes/   (markdown files)"
echo ""
echo "Logs:"
echo "  MkDocs:     tail -f /tmp/mkdocs.log"
echo "  API:        tail -f /tmp/paper_editor_api.log"
echo "  Refresh:    tail -f /tmp/refresh.log"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Wait for background processes
wait
