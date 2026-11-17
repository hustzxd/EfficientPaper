"""Own the editor processes and coalesce source updates before rebuilding."""

import fcntl
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("editor")


def source_snapshot(root):
    """Polling catches atomic replacements and directory moves on macOS too.

    Only inputs are watched: generated docs and the conference timeline must
    not trigger the generator again. MkDocs still watches other docs itself.
    """
    result = {}
    for directory, suffixes in (
        ("meta", {".prototxt"}),
        ("notes", {".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}),
        ("docs/weekly_paper", {".md"}),
    ):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if path.suffix.lower() not in suffixes or relative.as_posix() == "notes/conference_timeline.png":
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # An atomic save or archive may be in progress.
            if path.is_file():
                result[relative.as_posix()] = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    return result


class PendingChanges:
    def __init__(self, snapshot, debounce=2.0):
        self.snapshot = snapshot
        self.debounce = debounce
        self.paths = set()
        self.deadline = 0.0

    def observe(self, snapshot, now):
        changed = {p for p in self.snapshot.keys() | snapshot.keys()
                   if self.snapshot.get(p) != snapshot.get(p)}
        self.snapshot = snapshot
        if changed:
            self.paths.update(changed)
            self.deadline = now + self.debounce

    def take(self, now):
        if not self.paths or now < self.deadline:
            return set()
        paths, self.paths = self.paths, set()
        return paths

    def retry(self, paths, now):
        self.paths.update(paths)
        self.deadline = now + 30.0


def require_free_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


class Supervisor:
    def __init__(self, root=ROOT, log_dir=Path("/tmp")):
        self.root = root
        self.log_dir = log_dir
        self.pid_file = log_dir / "mkdocs_editor.pid"
        self.stopping = threading.Event()
        self.children = {}
        self.python = sys.executable

    def launch(self, name, command, logfile):
        # Servers stay in the panel's group. Refresh gets its own group so
        # timeout/shutdown can stop the entire generator pipeline.
        with (self.log_dir / logfile).open("a", buffering=1) as output:
            output.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} Starting {name} ---\n")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHON_BIN"] = self.python
            env["PATH"] = str(Path(self.python).parent) + os.pathsep + env.get("PATH", "")
            process = subprocess.Popen(command, cwd=self.root, env=env,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=(name == "refresh"))
        self.children[name] = process
        return process

    def stop_child(self, name):
        process = self.children.get(name)
        if process is None:
            return
        if name == "refresh":
            # Bash can exit before its foreground generator. Signal the owned
            # group even if the direct child has already exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                process.poll()  # Reap Bash while checking for descendants.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
        elif process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                LOG.warning("%s did not stop; killing owned child", name)
                process.kill()
                process.wait(timeout=1)
        else:
            process.wait()
        del self.children[name]
        if name == "mkdocs":
            self.pid_file.unlink(missing_ok=True)

    def wait_ready(self, name, url, timeout=90):
        process = self.children[name]
        deadline = time.monotonic() + timeout
        while not self.stopping.is_set() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with code {process.returncode}; see its log")
            try:
                with urlopen(url, timeout=1) as response:
                    if response.status == 200 and process.poll() is None:
                        return
            except (URLError, TimeoutError, OSError):
                pass
            self.stopping.wait(0.2)
        raise RuntimeError(f"{name} did not become ready at {url}")

    def start_mkdocs(self):
        require_free_port(8000)
        process = self.launch("mkdocs", [self.python, "-m", "mkdocs", "serve"], "mkdocs.log")
        try:
            self.wait_ready("mkdocs", "http://127.0.0.1:8000/")
        except Exception:
            self.stop_child("mkdocs")
            raise
        self.pid_file.write_text(f"{process.pid}\n")
        LOG.info("MkDocs ready on http://localhost:8000 (PID %s)", process.pid)

    def regenerate(self, timeout=300):
        LOG.info("Regenerating paper data; output: %s", self.log_dir / "refresh.log")
        process = self.launch("refresh", ["bash", "refresh_and_upload.sh"], "refresh.log")
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if self.stopping.wait(0.2):
                self.stop_child("refresh")
                raise RuntimeError("Refresh interrupted by shutdown")
            if time.monotonic() >= deadline:
                self.stop_child("refresh")
                LOG.error("Refresh exceeded %s seconds; keeping current MkDocs and retrying", timeout)
                return False
        self.stop_child("refresh")
        if process.returncode:
            LOG.error("Refresh failed (exit %s); keeping current MkDocs. See %s",
                      process.returncode, self.log_dir / "refresh.log")
            return False
        LOG.info("Regeneration complete")
        return True

    def run(self):
        # Lock is kept open for the lifetime of the supervisor. Do not unlink
        # the lock file: competing invocations must lock the same inode.
        with (self.log_dir / "efficientpaper_editor.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another EfficientPaper supervisor is already running") from exc
            try:
                require_free_port(8000)
                require_free_port(8001)
                pending = PendingChanges(source_snapshot(self.root))
                if not self.regenerate():
                    raise RuntimeError("Initial refresh failed; editor was not started")
                if self.stopping.is_set():
                    return
                self.start_mkdocs()
                self.launch("api", [self.python, "scripts/paper_editor_server.py"], "paper_editor_api.log")
                self.wait_ready("api", "http://127.0.0.1:8001/api/get-keywords")
                LOG.info("Editor ready. Watching meta/, notes/, docs/weekly_paper/ every 0.5s")
                while not self.stopping.wait(0.5):
                    for name in ("mkdocs", "api"):
                        process = self.children[name]
                        if process.poll() is not None:
                            raise RuntimeError(f"{name} exited unexpectedly ({process.returncode}); see its log")
                    pending.observe(source_snapshot(self.root), time.monotonic())
                    paths = pending.take(time.monotonic())
                    if not paths:
                        continue
                    LOG.info("Source changes: %s", ", ".join(sorted(paths)))
                    if any(p.startswith(("meta/", "notes/")) for p in paths):
                        if not self.regenerate():
                            pending.retry(paths, time.monotonic())
                            continue
                    if self.stopping.is_set():
                        break
                    LOG.info("Restarting owned MkDocs process")
                    self.stop_child("mkdocs")
                    self.start_mkdocs()
                    # Do not reset the snapshot here: edits during generation
                    # or startup must be picked up by the next polling pass.
            finally:
                for name in ("refresh", "api", "mkdocs"):
                    self.stop_child(name)
                LOG.info("Editor services stopped")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    supervisor = Supervisor()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: supervisor.stopping.set())
    try:
        supervisor.run()
    except Exception:
        if supervisor.stopping.is_set():
            return 0
        LOG.exception("Editor supervisor failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
