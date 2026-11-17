"""Isolated regression tests; never bind the live editor ports or edit papers."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from editor_supervisor import PendingChanges, Supervisor, require_free_port, source_snapshot


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("meta", "notes", "docs/weekly_paper", "logs"):
            (self.root / directory).mkdir(parents=True)
        self.supervisor = Supervisor(self.root, self.root / "logs")
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for name in list(self.supervisor.children):
            self.supervisor.stop_child(name)

    def test_create_modify_atomic_replace_delete_and_directory_move(self):
        pending = PendingChanges(source_snapshot(self.root))
        paper = self.root / "meta/paper.prototxt"
        paper.write_text("first")
        pending.observe(source_snapshot(self.root), 0)
        self.assertFalse(pending.take(1.9))
        paper.write_text("second edit")
        pending.observe(source_snapshot(self.root), 1)
        self.assertFalse(pending.take(2))
        self.assertEqual(pending.take(3), {"meta/paper.prototxt"})
        replacement = self.root / "meta/save.tmp"
        replacement.write_text("second edit")
        os.replace(replacement, paper)
        pending.observe(source_snapshot(self.root), 4)
        self.assertEqual(pending.take(6), {"meta/paper.prototxt"})
        folder = self.root / "notes/old"
        folder.mkdir()
        (folder / "note.md").write_text("note")
        pending.observe(source_snapshot(self.root), 7)
        pending.take(9)
        folder.rename(self.root / "notes/new")
        paper.unlink()
        pending.observe(source_snapshot(self.root), 10)
        self.assertEqual(pending.take(12), {
            "meta/paper.prototxt", "notes/old/note.md", "notes/new/note.md"})

    def test_weekly_create_archive_delete(self):
        pending = PendingChanges(source_snapshot(self.root))
        weekly = self.root / "docs/weekly_paper/2026-11-13.md"
        weekly.write_text("weekly")
        pending.observe(source_snapshot(self.root), 0)
        self.assertEqual(pending.take(2), {"docs/weekly_paper/2026-11-13.md"})
        archive = weekly.parent / "legacy"
        archive.mkdir()
        weekly.rename(archive / weekly.name)
        pending.observe(source_snapshot(self.root), 3)
        self.assertEqual(pending.take(5), {
            "docs/weekly_paper/2026-11-13.md", "docs/weekly_paper/legacy/2026-11-13.md"})
        (archive / weekly.name).unlink()
        pending.observe(source_snapshot(self.root), 6)
        self.assertEqual(pending.take(8), {"docs/weekly_paper/legacy/2026-11-13.md"})

    def test_changes_during_refresh_are_not_lost(self):
        pending = PendingChanges({})
        paper = self.root / "meta/paper.prototxt"
        paper.write_text("before")
        pending.observe(source_snapshot(self.root), 0)
        self.assertTrue(pending.take(2))
        paper.write_text("edited while refresh runs")
        pending.observe(source_snapshot(self.root), 5)
        self.assertEqual(pending.take(7), {"meta/paper.prototxt"})

    def test_generated_outputs_do_not_loop(self):
        before = source_snapshot(self.root)
        (self.root / "notes/conference_timeline.png").write_bytes(b"generated")
        (self.root / "docs/about.md").write_text("generated")
        (self.root / "meta/editor.tmp").write_text("incomplete")
        self.assertEqual(before, source_snapshot(self.root))
        (self.root / "notes/cover.png").write_bytes(b"cover")
        self.assertIn("notes/cover.png", source_snapshot(self.root))

    def test_refresh_failure_logged_and_pending_retry(self):
        (self.root / "refresh_and_upload.sh").write_text("printf 'generation failed\\n' >&2\nexit 7\n")
        self.assertFalse(self.supervisor.regenerate())
        self.assertIn("generation failed", (self.root / "logs/refresh.log").read_text())
        pending = PendingChanges({})
        pending.retry({"meta/paper.prototxt"}, 10)
        self.assertFalse(pending.take(39))
        self.assertEqual(pending.take(40), {"meta/paper.prototxt"})

    def test_start_failure_does_not_publish_pid(self):
        actual_launch = self.supervisor.launch

        def fail_launch(name, command, logfile):
            return actual_launch(name, [sys.executable, "-c", "raise SystemExit(9)"], logfile)

        with patch("editor_supervisor.require_free_port"), \
                patch.object(self.supervisor, "launch", side_effect=fail_launch), \
                patch("editor_supervisor.urlopen", side_effect=OSError("not ready")):
            with self.assertRaisesRegex(RuntimeError, "exited with code 9"):
                self.supervisor.start_mkdocs()
        self.assertFalse(self.supervisor.pid_file.exists())
        self.assertNotIn("mkdocs", self.supervisor.children)

    def test_stale_pid_is_never_signalled(self):
        self.supervisor.pid_file.write_text(str(os.getpid()))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(OSError):
                require_free_port(listener.getsockname()[1])
        with patch("editor_supervisor.require_free_port", side_effect=OSError("occupied")):
            with self.assertRaises(OSError):
                self.supervisor.start_mkdocs()
        self.assertEqual(self.supervisor.pid_file.read_text(), str(os.getpid()))

    def test_owned_process_is_reaped_and_logs_append(self):
        for i in range(2):
            process = self.supervisor.launch("mkdocs", [sys.executable, "-c", "import time; time.sleep(60)"], "mkdocs.log")
            self.supervisor.pid_file.write_text(str(process.pid))
            self.supervisor.stop_child("mkdocs")
            self.assertIsNotNone(process.poll())
            self.assertFalse(self.supervisor.pid_file.exists())
        self.assertEqual((self.root / "logs/mkdocs.log").read_text().count("Starting mkdocs"), 2)

    def make_hanging_refresh(self):
        (self.root / "child.py").write_text(
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "Path('child.pid').write_text(str(os.getpid()))\n"
            "time.sleep(60)\n")
        (self.root / "refresh_and_upload.sh").write_text(
            '"$PYTHON_BIN" child.py &\nwait\n')

    def assert_generator_stopped(self):
        child_pid = int((self.root / "child.pid").read_text())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="],
                                    capture_output=True, text=True)
            if result.returncode or result.stdout.strip().startswith("Z"):
                return
            time.sleep(0.05)
        self.fail(f"Generator {child_pid} is still running")

    def test_timeout_kills_generator_group_and_preserves_server(self):
        self.make_hanging_refresh()
        server = self.supervisor.launch("mkdocs", [sys.executable, "-c", "import time; time.sleep(60)"], "mkdocs.log")
        self.assertFalse(self.supervisor.regenerate(timeout=0.8))
        self.assert_generator_stopped()
        self.assertIsNone(server.poll())
        self.assertNotIn("refresh", self.supervisor.children)

    def test_shutdown_during_refresh_kills_generator_group(self):
        self.make_hanging_refresh()
        timer = threading.Timer(0.8, self.supervisor.stopping.set)
        timer.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "shutdown"):
                self.supervisor.regenerate()
        finally:
            timer.cancel()
            timer.join()
        self.assert_generator_stopped()
        self.assertNotIn("refresh", self.supervisor.children)

    def test_api_startup_failure_cleans_up_mkdocs(self):
        processes = []
        actual_launch = self.supervisor.launch

        def launch(name, command, logfile):
            process = actual_launch(name, [sys.executable, "-c", "import time; time.sleep(60)"], logfile)
            processes.append(process)
            return process

        def ready(name, url, timeout=90):
            if name == "api":
                raise RuntimeError("API startup failed")

        with patch("editor_supervisor.require_free_port"), \
                patch.object(self.supervisor, "regenerate", return_value=True), \
                patch.object(self.supervisor, "launch", side_effect=launch), \
                patch.object(self.supervisor, "wait_ready", side_effect=ready):
            with self.assertRaisesRegex(RuntimeError, "API startup failed"):
                self.supervisor.run()
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertEqual(self.supervisor.children, {})
        self.assertFalse(self.supervisor.pid_file.exists())

    def test_refresh_script_stops_at_first_failed_command(self):
        # Exercise the real shell script with fake tools in a temporary repo.
        script = Path(__file__).resolve().parent.parent / "refresh_and_upload.sh"
        (self.root / script.name).write_text(script.read_text())
        tools = self.root / "bin"
        tools.mkdir()
        protoc = tools / "protoc"
        protoc.write_text("#!/bin/sh\nexit 23\n")
        protoc.chmod(0o755)
        fake_python = tools / "python"
        fake_python.write_text("#!/bin/sh\ntouch should-not-run\n")
        fake_python.chmod(0o755)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHON_BIN", None)
        env["PATH"] = str(tools) + os.pathsep + env["PATH"]
        result = subprocess.run(["bash", str(self.root / script.name)], env=env, capture_output=True)
        self.assertEqual(result.returncode, 23)
        self.assertFalse((self.root / "should-not-run").exists())


if __name__ == "__main__":
    unittest.main()
