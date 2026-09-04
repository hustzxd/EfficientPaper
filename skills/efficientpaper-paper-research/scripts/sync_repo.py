#!/usr/bin/env python3
"""Clone or fast-forward the EfficientPaper data repository."""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_URL = "https://github.com/hustzxd/EfficientPaper.git"
DEFAULT_TARGET = Path("~/.codex/data/EfficientPaper")


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)


def same_remote(actual, expected):
    def normalize(value):
        value = value.strip().removesuffix("/")
        if value.endswith(".git"):
            value = value[:-4]
        if value.startswith("git@github.com:"):
            return "https://github.com/" + value.split(":", 1)[1]
        return value

    return normalize(actual) == normalize(expected)


def sync(repo_url, target, branch):
    target = target.expanduser().resolve()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--branch", branch, repo_url, str(target))
        return target

    if not target.is_dir() or not (target / ".git").exists():
        raise RuntimeError(f"target exists but is not a git repository: {target}")
    actual = run("git", "remote", "get-url", "origin", cwd=target).stdout
    if not same_remote(actual, repo_url):
        raise RuntimeError(f"origin does not match EfficientPaper: {actual.strip()}")
    dirty = run("git", "status", "--porcelain", cwd=target).stdout.strip()
    if dirty:
        raise RuntimeError(f"target has uncommitted changes; refusing to pull: {target}")
    run("git", "pull", "--ff-only", "origin", branch, cwd=target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_URL)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()
    try:
        target = sync(args.repo_url, args.target, args.branch)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
