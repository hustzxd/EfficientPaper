#!/usr/bin/env python3
"""Generate README.md and docs/about.md from readme_raw.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_PLACEHOLDER = "<root>"
PAPER_NUMBER_PLACEHOLDER = "<paper_number>"
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "readme_raw.md"
README_PATH = REPO_ROOT / "README.md"
ABOUT_PATH = REPO_ROOT / "docs" / "about.md"
PAPERS_JSON_PATH = REPO_ROOT / "docs" / "js" / "papers.json"
GENERATED_NOTICE = (
    "<!-- This file is generated from readme_raw.md. "
    "Edit readme_raw.md and run `python scripts/generate_readme_pages.py`. -->\n\n"
)


def render_readme(raw_text: str) -> str:
    """Render GitHub README paths."""
    return raw_text.replace(ROOT_PLACEHOLDER, "")


def render_about(raw_text: str) -> str:
    """Render MkDocs page paths.

    Files under docs/ are served from the MkDocs site root, while repository
    root paths such as notes/ are also copied to the deployed site root.
    """
    return raw_text.replace(f"{ROOT_PLACEHOLDER}docs/", "/").replace(
        ROOT_PLACEHOLDER, "/"
    )


def with_notice(text: str) -> str:
    return GENERATED_NOTICE + text.lstrip()


def read_paper_count() -> int:
    try:
        data = json.loads(PAPERS_JSON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing paper dataset: {PAPERS_JSON_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {PAPERS_JSON_PATH}: {exc}") from exc

    papers = data.get("papers") if isinstance(data, dict) else None
    if not isinstance(papers, list):
        raise SystemExit(f"Expected {PAPERS_JSON_PATH} to contain a papers list")
    return len(papers)


def read_source() -> str:
    try:
        return SOURCE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing source file: {SOURCE_PATH}") from exc


def write_if_changed(path: Path, content: str) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def apply_dynamic_values(raw_text: str, paper_count: int) -> str:
    return raw_text.replace(PAPER_NUMBER_PLACEHOLDER, str(paper_count))


def build_outputs(raw_text: str, paper_count: int) -> dict[Path, str]:
    rendered_source = apply_dynamic_values(raw_text, paper_count)
    return {
        README_PATH: with_notice(render_readme(rendered_source)),
        ABOUT_PATH: with_notice(render_about(rendered_source)),
    }


def check_outputs(outputs: dict[Path, str]) -> list[Path]:
    changed: list[Path] = []
    for path, expected in outputs.items():
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != expected:
            changed.append(path)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate README.md and docs/about.md from readme_raw.md."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with a non-zero status if generated files are stale.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_outputs(read_source(), read_paper_count())

    if args.check:
        stale_paths = check_outputs(outputs)
        if stale_paths:
            print("Generated documentation is stale:")
            for path in stale_paths:
                print(f"- {path.relative_to(REPO_ROOT)}")
            return 1
        print("Generated documentation is up to date")
        return 0

    changed_paths = [
        path.relative_to(REPO_ROOT)
        for path, content in outputs.items()
        if write_if_changed(path, content)
    ]

    if changed_paths:
        print("Updated generated documentation:")
        for path in changed_paths:
            print(f"- {path}")
    else:
        print("Generated documentation is already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
