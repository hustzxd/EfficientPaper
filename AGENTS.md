# AGENTS.md

This repository is a MkDocs-based paper collection site for efficient AI research. The primary user-facing page is `docs/index.md`, which acts as the main application surface rather than a simple Markdown landing page.

## Primary Surface

- Main page: `docs/index.md`
- Site navigation is defined in `mkdocs.yml`
- The `Home` nav entry points to `index.md`
- Any description of the current product UI should treat `docs/index.md` as the canonical main page

## Current Product Shape

The site currently presents EfficientPaper as a local-first paper workspace rather than a static list page.

- Current frontend dataset count: `docs/js/papers.json` reports `473` papers
- The README now documents the UI through screenshots in `docs/images/`
- If future agents update README or UI descriptions, keep `README.md`, `AGENTS.md`, and the actual UI behavior aligned

## Current UI and Features

### 1. Home search workspace

The top area of `docs/index.md` is a search-and-filter workspace for browsing papers.

- Search input supports plain keywords, quoted phrases, `AND`, negative terms with `-key`, and keyboard shortcuts:
  - `/` focuses the search box
  - `Esc` clears or closes active overlays
- Filter bar includes:
  - year filter
  - venue filter
  - keyword filter
  - rating filter
  - sort selector
  - reset button
  - statistics button
  - add-from-arXiv button
  - upload-to-GitHub button
  - pull-from-GitHub button
  - export-selected button
  - local PDF path button
- Search stats area shows:
  - current result count
  - select-all checkbox for the current filtered set

Representative screenshot:

- `docs/images/efficient_paper.png`
  - shows the Home page search bar, filters, result cards, and top-level actions

### 2. Paper list and card actions

Search results render as paginated paper cards, 10 items per page.

Each card can include:

- selection checkbox
- subtle delete button in the top-right corner
- cover image with lightbox preview
- title and optional abbreviation
- year and venue badges
- keyword badges
- interactive star rating
- update time
- authors
- institutions
- action links and buttons

Card actions currently include:

- `Copy`: copy title and URL
- `Share Link`: copy a deep link to the card
- `Code`: external repository link with async GitHub star count
- `Note`: open the paper note page
- `PDF`: open or locate a local PDF, with download fallback through the local server
- `Edit`: open `docs/edit.html` for metadata editing
- `Graph`: jump to the related component in `docs/baseline_methods_graph_interactive.md`
- `Delete`: a low-visibility card control that removes the paper's `.prototxt` and note folder after explicit confirmation

### 3. Detail drawer

Clicking a non-interactive area of a paper card opens a right-side detail drawer.

The detail drawer shows:

- larger cover image
- authors and institutions
- metadata badges
- rating
- note preview
- main links for the paper

On mobile, the drawer supports swipe-to-close behavior.

### 4. Statistics panel

The `Statistics` button opens an in-page panel with aggregate views over the current dataset.

Current sections include:

- top keywords
- year distribution
- rating distribution
- top venues
- top authors
- top institutions

### 5. Modal workflows on Home

The Home page contains several modal-based workflows:

- Add from arXiv
  - accepts arXiv ID
  - performs live lookup
  - shows detected title, authors, year, institutions, and code URL
  - can create a new paper entry with optional custom abbreviation
- Upload to GitHub
  - accepts a commit message
  - triggers local upload/deploy flow through the local server
- Pull from GitHub
  - opens a confirmation-style modal
  - runs `git pull` in the repository root through the local server
  - is intended to sync the latest remote changes into the local workspace
- Delete Paper
  - opens a confirmation modal from the paper card
  - shows the target `prototxt` and derived `note.md` path
  - requires typing `DELETE` before removal
- Set PDF Path
  - saves a local PDF directory path in the browser
  - supports manual input and folder browse
  - validates the path through the local server
- Export Selected Papers
  - exports selected items as Markdown, plain text, BibTeX, or JSON
  - supports copy-to-clipboard and file download

Representative screenshots:

- `docs/images/add_from_arxiv.png`
  - Add from arXiv modal
- `docs/images/upload_to_github.png`
  - Upload to GitHub modal

### 6. Interactive graph page

The graph page is `docs/baseline_methods_graph_interactive.md`.

- This is the current graph page linked from navigation
- It is not the older Mermaid-based `docs/baseline_methods_graph.md`
- It renders a custom interactive relationship graph
- Nodes can highlight related methods
- Node interactions can jump back to the Home page search/card

Representative screenshot:

- `docs/images/graph.png`
  - shows the interactive graph UI used by the current site

### 7. Local metadata editor

`docs/edit.html` is a standalone metadata editor for `.prototxt` entries and is intended for local use with the editor server.

Current editor capabilities include:

- title, abbreviation, URL, authors, and institutions editing
- venue and year editing
- keyword selection
- code URL editing
- rating editing
- cover filename editing
- cover image upload with preview
- baseline method linking
- save changes
- delete paper with confirmation flow

Representative screenshots:

- `docs/images/edit1.png`
  - top half of metadata editor
- `docs/images/edit2.png`
  - cover upload, preview, baseline methods, save, and delete controls

## Local-Server-Aware Behavior

Several controls depend on the local editor server at `http://localhost:8001`.

When the server is unavailable:

- `Add from arXiv` is disabled
- `Upload to GitHub` is disabled
- `Pull from GitHub` is disabled
- `Set PDF Path` is disabled
- per-card `PDF` actions are disabled
- per-card `Edit` links are disabled
- per-card delete controls are disabled

This graceful degradation is part of the current UX and should be preserved.

## Other User-Facing Pages

- `docs/baseline_methods_graph_interactive.md`
  - relationship graph page
  - renders the current custom interactive graph view
  - nodes link back to the Home page search
- `docs/weekly_paper/`
  - weekly paper digests
  - long-form curated summaries
- `docs/about.md`
  - repository overview / README-style page inside MkDocs
- `docs/contributors.md`
  - contributor listing
- `docs/edit.html`
  - standalone metadata editor for `.prototxt` entries
  - intended for local use with the editor server

## Note Editing

Paper note pages under `notes/<year>/<paper>/note.md` support in-browser editing when the local server is running.

The injected note editor:

- appears only on note pages
- loads note content from the local server
- uses EasyMDE
- saves back to the repository through the local API

## Data, Scripts, and Assets

Important files tied to the current interface:

- `docs/index.md`
  - main page HTML, CSS, and JavaScript
- `docs/baseline_methods_graph_interactive.md`
  - current interactive graph page
- `docs/edit.html`
  - local metadata editor
- `docs/js/papers.json`
  - frontend search dataset
- `docs/js/paper_graph_map.json`
  - mapping from paper IDs to graph anchors
- `docs/js/baseline_methods_graph_data.json`
  - graph data backing the interactive graph page
- `meta/<year>/*.prototxt`
  - structured paper metadata source
- `notes/<year>/<paper>/note.md`
  - note content
- `notes/<year>/<paper>/cover.*`
  - cover assets
- `scripts/paper_editor_server.py`
  - local API server for edit/save/search/upload/pull/delete/PDF/rating actions
- `scripts/generate_search_data.py`
  - search dataset generator
- `add_paper_info.sh`
  - convenience wrapper for adding a paper by arXiv ID
- `scripts/add_paper.py`
  - CLI that creates metadata from an arXiv ID
- `start_editor.sh`
  - starts MkDocs, the editor server, and the auto-refresh watcher
- `refresh_and_upload.sh`
  - regenerates derived assets and optionally commits/pushes/deploys

## Screenshot Assets

The screenshot files in `docs/images/` are now part of the repository documentation surface.

- `efficient_paper.png`
  - Home page overview
- `graph.png`
  - interactive graph page overview
- `edit1.png`
  - metadata editor top section
- `edit2.png`
  - metadata editor lower section
- `add_from_arxiv.png`
  - arXiv import modal
- `upload_to_github.png`
  - GitHub upload modal

If a UI change makes these screenshots stale, update the screenshot assets and the related README or AGENTS descriptions together.

## Change Guidance for Future Agents

- Treat `docs/index.md` as the primary application file for homepage UI changes
- Keep local-server-dependent actions optional and visibly disabled when unavailable
- Preserve existing routes between Home, Graph, Note, and Edit views
- Prefer updating generated data through existing scripts instead of hand-editing large generated JSON files unless the task explicitly requires it
- Keep `README.md` and `AGENTS.md` synchronized when changing user-visible workflows or screenshot-backed descriptions
- Do not reintroduce references to `docs/baseline_methods_graph.md` unless that file is intentionally restored to the product
- Document `./add_paper_info.sh` as taking an arXiv ID, not a local PDF path
