---
name: efficientpaper-paper-research
description: Search the EfficientPaper local corpus of efficient-AI papers and explain paper nodes, baseline-method relationships, families, and paths. Use for pruning, sparsity, quantization, KV cache, speculative decoding, efficient inference/training, and related LLM system optimization topics; do not use as a general web literature search or for unrelated research areas.
---

# EfficientPaper Research

Use this skill for questions about the local EfficientPaper collection. It is a focused collection, not an index of all research papers. Its primary scope is efficient AI and LLM optimization: pruning and sparsity, quantization, KV cache, speculative decoding, efficient inference and training, deployment, communication-computation overlap, performance modeling, kernel generation, network structure design, layer fusion, low-rank methods, benchmarks, and closely related system optimization topics. Treat the repository as the source of truth and distinguish indexed metadata from the author's notes.

If a request is clearly outside these themes, say that EfficientPaper does not claim coverage and do not present an empty local result as evidence that no such papers exist. For borderline requests, explain which indexed keyword or theme motivated the match. This skill can identify a paper by title or ID even when its metadata has no graph relationship, but it should not generalize the collection into a complete literature review.

## Query workflow

1. Prepare the fixed local data directory with `python skills/efficientpaper-paper-research/scripts/sync_repo.py --target ~/.codex/data/EfficientPaper`, then use the printed path as `--root`. It clones on first use and fast-forwards with `git pull --ff-only` on later uses. When the repository is already checked out locally, `--root .` can be used without syncing. This syncs the paper data, not the installed skill code.
2. Run `python skills/efficientpaper-paper-research/scripts/query.py --root <repo-root> search "<query>"` for keyword, title, author, venue, institution, or ID searches. Search supports quoted phrases, `AND`, `OR`, and negative terms prefixed with `-`.
3. Resolve a paper or graph node with `node "<name-or-year/name>"`. Prefer the canonical `year/id` node ID when available.
4. Use `related "<node>"` for direct baseline neighbors. Use `--direction upstream` for papers used as baselines, `--direction downstream` for papers that use the node, or `both` for both sides.
5. Use `path "<from-node>" "<to-node>"` to explain the shortest known baseline chain. An absent path means only that no path exists in the generated graph.

Always inspect the returned metadata before summarizing a paper. Include the paper's title, year, venue, URL, keywords, note path, and graph node ID when relevant. For relationship answers, state the edge direction and list the evidence node IDs. The graph is a curated baseline-method graph, not a complete citation graph; its generated data is transitively reduced and only retains baseline links whose endpoints share an explicit keyword.

## Output behavior

Answer in the user's language. Keep search results compact and rank exact ID/title matches ahead of broad keyword matches. For relationship requests, report:

- the resolved node(s);
- direct upstream/downstream relationships;
- a path or family component when useful;
- uncertainty when the graph has no relation or metadata is missing.

For an out-of-scope request, briefly state the supported themes and recommend a general academic search tool instead of inventing coverage.

Read [references/data-contract.md](references/data-contract.md) when interpreting IDs, graph semantics, or stale generated data. Regenerate the search and graph JSON with the repository's existing scripts only when the user asks for a refresh or the generated files are demonstrably stale.

## Updates

The data repository is intentionally synchronized explicitly before querying so a read-only question does not unexpectedly access the network. To update the installed skill code and instructions, rerun the repository installer with `--upgrade`:

```bash
curl -fsSL https://raw.githubusercontent.com/hustzxd/EfficientPaper/main/install_skill.sh \
  | bash -s -- --upgrade
```
