# EfficientPaper Data Contract

## Scope

EfficientPaper is a curated collection for efficient AI research rather than a general paper database. Its indexed themes include pruning and sparsity, quantization, KV Cache, speculative decoding, efficient inference/training, LLM deployment, communication-computation overlap, performance modeling, kernel generation, network structure design, layer fusion, low-rank methods, benchmarks, and related system optimization topics. The exact set of papers changes as the repository evolves; use the paper's indexed keywords as the evidence for a theme match.

## Sources

- `docs/js/papers.json`: generated paper metadata used by the Home search UI.
- `docs/js/baseline_methods_graph_data.json`: generated graph components, nodes, and edges.
- `docs/js/paper_graph_map.json`: paper ID to graph-family anchor mapping.
- `meta/<year>/<id>.prototxt`: editable source metadata, including explicit `baseline.methods`.
- `notes/<year>/<id>/note.md`: optional local research note; it may contain AI-generated claims and should not override the paper URL or indexed metadata.

## Identifiers

Paper records use `id` such as `PDD` plus a `year`. Graph nodes use `year/id`, for example `2026/PDD`. A graph node may be present even when its paper metadata is incomplete. Resolve names case-insensitively, but preserve the canonical spelling in output.

## Relationship semantics

An edge `A -> B` means B explicitly lists A as a baseline method in its metadata. It does not mean that A cites B, that B is strictly derived from A, or that the two papers share authors. The graph generator filters invalid/missing baseline references and requires a shared explicit keyword, then applies transitive reduction. The graph therefore describes known baseline lineage, not every possible research connection.

`paper_graph_map.json` maps a node's short paper ID to a family anchor. It is useful for locating the interactive graph page, but it is not an edge list.
