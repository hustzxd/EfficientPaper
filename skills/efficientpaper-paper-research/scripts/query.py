#!/usr/bin/env python3
"""Search EfficientPaper metadata and traverse its generated baseline graph."""

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict, deque


DEFAULT_SITE = "https://hustzxd.github.io/EfficientPaper/"


def load_site(site_url):
    base = site_url.rstrip("/")

    def read_json(name):
        with urllib.request.urlopen(f"{base}/js/{name}", timeout=20) as response:
            return json.load(response)

    papers = read_json("papers.json")["papers"]
    graph = read_json("baseline_methods_graph_data.json")
    families = read_json("paper_graph_map.json")
    by_node = {f"{p['year']}/{p['id']}".lower(): p for p in papers}
    nodes, edges, family_by_node = {}, [], {}
    for component in graph.get("components", []):
        for node in component.get("nodes", []):
            nodes[node["id"].lower()] = node
            family_by_node[node["id"].lower()] = component.get("anchor")
        edges.extend((e["source"], e["target"]) for e in component.get("edges", []))
    for short_id, anchor in families.items():
        for node_id in list(nodes):
            if node_id.rsplit("/", 1)[-1].lower() == short_id.lower():
                family_by_node[node_id] = anchor
    return papers, by_node, nodes, edges, family_by_node


def compact(paper, node=None, family=None):
    result = {k: paper.get(k) for k in (
        "id", "title", "abbr", "year", "venue", "url", "authors", "institutions",
        "keywords", "code_url", "note_url", "prototxt_path", "rating")}
    if node:
        result["node_id"] = node["id"]
        result["node_type"] = node.get("type")
    if family:
        result["family"] = family
    return result


def terms(query):
    groups = []
    for group in re.split(r"\s+OR\s+", query, flags=re.I):
        exact = re.findall(r'"([^"]+)"', group)
        remainder = re.sub(r'"[^"]+"', "", group)
        tokens = re.findall(r"(?<!\S)-?[^\s]+", remainder)
        positive = [t.lower() for t in tokens if not t.startswith("-") and t.upper() != "AND"]
        negative = [t[1:].lower() for t in tokens if t.startswith("-") and len(t) > 1]
        groups.append((exact, positive, negative))
    return groups


def searchable(paper):
    values = [paper.get("id"), paper.get("abbr"), paper.get("title"), paper.get("venue")]
    values += paper.get("authors", []) + paper.get("institutions", []) + paper.get("keywords", [])
    return " ".join(str(x) for x in values if x).lower()


def do_search(query, papers, limit):
    ranked = []
    for paper in papers:
        haystack = searchable(paper)
        score = 0
        for exact, positive, negative in terms(query):
            if not (all(p in haystack for p in positive) and all(n not in haystack for n in negative) and all(e.lower() in haystack for e in exact)):
                continue
            title = str(paper.get("title", "")).lower()
            score = max(score, sum(5 if p in title else 1 for p in positive) + 10 * sum(e.lower() in title for e in exact))
        if score or any(all(e.lower() in haystack for e in exact) and not positive for exact, positive, _ in terms(query)):
            ranked.append((score, paper))
    ranked.sort(key=lambda x: (-x[0], -int(x[1].get("year", 0)), x[1].get("title", "").lower()))
    return [{**paper, "match_score": score} for score, paper in ranked[:limit]]


def resolve(value, by_node, nodes):
    key = value.lower()
    if key in by_node:
        return key
    candidates = []
    for node_id, paper in by_node.items():
        node = nodes.get(node_id, {})
        names = [paper.get("id"), paper.get("abbr"), paper.get("title"), node.get("display_name"), node.get("search_name")]
        if any(key == str(name).lower() for name in names if name):
            candidates.append(node_id)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous node: {value} ({', '.join(candidates)})")
    raise ValueError(f"node not found: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-url", default=DEFAULT_SITE, help=f"Published EfficientPaper site (default: {DEFAULT_SITE})")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    node = sub.add_parser("node")
    node.add_argument("value")
    related = sub.add_parser("related")
    related.add_argument("value")
    related.add_argument("--direction", choices=("upstream", "downstream", "both"), default="both")
    path = sub.add_parser("path")
    path.add_argument("source")
    path.add_argument("target")
    args = parser.parse_args()
    try:
        papers, by_node, nodes, edges, families = load_site(args.site_url)
        if args.command == "search":
            matches = do_search(args.query, papers, args.limit)
            result = {"query": args.query, "count": len(matches), "papers": matches}
        elif args.command == "node":
            node_id = resolve(args.value, by_node, nodes)
            result = compact(by_node[node_id], nodes.get(node_id), families.get(node_id))
        elif args.command == "related":
            node_id = resolve(args.value, by_node, nodes)
            incoming = [s.lower() for s, t in edges if t.lower() == node_id]
            outgoing = [t.lower() for s, t in edges if s.lower() == node_id]
            selected = incoming if args.direction == "upstream" else outgoing if args.direction == "downstream" else incoming + outgoing
            result = {
                "node": compact(by_node[node_id], nodes.get(node_id), families.get(node_id)),
                "direction": args.direction,
                "relationships": [
                    {"relation": "upstream" if n in incoming else "downstream", "paper": compact(by_node[n], nodes.get(n), families.get(n))}
                    for n in selected if n in by_node
                ],
                "upstream": [compact(by_node[n], nodes.get(n), families.get(n)) for n in incoming if n in by_node],
                "downstream": [compact(by_node[n], nodes.get(n), families.get(n)) for n in outgoing if n in by_node],
                "unresolved_node_ids": [n for n in selected if n not in by_node],
            }
        else:
            source = resolve(args.source, by_node, nodes)
            target = resolve(args.target, by_node, nodes)
            graph = defaultdict(list)
            for left, right in edges:
                graph[left.lower()].append(right.lower())
                graph[right.lower()].append(left.lower())
            queue, previous = deque([source]), {source: None}
            while queue:
                current = queue.popleft()
                if current == target:
                    break
                for nxt in graph[current]:
                    if nxt not in previous:
                        previous[nxt] = current
                        queue.append(nxt)
            chain = []
            if target in previous:
                current = target
                while current is not None:
                    chain.append(current)
                    current = previous[current]
                chain.reverse()
            result = {"source": source, "target": target, "found": bool(chain), "path": [compact(by_node[n], nodes.get(n), families.get(n)) for n in chain]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
