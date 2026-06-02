from scripts.generate_paper_list import readMeta
import json
import networkx as nx
import os
import re


NODE_WIDTH = 168
NODE_HEIGHT = 54
LAYER_GAP = 108
ROW_GAP = 26
PADDING_X = 48
PADDING_Y = 40
MIN_GRAPH_HEIGHT = 260


def main():
    pinfos = readMeta()
    G = nx.DiGraph()
    for pinfo, f in pinfos:
        if len(pinfo.baseline.methods) >= 1:
            # print(pinfo, f)
            cur_name = f.replace(".prototxt", "")
            cur_year = pinfo.pub.year
            cur_node = f"{cur_year}/{cur_name}"
            G.add_node(cur_node, name=f"{cur_name}[{cur_year}]")
            for bl_method in pinfo.baseline.methods:
                if bl_method == "None":
                    continue
                # if not check_exist(bl_method):
                #     print(f"{f} Baseline Method: {bl_method} does not exist.")
                #     G.add_node(bl_method, name=f"{bl_method}")
                #     G.add_edge(bl_method, cur_node)
                if "/" not in bl_method:
                    print(f"{f} Baseline Method: {bl_method} missed year.")
                    G.add_node(bl_method, name=f"{bl_method}")
                    G.add_edge(bl_method, cur_node)
                else:
                    match = re.match(r"^(\d{4})/([a-z0-9_-]+)$", bl_method, flags=re.IGNORECASE)
                    year, name = match.groups()
                    G.add_node(bl_method, name=f"{name}[{year}]")
                    G.add_edge(bl_method, cur_node)

    # Transitive reduction: remove edges that are implied by longer paths
    # e.g., if A→B→C exists, remove A→C
    G_reduced = nx.transitive_reduction(G)
    # transitive_reduction loses node attributes, copy them back
    for node in G_reduced.nodes():
        G_reduced.nodes[node].update(G.nodes[node])

    components = list(nx.weakly_connected_components(G_reduced))
    # Sort components by size (descending) then by node names for consistency
    components.sort(key=lambda c: (-len(c), sorted(c)[0] if c else ''))
    subgraphs = [G_reduced.subgraph(c).copy() for c in components]

    interactive_components = []
    for subgraph in subgraphs:
        if subgraph.number_of_edges() >= 1:
            component_name, anchor = describe_component(subgraph, G_reduced)

            interactive_components.append(
                build_interactive_component(
                    subgraph=subgraph,
                    component_name=component_name,
                    anchor=anchor,
                )
            )

    # Build paper-to-family mapping for Index → Graph navigation
    # Key: paper name (matches paper.id in papers.json), Value: family anchor
    paper_family_map = {}
    for subgraph in subgraphs:
        if subgraph.number_of_edges() >= 1:
            _, anchor = describe_component(subgraph, G_reduced)

            for node in subgraph.nodes():
                # node is "year/name" or just "name" (for no-year nodes)
                name = node.split("/")[-1] if "/" in node else node
                paper_family_map[name] = anchor

    project_root = os.path.dirname(os.path.dirname(__file__))
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    # Write paper-family mapping JSON
    mapping_path = os.path.join(docs_dir, "js", "paper_graph_map.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(paper_family_map, f, ensure_ascii=False, indent=2)
    print(f"Written paper-family mapping to {mapping_path} ({len(paper_family_map)} entries)")

    interactive_data_path = os.path.join(docs_dir, "js", "baseline_methods_graph_data.json")
    interactive_payload = {
        "title": "Baseline Methods Graph Interactive",
        "component_count": len(interactive_components),
        "components": interactive_components,
    }
    with open(interactive_data_path, "w", encoding="utf-8") as f:
        json.dump(interactive_payload, f, ensure_ascii=False, indent=2)
    print(f"Written interactive graph data to {interactive_data_path} ({len(interactive_components)} components)")


def describe_component(subgraph: nx.DiGraph, full_graph: nx.DiGraph):
    # Use the node with highest out-degree (most cited as baseline by others)
    representative_node = max(subgraph.nodes(), key=lambda n: full_graph.out_degree(n))
    node_data = full_graph.nodes[representative_node]
    component_name = node_data.get("name", str(representative_node))
    component_name = re.sub(r"\[\d{4}\]", "", component_name).strip()
    return component_name, component_anchor(component_name)


def component_anchor(component_name: str) -> str:
    return f"{component_name.lower().replace(' ', '-')}-family"


def sort_node_key(graph: nx.DiGraph, node: str):
    year = 0
    if "/" in node:
        prefix = node.split("/", 1)[0]
        if prefix.isdigit():
            year = int(prefix)
    label = str(graph.nodes[node].get("name", node))
    return (year, label.lower(), str(node).lower())


def build_interactive_component(subgraph: nx.DiGraph, component_name: str, anchor: str):
    layers = build_layers(subgraph)
    column_heights = [
        len(layer) * NODE_HEIGHT + max(len(layer) - 1, 0) * ROW_GAP
        for layer in layers
    ] or [MIN_GRAPH_HEIGHT - 2 * PADDING_Y]
    inner_height = max(max(column_heights), MIN_GRAPH_HEIGHT - 2 * PADDING_Y)
    width = PADDING_X * 2 + len(layers) * NODE_WIDTH + max(len(layers) - 1, 0) * LAYER_GAP
    height = PADDING_Y * 2 + inner_height

    nodes = []
    for level, layer in enumerate(layers):
        column_height = len(layer) * NODE_HEIGHT + max(len(layer) - 1, 0) * ROW_GAP
        start_y = PADDING_Y + (inner_height - column_height) / 2 + NODE_HEIGHT / 2
        center_x = PADDING_X + NODE_WIDTH / 2 + level * (NODE_WIDTH + LAYER_GAP)
        for row, node in enumerate(layer):
            label = str(subgraph.nodes[node].get("name", node))
            display_name, year_label = split_node_label(label)
            search_name = re.sub(r"\[\d{4}\]", "", label).strip()
            center_y = start_y + row * (NODE_HEIGHT + ROW_GAP)
            node_type = classify_node(subgraph, node)
            nodes.append({
                "id": node,
                "label": label,
                "display_name": display_name,
                "year_label": year_label,
                "search_name": search_name,
                "type": node_type,
                "level": level,
                "row": row,
                "x": round(center_x, 2),
                "y": round(center_y, 2),
            })

    edges = [
        {"source": u, "target": v}
        for u, v in sorted(subgraph.edges(), key=lambda e: (str(e[0]), str(e[1])))
    ]
    return {
        "title": component_name,
        "anchor": anchor,
        "node_count": subgraph.number_of_nodes(),
        "edge_count": subgraph.number_of_edges(),
        "width": round(width, 2),
        "height": round(height, 2),
        "nodes": nodes,
        "edges": edges,
    }


def build_layers(graph: nx.DiGraph):
    try:
        return [
            sorted(generation, key=lambda node: sort_node_key(graph, node))
            for generation in nx.topological_generations(graph)
        ]
    except nx.NetworkXUnfeasible:
        # Baseline relationships should be acyclic, but keep a deterministic fallback.
        return [sorted(graph.nodes(), key=lambda node: sort_node_key(graph, node))]


def classify_node(graph: nx.DiGraph, node: str) -> str:
    if graph.in_degree(node) == 0:
        return "root"
    if graph.out_degree(node) == 0:
        return "leaf"
    return "default"


def split_node_label(label: str):
    match = re.match(r"^(.*)\[(\d{4})\]$", label)
    if match:
        return match.group(1).strip(), match.group(2)
    return label, ""

def check_exist(bl_method):
    if not isinstance(bl_method, str):
        return False

    match = re.match(r"^(\d{4})/([a-z0-9_-]+)$", bl_method, flags=re.IGNORECASE)
    if not match:
        return False

    year, name = match.groups()

    project_root = os.path.dirname(os.path.dirname(__file__))
    target_file = os.path.join(project_root, "meta", year, f"{name}.prototxt")
    return os.path.isfile(target_file)


if __name__ == "__main__":
    main()
