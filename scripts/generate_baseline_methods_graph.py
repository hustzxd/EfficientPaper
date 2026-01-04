from scripts.generate_paper_list import readMeta
import networkx as nx
import os
import re


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

    components = list(nx.weakly_connected_components(G))
    # Sort components by size (descending) then by node names for consistency
    components.sort(key=lambda c: (-len(c), sorted(c)[0] if c else ''))
    subgraphs = [G.subgraph(c).copy() for c in components]

    # collect markdown with multiple mermaid blocks
    md_lines = [
        "# Baseline Methods Graph",
        "",
        "This page visualizes baseline-method relationships extracted from meta files.",
        "",
        "Each component represents a family of related methods, showing how newer papers build upon previous baseline methods.",
        ""
    ]

    component_index = 0
    for subgraph in subgraphs:
        if subgraph.number_of_edges() >= 1:
            component_index += 1

            # Find the most representative name for this component
            # Use the node with highest out-degree (most cited as baseline by others)
            representative_node = max(subgraph.nodes(),
                                    key=lambda n: G.out_degree(n))

            # Extract readable name from node
            node_data = G.nodes[representative_node]
            component_name = node_data.get('name', str(representative_node))
            # Remove year suffix like [2020] for cleaner display
            component_name = re.sub(r'\[\d{4}\]', '', component_name).strip()

            mermaid_text = export_mermaid(subgraph)
            node_count = subgraph.number_of_nodes()
            edge_count = subgraph.number_of_edges()

            md_lines.append(f"## {component_name} Family")
            md_lines.append("")
            md_lines.append(f"*{node_count} methods, {edge_count} relationships*")
            md_lines.append("")
            md_lines.append("```mermaid")
            md_lines.append(mermaid_text)
            md_lines.append("```")
            md_lines.append("")

    project_root = os.path.dirname(os.path.dirname(__file__))
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    target_md = os.path.join(docs_dir, "baseline_methods_graph.md")
    with open(target_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Written Mermaid graphs to {target_md}")

def export_mermaid(graph: nx.DiGraph) -> str:
    """Export a NetworkX DiGraph to Mermaid flowchart text with enhanced styling.

    - Direction: left-to-right (LR) for better handling of many nodes
    - Node id: sanitized from original node key by replacing non [A-Za-z0-9_] with '_'
    - Node label: prefer node attribute 'name', fallback to node key
    - Edge direction: u --> v
    - Styling: colored nodes and styled edges for better visualization
    """
    lines = ["flowchart LR"]

    # Add custom styling
    lines.append("    classDef defaultNode fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff")
    lines.append("    classDef rootNode fill:#50C878,stroke:#2E7D4E,stroke-width:3px,color:#fff")
    lines.append("    classDef leafNode fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff")

    # Define link styles with different colors for better distinction
    lines.append("    linkStyle default stroke:#9370DB,stroke-width:2px")
    lines.append("")

    def sanitize_id(raw: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", raw)

    # Identify root nodes (no predecessors) and leaf nodes (no successors)
    root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]

    node_to_id = {}
    # Sort nodes for consistent output
    for node, data in sorted(graph.nodes(data=True), key=lambda x: str(x[0])):
        nid = sanitize_id(str(node))
        node_to_id[node] = nid
        label = str(data.get("name", node))

        # Use rounded rectangles for better appearance
        lines.append(f"    {nid}[\"{label}\"]")

        # Apply class styling
        if node in root_nodes:
            lines.append(f"    class {nid} rootNode")
        elif node in leaf_nodes:
            lines.append(f"    class {nid} leafNode")
        else:
            lines.append(f"    class {nid} defaultNode")

    lines.append("")

    # Define color palette for edges
    edge_colors = [
        "#9370DB",  # Medium Purple
        "#FF6347",  # Tomato
        "#20B2AA",  # Light Sea Green
        "#FFD700",  # Gold
        "#FF69B4",  # Hot Pink
        "#00CED1",  # Dark Turquoise
        "#FFA500",  # Orange
        "#7B68EE",  # Medium Slate Blue
    ]

    edge_index = 0
    # Sort edges for consistent output
    for u, v in sorted(graph.edges(), key=lambda e: (str(e[0]), str(e[1]))):
        uid = node_to_id[u]
        vid = node_to_id[v]
        # Assign different colors to edges for better visibility
        color = edge_colors[edge_index % len(edge_colors)]
        lines.append(f"    {uid} ==>|\" \"| {vid}")
        lines.append(f"    linkStyle {edge_index} stroke:{color},stroke-width:2.5px")
        edge_index += 1

    return "\n".join(lines)

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