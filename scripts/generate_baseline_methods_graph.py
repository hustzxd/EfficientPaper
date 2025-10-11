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
                if not check_exist(bl_method):
                    print(f"{f} Baseline Method: {bl_method} does not exist.")
                    G.add_node(bl_method, name=f"{bl_method}")
                    G.add_edge(bl_method, cur_node)
                else:
                    match = re.match(r"^(\d{4})/([a-z0-9_-]+)$", bl_method, flags=re.IGNORECASE)
                    year, name = match.groups()
                    G.add_node(bl_method, name=f"{name}[{year}]")
                    G.add_edge(bl_method, cur_node)

    components = list(nx.weakly_connected_components(G))
    subgraphs = [G.subgraph(c).copy() for c in components]

    # collect markdown with multiple mermaid blocks
    md_lines = ["# Baseline Methods Graph", "", "This page visualizes baseline-method relationships extracted from meta files.", ""]

    component_index = 0
    for subgraph in subgraphs:
        if subgraph.number_of_edges() >= 1:
            component_index += 1
            mermaid_text = export_mermaid(subgraph)
            md_lines.append(f"## Component {component_index}")
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
    """Export a NetworkX DiGraph to Mermaid flowchart text.

    - Direction: top-down (LR)
    - Node id: sanitized from original node key by replacing non [A-Za-z0-9_] with '_'
    - Node label: prefer node attribute 'name', fallback to node key
    - Edge direction: u --> v
    """
    lines = ["flowchart LR"]

    def sanitize_id(raw: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", raw)

    node_to_id = {}
    for node, data in graph.nodes(data=True):
        nid = sanitize_id(str(node))
        node_to_id[node] = nid
        label = str(data.get("name", node))
        # Mermaid node with label
        lines.append(f"    {nid}[\"{label}\"]")

    for u, v in graph.edges():
        uid = node_to_id[u]
        vid = node_to_id[v]
        lines.append(f"    {uid} --> {vid}")

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