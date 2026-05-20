import argparse
import json
import re
from pathlib import Path


COMPUTE_ID_RE = re.compile(r"^compute_(\d+)$")
SCHEMA_VERSION = "miqp_frontend_v0.1"


def parse_compute_id(value):
    if value is None:
        return None
    match = COMPUTE_ID_RE.match(value)
    if match is None:
        raise ValueError(f"Unexpected compute node id: {value}")
    return int(match.group(1))


def edge_category(edge):
    if edge.get("semantic_type") == "kv_cache":
        return "kv_cache"
    return edge.get("category", "intermediate_activation")


def build_miqp_input(workload_name, graph, source_format="zigzag_fx"):
    compute_nodes = [
        {
            "id": node["layer_id"],
            "name": node["layer_name"],
            "operator_type": node["operator_type"],
        }
        for node in graph.get("nodes", [])
    ]

    tensor_edges = [
        {
            "tensor": edge["tensor"],
            "producer": parse_compute_id(edge.get("producer")),
            "consumers": [parse_compute_id(consumer) for consumer in edge.get("consumers", [])],
            "size_elements": edge.get("size_elements", 0),
            "lifetime": edge.get("lifetime", 0),
            "reuse": edge.get("reuse", 0),
            "fanout": edge.get("fanout", 0),
            "category": edge_category(edge),
        }
        for edge in graph.get("edges", [])
    ]

    memory_accesses = []
    for node in graph.get("nodes", []):
        for access in node.get("memory_accesses", []):
            memory_accesses.append(
                {
                    "layer_id": node["layer_id"],
                    "operand": access["operand"],
                    "tensor": access["tensor"],
                    "tensor_category": access["tensor_category"],
                    "memory_level": access["memory_level"],
                    "reads": access["reads"],
                    "writes": access["writes"],
                    "assigned_memory": access["suggested_memory"],
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "workload_name": workload_name,
        "source_format": source_format,
        "compute_nodes": compute_nodes,
        "tensor_edges": tensor_edges,
        "memory_accesses": memory_accesses,
    }


def default_out_path(graph_path):
    path = Path(graph_path)
    stem = path.stem
    if stem.endswith("_expanded_memory_graph_assigned"):
        stem = stem.replace("_expanded_memory_graph_assigned", "_miqp_frontend_input")
    else:
        stem = stem + "_miqp_frontend_input"
    return path.with_name(f"{stem}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expanded_memory_graph_assigned_json")
    parser.add_argument("output_json", nargs="?")
    parser.add_argument("--workload-name", default=None)
    parser.add_argument("--source-format", default="zigzag_fx")
    args = parser.parse_args()

    graph_path = Path(args.expanded_memory_graph_assigned_json)
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    out_path = Path(args.output_json) if args.output_json else default_out_path(graph_path)
    workload_name = args.workload_name or out_path.name.replace("_miqp_frontend_input.json", "")
    miqp_input = build_miqp_input(workload_name, graph, args.source_format)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(miqp_input, f, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(miqp_input, indent=2))


if __name__ == "__main__":
    main()
