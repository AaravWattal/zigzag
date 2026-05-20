import argparse
import json
from collections import Counter
from pathlib import Path


def assign_memory_tech(access):
    category = access["tensor_category"]
    reads = access["reads"]
    writes = access["writes"]
    total = reads + writes
    write_ratio = writes / total if total else 0

    if category == "weights":
        return "LtRAM", "weights are persistent and read-dominated"

    if category == "kv_cache":
        if write_ratio < 0.25:
            return "LtRAM", "KV cache is long-lived and reads dominate"
        return "StRAM", "KV cache has substantial append/update writes"

    if category == "intermediate_activation":
        if write_ratio >= 0.10:
            return "StRAM", "intermediate activation with high write ratio"
        return "LtRAM", "intermediate activation with read-dominated traffic"

    return "StRAM", "default assignment"


def assign_graph(graph):
    assigned = dict(graph)
    nodes = []
    for node in graph["nodes"]:
        node = dict(node)
        accesses = []
        for access in node.get("memory_accesses", []):
            access = dict(access)
            memory, reason = assign_memory_tech(access)
            access["suggested_memory"] = memory
            access["assignment_reason"] = reason
            accesses.append(access)
        node["memory_accesses"] = accesses
        nodes.append(node)
    assigned["nodes"] = nodes
    assigned["memory_assignment_histogram"] = dict(sorted(memory_assignment_histogram(assigned).items()))
    assigned["tensor_category_histogram"] = dict(sorted(tensor_category_histogram(assigned).items()))
    return assigned


def iter_accesses(graph):
    for node in graph.get("nodes", []):
        yield from node.get("memory_accesses", [])


def memory_assignment_histogram(graph):
    return Counter(access["suggested_memory"] for access in iter_accesses(graph))


def tensor_category_histogram(graph):
    return Counter(access["tensor_category"] for access in iter_accesses(graph))


def default_out_path(graph_path):
    path = Path(graph_path)
    stem = path.stem
    if stem.endswith("_expanded_memory_graph"):
        stem = stem + "_assigned"
    else:
        stem = stem + "_memory_assigned"
    return path.with_name(f"{stem}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expanded_memory_graph_json")
    parser.add_argument("output_json", nargs="?")
    args = parser.parse_args()

    with open(args.expanded_memory_graph_json, "r", encoding="utf-8") as f:
        graph = json.load(f)

    assigned = assign_graph(graph)
    out_path = Path(args.output_json) if args.output_json else default_out_path(args.expanded_memory_graph_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(assigned, f, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(assigned, indent=2))


if __name__ == "__main__":
    main()
