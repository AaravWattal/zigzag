import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


MEMORY_TECHS = ("LtRAM", "StRAM")
TENSOR_CATEGORIES = ("weights", "intermediate_activation", "kv_cache")
SCHEMA_VERSION = "miqp_frontend_v0.1"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def category_from_kind(kind):
    kind = str(kind or "").lower()
    if kind in {"weight", "weights", "parameter"}:
        return "weights"
    if kind in {"kv", "kv_cache", "cache"}:
        return "kv_cache"
    return "intermediate_activation"


def assign_memory_tech(category, reads, writes):
    total = reads + writes
    write_ratio = writes / total if total else 0
    if category == "weights":
        return "LtRAM", "weights are persistent"
    if category == "kv_cache":
        if writes <= reads:
            return "LtRAM", "KV cache reads dominate writes"
        return "StRAM", "KV cache writes dominate reads"
    if category == "intermediate_activation":
        if write_ratio >= 0.10:
            return "StRAM", "intermediate activation with high write ratio"
        return "LtRAM", "intermediate activation with read-dominated traffic"
    return "StRAM", "default assignment"


def normalized_histogram(counter, keys):
    return {key: counter.get(key, 0) for key in keys}


def iter_memory_accesses(expanded_graph):
    for node in expanded_graph.get("nodes", []):
        for access in node.get("memory_accesses", []):
            yield node, access


def access_total_bytes(access):
    return (access.get("reads", 0) or 0) + (access.get("writes", 0) or 0)


def bytes_by_access_field(expanded_graph, field):
    totals = Counter()
    for _, access in iter_memory_accesses(expanded_graph):
        totals[access[field]] += access_total_bytes(access)
    return totals


def read_write_bytes_by_category(expanded_graph):
    reads = Counter()
    writes = Counter()
    for _, access in iter_memory_accesses(expanded_graph):
        category = access["tensor_category"]
        reads[category] += access.get("reads", 0) or 0
        writes[category] += access.get("writes", 0) or 0
    return reads, writes


def tensor_access_totals(accesses):
    totals = defaultdict(lambda: {"reads": 0, "writes": 0})
    for access in accesses:
        tensor_id = access["tensor_id"]
        bytes_per_exec = access.get("bytes_per_exec", 0) or 0
        if access.get("dir") == "write":
            totals[tensor_id]["writes"] += bytes_per_exec
        else:
            totals[tensor_id]["reads"] += bytes_per_exec
    return totals


def build_tensor_features(name, graph):
    nodes = graph.get("nodes", [])
    node_index = {node["id"]: idx for idx, node in enumerate(nodes)}
    outgoing = defaultdict(list)
    edge_sizes = defaultdict(int)
    for edge in graph.get("edges", []):
        outgoing[edge["source"]].append(edge["destination"])
        edge_sizes[(edge["source"], edge["destination"])] += edge.get("size", 0) or 0

    features = []
    for source, destinations in outgoing.items():
        if source not in node_index:
            continue
        consumers = [dst for dst in destinations if dst in node_index]
        if not consumers:
            continue
        producer_idx = node_index[source]
        last_consumer_idx = max(node_index[dst] for dst in consumers)
        size_bytes = sum(edge_sizes[(source, dst)] for dst in consumers)
        features.append(
            {
                "tensor": f"edge_{source}_out",
                "producer_layer_id": source,
                "consumer_layer_ids": consumers,
                "size_bytes": size_bytes,
                "fanout": len(set(consumers)),
                "lifetime": last_consumer_idx - producer_idx,
                "reuse": len(consumers),
                "semantic_type": "activation",
                "category": "intermediate_activation",
            }
        )

    tensor_by_id = {tensor["id"]: tensor for tensor in graph.get("tensors", [])}
    totals = tensor_access_totals(graph.get("accesses", []))
    for tensor_id, tensor in tensor_by_id.items():
        category = category_from_kind(tensor.get("kind"))
        if category == "intermediate_activation":
            continue
        total = totals.get(tensor_id, {"reads": 0, "writes": 0})
        features.append(
            {
                "tensor": tensor_id,
                "producer_layer_id": None,
                "consumer_layer_ids": [
                    access["op_id"]
                    for access in graph.get("accesses", [])
                    if access.get("tensor_id") == tensor_id
                ],
                "size_bytes": tensor.get("bytes_total", 0),
                "fanout": len(
                    {
                        access["op_id"]
                        for access in graph.get("accesses", [])
                        if access.get("tensor_id") == tensor_id
                    }
                ),
                "lifetime": 0,
                "reuse": total["reads"],
                "semantic_type": "weight" if category == "weights" else "kv_cache",
                "category": category,
            }
        )

    return features


def memory_level_accesses(graph):
    tensor_by_id = {tensor["id"]: tensor for tensor in graph.get("tensors", [])}
    layers = []
    accesses_by_node = defaultdict(list)
    for access in graph.get("accesses", []):
        accesses_by_node[access["op_id"]].append(access)

    for node in graph.get("nodes", []):
        operands = defaultdict(list)
        for access in accesses_by_node.get(node["id"], []):
            tensor = tensor_by_id.get(access["tensor_id"], {})
            category = category_from_kind(tensor.get("kind"))
            reads = access.get("bytes_per_exec", 0) if access.get("dir") != "write" else 0
            writes = access.get("bytes_per_exec", 0) if access.get("dir") == "write" else 0
            operands["external"].append(
                {
                    "level": "external_declared",
                    "tensor": access["tensor_id"],
                    "tensor_category": category,
                    "phase": access.get("phase", "any"),
                    "reads_up": 0,
                    "writes_down": writes,
                    "reads_down": reads,
                    "writes_up": 0,
                    "total_reads": reads,
                    "total_writes": writes,
                    "total_movement": reads + writes,
                }
            )
        layers.append(
            {
                "layer_id": node["id"],
                "layer_name": node.get("name", node["id"]),
                "operator_type": node.get("op", "unknown"),
                "operands": dict(operands),
            }
        )
    return {"layers": layers}


def build_edges(graph):
    nodes = graph.get("nodes", [])
    node_index = {node["id"]: idx for idx, node in enumerate(nodes)}
    outgoing = defaultdict(list)
    sizes = defaultdict(int)
    for edge in graph.get("edges", []):
        if edge["source"] in node_index and edge["destination"] in node_index:
            outgoing[edge["source"]].append(edge["destination"])
            sizes[edge["source"]] += edge.get("size", 0) or 0

    edges = []
    for source, destinations in outgoing.items():
        consumers = sorted(set(destinations), key=lambda dst: node_index[dst])
        producer_idx = node_index[source]
        last_consumer_idx = max(node_index[dst] for dst in consumers)
        edges.append(
            {
                "tensor": f"edge_{source}_out",
                "producer": source,
                "consumers": consumers,
                "size_bytes": sizes[source],
                "fanout": len(consumers),
                "lifetime": last_consumer_idx - producer_idx,
                "reuse": len(destinations),
                "category": "intermediate_activation",
            }
        )
    return edges


def build_expanded_graph(graph):
    tensor_by_id = {tensor["id"]: tensor for tensor in graph.get("tensors", [])}
    accesses_by_node = defaultdict(list)
    for access in graph.get("accesses", []):
        accesses_by_node[access["op_id"]].append(access)

    nodes = []
    for node in graph.get("nodes", []):
        memory_accesses = []
        inputs = []
        for access in accesses_by_node.get(node["id"], []):
            tensor = tensor_by_id.get(access["tensor_id"], {})
            category = category_from_kind(tensor.get("kind"))
            reads = access.get("bytes_per_exec", 0) if access.get("dir") != "write" else 0
            writes = access.get("bytes_per_exec", 0) if access.get("dir") == "write" else 0
            memory, reason = assign_memory_tech(category, reads, writes)
            memory_accesses.append(
                {
                    "tensor": access["tensor_id"],
                    "tensor_category": category,
                    "operand": "external",
                    "memory_level": "external_declared",
                    "reads": reads,
                    "writes": writes,
                    "phase": access.get("phase", "any"),
                    "total_movement": reads + writes,
                    "suggested_memory": memory,
                    "assignment_reason": reason,
                }
            )
            if reads and access["tensor_id"] not in inputs:
                inputs.append(access["tensor_id"])

        nodes.append(
            {
                "node_id": node["id"],
                "kind": "compute",
                "layer_id": node["id"],
                "layer_name": node.get("name", node["id"]),
                "operator_type": node.get("op", "unknown"),
                "compute_energy": node.get("compute_energy", node.get("energy")),
                "compute_latency": node.get("compute_latency", node.get("latency")),
                "local_scratch_bytes": node.get("local_scratch_bytes", 0),
                "inputs": inputs,
                "outputs": [f"edge_{node['id']}_out"],
                "memory_accesses": memory_accesses,
            }
        )

    expanded = {"nodes": nodes, "edges": build_edges(graph)}
    expanded["memory_assignment_histogram"] = normalized_histogram(
        Counter(access["suggested_memory"] for node in nodes for access in node["memory_accesses"]),
        MEMORY_TECHS,
    )
    expanded["tensor_category_histogram"] = normalized_histogram(
        Counter(access["tensor_category"] for node in nodes for access in node["memory_accesses"]),
        TENSOR_CATEGORIES,
    )
    expanded["memory_assignment_bytes"] = normalized_histogram(
        bytes_by_access_field(expanded, "suggested_memory"),
        MEMORY_TECHS,
    )
    expanded["tensor_category_bytes"] = normalized_histogram(
        bytes_by_access_field(expanded, "tensor_category"),
        TENSOR_CATEGORIES,
    )
    return expanded


def persistent_tensors(graph, expanded_graph):
    tensor_by_id = {tensor["id"]: tensor for tensor in graph.get("tensors", [])}
    access_memory = {}
    for node in expanded_graph["nodes"]:
        for access in node.get("memory_accesses", []):
            access_memory.setdefault(access["tensor"], access["suggested_memory"])

    persistent = []
    for tensor_id, tensor in tensor_by_id.items():
        category = category_from_kind(tensor.get("kind"))
        if category == "intermediate_activation":
            continue
        persistent.append(
            {
                "tensor": tensor_id,
                "category": category,
                "size_bytes": tensor.get("bytes_total", 0),
                "shardable": tensor.get("shardable", False),
                "num_shards": tensor.get("num_shards", 1),
                "assigned_memory": access_memory.get(tensor_id, "LtRAM" if category in {"weights", "kv_cache"} else "StRAM"),
            }
        )
    return persistent


def build_miqp_input(name, graph, expanded_graph, source_format):
    return {
        "schema_version": SCHEMA_VERSION,
        "workload_name": name,
        "source_format": source_format,
        "compute_nodes": [
            {
                "id": node["id"],
                "name": node.get("name", node["id"]),
                "operator_type": node.get("op", "unknown"),
                "compute_energy": node.get("compute_energy", node.get("energy")),
                "compute_latency": node.get("compute_latency", node.get("latency")),
                "local_scratch_bytes": node.get("local_scratch_bytes", 0),
            }
            for node in graph.get("nodes", [])
        ],
        "tensor_edges": expanded_graph["edges"],
        "persistent_tensors": persistent_tensors(graph, expanded_graph),
        "memory_accesses": [
            {
                "layer_id": node["layer_id"],
                "tensor": access["tensor"],
                "operand": access["operand"],
                "tensor_category": access["tensor_category"],
                "memory_level": access["memory_level"],
                "reads": access["reads"],
                "writes": access["writes"],
                "assigned_memory": access["suggested_memory"],
            }
            for node in expanded_graph["nodes"]
            for access in node.get("memory_accesses", [])
        ],
    }


def validate_decoupled(graph):
    if not graph.get("nodes"):
        raise ValueError("Expected at least one node")
    if not graph.get("edges"):
        raise ValueError("Expected at least one edge")
    if not graph.get("tensors"):
        raise ValueError("Expected at least one tensor")
    if not graph.get("accesses"):
        raise ValueError("Expected at least one access")

    tensor_by_id = {tensor["id"]: tensor for tensor in graph["tensors"]}
    categories = {category_from_kind(tensor.get("kind")) for tensor in graph["tensors"]}
    if "weights" not in categories:
        raise ValueError("Expected at least one weight tensor")
    if "kv_cache" not in categories:
        raise ValueError("Expected at least one KV-cache tensor")

    access_categories = {
        category_from_kind(tensor_by_id.get(access["tensor_id"], {}).get("kind"))
        for access in graph["accesses"]
    }
    if "weights" not in access_categories:
        raise ValueError("Expected at least one access to a weight tensor")
    if "kv_cache" not in access_categories:
        raise ValueError("Expected at least one access to a KV-cache tensor")
    accesses_by_tensor = defaultdict(list)
    for access in graph["accesses"]:
        accesses_by_tensor[access["tensor_id"]].append(access)
    for tensor in graph["tensors"]:
        category = category_from_kind(tensor.get("kind"))
        if category == "kv_cache" and "cache_k" not in tensor["id"] and "cache_v" not in tensor["id"]:
            raise ValueError(f"Unexpected KV-cache tensor name: {tensor['id']}")
        if category == "kv_cache" and tensor["id"] not in accesses_by_tensor:
            raise ValueError(f"KV-cache tensor has no access: {tensor['id']}")
        if category == "weights":
            reads = [access for access in accesses_by_tensor.get(tensor["id"], []) if access.get("dir") == "read"]
            if not reads:
                raise ValueError(f"Weight tensor has no read access: {tensor['id']}")


def coverage_metrics(graph, miqp_input):
    persistent = miqp_input["persistent_tensors"]
    accesses_by_tensor = defaultdict(list)
    for access in miqp_input["memory_accesses"]:
        accesses_by_tensor[access["tensor"]].append(access)

    persistent_with_accesses = [tensor for tensor in persistent if tensor["tensor"] in accesses_by_tensor]
    persistent_without_accesses = [tensor for tensor in persistent if tensor["tensor"] not in accesses_by_tensor]

    total_read_bytes_by_category = Counter()
    total_write_bytes_by_category = Counter()
    for access in miqp_input["memory_accesses"]:
        category = access["tensor_category"]
        total_read_bytes_by_category[category] += access.get("reads", 0) or 0
        total_write_bytes_by_category[category] += access.get("writes", 0) or 0

    return {
        "persistent_tensors_total": len(persistent),
        "persistent_tensors_with_accesses": len(persistent_with_accesses),
        "persistent_tensors_without_accesses": len(persistent_without_accesses),
        "persistent_tensors_without_access_ids": [tensor["tensor"] for tensor in persistent_without_accesses],
        "total_read_bytes_by_category": normalized_histogram(total_read_bytes_by_category, TENSOR_CATEGORIES),
        "total_write_bytes_by_category": normalized_histogram(total_write_bytes_by_category, TENSOR_CATEGORIES),
    }


def top_by_tensor(miqp_input, direction, limit=20):
    totals = Counter()
    for access in miqp_input["memory_accesses"]:
        totals[access["tensor"]] += access.get(direction, 0) or 0
    return [
        {"tensor": tensor, f"{direction}_bytes": amount}
        for tensor, amount in totals.most_common(limit)
        if amount
    ]


def top_ops_by_access(miqp_input, limit=20):
    totals = Counter()
    for access in miqp_input["memory_accesses"]:
        totals[access["layer_id"]] += (access.get("reads", 0) or 0) + (access.get("writes", 0) or 0)
    return [
        {"op_id": op_id, "total_access_bytes": amount}
        for op_id, amount in totals.most_common(limit)
        if amount
    ]


def handoff_report(name, graph, expanded_graph, miqp_input):
    total_reads = sum(access.get("reads", 0) or 0 for access in miqp_input["memory_accesses"])
    total_writes = sum(access.get("writes", 0) or 0 for access in miqp_input["memory_accesses"])
    report = {
        "workload_name": name,
        "schema_version": SCHEMA_VERSION,
        "source_format": miqp_input["source_format"],
        "num_compute_nodes": len(miqp_input["compute_nodes"]),
        "op_histogram": dict(sorted(Counter(node.get("op", "unknown") for node in graph.get("nodes", [])).items())),
        "num_tensor_edges": len(miqp_input["tensor_edges"]),
        "num_persistent_tensors": len(miqp_input["persistent_tensors"]),
        "tensor_category_histogram": expanded_graph["tensor_category_histogram"],
        "tensor_category_bytes": expanded_graph["tensor_category_bytes"],
        "memory_assignment_histogram": expanded_graph["memory_assignment_histogram"],
        "memory_assignment_bytes": expanded_graph["memory_assignment_bytes"],
        "total_read_bytes": total_reads,
        "total_write_bytes": total_writes,
        "top_20_tensors_by_read_bytes": top_by_tensor(miqp_input, "reads"),
        "top_20_tensors_by_write_bytes": top_by_tensor(miqp_input, "writes"),
        "top_20_ops_by_total_access_bytes": top_ops_by_access(miqp_input),
    }
    report.update(coverage_metrics(graph, miqp_input))
    return report


def summary(name, graph, expanded_graph, miqp_input):
    persistent_histogram = Counter(tensor["category"] for tensor in miqp_input["persistent_tensors"])
    result = {
        "name": name,
        "num_compute_nodes": len(graph.get("nodes", [])),
        "num_edges": len(graph.get("edges", [])),
        "num_persistent_tensors": len(miqp_input["persistent_tensors"]),
        "num_accesses": len(graph.get("accesses", [])),
        "op_histogram": dict(sorted(Counter(node.get("op", "unknown") for node in graph.get("nodes", [])).items())),
        "persistent_tensor_category_histogram": normalized_histogram(persistent_histogram, TENSOR_CATEGORIES),
        "tensor_category_histogram": expanded_graph["tensor_category_histogram"],
        "tensor_category_bytes": expanded_graph["tensor_category_bytes"],
        "memory_assignment_histogram": expanded_graph["memory_assignment_histogram"],
        "memory_assignment_bytes": expanded_graph["memory_assignment_bytes"],
    }
    result.update(coverage_metrics(graph, miqp_input))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default="workloads")
    parser.add_argument("--source-format", default="external_llama_decoupled")
    args = parser.parse_args()

    graph = load_json(Path(args.input))
    validate_decoupled(graph)

    out_dir = Path(args.output_dir)
    prefix = args.name
    tensor_features = build_tensor_features(prefix, graph)
    mem_levels = memory_level_accesses(graph)
    expanded = build_expanded_graph(graph)
    miqp = build_miqp_input(prefix, graph, expanded, args.source_format)
    summary_data = summary(prefix, graph, expanded, miqp)
    report = handoff_report(prefix, graph, expanded, miqp)

    write_json(out_dir / f"{prefix}_tensor_features.json", tensor_features)
    write_json(out_dir / f"{prefix}_memory_level_accesses.json", mem_levels)
    write_json(out_dir / f"{prefix}_expanded_memory_graph.json", expanded)
    write_json(out_dir / f"{prefix}_expanded_memory_graph_assigned.json", expanded)
    write_json(out_dir / f"{prefix}_miqp_frontend_input.json", miqp)
    write_json(out_dir / f"{prefix}_external_import_summary.json", summary_data)
    write_json(out_dir / f"{prefix}_handoff_report.json", report)

    print(json.dumps(summary_data, indent=2))


if __name__ == "__main__":
    main()
