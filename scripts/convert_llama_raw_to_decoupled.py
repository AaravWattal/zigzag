import argparse
import json
from collections import Counter
from pathlib import Path


COMPUTE_OPS = {"MatMul", "Add", "LayerNormalization", "Mul"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add_tensor(tensors_by_id, tensor_id, kind, bytes_total, shardable=False, num_shards=None):
    if not tensor_id:
        return
    existing = tensors_by_id.get(tensor_id)
    if existing is None:
        tensor = {
            "id": tensor_id,
            "kind": kind,
            "bytes_total": bytes_total,
            "shardable": shardable,
        }
        if num_shards is not None:
            tensor["num_shards"] = num_shards
        tensors_by_id[tensor_id] = tensor
        return
    existing["bytes_total"] = max(existing.get("bytes_total", 0), bytes_total)
    if existing.get("kind") != "kv":
        existing["kind"] = kind


def convert(raw):
    raw_nodes = raw.get("nodes", [])
    raw_by_id = {node["id"]: node for node in raw_nodes}
    compute_ids = {node["id"] for node in raw_nodes if node.get("op") in COMPUTE_OPS}

    nodes = []
    tensors_by_id = {}
    accesses = []

    for node in raw_nodes:
        op = node.get("op")
        if op in COMPUTE_OPS:
            nodes.append(
                {
                    "id": node["id"],
                    "name": node.get("name", node["id"]),
                    "op": op,
                    "compute_energy": node.get("energy", 0),
                    "compute_latency": node.get("latency", 0),
                    "local_scratch_bytes": node.get("size", 0),
                }
            )
            if node.get("param") and op in {"LayerNormalization", "Mul"}:
                add_tensor(tensors_by_id, node["param"], "weight", node.get("size", 0), shardable=False)
                accesses.append(
                    {
                        "op_id": node["id"],
                        "tensor_id": node["param"],
                        "dir": "read",
                        "bytes_per_exec": node.get("size", 0),
                        "phase": "any",
                    }
                )
        elif op == "Transpose" and node.get("param"):
            add_tensor(tensors_by_id, node["param"], "weight", node.get("size", 0), shardable=True, num_shards=2)
        elif op == "Slice" and node.get("cache", 0) > 0 and node.get("param"):
            add_tensor(tensors_by_id, node["param"], "kv", node.get("cache", 0), shardable=False)

    edges = []
    for edge in raw.get("edges", []):
        source = raw_by_id[edge["source"]]
        destination = raw_by_id[edge["destination"]]
        source_op = source.get("op")
        destination_op = destination.get("op")

        if edge["source"] in compute_ids and edge["destination"] in compute_ids:
            edges.append(
                {
                    "source": edge["source"],
                    "destination": edge["destination"],
                    "size": edge.get("size", 0),
                    "energy": edge.get("energy", 0),
                    "latency": edge.get("latency", 0),
                }
            )
            continue

        if destination["id"] not in compute_ids:
            continue

        if source_op == "Transpose" and source.get("param"):
            accesses.append(
                {
                    "op_id": destination["id"],
                    "tensor_id": source["param"],
                    "dir": "read",
                    "bytes_per_exec": source.get("size", edge.get("size", 0)),
                    "phase": "any",
                }
            )
            continue

        if source_op == "Slice" and source.get("cache", 0) > 0 and source.get("param"):
            accesses.append(
                {
                    "op_id": destination["id"],
                    "tensor_id": source["param"],
                    "dir": "read",
                    "bytes_per_exec": source.get("cache", source.get("size", edge.get("size", 0))),
                    "phase": "any",
                }
            )
            continue

    converted = {
        "directed": raw.get("directed", True),
        "multigraph": raw.get("multigraph", False),
        "graph": raw.get("graph", {}),
        "nodes": nodes,
        "edges": edges,
        "tensors": list(tensors_by_id.values()),
        "accesses": accesses,
    }
    validate(raw, converted)
    return converted


def validate(raw, converted):
    raw_counts = Counter(node.get("op") for node in raw.get("nodes", []))
    converted_counts = Counter(node.get("op") for node in converted.get("nodes", []))
    if converted_counts.get("Transpose", 0) or converted_counts.get("Slice", 0):
        raise ValueError("Converted compute nodes should not include Transpose or Slice")
    if converted_counts.get("MatMul", 0) != raw_counts.get("MatMul", 0):
        raise ValueError("Converted MatMul count should match raw MatMul count")
    if not any(tensor.get("kind") == "weight" for tensor in converted["tensors"]):
        raise ValueError("Expected weight tensors from raw Transpose/param nodes")
    if not any(tensor.get("kind") == "kv" for tensor in converted["tensors"]):
        raise ValueError("Expected KV tensors from raw Slice/cache nodes")
    if not converted["accesses"]:
        raise ValueError("Expected access records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = load_json(Path(args.input))
    converted = convert(raw)
    write_json(Path(args.output), converted)

    summary = {
        "output": args.output,
        "num_compute_nodes": len(converted["nodes"]),
        "num_edges": len(converted["edges"]),
        "num_tensors": len(converted["tensors"]),
        "num_accesses": len(converted["accesses"]),
        "op_histogram": dict(sorted(Counter(node["op"] for node in converted["nodes"]).items())),
        "tensor_kind_histogram": dict(sorted(Counter(tensor["kind"] for tensor in converted["tensors"]).items())),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
