import argparse
import json
from collections import Counter
from pathlib import Path


EXPECTED_SCHEMA = "miqp_frontend_v0.1"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def validate_llama_decoupled(data):
    persistent = data.get("persistent_tensors", [])
    accesses = data.get("memory_accesses", [])
    category_counts = Counter(tensor["category"] for tensor in persistent)
    kv_tensors = {tensor["tensor"] for tensor in persistent if tensor["category"] == "kv_cache"}
    weight_tensors = {tensor["tensor"] for tensor in persistent if tensor["category"] == "weights"}

    assertions = {
        "schema_version": data.get("schema_version") == EXPECTED_SCHEMA,
        "source_format": data.get("source_format") == "external_llama_decoupled",
        "compute_nodes": len(data.get("compute_nodes", [])) == 208,
        "tensor_edges": len(data.get("tensor_edges", [])) == 286,
        "persistent_tensors": len(persistent) == 177,
        "unique_kv_tensors": len(kv_tensors) == 32,
        "weight_tensors": len(weight_tensors) == 145,
        "all_kv_tensors_accessed": kv_tensors <= {access["tensor"] for access in accesses},
        "all_weight_tensors_read": weight_tensors
        <= {access["tensor"] for access in accesses if access.get("reads", 0) > 0},
    }

    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")

    total_reads = sum(access.get("reads", 0) or 0 for access in accesses)
    total_writes = sum(access.get("writes", 0) or 0 for access in accesses)
    access_categories = Counter(access["tensor_category"] for access in accesses)
    uniform_k1 = {
        "memory_class": "SRAM",
        "fabric_class": "default",
        "num_collapsed_access_records": len(accesses),
        "total_read_bytes": total_reads,
        "total_write_bytes": total_writes,
    }

    return {
        "input": data.get("workload_name"),
        "schema_version": data.get("schema_version"),
        "source_format": data.get("source_format"),
        "assertions": assertions,
        "ingested": {
            "compute_nodes": len(data.get("compute_nodes", [])),
            "tensor_edges": len(data.get("tensor_edges", [])),
            "persistent_tensors": len(persistent),
            "memory_access_records": len(accesses),
            "unique_kv_tensors": len(kv_tensors),
            "weight_tensors": len(weight_tensors),
            "persistent_tensor_categories": dict(sorted(category_counts.items())),
            "memory_access_categories": dict(sorted(access_categories.items())),
        },
        "objective_components_available": {
            "compute_energy": any("compute_energy" in node for node in data.get("compute_nodes", [])),
            "compute_latency": any("compute_latency" in node for node in data.get("compute_nodes", [])),
            "memory_read_bytes": total_reads > 0,
            "memory_write_bytes": total_writes > 0,
        },
        "k1_uniform_collapse": uniform_k1,
        "backend_solve": {
            "status": "not_run",
            "reason": "No backend solver or monolithic MIQP formulation entry point is present in this repository.",
            "variables": None,
            "constraints": None,
            "objective_components_in_model": [],
            "parity_against_monolithic": "not_checked",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = load_json(Path(args.input_json))
    report = validate_llama_decoupled(data)
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
