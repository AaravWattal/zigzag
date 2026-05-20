import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


TOY_WORKLOADS = [
    {
        "model": "two_conv",
        "prefix": "two_conv_auto_fx",
        "accelerator": "zigzag/inputs/hardware/tpu_like.yaml",
        "mapping": "zigzag/inputs/mapping/tpu_like.yaml",
    },
    {
        "model": "tiny_residual",
        "prefix": "tiny_residual_auto_fx",
        "accelerator": "zigzag/inputs/hardware/tpu_like.yaml",
        "mapping": "zigzag/inputs/mapping/tpu_like.yaml",
    },
    {
        "model": "one_linear",
        "prefix": "one_linear_fx",
        "accelerator": "zigzag/inputs/hardware/gemm_l1.yaml",
        "mapping": "zigzag/inputs/mapping/gemm_simple.yaml",
    },
    {
        "model": "tiny_cnn_linear",
        "prefix": "tiny_cnn_linear_auto_fx",
        "accelerator": "zigzag/inputs/hardware/tpu_like.yaml",
        "mapping": "zigzag/inputs/mapping/tpu_like_gemm_mixed.yaml",
    },
    {
        "model": "tiny_attention",
        "prefix": "tiny_attention_auto_fx",
        "accelerator": "zigzag/inputs/hardware/gemm_l1.yaml",
        "mapping": "zigzag/inputs/mapping/gemm_simple.yaml",
    },
]

MEMORY_TECHS = ("LtRAM", "StRAM")
TENSOR_CATEGORIES = ("weights", "intermediate_activation", "kv_cache")


def run_command(args):
    print("+", " ".join(args), flush=True)
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    return result.stdout


def parse_last_json_object(text):
    decoder = json.JSONDecoder()
    parsed = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end_idx = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if text[idx + end_idx :].strip():
            continue
        parsed = value
    if parsed is None:
        raise RuntimeError("Command output did not end with a JSON object")
    return parsed


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalized_histogram(counter, keys):
    return {key: counter.get(key, 0) for key in keys}


def first_existing_path(*paths):
    for path in paths:
        if (REPO_ROOT / path).exists():
            return path
    raise FileNotFoundError(f"None of these paths exist: {paths}")


def iter_accesses(graph):
    for node in graph.get("nodes", []):
        yield from node.get("memory_accesses", [])


def summarize_workload(config, run_summary, classified_features, assigned_graph):
    histogram = Counter(feature["class_name"] for feature in classified_features)
    lifetimes = [feature.get("lifetime", 0) or 0 for feature in classified_features]
    fanouts = [feature.get("fanout", 0) or 0 for feature in classified_features]
    reuses = [feature.get("reuse", 0) or 0 for feature in classified_features]
    accesses = list(iter_accesses(assigned_graph))
    memory_assignment_histogram = Counter(access["suggested_memory"] for access in accesses)
    tensor_category_histogram = Counter(access["tensor_category"] for access in accesses)

    yaml_path = f"workloads/{config['prefix']}.yaml"
    with open(REPO_ROOT / yaml_path, "r", encoding="utf-8") as f:
        num_layers = sum(1 for line in f if line.startswith("- id:"))

    return {
        "model": config["model"],
        "prefix": config["prefix"],
        "num_layers": num_layers,
        "num_cmes": run_summary["num_cmes"],
        "energy": run_summary["energy_pj"],
        "latency": run_summary["latency_cycles"],
        "num_tensors": len(classified_features),
        "max_lifetime": max(lifetimes, default=0),
        "max_fanout": max(fanouts, default=0),
        "max_reuse": max(reuses, default=0),
        "class_histogram": dict(sorted(histogram.items())),
        "num_compute_nodes": len(assigned_graph.get("nodes", [])),
        "num_memory_access_records": len(accesses),
        "memory_assignment_histogram": normalized_histogram(memory_assignment_histogram, MEMORY_TECHS),
        "tensor_category_histogram": normalized_histogram(tensor_category_histogram, TENSOR_CATEGORIES),
        "yaml": yaml_path,
        "cmes_pickle": run_summary["pickle_filename"],
        "tensor_features": f"workloads/{config['prefix']}_tensor_features.json",
        "tensor_features_with_reuse": f"workloads/{config['prefix']}_tensor_features_with_reuse.json",
        "tensor_features_classified": f"workloads/{config['prefix']}_tensor_features_classified.json",
        "memory_level_accesses": f"workloads/{config['prefix']}_memory_level_accesses.json",
        "expanded_memory_graph": f"workloads/{config['prefix']}_expanded_memory_graph.json",
        "expanded_memory_graph_assigned": f"workloads/{config['prefix']}_expanded_memory_graph_assigned.json",
        "miqp_frontend_input": f"workloads/{config['prefix']}_miqp_frontend_input.json",
        "dump_folder": run_summary["dump_folder"],
    }


def run_workload(config):
    prefix = config["prefix"]
    yaml_path = f"workloads/{prefix}.yaml"
    features_path = f"workloads/{prefix}_tensor_features.json"
    reuse_path = f"workloads/{prefix}_tensor_features_with_reuse.json"
    classified_path = f"workloads/{prefix}_tensor_features_classified.json"
    memory_levels_path = f"workloads/{prefix}_memory_level_accesses.json"
    expanded_graph_path = f"workloads/{prefix}_expanded_memory_graph.json"
    assigned_graph_path = f"workloads/{prefix}_expanded_memory_graph_assigned.json"
    miqp_path = f"workloads/{prefix}_miqp_frontend_input.json"

    run_command(
        [
            sys.executable,
            "scripts/export_fx_model_to_zigzag.py",
            "--model",
            config["model"],
            "--out-prefix",
            prefix,
        ]
    )

    run_stdout = run_command(
        [
            sys.executable,
            "-u",
            "scripts/run_zigzag_workload.py",
            "--workload",
            yaml_path,
            "--accelerator",
            config["accelerator"],
            "--mapping",
            config["mapping"],
            "--name",
            prefix,
        ]
    )
    run_summary = parse_last_json_object(run_stdout)

    run_command(
        [
            sys.executable,
            "scripts/augment_tensor_features_with_cme_accesses.py",
            features_path,
            run_summary["pickle_filename"],
            reuse_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/classify_tensor_features.py",
            reuse_path,
            classified_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/extract_cme_memory_levels.py",
            run_summary["pickle_filename"],
            reuse_path,
            memory_levels_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/build_expanded_memory_graph.py",
            classified_path,
            memory_levels_path,
            expanded_graph_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/assign_ltram_stram.py",
            expanded_graph_path,
            assigned_graph_path,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/build_miqp_frontend_input.py",
            assigned_graph_path,
            miqp_path,
            "--workload-name",
            prefix,
        ]
    )

    return summarize_workload(
        config,
        run_summary,
        load_json(REPO_ROOT / classified_path),
        load_json(REPO_ROOT / assigned_graph_path),
    )


def run_llama_imports():
    decoupled_input = first_existing_path("workloads/llama1b_decoupled.json", "llama1b_decoupled.json")
    raw_input = first_existing_path("workloads/llama1b.json", "llama1b.json")
    converted_path = "workloads/llama1b_converted_decoupled.json"

    decoupled_stdout = run_command(
        [
            sys.executable,
            "scripts/import_external_llama_graph.py",
            "--input",
            decoupled_input,
            "--name",
            "llama1b_decoupled",
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/convert_llama_raw_to_decoupled.py",
            "--input",
            raw_input,
            "--output",
            converted_path,
        ]
    )

    converted_stdout = run_command(
        [
            sys.executable,
            "scripts/import_external_llama_graph.py",
            "--input",
            converted_path,
            "--name",
            "llama1b_converted",
            "--source-format",
            "external_llama_raw_converted",
        ]
    )

    decoupled = parse_last_json_object(decoupled_stdout)
    converted = parse_last_json_object(converted_stdout)
    return {
        "llama1b_decoupled": decoupled,
        "llama1b_converted": converted,
        "consistency": llama_consistency_checks(decoupled, converted),
    }


def llama_consistency_checks(decoupled, converted):
    checks = {
        "compute_nodes_match_expected_208": decoupled["num_compute_nodes"] == converted["num_compute_nodes"] == 208,
        "edges_match_expected_286": decoupled["num_edges"] == converted["num_edges"] == 286,
        "matmul_count_match_expected_144": (
            decoupled["op_histogram"].get("MatMul") == converted["op_histogram"].get("MatMul") == 144
        ),
        "weights_count_match_expected_145": (
            decoupled["persistent_tensor_category_histogram"].get("weights")
            == converted["persistent_tensor_category_histogram"].get("weights")
            == 145
        ),
        "kv_cache_unique_count_match": (
            decoupled["persistent_tensor_category_histogram"].get("kv_cache")
            == converted["persistent_tensor_category_histogram"].get("kv_cache")
        ),
        "kv_cache_access_count_match": (
            decoupled["tensor_category_histogram"].get("kv_cache")
            == converted["tensor_category_histogram"].get("kv_cache")
        ),
        "kv_cache_mismatch_explanation": (
            "Both paths expose 32 unique KV tensors (16 layers x K/V). The preferred decoupled graph has "
            "64 KV access records because each cache tensor has one read and one write. The raw graph exposes "
            "Slice cache read nodes only, so the converted graph has 32 KV access records and no KV writes."
        ),
    }
    required = [
        "compute_nodes_match_expected_208",
        "edges_match_expected_286",
        "matmul_count_match_expected_144",
        "weights_count_match_expected_145",
        "kv_cache_unique_count_match",
    ]
    failed = [name for name in required if not checks[name]]
    if failed:
        raise RuntimeError(f"Llama consistency checks failed: {failed}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="workloads/frontend_regression_summary.json")
    parser.add_argument("--include-llama", action="store_true")
    args = parser.parse_args()

    summaries = [run_workload(config) for config in TOY_WORKLOADS]
    output = {
        "workloads": summaries,
        "class_histogram": dict(
            sorted(
                Counter(
                    class_name
                    for summary in summaries
                    for class_name, count in summary["class_histogram"].items()
                    for _ in range(count)
                ).items()
            )
        ),
        "memory_assignment_histogram": normalized_histogram(
            Counter(
                memory
                for summary in summaries
                for memory, count in summary["memory_assignment_histogram"].items()
                for _ in range(count)
            ),
            MEMORY_TECHS,
        ),
        "tensor_category_histogram": normalized_histogram(
            Counter(
                category
                for summary in summaries
                for category, count in summary["tensor_category_histogram"].items()
                for _ in range(count)
            ),
            TENSOR_CATEGORIES,
        ),
    }
    if args.include_llama:
        output["external_llama"] = run_llama_imports()

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {args.output}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
