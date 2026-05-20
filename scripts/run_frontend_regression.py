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


def summarize_workload(config, run_summary, classified_features):
    histogram = Counter(feature["class_name"] for feature in classified_features)
    lifetimes = [feature.get("lifetime", 0) or 0 for feature in classified_features]
    fanouts = [feature.get("fanout", 0) or 0 for feature in classified_features]
    reuses = [feature.get("reuse", 0) or 0 for feature in classified_features]

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
        "yaml": yaml_path,
        "cmes_pickle": run_summary["pickle_filename"],
        "tensor_features": f"workloads/{config['prefix']}_tensor_features.json",
        "tensor_features_with_reuse": f"workloads/{config['prefix']}_tensor_features_with_reuse.json",
        "tensor_features_classified": f"workloads/{config['prefix']}_tensor_features_classified.json",
        "dump_folder": run_summary["dump_folder"],
    }


def run_workload(config):
    prefix = config["prefix"]
    yaml_path = f"workloads/{prefix}.yaml"
    features_path = f"workloads/{prefix}_tensor_features.json"
    reuse_path = f"workloads/{prefix}_tensor_features_with_reuse.json"
    classified_path = f"workloads/{prefix}_tensor_features_classified.json"

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

    return summarize_workload(config, run_summary, load_json(REPO_ROOT / classified_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="workloads/frontend_regression_summary.json")
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
    }

    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {args.output}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
