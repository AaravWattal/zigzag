import argparse
import json
from pathlib import Path


LOW_REUSE_THRESHOLD = 4
HIGH_REUSE_THRESHOLD = 128

CLASSES = {
    "persistent_reused": {
        "class_id": 0,
        "recommended_memory": "RRAM/MRAM",
        "recommended_fabric": "broadcast fabric",
    },
    "broadcast_activation": {
        "class_id": 1,
        "recommended_memory": "SRAM/eDRAM",
        "recommended_fabric": "broadcast/tree fabric",
    },
    "ephemeral_activation": {
        "class_id": 2,
        "recommended_memory": "SRAM",
        "recommended_fabric": "local fabric",
    },
    "high_reuse_activation": {
        "class_id": 3,
        "recommended_memory": "eDRAM/SRAM",
        "recommended_fabric": "high-bandwidth fabric",
    },
    "standard_activation": {
        "class_id": 4,
        "recommended_memory": "SRAM/eDRAM",
        "recommended_fabric": "mesh/default fabric",
    },
}


def is_persistent_parameter(feature):
    semantic_type = str(feature.get("semantic_type", "")).lower()
    tensor_name = str(feature.get("tensor", "")).lower()
    return (
        semantic_type in {"weight", "weights", "parameter", "persistent_parameter"}
        or feature.get("persistent", False)
        or tensor_name.endswith("_w")
        or tensor_name.endswith("_weight")
    )


def classify_feature(feature):
    fanout = int(feature.get("fanout", 0) or 0)
    lifetime = int(feature.get("lifetime", 0) or 0)
    reuse = float(feature.get("reuse", feature.get("reuse_proxy", 0)) or 0)

    if is_persistent_parameter(feature):
        class_name = "persistent_reused"
        reason = "Tensor is marked as a persistent parameter or weight."
    elif fanout >= 3:
        class_name = "broadcast_activation"
        reason = f"Fanout {fanout} is at least 3, so broadcast delivery is preferred."
    elif lifetime <= 1 and reuse <= LOW_REUSE_THRESHOLD:
        class_name = "ephemeral_activation"
        reason = f"Lifetime {lifetime} is at most 1 and reuse {reuse:g} is low."
    elif reuse >= HIGH_REUSE_THRESHOLD:
        class_name = "high_reuse_activation"
        reason = f"Reuse {reuse:g} is at least {HIGH_REUSE_THRESHOLD}."
    else:
        class_name = "standard_activation"
        reason = "No specialized persistence, broadcast, ephemeral, or high-reuse rule matched."

    class_info = CLASSES[class_name]
    classified = dict(feature)
    classified.update(
        {
            "class_id": class_info["class_id"],
            "class_name": class_name,
            "recommended_memory": class_info["recommended_memory"],
            "recommended_fabric": class_info["recommended_fabric"],
            "reason": reason,
        }
    )
    return classified


def classify_features(features):
    return [classify_feature(feature) for feature in features]


def default_out_path(features_path):
    path = Path(features_path)
    stem = path.stem
    if stem.endswith("_tensor_features_with_reuse"):
        stem = stem.replace("_tensor_features_with_reuse", "_tensor_features_classified")
    elif stem.endswith("_with_reuse"):
        stem = stem.removesuffix("_with_reuse") + "_classified"
    else:
        stem = stem + "_classified"
    return path.with_name(f"{stem}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor_features_with_reuse_json")
    parser.add_argument("output_json", nargs="?")
    args = parser.parse_args()

    in_path = Path(args.tensor_features_with_reuse_json)
    out_path = Path(args.output_json) if args.output_json else default_out_path(in_path)

    with open(in_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    classified = classify_features(features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(classified, f, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(classified, indent=2))


if __name__ == "__main__":
    main()
