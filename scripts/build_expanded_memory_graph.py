import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def size_elements(shape):
    total = 1
    for dim in shape or []:
        total *= dim
    return total


def feature_by_producer(features):
    return {
        feature["producer_layer_id"]: feature
        for feature in features
        if feature.get("producer_layer_id") is not None
    }


def find_consumer_feature(features, layer_id, operand):
    for feature in features:
        if layer_id not in feature.get("consumer_layer_ids", []):
            continue
        roles = feature.get("consumer_operand_roles", {})
        if roles.get(str(layer_id), "I") == operand:
            return feature
    return None


def tensor_category(feature, operand, operator_type):
    if feature is not None and feature.get("semantic_type") == "kv_cache":
        return "kv_cache"
    if operand == "W" and feature is None:
        return "weights"
    if operand == "W" and operator_type == "Conv":
        return "weights"
    return "intermediate_activation"


def default_suggested_memory(category):
    if category == "weights":
        return "LtRAM"
    return "StRAM"


def operand_tensor_name(layer, operand, features, produced):
    layer_id = layer["layer_id"]
    layer_name = layer["layer_name"]

    if operand == "O":
        feature = produced.get(layer_id)
        return feature["tensor"] if feature else f"{layer_name}_O"

    feature = find_consumer_feature(features, layer_id, operand)
    if feature is not None:
        return feature["tensor"]

    if operand == "W":
        return f"{layer_name}_W"
    return "tensor_input"


def build_compute_node(layer, features, produced):
    memory_accesses = []
    for operand, levels in sorted(layer.get("operands", {}).items()):
        feature = None if operand == "O" else find_consumer_feature(features, layer["layer_id"], operand)
        category = tensor_category(feature, operand, layer["operator_type"])
        tensor_name = operand_tensor_name(layer, operand, features, produced)
        for level in levels:
            reads = level["total_reads"]
            writes = level["total_writes"]
            memory_accesses.append(
                {
                    "tensor": tensor_name,
                    "tensor_category": category,
                    "operand": operand,
                    "memory_level": level["level"],
                    "reads": reads,
                    "writes": writes,
                    "total_movement": level["total_movement"],
                    "reads_up": level["reads_up"],
                    "writes_down": level["writes_down"],
                    "reads_down": level["reads_down"],
                    "writes_up": level["writes_up"],
                    "suggested_memory": default_suggested_memory(category),
                }
            )

    inputs = []
    for operand in ("I", "W"):
        if operand in layer.get("operands", {}):
            inputs.append(operand_tensor_name(layer, operand, features, produced))

    outputs = []
    if "O" in layer.get("operands", {}):
        outputs.append(operand_tensor_name(layer, "O", features, produced))

    return {
        "node_id": f"compute_{layer['layer_id']}",
        "kind": "compute",
        "layer_id": layer["layer_id"],
        "layer_name": layer["layer_name"],
        "operator_type": layer["operator_type"],
        "inputs": inputs,
        "outputs": outputs,
        "memory_accesses": memory_accesses,
    }


def tensor_edge(feature):
    producer = feature.get("producer_layer_id")
    return {
        "tensor": feature["tensor"],
        "producer": f"compute_{producer}" if producer is not None else None,
        "consumers": [f"compute_{consumer}" for consumer in feature.get("consumer_layer_ids", [])],
        "metadata_consumers": feature.get("metadata_consumer_ids", []),
        "shape": feature.get("shape", []),
        "size_elements": size_elements(feature.get("shape", [])),
        "fanout": feature.get("fanout", 0),
        "lifetime": feature.get("lifetime", 0),
        "reuse": feature.get("reuse", 0),
        "class_name": feature.get("class_name"),
        "semantic_type": feature.get("semantic_type"),
        "category": "kv_cache" if feature.get("semantic_type") == "kv_cache" else "intermediate_activation",
    }


def build_expanded_graph(features, memory_levels):
    produced = feature_by_producer(features)
    layers = sorted(memory_levels["layers"], key=lambda layer: layer["layer_id"])
    return {
        "nodes": [build_compute_node(layer, features, produced) for layer in layers],
        "edges": [tensor_edge(feature) for feature in features],
    }


def default_out_path(classified_path):
    path = Path(classified_path)
    stem = path.stem
    if stem.endswith("_tensor_features_classified"):
        stem = stem.replace("_tensor_features_classified", "_expanded_memory_graph")
    else:
        stem = stem + "_expanded_memory_graph"
    return path.with_name(f"{stem}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor_features_classified_json")
    parser.add_argument("memory_level_accesses_json")
    parser.add_argument("output_json", nargs="?")
    args = parser.parse_args()

    features = load_json(args.tensor_features_classified_json)
    memory_levels = load_json(args.memory_level_accesses_json)
    graph = build_expanded_graph(features, memory_levels)

    out_path = Path(args.output_json) if args.output_json else default_out_path(args.tensor_features_classified_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(graph, indent=2))


if __name__ == "__main__":
    main()
