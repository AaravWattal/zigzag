import sys
import json
import yaml
from pathlib import Path

yaml_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else yaml_path.replace(".yaml", "_tensor_features.json")

with open(yaml_path) as f:
    layers = yaml.safe_load(f)

features = []

for layer in layers:
    producer_id = layer["id"]
    tensor_name = f"{layer.get('name', 'layer_' + str(producer_id))}_O"

    consumers = []
    for other in layers:
        if other["id"] == producer_id:
            continue
        src = other.get("operand_source", {}).get("I", None)
        if src == producer_id:
            consumers.append(other["id"])

    lifetime = max(consumers) - producer_id if consumers else 0

    features.append({
        "tensor": tensor_name,
        "producer_layer_id": producer_id,
        "consumer_layer_ids": consumers,
        "fanout": len(consumers),
        "lifetime": lifetime,
        "reuse_proxy": len(consumers),
        "semantic_type": "activation",
    })

Path(out_path).parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(features, f, indent=2)

print(f"Saved {out_path}")
print(json.dumps(features, indent=2))
