import argparse
import json
import operator
from pathlib import Path

import torch
import torch.fx as fx
import torch.nn as nn
import yaml
from torch.fx.passes.shape_prop import ShapeProp


class TwoConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 16, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        return x


class TinyResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        y = self.conv0(x)
        y = self.relu(y)
        y = self.conv1(y)
        return x + y


class OneLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc(x)


class TinyCNNLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def tensor_shape(node):
    meta = node.meta.get("tensor_meta")
    if meta is None:
        raise RuntimeError(f"Missing tensor metadata for FX node {node.name}")
    return tuple(meta.shape)


def normalize_source(input_source):
    return 0 if input_source is None else input_source


def conv2d_to_zigzag(layer_id, name, module, node, input_source):
    out_shape = tensor_shape(node)
    b, k, oy, ox = out_shape
    c = module.in_channels
    fy, fx_ = module.kernel_size
    sy, sx = module.stride
    dy, dx = module.dilation

    return {
        "id": layer_id,
        "name": name,
        "operator_type": "Conv",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": [
            f"ix={sx}*ox+{dx}*fx",
            f"iy={sy}*oy+{dy}*fy",
        ],
        "loop_dims": ["B", "K", "C", "OY", "OX", "FY", "FX"],
        "loop_sizes": [b, k, c, oy, ox, fy, fx_],
        "operand_precision": {
            "W": 8,
            "I": 8,
            "O": 16,
            "O_final": 8,
        },
        "operand_source": {
            "I": normalize_source(input_source),
            "W": 0,
        },
    }


def linear_to_zigzag(layer_id, name, module, node, input_source):
    out_shape = tensor_shape(node)
    if len(out_shape) < 2:
        raise RuntimeError(f"Linear output for {node.name} must have at least 2 dims, got {out_shape}")

    batch = 1
    for dim in out_shape[:-1]:
        batch *= dim

    return {
        "id": layer_id,
        "name": name,
        "operator_type": "Gemm",
        "equation": "O[d0][d1]+=I[d0][d2]*W[d2][d1]",
        "dimension_relations": [],
        "loop_dims": ["D0", "D1", "D2"],
        "loop_sizes": [batch, module.out_features, module.in_features],
        "operand_precision": {
            "W": 8,
            "I": 8,
            "O": 16,
            "O_final": 8,
        },
        "operand_source": {
            "I": normalize_source(input_source),
            "W": 0,
        },
    }


def make_tensor_feature(layer_name, layer_id, out_shape):
    return {
        "tensor": f"{layer_name}_O",
        "producer_layer_id": layer_id,
        "consumer_layer_ids": [],
        "shape": list(out_shape),
        "fanout": 0,
        "lifetime": 0,
        "reuse_proxy": 0,
        "semantic_type": "activation",
    }


def add_consumer(tensor_features, producer_layer_id, consumer_layer_id):
    if producer_layer_id is None:
        return
    consumers = tensor_features[producer_layer_id]["consumer_layer_ids"]
    if consumer_layer_id not in consumers:
        consumers.append(consumer_layer_id)


def first_node_arg(node):
    for arg in node.args:
        if isinstance(arg, fx.Node):
            return arg
    raise RuntimeError(f"Node {node.name} has no FX node input")


def export_model(model, example_input):
    gm = fx.symbolic_trace(model.eval())
    ShapeProp(gm).propagate(example_input)
    modules = dict(gm.named_modules())

    layers = []
    tensor_features = []
    node_to_producer_layer = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node_to_producer_layer[node] = None
            continue

        if node.op == "call_module":
            module = modules[node.target]
            input_node = first_node_arg(node)
            input_producer = node_to_producer_layer[input_node]

            if isinstance(module, nn.Conv2d):
                layer_id = len(layers)
                layer_name = str(node.target)
                layers.append(conv2d_to_zigzag(layer_id, layer_name, module, node, input_producer))
                tensor_features.append(make_tensor_feature(layer_name, layer_id, tensor_shape(node)))
                add_consumer(tensor_features, input_producer, layer_id)
                node_to_producer_layer[node] = layer_id
                continue

            if isinstance(module, nn.Linear):
                layer_id = len(layers)
                layer_name = str(node.target)
                layers.append(linear_to_zigzag(layer_id, layer_name, module, node, input_producer))
                tensor_features.append(make_tensor_feature(layer_name, layer_id, tensor_shape(node)))
                add_consumer(tensor_features, input_producer, layer_id)
                node_to_producer_layer[node] = layer_id
                continue

            if isinstance(module, (nn.ReLU, nn.Flatten)):
                node_to_producer_layer[node] = input_producer
                continue

            raise NotImplementedError(f"Unsupported module {node.target}: {type(module).__name__}")

        if node.op == "call_function" and node.target in {operator.add, torch.add}:
            input_producers = [
                node_to_producer_layer[arg]
                for arg in node.args
                if isinstance(arg, fx.Node)
            ]
            real_inputs = [producer for producer in input_producers if producer is not None]
            node_to_producer_layer[node] = real_inputs[0] if real_inputs else None
            continue

        if node.op == "output":
            continue

        raise NotImplementedError(f"Unsupported FX node {node.op} {node.target}")

    for feature in tensor_features:
        consumers = sorted(feature["consumer_layer_ids"])
        feature["consumer_layer_ids"] = consumers
        feature["fanout"] = len(consumers)
        feature["lifetime"] = max(consumers) - feature["producer_layer_id"] if consumers else 0
        feature["reuse_proxy"] = len(consumers)

    return layers, tensor_features


def build_model(name):
    if name == "two_conv":
        return TwoConv(), torch.randn(1, 3, 16, 16), "two_conv_auto_fx"
    if name == "tiny_residual":
        return TinyResidual(), torch.randn(1, 8, 16, 16), "tiny_residual_auto_fx"
    if name == "one_linear":
        return OneLinear(), torch.randn(1, 64), "one_linear_fx"
    if name == "tiny_cnn_linear":
        return TinyCNNLinear(), torch.randn(1, 3, 16, 16), "tiny_cnn_linear_auto_fx"
    raise ValueError(f"Unknown model {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["two_conv", "tiny_residual", "one_linear", "tiny_cnn_linear"],
        default="two_conv",
    )
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    model, example_input, default_prefix = build_model(args.model)
    out_prefix = args.out_prefix or default_prefix

    layers, tensor_features = export_model(model, example_input)

    out_dir = Path("workloads")
    out_dir.mkdir(exist_ok=True)
    yaml_path = out_dir / f"{out_prefix}.yaml"
    features_path = out_dir / f"{out_prefix}_tensor_features.json"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(layers, f, sort_keys=False)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(tensor_features, f, indent=2)

    print(f"Saved {yaml_path}")
    print(f"Saved {features_path}")
    print(f"Generated {len(layers)} ZigZag layers")
    print(json.dumps(tensor_features, indent=2))


if __name__ == "__main__":
    main()
