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
        self.stem = nn.Conv2d(8, 8, 1)
        self.conv0 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        residual = self.stem(x)
        y = self.conv0(residual)
        y = self.relu(y)
        y = self.conv1(y)
        return residual + y


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


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
        self.v = nn.Linear(16, 16)
        self.proj = nn.Linear(16, 16)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1))
        weights = self.softmax(scores)
        out = torch.matmul(weights, v)
        return self.proj(out)


def tensor_shape(node):
    meta = node.meta.get("tensor_meta")
    if meta is None:
        raise RuntimeError(f"Missing tensor metadata for FX node {node.name}")
    return tuple(meta.shape)


def normalize_source(input_source):
    if isinstance(input_source, tuple) and input_source[0] == "external":
        return 0
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


def gemm_to_zigzag(layer_id, name, d0, d1, d2, input_source, weight_source):
    operand_source = {
        "I": normalize_source(input_source),
        "W": normalize_source(weight_source),
    }

    return {
        "id": layer_id,
        "name": name,
        "operator_type": "Gemm",
        "equation": "O[d0][d1]+=I[d0][d2]*W[d2][d1]",
        "dimension_relations": [],
        "loop_dims": ["D0", "D1", "D2"],
        "loop_sizes": [d0, d1, d2],
        "operand_precision": {
            "W": 8,
            "I": 8,
            "O": 16,
            "O_final": 8,
        },
        "operand_source": operand_source,
    }


def linear_to_zigzag(layer_id, name, module, node, input_source):
    out_shape = tensor_shape(node)
    if len(out_shape) < 2:
        raise RuntimeError(f"Linear output for {node.name} must have at least 2 dims, got {out_shape}")

    batch = 1
    for dim in out_shape[:-1]:
        batch *= dim

    return gemm_to_zigzag(layer_id, name, batch, module.out_features, module.in_features, input_source, None)


def matmul_to_zigzag(layer_id, name, node, input_source, weight_source):
    lhs_shape = tensor_shape(node.args[0])
    rhs_shape = tensor_shape(node.args[1])
    out_shape = tensor_shape(node)

    d0 = 1
    for dim in lhs_shape[:-1]:
        d0 *= dim

    d1 = out_shape[-1]
    d2 = lhs_shape[-1]
    if rhs_shape[-2] != d2:
        raise RuntimeError(f"MatMul inner dimension mismatch for {node.name}: {lhs_shape} x {rhs_shape}")

    return gemm_to_zigzag(layer_id, name, d0, d1, d2, input_source, weight_source)


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


def add_consumer(tensor_features, producer_layer_id, consumer_layer_id, operand="I"):
    if producer_layer_id is None:
        return
    if isinstance(producer_layer_id, tuple):
        return
    feature = tensor_features[producer_layer_id]
    consumers = feature["consumer_layer_ids"]
    if consumer_layer_id not in consumers:
        consumers.append(consumer_layer_id)
    feature.setdefault("consumer_operand_roles", {})[str(consumer_layer_id)] = operand


def add_external_consumer(external_features, producer, consumer_layer_id, operand="I"):
    if not isinstance(producer, tuple) or producer[0] != "external":
        return
    feature = external_features[producer[1]]
    consumers = feature["consumer_layer_ids"]
    if consumer_layer_id not in consumers:
        consumers.append(consumer_layer_id)
    feature.setdefault("consumer_operand_roles", {})[str(consumer_layer_id)] = operand


def add_metadata_consumer(tensor_features, external_features, producer_layer_id, consumer_name):
    if producer_layer_id is None:
        return
    if isinstance(producer_layer_id, tuple):
        feature = external_features[producer_layer_id[1]]
    else:
        feature = tensor_features[producer_layer_id]
    consumers = feature.setdefault("metadata_consumer_ids", [])
    if consumer_name not in consumers:
        consumers.append(consumer_name)


def first_node_arg(node):
    for arg in node.args:
        if isinstance(arg, fx.Node):
            return arg
    raise RuntimeError(f"Node {node.name} has no FX node input")


def is_matmul_node(node):
    target = node.target
    return target in {torch.matmul, operator.matmul} or getattr(target, "__name__", None) == "matmul"


def export_model(model, example_input, include_external_inputs=False):
    gm = fx.symbolic_trace(model.eval())
    ShapeProp(gm).propagate(example_input)
    modules = dict(gm.named_modules())

    layers = []
    tensor_features = []
    external_features = []
    node_to_producer_layer = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if include_external_inputs:
                external_id = len(external_features)
                external_features.append(
                    {
                        "tensor": node.name,
                        "producer_layer_id": None,
                        "consumer_layer_ids": [],
                        "shape": list(tensor_shape(node)),
                        "fanout": 0,
                        "lifetime": 0,
                        "reuse_proxy": 0,
                        "semantic_type": "input",
                    }
                )
                node_to_producer_layer[node] = ("external", external_id)
            else:
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
                add_consumer(tensor_features, input_producer, layer_id, "I")
                add_external_consumer(external_features, input_producer, layer_id, "I")
                node_to_producer_layer[node] = layer_id
                continue

            if isinstance(module, nn.Linear):
                layer_id = len(layers)
                layer_name = str(node.target)
                layers.append(linear_to_zigzag(layer_id, layer_name, module, node, input_producer))
                tensor_features.append(make_tensor_feature(layer_name, layer_id, tensor_shape(node)))
                add_consumer(tensor_features, input_producer, layer_id, "I")
                add_external_consumer(external_features, input_producer, layer_id, "I")
                node_to_producer_layer[node] = layer_id
                continue

            if isinstance(module, (nn.ReLU, nn.Flatten, nn.Softmax)):
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
            for producer in real_inputs:
                add_metadata_consumer(tensor_features, external_features, producer, node.name)
            node_to_producer_layer[node] = real_inputs[0] if real_inputs else None
            continue

        if node.op == "call_function" and is_matmul_node(node):
            lhs, rhs = node.args[:2]
            if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
                raise RuntimeError(f"MatMul {node.name} expects FX node inputs")
            input_producer = node_to_producer_layer[lhs]
            weight_producer = node_to_producer_layer[rhs]
            layer_id = len(layers)
            layers.append(matmul_to_zigzag(layer_id, node.name, node, input_producer, weight_producer))
            tensor_features.append(make_tensor_feature(node.name, layer_id, tensor_shape(node)))
            add_consumer(tensor_features, input_producer, layer_id, "I")
            add_consumer(tensor_features, weight_producer, layer_id, "W")
            add_external_consumer(external_features, input_producer, layer_id, "I")
            add_external_consumer(external_features, weight_producer, layer_id, "W")
            node_to_producer_layer[node] = layer_id
            continue

        if node.op == "call_method" and node.target in {"transpose", "permute", "reshape", "view", "flatten"}:
            input_node = first_node_arg(node)
            node_to_producer_layer[node] = node_to_producer_layer[input_node]
            continue

        if node.op == "output":
            continue

        raise NotImplementedError(f"Unsupported FX node {node.op} {node.target}")

    for feature in external_features:
        consumers = sorted(feature["consumer_layer_ids"])
        metadata_consumers = sorted(feature.get("metadata_consumer_ids", []))
        feature["consumer_layer_ids"] = consumers
        if metadata_consumers:
            feature["metadata_consumer_ids"] = metadata_consumers
        feature["fanout"] = len(consumers) + len(metadata_consumers)
        feature["lifetime"] = max(consumers) if consumers else 0
        feature["reuse_proxy"] = feature["fanout"]

    for feature in tensor_features:
        consumers = sorted(feature["consumer_layer_ids"])
        metadata_consumers = sorted(feature.get("metadata_consumer_ids", []))
        feature["consumer_layer_ids"] = consumers
        if metadata_consumers:
            feature["metadata_consumer_ids"] = metadata_consumers
        feature["fanout"] = len(consumers) + len(metadata_consumers)
        feature["lifetime"] = max(consumers) - feature["producer_layer_id"] if consumers else 0
        feature["reuse_proxy"] = feature["fanout"]

    return layers, external_features + tensor_features


def build_model(name):
    if name == "two_conv":
        return TwoConv(), torch.randn(1, 3, 16, 16), "two_conv_auto_fx", False
    if name == "tiny_residual":
        return TinyResidual(), torch.randn(1, 8, 16, 16), "tiny_residual_auto_fx", False
    if name == "one_linear":
        return OneLinear(), torch.randn(1, 64), "one_linear_fx", False
    if name == "tiny_cnn_linear":
        return TinyCNNLinear(), torch.randn(1, 3, 16, 16), "tiny_cnn_linear_auto_fx", False
    if name == "tiny_attention":
        return TinyAttention(), torch.randn(1, 4, 16), "tiny_attention_auto_fx", True
    raise ValueError(f"Unknown model {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["two_conv", "tiny_residual", "one_linear", "tiny_cnn_linear", "tiny_attention"],
        default="two_conv",
    )
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    model, example_input, default_prefix, include_external_inputs = build_model(args.model)
    out_prefix = args.out_prefix or default_prefix

    layers, tensor_features = export_model(model, example_input, include_external_inputs=include_external_inputs)

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
