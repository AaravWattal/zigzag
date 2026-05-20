import yaml
import torch
import torch.nn as nn
import torch.fx as fx
from pathlib import Path


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def shape_prop(gm, example_input):
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(example_input)


def get_shape(node):
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None
    return tuple(meta.shape)


def conv_to_zigzag(layer_id, module, node, input_source):
    out_shape = get_shape(node)
    if out_shape is None:
        raise RuntimeError(f"Missing shape for node {node.name}")

    b, k, oy, ox = out_shape
    c = module.in_channels
    fy, fx = module.kernel_size
    sy, sx = module.stride
    dy, dx = module.dilation

    return {
        "id": layer_id,
        "name": node.name,
        "operator_type": "Conv",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": [f"ix={sx}*ox+{dx}*fx", f"iy={sy}*oy+{dy}*fy"],
        "loop_dims": ["B", "K", "C", "OY", "OX", "FY", "FX"],
        "loop_sizes": [b, k, c, oy, ox, fy, fx],
        "operand_precision": {"W": 8, "I": 8, "O": 16, "O_final": 8},
        "operand_source": {"I": input_source, "W": 0},
    }


def linear_to_zigzag(layer_id, module, node, input_source):
    out_shape = get_shape(node)
    if out_shape is None:
        raise RuntimeError(f"Missing shape for node {node.name}")

    if len(out_shape) == 2:
        b, n = out_shape
    else:
        b = 1
        n = out_shape[-1]

    k = module.in_features

    return {
        "id": layer_id,
        "name": node.name,
        "operator_type": "Gemm",
        "equation": "O[d0][d1]+=I[d0][d2]*W[d2][d1]",
        "dimension_relations": [],
        "loop_dims": ["D0", "D1", "D2"],
        "loop_sizes": [b, n, k],
        "operand_precision": {"W": 8, "I": 8, "O": 16, "O_final": 8},
        "operand_source": {"I": input_source, "W": 0},
    }


def main():
    Path("workloads").mkdir(exist_ok=True)
    model = TinyCNN().eval()
    example_input = torch.randn(1, 3, 32, 32)

    gm = fx.symbolic_trace(model)
    shape_prop(gm, example_input)
    modules = dict(gm.named_modules())

    layers = []
    last_real_layer_id = 0
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        module = modules[node.target]
        input_source = last_real_layer_id
        if isinstance(module, nn.Conv2d):
            layer_id = len(layers)
            layers.append(conv_to_zigzag(layer_id, module, node, input_source))
            last_real_layer_id = layer_id
        elif isinstance(module, nn.Linear):
            layer_id = len(layers)
            layers.append(linear_to_zigzag(layer_id, module, node, input_source))
            last_real_layer_id = layer_id

    out_path = "workloads/tiny_cnn_fx.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(layers, f, sort_keys=False)

    print(f"Saved {out_path}")
    print(f"Generated {len(layers)} ZigZag layers")
    for layer in layers:
        print(layer["id"], layer["operator_type"], layer["name"], layer["loop_sizes"])


if __name__ == "__main__":
    main()
