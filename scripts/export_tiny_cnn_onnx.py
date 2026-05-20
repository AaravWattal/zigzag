import torch
import torch.nn as nn
import onnx
from onnx import shape_inference


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


model = TinyCNN().eval()
x = torch.randn(1, 3, 32, 32)

raw_path = "workloads/tiny_cnn.raw.onnx"
inferred_path = "workloads/tiny_cnn.inferred.onnx"

torch.onnx.export(
    model,
    x,
    raw_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

m = onnx.load(raw_path)
onnx.checker.check_model(m)

m = shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, inferred_path)

print(f"Saved {inferred_path}")
