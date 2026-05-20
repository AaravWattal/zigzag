import torch
import torch.nn as nn
import onnx
from onnx import shape_inference

class TwoConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)

model = TwoConv().eval()
x = torch.randn(1, 3, 16, 16)

raw = "workloads/two_conv.raw.onnx"
inferred = "workloads/two_conv.inferred.onnx"

torch.onnx.export(
    model,
    x,
    raw,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

m = onnx.load(raw)
onnx.checker.check_model(m)
m = shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, inferred)

print(f"Saved {inferred}")
