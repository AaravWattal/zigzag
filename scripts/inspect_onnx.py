import sys
import onnx
from collections import Counter


path = sys.argv[1]
model = onnx.load(path)

ops = [node.op_type for node in model.graph.node]

print("ONNX op histogram:")
for op, count in Counter(ops).items():
    print(f"  {op}: {count}")

print("\nONNX nodes:")
for i, node in enumerate(model.graph.node):
    print(f"{i:03d}: {node.op_type:12s} name={node.name}")
    print(f"     inputs={list(node.input)}")
    print(f"     outputs={list(node.output)}")
