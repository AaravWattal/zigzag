import yaml
from pathlib import Path

Path("workloads").mkdir(exist_ok=True)

layers = [
    {
        "id": 0,
        "name": "conv0",
        "operator_type": "Conv",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": [
            "ix=1*ox+1*fx",
            "iy=1*oy+1*fy",
        ],
        "loop_dims": ["B", "K", "C", "OY", "OX", "FY", "FX"],
        "loop_sizes": [1, 8, 3, 16, 16, 3, 3],
        "operand_precision": {
            "W": 8,
            "I": 8,
            "O": 16,
            "O_final": 8,
        },
        "operand_source": {
            "I": 0,
            "W": 0,
        },
    },
    {
        "id": 1,
        "name": "conv1",
        "operator_type": "Conv",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": [
            "ix=1*ox+1*fx",
            "iy=1*oy+1*fy",
        ],
        "loop_dims": ["B", "K", "C", "OY", "OX", "FY", "FX"],
        "loop_sizes": [1, 16, 8, 16, 16, 3, 3],
        "operand_precision": {
            "W": 8,
            "I": 8,
            "O": 16,
            "O_final": 8,
        },
        "operand_source": {
            "I": 0,
            "W": 0,
        },
    },
]

with open("workloads/two_conv_fx.yaml", "w") as f:
    yaml.safe_dump(layers, f, sort_keys=False)

print("Saved workloads/two_conv_fx.yaml")
