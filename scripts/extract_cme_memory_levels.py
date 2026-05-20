import argparse
import json
import pickle
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from zigzag.hardware.architecture.memory_port import DataDirection
except Exception:
    DataDirection = None


MOVE_RE = re.compile(
    r"rd ↑: (?P<reads_up>[0-9]+(?:\.[0-9]+)?), "
    r"wr ↓: (?P<writes_down>[0-9]+(?:\.[0-9]+)?), "
    r"rd ↓: (?P<reads_down>[0-9]+(?:\.[0-9]+)?), "
    r"wr ↑: (?P<writes_up>[0-9]+(?:\.[0-9]+)?)"
)


def unwrap_cme(item):
    return item[0] if isinstance(item, tuple) else item


def as_number(value):
    value = float(value)
    return int(value) if value.is_integer() else value


def movement_to_record(level, movement):
    data = getattr(movement, "data", None)
    if data is not None and DataDirection is not None:
        reads_up = data.get(DataDirection.RD_OUT_TO_HIGH, 0)
        writes_down = data.get(DataDirection.WR_IN_BY_HIGH, 0)
        reads_down = data.get(DataDirection.RD_OUT_TO_LOW, 0)
        writes_up = data.get(DataDirection.WR_IN_BY_LOW, 0)
    else:
        match = MOVE_RE.search(str(movement))
        if match is None:
            raise RuntimeError(f"Could not parse memory movement: {movement}")
        reads_up = as_number(match.group("reads_up"))
        writes_down = as_number(match.group("writes_down"))
        reads_down = as_number(match.group("reads_down"))
        writes_up = as_number(match.group("writes_up"))

    return {
        "level": level,
        "reads_up": reads_up,
        "writes_down": writes_down,
        "reads_down": reads_down,
        "writes_up": writes_up,
        "total_reads": reads_up + reads_down,
        "total_writes": writes_down + writes_up,
        "total_movement": reads_up + writes_down + reads_down + writes_up,
    }


def layer_accesses(cme):
    accesses = getattr(cme, "memory_word_accesses", None)
    if accesses is None:
        accesses = getattr(cme, "memory_word_access", None)
    if accesses is None:
        return {}

    operands = {}
    for operand, movements in accesses.items():
        operands[str(operand)] = [
            movement_to_record(level, movement)
            for level, movement in enumerate(movements)
        ]
    return operands


def extract_memory_levels(cmes):
    layers = []
    for idx, item in enumerate(cmes):
        cme = unwrap_cme(item)
        layer = getattr(cme, "layer", None)
        layers.append(
            {
                "layer_id": getattr(layer, "id", idx),
                "layer_name": str(getattr(layer, "name", f"layer_{idx}")),
                "operator_type": str(getattr(layer, "type", "unknown")),
                "operands": layer_accesses(cme),
            }
        )
    return {"layers": layers}


def default_out_path(features_path):
    path = Path(features_path)
    stem = path.stem
    if stem.endswith("_tensor_features_with_reuse"):
        stem = stem.replace("_tensor_features_with_reuse", "_memory_level_accesses")
    elif stem.endswith("_with_reuse"):
        stem = stem.removesuffix("_with_reuse") + "_memory_level_accesses"
    else:
        stem = stem + "_memory_level_accesses"
    return path.with_name(f"{stem}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmes_pickle")
    parser.add_argument("tensor_features_with_reuse_json")
    parser.add_argument("output_json", nargs="?")
    args = parser.parse_args()

    with open(args.cmes_pickle, "rb") as f:
        cmes = pickle.load(f)

    out_path = Path(args.output_json) if args.output_json else default_out_path(args.tensor_features_with_reuse_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = extract_memory_levels(cmes)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
