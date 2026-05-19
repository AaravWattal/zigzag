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


READ_RE = re.compile(r"rd [↑↓]: ([0-9]+(?:\.[0-9]+)?)")


def unwrap_cme(item):
    if isinstance(item, tuple):
        return item[0]
    return item


def movement_total_reads(moves):
    total = 0
    for move in moves:
        data = getattr(move, "data", None)
        if data is not None and DataDirection is not None:
            total += data.get(DataDirection.RD_OUT_TO_HIGH, 0)
            total += data.get(DataDirection.RD_OUT_TO_LOW, 0)
            continue

        for match in READ_RE.finditer(str(move)):
            total += float(match.group(1))

    return int(total) if float(total).is_integer() else total


def operand_reads(cme, operand_name):
    accesses = getattr(cme, "memory_word_accesses", None)
    if accesses is None:
        accesses = getattr(cme, "memory_word_access", None)
    if accesses is None:
        return 0

    total = 0
    for operand, moves in accesses.items():
        if str(operand) == operand_name:
            total += movement_total_reads(moves)
    return total


def augment_tensor_features(features, cmes):
    cmes = [unwrap_cme(cme) for cme in cmes]
    augmented = []

    for feature in features:
        feature = dict(feature)
        reuse = 0
        for consumer_layer_id in feature.get("consumer_layer_ids", []):
            if consumer_layer_id >= len(cmes):
                continue
            reuse += operand_reads(cmes[consumer_layer_id], "I")
        feature["reuse"] = reuse
        augmented.append(feature)

    return augmented


def default_out_path(features_path):
    path = Path(features_path)
    return path.with_name(f"{path.stem}_with_reuse.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor_features_json")
    parser.add_argument("cmes_pickle")
    parser.add_argument("output_json", nargs="?")
    args = parser.parse_args()

    with open(args.tensor_features_json, "r", encoding="utf-8") as f:
        features = json.load(f)

    with open(args.cmes_pickle, "rb") as f:
        cmes = pickle.load(f)

    out_path = Path(args.output_json) if args.output_json else default_out_path(args.tensor_features_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    augmented = augment_tensor_features(features, cmes)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(augmented, indent=2))


if __name__ == "__main__":
    main()
