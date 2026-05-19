import sys
import pickle
from pathlib import Path

# Needed because pickled ZigZag objects reference the local zigzag package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

path = sys.argv[1]

with open(path, "rb") as f:
    cmes = pickle.load(f)


def movement_reads(moves):
    total = 0
    for move in moves:
        data = getattr(move, "data", {})
        for direction, value in data.items():
            if "rd_out" in str(direction):
                total += value
    return total


print(f"Loaded {len(cmes)} CMEs")

for i, item in enumerate(cmes):
    # ZigZag may store either CME directly or (CME, extra_info)
    if isinstance(item, tuple):
        cme = item[0]
        extra_info = item[1] if len(item) > 1 else None
    else:
        cme = item
        extra_info = None

    layer = getattr(cme, "layer", None)

    print(f"\nCME {i}")
    print(f"  cme_type: {type(cme)}")
    print(f"  layer: {layer}")
    print(f"  energy_total: {getattr(cme, 'energy_total', None)}")
    print(f"  latency_total0: {getattr(cme, 'latency_total0', None)}")
    print(f"  latency_total1: {getattr(cme, 'latency_total1', None)}")
    print(f"  latency_total2: {getattr(cme, 'latency_total2', None)}")

    mem_accesses = getattr(cme, "memory_word_access", None)
    if mem_accesses is None:
        mem_accesses = getattr(cme, "memory_word_accesses", None)

    print(f"  memory_word_accesses: {mem_accesses}")
    if mem_accesses is not None:
        for operand, moves in mem_accesses.items():
            print(f"    operand {operand}: total_reads={movement_reads(moves)}")
            if moves:
                first = moves[0]
                attrs = [name for name in dir(first) if not name.startswith("_")]
                print(f"      movement_type: {type(first)}")
                print(f"      movement_attrs: {attrs}")
    print(f"  extra_info_type: {type(extra_info) if extra_info is not None else None}")
