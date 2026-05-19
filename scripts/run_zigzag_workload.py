import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zigzag import api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--accelerator", default="zigzag/inputs/hardware/tpu_like.yaml")
    parser.add_argument("--mapping", default="zigzag/inputs/mapping/tpu_like.yaml")
    parser.add_argument("--name", default="run")
    parser.add_argument("--lpf-limit", type=int, default=2)
    parser.add_argument("--nb-spatial-mappings", type=int, default=1)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_folder = f"outputs/{args.name}_{ts}"
    pickle_filename = f"{dump_folder}/cmes.pickle"
    Path(dump_folder).mkdir(parents=True, exist_ok=True)

    energy, latency, cmes = api.get_hardware_performance_zigzag(
        workload=args.workload,
        accelerator=args.accelerator,
        mapping=args.mapping,
        opt="energy",
        dump_folder=dump_folder,
        pickle_filename=pickle_filename,
        lpf_limit=args.lpf_limit,
        nb_spatial_mappings_generated=args.nb_spatial_mappings,
        loma_show_progress_bar=False,
    )

    try:
        with open(pickle_filename, "rb") as f:
            saved_cmes = pickle.load(f)
        num_saved_cmes = len(saved_cmes)
    except Exception:
        num_saved_cmes = len(cmes)

    summary = {
        "workload": args.workload,
        "accelerator": args.accelerator,
        "mapping": args.mapping,
        "energy_pj": energy,
        "latency_cycles": latency,
        "num_cmes": num_saved_cmes,
        "dump_folder": dump_folder,
        "pickle_filename": pickle_filename,
        "lpf_limit": args.lpf_limit,
        "nb_spatial_mappings": args.nb_spatial_mappings,
    }

    with open(f"{dump_folder}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
