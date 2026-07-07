"""Qualitative-case packaging for HUMAN analysis (material only, no conclusions).

Samples NN-cell and scaffold-only cases from the frozen round-1 reference
evaluations. The protocol requires that qualitative conclusions be written by
a human; this module only collects and packages the material.

CLI: python -m edr.analysis.qualitative [--output-dir outputs/ablations]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edr.config import load_config
from edr.io_utils import read_json, write_json
from edr.paths import ABLATION_OUT, REFERENCE_EVALS


def collect(args, config):
    seed_dirs = sorted(Path(REFERENCE_EVALS).glob("seed*"))
    if not seed_dirs:
        raise FileNotFoundError(f"no reference evaluations under {REFERENCE_EVALS}")

    by_episode = {}
    seeds = []
    for seed_dir in seed_dirs:
        heldout = read_json(seed_dir / "heldout.json")
        seed = heldout["summary"]["seed"]
        seeds.append(str(seed))
        for record in heldout["records"]:
            entry = by_episode.setdefault(
                record["episode_id"],
                {
                    "episode_id": record["episode_id"],
                    "generalization": record.get("generalization"),
                    "partition": record.get("partition"),
                    "teacher_prediction": record.get("teacher_prediction"),
                    "teacher_success": record.get("teacher_success"),
                    "per_seed": {},
                },
            )
            entry["per_seed"][str(seed)] = {
                "success": record["success"],
                "prediction": record.get("prediction"),
                "raw_model_output": record.get("raw_model_output"),
                "teacher_agree": record.get("teacher_agree"),
            }

    def majority_success(entry):
        values = [row["success"] for row in entry["per_seed"].values()]
        return sum(values) >= (len(values) / 2.0)

    nn = [entry for entry in by_episode.values() if entry["generalization"] == "NN"]
    scaffold_only = [entry for entry in by_episode.values() if entry["partition"] == "scaffold_only"]
    so_success = [entry for entry in scaffold_only if majority_success(entry)]
    so_failure = [entry for entry in scaffold_only if not majority_success(entry)]

    sample_size = config["qualitative"]["sample_size"]
    package = []
    for entry in nn:
        package.append({**entry, "stratum": "NN", "majority_success": majority_success(entry)})
    fill = sample_size - len(package)
    half = max(0, fill) // 2
    for entry in sorted(so_success, key=lambda e: e["episode_id"])[:half]:
        package.append({**entry, "stratum": "scaffold_only_success", "majority_success": True})
    for entry in sorted(so_failure, key=lambda e: e["episode_id"])[: max(0, fill) - half]:
        package.append({**entry, "stratum": "scaffold_only_failure", "majority_success": False})

    payload = {
        "probe": "qualitative_case_package",
        "note": "MATERIAL ONLY — qualitative conclusions must be written by a human.",
        "nn_total": len(nn),
        "scaffold_only_total": len(scaffold_only),
        "package_size": len(package),
        "seeds": seeds,
        "cases": package,
    }
    output = Path(args.output_dir) / "qualitative" / "case_package.json"
    write_json(output, payload)
    print(f"[qualitative] NN={len(nn)} package={len(package)} -> {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ABLATION_OUT))
    args = parser.parse_args(argv)
    collect(args, load_config("ablations"))


if __name__ == "__main__":
    main()
