#!/usr/bin/env python3
"""Step 1.5 material collection: sample NN-cell cases for HUMAN analysis.

Discipline note (PROJECT_MASTER_PLAN Step 1.5): this step is judgment-dense.
The agent/launcher only samples and packages material; conclusions are written
by a human. This script therefore produces prompts + model outputs + teacher
references and STOPS.

Source: the five A3 rep2 heldout evaluations (repro_rep2/artifacts). NN cells
on heldout are tiny (n=2 in the Phase 0 run), so the package pads with
scaffold-only successes and failures up to the configured sample size, each
record labelled with its stratum.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phase1.common import (
    add_common_args,
    load_config,
    read_json,
    write_json,
    REPRO_ROOT,
)


def collect(args, config):
    artifact_root = REPRO_ROOT / "artifacts"
    seed_dirs = sorted(artifact_root.glob("main_rep2_seed*"))
    if not seed_dirs:
        raise FileNotFoundError(f"no rep2 artifacts under {artifact_root}")

    by_episode = {}
    for seed_dir in seed_dirs:
        heldout = read_json(seed_dir / "heldout.json")
        seed = heldout["summary"]["seed"]
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

    sample_size = config["phase1"]["step15"]["sample_size"]
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
        "probe": "step15_nn_sample_package",
        "note": "MATERIAL ONLY — qualitative conclusions must be written by a human (Master Plan Step 1.5).",
        "nn_total": len(nn),
        "scaffold_only_total": len(scaffold_only),
        "package_size": len(package),
        "seeds": [str(read_json(d / "heldout.json")["summary"]["seed"]) for d in seed_dirs],
        "cases": package,
    }
    output = Path(args.output_dir) / "step15" / "step15_nn_package.json"
    write_json(output, payload)
    print(f"[step15] NN={len(nn)} package={len(package)} -> {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    collect(args, load_config(args.config))


if __name__ == "__main__":
    main()
