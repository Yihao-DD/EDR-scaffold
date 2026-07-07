#!/usr/bin/env python3
"""A7/A8 dataset builders (CPU-only, deterministic).

A7 replay-ablation: the rep2 core (148 rows) with ZERO replay rows.
  Physical corollary (v1.26 ruling): the KL anchor fires on replay rows only,
  so removing replay also removes the KL anchor. A7 is therefore
  "rep2 minus replay minus KL" by definition; the launcher passes
  --kl-anchor-lambda 0 explicitly and this is reported as such.

A8 data-scale: episode-level subsets of the rep2 core at 25% / 50%, with the
  replay block scaled to keep the 2:1 replay:core ROW RATIO (the ratio is part
  of the recipe; the absolute count is not — v1.26 ruling). 100% is A3 itself
  (rep2 artifacts are reused, not retrained).

Both reuse the frozen rep2 rows verbatim — no re-generation, no new teacher
forwards — so the isolated variable is exactly replay presence (A7) or data
quantity (A8).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from phase1.common import (
    add_common_args,
    assert_training_disjoint,
    deterministic_sample,
    load_config,
    replay_rows_from_capped,
    write_json,
    write_jsonl,
)


def subsample_episodes(core_rows, fraction, seed):
    """Deterministic episode-level subset: hash-order episodes, take the prefix."""

    episodes = sorted({row["episode_id"] for row in core_rows})
    keyed = sorted(
        (hashlib.sha256(f"{seed}:a8:{episode_id}".encode("utf-8")).hexdigest(), episode_id)
        for episode_id in episodes
    )
    take = max(1, round(len(episodes) * fraction))
    selected = {episode_id for _, episode_id in keyed[:take]}
    return [row for row in core_rows if row["episode_id"] in selected], selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = config["dataset_seed"]
    core, replay = replay_rows_from_capped()
    output_dir = Path(args.output_dir)

    summaries = {}

    # --- A7: core only, zero replay -------------------------------------
    a7_rows = list(core)
    leakage = assert_training_disjoint(a7_rows, "a7")
    a7_path = output_dir / "a7" / "distill_a7_noreplay.jsonl"
    summaries["a7"] = {
        "probe": "a7_noreplay_dataset",
        "core_rows": len(a7_rows),
        "replay_rows": 0,
        "total_rows": len(a7_rows),
        "kl_anchor": "inert (no replay rows) — launcher passes --kl-anchor-lambda 0; arm is rep2 minus replay minus KL",
        "leakage_asserts": leakage,
        "dataset_output": str(a7_path),
    }
    if not args.dry_run:
        write_jsonl(a7_path, a7_rows)

    # --- A8: fraction subsets with 2:1 replay ratio ----------------------
    for fraction in config["phase1"]["a8"]["fractions"]:
        tag = f"{int(fraction * 100)}"
        sub_core, selected_episodes = subsample_episodes(core, fraction, seed)
        replay_target = 2 * len(sub_core)
        sub_replay = deterministic_sample(replay, min(replay_target, len(replay)), seed, f"a8_replay_{tag}")
        rows = sub_core + sub_replay
        leakage = assert_training_disjoint(rows, f"a8_{tag}")
        path = output_dir / "a8" / f"distill_a8_{tag}pct.jsonl"
        summaries[f"a8_{tag}"] = {
            "probe": f"a8_{tag}pct_dataset",
            "fraction": fraction,
            "core_rows": len(sub_core),
            "core_unique_episodes": len(selected_episodes),
            "replay_rows": len(sub_replay),
            "replay_ratio_target": "2:1",
            "total_rows": len(rows),
            "leakage_asserts": leakage,
            "dataset_output": str(path),
        }
        if not args.dry_run:
            write_jsonl(path, rows)

    summary_path = output_dir / "a7a8_dataset_summary.json"
    payload = {"probe": "a7_a8_datasets", "dataset_seed": seed, "dry_run": bool(args.dry_run), "datasets": summaries}
    write_json(summary_path, payload)
    for name, item in summaries.items():
        print(f"[a7a8-build] {name}: total={item['total_rows']} core={item['core_rows']} replay={item['replay_rows']}")


if __name__ == "__main__":
    main()
