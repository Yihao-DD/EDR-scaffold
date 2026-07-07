#!/usr/bin/env python3
"""A5 step 2: build the unverified-distilled dataset (AST filter OFF).

Pipeline parity with A3 (single-variable isolation, v1.26 ruling):
- same prompt construction (original no-patch input),
- same canonical output serialization,
- same dedupe rule (episode_id, source, output),
- same deterministic downsample, same replay block (the frozen 296 rep2
  replay rows, verbatim), same row budget (148 core rows).
The ONLY change: samples are admitted regardless of the AST verdict.

Unparseable raw outputs cannot be serialized by the shared pipeline
(output_text requires a prediction); they are counted and reported, not
trained on. This keeps parsing — a mechanical pipeline stage — fixed, and
switches off only the verifier. Registered in CHANGELOG v1.26.

CPU-only. Consumes a5_teacher_all_shard_*.json from a5_resample_teacher.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phase1.common import (
    add_common_args,
    assert_training_disjoint,
    deterministic_sample,
    load_config,
    load_inputs,
    load_probe_module,
    read_json,
    replay_rows_from_capped,
    write_json,
    write_jsonl,
)


def collect_unfiltered_rows(shard_paths, split, baseline, probe):
    train_by_id = {item["episode_id"]: item for item in split["train"]}
    rows = []
    unparseable = 0
    total = 0
    verified = 0
    for path in shard_paths:
        payload = read_json(path)
        if payload.get("probe") != "a5_teacher_all_samples":
            raise AssertionError({"assert": "a5_shard_kind", "path": str(path), "probe": payload.get("probe")})
        for record in payload["records"]:
            episode = train_by_id.get(record["episode_id"])
            if episode is None:
                raise AssertionError({"assert": "a5_episode_in_train_share", "episode_id": record["episode_id"]})
            for sample in record["samples"]:
                total += 1
                if sample["success"]:
                    verified += 1
                if sample["prediction"] is None:
                    unparseable += 1
                    continue
                source = "teacher_unverified_t0" if sample["kind"] == "t0" else "teacher_unverified_t08"
                row = probe.make_distill_row(
                    episode,
                    baseline["prompt_template"],
                    sample["prediction"],
                    source,
                    "unverified",
                    sample.get("sample_index"),
                )
                row["ast_pass"] = bool(sample["success"])
                rows.append(row)
    return rows, {"samples_total": total, "samples_ast_pass": verified, "samples_unparseable": unparseable}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--shard-glob", default="a5/a5_teacher_all_shard_*.json")
    parser.add_argument("--dataset-output", default=None)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    a5 = config["phase1"]["a5"]
    probe = load_probe_module()
    baseline, exp2_result, split, patches, evo = load_inputs(config["split_seed"])

    output_dir = Path(args.output_dir)
    shard_paths = sorted(output_dir.glob(args.shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"no A5 teacher shards matched {args.shard_glob} under {output_dir}")

    rows, funnel = collect_unfiltered_rows(shard_paths, split, baseline, probe)
    rows = probe.dedupe_rows(rows)
    core = deterministic_sample(rows, a5["core_rows"], config["dataset_seed"], "a5_core")
    if len(core) < a5["core_rows"]:
        print(f"[a5-build:warn] only {len(core)} unique rows available for a {a5['core_rows']}-row budget")

    _, replay = replay_rows_from_capped()
    dataset = core + replay

    leakage = assert_training_disjoint(dataset, "a5")

    dataset_output = args.dataset_output or (output_dir / "a5" / "distill_a5_unverified.jsonl")
    summary_output = args.summary_output or (output_dir / "a5" / "a5_dataset_summary.json")

    core_episodes = {row["episode_id"] for row in core}
    summary = {
        "probe": "a5_unverified_dataset",
        "mode": a5["mode"],
        "funnel": funnel,
        "unfiltered_rows_available": len(rows),
        "core_rows": len(core),
        "core_unique_episodes": len(core_episodes),
        "core_ast_pass_rows": sum(1 for row in core if row.get("ast_pass")),
        "core_ast_fail_rows": sum(1 for row in core if not row.get("ast_pass")),
        "replay_rows": len(replay),
        "total_rows": len(dataset),
        "row_matched_to": "distill_main_replay2_capped.jsonl (444 rows)",
        "leakage_asserts": leakage,
        "dataset_seed": config["dataset_seed"],
        "shards": [str(path) for path in shard_paths],
    }

    if args.dry_run:
        summary["dry_run"] = True
        write_json(summary_output, summary)
        print(f"[a5-build:dry-run] core={len(core)} (ast_fail={summary['core_ast_fail_rows']}) total={len(dataset)}")
        return

    write_jsonl(dataset_output, dataset)
    summary["dataset_output"] = str(dataset_output)
    write_json(summary_output, summary)
    print(
        f"[a5-build] rows={len(dataset)} core={len(core)} "
        f"(ast_fail={summary['core_ast_fail_rows']}/{len(core)}) episodes={len(core_episodes)} -> {dataset_output}"
    )


if __name__ == "__main__":
    main()
