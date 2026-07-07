"""Round-1 data reproduction pipeline (from-scratch rebuild of the shipped
frozen datasets). The outputs of this pipeline are already shipped under
data/round1/, so running it is only needed for full-provenance reproduction.

Subcommands (identical semantics to the frozen round-1 probe):
  prepare               split sizes + regression sets manifest
  run-shard             teacher T=0 + base pass@16 over train∪val failures (GPU)
  sample-teacher-shard  teacher T=0.8 augmentation over repaired episodes (GPU)
  merge-shards          inventory + pass@16 partition + repaired list
  build-datasets        distillation sets (train-share only, leakage-asserted)
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from edr.io_utils import read_json, write_json
from edr.data.distill import dedupe_rows, deterministic_sample, make_distill_row
from edr.data.leakage import assert_no_leakage, assert_no_selection_leakage
from edr.data.splits import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGRESSION_SEED,
    DEFAULT_SPLIT_SEED,
    failure_type,
    function_name,
    load_round1_inputs,
    partition_episode,
    select_regression_sets,
)
from edr.io_utils import read_jsonl, write_jsonl
from edr.paths import BASE_FAILURES, H1_EVOLUTION, OUTPUT_DIR


def shard_items(items, shard_index, shard_count):
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [item for index, item in enumerate(items) if index % shard_count == shard_index]


def run_shard(args):
    from edr.modeling import GenerationRunner
    from edr.scaffold.teacher import evaluate_teacher_episode, sample_base_episode

    baseline, _evolution, split, patches = load_round1_inputs(args.split_seed, args.failures, args.evolution)
    train_val = split["train"] + split["validation"]
    shard = shard_items(train_val, args.shard_index, args.shard_count)
    runner = GenerationRunner(args.model_id, cache_dir=args.model_cache_dir)
    records = []
    for index, episode in enumerate(shard, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(shard):
            print(
                f"[round1-shard {args.shard_index}/{args.shard_count}] episode {index}/{len(shard)} "
                f"{episode['episode_id']}",
                flush=True,
            )
        teacher = evaluate_teacher_episode(runner, episode, patches, args.max_new_tokens)
        pass16 = sample_base_episode(
            runner,
            episode,
            baseline["prompt_template"],
            args.sample_count,
            args.sample_batch_size,
            args.temperature,
            args.max_new_tokens,
            args.split_seed,
        )
        records.append(
            {
                "episode_id": episode["episode_id"],
                "failure_class": failure_type(episode),
                "function_name": function_name(episode),
                "split_partition": "train" if episode in split["train"] else "validation",
                "teacher": teacher,
                "pass16": pass16,
                "partition": partition_episode(pass16["pass16_hits"], teacher["success"]),
            }
        )
    payload = {
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sample_count": args.sample_count,
        "temperature": args.temperature,
        "records": records,
    }
    write_json(args.output, payload)


def run_teacher_sample_shard(args):
    from edr.modeling import GenerationRunner
    from edr.scaffold.teacher import sample_teacher_episode

    baseline, _evolution, split, patches = load_round1_inputs(args.split_seed, args.failures, args.evolution)
    train_val_by_id = {item["episode_id"]: item for item in split["train"] + split["validation"]}
    train_ids = {item["episode_id"] for item in split["train"]}
    repaired = read_json(args.repaired_input)
    repaired = [row for row in repaired if row["episode_id"] in train_val_by_id]
    if args.train_only:
        repaired = [row for row in repaired if row["episode_id"] in train_ids]
    shard = shard_items(repaired, args.shard_index, args.shard_count)
    runner = GenerationRunner(args.model_id, cache_dir=args.model_cache_dir)
    records = []
    for index, row in enumerate(shard, start=1):
        episode = train_val_by_id[row["episode_id"]]
        if index == 1 or index % args.progress_every == 0 or index == len(shard):
            print(
                f"[teacher-sample {args.shard_index}/{args.shard_count}] episode {index}/{len(shard)} "
                f"{episode['episode_id']}",
                flush=True,
            )
        sample = sample_teacher_episode(
            runner,
            episode,
            patches,
            args.sample_count,
            args.sample_batch_size,
            args.temperature,
            args.max_new_tokens,
            args.split_seed,
        )
        sample.update(
            {
                "failure_class": failure_type(episode),
                "function_name": function_name(episode),
                "partition": row.get("partition", "unknown"),
            }
        )
        records.append(sample)
    write_json(
        args.output,
        {
            "model_id": args.model_id,
            "split_seed": args.split_seed,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "sample_count": args.sample_count,
            "temperature": args.temperature,
            "records": records,
        },
    )


def merge_shard_payloads(shard_payloads, expected_episode_ids):
    by_id = {}
    for payload in shard_payloads:
        for row in payload["records"]:
            episode_id = row["episode_id"]
            if episode_id in by_id:
                raise ValueError(f"duplicate episode in shards: {episode_id}")
            by_id[episode_id] = row
    missing = sorted(set(expected_episode_ids) - set(by_id))
    extra = sorted(set(by_id) - set(expected_episode_ids))
    if missing or extra:
        raise ValueError({"missing": missing[:20], "extra": extra[:20], "n_missing": len(missing), "n_extra": len(extra)})
    return [by_id[episode_id] for episode_id in expected_episode_ids]


def summarize_counter(rows, key):
    return dict(sorted(Counter(row.get(key, "") for row in rows).items()))


def merge_shards(args):
    baseline, _evolution, split, _patches = load_round1_inputs(args.split_seed, args.failures, args.evolution)
    train_val = split["train"] + split["validation"]
    expected_ids = [item["episode_id"] for item in train_val]
    shard_payloads = [read_json(path) for path in sorted(Path(args.shard_dir).glob(args.shard_glob))]
    if not shard_payloads:
        raise ValueError(f"no shard files matched {args.shard_glob} in {args.shard_dir}")
    records = merge_shard_payloads(shard_payloads, expected_ids)

    repaired = [row for row in records if row["teacher"]["success"]]
    regression_sets = select_regression_sets(baseline["records"], split, args.regression_seed, args.r_success_size)
    assert_no_leakage(
        [row["episode_id"] for row in repaired],
        regression_sets["r_heldout"],
        regression_sets["r_success_eval"],
    )
    inventory = {
        "probe": "round1_inventory",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "train_val_failure_count": len(records),
        "repaired_count": len(repaired),
        "gate": "GO" if len(repaired) >= 50 else "STOP_LT_50",
        "repaired_episode_ids": [row["episode_id"] for row in repaired],
        "repaired_function_distribution": summarize_counter(repaired, "function_name"),
        "repaired_failure_type_distribution": summarize_counter(repaired, "failure_class"),
        "regression_sets": regression_sets,
    }
    partition_counts = Counter(row["partition"] for row in records)
    repaired_ids = {row["episode_id"] for row in repaired}
    sampling_repaired = [row for row in records if row["episode_id"] in repaired_ids and row["partition"] == "sampling_rescuable"]
    scaffold_only = [row for row in records if row["partition"] == "scaffold_only"]
    stats = {
        "total_failures": len(records),
        "sampling_rescuable": partition_counts["sampling_rescuable"],
        "scaffold_only": partition_counts["scaffold_only"],
        "neither": partition_counts["neither"],
        "repaired_count": len(repaired),
        "scaffold_only_repaired_rate": len(scaffold_only) / len(repaired) if repaired else 0.0,
        "sampling_rescuable_repaired_rate": len(sampling_repaired) / len(repaired) if repaired else 0.0,
    }
    partition = {
        "probe": "round1_pass16_partition",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "sample_count": shard_payloads[0].get("sample_count", args.sample_count),
        "temperature": shard_payloads[0].get("temperature", args.temperature),
        "stats": stats,
        "episodes": [
            {
                "episode_id": row["episode_id"],
                "failure_class": row["failure_class"],
                "function_name": row["function_name"],
                "teacher_success": row["teacher"]["success"],
                "pass16_hits": row["pass16"]["pass16_hits"],
                "partition": row["partition"],
            }
            for row in records
        ],
    }
    repaired_rows = [
        {
            "episode_id": row["episode_id"],
            "failure_class": row["failure_class"],
            "function_name": row["function_name"],
            "partition": row["partition"],
            "teacher_prediction": row["teacher"]["prediction"],
            "teacher_raw_model_output": row["teacher"]["raw_model_output"],
            "patch_ids": row["teacher"]["patch_ids"],
        }
        for row in repaired
    ]
    pass16_success_rows = []
    for row in records:
        for sample in row["pass16"]["successful_samples"]:
            pass16_success_rows.append(
                {
                    "episode_id": row["episode_id"],
                    "failure_class": row["failure_class"],
                    "function_name": row["function_name"],
                    "partition": row["partition"],
                    "sample_index": sample["sample_index"],
                    "prediction": sample["prediction"],
                    "raw_model_output": sample["raw_model_output"],
                }
            )

    write_json(args.inventory_output, inventory)
    write_json(args.partition_output, partition)
    write_json(args.repaired_output, repaired_rows)
    write_jsonl(args.pass16_success_output, pass16_success_rows)
    print(f"inventory repaired_count={inventory['repaired_count']} gate={inventory['gate']}")
    print(f"partition scaffold_only={stats['scaffold_only']} sampling_rescuable={stats['sampling_rescuable']}")


def build_datasets(args):
    baseline, _evolution, split, _patches = load_round1_inputs(args.split_seed, args.failures, args.evolution)
    train_ids = {item["episode_id"] for item in split["train"]}
    val_ids = {item["episode_id"] for item in split["validation"]}
    train_val_by_id = {item["episode_id"]: item for item in split["train"] + split["validation"]}
    repaired = read_json(args.repaired_input)
    partition = read_json(args.partition_input)
    partition_by_id = {row["episode_id"]: row["partition"] for row in partition["episodes"]}
    inventory = read_json(args.inventory_input)
    regression_sets = inventory["regression_sets"]
    heldout_ids = set(regression_sets["r_heldout"])
    r_success_eval_ids = set(regression_sets["r_success_eval"])

    teacher_rows = []
    for row in repaired:
        if args.train_only and row["episode_id"] not in train_ids:
            continue
        episode = train_val_by_id[row["episode_id"]]
        teacher_rows.append(
            make_distill_row(
                episode,
                baseline["prompt_template"],
                row["teacher_prediction"],
                "teacher_t0",
                row.get("partition", partition_by_id.get(row["episode_id"], "unknown")),
            )
        )
    for shard_path in sorted(Path(args.teacher_sample_dir).glob(args.teacher_sample_glob)):
        payload = read_json(shard_path)
        for sample_record in payload["records"]:
            if args.train_only and sample_record["episode_id"] not in train_ids:
                continue
            episode = train_val_by_id[sample_record["episode_id"]]
            for sample in sample_record["successful_samples"]:
                teacher_rows.append(
                    make_distill_row(
                        episode,
                        baseline["prompt_template"],
                        sample["prediction"],
                        "teacher_t08",
                        sample_record.get("partition", partition_by_id.get(sample_record["episode_id"], "unknown")),
                        sample.get("sample_index"),
                    )
                )
    teacher_rows = dedupe_rows(teacher_rows)
    star_rows = []
    for row in read_jsonl(args.pass16_success_input):
        if args.train_only and row["episode_id"] not in train_ids:
            continue
        episode = train_val_by_id[row["episode_id"]]
        star_rows.append(
            make_distill_row(
                episode,
                baseline["prompt_template"],
                row["prediction"],
                "star_pass16",
                row.get("partition", partition_by_id.get(row["episode_id"], "unknown")),
                row.get("sample_index"),
            )
        )
    # STaR uses every successful pass@16 trajectory. Keep duplicate outputs.

    max_main_core_for_balance = int(len(star_rows) / 0.9) if star_rows else len(teacher_rows)
    main_core_target = min(args.max_teacher_core, len(teacher_rows), max_main_core_for_balance)
    projected_total = round(main_core_target * (1.0 + args.replay_ratio))
    if projected_total < args.min_total and len(teacher_rows) > main_core_target and not star_rows:
        needed_core = math.ceil(args.min_total / (1.0 + args.replay_ratio))
        main_core_target = min(len(teacher_rows), args.max_teacher_core, needed_core)
    main_core = deterministic_sample(teacher_rows, main_core_target, args.dataset_seed, "main_core")
    star_core_target = min(len(star_rows), max(1, round(len(main_core) * args.star_core_ratio)))
    star_core = deterministic_sample(star_rows, star_core_target, args.dataset_seed, "star_core")

    assert_no_leakage([row["episode_id"] for row in main_core], heldout_ids, r_success_eval_ids)
    assert_no_leakage([row["episode_id"] for row in star_core], heldout_ids, r_success_eval_ids)
    assert_no_selection_leakage([row["episode_id"] for row in main_core], val_ids)
    assert_no_selection_leakage([row["episode_id"] for row in star_core], val_ids)

    replay_pool = [
        item
        for item in baseline["records"]
        if item.get("split") == "multiple"
        and item.get("call_success") is True
        and item["episode_id"] not in heldout_ids
        and item["episode_id"] not in r_success_eval_ids
    ]
    replay_rows = [
        make_distill_row(
            item,
            baseline["prompt_template"],
            item["predicted_call"],
            "replay_base_success",
            "replay",
        )
        for item in replay_pool
    ]
    replay_rows = dedupe_rows(replay_rows)
    main_replay_count = min(len(replay_rows), round(len(main_core) * args.replay_ratio))
    star_replay_count = min(len(replay_rows), round(len(star_core) * args.replay_ratio))
    main_replay = deterministic_sample(replay_rows, main_replay_count, args.dataset_seed, "main_replay")
    star_replay = deterministic_sample(replay_rows, star_replay_count, args.dataset_seed, "star_replay")

    distill_main = main_core + main_replay
    distill_star = star_core + star_replay
    assert_no_leakage([row["episode_id"] for row in distill_main], heldout_ids, r_success_eval_ids)
    assert_no_leakage([row["episode_id"] for row in distill_star], heldout_ids, r_success_eval_ids)
    assert_no_selection_leakage([row["episode_id"] for row in distill_main], val_ids)
    assert_no_selection_leakage([row["episode_id"] for row in distill_star], val_ids)

    write_jsonl(args.main_output, distill_main)
    write_jsonl(args.star_output, distill_star)
    summary = {
        "probe": "round1_datasets",
        "split_seed": args.split_seed,
        "dataset_seed": args.dataset_seed,
        "teacher_core_available": len(teacher_rows),
        "teacher_core_selected": len(main_core),
        "star_core_available": len(star_rows),
        "star_core_selected": len(star_core),
        "replay_pool_available": len(replay_rows),
        "replay_ratio_target": args.replay_ratio,
        "main_replay_count": len(main_replay),
        "star_replay_count": len(star_replay),
        "distill_main_count": len(distill_main),
        "distill_star_count": len(distill_star),
        "main_partition_distribution": summarize_counter(distill_main, "partition"),
        "star_partition_distribution": summarize_counter(distill_star, "partition"),
    }
    write_json(args.summary_output, summary)
    print(
        f"datasets main={summary['distill_main_count']} star={summary['distill_star_count']} "
        f"teacher_core={summary['teacher_core_selected']}/{summary['teacher_core_available']}"
    )


def prepare(args):
    baseline, evolution, split, patches = load_round1_inputs(args.split_seed, args.failures, args.evolution)
    regression_sets = select_regression_sets(baseline["records"], split, args.regression_seed, args.r_success_size)
    manifest = {
        "probe": "round1_manifest",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "split_sizes": {key: len(value) for key, value in split.items()},
        "nl_evo_accepted_patches": len(patches),
        "regression_sets": regression_sets,
    }
    write_json(args.output, manifest)
    import json as _json

    print(_json.dumps(manifest, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", default=str(BASE_FAILURES))
    parser.add_argument("--evolution", default=str(H1_EVOLUTION))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--regression-seed", type=int, default=DEFAULT_REGRESSION_SEED)
    parser.add_argument("--r-success-size", type=int, default=400)
    subparsers = parser.add_subparsers(dest="command", required=True)
    out = OUTPUT_DIR / "round1_build"

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", default=str(out / "manifest.json"))
    prepare_parser.set_defaults(func=prepare)

    shard_parser = subparsers.add_parser("run-shard")
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--shard-count", type=int, required=True)
    shard_parser.add_argument("--output", required=True)
    shard_parser.add_argument("--sample-count", type=int, default=16)
    shard_parser.add_argument("--sample-batch-size", type=int, default=4)
    shard_parser.add_argument("--temperature", type=float, default=0.8)
    shard_parser.add_argument("--max-new-tokens", type=int, default=256)
    shard_parser.add_argument("--progress-every", type=int, default=10)
    shard_parser.set_defaults(func=run_shard)

    teacher_parser = subparsers.add_parser("sample-teacher-shard")
    teacher_parser.add_argument("--repaired-input", default=str(out / "repaired_train_val.json"))
    teacher_parser.add_argument("--shard-index", type=int, required=True)
    teacher_parser.add_argument("--shard-count", type=int, required=True)
    teacher_parser.add_argument("--output", required=True)
    teacher_parser.add_argument("--sample-count", type=int, default=4)
    teacher_parser.add_argument("--sample-batch-size", type=int, default=4)
    teacher_parser.add_argument("--temperature", type=float, default=0.8)
    teacher_parser.add_argument("--max-new-tokens", type=int, default=256)
    teacher_parser.add_argument("--progress-every", type=int, default=10)
    teacher_parser.add_argument("--allow-val-sampling", action="store_false", dest="train_only")
    teacher_parser.set_defaults(train_only=True)
    teacher_parser.set_defaults(func=run_teacher_sample_shard)

    merge_parser = subparsers.add_parser("merge-shards")
    merge_parser.add_argument("--shard-dir", default=str(out / "shards"))
    merge_parser.add_argument("--shard-glob", default="shard_*.json")
    merge_parser.add_argument("--sample-count", type=int, default=16)
    merge_parser.add_argument("--temperature", type=float, default=0.8)
    merge_parser.add_argument("--inventory-output", default=str(out / "inventory.json"))
    merge_parser.add_argument("--partition-output", default=str(out / "pass16_partition.json"))
    merge_parser.add_argument("--repaired-output", default=str(out / "repaired_train_val.json"))
    merge_parser.add_argument("--pass16-success-output", default=str(out / "pass16_success_trajectories.jsonl"))
    merge_parser.set_defaults(func=merge_shards)

    dataset_parser = subparsers.add_parser("build-datasets")
    dataset_parser.add_argument("--repaired-input", default=str(out / "repaired_train_val.json"))
    dataset_parser.add_argument("--inventory-input", default=str(out / "inventory.json"))
    dataset_parser.add_argument("--partition-input", default=str(out / "pass16_partition.json"))
    dataset_parser.add_argument("--pass16-success-input", default=str(out / "pass16_success_trajectories.jsonl"))
    dataset_parser.add_argument("--teacher-sample-dir", default=str(out / "teacher_samples"))
    dataset_parser.add_argument("--teacher-sample-glob", default="teacher_shard_*.json")
    dataset_parser.add_argument("--main-output", default=str(out / "distill_main.jsonl"))
    dataset_parser.add_argument("--star-output", default=str(out / "distill_star.jsonl"))
    dataset_parser.add_argument("--summary-output", default=str(out / "datasets_summary.json"))
    dataset_parser.add_argument("--dataset-seed", type=int, default=20260702)
    dataset_parser.add_argument("--replay-ratio", type=float, default=0.4)
    dataset_parser.add_argument("--star-core-ratio", type=float, default=1.0)
    dataset_parser.add_argument("--min-total", type=int, default=300)
    dataset_parser.add_argument("--max-teacher-core", type=int, default=500)
    dataset_parser.add_argument("--allow-val-training", action="store_false", dest="train_only")
    dataset_parser.set_defaults(train_only=True)
    dataset_parser.set_defaults(func=build_datasets)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
