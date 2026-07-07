"""Ablation/control-arm dataset builders (A5 / A7 / A8).

All arms reuse the frozen round-1 rows verbatim wherever possible, so the
isolated variable is exactly the one the arm names. Recipes and balance
procedures are preregistered (see docs/experiments.md):

- A5 unverified: teacher re-sampled with the AST filter OFF; core = 148 rows
  by EPISODE-LEVEL UNIFORM downsampling (T=0 row first per episode) over the
  full train-share pool; replay = the frozen 296-row block, verbatim.
- A7 replay-ablation: the 148-row core with ZERO replay (the KL anchor rides
  on replay rows, so A7 is by definition "recipe minus replay minus KL").
- A8 data-scale: episode-level 25%/50% subsets of the core, replay scaled to
  keep the 2:1 replay:core ROW ratio.

Subcommands:
  resample-teacher   GPU: teacher forward keeping ALL samples (for A5)
  build-a5           CPU: unverified dataset from the resample shards
  build-a7a8         CPU: no-replay + scale-subset datasets
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from edr.config import load_config, resolve_model_id
from edr.data.distill import dedupe_rows, deterministic_sample, make_distill_row, replay_rows_from_train
from edr.data.leakage import assert_training_disjoint
from edr.data.splits import load_round1_inputs
from edr.io_utils import read_json, write_json, write_jsonl
from edr.paths import ABLATION_OUT, DISTILL_CORE, resolve


# ---------------------------------------------------------------------------
# A5 step 1: teacher resample keeping every sample
# ---------------------------------------------------------------------------

def episode_pool(mode, split):
    train = split["train"]
    if mode == "no_verifier_full":
        return train
    if mode == "repaired_episodes_only":
        from edr.io_utils import read_jsonl

        core_ids = {row["episode_id"] for row in read_jsonl(DISTILL_CORE)}
        return [item for item in train if item["episode_id"] in core_ids]
    raise ValueError(f"unknown a5 mode: {mode}")


def resample_teacher(args, config):
    a5 = config["a5"]
    split_seed = config["split_seed"]
    baseline, _evolution, split, patches = load_round1_inputs(split_seed)

    pool = episode_pool(a5["mode"], split)
    shard = [item for index, item in enumerate(pool) if index % args.shard_count == args.shard_index]
    if args.limit:
        shard = shard[: args.limit]

    output = resolve(args.output) if args.output else (Path(args.output_dir) / "a5" / f"a5_teacher_all_shard_{args.shard_index}.json")

    if args.dry_run:
        write_json(
            output,
            {
                "probe": "a5_resample_teacher_dry_run",
                "mode": a5["mode"],
                "pool_episodes": len(pool),
                "shard_episodes": len(shard),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "sample_count": a5["sample_count"],
                "temperature": a5["temperature"],
                "episode_ids_head": [item["episode_id"] for item in shard[:10]],
            },
        )
        print(f"[a5-resample:dry-run] pool={len(pool)} shard={len(shard)} -> {output}")
        return

    from edr.modeling import GenerationRunner
    from edr.scaffold.teacher import sample_episode_keep_all

    runner = GenerationRunner(resolve_model_id(config), cache_dir=config.get("model_cache_dir"))
    records = []
    for index, episode in enumerate(shard, start=1):
        if index == 1 or index % 10 == 0 or index == len(shard):
            print(f"[a5-resample {args.shard_index}/{args.shard_count}] {index}/{len(shard)} {episode['episode_id']}", flush=True)
        records.append(
            sample_episode_keep_all(
                runner,
                episode,
                patches,
                a5["sample_count"],
                a5["temperature"],
                config["eval"]["max_new_tokens"],
                split_seed,
            )
        )
    write_json(
        output,
        {
            "probe": "a5_teacher_all_samples",
            "model_id": resolve_model_id(config),
            "split_seed": split_seed,
            "mode": a5["mode"],
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "sample_count": a5["sample_count"],
            "temperature": a5["temperature"],
            "records": records,
        },
    )
    total = sum(len(row["samples"]) for row in records)
    passed = sum(row["ast_pass_count"] for row in records)
    print(f"[a5-resample] episodes={len(records)} samples={total} ast_pass={passed} -> {output}")


# ---------------------------------------------------------------------------
# A5 step 2: unverified dataset (AST filter OFF, episode-uniform balance)
# ---------------------------------------------------------------------------

def select_core_episode_uniform(rows, budget, seed):
    """Hash-order episodes; each contributes its T=0 row first; deeper
    (sampled) rows fill only after every episode contributed one."""

    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_id"], []).append(row)
    for items in by_episode.values():
        items.sort(
            key=lambda row: (
                0 if row["source"].endswith("_t0") else 1,
                row.get("sample_index") if row.get("sample_index") is not None else 1 << 30,
            )
        )
    order = sorted(
        by_episode,
        key=lambda episode_id: hashlib.sha256(f"{seed}:a5_core:{episode_id}".encode("utf-8")).hexdigest(),
    )
    core = []
    depth = 0
    while len(core) < budget:
        added = False
        for episode_id in order:
            items = by_episode[episode_id]
            if depth < len(items):
                core.append(items[depth])
                added = True
                if len(core) >= budget:
                    break
        if not added:
            break
        depth += 1
    return core


def collect_unfiltered_rows(shard_paths, split, baseline):
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
                row = make_distill_row(
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


def build_a5(args, config):
    a5 = config["a5"]
    baseline, _evolution, split, _patches = load_round1_inputs(config["split_seed"])

    output_dir = Path(args.output_dir)
    shard_paths = sorted(output_dir.glob("a5/a5_teacher_all_shard_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no A5 teacher shards under {output_dir / 'a5'}")

    rows, funnel = collect_unfiltered_rows(shard_paths, split, baseline)
    rows = dedupe_rows(rows)
    core = select_core_episode_uniform(rows, a5["core_rows"], config["dataset_seed"])
    if len(core) < a5["core_rows"]:
        print(f"[a5-build:warn] only {len(core)} unique rows available for a {a5['core_rows']}-row budget")

    _, replay = replay_rows_from_train()
    dataset = core + replay
    leakage = assert_training_disjoint(dataset, "a5")

    dataset_output = output_dir / "a5" / "distill_a5_unverified.jsonl"
    summary_output = output_dir / "a5" / "a5_dataset_summary.json"

    core_episodes = {row["episode_id"] for row in core}
    summary = {
        "probe": "a5_unverified_dataset",
        "mode": a5["mode"],
        "balance": "episode-level uniform, T=0 first; rows matched to the round-1 core 148 +/-10%",
        "unverified_meaning": "not correctness-verified (AST filter off); parsing is a mechanical stage, unparseable outputs counted below and never trained on",
        "funnel": funnel,
        "unfiltered_rows_available": len(rows),
        "core_rows": len(core),
        "core_unique_episodes": len(core_episodes),
        "core_t0_rows": sum(1 for row in core if row["source"].endswith("_t0")),
        "core_ast_pass_rows": sum(1 for row in core if row.get("ast_pass")),
        "core_ast_fail_rows": sum(1 for row in core if not row.get("ast_pass")),
        "replay_rows": len(replay),
        "total_rows": len(dataset),
        "row_matched_to": "data/round1/distill_train.jsonl (444 rows)",
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


# ---------------------------------------------------------------------------
# A7 / A8
# ---------------------------------------------------------------------------

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


def build_a7a8(args, config):
    seed = config["dataset_seed"]
    core, replay = replay_rows_from_train()
    output_dir = Path(args.output_dir)

    summaries = {}

    a7_rows = list(core)
    leakage = assert_training_disjoint(a7_rows, "a7")
    a7_path = output_dir / "a7" / "distill_a7_noreplay.jsonl"
    summaries["a7"] = {
        "probe": "a7_noreplay_dataset",
        "core_rows": len(a7_rows),
        "replay_rows": 0,
        "total_rows": len(a7_rows),
        "kl_anchor": "inert (no replay rows) — the runner passes kl_anchor_lambda=0; arm is recipe minus replay minus KL",
        "leakage_asserts": leakage,
        "dataset_output": str(a7_path),
    }
    if not args.dry_run:
        write_jsonl(a7_path, a7_rows)

    for fraction in config["a8"]["fractions"]:
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
    write_json(summary_path, {"probe": "a7_a8_datasets", "dataset_seed": seed, "dry_run": bool(args.dry_run), "datasets": summaries})
    for name, item in summaries.items():
        print(f"[a7a8-build] {name}: total={item['total_rows']} core={item['core_rows']} replay={item['replay_rows']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resample_parser = subparsers.add_parser("resample-teacher")
    resample_parser.add_argument("--shard-index", type=int, default=0)
    resample_parser.add_argument("--shard-count", type=int, default=2)
    resample_parser.add_argument("--output", default=None)

    a5_parser = subparsers.add_parser("build-a5")
    a7a8_parser = subparsers.add_parser("build-a7a8")

    for sub in (resample_parser, a5_parser, a7a8_parser):
        sub.add_argument("--output-dir", default=str(ABLATION_OUT))
        sub.add_argument("--limit", type=int, default=None)
        sub.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    config = load_config("ablations")
    if args.command == "resample-teacher":
        resample_teacher(args, config)
    elif args.command == "build-a5":
        build_a5(args, config)
    else:
        build_a7a8(args, config)


if __name__ == "__main__":
    main()
