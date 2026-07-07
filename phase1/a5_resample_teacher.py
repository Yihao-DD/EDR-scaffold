#!/usr/bin/env python3
"""A5 step 1: re-sample the H1 teacher keeping ALL outputs (verified and not).

Why this exists: the Phase 0 teacher shards only stored `successful_samples`
(AST-verified). A5 needs the unfiltered stream, so the teacher forward is
re-run with the same prompts, same patch-injection path, and the same
deterministic per-episode sampling seeds as Phase 0 (stable_int(split_seed,
episode_id, "teacher", offset)), but every sample is retained with its
verifier flag recorded for audit.

Episode pool (config phase1.a5.mode):
- no_verifier_full (default): every train-share failure episode. This is the
  honest no-verifier counterfactual: without a verifier you cannot even know
  which episodes were repaired.
- repaired_episodes_only: the 74 rep2 core episodes (narrow variant).

GPU required unless --dry-run. Shardable via --shard-index/--shard-count.
"""

from __future__ import annotations

import argparse
import time

from phase1.common import (
    add_common_args,
    load_config,
    load_inputs,
    load_probe_module,
    read_jsonl,
    resolve_model_id,
    write_json,
    REPRO_ROOT,
)
from pathlib import Path


def episode_pool(mode, split, core_jsonl):
    train = split["train"]
    if mode == "no_verifier_full":
        return train
    if mode == "repaired_episodes_only":
        core_ids = {row["episode_id"] for row in read_jsonl(core_jsonl)}
        return [item for item in train if item["episode_id"] in core_ids]
    raise ValueError(f"unknown a5 mode: {mode}")


def sample_episode_keep_all(evo, probe, runner, episode, patches, sample_count, temperature, max_new_tokens, split_seed):
    """T=0 x1 plus T=temperature x sample_count, retaining every sample."""

    relevant = evo.relevant_patches_for_episode(patches, episode)
    prompt = evo.build_prompt(episode, relevant)
    samples = []
    started = time.time()

    def record(raw, sample_index, kind):
        prediction, parse_error = probe.parse_prediction(evo, raw)
        success = probe.success_for_prediction(evo, prediction, episode)
        samples.append(
            {
                "sample_index": sample_index,
                "kind": kind,
                "success": success,
                "prediction": prediction,
                "raw_model_output": raw,
                "parse_error": parse_error,
            }
        )

    greedy = runner.generate_many(prompt, n=1, do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens)[0]
    record(greedy, -1, "t0")
    offset = 0
    remaining = sample_count
    while remaining:
        batch = min(4, remaining)
        seed = probe.stable_int(split_seed, episode["episode_id"], "teacher", offset)
        raws = runner.generate_many(
            prompt,
            n=batch,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        for raw in raws:
            record(raw, offset, "t_sampled")
            offset += 1
        remaining -= batch
    return {
        "episode_id": episode["episode_id"],
        "failure_class": episode.get("failure_class", "unknown"),
        "function_name": episode.get("ground_truth_call", {}).get("name", ""),
        "patch_ids": [patch.patch_id for patch in relevant],
        "n_patches": len(relevant),
        "sample_count": sample_count,
        "temperature": temperature,
        "latency_ms": int((time.time() - started) * 1000),
        "samples": samples,
        "ast_pass_count": sum(1 for s in samples if s["success"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    a5 = config["phase1"]["a5"]
    split_seed = config["split_seed"]
    baseline, exp2_result, split, patches, evo = load_inputs(split_seed)
    probe = load_probe_module()

    pool = episode_pool(a5["mode"], split, REPRO_ROOT / "data" / "distill_main_core.jsonl")
    shard = probe.shard_items(pool, args.shard_index, args.shard_count)
    if args.limit:
        shard = shard[: args.limit]

    output = args.output or (
        Path(args.output_dir) / "a5" / f"a5_teacher_all_shard_{args.shard_index}.json"
    )

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

    runner = probe.Phase0Runner(resolve_model_id(config), cache_dir=config.get("model_cache_dir"))
    records = []
    for index, episode in enumerate(shard, start=1):
        if index == 1 or index % 10 == 0 or index == len(shard):
            print(f"[a5-resample {args.shard_index}/{args.shard_count}] {index}/{len(shard)} {episode['episode_id']}", flush=True)
        records.append(
            sample_episode_keep_all(
                evo,
                probe,
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


if __name__ == "__main__":
    main()
