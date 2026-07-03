#!/usr/bin/env python3
"""Collect F2: failures of M0+A1 on the round-2 train share."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from common import add_common_args, read_json, slice_limit, write_json


def load_train_share(data_root: Path) -> list[dict]:
    pass16 = read_json(data_root / "train_val_pass16_partition.json")
    val_ids = set(read_json(data_root / "episode_ids" / "D_val_failures.json")["episode_ids"])
    # D_val is excluded from train. Train share is the complement inside train+val failures.
    return [row for row in pass16["episodes"] if row["episode_id"] not in val_ids]


def summarize(rows: list[dict], key: str) -> dict:
    return dict(sorted(Counter(row.get(key, "unknown") for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--m1-adapter", required=True, help="Path to designated M1 adapter.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--output", default="round2_outputs/f2_failures.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_share = slice_limit(load_train_share(data_root), args.limit)
    if args.dry_run:
        failures = [
            {
                "episode_id": row["episode_id"],
                "function_name": row.get("function_name"),
                "failure_class": row.get("failure_class"),
                "dry_run_status": "not_inferred",
            }
            for row in train_share
        ]
    else:
        sys.path.insert(0, str(Path(args.repro_root)))
        from scripts.lora_phase0 import evaluate_episode, load_model_for_eval, load_tokenizer
        from scripts.phase0_probe import load_exp2

        baseline = read_json(args.failures)
        baseline_by_id = {row["episode_id"]: row for row in baseline["records"]}
        missing = [row["episode_id"] for row in train_share if row["episode_id"] not in baseline_by_id]
        if missing:
            raise AssertionError({"missing_train_share_in_baseline": missing[:20], "count": len(missing)})

        evo = load_exp2(args.exp2_root)
        tokenizer = load_tokenizer(args.model_id)
        model = load_model_for_eval(
            args.model_id,
            adapter_dir=None,
            base_adapter_dir=args.m1_adapter,
            qlora=not args.no_qlora,
        )
        failures = []
        eliminated = []
        for index, row in enumerate(train_share, start=1):
            if index == 1 or index % args.progress_every == 0 or index == len(train_share):
                print(f"[collect-f2] {index}/{len(train_share)} {row['episode_id']}", flush=True)
            baseline_row = baseline_by_id[row["episode_id"]]
            result = evaluate_episode(
                evo,
                model,
                tokenizer,
                baseline_row,
                baseline["prompt_template"],
                args.max_new_tokens,
            )
            merged = {**baseline_row, **row, **result}
            if result["success"]:
                eliminated.append(merged)
            else:
                failures.append(merged)

    payload = {
        "probe": "round2_collect_failures",
        "m1_adapter": args.m1_adapter,
        "model_stack": ["M0", "A1_merged_in_memory"],
        "prompt_template": None if args.dry_run else baseline.get("prompt_template"),
        "train_share_checked": len(train_share),
        "f2_count": len(failures),
        "internalized_count": len(train_share) - len(failures),
        "f1_by_partition": summarize(train_share, "partition"),
        "f2_by_partition": summarize(failures, "partition"),
        "internalized_by_partition": summarize(eliminated if not args.dry_run else [], "partition"),
        "f2_by_failure_class": summarize(failures, "failure_class"),
        "f2": failures,
        "failures": [dict(row, split="multiple") for row in failures],
        "note": "Dry run enumerates the train share; full mode records M0+A1 T=0 AST failures.",
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} f2_count={len(failures)}")


if __name__ == "__main__":
    main()
