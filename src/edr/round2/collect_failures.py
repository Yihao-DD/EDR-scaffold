"""Collect F2: failures of M1 (= M0 + A1, merged) on the round-2 train share.

CLI:
  python -m edr.round2.collect_failures [--m1-adapter <dir>] [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
from collections import Counter

from edr.config import load_config, resolve_model_id
from edr.io_utils import read_json, write_json
from edr.paths import BASE_FAILURES, ROUND2_OUT, resolve
from edr.round2.shared import load_train_share, resolve_m1_adapter


def summarize(rows, key):
    return dict(sorted(Counter(row.get(key, "unknown") for row in rows).items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--failures", default=str(BASE_FAILURES))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=str(ROUND2_OUT / "f2_failures.json"))
    args = parser.parse_args(argv)

    config = load_config("round2")
    m1_adapter = resolve_m1_adapter(args.m1_adapter)
    train_share = load_train_share()
    if args.limit is not None:
        train_share = train_share[: args.limit]

    baseline = None
    eliminated = []
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
        from edr.evaluation.heldout import evaluate_episode
        from edr.modeling import load_model_for_eval, load_tokenizer

        baseline = read_json(resolve(args.failures))
        baseline_by_id = {row["episode_id"]: row for row in baseline["records"]}
        missing = [row["episode_id"] for row in train_share if row["episode_id"] not in baseline_by_id]
        if missing:
            raise AssertionError({"missing_train_share_in_baseline": missing[:20], "count": len(missing)})

        tokenizer = load_tokenizer(resolve_model_id(config))
        model = load_model_for_eval(
            resolve_model_id(config),
            adapter_dir=None,
            base_adapter_dir=m1_adapter,
            qlora=not args.no_qlora,
        )
        failures = []
        for index, row in enumerate(train_share, start=1):
            if index == 1 or index % args.progress_every == 0 or index == len(train_share):
                print(f"[collect-f2] {index}/{len(train_share)} {row['episode_id']}", flush=True)
            baseline_row = baseline_by_id[row["episode_id"]]
            result = evaluate_episode(model, tokenizer, baseline_row, baseline["prompt_template"], args.max_new_tokens)
            merged = {**baseline_row, **row, **result}
            if result["success"]:
                eliminated.append(merged)
            else:
                failures.append(merged)

    payload = {
        "probe": "round2_collect_failures",
        "m1_adapter": str(m1_adapter),
        "model_stack": ["M0", "A1_merged_in_memory"],
        "prompt_template": None if args.dry_run else baseline.get("prompt_template"),
        "train_share_checked": len(train_share),
        "f2_count": len(failures),
        "internalized_count": len(train_share) - len(failures),
        "eliminated_episode_ids": [] if args.dry_run else [row["episode_id"] for row in eliminated],
        "f1_by_partition": summarize(train_share, "partition"),
        "f2_by_partition": summarize(failures, "partition"),
        "internalized_by_partition": summarize(eliminated if not args.dry_run else [], "partition"),
        "f2_by_failure_class": summarize(failures, "failure_class"),
        "f2": failures,
        "failures": [dict(row, split="multiple") for row in failures],
        "note": "Dry run enumerates the train share; full mode records M1 T=0 AST failures.",
    }
    write_json(resolve(args.output), payload)
    print(f"wrote {args.output} f2_count={len(failures)}")


if __name__ == "__main__":
    main()
