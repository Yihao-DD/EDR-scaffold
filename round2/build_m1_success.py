#!/usr/bin/env python3
"""Build the round-2 replay source pool: M1 T=0 successes (V=1).

Pool spec (locked): ``M1 T=0 @ {capped-120 ∪ eliminated}`` keeping only V=1.

- ``capped-120``  : the 120 unique replay episodes of the round-1 rep2 replay pool
  (``repro_rep2/data/episode_ids/capped120_replay.json``).
- ``eliminated``  : train-share failures that M1 now repairs, taken from the
  ``eliminated_episode_ids`` field written by ``collect_failures.py``.

M1 = M0 + A1 is merged in memory (base_adapter_dir); raw M0 is never used. Every
pool episode is re-scored with the deterministic AST verifier and only V=1 rows
are written. The number of V=0 exclusions is reported, and the pool is asserted
disjoint from every evaluation surface. Output rows carry ``input``/``output`` so
``build_replay2.py`` can stamp the replay marker and repeat them to the 2:1 ratio.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from common import add_common_args, assert_disjoint, load_episode_id_file, read_json, write_json, write_jsonl


def _load_pool(capped_ids_path: Path, collect_json_path: Path | None) -> "OrderedDict[str, str]":
    """Return ordered {episode_id: origin} for capped-120 ∪ eliminated."""
    pool: "OrderedDict[str, str]" = OrderedDict()
    for episode_id in sorted(load_episode_id_file(capped_ids_path)):
        pool[str(episode_id)] = "capped120"
    if collect_json_path is not None:
        eliminated = read_json(collect_json_path).get("eliminated_episode_ids") or []
        for episode_id in eliminated:
            key = str(episode_id)
            pool[key] = "both" if key in pool else "eliminated"
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--m1-adapter", default=None, help="M1 adapter dir; if omitted, resolved from --manifest. Raw M0 is refused.")
    parser.add_argument("--manifest", default="repro_rep2/MANIFEST.json", help="Used to resolve M1 when --m1-adapter is omitted.")
    parser.add_argument("--capped-ids", default=None, help="capped-120 episode-id list (default: <data_root>/episode_ids/capped120_replay.json).")
    parser.add_argument("--collect-json", default=None, help="collect_failures.py output; supplies eliminated_episode_ids.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--output", default="round2_outputs/m1_train_success.jsonl")
    parser.add_argument("--summary-output", default="round2_outputs/m1_train_success_summary.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    capped_ids_path = Path(args.capped_ids) if args.capped_ids else data_root / "episode_ids" / "capped120_replay.json"
    collect_json_path = Path(args.collect_json) if args.collect_json else None
    pool = _load_pool(capped_ids_path, collect_json_path)
    pool_ids = list(pool.keys())
    if args.limit is not None:
        pool_ids = pool_ids[: args.limit]

    # Leakage asserts run before any inference; the pool must never touch an eval surface.
    assertions = []
    for name, rel in [
        ("heldout", "D_heldout_failures.json"),
        ("d_val", "D_val_failures.json"),
        ("old400", "old400_success_eval.json"),
        ("sibling300", "sibling300_arena.json"),
    ]:
        assertions.append(assert_disjoint(f"m1_success_intersect_{name}", pool_ids, load_episode_id_file(data_root / "episode_ids" / rel)))

    if args.dry_run:
        write_jsonl(args.output, [{"episode_id": episode_id, "pool_origin": pool[episode_id], "dry_run_status": "not_inferred"} for episode_id in pool_ids])
        write_json(
            args.summary_output,
            {
                "probe": "round2_build_m1_success",
                "dry_run": True,
                "pool_size": len(pool_ids),
                "capped120_from": str(capped_ids_path),
                "collect_json": str(collect_json_path) if collect_json_path else None,
                "assertions": assertions,
            },
        )
        print(f"DRY_RUN wrote {args.output} pool={len(pool_ids)} (no inference)")
        return

    # Resolve M1 and refuse raw M0: the replay pool must be M1 T=0 successes.
    m1_adapter = args.m1_adapter
    if not m1_adapter:
        m1 = read_json(args.manifest).get("m1_designated", {})
        adapter_file = (m1.get("adapter") or {}).get("path") or m1.get("adapter_path")
        m1_adapter = str(Path(adapter_file).parent) if adapter_file else None
    if not m1_adapter:
        raise SystemExit("M1 adapter is required (raw M0 pool is invalid); pass --m1-adapter or a manifest with m1_designated.")

    sys.path.insert(0, str(Path(args.repro_root)))
    from scripts.lora_phase0 import evaluate_episode, load_model_for_eval, load_tokenizer
    from scripts.phase0_probe import build_base_prompt, load_exp2

    baseline = read_json(args.failures)
    prompt_template = baseline["prompt_template"]
    baseline_by_id = {str(row["episode_id"]): row for row in baseline["records"]}
    missing = [episode_id for episode_id in pool_ids if episode_id not in baseline_by_id]
    if missing:
        raise AssertionError({"assert": "m1_success_pool_in_baseline", "missing_count": len(missing), "examples": missing[:20]})

    evo = load_exp2(args.exp2_root)
    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(args.model_id, adapter_dir=None, base_adapter_dir=m1_adapter, qlora=not args.no_qlora)

    kept: list[dict] = []
    excluded: list[str] = []
    for index, episode_id in enumerate(pool_ids, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(pool_ids):
            print(f"[m1-success] {index}/{len(pool_ids)} {episode_id}", flush=True)
        episode = baseline_by_id[episode_id]
        result = evaluate_episode(evo, model, tokenizer, episode, prompt_template, args.max_new_tokens)
        if result["success"]:
            kept.append(
                {
                    "episode_id": episode_id,
                    "input": build_base_prompt(episode, prompt_template),
                    "output": result["raw_model_output"],
                    "function_name": episode.get("ground_truth_call", {}).get("name"),
                    "failure_class": episode.get("failure_class", "unknown"),
                    "source": "m1_train_success",
                    "pool_origin": pool[episode_id],
                }
            )
        else:
            excluded.append(episode_id)

    write_jsonl(args.output, kept)
    write_json(
        args.summary_output,
        {
            "probe": "round2_build_m1_success",
            "dry_run": False,
            "model_stack": ["M0", "A1_merged_in_memory"],
            "pool_size": len(pool_ids),
            "capped120_from": str(capped_ids_path),
            "collect_json": str(collect_json_path) if collect_json_path else None,
            "kept_v1": len(kept),
            "excluded_v0": len(excluded),
            "excluded_episode_ids": excluded,
            "assertions": assertions,
        },
    )
    print(f"wrote {args.output} kept_v1={len(kept)} excluded_v0={len(excluded)} pool={len(pool_ids)}")


if __name__ == "__main__":
    main()
