"""Round-2 replay source pool: M1 T=0 successes over {capped-120 ∪ eliminated}.

Only V=1 rows are kept; V=0 exclusions are counted. Leakage asserts run
BEFORE any inference. Raw M0 is refused.

CLI:
  python -m edr.round2.build_m1_success --collect-json <f2 json> [--m1-adapter <dir>]
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

from edr.config import load_config, resolve_model_id
from edr.data.leakage import assert_disjoint, load_episode_id_file, load_eval_surface_ids
from edr.data.splits import build_base_prompt
from edr.io_utils import read_json, write_json, write_jsonl
from edr.paths import BASE_FAILURES, EPISODE_ID_DIR, ROUND2_OUT, resolve
from edr.round2.shared import resolve_m1_adapter


def _load_pool(capped_ids_path, collect_json_path):
    """Ordered {episode_id: origin} for capped-120 ∪ eliminated."""

    pool = OrderedDict()
    for episode_id in sorted(load_episode_id_file(capped_ids_path)):
        pool[str(episode_id)] = "capped120"
    if collect_json_path is not None:
        eliminated = read_json(collect_json_path).get("eliminated_episode_ids") or []
        for episode_id in eliminated:
            key = str(episode_id)
            pool[key] = "both" if key in pool else "eliminated"
    return pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--capped-ids", default=str(EPISODE_ID_DIR / "capped120_replay.json"))
    parser.add_argument("--collect-json", default=None)
    parser.add_argument("--failures", default=str(BASE_FAILURES))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=str(ROUND2_OUT / "m1_train_success.jsonl"))
    parser.add_argument("--summary-output", default=str(ROUND2_OUT / "m1_train_success_summary.json"))
    args = parser.parse_args(argv)

    config = load_config("round2")
    capped_ids_path = resolve(args.capped_ids)
    collect_json_path = resolve(args.collect_json) if args.collect_json else None
    pool = _load_pool(capped_ids_path, collect_json_path)
    pool_ids = list(pool.keys())
    if args.limit is not None:
        pool_ids = pool_ids[: args.limit]

    assertions = []
    for name, surface in load_eval_surface_ids().items():
        assertions.append(assert_disjoint(f"m1_success_intersect_{name}", pool_ids, surface))

    if args.dry_run:
        write_jsonl(
            resolve(args.output),
            [{"episode_id": episode_id, "pool_origin": pool[episode_id], "dry_run_status": "not_inferred"} for episode_id in pool_ids],
        )
        write_json(
            resolve(args.summary_output),
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

    m1_adapter = resolve_m1_adapter(args.m1_adapter)

    from edr.evaluation.heldout import evaluate_episode
    from edr.modeling import load_model_for_eval, load_tokenizer

    baseline = read_json(resolve(args.failures))
    prompt_template = baseline["prompt_template"]
    baseline_by_id = {str(row["episode_id"]): row for row in baseline["records"]}
    missing = [episode_id for episode_id in pool_ids if episode_id not in baseline_by_id]
    if missing:
        raise AssertionError({"assert": "m1_success_pool_in_baseline", "missing_count": len(missing), "examples": missing[:20]})

    tokenizer = load_tokenizer(resolve_model_id(config))
    model = load_model_for_eval(resolve_model_id(config), adapter_dir=None, base_adapter_dir=m1_adapter, qlora=not args.no_qlora)

    kept = []
    excluded = []
    for index, episode_id in enumerate(pool_ids, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(pool_ids):
            print(f"[m1-success] {index}/{len(pool_ids)} {episode_id}", flush=True)
        episode = baseline_by_id[episode_id]
        result = evaluate_episode(model, tokenizer, episode, prompt_template, args.max_new_tokens)
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

    write_jsonl(resolve(args.output), kept)
    write_json(
        resolve(args.summary_output),
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
