"""Self-evolution loop: propose patches from failures, accept via AST-verified
validation gains, never regress (accept iff fixed>0 and regressed==0).

Ported verbatim from the frozen loop. Round-2 invariant: patch search must
run on merged M1 — the CLI refuses to start without an adapter (raw-M0 patch
search is invalid for the second iteration).

CLI:
  python -m edr.scaffold.loop --base-adapter-dir <M1> \
      [--input data/round1/base_failures.json] [--seeds 20260630] \
      --output outputs/round2/loop/evolution_loop.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from edr.io_utils import read_json, write_json
from edr.modeling import ModelRunner
from edr.paths import BASE_FAILURES, ROUND2_OUT, resolve
from edr.scaffold.patches import (
    FAMILIES,
    materialize_candidate,
    patch_relevant_to_episode,
    relevant_patches_for_episode,
    build_prompt,
)
from edr.verifier import call_success, parse_model_output
from edr.data.splits import deterministic_split

METHODS = ["no-evolution", "NL-evo", "NL-padded-evo", "STRUCT-evo", "oracle"]


def evaluate_episode(episode, patches, runner, max_new_tokens=256):
    prompt = build_prompt(episode, patches)
    started = time.time()
    raw = runner.generate(prompt, max_new_tokens=max_new_tokens)
    latency_ms = int((time.time() - started) * 1000)
    parse_error = None
    try:
        prediction = parse_model_output(raw)
    except Exception as error:
        prediction = None
        parse_error = str(error)
    success = call_success(prediction or {"name": "", "arguments": {}}, episode["ground_truth_call"])
    return {
        "episode_id": episode["episode_id"],
        "success": success,
        "prediction": prediction,
        "raw_model_output": raw,
        "parse_error": parse_error,
        "latency_ms": latency_ms,
        "n_patches": len(patches),
        "patch_ids": [patch.patch_id for patch in patches],
    }


def update_validation_state(current, proposed):
    fixed = sum(1 for key, value in proposed.items() if value and not current.get(key, False))
    regressed = sum(1 for key, value in proposed.items() if current.get(key, False) and not value)
    accepted = fixed > 0 and regressed == 0
    stats = {"fixed": fixed, "regressed": regressed, "current_pass": sum(current.values()), "proposed_pass": sum(proposed.values())}
    return accepted, (dict(proposed) if accepted else dict(current)), stats


def _validation_scope(candidate, validation):
    return [episode for episode in validation if patch_relevant_to_episode(candidate, episode)]


def run_family_loop(family, split, runner, max_new_tokens=256):
    accepted_patches = []
    validation_state = {episode["episode_id"]: False for episode in split["validation"]}
    accepted_log = []
    rejected_log = []
    candidates = [materialize_candidate(failure, family, tokenizer=runner.tokenizer) for failure in split["train"]]
    print(f"[loop:{family}] candidates={len(candidates)} validation={len(split['validation'])}", flush=True)
    for index, candidate in enumerate(candidates, start=1):
        if index == 1 or index % 25 == 0 or index == len(candidates):
            print(
                f"[loop:{family}] candidate {index}/{len(candidates)} accepted={len(accepted_patches)}",
                flush=True,
            )
        scope = _validation_scope(candidate, split["validation"])
        if not scope:
            rejected_log.append({"patch": candidate.to_json_dict(), "reason": "no_validation_scope"})
            continue
        current_subset = {episode["episode_id"]: validation_state.get(episode["episode_id"], False) for episode in scope}
        proposed_subset = {}
        for episode in scope:
            relevant = relevant_patches_for_episode(accepted_patches + [candidate], episode)
            proposed_subset[episode["episode_id"]] = evaluate_episode(episode, relevant, runner, max_new_tokens=max_new_tokens)["success"]
        accepted, updated_subset, stats = update_validation_state(current_subset, proposed_subset)
        log_item = {
            "patch": candidate.to_json_dict(),
            "validation_scope_size": len(scope),
            "validation_result": stats,
        }
        if accepted:
            accepted_patches.append(candidate)
            validation_state.update(updated_subset)
            accepted_log.append(log_item)
        else:
            rejected_log.append(log_item)
    return {
        "family": family,
        "accepted_patches": accepted_patches,
        "accepted_log": accepted_log,
        "rejected_log": rejected_log,
        "validation_pass_count": sum(validation_state.values()),
        "validation_size": len(split["validation"]),
    }


def evaluate_held_out(method, held_out, patches, runner, max_new_tokens=256):
    records = []
    if method == "no-evolution":
        for episode in held_out:
            records.append(
                {
                    "episode_id": episode["episode_id"],
                    "success": False,
                    "failure_class": episode["failure_class"],
                    "n_patches": 0,
                    "patch_ids": [],
                }
            )
        return records
    print(f"[heldout:{method}] episodes={len(held_out)} patches={len(patches)}", flush=True)
    for index, episode in enumerate(held_out, start=1):
        if index == 1 or index % 25 == 0 or index == len(held_out):
            print(f"[heldout:{method}] episode {index}/{len(held_out)}", flush=True)
        relevant = relevant_patches_for_episode(patches, episode)
        if not relevant:
            records.append(
                {
                    "episode_id": episode["episode_id"],
                    "success": False,
                    "failure_class": episode["failure_class"],
                    "n_patches": 0,
                    "patch_ids": [],
                }
            )
            continue
        result = evaluate_episode(episode, relevant, runner, max_new_tokens=max_new_tokens)
        result["failure_class"] = episode["failure_class"]
        records.append(result)
    return records


def evaluate_oracle(held_out, train, runner, max_new_tokens=256):
    candidate_bank = []
    for family in FAMILIES:
        candidate_bank.extend(materialize_candidate(failure, family, tokenizer=runner.tokenizer) for failure in train)
    records = []
    print(f"[oracle] heldout={len(held_out)} candidates={len(candidate_bank)}", flush=True)
    for index, episode in enumerate(held_out, start=1):
        if index == 1 or index % 25 == 0 or index == len(held_out):
            print(f"[oracle] episode {index}/{len(held_out)}", flush=True)
        relevant = relevant_patches_for_episode(candidate_bank, episode)
        success = False
        tried = []
        for candidate in relevant:
            tried.append(candidate.patch_id)
            result = evaluate_episode(episode, [candidate], runner, max_new_tokens=max_new_tokens)
            if result["success"]:
                success = True
                break
        records.append(
            {
                "episode_id": episode["episode_id"],
                "success": success,
                "failure_class": episode["failure_class"],
                "n_candidates_tried": len(tried),
                "candidate_ids_tried": tried,
            }
        )
    return records


def pass_rates(records):
    overall = sum(record["success"] for record in records) / len(records) if records else 0.0
    router = [record for record in records if record.get("failure_class") == "router_fail"]
    validator = [record for record in records if record.get("failure_class") == "validator_fail"]
    return {
        "overall_pass_rate": overall,
        "router_pass_rate": sum(record["success"] for record in router) / len(router) if router else None,
        "validator_pass_rate": sum(record["success"] for record in validator) / len(validator) if validator else None,
        "overall_n": len(records),
        "router_n": len(router),
        "validator_n": len(validator),
        "overall_pass": sum(record["success"] for record in records),
        "router_pass": sum(record["success"] for record in router),
        "validator_pass": sum(record["success"] for record in validator),
    }


def compute_ci95(values):
    if not values:
        return {"mean": None, "ci95": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci95": 0.0, "std": 0.0, "n": 1}
    std = statistics.stdev(values)
    return {"mean": mean, "ci95": 1.96 * std / math.sqrt(len(values)), "std": std, "n": len(values)}


def summarize_seed_results(seed_results):
    summary = {}
    for method in METHODS:
        rows = [row for row in seed_results if row["method"] == method]
        if not rows:
            continue
        summary[method] = {
            "overall": compute_ci95([row["overall_pass_rate"] for row in rows]),
            "router": compute_ci95([row["router_pass_rate"] for row in rows if row["router_pass_rate"] is not None]),
            "validator": compute_ci95([row["validator_pass_rate"] for row in rows if row["validator_pass_rate"] is not None]),
        }
    return summary


def context_summary(loop_outputs):
    result = {}
    for family, output in loop_outputs.items():
        patches = output["accepted_patches"]
        token_counts = [patch.token_count for patch in patches]
        result[f"{family}-evo"] = {
            "accepted_patch_count": len(patches),
            "mean_patch_tokens": sum(token_counts) / len(token_counts) if token_counts else 0.0,
            "total_patch_tokens": sum(token_counts),
        }
    return result


def load_multiple_failures(input_path):
    data = read_json(input_path)
    return [item for item in data.get("failures", []) if item.get("split") == "multiple"]


def run_seed(seed, failures, runner, max_new_tokens=256, families=None):
    split = deterministic_split(failures, seed=seed)
    print(f"[seed:{seed}] split_sizes={{{', '.join(f'{key}: {len(value)}' for key, value in split.items())}}}", flush=True)
    loop_outputs = {}
    for family in families or FAMILIES:
        print(f"[seed:{seed}] start {family}-evo", flush=True)
        loop_outputs[family] = run_family_loop(family, split, runner, max_new_tokens=max_new_tokens)
        print(
            f"[seed:{seed}] done {family}-evo accepted={len(loop_outputs[family]['accepted_patches'])}",
            flush=True,
        )
    method_records = {"no-evolution": evaluate_held_out("no-evolution", split["held_out"], [], runner, max_new_tokens=max_new_tokens)}
    for family, output in loop_outputs.items():
        print(f"[seed:{seed}] evaluate {family}-evo held-out", flush=True)
        method_records[f"{family}-evo"] = evaluate_held_out(
            f"{family}-evo",
            split["held_out"],
            output["accepted_patches"],
            runner,
            max_new_tokens=max_new_tokens,
        )
    seed_rows = []
    for method, records in method_records.items():
        row = {"seed": seed, "method": method, **pass_rates(records)}
        seed_rows.append(row)
    return {
        "seed": seed,
        "split_sizes": {key: len(value) for key, value in split.items()},
        "seed_rows": seed_rows,
        "held_out_records": method_records,
        "loop_outputs": {
            family: {
                "accepted_patches": [patch.to_json_dict() for patch in output["accepted_patches"]],
                "accepted_log": output["accepted_log"],
                "rejected_log": output["rejected_log"],
                "validation_pass_count": output["validation_pass_count"],
                "validation_size": output["validation_size"],
            }
            for family, output in loop_outputs.items()
        },
        "context_summary": context_summary(loop_outputs),
    }


def build_result(args, seeds, failures, per_seed):
    seed_rows = [row for seed_result in per_seed for row in seed_result["seed_rows"]]
    return {
        "model_id": args.model_id,
        "model_stack": ["M0", "adapter_merged_in_memory"] if args.base_adapter_dir else ["M0"],
        "base_adapter_dir": args.base_adapter_dir,
        "source": args.input,
        "seeds": seeds,
        "completed_seeds": [seed_result["seed"] for seed_result in per_seed],
        "complete": len(per_seed) == len(seeds),
        "n_failures": len(failures),
        "methods": METHODS,
        "summary": summarize_seed_results(seed_rows),
        "seed_rows": seed_rows,
        "per_seed": per_seed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(BASE_FAILURES))
    parser.add_argument("--output", default=str(ROUND2_OUT / "loop" / "evolution_loop.json"))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--base-adapter-dir", required=True, help="M1 adapter dir; round-2 patch search must run on merged M1.")
    parser.add_argument("--seeds", default="20260630")
    parser.add_argument("--families", default="NL", help="Comma-separated patch families (round 2 locks NL).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    families = [item for item in args.families.split(",") if item.strip()]
    failures = load_multiple_failures(resolve(args.input))
    runner = ModelRunner(
        args.model_id,
        cache_dir=args.model_cache_dir,
        base_adapter_dir=resolve(args.base_adapter_dir),
        require_adapter=True,
    )
    per_seed = []
    output_path = resolve(args.output)
    partial_output = str(Path(output_path).with_suffix(".partial.json"))
    for seed in seeds:
        print(f"[loop] start seed={seed}", flush=True)
        per_seed.append(run_seed(seed, failures, runner, max_new_tokens=args.max_new_tokens, families=families))
        write_json(partial_output, build_result(args, seeds, failures, per_seed))
        print(f"[loop] completed seed={seed}; partial={partial_output}", flush=True)
    result = build_result(args, seeds, failures, per_seed)
    write_json(output_path, result)
    print(json.dumps({"summary": result["summary"], "seeds": seeds, "n_failures": len(failures)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
