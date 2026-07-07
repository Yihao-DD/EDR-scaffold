"""Sibling-300 regression arena evaluation (the `forget` surface).

The arena mixes parallel-category and single-call episodes that the base
model solves; `forget` = fraction newly wrong under an adapter stack.

CLI:
  python -m edr.evaluation.sibling --model-id ... --adapter-dir ... \
      --arm <label> --config <label> --seed N --run-id <id> --output <json>
"""

from __future__ import annotations

import argparse
import json
import time

from edr.io_utils import read_json, write_json
from edr.modeling import generate_one, load_model_for_eval, load_tokenizer
from edr.paths import BASE_FAILURES, SIBLING_ARENA, resolve
from edr.data.splits import build_base_prompt
from edr.evaluation.parallel_arena import (
    PARALLEL_CATEGORIES,
    parallel_call_success,
    parallel_router_success,
    parse_model_output_multi,
    render_parallel_prompt,
    single_call_success,
)


def rate(rows):
    return sum(1 for row in rows if row["call_success"]) / len(rows) if rows else 0.0


def summarize(records, key):
    output = {}
    for value in sorted({row[key] for row in records}):
        rows = [row for row in records if row[key] == value]
        success = sum(1 for row in rows if row["call_success"])
        output[value] = {
            "n": len(rows),
            "success": success,
            "success_rate": success / len(rows) if rows else 0.0,
            "forget_rate": 1 - (success / len(rows) if rows else 0.0),
        }
    return output


def evaluate(args):
    arena = read_json(resolve(args.arena))
    records_in = arena["records"]
    baseline = read_json(resolve(args.failures))

    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(
        args.model_id,
        adapter_dir=resolve(args.adapter_dir) if args.adapter_dir else None,
        qlora=not args.no_qlora,
        base_adapter_dir=resolve(args.base_adapter_dir) if args.base_adapter_dir else None,
    )

    records = []
    for index, row in enumerate(records_in, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(records_in):
            print(f"[sibling-eval] {args.run_id} {index}/{len(records_in)} {row['episode_id']}", flush=True)
        if row["category"] in PARALLEL_CATEGORIES:
            prompt = render_parallel_prompt(row)
            ground_truth_calls = row["ground_truth_calls"]
            started = time.time()
            raw = generate_one(model, tokenizer, prompt, args.max_new_tokens)
            latency_ms = int((time.time() - started) * 1000)
            parse_error = None
            try:
                prediction = parse_model_output_multi(raw)
            except Exception as error:
                prediction = []
                parse_error = str(error)
            router = parallel_router_success(prediction, ground_truth_calls)
            call = parallel_call_success(prediction, ground_truth_calls)
            validator = call
        else:
            prompt = build_base_prompt(
                {"query": row["query"], "function_pool": row["function_pool"]},
                baseline["prompt_template"],
            )
            started = time.time()
            raw = generate_one(model, tokenizer, prompt, args.max_new_tokens)
            latency_ms = int((time.time() - started) * 1000)
            parse_error = None
            try:
                prediction = parse_model_output_multi(raw)
            except Exception as error:
                prediction = []
                parse_error = str(error)
            if len(prediction) == 1:
                router = prediction[0].get("name") == row["ground_truth_call"]["name"]
                call = single_call_success(prediction[0], row["ground_truth_call"])
                validator = call
            else:
                router = False
                validator = False
                call = False

        records.append(
            {
                "episode_id": row["episode_id"],
                "category": row["category"],
                "segment": row["segment"],
                "predicted_calls": prediction,
                "raw_model_output": raw,
                "parse_error": parse_error,
                "router_success": router,
                "validator_success": validator,
                "call_success": call,
                "latency_ms": latency_ms,
            }
        )

    success = sum(1 for row in records if row["call_success"])
    summary = {
        "run_id": args.run_id,
        "arm": args.arm,
        "config": args.config,
        "seed": args.seed,
        "adapter_dir": args.adapter_dir,
        "base_adapter_dir": args.base_adapter_dir,
        "arena": str(args.arena),
        "arena_n": len(records),
        "success": success,
        "success_rate": success / len(records) if records else 0.0,
        "forget_sibling": 1 - (success / len(records) if records else 0.0),
        "by_category": summarize(records, "category"),
        "by_segment": summarize(records, "segment"),
    }
    payload = {
        "probe": "sibling_arena_eval",
        "model_id": args.model_id,
        "summary": summary,
        "records": records,
    }
    write_json(resolve(args.output), payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-adapter-dir", default=None)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arena", default=str(SIBLING_ARENA))
    parser.add_argument("--failures", default=str(BASE_FAILURES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
