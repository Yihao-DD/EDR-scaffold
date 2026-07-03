import argparse
import json
import time
from pathlib import Path

from scripts.lora_phase0 import generate_one, load_model_for_eval, load_tokenizer
from scripts.phase0_probe import write_json
from scripts.v17_parallel_arena import (
    PARALLEL_CATEGORIES,
    build_base_prompt,
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
    root = Path(args.root)
    arena = json.loads((root / args.arena).read_text(encoding="utf-8"))
    records_in = arena["records"]
    baseline = json.loads((root / args.failures).read_text(encoding="utf-8"))

    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(
        args.model_id,
        adapter_dir=args.adapter_dir,
        qlora=not args.no_qlora,
        base_adapter_dir=args.base_adapter_dir,
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
        "arena": args.arena,
        "arena_n": len(records),
        "success": success,
        "success_rate": success / len(records) if records else 0.0,
        "forget_sibling": 1 - (success / len(records) if records else 0.0),
        "by_category": summarize(records, "category"),
        "by_segment": summarize(records, "segment"),
    }
    payload = {
        "probe": "s07_sibling_arena_eval",
        "model_id": args.model_id,
        "summary": summary,
        "records": records,
    }
    write_json(root / args.output, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="s07 final sibling-arena regression evaluation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-adapter-dir", default=None)
    parser.add_argument("--arm", required=True, choices=["main", "star"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arena", default="results/v17_sibling_arena.json")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
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
