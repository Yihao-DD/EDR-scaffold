import argparse
import json
import sys
from pathlib import Path

from scripts.lora_phase0 import generate_one, load_model_for_eval, load_tokenizer
from scripts.phase0_probe import load_exp2, read_json, write_json


def evaluate(args):
    evo = load_exp2(args.exp2_root)
    baseline = read_json(args.failures)
    by_id = {row["episode_id"]: row for row in baseline["records"]}
    arena = read_json(args.arena)
    records = arena.get("arena_records", [])
    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(args.model_id, adapter_dir=args.adapter_dir, qlora=not args.no_qlora)
    eval_records = []
    for index, row in enumerate(records, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(records):
            print(f"[eval:capped-arena] {index}/{len(records)}", flush=True)
        raw = generate_one(model, tokenizer, row["input"], args.max_new_tokens)
        try:
            prediction = evo.parse_model_output(raw)
            parse_error = None
        except Exception as error:
            prediction = None
            parse_error = str(error)
        episode = by_id[row["episode_id"]]
        success = evo.call_success(prediction or {"name": "", "arguments": {}}, episode["ground_truth_call"])
        eval_records.append(
            {
                "episode_id": row["episode_id"],
                "success": success,
                "prediction": prediction,
                "raw_model_output": raw,
                "parse_error": parse_error,
                "target_output_call": row.get("output_call"),
                "function_name": row.get("function_name"),
            }
        )
    ok = sum(row["success"] for row in eval_records)
    result = {
        "probe": "capped_arena_eval",
        "adapter_dir": args.adapter_dir,
        "model_id": args.model_id,
        "arena": args.arena,
        "arena_n": len(eval_records),
        "arena_success_rate": ok / len(eval_records) if eval_records else 0.0,
        "arena_forget_rate": (len(eval_records) - ok) / len(eval_records) if eval_records else 0.0,
        "records": eval_records,
    }
    write_json(args.output, result)
    print(
        f"[eval] capped_arena_success={ok}/{len(eval_records)} "
        f"forget={(len(eval_records)-ok)}/{len(eval_records)}={result['arena_forget_rate']:.4f}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate adapter on v1.13 capped-pool arena")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--arena", default="results/v13_capped_pool_arena.json")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
