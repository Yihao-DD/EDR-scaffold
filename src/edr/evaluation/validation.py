"""Validation-156 + regression-400 monitor evaluation (selection surfaces).

These surfaces monitored round-1 model selection; heldout/sibling are the
adjudication surfaces. Regression sets are recomputed deterministically from
the frozen split seeds (identical to round 1).

CLI:
  python -m edr.evaluation.validation --model-id ... --adapter-dir ... --output <json>
"""

from __future__ import annotations

import argparse

from edr.io_utils import write_json
from edr.modeling import generate_one, load_model_for_eval, load_tokenizer
from edr.paths import resolve
from edr.data.splits import (
    DEFAULT_REGRESSION_SEED,
    DEFAULT_SPLIT_SEED,
    build_base_prompt,
    load_round1_inputs,
    select_regression_sets,
)
from edr.scaffold.teacher import parse_prediction, success_for_prediction


def rate(records):
    return sum(1 for row in records if row["success"]) / len(records) if records else 0.0


def evaluate_episode(model, tokenizer, episode, prompt_template, max_new_tokens):
    raw = generate_one(model, tokenizer, build_base_prompt(episode, prompt_template), max_new_tokens)
    prediction, parse_error = parse_prediction(raw)
    return {
        "episode_id": episode["episode_id"],
        "success": success_for_prediction(prediction, episode),
        "prediction": prediction,
        "raw_model_output": raw,
        "parse_error": parse_error,
        "failure_class": episode.get("failure_class", "unknown"),
    }


def evaluate(args):
    baseline, _evolution, split, _patches = load_round1_inputs(args.split_seed)
    regression = select_regression_sets(baseline["records"], split, args.regression_seed)
    r_success_ids = set(regression["r_success_eval"])
    val_episodes = split["validation"]
    r_success = [row for row in baseline["records"] if row["episode_id"] in r_success_ids]
    if args.eval_limit:
        val_episodes = val_episodes[: args.eval_limit]
        r_success = r_success[: args.eval_limit]
    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(
        args.model_id,
        adapter_dir=resolve(args.adapter_dir) if args.adapter_dir else None,
        qlora=not args.no_qlora,
        base_adapter_dir=resolve(args.base_adapter_dir) if args.base_adapter_dir else None,
    )
    val_records = []
    for index, episode in enumerate(val_episodes, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(val_episodes):
            print(f"[eval:val] {index}/{len(val_episodes)}", flush=True)
        val_records.append(evaluate_episode(model, tokenizer, episode, baseline["prompt_template"], args.max_new_tokens))
    success_records = []
    for index, episode in enumerate(r_success, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(r_success):
            print(f"[eval:r-success] {index}/{len(r_success)}", flush=True)
        success_records.append(evaluate_episode(model, tokenizer, episode, baseline["prompt_template"], args.max_new_tokens))
    result = {
        "adapter_dir": args.adapter_dir,
        "base_adapter_dir": args.base_adapter_dir,
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "val_n": len(val_records),
        "val_repair_rate": rate(val_records),
        "r_success_n": len(success_records),
        "r_success_rate": rate(success_records),
        "val_records": val_records,
        "r_success_records": success_records,
    }
    write_json(resolve(args.output), result)
    print(
        f"[eval] val_repair_rate={result['val_repair_rate']:.4f} "
        f"r_success_rate={result['r_success_rate']:.4f}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--base-adapter-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--regression-seed", type=int, default=DEFAULT_REGRESSION_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-qlora", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
