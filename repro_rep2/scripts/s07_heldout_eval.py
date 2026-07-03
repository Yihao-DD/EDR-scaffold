import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.lora_phase0 import evaluate_episode, load_model_for_eval, load_tokenizer, rate
from scripts.phase0_probe import DEFAULT_SPLIT_SEED, load_exp2, read_json, write_json


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def canonical_call(call):
    if not call:
        return None
    name = call.get("name")
    arguments = call.get("arguments") or {}
    return {
        "name": name,
        "arguments": {key: arguments[key] for key in sorted(arguments)},
    }


def calls_equal(left, right):
    return canonical_call(left) == canonical_call(right)


def train_signature(dataset_path):
    rows = read_jsonl(dataset_path)
    functions = {row.get("function_name") for row in rows if row.get("function_name")}
    failures = {row.get("failure_class") for row in rows if row.get("failure_class")}
    episodes = {row.get("episode_id") for row in rows if row.get("episode_id")}
    return {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "unique_episodes": len(episodes),
        "functions": sorted(functions),
        "failure_classes": sorted(failures),
    }


def generalization_label(function_name, failure_class, signature):
    seen_function = function_name in set(signature["functions"])
    seen_failure = failure_class in set(signature["failure_classes"])
    if seen_function and seen_failure:
        return "SS"
    if seen_function and not seen_failure:
        return "SN"
    if not seen_function and seen_failure:
        return "NS"
    return "NN"


def summarize(records, key):
    output = {}
    for value in sorted({row[key] for row in records}):
        rows = [row for row in records if row[key] == value]
        output[value] = {
            "n": len(rows),
            "success": sum(1 for row in rows if row["success"]),
            "rate": rate(rows),
        }
    return output


def teacher_agree_summary(records, selector):
    rows = [row for row in records if row["success"] and selector(row)]
    if not rows:
        return {"n": 0, "agree": 0, "rate": None}
    agree = sum(1 for row in rows if row.get("teacher_agree") is True)
    return {"n": len(rows), "agree": agree, "rate": agree / len(rows)}


def evaluate_heldout(args):
    root = Path(args.root)
    baseline = read_json(args.failures)
    evo = load_exp2(args.exp2_root)
    baseline_by_id = {row["episode_id"]: row for row in baseline["records"]}
    heldout_payload = read_json(root / args.heldout_partition)
    heldout_rows = heldout_payload["episodes"]
    train_sig = train_signature(root / args.train_dataset)

    missing = [row["episode_id"] for row in heldout_rows if row["episode_id"] not in baseline_by_id]
    if missing:
        raise AssertionError({"missing_heldout_in_baseline": missing[:20], "count": len(missing)})

    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(
        args.model_id,
        adapter_dir=args.adapter_dir,
        qlora=not args.no_qlora,
        base_adapter_dir=args.base_adapter_dir,
    )

    records = []
    for index, row in enumerate(heldout_rows, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(heldout_rows):
            print(f"[heldout-eval] {args.run_id} {index}/{len(heldout_rows)}", flush=True)
        episode = baseline_by_id[row["episode_id"]]
        result = evaluate_episode(evo, model, tokenizer, episode, baseline["prompt_template"], args.max_new_tokens)
        function_name = row.get("function_name") or episode["ground_truth_call"]["name"]
        failure_class = row.get("failure_class") or episode.get("failure_class", "unknown")
        teacher = row.get("teacher") or {}
        prediction = result.get("prediction")
        teacher_prediction = teacher.get("prediction")
        label = generalization_label(function_name, failure_class, train_sig)
        records.append(
            {
                **result,
                "partition": row.get("partition"),
                "function_name": function_name,
                "generalization": label,
                "teacher_success": bool(teacher.get("success")),
                "teacher_prediction": teacher_prediction,
                "teacher_agree": calls_equal(prediction, teacher_prediction) if result["success"] and teacher_prediction else False,
            }
        )

    teacher_success_records = [row for row in records if row["teacher_success"]]
    teacher_success_n = len(teacher_success_records)
    repair_n = sum(1 for row in records if row["success"])
    summary = {
        "run_id": args.run_id,
        "arm": args.arm,
        "config": args.config,
        "seed": args.seed,
        "adapter_dir": args.adapter_dir,
        "base_adapter_dir": args.base_adapter_dir,
        "heldout_n": len(records),
        "heldout_repair": repair_n,
        "heldout_repair_rate": repair_n / len(records) if records else 0.0,
        "teacher_success_n": teacher_success_n,
        "retention_ratio_vs_partition_teacher": (repair_n / len(records)) / (teacher_success_n / len(records))
        if teacher_success_n
        else None,
        "conditional_recovery_on_teacher_success": rate(teacher_success_records) if teacher_success_records else None,
        "partition": summarize(records, "partition"),
        "generalization": summarize(records, "generalization"),
        "teacher_agree": {
            "all_success": teacher_agree_summary(records, lambda row: True),
            "teacher_success_successes": teacher_agree_summary(records, lambda row: row["teacher_success"]),
            "nn_successes": teacher_agree_summary(records, lambda row: row["generalization"] == "NN"),
            "scaffold_only_successes": teacher_agree_summary(records, lambda row: row["partition"] == "scaffold_only"),
        },
        "train_signature": {
            "dataset": train_sig["dataset"],
            "rows": train_sig["rows"],
            "unique_episodes": train_sig["unique_episodes"],
            "function_count": len(train_sig["functions"]),
            "failure_class_count": len(train_sig["failure_classes"]),
        },
    }
    payload = {
        "probe": "s07_heldout_eval",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "heldout_partition": args.heldout_partition,
        "summary": summary,
        "records": records,
    }
    write_json(root / args.output, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="s07 D2 heldout@158 pure-forward evaluation")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-adapter-dir", default=None)
    parser.add_argument("--arm", required=True, choices=["main", "star"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--heldout-partition", default="results/s01_heldout_pass16_partition.json")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    evaluate_heldout(args)


if __name__ == "__main__":
    main()
