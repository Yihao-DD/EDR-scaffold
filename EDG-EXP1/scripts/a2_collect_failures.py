import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from scripts.bfcl_common import (
    call_success,
    canonical_ground_truth,
    ensure_raw_data,
    load_bfcl_splits,
    parse_bfcl_call,
    router_success,
    validator_success,
    write_json,
)


PROMPT_TEMPLATE = """You are a function-calling model. Choose exactly one function from the function document and produce exactly one JSON object.

Output format:
{{"name": "<function_name>", "arguments": {{"<argument_name>": <argument_value>}}}}

User query:
{query}

Function document:
{function_doc}

Return only the JSON object. Do not explain."""

GENERATION_PARAMS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 256,
}


def parse_model_output(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "name" in obj and "arguments" in obj:
                return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
            if len(obj) == 1:
                name, arguments = next(iter(obj.items()))
                if isinstance(arguments, dict):
                    return {"name": name, "arguments": arguments}
    match = re.search(r"[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)", text, re.S)
    if match:
        return parse_bfcl_call(match.group(0))
    raise ValueError("No parseable function call found")


def build_prompt(row):
    return PROMPT_TEMPLATE.format(
        query=row.get("query", ""),
        function_doc=json.dumps(row.get("function_pool", []), ensure_ascii=False),
    )


def classify_failure(router, validator):
    if not router:
        return "router_fail"
    if not validator:
        return "validator_fail"
    return "none"


def _basic_stats(values):
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def summarize_records(records):
    by_split = defaultdict(lambda: {"n_total": 0, "n_failures": 0, "router_fail": 0, "validator_fail": 0})
    by_category = defaultdict(lambda: {"n_total": 0, "n_failures": 0, "router_fail": 0, "validator_fail": 0})
    query_lengths = []
    param_counts = []
    for record in records:
        failure_class = record.get("failure_class", classify_failure(record["router_success"], record["validator_success"]))
        for bucket in [by_split[record["split"]], by_category[record["category"]]]:
            bucket["n_total"] += 1
            bucket["n_failures"] += int(not record["call_success"])
            bucket["router_fail"] += int(failure_class == "router_fail")
            bucket["validator_fail"] += int(failure_class == "validator_fail")
        query_lengths.append(len(record.get("query", "").split()))
        param_counts.append(len(record.get("ground_truth_call", {}).get("accepted_arguments", {})))

    n_total = len(records)
    n_failures = sum(1 for record in records if not record["call_success"])
    overall = {
        "n_total_episodes": n_total,
        "n_failures": n_failures,
        "call_accuracy": (n_total - n_failures) / n_total if n_total else 0.0,
        "failure_rate": n_failures / n_total if n_total else 0.0,
        "n_router_fail": sum(bucket["router_fail"] for bucket in by_split.values()),
        "n_validator_fail": sum(bucket["validator_fail"] for bucket in by_split.values()),
        "query_length_words": _basic_stats(query_lengths),
        "ground_truth_parameter_count": _basic_stats(param_counts),
    }
    return {
        "overall": overall,
        "by_split": dict(by_split),
        "by_category": dict(by_category),
    }


def _load_existing(path):
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def _load_model(model_id, cache_dir=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def _generate(tokenizer, model, prompt, max_new_tokens):
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(tokenizer, "apply_chat_template") else prompt
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[:, inputs.input_ids.shape[-1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]


def build_record(row, raw_output, latency_ms):
    parse_error = None
    try:
        predicted_call = parse_model_output(raw_output)
    except Exception as error:
        predicted_call = None
        parse_error = str(error)
    gt = canonical_ground_truth(row)
    prediction = predicted_call or {"name": "", "arguments": {}}
    router = router_success(prediction, gt)
    validator = validator_success(prediction, gt)
    call = call_success(prediction, gt)
    return {
        "episode_id": row["id"],
        "split": row["split"],
        "category": row["category"],
        "query": row["query"],
        "function_pool": row["function_pool"],
        "ground_truth_call": gt,
        "predicted_call": predicted_call,
        "raw_model_output": raw_output,
        "parse_error": parse_error,
        "router_success": router,
        "validator_success": validator,
        "call_success": call,
        "failure_class": classify_failure(router, validator),
        "latency_ms": latency_ms,
    }


def write_result(output, records, args):
    summary = summarize_records(records)
    failures = [record for record in records if not record["call_success"]]
    split_asserts = {}
    for split, counts in summary["by_split"].items():
        split_asserts[f"{split}_n_failures_gt_0"] = counts["n_failures"] > 0
        split_asserts[f"{split}_n_failures_lt_total"] = counts["n_failures"] < counts["n_total"]
    result = {
        "model_id": args.model_id,
        "generation_params": {**GENERATION_PARAMS, "max_new_tokens": args.max_new_tokens},
        "prompt_template": PROMPT_TEMPLATE,
        "summary": summary,
        "records": records,
        "failures": failures,
        "asserts": split_asserts,
    }
    write_json(output, result)
    return result


def run_collection(args):
    ensure_raw_data(args.source_data_dir, args.raw_dir)
    loaded = load_bfcl_splits(args.raw_dir)
    rows = [row for split in ["multiple", "simple"] for row in loaded[split]]
    rows.sort(key=lambda row: (row["split"], row["category"], row["id"]))
    if args.limit is not None:
        rows = rows[: args.limit]
    existing = _load_existing(args.output) if args.resume else []
    seen = {record["episode_id"] for record in existing}
    records = list(existing)
    tokenizer, model = _load_model(args.model_id, cache_dir=args.model_cache_dir)
    for index, row in enumerate(rows, start=1):
        if row["id"] in seen:
            continue
        prompt = build_prompt(row)
        started = time.time()
        raw_output = _generate(tokenizer, model, prompt, args.max_new_tokens)
        latency_ms = int((time.time() - started) * 1000)
        record = build_record(row, raw_output, latency_ms)
        records.append(record)
        if len(records) % args.save_every == 0:
            write_result(args.output, records, args)
        print(f"{index}/{len(rows)} {row['split']} {row['id']} call={record['call_success']} failure={record['failure_class']}", flush=True)
    result = write_result(args.output, records, args)
    for value in result["asserts"].values():
        assert value is True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="A2 frozen Qwen collection for BFCL Multiple and Simple")
    parser.add_argument("--source-data-dir", default="../EDG-BFCL/deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--raw-dir", default="data/raw/bfcl")
    parser.add_argument("--output", default="results/a2_failures.json")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_collection(args)
    print(json.dumps({"summary": result["summary"], "asserts": result["asserts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
