import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.a0_bfcl_load import CONNECTED_CATEGORIES, _canonical_ground_truth, _write_json, load_connected_categories, query_text
from scripts.a1_signal_check import call_success, router_success, validator_success
from scripts.a0_bfcl_load import parse_bfcl_call


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


def classify_failure(router, validator):
    if not router:
        return "router_fail"
    if not validator:
        return "validator_fail"
    return "none"


def build_prompt(row):
    return PROMPT_TEMPLATE.format(
        query=query_text(row),
        function_doc=json.dumps(row.get("function"), ensure_ascii=False),
    )


def summarize_records(records):
    n_total = len(records)
    n_failures = sum(1 for record in records if not record["call_success"])
    n_router_fail = sum(
        1
        for record in records
        if record.get("failure_class", classify_failure(record["router_success"], record["validator_success"])) == "router_fail"
    )
    n_validator_fail = sum(
        1
        for record in records
        if record.get("failure_class", classify_failure(record["router_success"], record["validator_success"]))
        == "validator_fail"
    )
    by_category = defaultdict(lambda: {"n_total": 0, "n_failures": 0, "router_fail": 0, "validator_fail": 0})
    query_lengths = []
    param_counts = []
    for record in records:
        bucket = by_category[record["category"]]
        bucket["n_total"] += 1
        bucket["n_failures"] += int(not record["call_success"])
        failure_class = record.get("failure_class", classify_failure(record["router_success"], record["validator_success"]))
        bucket["router_fail"] += int(failure_class == "router_fail")
        bucket["validator_fail"] += int(failure_class == "validator_fail")
        query_lengths.append(len(record.get("query", "").split()))
        param_counts.append(len(record.get("ground_truth_call", {}).get("accepted_arguments", {})))
    return {
        "n_total_episodes": n_total,
        "n_failures": n_failures,
        "call_accuracy": (n_total - n_failures) / n_total if n_total else 0.0,
        "n_router_fail": n_router_fail,
        "n_validator_fail": n_validator_fail,
        "failure_rate": n_failures / n_total if n_total else 0.0,
        "router_fail_rate": n_router_fail / n_total if n_total else 0.0,
        "validator_fail_rate": n_validator_fail / n_total if n_total else 0.0,
        "by_category": dict(by_category),
        "query_length_words": {
            "min": min(query_lengths) if query_lengths else None,
            "max": max(query_lengths) if query_lengths else None,
            "mean": sum(query_lengths) / len(query_lengths) if query_lengths else None,
        },
        "ground_truth_parameter_count": {
            "min": min(param_counts) if param_counts else None,
            "max": max(param_counts) if param_counts else None,
            "mean": sum(param_counts) / len(param_counts) if param_counts else None,
        },
    }


def _load_existing(output):
    path = Path(output)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("records", [])
    except Exception:
        return []


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
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
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


def run_collection(args):
    loaded = load_connected_categories(args.raw_dir, CONNECTED_CATEGORIES)
    rows = [row for category_rows in loaded.values() for row in category_rows]
    rows.sort(key=lambda row: row["id"])
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
        parse_error = None
        try:
            predicted_call = parse_model_output(raw_output)
        except Exception as error:
            predicted_call = None
            parse_error = str(error)
        ground_truth = _canonical_ground_truth(row["ground_truth"])
        router = router_success(predicted_call or {"name": "", "arguments": {}}, ground_truth)
        validator = validator_success(predicted_call or {"name": "", "arguments": {}}, ground_truth)
        call = call_success(predicted_call or {"name": "", "arguments": {}}, ground_truth)
        record = {
            "episode_id": row["id"],
            "category": row["category"],
            "query": query_text(row),
            "ground_truth_call": ground_truth,
            "predicted_call": predicted_call,
            "raw_model_output": raw_output,
            "parse_error": parse_error,
            "router_success": router,
            "validator_success": validator,
            "call_success": call,
            "failure_class": classify_failure(router, validator),
            "latency_ms": latency_ms,
            "prompt_template": "PROMPT_TEMPLATE",
        }
        records.append(record)
        if len(records) % args.save_every == 0:
            write_result(args.output, records, args)
        print(f"{index}/{len(rows)} {row['id']} call={call} failure={record['failure_class']}", flush=True)
    write_result(args.output, records, args)
    return records


def write_result(output, records, args):
    failures = [record for record in records if not record["call_success"]]
    result = {
        "model_id": args.model_id,
        "generation_params": {**GENERATION_PARAMS, "max_new_tokens": args.max_new_tokens},
        "prompt_template": PROMPT_TEMPLATE,
        "summary": summarize_records(records),
        "records": records,
        "failures": failures,
        "asserts": {
            "n_failures_gt_0": len(failures) > 0,
            "n_failures_lt_total": len(failures) < len(records) if records else False,
        },
    }
    _write_json(output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="A2 frozen Qwen natural failure collection")
    parser.add_argument("--raw-dir", default="data/raw/bfcl")
    parser.add_argument("--output", default="results/a2_natural_failures.json")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    records = run_collection(args)
    summary = summarize_records(records)
    assert summary["n_failures"] > 0, "frozen agent failed on nothing"
    assert summary["n_failures"] < summary["n_total_episodes"], "frozen agent failed on everything"
    with Path(args.output).open(encoding="utf-8") as handle:
        result = json.load(handle)
    json.dump({"summary": result["summary"], "asserts": result["asserts"]}, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
