import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.patch_pair_probe import build_pair


FAMILIES = ["NL", "NL-padded", "STRUCT"]
METHODS = ["no-evolution", "NL-evo", "NL-padded-evo", "STRUCT-evo", "oracle"]
PROMPT_TEMPLATE = """You are a function-calling model. Choose exactly one function from the function document and produce exactly one JSON object.

Output format:
{{"name": "<function_name>", "arguments": {{"<argument_name>": <argument_value>}}}}

Harness patch guidance:
{patch_context}

User query:
{query}

Function document:
{function_doc}

Return only the JSON object. Do not explain."""


@dataclass
class PatchCandidate:
    patch_id: str
    family: str
    component: str
    trigger_episode_id: str
    target_function: str
    selected_function: str
    parameter_names: list[str]
    patch_text: str
    token_count: int
    iu_ids: list[str]

    def to_json_dict(self):
        return asdict(self)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def parse_bfcl_call(call):
    if isinstance(call, dict) and "name" in call and "arguments" in call:
        return {"name": call["name"], "arguments": dict(call.get("arguments") or {})}
    if isinstance(call, dict) and len(call) == 1:
        name, arguments = next(iter(call.items()))
        return {"name": name, "arguments": dict(arguments or {})}
    if not isinstance(call, str):
        raise ValueError(f"Unsupported call shape: {call!r}")
    expression = ast.parse(call, mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"Expected a function call expression: {call!r}")
    arguments = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("Variadic keyword arguments are not supported")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return {"name": _call_name(expression.func), "arguments": arguments}


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


def _has_concrete(value):
    if value == "":
        return False
    if isinstance(value, dict):
        return any(_has_concrete(nested) for nested in value.values())
    return True


def _matches_accepted(value, accepted):
    values = accepted if isinstance(accepted, list) else [accepted]
    non_empty = [candidate for candidate in values if candidate != ""]
    if not non_empty:
        return value in ("", None)
    return any(_value_matches(value, candidate) for candidate in non_empty)


def _value_matches(value, accepted):
    if isinstance(accepted, dict):
        if not isinstance(value, dict):
            return False
        required = {
            key
            for key, nested in accepted.items()
            if any(_has_concrete(v) for v in (nested if isinstance(nested, list) else [nested]))
        }
        if set(value) != required:
            return False
        return all(_matches_accepted(value[key], accepted[key]) for key in required)
    return value == accepted


def router_success(prediction, ground_truth):
    try:
        return parse_bfcl_call(prediction)["name"] == ground_truth["name"]
    except Exception:
        return False


def validator_success(prediction, ground_truth):
    try:
        parsed = parse_bfcl_call(prediction)
        if parsed["name"] != ground_truth["name"]:
            return False
        required = {
            key
            for key, accepted in ground_truth["accepted_arguments"].items()
            if any(_has_concrete(value) for value in (accepted if isinstance(accepted, list) else [accepted]))
        }
        if set(parsed["arguments"]) != required:
            return False
        return all(_matches_accepted(parsed["arguments"][key], ground_truth["accepted_arguments"][key]) for key in required)
    except Exception:
        return False


def call_success(prediction, ground_truth):
    return router_success(prediction, ground_truth) and validator_success(prediction, ground_truth)


def deterministic_split(failures, seed, train_ratio=0.5, validation_ratio=0.25):
    keyed = []
    for item in failures:
        digest = hashlib.sha256(f"{seed}:{item['episode_id']}".encode("utf-8")).hexdigest()
        keyed.append((digest, item))
    ordered = [item for _, item in sorted(keyed)]
    n = len(ordered)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "held_out": ordered[validation_end:],
    }


def materialize_candidate(failure, family, tokenizer=None):
    pair = build_pair(failure, tokenizer=tokenizer)
    if family == "STRUCT":
        patch_text = pair["struct_patch"]
        token_count = pair["struct_token_count"]
        iu_ids = pair["struct_iu_ids"]
    elif family == "NL":
        patch_text = pair["nl_patch"]
        token_count = pair["nl_token_count"]
        iu_ids = pair["nl_iu_ids"]
    elif family == "NL-padded":
        patch_text = pair["nl_padded_patch"]
        token_count = pair["nl_padded_token_count"]
        iu_ids = pair["nl_padded_iu_ids"]
    else:
        raise ValueError(f"Unknown patch family: {family}")
    component = "router" if failure.get("failure_class") == "router_fail" else "validator"
    return PatchCandidate(
        patch_id=f"{family}:{failure['episode_id']}",
        family=family,
        component=component,
        trigger_episode_id=failure["episode_id"],
        target_function=failure.get("ground_truth_call", {}).get("name", ""),
        selected_function=(failure.get("predicted_call") or {}).get("name", ""),
        parameter_names=list((failure.get("ground_truth_call", {}).get("accepted_arguments") or {}).keys()),
        patch_text=patch_text,
        token_count=token_count,
        iu_ids=iu_ids,
    )


def patch_relevant_to_episode(patch, episode):
    gt_name = episode.get("ground_truth_call", {}).get("name", "")
    if patch.component == "router":
        function_names = {function.get("name") for function in episode.get("function_pool", []) if isinstance(function, dict)}
        selected = (episode.get("predicted_call") or {}).get("name", "")
        return patch.target_function == gt_name and (patch.selected_function == selected or patch.selected_function in function_names)
    episode_params = set((episode.get("ground_truth_call", {}).get("accepted_arguments") or {}).keys())
    return patch.target_function == gt_name and bool(episode_params.intersection(patch.parameter_names))


def relevant_patches_for_episode(patches, episode):
    return [patch for patch in patches if patch_relevant_to_episode(patch, episode)]


def build_prompt(episode, patches):
    patch_context = "\n\n".join(f"[PATCH {index}] {patch.patch_text}" for index, patch in enumerate(patches, start=1))
    if not patch_context:
        patch_context = "None."
    return PROMPT_TEMPLATE.format(
        patch_context=patch_context,
        query=episode.get("query", ""),
        function_doc=json.dumps(episode.get("function_pool", []), ensure_ascii=False),
    )


class ModelRunner:
    def __init__(self, model_id, cache_dir=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, prompt, max_new_tokens=256):
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if hasattr(self.tokenizer, "apply_chat_template") else prompt
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs.input_ids.shape[-1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


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
    scoped = [episode for episode in validation if patch_relevant_to_episode(candidate, episode)]
    return scoped


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


def _load_multiple_failures(input_path):
    data = _read_json(input_path)
    return [item for item in data.get("failures", []) if item.get("split") == "multiple"]


def run_seed(seed, failures, runner, max_new_tokens=256):
    split = deterministic_split(failures, seed=seed)
    print(f"[seed:{seed}] split_sizes={{{', '.join(f'{key}: {len(value)}' for key, value in split.items())}}}", flush=True)
    loop_outputs = {}
    for family in FAMILIES:
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
    print(f"[seed:{seed}] evaluate oracle", flush=True)
    method_records["oracle"] = evaluate_oracle(split["held_out"], split["train"], runner, max_new_tokens=max_new_tokens)
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


def write_markdown(result, output_path):
    lines = [
        "# Three-Family Self-Evolution Main Result",
        "",
        f"PROBE signal (n_seeds={len(result['seeds'])}). BFCL Multiple natural failures are split train/validation/held-out by deterministic hash per seed.",
        "",
        "## Held-Out Pass Rate",
        "",
        "| method | overall mean | overall CI95 | router mean | router CI95 | validator mean | validator CI95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, stats in result["summary"].items():
        lines.append(
            f"| {method} | {stats['overall']['mean']:.4f} | {stats['overall']['ci95']:.4f} | "
            f"{stats['router']['mean']:.4f} | {stats['router']['ci95']:.4f} | "
            f"{stats['validator']['mean']:.4f} | {stats['validator']['ci95']:.4f} |"
        )
    lines.extend(["", "## Context Cost", "", "| seed | method | accepted patches | mean patch tokens | total patch tokens |", "|---:|---|---:|---:|---:|"])
    for seed_result in result["per_seed"]:
        for method, stats in seed_result["context_summary"].items():
            lines.append(
                f"| {seed_result['seed']} | {method} | {stats['accepted_patch_count']} | "
                f"{stats['mean_patch_tokens']:.2f} | {stats['total_patch_tokens']} |"
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Three-family self-evolution loop for BFCL Multiple failures")
    parser.add_argument("--input", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--output", default="results/evolution_main.json")
    parser.add_argument("--log", default="logs/evolution_main.md")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--seeds", default="20260630,20260631,20260632,20260633,20260634")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    failures = _load_multiple_failures(args.input)
    runner = ModelRunner(args.model_id, cache_dir=args.model_cache_dir)
    per_seed = []
    partial_output = str(Path(args.output).with_suffix(".partial.json"))
    partial_log = str(Path(args.log).with_suffix(".partial.md"))
    for seed in seeds:
        print(f"[main] start seed={seed}", flush=True)
        per_seed.append(run_seed(seed, failures, runner, max_new_tokens=args.max_new_tokens))
        partial_result = build_result(args, seeds, failures, per_seed)
        _write_json(partial_output, partial_result)
        write_markdown(partial_result, partial_log)
        print(f"[main] completed seed={seed}; partial={partial_output}", flush=True)
    result = build_result(args, seeds, failures, per_seed)
    _write_json(args.output, result)
    write_markdown(result, args.log)
    print(json.dumps({"summary": result["summary"], "seeds": seeds, "n_failures": len(failures)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
