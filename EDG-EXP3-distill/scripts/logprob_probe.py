import argparse
import json
import statistics
import sys
from pathlib import Path

from scripts.phase0_probe import (
    DEFAULT_MODEL_ID,
    DEFAULT_SPLIT_SEED,
    call_key,
    load_exp2,
    output_text,
    read_json,
    write_json,
)


DEFAULT_GRID_FILES = [
    "results/s03_grid_clean/main/main_r8_lr2e-5_ep1_seed20260703.json",
    "results/s03_grid_clean/main/main_r8_lr2e-5_ep2_seed20260703.json",
    "results/s03_grid_clean/star/star_r8_lr2e-5_ep1_seed20260703.json",
    "results/s03_grid_clean/star/star_r8_lr2e-5_ep2_seed20260703.json",
]


def load_tokenizer(model_id):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_for_eval(model_id, qlora=True):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def render_user_prompt(tokenizer, input_text):
    messages = [{"role": "user", "content": input_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return input_text


def render_training_text(tokenizer, input_text, output_text):
    messages = [{"role": "user", "content": input_text}, {"role": "assistant", "content": output_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{input_text}\n{output_text}"


def encode_row(tokenizer, input_text, output_text, max_length):
    prompt_text = render_user_prompt(tokenizer, input_text)
    full_text = render_training_text(tokenizer, input_text, output_text)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {"input_ids": full_ids, "labels": labels}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_name(path):
    return Path(path).stem.replace("_seed20260703", "")


def score_output(model, tokenizer, input_text, target_output, max_length):
    import torch

    encoded = encode_row(tokenizer, input_text, target_output, max_length=max_length)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=model.device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    token_count = int(mask.sum().item())
    if token_count == 0:
        return {"total_logprob": None, "mean_logprob": None, "token_count": 0}
    selected_logits = logits[mask]
    selected_labels = shifted_labels[mask]
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    token_log_probs = log_probs.gather(1, selected_labels[:, None]).squeeze(1)
    total = float(token_log_probs.sum().detach().cpu())
    return {
        "total_logprob": total,
        "mean_logprob": total / token_count,
        "token_count": token_count,
    }


def summarize(records, key):
    groups = {}
    for row in records:
        value = row.get(key, "unknown")
        score = row.get("correct_score") or row.get("score")
        if not score or score.get("mean_logprob") is None:
            continue
        groups.setdefault(value, []).append(score["mean_logprob"])
    summary = {}
    for value, scores in sorted(groups.items()):
        summary[value] = {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
            "min": min(scores),
            "max": max(scores),
        }
    return summary


def attach_grid_success(target, grid_val_by_name):
    episode_id = target["episode_id"]
    target["grid_success"] = {}
    for name, records in grid_val_by_name.items():
        row = records.get(episode_id)
        if row is not None:
            target["grid_success"][name] = bool(row["success"])


def build_targets(args):
    baseline = read_json(args.failures)
    evo = load_exp2(args.exp2_root)
    multiple_failures = [item for item in baseline["failures"] if item.get("split") == "multiple"]
    split = evo.deterministic_split(multiple_failures, seed=args.split_seed)
    train_ids = {item["episode_id"] for item in split["train"]}
    val_ids = {item["episode_id"] for item in split["validation"]}
    records_by_id = {item["episode_id"]: item for item in baseline["records"]}
    failures_by_id = {item["episode_id"]: item for item in baseline["failures"]}
    episodes_by_id = {item["episode_id"]: item for item in multiple_failures}
    inventory = read_json(args.s00_input)
    partition_by_id = {row["episode_id"]: row for row in read_json(args.s01_input)["episodes"]}

    grid_val_by_name = {}
    grid_success_by_name = {}
    for path in args.grid_json:
        payload = read_json(path)
        name = stable_name(path)
        grid_val_by_name[name] = {row["episode_id"]: row for row in payload["val_records"]}
        grid_success_by_name[name] = {row["episode_id"]: row for row in payload["r_success_records"]}

    repaired = read_json(args.repaired_input)
    targets = []
    seen = set()
    for row in repaired:
        episode_id = row["episode_id"]
        if episode_id not in train_ids and episode_id not in val_ids:
            continue
        episode = episodes_by_id[episode_id]
        part = row.get("partition", partition_by_id.get(episode_id, {}).get("partition", "unknown"))
        split_name = "train" if episode_id in train_ids else "validation"
        target = {
            "target_id": f"teacher:{split_name}:{episode_id}",
            "target_family": f"{split_name}_teacher_repaired",
            "episode_id": episode_id,
            "split": split_name,
            "partition": part,
            "input": args.prompt_template_key and "" or None,
            "target_output": output_text(row["teacher_prediction"]),
            "function_name": row.get("function_name") or episode.get("ground_truth_call", {}).get("name", ""),
        }
        target["input"] = args.prompt_template.format(
            query=episode.get("query", ""),
            function_doc=json.dumps(episode.get("function_pool", []), ensure_ascii=False),
        )
        if split_name == "validation":
            attach_grid_success(target, grid_val_by_name)
        key = (target["target_family"], target["episode_id"], target["target_output"])
        if key not in seen:
            seen.add(key)
            targets.append(target)

    pass16_seen = {}
    for row in read_jsonl(args.pass16_success_input):
        episode_id = row["episode_id"]
        if episode_id not in train_ids:
            continue
        key = (episode_id, call_key(row["prediction"]))
        pass16_seen.setdefault(key, row)
    for (episode_id, _), row in sorted(pass16_seen.items()):
        episode = episodes_by_id[episode_id]
        target = {
            "target_id": f"base_pass16:train:{episode_id}",
            "target_family": "train_base_pass16_success",
            "episode_id": episode_id,
            "split": "train",
            "partition": row.get("partition", partition_by_id.get(episode_id, {}).get("partition", "unknown")),
            "input": args.prompt_template.format(
                query=episode.get("query", ""),
                function_doc=json.dumps(episode.get("function_pool", []), ensure_ascii=False),
            ),
            "target_output": output_text(row["prediction"]),
            "function_name": row.get("function_name") or episode.get("ground_truth_call", {}).get("name", ""),
        }
        targets.append(target)

    r_success_ids = set(inventory["regression_sets"]["r_success_eval"])
    wrong_by_grid = {
        name: {episode_id for episode_id, row in records.items() if not row["success"]}
        for name, records in grid_success_by_name.items()
    }
    first_shared = wrong_by_grid.get("main_r8_lr2e-5_ep1", set()) & wrong_by_grid.get("star_r8_lr2e-5_ep1", set())
    ep2_union = wrong_by_grid.get("main_r8_lr2e-5_ep2", set()) | wrong_by_grid.get("star_r8_lr2e-5_ep2", set())
    for episode_id in sorted(r_success_ids):
        row = records_by_id[episode_id]
        correct_output = output_text(row["predicted_call"])
        target = {
            "target_id": f"r_success_correct:{episode_id}",
            "target_family": "r_success_correct",
            "episode_id": episode_id,
            "split": "r_success_eval",
            "partition": "r_success",
            "input": args.prompt_template.format(
                query=row.get("query", ""),
                function_doc=json.dumps(row.get("function_pool", []), ensure_ascii=False),
            ),
            "target_output": correct_output,
            "function_name": row.get("ground_truth_call", {}).get("name", ""),
            "first_shared_fragile": episode_id in first_shared,
            "ep2_union_fragile": episode_id in ep2_union,
        }
        target["flipped_outputs"] = {}
        for name, records in grid_success_by_name.items():
            pred = records[episode_id].get("prediction")
            if episode_id in wrong_by_grid.get(name, set()) and pred:
                target["flipped_outputs"][name] = output_text(pred)
        targets.append(target)
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description="Base teacher-forced log-probability probe.")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--s00-input", default="results/s00_inventory.json")
    parser.add_argument("--s01-input", default="results/s01_pass16_partition.json")
    parser.add_argument("--repaired-input", default="results/repaired_train_val.json")
    parser.add_argument("--pass16-success-input", default="data/pass16_success_trajectories.jsonl")
    parser.add_argument("--grid-json", action="append", default=list(DEFAULT_GRID_FILES))
    parser.add_argument("--output", default="results/s04_logprob_probe.json")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-qlora", action="store_true")
    args = parser.parse_args(argv)
    args.prompt_template = read_json(args.failures)["prompt_template"]
    args.prompt_template_key = False

    targets = build_targets(args)
    tokenizer = load_tokenizer(args.model_id)
    model = load_model_for_eval(args.model_id, qlora=not args.no_qlora)
    scored = []
    for index, target in enumerate(targets, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(targets):
            print(f"[logprob] target {index}/{len(targets)} {target['target_id']}", flush=True)
        row = {key: value for key, value in target.items() if key not in {"input", "target_output", "flipped_outputs"}}
        row["correct_score"] = score_output(model, tokenizer, target["input"], target["target_output"], args.max_length)
        if target.get("flipped_outputs"):
            row["flipped_scores"] = {
                name: score_output(model, tokenizer, target["input"], output, args.max_length)
                for name, output in target["flipped_outputs"].items()
            }
            row["margins"] = {
                name: row["correct_score"]["mean_logprob"] - score["mean_logprob"]
                for name, score in row["flipped_scores"].items()
                if row["correct_score"]["mean_logprob"] is not None and score["mean_logprob"] is not None
            }
        scored.append(row)

    payload = {
        "probe": "base_logprob_probe",
        "model_id": args.model_id,
        "target_count": len(scored),
        "records": scored,
        "summary_by_family": summarize(scored, "target_family"),
        "summary_by_partition": summarize(scored, "partition"),
    }
    write_json(args.output, payload)
    print(f"logprob targets={len(scored)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
