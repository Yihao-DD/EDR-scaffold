import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


DEFAULT_SPLIT_SEED = 20260630
DEFAULT_REGRESSION_SEED = 20260701
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_exp2(exp2_root):
    exp2_root = Path(exp2_root).resolve()
    if str(exp2_root) not in sys.path:
        sys.path.insert(0, str(exp2_root))
    from scripts import evolution_loop as evo

    return evo


def load_inputs(failures_path, exp2_result_path, exp2_root, split_seed):
    evo = load_exp2(exp2_root)
    baseline = read_json(failures_path)
    exp2_result = read_json(exp2_result_path)
    multiple_failures = [item for item in baseline["failures"] if item.get("split") == "multiple"]
    split = evo.deterministic_split(multiple_failures, seed=split_seed)
    seed_result = next((item for item in exp2_result["per_seed"] if item["seed"] == split_seed), None)
    if seed_result is None:
        raise ValueError(f"split seed {split_seed} not found in EXP2 result")
    patches = [evo.PatchCandidate(**item) for item in seed_result["loop_outputs"]["NL"]["accepted_patches"]]
    return baseline, exp2_result, split, patches, evo


def stable_int(*parts):
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def shard_items(items, shard_index, shard_count):
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [item for index, item in enumerate(items) if index % shard_count == shard_index]


def build_base_prompt(episode, prompt_template):
    return prompt_template.format(
        query=episode.get("query", ""),
        function_doc=json.dumps(episode.get("function_pool", []), ensure_ascii=False),
    )


def function_name(episode):
    return episode.get("ground_truth_call", {}).get("name", "")


def failure_type(episode):
    return episode.get("failure_class", "unknown")


def select_regression_sets(records, split, regression_seed=DEFAULT_REGRESSION_SEED, r_success_size=400):
    heldout_ids = {item["episode_id"] for item in split["held_out"]}
    multiple_success = [
        item
        for item in records
        if item.get("split") == "multiple" and item.get("call_success") is True and item["episode_id"] not in heldout_ids
    ]
    rng = random.Random(regression_seed)
    ordered = list(multiple_success)
    rng.shuffle(ordered)
    r_success = ordered[: min(r_success_size, len(ordered))]
    r_other = [item for item in records if item.get("split") == "simple"]
    return {
        "r_success_eval": [item["episode_id"] for item in r_success],
        "r_other": [item["episode_id"] for item in r_other],
        "r_heldout": sorted(heldout_ids),
        "r_success_pool_total": len(multiple_success),
        "r_success_eval_size": len(r_success),
        "r_other_size": len(r_other),
        "r_heldout_size": len(heldout_ids),
    }


def assert_no_leakage(distill_episode_ids, heldout_episode_ids, r_success_eval_ids):
    distill = set(distill_episode_ids)
    heldout = set(heldout_episode_ids)
    r_success = set(r_success_eval_ids)
    overlap_heldout = sorted(distill & heldout)
    overlap_success = sorted(distill & r_success)
    if overlap_heldout or overlap_success:
        raise AssertionError(
            {
                "distill_heldout_overlap": overlap_heldout[:20],
                "distill_r_success_overlap": overlap_success[:20],
                "n_distill_heldout_overlap": len(overlap_heldout),
                "n_distill_r_success_overlap": len(overlap_success),
            }
        )
    return True


def assert_no_selection_leakage(training_episode_ids, selection_episode_ids):
    training = set(training_episode_ids)
    selection = set(selection_episode_ids)
    overlap = sorted(training & selection)
    if overlap:
        raise AssertionError(
            {
                "training_selection_overlap": overlap[:20],
                "n_training_selection_overlap": len(overlap),
            }
        )
    return True


def partition_episode(pass16_hits, scaffold_success):
    if pass16_hits > 0:
        return "sampling_rescuable"
    if scaffold_success:
        return "scaffold_only"
    return "neither"


class Phase0Runner:
    def __init__(self, model_id, cache_dir=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = self.model.device

    def generate_many(
        self,
        prompt,
        n=1,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        seed=None,
    ):
        if seed is not None:
            self.torch.manual_seed(seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(seed)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
        with self.torch.no_grad():
            generated = self.model.generate(**inputs, **kwargs)
        new_tokens = generated[:, inputs.input_ids.shape[-1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def parse_prediction(evo, raw_text):
    try:
        prediction = evo.parse_model_output(raw_text)
        return prediction, None
    except Exception as error:
        return None, str(error)


def success_for_prediction(evo, prediction, episode):
    return evo.call_success(prediction or {"name": "", "arguments": {}}, episode["ground_truth_call"])


def evaluate_teacher_episode(evo, runner, episode, patches, max_new_tokens):
    relevant = evo.relevant_patches_for_episode(patches, episode)
    if not relevant:
        return {
            "episode_id": episode["episode_id"],
            "success": False,
            "prediction": None,
            "raw_model_output": "",
            "parse_error": "no_relevant_patch",
            "patch_ids": [],
            "n_patches": 0,
        }
    prompt = evo.build_prompt(episode, relevant)
    started = time.time()
    raw = runner.generate_many(prompt, n=1, do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens)[0]
    latency_ms = int((time.time() - started) * 1000)
    prediction, parse_error = parse_prediction(evo, raw)
    return {
        "episode_id": episode["episode_id"],
        "success": success_for_prediction(evo, prediction, episode),
        "prediction": prediction,
        "raw_model_output": raw,
        "parse_error": parse_error,
        "latency_ms": latency_ms,
        "patch_ids": [patch.patch_id for patch in relevant],
        "n_patches": len(relevant),
    }


def sample_base_episode(evo, runner, episode, prompt_template, sample_count, sample_batch_size, temperature, max_new_tokens, split_seed):
    prompt = build_base_prompt(episode, prompt_template)
    samples = []
    remaining = sample_count
    offset = 0
    started = time.time()
    while remaining:
        batch = min(sample_batch_size, remaining)
        seed = stable_int(split_seed, episode["episode_id"], "pass16", offset)
        raws = runner.generate_many(
            prompt,
            n=batch,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        for raw in raws:
            prediction, parse_error = parse_prediction(evo, raw)
            success = success_for_prediction(evo, prediction, episode)
            samples.append(
                {
                    "sample_index": offset,
                    "success": success,
                    "prediction": prediction,
                    "raw_model_output": raw,
                    "parse_error": parse_error,
                }
            )
            offset += 1
        remaining -= batch
    latency_ms = int((time.time() - started) * 1000)
    return {
        "episode_id": episode["episode_id"],
        "pass16_hits": sum(1 for sample in samples if sample["success"]),
        "sample_count": sample_count,
        "latency_ms": latency_ms,
        "successful_samples": [sample for sample in samples if sample["success"]],
    }


def sample_teacher_episode(evo, runner, episode, patches, sample_count, sample_batch_size, temperature, max_new_tokens, split_seed):
    relevant = evo.relevant_patches_for_episode(patches, episode)
    if not relevant:
        return {
            "episode_id": episode["episode_id"],
            "successful_samples": [],
            "sample_count": sample_count,
            "patch_ids": [],
            "error": "no_relevant_patch",
        }
    prompt = evo.build_prompt(episode, relevant)
    samples = []
    remaining = sample_count
    offset = 0
    started = time.time()
    while remaining:
        batch = min(sample_batch_size, remaining)
        seed = stable_int(split_seed, episode["episode_id"], "teacher", offset)
        raws = runner.generate_many(
            prompt,
            n=batch,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        for raw in raws:
            prediction, parse_error = parse_prediction(evo, raw)
            success = success_for_prediction(evo, prediction, episode)
            samples.append(
                {
                    "sample_index": offset,
                    "success": success,
                    "prediction": prediction,
                    "raw_model_output": raw,
                    "parse_error": parse_error,
                }
            )
            offset += 1
        remaining -= batch
    return {
        "episode_id": episode["episode_id"],
        "sample_count": sample_count,
        "temperature": temperature,
        "latency_ms": int((time.time() - started) * 1000),
        "patch_ids": [patch.patch_id for patch in relevant],
        "successful_samples": [sample for sample in samples if sample["success"]],
    }


def run_shard(args):
    baseline, exp2_result, split, patches, evo = load_inputs(
        args.failures,
        args.exp2_result,
        args.exp2_root,
        args.split_seed,
    )
    train_val = split["train"] + split["validation"]
    shard = shard_items(train_val, args.shard_index, args.shard_count)
    runner = Phase0Runner(args.model_id, cache_dir=args.model_cache_dir)
    records = []
    for index, episode in enumerate(shard, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(shard):
            print(
                f"[phase0-shard {args.shard_index}/{args.shard_count}] episode {index}/{len(shard)} "
                f"{episode['episode_id']}",
                flush=True,
            )
        teacher = evaluate_teacher_episode(evo, runner, episode, patches, args.max_new_tokens)
        pass16 = sample_base_episode(
            evo,
            runner,
            episode,
            baseline["prompt_template"],
            args.sample_count,
            args.sample_batch_size,
            args.temperature,
            args.max_new_tokens,
            args.split_seed,
        )
        records.append(
            {
                "episode_id": episode["episode_id"],
                "failure_class": failure_type(episode),
                "function_name": function_name(episode),
                "split_partition": "train" if episode in split["train"] else "validation",
                "teacher": teacher,
                "pass16": pass16,
                "partition": partition_episode(pass16["pass16_hits"], teacher["success"]),
            }
        )
    payload = {
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sample_count": args.sample_count,
        "temperature": args.temperature,
        "records": records,
    }
    write_json(args.output, payload)


def run_teacher_sample_shard(args):
    baseline, exp2_result, split, patches, evo = load_inputs(
        args.failures,
        args.exp2_result,
        args.exp2_root,
        args.split_seed,
    )
    train_val_by_id = {item["episode_id"]: item for item in split["train"] + split["validation"]}
    train_ids = {item["episode_id"] for item in split["train"]}
    repaired = read_json(args.repaired_input)
    repaired = [row for row in repaired if row["episode_id"] in train_val_by_id]
    if args.train_only:
        repaired = [row for row in repaired if row["episode_id"] in train_ids]
    shard = shard_items(repaired, args.shard_index, args.shard_count)
    runner = Phase0Runner(args.model_id, cache_dir=args.model_cache_dir)
    records = []
    for index, row in enumerate(shard, start=1):
        episode = train_val_by_id[row["episode_id"]]
        if index == 1 or index % args.progress_every == 0 or index == len(shard):
            print(
                f"[teacher-sample {args.shard_index}/{args.shard_count}] episode {index}/{len(shard)} "
                f"{episode['episode_id']}",
                flush=True,
            )
        sample = sample_teacher_episode(
            evo,
            runner,
            episode,
            patches,
            args.sample_count,
            args.sample_batch_size,
            args.temperature,
            args.max_new_tokens,
            args.split_seed,
        )
        sample.update(
            {
                "failure_class": failure_type(episode),
                "function_name": function_name(episode),
                "partition": row.get("partition", "unknown"),
            }
        )
        records.append(sample)
    write_json(
        args.output,
        {
            "model_id": args.model_id,
            "split_seed": args.split_seed,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "sample_count": args.sample_count,
            "temperature": args.temperature,
            "records": records,
        },
    )


def canonical_call(prediction):
    return {
        "name": prediction.get("name", ""),
        "arguments": prediction.get("arguments") or {},
    }


def call_key(prediction):
    return json.dumps(canonical_call(prediction), ensure_ascii=False, sort_keys=True)


def output_text(prediction):
    return json.dumps(canonical_call(prediction), ensure_ascii=False, sort_keys=True)


def make_distill_row(episode, prompt_template, prediction, source, partition, sample_index=None):
    return {
        "episode_id": episode["episode_id"],
        "source": source,
        "partition": partition,
        "failure_class": failure_type(episode),
        "function_name": function_name(episode),
        "sample_index": sample_index,
        "input": build_base_prompt(episode, prompt_template),
        "output": output_text(prediction),
        "output_call": canonical_call(prediction),
    }


def dedupe_rows(rows):
    result = []
    seen = set()
    for row in rows:
        key = (row["episode_id"], row["source"], row["output"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def deterministic_sample(rows, count, seed, label):
    if count >= len(rows):
        return list(rows)
    keyed = []
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{label}:{row['episode_id']}:{row.get('source')}:{row.get('output')}".encode("utf-8")).hexdigest()
        keyed.append((digest, row))
    return [row for _, row in sorted(keyed, key=lambda item: item[0])[:count]]


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset_log(path, summary):
    lines = [
        "# Step 0.2 Dataset Construction",
        "",
        f"PROBE signal (main_n={summary['distill_main_count']}, star_n={summary['distill_star_count']}).",
        "",
        f"- teacher_core_available: {summary['teacher_core_available']}",
        f"- teacher_core_selected: {summary['teacher_core_selected']}",
        f"- star_core_available: {summary['star_core_available']}",
        f"- star_core_selected: {summary['star_core_selected']}",
        f"- replay_ratio_target: {summary['replay_ratio_target']}",
        f"- main_replay_count: {summary['main_replay_count']}",
        f"- star_replay_count: {summary['star_replay_count']}",
        f"- dataset_size_ratio_star_over_main: {summary['dataset_size_ratio_star_over_main']:.4f}",
        f"- leakage assertions: {summary['leakage_asserts']}",
        "",
        "## Main Partition Distribution",
        "",
    ]
    for key, value in summary["main_partition_distribution"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## STaR Partition Distribution", ""])
    for key, value in summary["star_partition_distribution"].items():
        lines.append(f"- {key}: {value}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_datasets(args):
    baseline, exp2_result, split, patches, evo = load_inputs(
        args.failures,
        args.exp2_result,
        args.exp2_root,
        args.split_seed,
    )
    records_by_id = {item["episode_id"]: item for item in baseline["records"]}
    train_ids = {item["episode_id"] for item in split["train"]}
    val_ids = {item["episode_id"] for item in split["validation"]}
    train_val_by_id = {item["episode_id"]: item for item in split["train"] + split["validation"]}
    repaired = read_json(args.repaired_input)
    partition = read_json(args.s01_input)
    partition_by_id = {row["episode_id"]: row["partition"] for row in partition["episodes"]}
    inventory = read_json(args.s00_input)
    regression_sets = inventory["regression_sets"]
    heldout_ids = set(regression_sets["r_heldout"])
    r_success_eval_ids = set(regression_sets["r_success_eval"])

    teacher_rows = []
    for row in repaired:
        if args.train_only and row["episode_id"] not in train_ids:
            continue
        episode = train_val_by_id[row["episode_id"]]
        teacher_rows.append(
            make_distill_row(
                episode,
                baseline["prompt_template"],
                row["teacher_prediction"],
                "teacher_t0",
                row.get("partition", partition_by_id.get(row["episode_id"], "unknown")),
            )
        )
    for shard_path in sorted(Path(args.teacher_sample_dir).glob(args.teacher_sample_glob)):
        payload = read_json(shard_path)
        for sample_record in payload["records"]:
            if args.train_only and sample_record["episode_id"] not in train_ids:
                continue
            episode = train_val_by_id[sample_record["episode_id"]]
            for sample in sample_record["successful_samples"]:
                teacher_rows.append(
                    make_distill_row(
                        episode,
                        baseline["prompt_template"],
                        sample["prediction"],
                        "teacher_t08",
                        sample_record.get("partition", partition_by_id.get(sample_record["episode_id"], "unknown")),
                        sample.get("sample_index"),
                    )
                )
    teacher_rows = dedupe_rows(teacher_rows)
    star_rows = []
    for row in load_jsonl(args.pass16_success_input):
        if args.train_only and row["episode_id"] not in train_ids:
            continue
        episode = train_val_by_id[row["episode_id"]]
        star_rows.append(
            make_distill_row(
                episode,
                baseline["prompt_template"],
                row["prediction"],
                "star_pass16",
                row.get("partition", partition_by_id.get(row["episode_id"], "unknown")),
                row.get("sample_index"),
            )
        )
    # STaR uses every successful pass@16 trajectory. Keep duplicate outputs because
    # the probe explicitly treats sampled successful trajectories as the STaR data.

    max_main_core_for_balance = int(len(star_rows) / 0.9) if star_rows else len(teacher_rows)
    main_core_target = min(args.max_teacher_core, len(teacher_rows), max_main_core_for_balance)
    projected_total = round(main_core_target * (1.0 + args.replay_ratio))
    if projected_total < args.min_total and len(teacher_rows) > main_core_target and not star_rows:
        needed_core = math.ceil(args.min_total / (1.0 + args.replay_ratio))
        main_core_target = min(len(teacher_rows), args.max_teacher_core, needed_core)
    main_core = deterministic_sample(teacher_rows, main_core_target, args.dataset_seed, "main_core")
    star_core_target = min(len(star_rows), max(1, round(len(main_core) * args.star_core_ratio)))
    star_core = deterministic_sample(star_rows, star_core_target, args.dataset_seed, "star_core")

    assert_no_leakage([row["episode_id"] for row in main_core], heldout_ids, r_success_eval_ids)
    assert_no_leakage([row["episode_id"] for row in star_core], heldout_ids, r_success_eval_ids)
    assert_no_selection_leakage([row["episode_id"] for row in main_core], val_ids)
    assert_no_selection_leakage([row["episode_id"] for row in star_core], val_ids)

    replay_pool = [
        item
        for item in baseline["records"]
        if item.get("split") == "multiple"
        and item.get("call_success") is True
        and item["episode_id"] not in heldout_ids
        and item["episode_id"] not in r_success_eval_ids
    ]
    replay_rows = [
        make_distill_row(
            item,
            baseline["prompt_template"],
            item["predicted_call"],
            "replay_base_success",
            "replay",
        )
        for item in replay_pool
    ]
    replay_rows = dedupe_rows(replay_rows)
    main_replay_count = min(len(replay_rows), round(len(main_core) * args.replay_ratio))
    star_replay_count = min(len(replay_rows), round(len(star_core) * args.replay_ratio))
    main_replay = deterministic_sample(replay_rows, main_replay_count, args.dataset_seed, "main_replay")
    star_replay = deterministic_sample(replay_rows, star_replay_count, args.dataset_seed, "star_replay")

    distill_main = main_core + main_replay
    distill_star = star_core + star_replay
    assert_no_leakage([row["episode_id"] for row in distill_main], heldout_ids, r_success_eval_ids)
    assert_no_leakage([row["episode_id"] for row in distill_star], heldout_ids, r_success_eval_ids)
    assert_no_selection_leakage([row["episode_id"] for row in distill_main], val_ids)
    assert_no_selection_leakage([row["episode_id"] for row in distill_star], val_ids)

    write_jsonl(args.main_output, distill_main)
    write_jsonl(args.star_output, distill_star)
    summary = {
        "probe": "s02_datasets",
        "split_seed": args.split_seed,
        "dataset_seed": args.dataset_seed,
        "teacher_core_available": len(teacher_rows),
        "teacher_core_selected": len(main_core),
        "star_core_available": len(star_rows),
        "star_core_selected": len(star_core),
        "replay_pool_available": len(replay_rows),
        "replay_ratio_target": args.replay_ratio,
        "main_replay_count": len(main_replay),
        "star_replay_count": len(star_replay),
        "distill_main_count": len(distill_main),
        "distill_star_count": len(distill_star),
        "dataset_size_ratio_star_over_main": len(distill_star) / len(distill_main) if distill_main else 0.0,
        "main_partition_distribution": summarize_counter(distill_main, "partition"),
        "star_partition_distribution": summarize_counter(distill_star, "partition"),
        "main_function_distribution": summarize_counter(distill_main, "function_name"),
        "star_function_distribution": summarize_counter(distill_star, "function_name"),
        "leakage_asserts": {
            "main_intersect_heldout": 0,
            "main_intersect_r_success_eval": 0,
            "main_intersect_d_val": 0,
            "star_intersect_heldout": 0,
            "star_intersect_r_success_eval": 0,
            "star_intersect_d_val": 0,
        },
    }
    write_json(args.summary_output, summary)
    build_dataset_log(args.log_output, summary)
    print(
        f"s02 main={summary['distill_main_count']} star={summary['distill_star_count']} "
        f"teacher_core={summary['teacher_core_selected']}/{summary['teacher_core_available']} "
        f"star_core={summary['star_core_selected']}/{summary['star_core_available']}"
    )


def summarize_counter(rows, key):
    return dict(sorted(Counter(row.get(key, "") for row in rows).items()))


def merge_shard_payloads(shard_payloads, expected_episode_ids):
    by_id = {}
    for payload in shard_payloads:
        for row in payload["records"]:
            episode_id = row["episode_id"]
            if episode_id in by_id:
                raise ValueError(f"duplicate episode in shards: {episode_id}")
            by_id[episode_id] = row
    missing = sorted(set(expected_episode_ids) - set(by_id))
    extra = sorted(set(by_id) - set(expected_episode_ids))
    if missing or extra:
        raise ValueError({"missing": missing[:20], "extra": extra[:20], "n_missing": len(missing), "n_extra": len(extra)})
    return [by_id[episode_id] for episode_id in expected_episode_ids]


def write_inventory_log(path, inventory):
    lines = [
        "# Step 0.0 Inventory",
        "",
        f"PROBE signal (n={inventory['train_val_failure_count']}).",
        "",
        f"- split_seed: {inventory['split_seed']}",
        f"- train+validation failures: {inventory['train_val_failure_count']}",
        f"- repaired_train_val: {inventory['repaired_count']}",
        f"- gate: {inventory['gate']}",
        f"- R-success eval: {inventory['regression_sets']['r_success_eval_size']}",
        f"- R-other simple episodes: {inventory['regression_sets']['r_other_size']}",
        f"- R-heldout failures: {inventory['regression_sets']['r_heldout_size']}",
        f"- leakage assertions: {inventory['leakage_asserts']}",
        "",
        "## Repaired Function Distribution",
        "",
    ]
    for name, count in inventory["repaired_function_distribution"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Repaired Failure-Type Distribution", ""])
    for name, count in inventory["repaired_failure_type_distribution"].items():
        lines.append(f"- {name}: {count}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_partition_log(path, partition):
    stats = partition["stats"]
    lines = [
        "# Step 0.1 pass@16 Partition",
        "",
        f"PROBE signal (n={stats['total_failures']}). Base model sampled {partition['sample_count']} trajectories per failure at T={partition['temperature']}.",
        "",
        f"- sampling_rescuable: {stats['sampling_rescuable']} ({stats['sampling_rescuable_rate']:.4f})",
        f"- scaffold_only: {stats['scaffold_only']} ({stats['scaffold_only_rate']:.4f})",
        f"- neither: {stats['neither']} ({stats['neither_rate']:.4f})",
        f"- repaired_count: {stats['repaired_count']}",
        f"- scaffold_only / repaired_count: {stats['scaffold_only_repaired_rate']:.4f}",
        f"- sampling_rescuable / repaired_count: {stats['sampling_rescuable_repaired_rate']:.4f}",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_shards(args):
    baseline, exp2_result, split, patches, evo = load_inputs(
        args.failures,
        args.exp2_result,
        args.exp2_root,
        args.split_seed,
    )
    train_val = split["train"] + split["validation"]
    expected_ids = [item["episode_id"] for item in train_val]
    shard_payloads = [read_json(path) for path in sorted(Path(args.shard_dir).glob(args.shard_glob))]
    if not shard_payloads:
        raise ValueError(f"no shard files matched {args.shard_glob} in {args.shard_dir}")
    records = merge_shard_payloads(shard_payloads, expected_ids)

    repaired = [row for row in records if row["teacher"]["success"]]
    regression_sets = select_regression_sets(baseline["records"], split, args.regression_seed, args.r_success_size)
    assert_no_leakage(
        [row["episode_id"] for row in repaired],
        regression_sets["r_heldout"],
        regression_sets["r_success_eval"],
    )
    inventory = {
        "probe": "s00_inventory",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "exp2_reference_scaffold_on_mean": exp2_result["summary"]["NL-evo"]["overall"]["mean"],
        "train_val_failure_count": len(records),
        "repaired_count": len(repaired),
        "gate": "GO" if len(repaired) >= 50 else "STOP_LT_50",
        "repaired_episode_ids": [row["episode_id"] for row in repaired],
        "repaired_function_distribution": summarize_counter(repaired, "function_name"),
        "repaired_failure_type_distribution": summarize_counter(repaired, "failure_class"),
        "regression_sets": regression_sets,
        "leakage_asserts": {
            "distill_intersect_heldout": 0,
            "distill_intersect_r_success_eval": 0,
        },
    }
    partition_counts = Counter(row["partition"] for row in records)
    repaired_ids = {row["episode_id"] for row in repaired}
    sampling_repaired = [row for row in records if row["episode_id"] in repaired_ids and row["partition"] == "sampling_rescuable"]
    scaffold_only = [row for row in records if row["partition"] == "scaffold_only"]
    stats = {
        "total_failures": len(records),
        "sampling_rescuable": partition_counts["sampling_rescuable"],
        "scaffold_only": partition_counts["scaffold_only"],
        "neither": partition_counts["neither"],
        "sampling_rescuable_rate": partition_counts["sampling_rescuable"] / len(records) if records else 0.0,
        "scaffold_only_rate": partition_counts["scaffold_only"] / len(records) if records else 0.0,
        "neither_rate": partition_counts["neither"] / len(records) if records else 0.0,
        "repaired_count": len(repaired),
        "scaffold_only_repaired_rate": len(scaffold_only) / len(repaired) if repaired else 0.0,
        "sampling_rescuable_repaired_rate": len(sampling_repaired) / len(repaired) if repaired else 0.0,
    }
    partition = {
        "probe": "s01_pass16_partition",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "sample_count": shard_payloads[0].get("sample_count", args.sample_count),
        "temperature": shard_payloads[0].get("temperature", args.temperature),
        "stats": stats,
        "episodes": [
            {
                "episode_id": row["episode_id"],
                "failure_class": row["failure_class"],
                "function_name": row["function_name"],
                "teacher_success": row["teacher"]["success"],
                "pass16_hits": row["pass16"]["pass16_hits"],
                "partition": row["partition"],
            }
            for row in records
        ],
    }
    repaired_rows = [
        {
            "episode_id": row["episode_id"],
            "failure_class": row["failure_class"],
            "function_name": row["function_name"],
            "partition": row["partition"],
            "teacher_prediction": row["teacher"]["prediction"],
            "teacher_raw_model_output": row["teacher"]["raw_model_output"],
            "patch_ids": row["teacher"]["patch_ids"],
        }
        for row in repaired
    ]
    pass16_success_rows = []
    for row in records:
        for sample in row["pass16"]["successful_samples"]:
            pass16_success_rows.append(
                {
                    "episode_id": row["episode_id"],
                    "failure_class": row["failure_class"],
                    "function_name": row["function_name"],
                    "partition": row["partition"],
                    "sample_index": sample["sample_index"],
                    "prediction": sample["prediction"],
                    "raw_model_output": sample["raw_model_output"],
                }
            )

    write_json(args.s00_output, inventory)
    write_json(args.s01_output, partition)
    write_json(args.repaired_output, repaired_rows)
    write_jsonl(args.pass16_success_output, pass16_success_rows)
    write_inventory_log(args.s00_log, inventory)
    write_partition_log(args.s01_log, partition)
    print(f"s00 repaired_count={inventory['repaired_count']} gate={inventory['gate']}")
    print(f"s01 scaffold_only={stats['scaffold_only']} sampling_rescuable={stats['sampling_rescuable']}")


def prepare(args):
    baseline, exp2_result, split, patches, evo = load_inputs(
        args.failures,
        args.exp2_result,
        args.exp2_root,
        args.split_seed,
    )
    train_val = split["train"] + split["validation"]
    regression_sets = select_regression_sets(baseline["records"], split, args.regression_seed, args.r_success_size)
    manifest = {
        "probe": "phase0_manifest",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "split_sizes": {key: len(value) for key, value in split.items()},
        "train_val_failure_count": len(train_val),
        "nl_evo_accepted_patches": len(patches),
        "exp2_reference_scaffold_on_mean": exp2_result["summary"]["NL-evo"]["overall"]["mean"],
        "regression_sets": regression_sets,
    }
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="EXP3 Phase 0 inventory and pass@16 probe")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-result", default="../EDG-EXP2-struct/results/evolution_main_merged.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--regression-seed", type=int, default=DEFAULT_REGRESSION_SEED)
    parser.add_argument("--r-success-size", type=int, default=400)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", default="results/phase0_manifest.json")
    prepare_parser.set_defaults(func=prepare)

    shard_parser = subparsers.add_parser("run-shard")
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--shard-count", type=int, required=True)
    shard_parser.add_argument("--output", required=True)
    shard_parser.add_argument("--sample-count", type=int, default=16)
    shard_parser.add_argument("--sample-batch-size", type=int, default=4)
    shard_parser.add_argument("--temperature", type=float, default=0.8)
    shard_parser.add_argument("--max-new-tokens", type=int, default=256)
    shard_parser.add_argument("--progress-every", type=int, default=10)
    shard_parser.set_defaults(func=run_shard)

    teacher_parser = subparsers.add_parser("sample-teacher-shard")
    teacher_parser.add_argument("--repaired-input", default="results/repaired_train_val.json")
    teacher_parser.add_argument("--shard-index", type=int, required=True)
    teacher_parser.add_argument("--shard-count", type=int, required=True)
    teacher_parser.add_argument("--output", required=True)
    teacher_parser.add_argument("--sample-count", type=int, default=4)
    teacher_parser.add_argument("--sample-batch-size", type=int, default=4)
    teacher_parser.add_argument("--temperature", type=float, default=0.8)
    teacher_parser.add_argument("--max-new-tokens", type=int, default=256)
    teacher_parser.add_argument("--progress-every", type=int, default=10)
    teacher_parser.add_argument(
        "--allow-val-sampling",
        action="store_false",
        dest="train_only",
        help="Legacy escape hatch only: sample repaired D_val episodes.",
    )
    teacher_parser.set_defaults(train_only=True)
    teacher_parser.set_defaults(func=run_teacher_sample_shard)

    merge_parser = subparsers.add_parser("merge-shards")
    merge_parser.add_argument("--shard-dir", default="results/phase0_shards")
    merge_parser.add_argument("--shard-glob", default="shard_*.json")
    merge_parser.add_argument("--sample-count", type=int, default=16)
    merge_parser.add_argument("--temperature", type=float, default=0.8)
    merge_parser.add_argument("--s00-output", default="results/s00_inventory.json")
    merge_parser.add_argument("--s01-output", default="results/s01_pass16_partition.json")
    merge_parser.add_argument("--repaired-output", default="results/repaired_train_val.json")
    merge_parser.add_argument("--pass16-success-output", default="data/pass16_success_trajectories.jsonl")
    merge_parser.add_argument("--s00-log", default="logs/s00.md")
    merge_parser.add_argument("--s01-log", default="logs/s01.md")
    merge_parser.set_defaults(func=merge_shards)

    dataset_parser = subparsers.add_parser("build-datasets")
    dataset_parser.add_argument("--repaired-input", default="results/repaired_train_val.json")
    dataset_parser.add_argument("--s00-input", default="results/s00_inventory.json")
    dataset_parser.add_argument("--s01-input", default="results/s01_pass16_partition.json")
    dataset_parser.add_argument("--pass16-success-input", default="data/pass16_success_trajectories.jsonl")
    dataset_parser.add_argument("--teacher-sample-dir", default="results/s02_teacher_samples")
    dataset_parser.add_argument("--teacher-sample-glob", default="teacher_shard_*.json")
    dataset_parser.add_argument("--main-output", default="data/distill_main.jsonl")
    dataset_parser.add_argument("--star-output", default="data/distill_star.jsonl")
    dataset_parser.add_argument("--summary-output", default="results/s02_datasets.json")
    dataset_parser.add_argument("--log-output", default="logs/s02.md")
    dataset_parser.add_argument("--dataset-seed", type=int, default=20260702)
    dataset_parser.add_argument("--replay-ratio", type=float, default=0.4)
    dataset_parser.add_argument("--star-core-ratio", type=float, default=1.0)
    dataset_parser.add_argument("--min-total", type=int, default=300)
    dataset_parser.add_argument("--max-teacher-core", type=int, default=500)
    dataset_parser.add_argument(
        "--allow-val-training",
        action="store_false",
        dest="train_only",
        help="Legacy escape hatch only: allow D_val episodes into training data.",
    )
    dataset_parser.set_defaults(train_only=True)
    dataset_parser.set_defaults(func=build_datasets)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
