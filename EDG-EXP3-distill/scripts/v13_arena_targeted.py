import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SPLIT_SEED = 20260630
DEFAULT_DATASET_SEED = 20260702
TARGET_REPLAY_ROWS = 296
CAPPED_REPLAY_UNIQUE = 120


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def function_name(record):
    gt = record.get("ground_truth_call") or {}
    return gt.get("name", "")


def coarse_family(name):
    match = re.match(r"^([^_]+)_\d+_", name)
    if match:
        return match.group(1)
    if "." in name:
        return name.split(".", 1)[0]
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def deterministic_order(rows, seed, label):
    keyed = []
    for index, row in enumerate(rows):
        digest = hashlib.sha256(
            f"{seed}:{label}:{row.get('episode_id')}:{row.get('source')}:{row.get('output')}:{index}".encode("utf-8")
        ).hexdigest()
        keyed.append((digest, index, row))
    return [row for _, _, row in sorted(keyed)]


def take_with_repeats(rows, count, seed, label):
    if not rows:
        return []
    ordered = deterministic_order(rows, seed, label)
    selected = []
    round_index = 0
    while len(selected) < count:
        for row in deterministic_order(ordered, seed + round_index, f"{label}:round"):
            selected.append(dict(row))
            if len(selected) == count:
                break
        round_index += 1
    return selected


def load_split(exp2_root, failures_path, split_seed):
    exp2_root = Path(exp2_root).resolve()
    if str(exp2_root) not in sys.path:
        sys.path.insert(0, str(exp2_root))
    from scripts import evolution_loop as evo

    baseline = read_json(failures_path)
    multiple_failures = [item for item in baseline["failures"] if item.get("split") == "multiple"]
    return baseline, evo.deterministic_split(multiple_failures, seed=split_seed)


def collect_history_touches(root):
    root = Path(root)
    touched = defaultdict(set)
    s00 = read_json(root / "results/s00_inventory.json")
    for key, value in s00["regression_sets"].items():
        if isinstance(value, list):
            touched[f"eval:{key}"].update(value)
    for path in [root / "results/s01_pass16_partition.json", root / "results/s01_heldout_pass16_partition.json"]:
        if path.exists():
            payload = read_json(path)
            touched[f"eval:{path.stem}"].update(row["episode_id"] for row in payload.get("episodes", []))
    for phase in ["s03_grid_clean", "s04_upgrade", "s05_blocked_kl"]:
        for path in (root / f"results/{phase}").glob("*/*.json"):
            payload = read_json(path)
            touched[f"eval:{phase}:val"].update(row["episode_id"] for row in payload.get("val_records", []))
            touched[f"eval:{phase}:r_success"].update(row["episode_id"] for row in payload.get("r_success_records", []))
    for path in (root / "data").glob("*.jsonl"):
        if path.name.startswith("distill_") or path.name.startswith("pass16_"):
            for row in read_jsonl(path):
                touched[f"data:{path.name}"].add(row["episode_id"])
    return touched


def current_eval_parseable_ground_truth(ground_truth):
    if not isinstance(ground_truth, list) or len(ground_truth) != 1:
        return False
    item = ground_truth[0]
    if isinstance(item, dict) and len(item) == 1:
        arguments = next(iter(item.values()))
        return isinstance(arguments, dict)
    if isinstance(item, str):
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*\s*\(.*\)$", item.strip(), re.S))
    return False


def raw_category_counts(raw_root, recorded_episode_ids=None):
    raw_root = Path(raw_root)
    answer_root = raw_root / "possible_answer"
    recorded_episode_ids = recorded_episode_ids or set()
    rows = []
    for path in sorted(raw_root.glob("BFCL_v4_*.json")):
        category = path.stem.replace("BFCL_v4_", "")
        answer_path = answer_root / path.name
        if not answer_path.exists():
            continue
        question_ids = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    question_ids.append(json.loads(line)["id"])
        answers = []
        with answer_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    answers.append(json.loads(line))
        q_count = len(question_ids)
        a_count = len(answers)
        first_gt = answers[0].get("ground_truth") if answers else None
        required_calls = len(first_gt) if isinstance(first_gt, list) else 1
        parseable = sum(current_eval_parseable_ground_truth(row.get("ground_truth")) for row in answers)
        unrecorded_ids = set(question_ids) - set(recorded_episode_ids)
        rows.append(
            {
                "category": category,
                "question_rows": q_count,
                "answer_rows": a_count,
                "ground_truth_calls_per_episode_first_row": required_calls,
                "current_eval_parseable_rows": parseable,
                "single_call_compatible_with_current_eval": required_calls == 1 and parseable == a_count,
                "unrecorded_raw_episode_rows": len(unrecorded_ids),
            }
        )
    return rows


def category_touch_audit(args):
    baseline, split = load_split(args.exp2_root, args.failures, args.split_seed)
    by_record_id = {row["episode_id"]: row for row in baseline["records"]}
    touched = collect_history_touches(args.root)
    all_touched = set().union(*touched.values()) if touched else set()
    category_rows = []
    for category in sorted({row.get("category") for row in baseline["records"]}):
        rows = [row for row in baseline["records"] if row.get("category") == category]
        ids = {row["episode_id"] for row in rows}
        untouched = ids - all_touched
        category_rows.append(
            {
                "category": category,
                "recorded_rows": len(rows),
                "recorded_successes": sum(row.get("call_success") is True for row in rows),
                "touched_any": len(ids & all_touched),
                "untouched_recorded_rows": len(untouched),
                "untouched_recorded_successes": sum(
                    by_record_id[episode_id].get("call_success") is True for episode_id in untouched
                ),
            }
        )

    recorded_episode_ids = {row["episode_id"] for row in baseline["records"]}
    raw_rows = raw_category_counts(args.raw_root, recorded_episode_ids=recorded_episode_ids)
    recorded_categories = {row["category"] for row in category_rows}
    unrecorded_raw = [row for row in raw_rows if row["category"] not in recorded_categories]
    sibling_single_call = [
        row
        for row in unrecorded_raw
        if row["single_call_compatible_with_current_eval"]
        and row["category"] not in {"format_sensitivity"}
    ]
    sibling_multi_call = [
        row
        for row in unrecorded_raw
        if not row["single_call_compatible_with_current_eval"]
    ]
    total_single_call_rows = sum(row["question_rows"] for row in sibling_single_call)
    payload = {
        "probe": "v13_category_touch_audit",
        "split_seed": args.split_seed,
        "current_recorded_categories": category_rows,
        "history_touch_sets": {key: len(value) for key, value in sorted(touched.items())},
        "raw_categories": raw_rows,
        "unrecorded_single_call_compatible_categories": sibling_single_call,
        "unrecorded_multi_call_categories": sibling_multi_call,
        "fresh_arena_primary_status": (
            "needs_base_t0_forward_on_unrecorded_single_call_categories"
            if total_single_call_rows >= args.min_fresh_candidates
            else "insufficient_single_call_unrecorded_categories"
        ),
        "unrecorded_single_call_total_rows": total_single_call_rows,
        "strict_recorded_pool_available_successes": sum(row["untouched_recorded_successes"] for row in category_rows),
        "strict_recorded_pool_note": (
            "All categories already present in EXP1 baseline records have zero untouched successes after excluding historical eval/training/replay."
        ),
        "raw_unrecorded_parseable_rows": sum(
            row["unrecorded_raw_episode_rows"]
            for row in raw_rows
            if row["single_call_compatible_with_current_eval"]
        ),
    }
    write_json(args.output, payload)
    return payload


def build_replay_row_from_record(record, prompt_template):
    return {
        "episode_id": record["episode_id"],
        "source": "replay_base_success",
        "partition": "replay",
        "failure_class": record.get("failure_class", "none"),
        "function_name": function_name(record),
        "sample_index": None,
        "input": prompt_template.format(
            query=record.get("query", ""),
            function_doc=json.dumps(record.get("function_pool", []), ensure_ascii=False),
        ),
        "output": json.dumps(record["predicted_call"], ensure_ascii=False, sort_keys=True),
        "output_call": record["predicted_call"],
    }


def assert_disjoint(label, left, right):
    overlap = sorted(set(left) & set(right))
    if overlap:
        raise AssertionError({label: overlap[:20], f"n_{label}": len(overlap)})


def build_targeted(args):
    baseline, split = load_split(args.exp2_root, args.failures, args.split_seed)
    s00 = read_json(Path(args.root) / "results/s00_inventory.json")
    train_failure_functions = {function_name(row) for row in split["train"]}
    train_failure_families = {coarse_family(name) for name in train_failure_functions}
    exclude = set(s00["regression_sets"]["r_success_eval"]) | set(s00["regression_sets"]["r_heldout"])
    val_ids = {row["episode_id"] for row in split["validation"]}
    exclude |= val_ids
    candidates = [
        row
        for row in baseline["records"]
        if row.get("split") == "multiple"
        and row.get("call_success") is True
        and row["episode_id"] not in exclude
    ]
    replay_rows = [build_replay_row_from_record(row, baseline["prompt_template"]) for row in candidates]
    exact = [row for row in replay_rows if row["function_name"] in train_failure_functions]
    exact_ids = {row["episode_id"] for row in exact}
    coarse = [
        row
        for row in replay_rows
        if row["episode_id"] not in exact_ids and coarse_family(row["function_name"]) in train_failure_families
    ]
    used_ids = exact_ids | {row["episode_id"] for row in coarse}
    uniform = [row for row in replay_rows if row["episode_id"] not in used_ids]
    selected = []
    stages = []
    for label, rows in [("exact_train_failure_function", exact), ("coarse_train_failure_family", coarse), ("uniform_remainder", uniform)]:
        needed = TARGET_REPLAY_ROWS - len(selected)
        if needed <= 0:
            break
        take = take_with_repeats(rows, needed, args.dataset_seed, label)
        selected.extend(take)
        stages.append(
            {
                "stage": label,
                "available_unique_rows": len(rows),
                "selected_rows": len(take),
                "selected_unique_episodes": len({row["episode_id"] for row in take}),
            }
        )
    if len(selected) != TARGET_REPLAY_ROWS:
        raise ValueError({"target_replay_rows": TARGET_REPLAY_ROWS, "selected": len(selected)})

    core_paths = {
        "main": Path(args.root) / "data/distill_main.jsonl",
        "star": Path(args.root) / "data/distill_star.jsonl",
    }
    outputs = {}
    heldout = set(s00["regression_sets"]["r_heldout"])
    r_success = set(s00["regression_sets"]["r_success_eval"])
    for arm, core_path in core_paths.items():
        core = [row for row in read_jsonl(core_path) if row.get("partition") != "replay"]
        data = core + selected
        ids = [row["episode_id"] for row in data]
        assert_disjoint(f"{arm}_intersect_heldout", ids, heldout)
        assert_disjoint(f"{arm}_intersect_r_success_eval", ids, r_success)
        assert_disjoint(f"{arm}_intersect_d_val", ids, val_ids)
        out_path = Path(args.root) / f"data/distill_{arm}_targeted_replay2.jsonl"
        write_jsonl(out_path, data)
        outputs[arm] = {
            "path": str(out_path),
            "core_rows": len(core),
            "targeted_replay_rows": len(selected),
            "total_rows": len(data),
            "unique_episode_count": len(set(ids)),
            "targeted_replay_unique_episode_count": len({row["episode_id"] for row in selected}),
            "source_distribution": dict(sorted(Counter(row.get("source") for row in data).items())),
            "function_distribution_top20": Counter(row.get("function_name") for row in selected).most_common(20),
        }
    summary = {
        "probe": "v13_targeted_replay2",
        "dataset_seed": args.dataset_seed,
        "target_replay_rows_per_arm": TARGET_REPLAY_ROWS,
        "train_failure_exact_function_count": len(train_failure_functions),
        "train_failure_coarse_family_count": len(train_failure_families),
        "candidate_unique_replay_rows": len(replay_rows),
        "selection_stages": stages,
        "selection_stage_row_share": {
            stage["stage"]: stage["selected_rows"] / TARGET_REPLAY_ROWS for stage in stages
        },
        "outputs": outputs,
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
    return summary


def replay_candidates(args):
    baseline, split = load_split(args.exp2_root, args.failures, args.split_seed)
    s00 = read_json(Path(args.root) / "results/s00_inventory.json")
    exclude = set(s00["regression_sets"]["r_success_eval"]) | set(s00["regression_sets"]["r_heldout"])
    val_ids = {row["episode_id"] for row in split["validation"]}
    exclude |= val_ids
    candidates = [
        row
        for row in baseline["records"]
        if row.get("split") == "multiple"
        and row.get("call_success") is True
        and row["episode_id"] not in exclude
    ]
    replay_rows = [build_replay_row_from_record(row, baseline["prompt_template"]) for row in candidates]
    return baseline, split, s00, val_ids, candidates, replay_rows


def write_capped_dataset(args, arm, core, replay_rows, suffix):
    data = core + replay_rows
    out_path = Path(args.root) / f"data/distill_{arm}_{suffix}.jsonl"
    write_jsonl(out_path, data)
    return {
        "path": str(out_path),
        "core_rows": len(core),
        "replay_rows": len(replay_rows),
        "total_rows": len(data),
        "core_unique_episode_count": len({row["episode_id"] for row in core}),
        "replay_unique_episode_count": len({row["episode_id"] for row in replay_rows}),
        "unique_episode_count": len({row["episode_id"] for row in data}),
        "source_distribution": dict(sorted(Counter(row.get("source") for row in data).items())),
    }


def build_capped(args):
    baseline, split, s00, val_ids, candidates, replay_rows = replay_candidates(args)
    if len(replay_rows) < CAPPED_REPLAY_UNIQUE:
        raise ValueError({"available_replay_unique": len(replay_rows), "required": CAPPED_REPLAY_UNIQUE})
    ordered = deterministic_order(replay_rows, args.dataset_seed, "capped_arena_split")
    arena_rows = ordered[CAPPED_REPLAY_UNIQUE:]
    allowed_replay_unique = ordered[:CAPPED_REPLAY_UNIQUE]
    heldout = set(s00["regression_sets"]["r_heldout"])
    r_success = set(s00["regression_sets"]["r_success_eval"])
    arena_ids = {row["episode_id"] for row in arena_rows}
    replay_ids = {row["episode_id"] for row in allowed_replay_unique}
    assert_disjoint("capped_arena_intersect_allowed_replay", arena_ids, replay_ids)
    assert_disjoint("capped_arena_intersect_heldout", arena_ids, heldout)
    assert_disjoint("capped_arena_intersect_r_success_eval", arena_ids, r_success)
    assert_disjoint("capped_arena_intersect_d_val", arena_ids, val_ids)

    uniform_replay = take_with_repeats(allowed_replay_unique, TARGET_REPLAY_ROWS, args.dataset_seed, "capped_uniform_replay2")

    train_failure_functions = {function_name(row) for row in split["train"]}
    train_failure_families = {coarse_family(name) for name in train_failure_functions}
    exact = [row for row in allowed_replay_unique if row["function_name"] in train_failure_functions]
    exact_ids = {row["episode_id"] for row in exact}
    coarse = [
        row
        for row in allowed_replay_unique
        if row["episode_id"] not in exact_ids and coarse_family(row["function_name"]) in train_failure_families
    ]
    used_ids = exact_ids | {row["episode_id"] for row in coarse}
    uniform_tail = [row for row in allowed_replay_unique if row["episode_id"] not in used_ids]
    targeted_replay = []
    targeted_stages = []
    for label, rows in [("exact_train_failure_function", exact), ("coarse_train_failure_family", coarse), ("uniform_remainder", uniform_tail)]:
        needed = TARGET_REPLAY_ROWS - len(targeted_replay)
        if needed <= 0:
            break
        take = take_with_repeats(rows, needed, args.dataset_seed, f"capped_targeted:{label}")
        targeted_replay.extend(take)
        targeted_stages.append(
            {
                "stage": label,
                "available_unique_rows": len(rows),
                "selected_rows": len(take),
                "selected_unique_episodes": len({row["episode_id"] for row in take}),
            }
        )
    if len(targeted_replay) != TARGET_REPLAY_ROWS:
        raise ValueError({"targeted_replay_rows": len(targeted_replay), "target": TARGET_REPLAY_ROWS})

    outputs = {"uniform": {}, "targeted": {}}
    for arm in ["main", "star"]:
        core = [row for row in read_jsonl(Path(args.root) / f"data/distill_{arm}.jsonl") if row.get("partition") != "replay"]
        for label, rows, suffix in [
            ("uniform", uniform_replay, "replay2_capped"),
            ("targeted", targeted_replay, "targeted_replay2_capped"),
        ]:
            ids = [row["episode_id"] for row in core + rows]
            assert_disjoint(f"{arm}_{label}_intersect_heldout", ids, heldout)
            assert_disjoint(f"{arm}_{label}_intersect_r_success_eval", ids, r_success)
            assert_disjoint(f"{arm}_{label}_intersect_d_val", ids, val_ids)
            assert_disjoint(f"{arm}_{label}_intersect_capped_arena", ids, arena_ids)
            outputs[label][arm] = write_capped_dataset(args, arm, core, rows, suffix)

    arena_payload = {
        "probe": "v13_capped_pool_arena",
        "seed": args.dataset_seed,
        "source": "heldout_subset_of_preexisting_clean_replay_pool",
        "status": "secondary_capped_pool_arena_ready",
        "available_replay_unique": len(replay_rows),
        "allowed_replay_unique": len(allowed_replay_unique),
        "arena_size": len(arena_rows),
        "arena_episode_ids": [row["episode_id"] for row in arena_rows],
        "arena_records": arena_rows,
        "function_distribution": dict(sorted(Counter(row["function_name"] for row in arena_rows).items())),
        "asserts": {
            "arena_intersect_allowed_replay": 0,
            "arena_intersect_heldout": 0,
            "arena_intersect_r_success_eval": 0,
            "arena_intersect_d_val": 0,
        },
        "caveat": "This is a secondary PROBE arena: the source pool was used in earlier historical replay variants, but is held out from all capped-pool D3/D1 runs.",
    }
    write_json(Path(args.root) / "results/v13_capped_pool_arena.json", arena_payload)
    summary = {
        "probe": "v13_capped_pool_datasets",
        "dataset_seed": args.dataset_seed,
        "target_replay_rows_per_arm": TARGET_REPLAY_ROWS,
        "available_replay_unique": len(replay_rows),
        "allowed_replay_unique": len(allowed_replay_unique),
        "arena_size": len(arena_rows),
        "uniform_replay_unique_episode_count": len({row["episode_id"] for row in uniform_replay}),
        "targeted_selection_stages": targeted_stages,
        "targeted_stage_row_share": {
            stage["stage"]: stage["selected_rows"] / TARGET_REPLAY_ROWS for stage in targeted_stages
        },
        "outputs": outputs,
        "leakage_asserts": {
            "main_uniform_intersect_capped_arena": 0,
            "star_uniform_intersect_capped_arena": 0,
            "main_targeted_intersect_capped_arena": 0,
            "star_targeted_intersect_capped_arena": 0,
            "intersect_heldout": 0,
            "intersect_r_success_eval": 0,
            "intersect_d_val": 0,
        },
    }
    write_json(args.summary_output, summary)
    return summary


def residual_coverage(args):
    baseline = read_json(args.failures)
    records = {row["episode_id"]: row for row in baseline["records"]}
    _, split = load_split(args.exp2_root, args.failures, args.split_seed)
    train_failure_functions = {function_name(row) for row in split["train"]}
    train_failure_families = {coarse_family(name) for name in train_failure_functions}
    rows = {}
    for arm, path in {
        "main": Path(args.root) / "results/s05_blocked_kl/main/main_kl_r16_lr5e-5_ep2_replay2_lam2_seed20260703.json",
        "star": Path(args.root) / "results/s05_blocked_kl/star/star_kl_r16_lr1e-4_ep2_replay2_lam2_seed20260703.json",
    }.items():
        payload = read_json(path)
        wrong = [row for row in payload["r_success_records"] if not row["success"]]
        exact = 0
        coarse = 0
        for row in wrong:
            name = function_name(records[row["episode_id"]])
            exact += int(name in train_failure_functions)
            coarse += int(coarse_family(name) in train_failure_families)
        rows[arm] = {
            "new_wrong": len(wrong),
            "exact_train_failure_function_cover": exact,
            "coarse_train_failure_family_cover": coarse,
            "exact_perfect_protection_residual_forget": (len(wrong) - exact) / 400,
            "coarse_perfect_protection_residual_forget": (len(wrong) - coarse) / 400,
        }
    write_json(args.output, {"probe": "v13_residual_targeted_coverage", "arms": rows})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="v1.13 arena feasibility and targeted replay helpers")
    parser.add_argument("--root", default=".")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--raw-root", default="../EDG-BFCL/deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--dataset-seed", type=int, default=DEFAULT_DATASET_SEED)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("category-audit")
    audit.add_argument("--output", default="results/v13_category_touch_audit.json")
    audit.add_argument("--min-fresh-candidates", type=int, default=250)
    audit.set_defaults(func=category_touch_audit)

    targeted = subparsers.add_parser("build-targeted")
    targeted.add_argument("--summary-output", default="results/s06_targeted_replay2.json")
    targeted.set_defaults(func=build_targeted)

    capped = subparsers.add_parser("build-capped")
    capped.add_argument("--summary-output", default="results/s06_capped_pool_datasets.json")
    capped.set_defaults(func=build_capped)

    coverage = subparsers.add_parser("residual-coverage")
    coverage.add_argument("--output", default="results/v13_residual_targeted_coverage.json")
    coverage.set_defaults(func=residual_coverage)

    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
