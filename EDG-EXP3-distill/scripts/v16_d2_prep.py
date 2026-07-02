import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SPLIT_SEED = 20260630


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_exp2(exp2_root):
    exp2_root = Path(exp2_root).resolve()
    if str(exp2_root) not in sys.path:
        sys.path.insert(0, str(exp2_root))
    from scripts import evolution_loop as evo

    return evo


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


def raw_category_counts(raw_root, recorded_episode_ids):
    raw_root = Path(raw_root)
    answer_root = raw_root / "possible_answer"
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
        first_gt = answers[0].get("ground_truth") if answers else None
        required_calls = len(first_gt) if isinstance(first_gt, list) else 1
        parseable = sum(current_eval_parseable_ground_truth(row.get("ground_truth")) for row in answers)
        unrecorded_ids = sorted(set(question_ids) - set(recorded_episode_ids))
        rows.append(
            {
                "category": category,
                "question_rows": len(question_ids),
                "answer_rows": len(answers),
                "ground_truth_calls_per_episode_first_row": required_calls,
                "current_eval_parseable_rows": parseable,
                "single_call_compatible_with_current_eval": required_calls == 1 and parseable == len(answers),
                "unrecorded_raw_episode_rows": len(unrecorded_ids),
                "unrecorded_raw_episode_ids": unrecorded_ids[:20],
            }
        )
    return rows


def add_touch(touches, source, episode_id):
    if episode_id:
        touches[source].add(episode_id)


def collect_training_touches(root):
    root = Path(root)
    touches = defaultdict(set)
    for folder in [root / "data", root / "contaminated_gridv1/data"]:
        if not folder.exists():
            continue
        for path in folder.glob("*.jsonl"):
            if not (
                path.name.startswith("distill_")
                or path.name.startswith("pass16_")
            ):
                continue
            for row in read_jsonl(path):
                add_touch(touches, f"train:{path.relative_to(root)}", row.get("episode_id"))
    return touches


def collect_eval_touches(root):
    root = Path(root)
    touches = defaultdict(set)
    s00_path = root / "results/s00_inventory.json"
    if s00_path.exists():
        s00 = read_json(s00_path)
        for key, value in s00.get("regression_sets", {}).items():
            if isinstance(value, list):
                touches[f"eval:s00:{key}"].update(value)
    for path in [
        root / "results/s01_pass16_partition.json",
        root / "results/s01_heldout_pass16_partition.json",
        root / "contaminated_gridv1/results/s01_pass16_partition.json",
    ]:
        if path.exists():
            payload = read_json(path)
            for row in payload.get("episodes", []):
                add_touch(touches, f"eval:{path.relative_to(root)}", row.get("episode_id"))
    for path in (root / "results").glob("s0*_*/**/*.json"):
        payload = read_json(path)
        for key in ["val_records", "r_success_records", "records"]:
            for row in payload.get(key, []):
                add_touch(touches, f"eval:{path.relative_to(root)}:{key}", row.get("episode_id"))
    for path in (root / "results").glob("s03_grid_clean/**/*.json"):
        payload = read_json(path)
        for key in ["val_records", "r_success_records"]:
            for row in payload.get(key, []):
                add_touch(touches, f"eval:{path.relative_to(root)}:{key}", row.get("episode_id"))
    for path in (root / "results").glob("s04_upgrade/**/*.json"):
        payload = read_json(path)
        for key in ["val_records", "r_success_records"]:
            for row in payload.get(key, []):
                add_touch(touches, f"eval:{path.relative_to(root)}:{key}", row.get("episode_id"))
    for path in (root / "results").glob("s05_blocked_kl/**/*.json"):
        payload = read_json(path)
        for key in ["val_records", "r_success_records"]:
            for row in payload.get(key, []):
                add_touch(touches, f"eval:{path.relative_to(root)}:{key}", row.get("episode_id"))
    for path in (root / "results").glob("s06_capped_arena_eval/**/*.json"):
        payload = read_json(path)
        for row in payload.get("records", []):
            add_touch(touches, f"eval:{path.relative_to(root)}:records", row.get("episode_id"))
    for path in (root / "results").glob("s06_targeted_c/**/*.json"):
        payload = read_json(path)
        for key in ["val_records", "r_success_records"]:
            for row in payload.get(key, []):
                add_touch(touches, f"eval:{path.relative_to(root)}:{key}", row.get("episode_id"))
    return touches


def category_from_id(episode_id, baseline_by_id):
    if episode_id in baseline_by_id:
        return baseline_by_id[episode_id].get("category")
    if episode_id.startswith("live_multiple_"):
        return "live_multiple"
    if episode_id.startswith("live_simple_"):
        return "live_simple"
    if episode_id.startswith("simple_python_"):
        return "simple_python"
    if episode_id.startswith("simple_java_"):
        return "simple_java"
    if episode_id.startswith("simple_javascript_"):
        return "simple_javascript"
    if episode_id.startswith("multiple_"):
        return "multiple"
    return episode_id.split("_", 1)[0]


def status_for_category(category, train_ids, eval_ids, baseline_category_ids):
    ids = baseline_category_ids.get(category, set())
    train_hit = bool(ids & train_ids)
    eval_hit = bool(ids & eval_ids)
    if train_hit:
        return "train_touched"
    if eval_hit:
        return "eval_touched_only"
    return "never_touched"


def heldout_diff_and_success_decomp(root, baseline, evo, split_seed, training_ids, eval_ids):
    multiple_records = [row for row in baseline["records"] if row.get("split") == "multiple"]
    multiple_failures = [row for row in baseline["failures"] if row.get("split") == "multiple"]
    split_all = evo.deterministic_split(multiple_records, seed=split_seed)
    split_fail = evo.deterministic_split(multiple_failures, seed=split_seed)
    held_all = split_all["held_out"]
    held_success = [row for row in held_all if row.get("call_success") is True]
    held_fail_recomputed = [row for row in held_all if row.get("call_success") is not True]
    frozen_path = Path(root) / "results/s01_heldout_pass16_partition.json"
    frozen = read_json(frozen_path)
    frozen_ids = [row["episode_id"] for row in frozen.get("episodes", [])]
    recomputed_failure_ids = [row["episode_id"] for row in split_fail["held_out"]]
    held_all_failure_ids = [row["episode_id"] for row in held_fail_recomputed]
    baseline_by_id = {row["episode_id"]: row for row in baseline["records"]}
    frozen_set = set(frozen_ids)
    recomputed_set = set(recomputed_failure_ids)
    held_all_fail_set = set(held_all_failure_ids)
    extra_in_frozen = sorted(frozen_set - recomputed_set)
    missing_from_frozen = sorted(recomputed_set - frozen_set)
    diff_rows = []
    for episode_id in extra_in_frozen:
        record = baseline_by_id.get(episode_id, {})
        diff_rows.append(
            {
                "episode_id": episode_id,
                "direction": "in_frozen_158_not_in_recomputed_failure_151",
                "baseline_call_success": record.get("call_success"),
                "category": record.get("category"),
            }
        )
    for episode_id in missing_from_frozen:
        record = baseline_by_id.get(episode_id, {})
        diff_rows.append(
            {
                "episode_id": episode_id,
                "direction": "in_recomputed_failure_151_not_in_frozen_158",
                "baseline_call_success": record.get("call_success"),
                "category": record.get("category"),
            }
        )
    held_success_ids = {row["episode_id"] for row in held_success}
    s00 = read_json(Path(root) / "results/s00_inventory.json")
    old_r_success = set(s00["regression_sets"]["r_success_eval"])
    training_overlap = sorted(held_success_ids & training_ids)
    old_eval_overlap = sorted(held_success_ids & old_r_success)
    residual = sorted(held_success_ids - set(training_overlap) - set(old_eval_overlap))
    return {
        "split_all_multiple": len(multiple_records),
        "held_all_total": len(held_all),
        "held_all_base_success": len(held_success),
        "held_all_base_failure": len(held_all_failure_ids),
        "failure_split_heldout": len(recomputed_failure_ids),
        "frozen_heldout_failures": len(frozen_ids),
        "frozen_vs_recomputed": {
            "extra_in_frozen_count": len(extra_in_frozen),
            "missing_from_frozen_count": len(missing_from_frozen),
            "diff_rows": diff_rows,
        },
        "frozen_vs_held_all_failures": {
            "extra_in_frozen_count": len(frozen_set - held_all_fail_set),
            "missing_from_frozen_count": len(held_all_fail_set - frozen_set),
        },
        "heldout_success_decomposition": {
            "total": len(held_success_ids),
            "intersect_historical_training_or_replay": len(training_overlap),
            "intersect_old_r_success_eval": len(old_eval_overlap),
            "residual_after_training_and_old_eval": len(residual),
            "training_overlap_ids": training_overlap[:50],
            "old_eval_overlap_ids": old_eval_overlap[:50],
            "residual_ids": residual[:50],
        },
    }


def teacher_output_check(root):
    path = Path(root) / "results/s01_heldout_pass16_partition.json"
    payload = read_json(path)
    episodes = payload.get("episodes", [])
    teacher_with_output = [
        row["episode_id"]
        for row in episodes
        if row.get("teacher", {}).get("success") and row.get("teacher", {}).get("raw_model_output") is not None
    ]
    return {
        "path": str(path),
        "episode_count": len(episodes),
        "teacher_success_count": sum(1 for row in episodes if row.get("teacher", {}).get("success")),
        "teacher_success_with_raw_output": len(teacher_with_output),
        "teacher_outputs_present": len(teacher_with_output)
        == sum(1 for row in episodes if row.get("teacher", {}).get("success")),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="v1.16 D2 preflight audit")
    parser.add_argument("--root", default=".")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--raw-root", default="../EDG-BFCL/deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--output", default="results/v16_d2_step0_audit.json")
    args = parser.parse_args(argv)

    root = Path(args.root)
    baseline = read_json(args.failures)
    baseline_by_id = {row["episode_id"]: row for row in baseline["records"]}
    baseline_category_ids = defaultdict(set)
    for row in baseline["records"]:
        baseline_category_ids[row["category"]].add(row["episode_id"])

    training_touches = collect_training_touches(root)
    eval_touches = collect_eval_touches(root)
    training_ids = set().union(*training_touches.values()) if training_touches else set()
    eval_ids = set().union(*eval_touches.values()) if eval_touches else set()
    raw_rows = raw_category_counts(args.raw_root, set(baseline_by_id))

    categories = sorted({row["category"] for row in raw_rows} | set(baseline_category_ids))
    tri_state = []
    for category in categories:
        raw = next((row for row in raw_rows if row["category"] == category), None)
        train_count = sum(1 for episode_id in training_ids if category_from_id(episode_id, baseline_by_id) == category)
        eval_count = sum(1 for episode_id in eval_ids if category_from_id(episode_id, baseline_by_id) == category)
        status = status_for_category(category, training_ids, eval_ids, baseline_category_ids)
        if raw and category not in baseline_category_ids and train_count == 0 and eval_count == 0:
            status = "never_touched"
        tri_state.append(
            {
                "category": category,
                "status": status,
                "recorded_rows": len(baseline_category_ids.get(category, set())),
                "recorded_train_touched_episode_count": train_count,
                "recorded_eval_touched_episode_count": eval_count,
                "raw_question_rows": raw["question_rows"] if raw else 0,
                "raw_unrecorded_rows": raw["unrecorded_raw_episode_rows"] if raw else 0,
                "single_call_compatible_with_current_eval": raw["single_call_compatible_with_current_eval"] if raw else False,
                "current_eval_parseable_rows": raw["current_eval_parseable_rows"] if raw else 0,
            }
        )

    never_touched_compatible = [
        row
        for row in tri_state
        if row["status"] == "never_touched" and row["single_call_compatible_with_current_eval"]
    ]
    downgrade_candidates = [
        row
        for row in tri_state
        if row["status"] == "eval_touched_only"
        and row["single_call_compatible_with_current_eval"]
        and row["raw_unrecorded_rows"] > 0
    ]
    evo = load_exp2(args.exp2_root)
    payload = {
        "probe": "v16_d2_step0_audit",
        "split_seed": args.split_seed,
        "category_touch_table": tri_state,
        "touch_summary": {
            "training_touch_sets": {key: len(value) for key, value in sorted(training_touches.items())},
            "eval_touch_sets": {key: len(value) for key, value in sorted(eval_touches.items())},
        },
        "sibling_arena_constructibility": {
            "never_touched_single_call_compatible_categories": never_touched_compatible,
            "never_touched_single_call_total_rows": sum(row["raw_question_rows"] for row in never_touched_compatible),
            "eval_touched_only_unrecorded_single_call_candidates": downgrade_candidates,
            "eval_touched_only_unrecorded_total_rows": sum(row["raw_unrecorded_rows"] for row in downgrade_candidates),
            "status": "blocked_no_never_touched_single_call_compatible_categories"
            if not never_touched_compatible
            else "needs_base_forward",
        },
        "heldout_diff": heldout_diff_and_success_decomp(root, baseline, evo, args.split_seed, training_ids, eval_ids),
        "teacher_heldout_output_check": teacher_output_check(root),
    }
    write_json(root / args.output, payload)
    print(json.dumps(payload["sibling_arena_constructibility"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["heldout_diff"]["heldout_success_decomposition"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
