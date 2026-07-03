import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.load(open(path, encoding="utf-8"))


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


def collect_training_ids(root):
    ids = set()
    for folder in [root / "data", root / "contaminated_gridv1/data"]:
        if not folder.exists():
            continue
        for path in folder.glob("*.jsonl"):
            if path.name.startswith("distill_") or path.name.startswith("pass16_"):
                for row in read_jsonl(path):
                    if row.get("episode_id"):
                        ids.add(row["episode_id"])
    return ids


def stable_sample(rows, count, seed, label):
    keyed = []
    for row in rows:
        episode_id = row["episode_id"]
        digest = hashlib.sha256(f"{seed}:{label}:{episode_id}".encode("utf-8")).hexdigest()
        keyed.append((digest, row))
    return [row for _, row in sorted(keyed)[:count]]


def main():
    parser = argparse.ArgumentParser(description="Freeze v1.18 sibling arena after decision-tree landing")
    parser.add_argument("--root", default=".")
    parser.add_argument("--base-eval", default="results/v17_sibling_base_eval.json")
    parser.add_argument("--output", default="results/v17_sibling_arena.json")
    parser.add_argument("--arena-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    root = Path(args.root)
    base = read_json(root / args.base_eval)
    parallel_success = [
        row
        for row in base["records"]
        if row.get("segment") == "parallel_family" and row.get("call_success") is True
    ]
    simple_success = [
        row
        for row in base["records"]
        if row.get("segment") == "simple_python_unrecorded" and row.get("call_success") is True
    ]
    if len(parallel_success) >= args.arena_size:
        landing = "a_never_touched_parallel_family_success_ge_300"
        source = "parallel_family_never_touched_base_successes_only"
        selected = stable_sample(parallel_success, args.arena_size, args.seed, "v17_sibling_arena")
    elif len(parallel_success) + len(simple_success) >= args.arena_size:
        landing = "b_composite_parallel_plus_simple_python_success_ge_300"
        source = "parallel_family_plus_simple_python_unrecorded_successes"
        selected = stable_sample(parallel_success + simple_success, args.arena_size, args.seed, "v17_sibling_arena")
    else:
        landing = "c_composite_lt_300_probe"
        source = "parallel_family_plus_simple_python_unrecorded_successes_probe"
        selected = stable_sample(
            parallel_success + simple_success,
            len(parallel_success) + len(simple_success),
            args.seed,
            "v17_sibling_arena",
        )

    ids = {row["episode_id"] for row in selected}
    training = collect_training_ids(root)
    s00 = read_json(root / "results/s00_inventory.json")
    old400 = set(s00["regression_sets"]["r_success_eval"])
    capped = set(read_json(root / "results/v13_capped_pool_arena.json").get("episode_ids", []))
    d_val_file = root / "results/s05_blocked_kl/main/main_ep3_r16_lr5e-5_replay1_seed20260703.json"
    d_val = {row["episode_id"] for row in read_json(d_val_file).get("val_records", [])}
    d_heldout = {
        row["episode_id"]
        for row in read_json(root / "results/s01_heldout_pass16_partition.json").get("episodes", [])
    }
    asserts = {
        "arena_intersect_all_training_rows": len(ids & training),
        "arena_intersect_old400": len(ids & old400),
        "arena_intersect_capped106": len(ids & capped),
        "arena_intersect_d_val": len(ids & d_val),
        "arena_intersect_d_heldout": len(ids & d_heldout),
    }
    if any(asserts.values()):
        raise AssertionError(asserts)

    payload = {
        "probe": "v17_sibling_arena",
        "decision_tree_landing": landing,
        "source": source,
        "seed": args.seed,
        "requested_arena_size": args.arena_size,
        "available_parallel_family_successes": len(parallel_success),
        "available_simple_python_unrecorded_successes": len(simple_success),
        "selected_size": len(selected),
        "status": "frozen_pending_pytest_registration",
        "segment_distribution": dict(Counter(row["segment"] for row in selected)),
        "category_distribution": dict(Counter(row["category"] for row in selected)),
        "asserts": asserts,
        "episode_ids": [row["episode_id"] for row in selected],
        "records": selected,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in [
        "decision_tree_landing",
        "available_parallel_family_successes",
        "available_simple_python_unrecorded_successes",
        "selected_size",
        "segment_distribution",
        "category_distribution",
        "asserts",
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
