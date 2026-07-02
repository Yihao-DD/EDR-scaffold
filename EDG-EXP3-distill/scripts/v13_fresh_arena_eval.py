import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


DEFAULT_CATEGORIES = ["memory", "web_search"]
DEFAULT_SEED = 20260704


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_exp1(exp1_root):
    exp1_root = Path(exp1_root).resolve()
    if str(exp1_root) not in sys.path:
        sys.path.insert(0, str(exp1_root))
    from scripts import a2_collect_failures as a2
    from scripts import bfcl_common as bfcl

    return a2, bfcl


def stable_sample(rows, count, seed, label):
    if count >= len(rows):
        return list(rows)
    keyed = []
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{label}:{row['episode_id']}".encode("utf-8")).hexdigest()
        keyed.append((digest, row))
    return [row for _, row in sorted(keyed)[:count]]


def touched_episode_ids(root):
    root = Path(root)
    touched = set()
    for path in [
        root / "results/s00_inventory.json",
        root / "results/phase0_manifest.json",
    ]:
        if not path.exists():
            continue
        payload = read_json(path)
        for value in payload.get("regression_sets", {}).values():
            if isinstance(value, list):
                touched.update(value)
    for path in [root / "results/s01_pass16_partition.json", root / "results/s01_heldout_pass16_partition.json"]:
        if path.exists():
            payload = read_json(path)
            touched.update(row["episode_id"] for row in payload.get("episodes", []))
    for phase in ["s03_grid_clean", "s04_upgrade", "s05_blocked_kl"]:
        for path in (root / f"results/{phase}").glob("*/*.json"):
            payload = read_json(path)
            touched.update(row["episode_id"] for row in payload.get("val_records", []))
            touched.update(row["episode_id"] for row in payload.get("r_success_records", []))
    for path in (root / "data").glob("*.jsonl"):
        if path.name.startswith("distill_") or path.name.startswith("pass16_"):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        touched.add(json.loads(line)["episode_id"])
    return touched


def load_existing_records(path):
    path = Path(path)
    if not path.exists():
        return []
    return read_json(path).get("records", [])


def evaluate(args):
    a2, bfcl = load_exp1(args.exp1_root)
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    bfcl.ensure_raw_data(args.source_data_dir, args.raw_dir, categories=categories)
    loaded = bfcl.load_categories(args.raw_dir, categories)
    rows = [row for category in categories for row in loaded[category]]
    rows.sort(key=lambda row: (row["category"], row["id"]))
    existing = load_existing_records(args.output) if args.resume else []
    seen = {row["episode_id"] for row in existing}
    records = list(existing)
    tokenizer, model = a2._load_model(args.model_id, cache_dir=args.model_cache_dir)
    for index, row in enumerate(rows, start=1):
        if row["id"] in seen:
            continue
        prompt = a2.build_prompt(row)
        started = time.time()
        raw_output = a2._generate(tokenizer, model, prompt, args.max_new_tokens)
        latency_ms = int((time.time() - started) * 1000)
        record = a2.build_record(row, raw_output, latency_ms)
        records.append(record)
        if len(records) % args.save_every == 0:
            write_payload(args.output, records, categories, args)
        if index == 1 or index % args.progress_every == 0 or index == len(rows):
            print(
                f"[fresh-base] {index}/{len(rows)} {row['category']} {row['id']} "
                f"call={record['call_success']}",
                flush=True,
            )
    payload = write_payload(args.output, records, categories, args)
    if args.arena_output:
        build_arena_from_payload(payload, args)
    return payload


def write_payload(output, records, categories, args):
    summary = {}
    for category in categories:
        rows = [row for row in records if row["category"] == category]
        success = sum(row.get("call_success") is True for row in rows)
        summary[category] = {
            "n": len(rows),
            "success": success,
            "success_rate": success / len(rows) if rows else 0.0,
        }
    payload = {
        "probe": "v13_fresh_sibling_base_eval",
        "model_id": args.model_id,
        "categories": categories,
        "generation_params": {"do_sample": False, "temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "records": records,
    }
    write_json(output, payload)
    return payload


def build_arena_from_payload(payload, args):
    touched = touched_episode_ids(args.root)
    successes = [row for row in payload["records"] if row.get("call_success") is True]
    clean_successes = [row for row in successes if row["episode_id"] not in touched]
    selected = stable_sample(clean_successes, min(args.arena_size, len(clean_successes)), args.seed, "fresh_arena")
    selected_ids = [row["episode_id"] for row in selected]
    overlap = sorted(set(selected_ids) & touched)
    if overlap:
        raise AssertionError({"fresh_intersect_historical_touch": overlap[:20], "n_overlap": len(overlap)})
    category_distribution = {}
    for row in selected:
        category_distribution[row["category"]] = category_distribution.get(row["category"], 0) + 1
    payload = {
        "probe": "v13_fresh_arena",
        "seed": args.seed,
        "source": "unrecorded_single_call_bfcl_sibling_categories",
        "requested_arena_size": args.arena_size,
        "min_success_threshold": args.min_success_threshold,
        "available_clean_successes": len(clean_successes),
        "selected_size": len(selected),
        "status": "primary_fresh_arena_ready" if len(clean_successes) >= args.min_success_threshold else "primary_fresh_arena_insufficient",
        "category_distribution": dict(sorted(category_distribution.items())),
        "asserts": {
            "fresh_intersect_historical_eval_training_replay": 0,
        },
        "episode_ids": selected_ids,
        "records": selected,
    }
    write_json(args.arena_output, payload)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate unrecorded sibling BFCL categories for v1.13 fresh arena")
    parser.add_argument("--root", default=".")
    parser.add_argument("--exp1-root", default="../EDG-EXP1")
    parser.add_argument("--source-data-dir", default="../EDG-BFCL/deps/bfcl_eval_pkg/bfcl_eval/data")
    parser.add_argument("--raw-dir", default="data/raw/bfcl_v13")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="results/v13_fresh_sibling_base_eval.json")
    parser.add_argument("--arena-output", default="results/v13_fresh_arena.json")
    parser.add_argument("--arena-size", type=int, default=300)
    parser.add_argument("--min-success-threshold", type=int, default=250)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = evaluate(args)
    print(json.dumps({"summary": result["summary"], "output": args.output, "arena_output": args.arena_output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
