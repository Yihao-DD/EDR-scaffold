import argparse
import hashlib
import json
import sys
from pathlib import Path


def deterministic_split(episode_ids, seed=20260630, train_ratio=0.5, validation_ratio=0.25):
    keyed = []
    for episode_id in episode_ids:
        digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()
        keyed.append((digest, episode_id))
    ordered = [episode_id for _, episode_id in sorted(keyed)]
    n = len(ordered)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)
    return {
        "train_update": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "held_out": ordered[validation_end:],
    }


def build_split_result(failure_ids, seed=20260630):
    splits = deterministic_split(failure_ids, seed=seed)
    train = set(splits["train_update"])
    validation = set(splits["validation"])
    held_out = set(splits["held_out"])
    all_split_ids = train | validation | held_out
    asserts = {
        "train_validation_disjoint": train.isdisjoint(validation),
        "train_held_out_disjoint": train.isdisjoint(held_out),
        "validation_held_out_disjoint": validation.isdisjoint(held_out),
        "all_failures_accounted_for": len(all_split_ids) == len(failure_ids),
    }
    assert all(asserts.values())
    return {
        "seed": seed,
        "n_failures": len(failure_ids),
        "splits": splits,
        "sizes": {k: len(v) for k, v in splits.items()},
        "asserts": asserts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="B0 deterministic split of natural failure episodes")
    parser.add_argument("--input", default="results/a2_natural_failures.json")
    parser.add_argument("--output", default="results/b0_splits.json")
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    failure_ids = [record["episode_id"] for record in data.get("failures", [])]
    result = build_split_result(failure_ids, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
