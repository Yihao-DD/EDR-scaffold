#!/usr/bin/env python3
"""Build round-2 replay pool from M1 T=0 train successes."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, assert_disjoint, episode_ids, load_episode_id_file, read_jsonl, write_json, write_jsonl


def replay_row(row: dict, repeat_index: int) -> dict:
    out = dict(row)
    out["source"] = "replay_base_success"
    out["partition"] = "replay"
    out["round2_source"] = out.get("round2_source", "m1_train_success")
    out["replay_repeat_index"] = repeat_index
    return out


def repeat_to_count(rows: list[dict], target: int) -> list[dict]:
    if not rows:
        return []
    out = []
    i = 0
    while len(out) < target:
        out.append(replay_row(rows[i % len(rows)], i // len(rows)))
        i += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--m1-success-jsonl", required=True, help="M1 T=0 train-share success trajectories.")
    parser.add_argument("--t2-jsonl", default=None, help="T2 JSONL. If supplied, target rows default to 2x |T2|.")
    parser.add_argument("--target-rows", type=int, default=None)
    parser.add_argument("--output", default="round2_outputs/replay2.jsonl")
    parser.add_argument("--summary-output", default="round2_outputs/replay2_summary.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = read_jsonl(args.m1_success_jsonl)
    ids = episode_ids(rows)
    missing_ids = [index for index, row in enumerate(rows) if not row.get("episode_id")]
    if missing_ids:
        raise AssertionError(
            {
                "assert": "m1_success_episode_id_present",
                "missing_count": len(missing_ids),
                "examples": missing_ids[:20],
            }
        )
    assertions = []
    for name, rel in [
        ("heldout", "D_heldout_failures.json"),
        ("d_val", "D_val_failures.json"),
        ("old400", "old400_success_eval.json"),
        ("sibling300", "sibling300_arena.json"),
    ]:
        assertions.append(assert_disjoint(f"replay2_intersect_{name}", ids, load_episode_id_file(data_root / "episode_ids" / rel)))
    if args.target_rows is None:
        if not args.t2_jsonl:
            raise SystemExit("--target-rows is required unless --t2-jsonl is supplied")
        t2_rows = read_jsonl(args.t2_jsonl)
        target_rows = 2 * len(t2_rows)
    else:
        target_rows = args.target_rows
    selected = repeat_to_count(rows, target_rows)
    write_jsonl(args.output, selected)
    write_json(
        args.summary_output,
        {
            "probe": "round2_build_replay2",
            "target_rows": target_rows,
            "t2_jsonl": args.t2_jsonl,
            "replay_marker": {"source": "replay_base_success", "partition": "replay"},
            "unique_rows": len(rows),
            "unique_episodes": len(ids),
            "assertions": assertions,
        },
    )
    print(f"wrote {args.output} rows={len(selected)} unique_episodes={len(ids)}")


if __name__ == "__main__":
    main()
