#!/usr/bin/env python3
"""Collect F2: failures of M0+A1 on the round-2 train share."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, read_json, slice_limit, write_json


def load_train_share(data_root: Path) -> list[dict]:
    pass16 = read_json(data_root / "train_val_pass16_partition.json")
    val_ids = set(read_json(data_root / "episode_ids" / "D_val_failures.json")["episode_ids"])
    # D_val is excluded from train. Train share is the complement inside train+val failures.
    return [row for row in pass16["episodes"] if row["episode_id"] not in val_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--m1-adapter", required=True, help="Path to designated M1 adapter.")
    parser.add_argument("--output", default="round2_outputs/f2_failures.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_share = slice_limit(load_train_share(data_root), args.limit)
    if args.dry_run:
        failures = [
            {
                "episode_id": row["episode_id"],
                "function_name": row.get("function_name"),
                "failure_class": row.get("failure_class"),
                "dry_run_status": "not_inferred",
            }
            for row in train_share
        ]
    else:
        raise SystemExit(
            "GPU inference is intentionally routed through the project runner. "
            "Use repro_rep2/scripts/s07_heldout_eval.py-style generation with --adapter, "
            "then pass the output JSON to build_t2.py."
        )

    payload = {
        "probe": "round2_collect_failures",
        "m1_adapter": args.m1_adapter,
        "train_share_checked": len(train_share),
        "f2_count": len(failures),
        "f2": failures,
        "note": "Dry run enumerates the train share. Full mode should record M0+A1 T=0 AST failures.",
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} f2_count={len(failures)}")


if __name__ == "__main__":
    main()
