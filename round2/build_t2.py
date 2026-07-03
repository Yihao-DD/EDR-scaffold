#!/usr/bin/env python3
"""Build T2 scaffold-taught rows from F2 and teacher-2 outputs."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from common import (
    add_common_args,
    assert_disjoint,
    episode_ids,
    load_episode_id_file,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def dedupe_by_episode_output(rows: list[dict]) -> list[dict]:
    seen = OrderedDict()
    for row in rows:
        key = (row.get("episode_id"), row.get("output") or row.get("prediction"))
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--teacher2-samples", required=True, help="JSONL rows from M1+H2, train split only.")
    parser.add_argument("--output", default="round2_outputs/t2.jsonl")
    parser.add_argument("--summary-output", default="round2_outputs/t2_summary.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_rows = read_jsonl(args.teacher2_samples)
    missing_ast = [row.get("episode_id") for row in raw_rows if "ast_pass" not in row]
    if missing_ast:
        raise AssertionError(
            {
                "assert": "teacher2_ast_pass_field_present",
                "missing_count": len(missing_ast),
                "examples": missing_ast[:20],
            }
        )
    rejected_ast = [row.get("episode_id") for row in raw_rows if row.get("ast_pass") is not True]
    rows = [dict(row, round2_source=row.get("round2_source", "teacher2")) for row in raw_rows if row.get("ast_pass") is True]
    rows = dedupe_by_episode_output(rows)
    ids = episode_ids(rows)

    assertions = []
    for name, rel in [
        ("heldout", "D_heldout_failures.json"),
        ("d_val", "D_val_failures.json"),
        ("old400", "old400_success_eval.json"),
        ("sibling300", "sibling300_arena.json"),
    ]:
        assertions.append(assert_disjoint(f"t2_intersect_{name}", ids, load_episode_id_file(data_root / "episode_ids" / rel)))

    status = "OK" if len(rows) >= 30 else "MATERIAL_EXHAUSTION"
    write_jsonl(args.output, rows)
    write_json(
        args.summary_output,
        {
            "probe": "round2_build_t2",
            "rows": len(rows),
            "raw_rows": len(raw_rows),
            "ast_rejected_rows": len(rejected_ast),
            "unique_episodes": len(ids),
            "status": status,
            "assertions": assertions,
        },
    )
    print(f"wrote {args.output} rows={len(rows)} status={status}")


if __name__ == "__main__":
    main()
