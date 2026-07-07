"""Round-2 replay block: M1 successes repeated to a 2:1 replay:core row ratio.

Every replay row is stamped source=replay_base_success / partition=replay so
the KL-anchor path fires during training.

CLI:
  python -m edr.round2.build_replay2 --m1-success-jsonl <jsonl> --t2-jsonl <jsonl>
"""

from __future__ import annotations

import argparse

from edr.data.leakage import assert_disjoint, episode_ids, load_eval_surface_ids
from edr.io_utils import read_jsonl, write_json, write_jsonl
from edr.paths import ROUND2_OUT, resolve


def replay_row(row, repeat_index):
    out = dict(row)
    out["source"] = "replay_base_success"
    out["partition"] = "replay"
    out["round2_source"] = out.get("round2_source", "m1_train_success")
    out["replay_repeat_index"] = repeat_index
    return out


def repeat_to_count(rows, target):
    if not rows:
        return []
    out = []
    i = 0
    while len(out) < target:
        out.append(replay_row(rows[i % len(rows)], i // len(rows)))
        i += 1
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-success-jsonl", required=True)
    parser.add_argument("--t2-jsonl", default=None, help="If supplied, target rows default to 2x |T2|.")
    parser.add_argument("--target-rows", type=int, default=None)
    parser.add_argument("--output", default=str(ROUND2_OUT / "replay2.jsonl"))
    parser.add_argument("--summary-output", default=str(ROUND2_OUT / "replay2_summary.json"))
    args = parser.parse_args(argv)

    rows = read_jsonl(resolve(args.m1_success_jsonl))
    ids = episode_ids(rows)
    assertions = []
    for name, surface in load_eval_surface_ids().items():
        assertions.append(assert_disjoint(f"replay2_intersect_{name}", ids, surface))
    if args.target_rows is None:
        if not args.t2_jsonl:
            raise SystemExit("--target-rows is required unless --t2-jsonl is supplied")
        t2_rows = read_jsonl(resolve(args.t2_jsonl))
        target_rows = 2 * len(t2_rows)
    else:
        target_rows = args.target_rows
    selected = repeat_to_count(rows, target_rows)
    write_jsonl(resolve(args.output), selected)
    write_json(
        resolve(args.summary_output),
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
