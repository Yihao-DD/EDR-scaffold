"""No-patch equivalence smoke: the M1-merged loop runner must reproduce the F2
collector's raw outputs exactly (model-loading invariant check).

CLI:
  python -m edr.round2.equivalence_smoke --collect-json <f2 json> [--limit 10]
"""

from __future__ import annotations

import argparse

from edr.config import load_config, resolve_model_id
from edr.data.splits import build_base_prompt
from edr.io_utils import read_json, write_json
from edr.paths import ROUND2_OUT, resolve
from edr.round2.shared import resolve_m1_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-json", required=True)
    parser.add_argument("--output", default=str(ROUND2_OUT / "loop" / "no_patch_equivalence.json"))
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    config = load_config("round2")
    payload = read_json(resolve(args.collect_json))
    prompt_template = payload.get("prompt_template")
    if not prompt_template:
        raise SystemExit("collect JSON is missing prompt_template; run collect_failures in full mode")
    rows = payload.get("failures") or payload.get("f2") or []
    rows = rows[: args.limit]
    missing = [row.get("episode_id") for row in rows if "raw_model_output" not in row]
    if missing:
        raise SystemExit(f"collect JSON rows are missing raw_model_output: {missing[:10]}")

    from edr.modeling import ModelRunner

    runner = ModelRunner(
        resolve_model_id(config),
        cache_dir=args.model_cache_dir,
        base_adapter_dir=resolve_m1_adapter(args.m1_adapter),
        require_adapter=True,
    )
    records = []
    mismatches = []
    for row in rows:
        prompt = build_base_prompt(row, prompt_template)
        raw = runner.generate(prompt)
        expected = row["raw_model_output"]
        match = raw == expected
        record = {
            "episode_id": row["episode_id"],
            "match": match,
            "expected": expected,
            "actual": raw,
        }
        records.append(record)
        if not match:
            mismatches.append(record)

    write_json(
        resolve(args.output),
        {
            "probe": "round2_loop_no_patch_equivalence",
            "model_stack": ["M0", "A1_merged_in_memory"],
            "collect_json": args.collect_json,
            "checked": len(records),
            "mismatch_count": len(mismatches),
            "records": records,
        },
    )
    if mismatches:
        raise SystemExit(f"no-patch equivalence failed: {len(mismatches)}/{len(records)} mismatches; see {args.output}")
    print(f"no_patch_equivalence_ok checked={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
