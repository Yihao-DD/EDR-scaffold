#!/usr/bin/env python3
"""Check that the M1-aware loop entry matches collect_failures no-patch outputs.

This smoke isolates the model-loading invariant. It uses the same M1-merged
ModelRunner as the patch loop, but renders the base prompt used by
collect_failures.py. Any mismatch means the loop is not equivalent to the F2
collector before patches are introduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOOP_DIR.parents[1]
sys.path.insert(0, str(LOOP_DIR))
sys.path.insert(0, str(REPO_ROOT / "repro_rep2"))

from evolution_loop_m1 import ModelRunner, _manifest_m1_adapter_dir, _resolve_repo_path
from scripts.phase0_probe import build_base_prompt


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-json", required=True)
    parser.add_argument("--output", default="round2_outputs/loop/no_patch_equivalence.json")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--manifest", default="repro_rep2/MANIFEST.json")
    parser.add_argument("--base-adapter-dir", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    payload = read_json(args.collect_json)
    prompt_template = payload.get("prompt_template")
    if not prompt_template:
        raise SystemExit("collect JSON is missing prompt_template; run collect_failures.py in full mode")
    rows = payload.get("failures") or payload.get("f2") or []
    rows = rows[: args.limit]
    missing = [row.get("episode_id") for row in rows if "raw_model_output" not in row]
    if missing:
        raise SystemExit(f"collect JSON rows are missing raw_model_output: {missing[:10]}")

    base_adapter = (
        _resolve_repo_path(args.base_adapter_dir)
        if args.base_adapter_dir
        else _manifest_m1_adapter_dir(args.manifest)
    )
    runner = ModelRunner(args.model_id, cache_dir=args.model_cache_dir, base_adapter_dir=str(base_adapter))
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

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "probe": "round2_loop_no_patch_equivalence",
                "model_stack": ["M0", "A1_merged_in_memory"],
                "collect_json": args.collect_json,
                "checked": len(records),
                "mismatch_count": len(mismatches),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if mismatches:
        raise SystemExit(f"no-patch equivalence failed: {len(mismatches)}/{len(records)} mismatches; see {out}")
    print(f"no_patch_equivalence_ok checked={len(records)} output={out}")


if __name__ == "__main__":
    main()
