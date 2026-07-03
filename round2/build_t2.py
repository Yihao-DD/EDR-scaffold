#!/usr/bin/env python3
"""Build T2 scaffold-taught rows from F2 and teacher-2 outputs."""

from __future__ import annotations

import argparse
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = REPO_ROOT / "round2" / "loop"
sys.path.insert(0, str(LOOP_DIR))

from evolution_loop_m1 import (  # noqa: E402
    ModelRunner,
    PatchCandidate,
    _manifest_m1_adapter_dir,
    _resolve_repo_path,
    build_prompt,
    call_success,
    parse_model_output,
    relevant_patches_for_episode,
)


def dedupe_by_episode_output(rows: list[dict]) -> list[dict]:
    seen = OrderedDict()
    for row in rows:
        key = (row.get("episode_id"), row.get("output") or row.get("prediction"))
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def _loop_family_from_method(method: str) -> str:
    return method.replace("-evo", "")


def load_h2_patches(loop_output: Path, family: str = "NL", seed: int | None = None) -> list[PatchCandidate]:
    payload = read_json(loop_output)
    per_seed = payload.get("per_seed") or []
    if not per_seed:
        raise AssertionError({"assert": "loop_output_has_per_seed", "path": str(loop_output)})
    selected = None
    for item in per_seed:
        if seed is None or item.get("seed") == seed:
            selected = item
            break
    if selected is None:
        raise AssertionError({"assert": "loop_seed_present", "seed": seed, "path": str(loop_output)})
    loop_outputs = selected.get("loop_outputs") or {}
    if family not in loop_outputs:
        raise AssertionError({"assert": "loop_family_present", "family": family, "available": sorted(loop_outputs)})
    return [PatchCandidate(**row) for row in loop_outputs[family].get("accepted_patches", [])]


def teacher2_generate_row(runner: ModelRunner, episode: dict, patches: list[PatchCandidate], sample: dict, max_new_tokens: int) -> dict:
    relevant = relevant_patches_for_episode(patches, episode)
    prompt = build_prompt(episode, relevant)
    raw = runner.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=sample["do_sample"],
        temperature=sample.get("temperature"),
    )
    try:
        prediction = parse_model_output(raw)
        parse_error = None
        ast_pass = call_success(prediction, episode["ground_truth_call"])
    except Exception as error:
        prediction = None
        parse_error = str(error)
        ast_pass = False
    return {
        **episode,
        "input": prompt,
        "output": raw,
        "raw_model_output": raw,
        "prediction": prediction,
        "parse_error": parse_error,
        "ast_pass": ast_pass,
        "round2_source": "teacher2_m1_h2",
        "model_stack": "M1_merged_plus_H2_patch_in_context",
        "patch_ids": [patch.patch_id for patch in relevant],
        "sample_temperature": sample.get("temperature", 0.0),
        "sample_index": sample["index"],
    }


def generate_teacher2_samples(args) -> list[dict]:
    if not args.f2_json or not args.loop_output:
        raise SystemExit("--teacher2-samples or both --f2-json and --loop-output are required")
    f2_payload = read_json(args.f2_json)
    episodes = f2_payload.get("failures") or f2_payload.get("f2") or []
    patches = load_h2_patches(Path(args.loop_output), family=_loop_family_from_method(args.loop_method), seed=args.loop_seed)
    base_adapter = (
        _resolve_repo_path(args.m1_adapter)
        if args.m1_adapter
        else _manifest_m1_adapter_dir(args.manifest)
    )
    runner = ModelRunner(args.model_id, base_adapter_dir=str(base_adapter))
    rows = []
    for episode in episodes:
        samples = [{"index": "t0_0", "do_sample": False, "temperature": 0.0}]
        for idx in range(args.t08_samples):
            samples.append({"index": f"t08_{idx}", "do_sample": True, "temperature": 0.8})
        attempts = args.t08_samples
        ast_t08 = 0
        for sample in samples:
            row = teacher2_generate_row(runner, episode, patches, sample, args.max_new_tokens)
            rows.append(row)
            if sample["do_sample"] and row["ast_pass"]:
                ast_t08 += 1
        while ast_t08 < args.t08_samples and attempts < args.augment_to:
            sample = {"index": f"t08_extra_{attempts}", "do_sample": True, "temperature": 0.8}
            row = teacher2_generate_row(runner, episode, patches, sample, args.max_new_tokens)
            rows.append(row)
            attempts += 1
            if row["ast_pass"]:
                ast_t08 += 1
    if args.teacher2_samples_output:
        write_jsonl(args.teacher2_samples_output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--teacher2-samples", default=None, help="Precomputed JSONL rows from M1+H2, train split only.")
    parser.add_argument("--f2-json", default=None, help="F2 JSON from collect_failures.py; used to generate teacher-2 samples.")
    parser.add_argument("--loop-output", default=None, help="M1-aware loop output containing accepted H2 patches.")
    parser.add_argument("--loop-method", default="NL-evo", choices=["NL-evo", "NL-padded-evo", "STRUCT-evo"])
    parser.add_argument("--loop-seed", type=int, default=None)
    parser.add_argument("--manifest", default="repro_rep2/MANIFEST.json")
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--t08-samples", type=int, default=4)
    parser.add_argument("--augment-to", type=int, default=8)
    parser.add_argument("--teacher2-samples-output", default="round2_outputs/teacher2_samples.jsonl")
    parser.add_argument("--output", default="round2_outputs/t2.jsonl")
    parser.add_argument("--summary-output", default="round2_outputs/t2_summary.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    raw_rows = read_jsonl(args.teacher2_samples) if args.teacher2_samples else generate_teacher2_samples(args)
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
