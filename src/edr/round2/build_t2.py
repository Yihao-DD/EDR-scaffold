"""Build T2 (round-2 scaffold-taught rows) from F2 + H2 patches. AST fail-closed.

Also writes t1_t2.jsonl (T1 core ∪ T2, episode_id-deduped) for the 2x2 "seen"
signature at evaluation time. |T2| < 30 is flagged MATERIAL_EXHAUSTION — a
preregistered finding, not an error.

CLI:
  python -m edr.round2.build_t2 --f2-json <json> --loop-output <json> [--m1-adapter <dir>]
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from edr.config import load_config, resolve_model_id
from edr.data.leakage import assert_disjoint, episode_ids, load_eval_surface_ids
from edr.io_utils import read_json, read_jsonl, write_json, write_jsonl
from edr.paths import DISTILL_CORE, ROUND2_OUT, resolve
from edr.round2.shared import dedupe_by_episode_output, load_h2_patches, loop_family_from_method, resolve_m1_adapter
from edr.scaffold.patches import build_prompt, relevant_patches_for_episode
from edr.verifier import call_success, parse_model_output


def assert_main_arm(rows, source_path):
    """Guard the T1 core against a STaR fork before it feeds the T1∪T2 signature."""

    if "star" in Path(source_path).name.lower():
        raise AssertionError({"assert": "t1_core_is_main_arm", "path": str(source_path)})
    star_rows = [
        row.get("episode_id")
        for row in rows
        if any(str(row.get(field, "")).lower().lstrip().startswith("star") for field in ("source", "arm", "round2_source"))
    ]
    if star_rows:
        raise AssertionError({"assert": "t1_core_no_star_source", "count": len(star_rows), "examples": star_rows[:20]})


def build_t1_t2(t1_rows, t2_rows):
    """Union T1 core and T2, deduped by episode_id (T1 kept on collision)."""

    seen = OrderedDict()
    for row in [*t1_rows, *t2_rows]:
        key = row.get("episode_id")
        if not key:
            raise AssertionError({"assert": "t1_t2_episode_id_present"})
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def teacher2_generate_row(runner, episode, patches, sample, max_new_tokens):
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


def generate_teacher2_samples(args, config):
    from edr.modeling import ModelRunner

    if not args.f2_json or not args.loop_output:
        raise SystemExit("--teacher2-samples or both --f2-json and --loop-output are required")
    f2_payload = read_json(resolve(args.f2_json))
    episodes = f2_payload.get("failures") or f2_payload.get("f2") or []
    patches = load_h2_patches(resolve(args.loop_output), family=loop_family_from_method(args.loop_method), seed=args.loop_seed)
    runner = ModelRunner(
        resolve_model_id(config),
        base_adapter_dir=resolve_m1_adapter(args.m1_adapter),
        require_adapter=True,
    )
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
        write_jsonl(resolve(args.teacher2_samples_output), rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher2-samples", default=None, help="Precomputed JSONL rows from M1+H2, train split only.")
    parser.add_argument("--f2-json", default=None)
    parser.add_argument("--loop-output", default=None)
    parser.add_argument("--loop-method", default="NL-evo", choices=["NL-evo", "NL-padded-evo", "STRUCT-evo"])
    parser.add_argument("--loop-seed", type=int, default=None)
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--t08-samples", type=int, default=4)
    parser.add_argument("--augment-to", type=int, default=8)
    parser.add_argument("--teacher2-samples-output", default=str(ROUND2_OUT / "teacher2_samples.jsonl"))
    parser.add_argument("--output", default=str(ROUND2_OUT / "t2.jsonl"))
    parser.add_argument("--t1-core", default=str(DISTILL_CORE))
    parser.add_argument("--t1t2-output", default=str(ROUND2_OUT / "t1_t2.jsonl"))
    parser.add_argument("--summary-output", default=str(ROUND2_OUT / "t2_summary.json"))
    args = parser.parse_args(argv)

    config = load_config("round2")
    raw_rows = read_jsonl(resolve(args.teacher2_samples)) if args.teacher2_samples else generate_teacher2_samples(args, config)
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
    for name, surface in load_eval_surface_ids().items():
        assertions.append(assert_disjoint(f"t2_intersect_{name}", ids, surface))

    status = "OK" if len(rows) >= 30 else "MATERIAL_EXHAUSTION"
    write_jsonl(resolve(args.output), rows)

    t1_core_path = resolve(args.t1_core)
    t1_rows = read_jsonl(t1_core_path)
    assert_main_arm(t1_rows, t1_core_path)
    t1_t2_rows = build_t1_t2(t1_rows, rows)
    write_jsonl(resolve(args.t1t2_output), t1_t2_rows)

    write_json(
        resolve(args.summary_output),
        {
            "probe": "round2_build_t2",
            "rows": len(rows),
            "raw_rows": len(raw_rows),
            "ast_rejected_rows": len(rejected_ast),
            "unique_episodes": len(ids),
            "status": status,
            "t1_core": str(t1_core_path),
            "t1_core_rows": len(t1_rows),
            "t1_t2_rows": len(t1_t2_rows),
            "t1_t2_output": str(args.t1t2_output),
            "assertions": assertions,
        },
    )
    print(f"wrote {args.output} rows={len(rows)} status={status}; wrote {args.t1t2_output} t1_t2_rows={len(t1_t2_rows)}")


if __name__ == "__main__":
    main()
