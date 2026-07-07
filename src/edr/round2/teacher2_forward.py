"""Teacher-2 = M1(merged) + H2 patch-in-context, forwarded on an episode set.

Seed-independent; provides the retention_2 denominator.

CLI:
  python -m edr.round2.teacher2_forward --loop-output <json> [--m1-adapter <dir>]
"""

from __future__ import annotations

import argparse

from edr.config import load_config, resolve_model_id
from edr.data.leakage import load_episode_id_file
from edr.io_utils import read_json, write_json
from edr.paths import BASE_FAILURES, EPISODE_ID_DIR, ROUND2_OUT, resolve
from edr.round2.shared import load_h2_patches, loop_family_from_method, resolve_m1_adapter
from edr.scaffold.patches import build_prompt, relevant_patches_for_episode
from edr.verifier import call_success, parse_model_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", default=str(BASE_FAILURES))
    parser.add_argument("--episode-ids", default=str(EPISODE_ID_DIR / "d_heldout_failures.json"))
    parser.add_argument("--loop-output", required=True)
    parser.add_argument("--loop-method", default="NL-evo", choices=["NL-evo", "NL-padded-evo", "STRUCT-evo"])
    parser.add_argument("--loop-seed", type=int, default=None)
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default=str(ROUND2_OUT / "eval" / "teacher2.heldout.json"))
    args = parser.parse_args(argv)

    config = load_config("round2")
    baseline = read_json(resolve(args.failures))
    ids = load_episode_id_file(resolve(args.episode_ids))
    episodes = [row for row in baseline["records"] if str(row["episode_id"]) in ids]
    if len(episodes) != len(ids):
        raise SystemExit({"missing_episode_count": len(ids) - len(episodes)})
    patches = load_h2_patches(resolve(args.loop_output), loop_family_from_method(args.loop_method), args.loop_seed)

    from edr.modeling import ModelRunner

    runner = ModelRunner(
        resolve_model_id(config),
        cache_dir=args.model_cache_dir,
        base_adapter_dir=resolve_m1_adapter(args.m1_adapter),
        require_adapter=True,
    )
    records = []
    for episode in episodes:
        relevant = relevant_patches_for_episode(patches, episode)
        prompt = build_prompt(episode, relevant)
        raw = runner.generate(prompt, max_new_tokens=args.max_new_tokens)
        try:
            prediction = parse_model_output(raw)
            parse_error = None
            success = call_success(prediction, episode["ground_truth_call"])
        except Exception as error:
            prediction = None
            parse_error = str(error)
            success = False
        records.append(
            {
                "episode_id": episode["episode_id"],
                "success": success,
                "prediction": prediction,
                "raw_model_output": raw,
                "parse_error": parse_error,
                "patch_ids": [patch.patch_id for patch in relevant],
                "n_patches": len(relevant),
            }
        )
    output = {
        "probe": "teacher2_forward",
        "model_stack": ["M0", "A1_merged_in_memory", "H2_patch_in_context"],
        "loop_output": args.loop_output,
        "loop_method": args.loop_method,
        "loop_seed": args.loop_seed,
        "episode_n": len(records),
        "success": sum(row["success"] for row in records),
        "success_rate": sum(row["success"] for row in records) / len(records) if records else 0.0,
        "records": records,
    }
    write_json(resolve(args.output), output)
    print(f"wrote {args.output} success={output['success']}/{output['episode_n']}")


if __name__ == "__main__":
    main()
