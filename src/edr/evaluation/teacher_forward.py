"""Teacher (scaffold-on) forward: M0 + H1 patches on an episode set.

Reproduces the retention denominator — the frozen result is
data/round1/teacher_heldout_reference.json (50/158 on heldout). This is the
round-1 counterpart of edr.round2.teacher2_forward (which runs M1 + H2).

CLI:
  python -m edr.evaluation.teacher_forward \
      [--episode-ids data/round1/episode_ids/d_heldout_failures.json] \
      [--output outputs/round1_build/teacher_heldout.json]
"""

from __future__ import annotations

import argparse

from edr.config import load_config, resolve_model_id
from edr.data.leakage import load_episode_id_file
from edr.data.splits import load_round1_inputs
from edr.io_utils import write_json
from edr.paths import EPISODE_ID_DIR, OUTPUT_DIR, resolve
from edr.scaffold.patches import build_prompt, relevant_patches_for_episode
from edr.verifier import call_success, parse_model_output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-ids", default=str(EPISODE_ID_DIR / "d_heldout_failures.json"))
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "round1_build" / "teacher_heldout.json"))
    args = parser.parse_args(argv)

    config = load_config()
    baseline, _evolution, _split, patches = load_round1_inputs(config["split_seed"])
    ids = load_episode_id_file(resolve(args.episode_ids))
    episodes = [row for row in baseline["records"] if str(row["episode_id"]) in ids]
    if len(episodes) != len(ids):
        raise SystemExit({"missing_episode_count": len(ids) - len(episodes)})

    from edr.modeling import ModelRunner

    runner = ModelRunner(resolve_model_id(config), cache_dir=args.model_cache_dir)
    records = []
    for index, episode in enumerate(episodes, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(episodes):
            print(f"[teacher-forward] {index}/{len(episodes)} {episode['episode_id']}", flush=True)
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
        "probe": "teacher_forward",
        "model_stack": ["M0", "H1_patch_in_context"],
        "episode_n": len(records),
        "success": sum(row["success"] for row in records),
        "success_rate": sum(row["success"] for row in records) / len(records) if records else 0.0,
        "records": records,
    }
    write_json(resolve(args.output), output)
    print(f"wrote {args.output} success={output['success']}/{output['episode_n']}")


if __name__ == "__main__":
    main()
