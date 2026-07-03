#!/usr/bin/env python3
"""Evaluate teacher-2 = M1(merged) + H2 patch-in-context on an episode set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOOP_DIR.parents[1]
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


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_episode_id_file(path: str | Path) -> set[str]:
    payload = read_json(path)
    values = payload.get("episode_ids") if isinstance(payload, dict) else payload
    if values is None:
        raise SystemExit(f"{path} does not contain episode_ids")
    return {str(item) for item in values}


def loop_family_from_method(method: str) -> str:
    return method.replace("-evo", "")


def load_h2_patches(loop_output: Path, family: str, seed: int | None) -> list[PatchCandidate]:
    payload = read_json(loop_output)
    per_seed = payload.get("per_seed") or []
    selected = None
    for item in per_seed:
        if seed is None or item.get("seed") == seed:
            selected = item
            break
    if selected is None:
        raise SystemExit(f"loop seed not found: {seed}")
    loop_outputs = selected.get("loop_outputs") or {}
    if family not in loop_outputs:
        raise SystemExit(f"loop family not found: {family}; available={sorted(loop_outputs)}")
    return [PatchCandidate(**row) for row in loop_outputs[family].get("accepted_patches", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", default="EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--episode-ids", default="repro_rep2/data/episode_ids/D_heldout_failures.json")
    parser.add_argument("--loop-output", required=True)
    parser.add_argument("--loop-method", default="NL-evo", choices=["NL-evo", "NL-padded-evo", "STRUCT-evo"])
    parser.add_argument("--loop-seed", type=int, default=None)
    parser.add_argument("--manifest", default="repro_rep2/MANIFEST.json")
    parser.add_argument("--base-adapter-dir", default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="round2_outputs/eval/teacher2.heldout.json")
    args = parser.parse_args()

    baseline = read_json(args.failures)
    ids = load_episode_id_file(args.episode_ids)
    episodes = [row for row in baseline["records"] if str(row["episode_id"]) in ids]
    if len(episodes) != len(ids):
        raise SystemExit({"missing_episode_count": len(ids) - len(episodes)})
    patches = load_h2_patches(Path(args.loop_output), loop_family_from_method(args.loop_method), args.loop_seed)
    base_adapter = (
        _resolve_repo_path(args.base_adapter_dir)
        if args.base_adapter_dir
        else _manifest_m1_adapter_dir(args.manifest)
    )
    runner = ModelRunner(args.model_id, cache_dir=args.model_cache_dir, base_adapter_dir=str(base_adapter))
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
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} success={output['success']}/{output['episode_n']}")


if __name__ == "__main__":
    main()
