import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.phase0_probe import (
    DEFAULT_MODEL_ID,
    DEFAULT_SPLIT_SEED,
    Phase0Runner,
    evaluate_teacher_episode,
    failure_type,
    function_name,
    load_inputs,
    read_json,
    write_json,
)


def compare_reference(records, reference_path):
    if not reference_path:
        return None
    reference = read_json(reference_path)
    ref_by_id = {row["episode_id"]: row for row in reference.get("episodes", [])}
    changed = []
    missing = []
    for row in records:
        ref = ref_by_id.get(row["episode_id"])
        if ref is None:
            missing.append(row["episode_id"])
            continue
        if bool(ref.get("teacher", {}).get("success")) != bool(row["teacher"]["success"]):
            changed.append(
                {
                    "episode_id": row["episode_id"],
                    "reference_success": bool(ref.get("teacher", {}).get("success")),
                    "reanchor_success": bool(row["teacher"]["success"]),
                }
            )
    return {
        "reference_path": str(reference_path),
        "reference_probe": reference.get("probe"),
        "reference_model_id": reference.get("model_id"),
        "missing_in_reference": missing,
        "changed_teacher_success": changed,
        "changed_teacher_success_count": len(changed),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-run teacher T=0 on frozen heldout episodes in the pinned eval environment.")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-result", default="../EDG-EXP2-struct/results/evolution_main_merged.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--reference-partition", default="results/s01_heldout_pass16_partition.json")
    parser.add_argument("--output", default="results/s07_teacher_reanchor_heldout.json")
    args = parser.parse_args(argv)

    _, _, split, patches, evo = load_inputs(
        args.failures,
        args.exp2_result,
        args.exp2_root,
        args.split_seed,
    )
    heldout = split["held_out"]
    runner = Phase0Runner(args.model_id, cache_dir=args.model_cache_dir)

    records = []
    for index, episode in enumerate(heldout, start=1):
        if index == 1 or index % args.progress_every == 0 or index == len(heldout):
            print(f"[teacher-reanchor] episode {index}/{len(heldout)} {episode['episode_id']}", flush=True)
        teacher = evaluate_teacher_episode(evo, runner, episode, patches, args.max_new_tokens)
        records.append(
            {
                "episode_id": episode["episode_id"],
                "failure_class": failure_type(episode),
                "function_name": function_name(episode),
                "teacher": teacher,
            }
        )

    total = len(records)
    teacher_success = sum(1 for row in records if row["teacher"]["success"])
    patch_counts = Counter(row["teacher"].get("n_patches", 0) for row in records)
    payload = {
        "probe": "s07_teacher_reanchor_heldout",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "max_new_tokens": args.max_new_tokens,
        "stats": {
            "total": total,
            "teacher_success": teacher_success,
            "teacher_success_rate": teacher_success / total if total else 0.0,
            "patch_count_distribution": {str(key): value for key, value in sorted(patch_counts.items())},
        },
        "reference_comparison": compare_reference(records, args.reference_partition),
        "records": records,
    }
    write_json(args.output, payload)
    print(json.dumps(payload["stats"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
