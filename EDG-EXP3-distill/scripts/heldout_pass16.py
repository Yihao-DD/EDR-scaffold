import argparse
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
    partition_episode,
    sample_base_episode,
    write_json,
)


def write_log(path, payload):
    stats = payload["stats"]
    lines = [
        "# Heldout pass@16 Partition",
        "",
        "Status: pure-forward evaluation label generation. This does not touch training data or model selection.",
        "",
        f"- split_seed: {payload['split_seed']}",
        f"- sample_count: {payload['sample_count']}",
        f"- temperature: {payload['temperature']}",
        f"- total_heldout: {stats['total']}",
        f"- sampling_rescuable: {stats['sampling_rescuable']} ({stats['sampling_rescuable_rate']:.4f})",
        f"- scaffold_only: {stats['scaffold_only']} ({stats['scaffold_only_rate']:.4f})",
        f"- neither: {stats['neither']} ({stats['neither_rate']:.4f})",
        f"- teacher_success: {stats['teacher_success']} ({stats['teacher_success_rate']:.4f})",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pure-forward heldout pass@16 and scaffold partition labels.")
    parser.add_argument("--failures", default="../EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-result", default="../EDG-EXP2-struct/results/evolution_main_merged.json")
    parser.add_argument("--exp2-root", default="../EDG-EXP2-struct")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", default="results/s01_heldout_pass16_partition.json")
    parser.add_argument("--log-output", default="logs/s01_heldout_pass16.md")
    args = parser.parse_args(argv)

    baseline, exp2_result, split, patches, evo = load_inputs(
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
            print(f"[heldout-pass16] episode {index}/{len(heldout)} {episode['episode_id']}", flush=True)
        teacher = evaluate_teacher_episode(evo, runner, episode, patches, args.max_new_tokens)
        pass16 = sample_base_episode(
            evo,
            runner,
            episode,
            baseline["prompt_template"],
            args.sample_count,
            args.sample_batch_size,
            args.temperature,
            args.max_new_tokens,
            args.split_seed,
        )
        records.append(
            {
                "episode_id": episode["episode_id"],
                "failure_class": failure_type(episode),
                "function_name": function_name(episode),
                "teacher": teacher,
                "pass16": pass16,
                "partition": partition_episode(pass16["pass16_hits"], teacher["success"]),
            }
        )

    counts = Counter(row["partition"] for row in records)
    teacher_success = sum(1 for row in records if row["teacher"]["success"])
    total = len(records)
    stats = {
        "total": total,
        "sampling_rescuable": counts["sampling_rescuable"],
        "scaffold_only": counts["scaffold_only"],
        "neither": counts["neither"],
        "sampling_rescuable_rate": counts["sampling_rescuable"] / total if total else 0.0,
        "scaffold_only_rate": counts["scaffold_only"] / total if total else 0.0,
        "neither_rate": counts["neither"] / total if total else 0.0,
        "teacher_success": teacher_success,
        "teacher_success_rate": teacher_success / total if total else 0.0,
    }
    payload = {
        "probe": "heldout_pass16_partition",
        "model_id": args.model_id,
        "split_seed": args.split_seed,
        "sample_count": args.sample_count,
        "temperature": args.temperature,
        "stats": stats,
        "episodes": records,
    }
    write_json(args.output, payload)
    write_log(args.log_output, payload)
    print(
        f"heldout total={total} scaffold_only={stats['scaffold_only']} "
        f"sampling_rescuable={stats['sampling_rescuable']} neither={stats['neither']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
