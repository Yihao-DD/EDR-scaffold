import argparse
import subprocess
import sys
from pathlib import Path


MODEL_ID = "/root/autodl-tmp/hf/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"


def run_specs():
    specs = []
    seeds = [20260703, 20260704, 20260705, 20260706, 20260707]
    for seed in seeds:
        if seed == 20260703:
            main_rep1 = "adapters/s05_blocked_kl/main/main_ep3_r16_lr5e-5_replay1_seed20260703"
            star_rep1 = "adapters/s05_blocked_kl/star/star_ep3_r16_lr1e-4_replay1_seed20260703"
            main_rep2 = "adapters/s06_d3_capped/main/main_d3c_r16_lr5e-5_ep3_replay2_lam2_seed20260703"
            star_rep2 = "adapters/s07_recovered/star/star_d3c_r16_lr1e-4_ep3_replay2_lam2_seed20260703_v2"
        elif seed == 20260704:
            main_rep1 = "adapters/s07_recovered/main/main_d2_rep1_r16_lr5e-5_ep3_replay1_seed20260704_v2"
            star_rep1 = "adapters/s07_d2_5seed/star/star_d2_rep1_r16_lr1e-4_ep3_replay1_seed20260704"
            main_rep2 = "adapters/s07_d2_5seed/main/main_d2_rep2_r16_lr5e-5_ep3_replay2_capped_lam2_seed20260704"
            star_rep2 = "adapters/s07_d2_5seed/star/star_d2_rep2_r16_lr1e-4_ep3_replay2_capped_lam2_seed20260704"
        else:
            main_rep1 = f"adapters/s07_d2_5seed/main/main_d2_rep1_r16_lr5e-5_ep3_replay1_seed{seed}"
            star_rep1 = f"adapters/s07_d2_5seed/star/star_d2_rep1_r16_lr1e-4_ep3_replay1_seed{seed}"
            main_rep2 = f"adapters/s07_d2_5seed/main/main_d2_rep2_r16_lr5e-5_ep3_replay2_capped_lam2_seed{seed}"
            star_rep2 = f"adapters/s07_d2_5seed/star/star_d2_rep2_r16_lr1e-4_ep3_replay2_capped_lam2_seed{seed}"

        specs.extend(
            [
                {
                    "arm": "main",
                    "config": "rep1_ep3_replay1",
                    "seed": seed,
                    "adapter": main_rep1,
                    "dataset": "data/distill_main_replay1.jsonl",
                },
                {
                    "arm": "star",
                    "config": "rep1_ep3_replay1",
                    "seed": seed,
                    "adapter": star_rep1,
                    "dataset": "data/distill_star_replay1.jsonl",
                },
                {
                    "arm": "main",
                    "config": "rep2_ep3_replay2_lam2_capped",
                    "seed": seed,
                    "adapter": main_rep2,
                    "dataset": "data/distill_main_replay2_capped.jsonl",
                },
                {
                    "arm": "star",
                    "config": "rep2_ep3_replay2_lam2_capped",
                    "seed": seed,
                    "adapter": star_rep2,
                    "dataset": "data/distill_star_replay2_capped.jsonl",
                },
            ]
        )
    return specs


def run_command(command):
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run s07 final heldout and sibling forward evaluations.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)

    root = Path(args.root)
    failures = "../EDG-EXP1/results/a2_failures.json"
    exp2_root = "../EDG-EXP2-struct"
    failures_path = root / failures
    exp2_path = root / exp2_root
    if not failures_path.exists():
        raise SystemExit(f"missing failures path: {failures_path}")
    if not exp2_path.exists():
        raise SystemExit(f"missing exp2 root: {exp2_path}")

    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, shard-count)")

    specs = [
        spec
        for index, spec in enumerate(run_specs())
        if index % args.shard_count == args.shard_index
    ]
    print(
        f"[queue] shard {args.shard_index}/{args.shard_count} has {len(specs)} checkpoints",
        flush=True,
    )

    for spec in specs:
        adapter = root / spec["adapter"]
        if not adapter.exists():
            raise SystemExit(f"missing adapter: {adapter}")
        run_id = f"{spec['arm']}_{spec['config']}_seed{spec['seed']}"
        heldout_out = root / "results" / "s07_final_heldout" / spec["arm"] / f"{run_id}.json"
        sibling_out = root / "results" / "s07_final_sibling" / spec["arm"] / f"{run_id}.json"

        if not heldout_out.exists():
            heldout_out.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.s07_heldout_eval",
                    "--model-id",
                    args.model_id,
                    "--adapter-dir",
                    spec["adapter"],
                    "--arm",
                    spec["arm"],
                    "--config",
                    spec["config"],
                    "--seed",
                    str(spec["seed"]),
                    "--run-id",
                    run_id,
                    "--train-dataset",
                    spec["dataset"],
                    "--output",
                    str(heldout_out.relative_to(root)),
                    "--progress-every",
                    "25",
                ]
            )
        else:
            print(f"[skip] heldout exists {heldout_out}", flush=True)

        if not sibling_out.exists():
            sibling_out.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.s07_sibling_arena_eval",
                    "--model-id",
                    args.model_id,
                    "--adapter-dir",
                    spec["adapter"],
                    "--arm",
                    spec["arm"],
                    "--config",
                    spec["config"],
                    "--seed",
                    str(spec["seed"]),
                    "--run-id",
                    run_id,
                    "--output",
                    str(sibling_out.relative_to(root)),
                    "--progress-every",
                    "25",
                ]
            )
        else:
            print(f"[skip] sibling exists {sibling_out}", flush=True)


if __name__ == "__main__":
    main()
