#!/usr/bin/env python3
"""Train A2 over M0+A1 with KL anchored to frozen M1.

This is a thin, auditable launcher. The Phase 0 trainer already implements
LoRA, replay, KL, gradient clipping, and non-finite guards. Round 2 changes the
anchor semantics: KL reference is M1 rather than M0. The launcher writes an
explicit run plan and refuses to run unless an M1 adapter path is supplied.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import ROUND2_RECIPE, ROUND2_SEEDS, add_common_args, write_json


def combine_dataset(train_jsonl: Path, replay_jsonl: Path, output_dir: Path, seed: int) -> Path:
    combined = output_dir / f"round2_train_seed{seed}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with combined.open("w", encoding="utf-8") as out:
        for source in (train_jsonl, replay_jsonl):
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--m1-adapter", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--replay-jsonl", required=True)
    parser.add_argument("--seed", type=int, choices=ROUND2_SEEDS, required=True)
    parser.add_argument("--trainer", default="repro_rep2/scripts/lora_phase0.py")
    parser.add_argument("--output-adapter-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--plan-output", default=None)
    args = parser.parse_args()

    train_jsonl = Path(args.train_jsonl)
    replay_jsonl = Path(args.replay_jsonl)
    output_adapter_dir = Path(args.output_adapter_dir)
    combined_dataset = combine_dataset(train_jsonl, replay_jsonl, output_adapter_dir, args.seed)

    plan = {
        "probe": "round2_train_plan",
        "seed": args.seed,
        "recipe": ROUND2_RECIPE,
        "m1_adapter": args.m1_adapter,
        "train_jsonl": args.train_jsonl,
        "replay_jsonl": args.replay_jsonl,
        "combined_dataset": str(combined_dataset),
        "output_adapter_dir": args.output_adapter_dir,
        "output_json": args.output_json,
        "anchor_invariant": "KL reference model must be M1 (base + merged A1), not raw M0.",
    }
    plan_output = args.plan_output or str(Path(args.output_json).with_suffix(".plan.json"))
    write_json(plan_output, plan)

    cmd = [
        "python3",
        args.trainer,
        "--model-id",
        "Qwen/Qwen2.5-7B-Instruct",
        "train",
        "--dataset",
        str(combined_dataset),
        "--base-adapter-dir",
        args.m1_adapter,
        "--output-dir",
        args.output_adapter_dir,
        "--rank",
        "16",
        "--lr",
        "5e-5",
        "--seed",
        str(args.seed),
        "--epochs",
        "3",
        "--kl-anchor-lambda",
        "2",
        "--max-grad-norm",
        "1.0",
    ]
    if args.dry_run:
        print("DRY_RUN", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
