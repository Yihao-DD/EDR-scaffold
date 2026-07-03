#!/usr/bin/env python3
"""Evaluate M2 = M0 + A1 + A2 on Phase 2 surfaces."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import add_common_args, write_json


def run(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print("DRY_RUN", " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--m1-adapter", required=True)
    parser.add_argument("--a2-adapter", required=True)
    parser.add_argument("--train-signature-jsonl", default="round2_outputs/t1_t2.jsonl")
    parser.add_argument("--teacher2-loop-output", default=None, help="Loop output with H2 patches for teacher-2 = M1+H2.")
    parser.add_argument("--teacher2-loop-method", default="NL-evo")
    parser.add_argument("--teacher2-loop-seed", type=int, default=None)
    parser.add_argument("--output-prefix", default="round2_outputs/eval/m2")
    parser.add_argument("--heldout-eval", default="repro_rep2/scripts/s07_heldout_eval.py")
    parser.add_argument("--sibling-eval", default="repro_rep2/scripts/s07_sibling_arena_eval.py")
    parser.add_argument("--phase0-eval", default="repro_rep2/scripts/lora_phase0.py")
    parser.add_argument("--phase0-failures", default=None, help="Optional EXP1 failures JSON for val@156 and old400 continuity.")
    parser.add_argument("--phase0-exp2-root", default=None, help="Optional EXP2 root for val@156 and old400 continuity.")
    parser.add_argument("--phase0-s00-input", default=None, help="Optional s00 inventory for val@156 and old400 continuity.")
    parser.add_argument("--teacher2-forward", default="round2/loop/teacher2_forward.py")
    args = parser.parse_args()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "probe": "round2_eval_plan",
        "model_stack": ["M0", args.m1_adapter, args.a2_adapter],
        "teacher2_loop_output": args.teacher2_loop_output,
        "surfaces": ["heldout158", "sibling300", "old400", "val156"],
        "seen_definition": "T1 union T2 for 2x2 function/error labels",
        "train_signature_jsonl": args.train_signature_jsonl,
    }
    write_json(str(prefix) + ".plan.json", plan)
    common = [
        "--model-id",
        "Qwen/Qwen2.5-7B-Instruct",
        "--adapter-dir",
        args.a2_adapter,
        "--base-adapter-dir",
        args.m1_adapter,
        "--arm",
        "main",
        "--config",
        "round2",
        "--seed",
        "0",
        "--run-id",
        "round2_eval",
    ]
    run(
        [
            "python3",
            args.heldout_eval,
            *common,
            "--train-dataset",
            args.train_signature_jsonl,
            "--output",
            str(prefix) + ".heldout.json",
        ],
        args.dry_run,
    )
    run(["python3", args.sibling_eval, *common, "--output", str(prefix) + ".sibling.json"], args.dry_run)
    if args.phase0_failures and args.phase0_exp2_root and args.phase0_s00_input:
        run(
            [
                "python3",
                args.phase0_eval,
                "--model-id",
                "Qwen/Qwen2.5-7B-Instruct",
                "--failures",
                args.phase0_failures,
                "--exp2-root",
                args.phase0_exp2_root,
                "--s00-input",
                args.phase0_s00_input,
                "eval",
                "--adapter-dir",
                args.a2_adapter,
                "--base-adapter-dir",
                args.m1_adapter,
                "--output",
                str(prefix) + ".val_old400.json",
            ],
            args.dry_run,
        )
    else:
        print("SKIP val_old400: pass --phase0-failures, --phase0-exp2-root, and --phase0-s00-input to enable it.")
    if args.teacher2_loop_output:
        cmd = [
            "python3",
            args.teacher2_forward,
            "--loop-output",
            args.teacher2_loop_output,
            "--loop-method",
            args.teacher2_loop_method,
            "--manifest",
            "repro_rep2/MANIFEST.json",
            "--base-adapter-dir",
            args.m1_adapter,
            "--output",
            str(prefix) + ".teacher2_heldout.json",
        ]
        if args.teacher2_loop_seed is not None:
            cmd.extend(["--loop-seed", str(args.teacher2_loop_seed)])
        run(cmd, args.dry_run)
    else:
        print("SKIP teacher2: pass --teacher2-loop-output to evaluate M1+H2 with the adapter-aware loop entry.")


if __name__ == "__main__":
    main()
