"""Evaluate M2 = M0 + A1 + A2 on the round-2 surfaces (heldout + sibling,
optional val/old400 monitor), plus retention_2 against teacher-2.

Evaluators are invoked in-process (no subprocess path fragility).

CLI:
  python -m edr.round2.evaluate --a2-adapter <dir> [--m1-adapter <dir>] \
      [--teacher2-loop-output <loop json>] --output-prefix outputs/round2/eval/m2_seed<seed>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edr.config import load_config, resolve_model_id
from edr.io_utils import read_json, write_json
from edr.paths import ROUND2_OUT, resolve
from edr.round2.shared import resolve_m1_adapter


def write_retention2(prefix):
    """retention_2 = repair(M2 no-patch) / repair(M1+H2), both on heldout."""

    heldout = read_json(str(prefix) + ".heldout.json")["summary"]
    teacher2 = read_json(str(prefix) + ".teacher2_heldout.json")
    m2_rate = heldout["heldout_repair_rate"]
    teacher2_rate = teacher2["success_rate"]
    retention2 = (m2_rate / teacher2_rate) if teacher2_rate else None
    payload = {
        "probe": "round2_retention2",
        "definition": "repair(M2 no-patch, heldout) / repair(M1+H2, heldout)",
        "m2_heldout_repair": heldout["heldout_repair"],
        "m2_heldout_n": heldout["heldout_n"],
        "m2_repair_rate": m2_rate,
        "teacher2_success": teacher2["success"],
        "teacher2_n": teacher2["episode_n"],
        "teacher2_repair_rate": teacher2_rate,
        "retention_2": retention2,
    }
    write_json(str(prefix) + ".retention2.json", payload)
    print(f"retention_2={retention2} (M2={m2_rate:.4f} / teacher2={teacher2_rate:.4f})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--a2-adapter", required=True)
    parser.add_argument("--train-signature-jsonl", default=str(ROUND2_OUT / "t1_t2.jsonl"))
    parser.add_argument("--teacher2-loop-output", default=None, help="Loop output with H2 patches; runs teacher-2 + retention_2.")
    parser.add_argument("--teacher2-loop-method", default="NL-evo")
    parser.add_argument("--teacher2-loop-seed", type=int, default=None)
    parser.add_argument("--output-prefix", default=str(ROUND2_OUT / "eval" / "m2"))
    parser.add_argument("--with-val-old400", action="store_true", help="Also run the val/regression monitor surface.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config("round2")
    model_id = resolve_model_id(config)
    m1_adapter = resolve_m1_adapter(args.m1_adapter)
    a2_adapter = resolve(args.a2_adapter)
    prefix = resolve(args.output_prefix)
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    plan = {
        "probe": "round2_eval_plan",
        "model_stack": ["M0", str(m1_adapter), str(a2_adapter)],
        "teacher2_loop_output": args.teacher2_loop_output,
        "surfaces": ["heldout158", "sibling300"] + (["val156+old400"] if args.with_val_old400 else []),
        "seen_definition": "T1 union T2 for 2x2 function/error labels",
        "train_signature_jsonl": args.train_signature_jsonl,
    }
    write_json(str(prefix) + ".plan.json", plan)

    common = [
        "--model-id",
        model_id,
        "--adapter-dir",
        str(a2_adapter),
        "--base-adapter-dir",
        str(m1_adapter),
        "--arm",
        "round2",
        "--config",
        "round2_locked",
        "--seed",
        "0",
        "--run-id",
        "round2_eval",
    ]
    heldout_argv = [
        *common,
        "--train-dataset",
        args.train_signature_jsonl,
        "--output",
        str(prefix) + ".heldout.json",
    ]
    sibling_argv = [*common, "--output", str(prefix) + ".sibling.json"]

    if args.dry_run:
        print("DRY_RUN python -m edr.evaluation.heldout " + " ".join(heldout_argv))
        print("DRY_RUN python -m edr.evaluation.sibling " + " ".join(sibling_argv))
        if args.teacher2_loop_output:
            print("DRY_RUN python -m edr.round2.teacher2_forward --loop-output " + args.teacher2_loop_output)
            print("SKIP retention_2: requires a full (non-dry-run) M2 heldout + teacher-2 forward.")
        return

    from edr.evaluation.heldout import main as heldout_main
    from edr.evaluation.sibling import main as sibling_main

    heldout_main(heldout_argv)
    sibling_main(sibling_argv)

    if args.with_val_old400:
        from edr.evaluation.validation import main as validation_main

        validation_main(
            [
                "--model-id",
                model_id,
                "--adapter-dir",
                str(a2_adapter),
                "--base-adapter-dir",
                str(m1_adapter),
                "--output",
                str(prefix) + ".val_old400.json",
            ]
        )

    if args.teacher2_loop_output:
        from edr.round2.teacher2_forward import main as teacher2_main

        teacher2_argv = [
            "--loop-output",
            args.teacher2_loop_output,
            "--loop-method",
            args.teacher2_loop_method,
            "--output",
            str(prefix) + ".teacher2_heldout.json",
        ]
        if args.m1_adapter:
            teacher2_argv.extend(["--m1-adapter", args.m1_adapter])
        if args.teacher2_loop_seed is not None:
            teacher2_argv.extend(["--loop-seed", str(args.teacher2_loop_seed)])
        teacher2_main(teacher2_argv)
        write_retention2(prefix)
    else:
        print("SKIP teacher2: pass --teacher2-loop-output to evaluate M1+H2 and compute retention_2.")


if __name__ == "__main__":
    main()
