"""Train A2 over M0+A1 with KL anchored to frozen M1 (locked recipe).

Thin, auditable launcher over edr.training.lora (called in-process). Recipe
and seeds come from configs/{base,round2}.json and are preregistered; the
launcher writes an explicit run plan and refuses to run without M1.

CLI:
  python -m edr.round2.train --train-jsonl <t2> --replay-jsonl <replay2> \
      --seed 20260708 --output-adapter-dir <dir> [--m1-adapter <dir>] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edr.config import load_config, resolve_model_id
from edr.io_utils import write_json
from edr.paths import resolve
from edr.round2.shared import resolve_m1_adapter


def combine_dataset(train_jsonl, replay_jsonl, output_dir, seed):
    combined = Path(output_dir) / f"round2_train_seed{seed}.jsonl"
    combined.parent.mkdir(parents=True, exist_ok=True)
    with combined.open("w", encoding="utf-8") as out:
        for source in (train_jsonl, replay_jsonl):
            with Path(source).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--replay-jsonl", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-adapter-dir", required=True)
    parser.add_argument("--plan-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config("round2")
    if args.seed not in config["seeds"]:
        raise SystemExit(f"seed {args.seed} is not in the preregistered round-2 seed block {config['seeds']}")
    recipe = config["recipe_locked"]
    m1_adapter = resolve_m1_adapter(args.m1_adapter)
    output_adapter_dir = resolve(args.output_adapter_dir)
    combined_dataset = combine_dataset(resolve(args.train_jsonl), resolve(args.replay_jsonl), output_adapter_dir, args.seed)

    plan = {
        "probe": "round2_train_plan",
        "seed": args.seed,
        "recipe": recipe,
        "m1_adapter": str(m1_adapter),
        "train_jsonl": args.train_jsonl,
        "replay_jsonl": args.replay_jsonl,
        "combined_dataset": str(combined_dataset),
        "output_adapter_dir": str(output_adapter_dir),
        "anchor_invariant": "KL reference model must be M1 (base + merged A1), not raw M0.",
    }
    plan_output = args.plan_output or str(output_adapter_dir / "train_plan.json")
    write_json(plan_output, plan)

    trainer_argv = [
        "--model-id",
        resolve_model_id(config),
        "--dataset",
        str(combined_dataset),
        "--base-adapter-dir",
        str(m1_adapter),
        "--output-dir",
        str(output_adapter_dir),
        "--rank",
        str(recipe["rank"]),
        "--lr",
        str(recipe["lr"]),
        "--seed",
        str(args.seed),
        "--epochs",
        str(recipe["epochs"]),
        "--kl-anchor-lambda",
        str(recipe["kl_anchor_lambda"]),
        "--max-grad-norm",
        "1.0",
    ]
    if args.dry_run:
        print("DRY_RUN python -m edr.training.lora " + " ".join(trainer_argv))
        return
    from edr.training.lora import main as train_main

    train_main(trainer_argv)


if __name__ == "__main__":
    main()
