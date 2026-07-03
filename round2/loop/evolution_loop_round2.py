#!/usr/bin/env python3
"""Minimal Phase 2 NL-evo loop entrypoint.

This wrapper records immutable inputs and delegates the actual patch search to
the vendored EDG-EXP2-struct loop. It is intentionally thin so reviewers can
verify that acceptance remains D_val-only and AST-only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from round2.common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-loop", default="EDG-EXP2-struct/scripts/evolution_loop.py")
    parser.add_argument("--m1-adapter", required=True)
    parser.add_argument("--f2", required=True)
    parser.add_argument("--d-val", default="repro_rep2/data/episode_ids/D_val_failures.json")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seeds", default="20260630")
    parser.add_argument("--output-dir", default="round2_outputs/loop")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plan = {
        "probe": "round2_loop_plan",
        "vendor_loop": args.vendor_loop,
        "vendor_source": "EDG-EXP2-struct/scripts/evolution_loop.py from this repository.",
        "m1_adapter": args.m1_adapter,
        "f2": args.f2,
        "acceptance": "D_val AST only; F2 was collected from M0+A1 before entering this patch loop.",
        "d_val": args.d_val,
        "adapter_note": "The upstream patch loop does not load LoRA adapters. Adapter conditioning happens in F2 collection; adapter-aware patch search remains a future extension.",
    }
    write_json(Path(args.output_dir) / "loop_plan.json", plan)
    vendor_loop = Path(args.vendor_loop)
    if not vendor_loop.exists():
        raise SystemExit(f"vendor loop not found: {vendor_loop}")
    cmd = [
        "python3",
        str(vendor_loop),
        "--input",
        args.f2,
        "--output",
        str(Path(args.output_dir) / "evolution_loop.json"),
        "--log",
        str(Path(args.output_dir) / "evolution_loop.md"),
        "--model-id",
        args.model_id,
        "--seeds",
        args.seeds,
    ]
    if args.dry_run:
        print("DRY_RUN", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
