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
    parser.add_argument("--vendor-loop", default="EDG-EXP2-struct/scripts/evolution_main.py")
    parser.add_argument("--m1-adapter", required=True)
    parser.add_argument("--f2", required=True)
    parser.add_argument("--d-val", default="repro_rep2/data/episode_ids/D_val_failures.json")
    parser.add_argument("--output-dir", default="round2_outputs/loop")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plan = {
        "probe": "round2_loop_plan",
        "vendor_loop": args.vendor_loop,
        "vendor_source": "EDG-EXP2-struct minimal NL-evo loop; record source commit before production use.",
        "m1_adapter": args.m1_adapter,
        "f2": args.f2,
        "acceptance": "D_val AST only",
        "d_val": args.d_val,
    }
    write_json(Path(args.output_dir) / "loop_plan.json", plan)
    cmd = ["python3", args.vendor_loop, "--adapter", args.m1_adapter, "--failures", args.f2, "--validation-ids", args.d_val]
    if args.dry_run:
        print("DRY_RUN", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
