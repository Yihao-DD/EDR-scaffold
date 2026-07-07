#!/usr/bin/env python3
"""Acceptance check: this machine must reproduce the designated round-1 model.

Static mode (--skip-model-check): pytest + shipped-file validation, prints
PACKAGE_ONLY. Full mode: re-evaluates M1 on heldout and sibling surfaces and
prints ACCEPTED iff the results match the frozen reference within tolerance
(heldout ±1 episode, sibling ±2). Running the round-2 iteration on a machine
that does not reproduce M1 is invalid.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edr.config import load_config, resolve_model_id  # noqa: E402
from edr.io_utils import read_json  # noqa: E402
from edr.paths import RECONCILE_OUT, resolve  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true", help="Static package validation only; does not print ACCEPTED.")
    parser.add_argument("--m1-adapter", default=None)
    parser.add_argument("--output-prefix", default=str(RECONCILE_OUT / "m1"))
    args = parser.parse_args(argv)

    config = load_config("round2")
    m1_adapter = resolve(args.m1_adapter or config["m1_adapter"])
    reference = config["m1_reference"]

    if not args.skip_pytest:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=REPO_ROOT, check=True)

    if args.skip_model_check:
        print("PACKAGE_ONLY: static files and pytest passed; model re-evaluation was skipped.")
        return

    adapter_file = m1_adapter / "adapter_model.safetensors"
    if not adapter_file.exists() or adapter_file.stat().st_size < 1_000_000:
        raise SystemExit(f"M1 adapter missing or incomplete: {adapter_file} — run tools/fetch_adapters.py")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    heldout_json = Path(str(prefix) + ".heldout.json")
    sibling_json = Path(str(prefix) + ".sibling.json")

    from edr.evaluation.heldout import main as heldout_main
    from edr.evaluation.sibling import main as sibling_main

    common = [
        "--model-id",
        resolve_model_id(config),
        "--adapter-dir",
        str(m1_adapter),
        "--arm",
        "round1_main",
        "--config",
        "m1_reconcile",
        "--seed",
        str(reference["seed"]),
        "--run-id",
        "m1_reconcile",
    ]
    heldout_main([*common, "--train-dataset", "data/round1/distill_train.jsonl", "--output", str(heldout_json)])
    sibling_main([*common, "--output", str(sibling_json)])

    heldout = read_json(heldout_json)["summary"]["heldout_repair"]
    sibling = read_json(sibling_json)["summary"]["success"]
    if abs(heldout - reference["expected_heldout_repair_count"]) > 1:
        raise SystemExit(f"heldout reference outside +/-1 episode (got {heldout}, expected {reference['expected_heldout_repair_count']})")
    if abs(sibling - reference["expected_sibling_success_count"]) > 2:
        raise SystemExit(f"sibling reference outside +/-2 episodes (got {sibling}, expected {reference['expected_sibling_success_count']})")
    print("ACCEPTED")


if __name__ == "__main__":
    main()
