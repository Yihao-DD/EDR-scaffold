#!/usr/bin/env python3
"""Company acceptance check for the rep2 handoff package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from common import add_common_args, read_json, REPO_ROOT


def _run(cmd: list[str]) -> None:
    """v1.26 audit fix: spawned s07 evaluators import `scripts.*` against repro_rep2."""

    env = dict(os.environ)
    repro = str(REPO_ROOT / "repro_rep2")
    env["PYTHONPATH"] = repro + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--manifest", default="repro_rep2/MANIFEST.json")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true", help="Only validate static package files; do not print ACCEPTED.")
    parser.add_argument("--model-id", default=None, help="Model path or HF id used to re-evaluate M1.")
    parser.add_argument("--heldout-eval", default="repro_rep2/scripts/s07_heldout_eval.py")
    parser.add_argument("--sibling-eval", default="repro_rep2/scripts/s07_sibling_arena_eval.py")
    # v1.26 audit fix: repo-root-relative defaults ("../" escaped the repository).
    parser.add_argument("--failures", default="EDG-EXP1/results/a2_failures.json")
    parser.add_argument("--exp2-root", default="EDG-EXP2-struct")
    parser.add_argument("--heldout-partition", default="repro_rep2/data/heldout_pass16_partition.json")
    parser.add_argument("--sibling-arena", default="repro_rep2/data/v17_sibling_arena.json")
    parser.add_argument("--output-prefix", default="round2_outputs/reconcile/m1")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    m1 = manifest["m1_designated"]
    adapter_path = m1.get("adapter_path") or (m1.get("adapter") or {}).get("path")
    if not adapter_path:
        raise SystemExit("manifest m1_designated is missing adapter path")
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise SystemExit(f"M1 adapter missing: {adapter}")
    expected_heldout = m1["reference"]["expected_heldout_repair_count"]
    expected_sibling = m1["reference"]["expected_sibling_success_count"]
    if not args.skip_pytest:
        subprocess.run(["python3", "-m", "pytest", "repro_rep2/tests/test_repro_rep2_asserts.py"], check=True)
    if args.skip_model_check:
        print("PACKAGE_ONLY: static files and pytest passed; model re-evaluation was skipped.")
        return
    if not args.model_id:
        raise SystemExit("--model-id is required unless --skip-model-check is set")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    heldout_json = Path(str(prefix) + ".heldout.json")
    sibling_json = Path(str(prefix) + ".sibling.json")
    _run(
        [
            "python3",
            args.heldout_eval,
            "--model-id",
            args.model_id,
            "--adapter-dir",
            str(adapter.parent),
            "--arm",
            "main",
            "--config",
            "rep2_reconcile_m1",
            "--seed",
            str(m1["seed"]),
            "--run-id",
            "m1_reconcile",
            "--train-dataset",
            "repro_rep2/data/distill_main_replay2_capped.jsonl",
            "--output",
            str(heldout_json),
            "--failures",
            args.failures,
            "--exp2-root",
            args.exp2_root,
            "--heldout-partition",
            args.heldout_partition,
        ]
    )
    _run(
        [
            "python3",
            args.sibling_eval,
            "--model-id",
            args.model_id,
            "--adapter-dir",
            str(adapter.parent),
            "--arm",
            "main",
            "--config",
            "rep2_reconcile_m1",
            "--seed",
            str(m1["seed"]),
            "--run-id",
            "m1_reconcile",
            "--arena",
            args.sibling_arena,
            "--output",
            str(sibling_json),
            "--failures",
            args.failures,
        ]
    )

    heldout_payload = json.loads(heldout_json.read_text(encoding="utf-8"))
    sibling_payload = json.loads(sibling_json.read_text(encoding="utf-8"))
    heldout = heldout_payload["summary"]["heldout_repair"]
    sibling = sibling_payload["summary"]["success"]
    if abs(heldout - expected_heldout) > 1:
        raise SystemExit("heldout reference outside +/-1 episode")
    if abs(sibling - expected_sibling) > 2:
        raise SystemExit("sibling reference outside +/-2 episodes")
    print("ACCEPTED")


if __name__ == "__main__":
    main()
