#!/usr/bin/env python3
"""EDR unified runner. One command per stage; resumable; no babysitting.

Usage:
  python3 scripts/run.py preflight              # environment + data + package checks (no GPU)
  python3 scripts/run.py ablations [--dry-run]  # A5/A6/A7/A8/A11 arms + Gate-1 tally
  python3 scripts/run.py round2    [--dry-run]  # second EDR iteration + Gate-2 classification
  python3 scripts/run.py all       [--dry-run]
  python3 scripts/run.py status                 # step table from outputs/state.json
  python3 scripts/run.py report                 # regenerate gate reports + REPORT.md

Recipes and gates are preregistered constants in configs/; the runner
executes them mechanically. Gate reports are mechanical tallies —
interpretation requires human sign-off. See README.md and docs/reproduce.md.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edr.config import load_config  # noqa: E402
from edr.paths import OUTPUT_DIR, STATE_FILE  # noqa: E402
from edr.runner.gpus import detect_lanes  # noqa: E402
from edr.runner.scheduler import Runner  # noqa: E402
from edr.runner.steps import ablation_steps, round2_steps  # noqa: E402

REQUIRED_FILES = [
    "data/round1/base_failures.json",
    "data/round1/evolution_h1.json",
    "data/round1/distill_core.jsonl",
    "data/round1/distill_train.jsonl",
    "data/round1/heldout_pass16_partition.json",
    "data/round1/train_val_pass16_partition.json",
    "data/round1/teacher_heldout_reference.json",
    "data/round1/sibling_arena.json",
    "data/round1/episode_ids/d_val_failures.json",
    "data/round1/episode_ids/d_heldout_failures.json",
    "data/round1/episode_ids/old400_success_eval.json",
    "data/round1/episode_ids/sibling300_arena.json",
    "data/round1/episode_ids/capped120_replay.json",
]


def preflight(args):
    config = load_config()
    report = {"checks": [], "pass": True}

    def check(name, ok, detail=""):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            report["pass"] = False
        print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + str(detail) if detail else ''}")

    check("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    for module in ("torch", "transformers", "peft", "numpy", "pytest"):
        try:
            mod = importlib.import_module(module)
            check(f"import {module}", True, getattr(mod, "__version__", ""))
        except Exception as error:  # noqa: BLE001
            check(f"import {module}", False, error)
    try:
        importlib.import_module("sentence_transformers")
        check("import sentence_transformers (A6 dense)", True)
    except Exception:  # noqa: BLE001
        check("import sentence_transformers (A6 dense)", False, "pip install -e '.[retrieval]' or the A6 dense cells cannot run")

    for rel in REQUIRED_FILES:
        check(f"file {rel}", (REPO_ROOT / rel).exists())

    round2_config = load_config("round2")
    m1 = REPO_ROOT / round2_config["m1_adapter"] / "adapter_model.safetensors"
    if m1.exists():
        check("M1 adapter present (tools/fetch_adapters.py done)", m1.stat().st_size > 1_000_000, f"{m1.stat().st_size} bytes")
    else:
        check("M1 adapter present", False, f"{m1} missing — run tools/fetch_adapters.py (needed for round2; ablations can run without it)")

    lanes = detect_lanes(config.get("gpu", {}))
    check("gpu lanes (informational)", True, lanes or "none detected — GPU steps will refuse to start")

    if not args.skip_pytest:
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=REPO_ROOT).returncode
        check("pytest", rc == 0)

    smoke_dir = OUTPUT_DIR / "preflight_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smokes = [
        ("a7a8 builder dry-run", ["-m", "edr.data.ablations", "build-a7a8", "--dry-run", "--output-dir", str(smoke_dir)]),
        ("a5 resample dry-run", ["-m", "edr.data.ablations", "resample-teacher", "--dry-run", "--output-dir", str(smoke_dir)]),
        ("a11 scramble dry-run", ["-m", "edr.evaluation.placebo", "scramble", "--dry-run", "--output-dir", str(smoke_dir)]),
        ("a6 index dry-run (bm25, 5 episodes)", ["-m", "edr.retrieval.patch_retrieval", "build-index", "--dry-run", "--limit", "5", "--output-dir", str(smoke_dir)]),
    ]
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for name, argv in smokes:
        rc = subprocess.run([sys.executable, *argv], cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        tail = (rc.stdout or rc.stderr).strip().splitlines()[-1] if (rc.stdout or rc.stderr) else ""
        check(name, rc.returncode == 0, tail)

    out = OUTPUT_DIR / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\npreflight {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


def status():
    if not STATE_FILE.exists():
        print("no outputs/state.json yet — nothing has been launched")
        return 0
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    counts = {}
    for step_id in sorted(state):
        entry = state[step_id]
        counts[entry.get("status")] = counts.get(entry.get("status"), 0) + 1
        print(f"{entry.get('status', '?'):8s} {step_id:40s} {entry.get('log', '')}")
    print(f"\ntotals: {counts}")
    return 0


def report():
    pieces = []
    gate1 = OUTPUT_DIR / "ablations" / "gate1_report.md"
    gate2 = OUTPUT_DIR / "round2" / "gate2_report.md"
    if not gate1.exists():
        subprocess.run([sys.executable, "-m", "edr.analysis.gate1"], cwd=REPO_ROOT, env=_env())
    for path, title in ((gate1, "Ablations — Gate 1"), (gate2, "Round 2 — Gate 2")):
        if path.exists():
            pieces.append(f"\n\n---\n\n## {title}\n\n" + path.read_text(encoding="utf-8"))
        else:
            pieces.append(f"\n\n---\n\n## {title}\n\nNOT AVAILABLE YET (`{path.relative_to(REPO_ROOT)}` missing).")
    header = (
        "# EDR run report\n\n"
        "> Generated by `python3 scripts/run.py report`. All gate content is a MECHANICAL tally; "
        "interpretation and gate verdicts require human sign-off.\n"
    )
    out = REPO_ROOT / "REPORT.md"
    out.write_text(header + "".join(pieces) + "\n", encoding="utf-8")
    print(f"report -> {out}")
    return 0


def _env():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["preflight", "ablations", "round2", "all", "status", "report"])
    parser.add_argument("--dry-run", action="store_true", help="Print the step plan; execute nothing.")
    parser.add_argument("--skip-pytest", action="store_true", help="Preflight only: skip the pytest block.")
    args = parser.parse_args(argv)

    if args.command == "preflight":
        return preflight(args)
    if args.command == "status":
        return status()
    if args.command == "report":
        return report()

    steps = []
    if args.command in ("ablations", "all"):
        steps.extend(ablation_steps())
    if args.command in ("round2", "all"):
        steps.extend(round2_steps())

    runner = Runner(steps, load_config())
    if args.dry_run:
        runner.dry_run()
        return 0
    counts = runner.run()
    if not counts.get("failed") and not counts.get("blocked"):
        report()
    return 1 if counts.get("failed") or counts.get("blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
