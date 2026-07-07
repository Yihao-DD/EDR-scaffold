#!/usr/bin/env python3
"""EDR unified launcher. One command per phase; resumable; no babysitting.

Usage:
  python3 run.py preflight              # environment + package + data checks (no GPU needed)
  python3 run.py phase1 [--dry-run]     # all Phase-1 ablation arms (A5/A6/A7/A8/A11) + Gate-1 tally
  python3 run.py phase2 [--dry-run]     # full round-2 iteration (F2 -> H2 -> T2 -> A2 x5 -> M2 eval -> Gate-2)
  python3 run.py all [--dry-run]        # phase1 then phase2
  python3 run.py status                 # step table from run_state/state.json
  python3 run.py report                 # regenerate Gate reports + top-level REPORT.md

Read RUNBOOK.md first. Recipes and gates are preregistered; the launcher
executes them mechanically and never invents parameters. Gate reports are
mechanical tallies — interpretation requires human sign-off.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from launcher.gpus import detect_lanes  # noqa: E402
from launcher.runner import Runner  # noqa: E402
from launcher.steps import phase1_steps, phase2_steps  # noqa: E402


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "repro_rep2/MANIFEST.json",
    "repro_rep2/data/distill_main_core.jsonl",
    "repro_rep2/data/distill_main_replay2_capped.jsonl",
    "repro_rep2/data/heldout_pass16_partition.json",
    "repro_rep2/data/v17_sibling_arena.json",
    "repro_rep2/data/episode_ids/D_val_failures.json",
    "repro_rep2/data/episode_ids/D_heldout_failures.json",
    "repro_rep2/data/episode_ids/old400_success_eval.json",
    "repro_rep2/data/episode_ids/sibling300_arena.json",
    "EDG-EXP1/results/a2_failures.json",
    "EDG-EXP2-struct/results/evolution_main_merged.json",
    "EDG-EXP2-struct/scripts/evolution_loop.py",
]


def preflight(config, args):
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
        check("import sentence_transformers (A6 dense)", False, "install requirements-phase1.txt or A6 dense cells cannot run")

    for rel in REQUIRED_FILES:
        check(f"file {rel}", (REPO_ROOT / rel).exists())

    manifest = load_config(REPO_ROOT / config["phase2"]["m1_manifest"])
    adapter = REPO_ROOT / manifest["m1_designated"]["adapter"]["path"]
    if adapter.exists():
        size = adapter.stat().st_size
        check("M1 adapter is a real file (git lfs pull done)", size > 1_000_000, f"{size} bytes")
    else:
        check("M1 adapter exists", False, str(adapter))

    lanes = detect_lanes(config.get("gpu", {}))
    check("gpu lanes (informational)", True, lanes or "none detected — GPU steps will refuse to start")

    if not args.skip_pytest:
        rc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "repro_rep2/tests/test_repro_rep2_asserts.py", "repro_rep2/tests/test_round2_contracts.py"],
            cwd=REPO_ROOT,
        ).returncode
        check("focused pytest", rc == 0)

    smoke_dir = REPO_ROOT / "run_state" / "preflight_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smokes = [
        ("a7a8 builder dry-run", ["-m", "phase1.a7_a8_build", "--dry-run", "--output-dir", str(smoke_dir)]),
        ("a5 resample dry-run", ["-m", "phase1.a5_resample_teacher", "--dry-run", "--output-dir", str(smoke_dir)]),
        ("a11 scramble dry-run", ["-m", "phase1.a11_placebo", "scramble", "--dry-run", "--output-dir", str(smoke_dir)]),
        ("a6 index dry-run (bm25, 5 episodes)", ["-m", "phase1.a6_retrieval", "build-index", "--dry-run", "--limit", "5", "--output-dir", str(smoke_dir)]),
    ]
    for name, argv in smokes:
        rc = subprocess.run([sys.executable, *argv], cwd=REPO_ROOT, capture_output=True, text=True)
        check(name, rc.returncode == 0, (rc.stdout or rc.stderr).strip().splitlines()[-1] if (rc.stdout or rc.stderr) else "")

    out = REPO_ROOT / "run_state" / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\npreflight {'PASS' if report['pass'] else 'FAIL'} -> {out}")
    return 0 if report["pass"] else 1


# ---------------------------------------------------------------------------
# status / report
# ---------------------------------------------------------------------------

def status(config):
    state_path = REPO_ROOT / "run_state" / "state.json"
    if not state_path.exists():
        print("no run_state/state.json yet — nothing has been launched")
        return 0
    state = load_config(state_path)
    counts = {}
    for step_id in sorted(state):
        entry = state[step_id]
        counts[entry.get("status")] = counts.get(entry.get("status"), 0) + 1
        print(f"{entry.get('status', '?'):8s} {step_id:40s} {entry.get('log', '')}")
    print(f"\ntotals: {counts}")
    return 0


def report(config):
    pieces = []
    gate1 = REPO_ROOT / "phase1_outputs" / "gate1_report.md"
    gate2 = REPO_ROOT / "round2_outputs" / "gate2_report.md"
    if not gate1.exists():
        subprocess.run([sys.executable, "-m", "phase1.gate1_report"], cwd=REPO_ROOT)
    for path, title in ((gate1, "Phase 1 — Gate 1"), (gate2, "Phase 2 — Gate 2")):
        if path.exists():
            pieces.append(f"\n\n---\n\n## {title}\n\n" + path.read_text(encoding="utf-8"))
        else:
            pieces.append(f"\n\n---\n\n## {title}\n\nNOT AVAILABLE YET (`{path.relative_to(REPO_ROOT)}` missing).")
    header = (
        "# EDR run report\n\n"
        "> Generated by `python3 run.py report`. All Gate content is a MECHANICAL tally; "
        "interpretation and Gate verdicts require human sign-off (anti-self-deception protocol #7).\n"
    )
    out = REPO_ROOT / "REPORT.md"
    out.write_text(header + "".join(pieces) + "\n", encoding="utf-8")
    print(f"report -> {out}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["preflight", "phase1", "phase2", "all", "status", "report"])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "launch.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print the step plan; execute nothing.")
    parser.add_argument("--skip-pytest", action="store_true", help="Preflight only: skip the pytest block.")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "preflight":
        return preflight(config, args)
    if args.command == "status":
        return status(config)
    if args.command == "report":
        return report(config)

    steps = []
    if args.command in ("phase1", "all"):
        steps.extend(phase1_steps(config))
    if args.command in ("phase2", "all"):
        steps.extend(phase2_steps(config))

    runner = Runner(steps, config)
    if args.dry_run:
        runner.dry_run()
        return 0
    counts = runner.run()
    if args.command in ("phase1", "all") and not counts.get("failed") and not counts.get("blocked"):
        report(config)
    return 1 if counts.get("failed") or counts.get("blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
