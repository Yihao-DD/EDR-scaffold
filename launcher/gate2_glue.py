#!/usr/bin/env python3
"""Gate 2 glue: measured forget + retention_2 per seed, then mechanical classification.

Runs round2/classify_gate2.py with MEASURED inputs (no hand-typed numbers):
- f2-size from round2_outputs/f2_failures.json,
- forget = mean sibling forget over the five M2 seeds,
- m1-heldout from the reconcile step (fresh M1 forward on this machine),
- retention_2 for every seed from the single teacher-2 forward.

Output: round2_outputs/eval/gate2.json + round2_outputs/gate2_report.md.
The classification is mechanical (compound/converge/collapse); interpretation
requires human sign-off.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "launch.json"))
    parser.add_argument("--outputs", default=str(REPO_ROOT / "round2_outputs"))
    args = parser.parse_args(argv)
    config = read_json(args.config)
    outputs = Path(args.outputs)

    f2 = read_json(outputs / "f2_failures.json")
    f2_size = f2.get("f2_count") or len(f2.get("failures", []))

    heldout_paths = sorted((outputs / "eval").glob("m2_seed*.heldout.json"))
    sibling_paths = sorted((outputs / "eval").glob("m2_seed*.sibling.json"))
    if not heldout_paths or not sibling_paths:
        raise SystemExit("no M2 eval outputs found; run the p2.eval steps first")

    forgets = [read_json(path)["summary"]["forget_sibling"] for path in sibling_paths]
    forget_mean = sum(forgets) / len(forgets)

    teacher2_paths = sorted((outputs / "eval").glob("m2_seed*.teacher2_heldout.json"))
    retention = {}
    if teacher2_paths:
        teacher2 = read_json(teacher2_paths[0])
        teacher2_rate = teacher2.get("success_rate")
        for path in heldout_paths:
            payload = read_json(path)
            seed_tag = path.name.replace(".heldout.json", "")
            m2_rate = payload["summary"]["heldout_repair_rate"]
            retention[seed_tag] = {
                "m2_repair": payload["summary"]["heldout_repair"],
                "m2_rate": m2_rate,
                "teacher2_rate": teacher2_rate,
                "retention_2": (m2_rate / teacher2_rate) if teacher2_rate else None,
            }

    gate2_output = outputs / "eval" / "gate2.json"
    cmd = [
        "python3",
        "round2/classify_gate2.py",
        "--m1-heldout",
        str(outputs / "reconcile" / "m1.heldout.json"),
        "--m2-heldout",
        *[str(path) for path in heldout_paths],
        "--f1-size",
        "74",
        "--f2-size",
        str(f2_size),
        "--forget",
        str(forget_mean),
        "--forget-gate",
        str(config["phase2"]["forget_gate"]),
        "--output",
        str(gate2_output),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    gate2 = read_json(gate2_output)

    lines = [
        "# Gate 2 mechanical classification",
        "",
        "> MECHANICAL CLASSIFICATION ONLY — interpretation requires human sign-off.",
        "",
        f"- classification: **{gate2.get('classification', gate2)}**",
        f"- f1_size=74, f2_size={f2_size}",
        f"- sibling forget per seed: {[round(x, 4) for x in forgets]} (mean {forget_mean:.4f}, gate {config['phase2']['forget_gate']})",
        "",
        "## retention_2 per seed (denominator: single teacher-2 forward)",
    ]
    for seed_tag, item in sorted(retention.items()):
        r2 = item["retention_2"]
        lines.append(f"- {seed_tag}: repair={item['m2_repair']} retention_2={r2:.4f}" if r2 is not None else f"- {seed_tag}: repair={item['m2_repair']} retention_2=n/a")
    lines.append("")
    lines.append(f"Raw classifier output: `{gate2_output}`")
    report = outputs / "gate2_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gate2] classification={gate2.get('classification')} -> {report}")


if __name__ == "__main__":
    main()
