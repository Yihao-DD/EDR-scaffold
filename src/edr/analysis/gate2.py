"""Gate 2 dynamics classifier + report glue (mechanical; no narrative).

Classification rules (5-seed paired bootstrap on heldout repair):
- compound : repair(M2) > repair(M1), 95% CI of the paired diff excludes 0.
- collapse : repair(M2) < repair(M1) with CI excluding 0, OR forget > gate.
- converge : CI overlaps 0 AND |F2| < 0.5 * |F1| (healthy convergence).
- flat     : CI overlaps 0 otherwise; reported honestly as undetermined.

The report glue computes MEASURED inputs (f2 size from the collect output,
forget = mean sibling forget over M2 seeds, M1 reference from the reconcile
forward, retention_2 per seed from the single teacher-2 forward).

CLI:
  python -m edr.analysis.gate2 classify --m1-heldout <json> --m2-heldout <json...> \
      --f1-size 74 --f2-size N --forget F
  python -m edr.analysis.gate2 report        # measured-input glue + markdown
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from edr.config import load_config
from edr.io_utils import read_json, write_json
from edr.paths import RECONCILE_OUT, ROUND2_OUT, resolve


def _per_episode_success(payload):
    return {str(row["episode_id"]): (1.0 if row["success"] else 0.0) for row in payload["records"]}


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q / 100.0 * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def classify(m1_success, m2_seed_successes, f1_size, f2_size, forget=None, forget_gate=0.02, bootstrap=10000, seed=20260708):
    episode_ids = sorted(set(m1_success) & set.intersection(*[set(s) for s in m2_seed_successes]))
    if not episode_ids:
        raise SystemExit("no shared heldout episodes across M1 and M2 seeds")
    m1 = [m1_success[e] for e in episode_ids]
    m2 = [sum(s[e] for s in m2_seed_successes) / len(m2_seed_successes) for e in episode_ids]
    n = len(episode_ids)
    m1_rate = sum(m1) / n
    m2_rate = sum(m2) / n
    point = m2_rate - m1_rate

    rng = random.Random(seed)
    diffs = []
    for _ in range(bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(m2[i] - m1[i] for i in idx) / n)
    ci_lo = _percentile(diffs, 2.5)
    ci_hi = _percentile(diffs, 97.5)
    ci_excludes_zero = ci_lo > 0 or ci_hi < 0

    forget_broken = forget is not None and forget > forget_gate
    if forget_broken:
        classification = "collapse"
        reason = f"forgetting gate broken (forget={forget:.4f} > {forget_gate})"
    elif ci_excludes_zero and point > 0:
        classification = "compound"
        reason = "repair(M2) > repair(M1), paired CI excludes 0"
    elif ci_excludes_zero and point < 0:
        classification = "collapse"
        reason = "repair(M2) < repair(M1), paired CI excludes 0"
    elif f1_size and f2_size < 0.5 * f1_size:
        classification = "converge"
        reason = f"CI overlaps 0 and |F2|={f2_size} < 0.5*|F1|={0.5 * f1_size}"
    else:
        classification = "flat"
        reason = "CI overlaps 0 and |F2| not < 0.5*|F1|; undetermined (not a Gate-2 category)"

    return {
        "probe": "gate2_classifier",
        "classification": classification,
        "reason": reason,
        "n_heldout_episodes": n,
        "m1_repair_rate": m1_rate,
        "m2_repair_rate_mean_over_seeds": m2_rate,
        "m2_seed_repair_rates": [sum(s[e] for e in episode_ids) / n for s in m2_seed_successes],
        "paired_diff_point": point,
        "paired_diff_ci95": [ci_lo, ci_hi],
        "ci_excludes_zero": ci_excludes_zero,
        "f1_size": f1_size,
        "f2_size": f2_size,
        "forget": forget,
        "forget_gate": forget_gate,
        "forget_gate_broken": forget_broken,
        "bootstrap": bootstrap,
        "seed": seed,
    }


def cmd_classify(args):
    result = classify(
        _per_episode_success(read_json(resolve(args.m1_heldout))),
        [_per_episode_success(read_json(resolve(path))) for path in args.m2_heldout],
        f1_size=args.f1_size,
        f2_size=args.f2_size,
        forget=args.forget,
        forget_gate=args.forget_gate,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_json(resolve(args.output), result)
    print(json.dumps({k: result[k] for k in ("classification", "reason", "paired_diff_point", "paired_diff_ci95")}, ensure_ascii=False))


def cmd_report(args):
    """Measured-input glue: forget/f2/retention_2 from run outputs, then classify + markdown."""

    config = load_config("round2")
    outputs = resolve(args.outputs)

    f2 = read_json(outputs / "f2_failures.json")
    f2_size = f2.get("f2_count") or len(f2.get("failures", []))

    heldout_paths = sorted((outputs / "eval").glob("m2_seed*.heldout.json"))
    sibling_paths = sorted((outputs / "eval").glob("m2_seed*.sibling.json"))
    if not heldout_paths or not sibling_paths:
        raise SystemExit("no M2 eval outputs found; run the round-2 eval steps first")

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

    m1_heldout_path = RECONCILE_OUT / "m1.heldout.json"
    result = classify(
        _per_episode_success(read_json(m1_heldout_path)),
        [_per_episode_success(read_json(path)) for path in heldout_paths],
        f1_size=args.f1_size,
        f2_size=f2_size,
        forget=forget_mean,
        forget_gate=config["forget_gate"],
    )
    gate2_output = outputs / "eval" / "gate2.json"
    write_json(gate2_output, result)

    lines = [
        "# Gate 2 mechanical classification",
        "",
        "> MECHANICAL CLASSIFICATION ONLY — interpretation requires human sign-off.",
        "",
        f"- classification: **{result['classification']}** ({result['reason']})",
        f"- f1_size={args.f1_size}, f2_size={f2_size}",
        f"- paired diff M2−M1: {result['paired_diff_point']:.4f} CI95={result['paired_diff_ci95']}",
        f"- sibling forget per seed: {[round(x, 4) for x in forgets]} (mean {forget_mean:.4f}, gate {config['forget_gate']})",
        "",
        "## retention_2 per seed (denominator: single teacher-2 forward)",
    ]
    for seed_tag, item in sorted(retention.items()):
        r2 = item["retention_2"]
        lines.append(
            f"- {seed_tag}: repair={item['m2_repair']} retention_2={r2:.4f}" if r2 is not None else f"- {seed_tag}: repair={item['m2_repair']} retention_2=n/a"
        )
    lines.append("")
    lines.append(f"Raw classifier output: `{gate2_output}`")
    report = outputs / "gate2_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gate2] classification={result['classification']} -> {report}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument("--m1-heldout", required=True)
    classify_parser.add_argument("--m2-heldout", nargs="+", required=True)
    classify_parser.add_argument("--f1-size", type=int, required=True)
    classify_parser.add_argument("--f2-size", type=int, required=True)
    classify_parser.add_argument("--forget", type=float, default=None)
    classify_parser.add_argument("--forget-gate", type=float, default=0.02)
    classify_parser.add_argument("--bootstrap", type=int, default=10000)
    classify_parser.add_argument("--seed", type=int, default=20260708)
    classify_parser.add_argument("--output", default=str(ROUND2_OUT / "eval" / "gate2.json"))
    classify_parser.set_defaults(func=cmd_classify)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--outputs", default=str(ROUND2_OUT))
    report_parser.add_argument("--f1-size", type=int, default=74)
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
