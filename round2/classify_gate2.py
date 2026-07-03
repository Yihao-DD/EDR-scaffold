#!/usr/bin/env python3
"""Gate 2 dynamics classifier (mechanical; no narrative adjudication).

Master-plan Gate 2 rules, decided by 5-seed paired bootstrap on heldout repair:

- compound : repair(M2) > repair(M1) with the 95% CI of the paired difference
             excluding 0.
- collapse : repair(M2) < repair(M1) with the CI excluding 0, OR the forgetting
             gate is broken (forget > forget_gate).
- converge : CI overlaps 0 AND the round-2 failure set shrank below half of
             round 1 (|F2| < 0.5 * |F1|) -- failures were eliminated, so there is
             less to teach (healthy convergence).
- flat     : CI overlaps 0 and |F2| is not < 0.5 * |F1|. Not one of the three
             Gate-2 categories; reported honestly as undetermined.

M2 per-episode success is the mean over the seed adapters (in [0, 1]); M1 is the
single designated reference. The paired bootstrap resamples heldout episodes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _per_episode_success(payload) -> dict:
    """Map episode_id -> 0/1 success from an s07 heldout-eval payload."""
    return {str(row["episode_id"]): (1.0 if row["success"] else 0.0) for row in payload["records"]}


def _percentile(values: list[float], q: float) -> float:
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


def classify(
    m1_success: dict,
    m2_seed_successes: list[dict],
    f1_size: int,
    f2_size: int,
    forget: float | None = None,
    forget_gate: float = 0.02,
    bootstrap: int = 10000,
    seed: int = 20260708,
) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-heldout", required=True, help="s07 heldout-eval JSON for the designated M1.")
    parser.add_argument("--m2-heldout", nargs="+", required=True, help="s07 heldout-eval JSONs for the 5 M2 seeds.")
    parser.add_argument("--f1-size", type=int, required=True, help="Round-1 failure-set size |F1|.")
    parser.add_argument("--f2-size", type=int, required=True, help="Round-2 failure-set size |F2| (from collect_failures f2_count).")
    parser.add_argument("--forget", type=float, default=None, help="Max forgetting across M2 seeds (regression surface).")
    parser.add_argument("--forget-gate", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--output", default="round2_outputs/eval/gate2.json")
    args = parser.parse_args()

    result = classify(
        _per_episode_success(_read_json(args.m1_heldout)),
        [_per_episode_success(_read_json(path)) for path in args.m2_heldout],
        f1_size=args.f1_size,
        f2_size=args.f2_size,
        forget=args.forget,
        forget_gate=args.forget_gate,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("classification", "reason", "paired_diff_point", "paired_diff_ci95")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
