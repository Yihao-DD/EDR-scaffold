"""Shared statistics: paired bootstrap and Wilson intervals. Deterministic."""

from __future__ import annotations

import random

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260713


def per_episode_success(heldout_payloads):
    """episode_id -> mean success across seeds for one arm."""

    totals = {}
    counts = {}
    for payload in heldout_payloads:
        for record in payload["records"]:
            episode_id = record["episode_id"]
            totals[episode_id] = totals.get(episode_id, 0) + (1 if record["success"] else 0)
            counts[episode_id] = counts.get(episode_id, 0) + 1
    return {episode_id: totals[episode_id] / counts[episode_id] for episode_id in totals}


def paired_bootstrap(left, right, episode_ids, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    """Bootstrap CI of mean(left - right) over episodes."""

    diffs = [left[e] - right[e] for e in episode_ids]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean_diff": None, "ci95": None}
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[int(0.975 * iterations) - 1]
    return {
        "n": n,
        "mean_diff": sum(diffs) / n,
        "ci95": [lo, hi],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]
