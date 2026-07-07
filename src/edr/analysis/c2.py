"""C2 headline recomputation from the shipped frozen reference evaluations.

C2 (mechanism claim): on failures that base-model sampling cannot reach
(scaffold-only stratum), distilling scaffold-taught trajectories beats
distilling the model's own lucky samples (STaR). The adjudication is the
episode-level paired bootstrap of (main − star) per-episode success means
over the five frozen seeds.

This module makes the round-1 headline number reproducible inside this repo
with one command — no GPU, pure arithmetic over data/round1/reference_evals/:

  python -m edr.analysis.c2
  # scaffold_only_47: mean diff +0.1532, 95% CI excludes 0

The point estimate is exact arithmetic (main 24.6/47 − star 17.4/47 = 7.2/47);
CI endpoints depend on the bootstrap seed and are pinned by test to the
historical interval [+0.064, +0.251] within tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edr.analysis.stats import paired_bootstrap, per_episode_success
from edr.io_utils import read_json, write_json
from edr.paths import REFERENCE_EVALS


def load_arm(arm):
    payloads = [read_json(path) for path in sorted((Path(REFERENCE_EVALS) / arm).glob("seed*/heldout.json"))]
    if not payloads:
        raise FileNotFoundError(f"no reference heldout evals for arm '{arm}' under {REFERENCE_EVALS}")
    return payloads


def strata_of(payloads):
    strata = {}
    for payload in payloads:
        for record in payload["records"]:
            strata[record["episode_id"]] = record.get("partition")
    return strata


def compute(output=None):
    main_payloads = load_arm("main")
    star_payloads = load_arm("star")
    main = per_episode_success(main_payloads)
    star = per_episode_success(star_payloads)
    common = sorted(set(main) & set(star))
    strata = strata_of(main_payloads)

    result = {
        "probe": "c2_recomputation",
        "definition": "paired bootstrap of per-episode success means, main(scaffold-taught) − star(self-sampled), 5 frozen seeds each",
        "seeds": {
            "main": [p["summary"]["seed"] for p in main_payloads],
            "star": [p["summary"]["seed"] for p in star_payloads],
        },
        "heldout_repair_per_seed": {
            "main": [p["summary"]["heldout_repair"] for p in main_payloads],
            "star": [p["summary"]["heldout_repair"] for p in star_payloads],
        },
        "strata": {},
    }
    groups = {"all_158": common}
    for name in ("scaffold_only", "sampling_rescuable", "neither"):
        groups[f"{name}_{sum(1 for e in common if strata.get(e) == name)}"] = [e for e in common if strata.get(e) == name]
    for name, episode_ids in groups.items():
        ci = paired_bootstrap(main, star, episode_ids)
        ci["probe"] = f"PROBE signal (n={ci['n']})" if ci["n"] < 30 else None
        result["strata"][name] = ci

    if output:
        write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)
    result = compute(args.output)
    for name, ci in result["strata"].items():
        flag = " <- C2 adjudication stratum" if name.startswith("scaffold_only") else ""
        print(
            f"{name}: mean diff {ci['mean_diff']:+.4f}  95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]"
            f"  excludes 0: {ci['ci_excludes_zero']}{flag}"
        )
    if args.output:
        print(f"-> {args.output}")


if __name__ == "__main__":
    main()
