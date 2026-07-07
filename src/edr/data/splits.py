"""Frozen data splits and round-1 input loading.

The deterministic hash split (seed 20260630) and the regression-set selection
are preregistered constants; every experiment in the project reads the same
split. Ported verbatim from the frozen round-1 code.
"""

from __future__ import annotations

import hashlib
import random

from edr.io_utils import read_json
from edr.paths import BASE_FAILURES, H1_EVOLUTION
from edr.scaffold.patches import PatchCandidate

DEFAULT_SPLIT_SEED = 20260630
DEFAULT_REGRESSION_SEED = 20260701
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def deterministic_split(failures, seed, train_ratio=0.5, validation_ratio=0.25):
    keyed = []
    for item in failures:
        digest = hashlib.sha256(f"{seed}:{item['episode_id']}".encode("utf-8")).hexdigest()
        keyed.append((digest, item))
    ordered = [item for _, item in sorted(keyed)]
    n = len(ordered)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "held_out": ordered[validation_end:],
    }


def load_round1_inputs(split_seed=DEFAULT_SPLIT_SEED, failures_path=None, evolution_path=None):
    """Baseline records, deterministic split, and the frozen H1 patch set.

    Returns (baseline, evolution_result, split, patches) — the same objects
    round 1 was built and judged on.
    """

    baseline = read_json(failures_path or BASE_FAILURES)
    evolution_result = read_json(evolution_path or H1_EVOLUTION)
    multiple_failures = [item for item in baseline["failures"] if item.get("split") == "multiple"]
    split = deterministic_split(multiple_failures, seed=split_seed)
    seed_result = next((item for item in evolution_result["per_seed"] if item["seed"] == split_seed), None)
    if seed_result is None:
        raise ValueError(f"split seed {split_seed} not found in the frozen evolution result")
    patches = [PatchCandidate(**item) for item in seed_result["loop_outputs"]["NL"]["accepted_patches"]]
    return baseline, evolution_result, split, patches


def build_base_prompt(episode, prompt_template):
    import json

    return prompt_template.format(
        query=episode.get("query", ""),
        function_doc=json.dumps(episode.get("function_pool", []), ensure_ascii=False),
    )


def function_name(episode):
    return episode.get("ground_truth_call", {}).get("name", "")


def failure_type(episode):
    return episode.get("failure_class", "unknown")


def partition_episode(pass16_hits, scaffold_success):
    if pass16_hits > 0:
        return "sampling_rescuable"
    if scaffold_success:
        return "scaffold_only"
    return "neither"


def select_regression_sets(records, split, regression_seed=DEFAULT_REGRESSION_SEED, r_success_size=400):
    heldout_ids = {item["episode_id"] for item in split["held_out"]}
    multiple_success = [
        item
        for item in records
        if item.get("split") == "multiple" and item.get("call_success") is True and item["episode_id"] not in heldout_ids
    ]
    rng = random.Random(regression_seed)
    ordered = list(multiple_success)
    rng.shuffle(ordered)
    r_success = ordered[: min(r_success_size, len(ordered))]
    r_other = [item for item in records if item.get("split") == "simple"]
    return {
        "r_success_eval": [item["episode_id"] for item in r_success],
        "r_other": [item["episode_id"] for item in r_other],
        "r_heldout": sorted(heldout_ids),
        "r_success_pool_total": len(multiple_success),
        "r_success_eval_size": len(r_success),
        "r_other_size": len(r_other),
        "r_heldout_size": len(heldout_ids),
    }
