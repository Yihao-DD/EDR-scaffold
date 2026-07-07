"""Shared helpers for the round-2 pipeline."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from edr.config import load_config
from edr.io_utils import read_json
from edr.paths import EPISODE_ID_DIR, TRAINVAL_PARTITION, resolve
from edr.scaffold.patches import PatchCandidate


def resolve_m1_adapter(cli_value=None):
    """M1 adapter dir: CLI value wins, else configs/round2.json m1_adapter."""

    if cli_value:
        return resolve(cli_value)
    config = load_config("round2")
    return resolve(config["m1_adapter"])


def load_train_share():
    """Round-2 train share = (train∪val failures) minus D_val, from the frozen partition."""

    pass16 = read_json(TRAINVAL_PARTITION)
    val_ids = read_json(EPISODE_ID_DIR / "d_val_failures.json")
    val_ids = set(val_ids.get("episode_ids") if isinstance(val_ids, dict) else val_ids)
    return [row for row in pass16["episodes"] if row["episode_id"] not in val_ids]


def dedupe_by_episode_output(rows):
    seen = OrderedDict()
    for row in rows:
        key = (row.get("episode_id"), row.get("output") or row.get("prediction"))
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def load_h2_patches(loop_output, family="NL", seed=None):
    payload = read_json(loop_output)
    per_seed = payload.get("per_seed") or []
    if not per_seed:
        raise AssertionError({"assert": "loop_output_has_per_seed", "path": str(loop_output)})
    selected = None
    for item in per_seed:
        if seed is None or item.get("seed") == seed:
            selected = item
            break
    if selected is None:
        raise AssertionError({"assert": "loop_seed_present", "seed": seed, "path": str(loop_output)})
    loop_outputs = selected.get("loop_outputs") or {}
    if family not in loop_outputs:
        raise AssertionError({"assert": "loop_family_present", "family": family, "available": sorted(loop_outputs)})
    return [PatchCandidate(**row) for row in loop_outputs[family].get("accepted_patches", [])]


def loop_family_from_method(method):
    return method.replace("-evo", "")
