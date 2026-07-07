"""Leakage enforcement — code asserts, never convention.

Training data must be disjoint from every evaluation surface:
D_val, D_heldout, old-400 regression set, and the sibling-300 arena.
All asserts fail closed: a row without an episode_id raises instead of
slipping through.
"""

from __future__ import annotations

from edr.io_utils import read_json
from edr.paths import EPISODE_ID_DIR


def load_episode_id_file(path):
    payload = read_json(path)
    if isinstance(payload, dict):
        values = payload.get("episode_ids") or payload.get("arena_episode_ids")
    else:
        values = payload
    if values is None:
        raise ValueError(f"{path} does not contain episode ids")
    return {str(item) for item in values}


def load_eval_surface_ids():
    return {
        "d_val": load_episode_id_file(EPISODE_ID_DIR / "d_val_failures.json"),
        "d_heldout": load_episode_id_file(EPISODE_ID_DIR / "d_heldout_failures.json"),
        "old400": load_episode_id_file(EPISODE_ID_DIR / "old400_success_eval.json"),
        "sibling300": load_episode_id_file(EPISODE_ID_DIR / "sibling300_arena.json"),
    }


def episode_ids(rows):
    """Collect episode ids, failing closed on rows without one."""

    ids = set()
    missing = []
    for index, row in enumerate(rows):
        episode_id = row.get("episode_id")
        if not episode_id:
            missing.append(index)
            continue
        ids.add(str(episode_id))
    if missing:
        raise AssertionError(
            {"assert": "episode_id_present", "missing_count": len(missing), "examples": missing[:20]}
        )
    return ids


def assert_disjoint(name, left, right):
    overlap = sorted(set(left) & set(right))
    if overlap:
        raise AssertionError({"assert": name, "overlap_count": len(overlap), "examples": overlap[:20]})
    return {"assert": name, "overlap_count": 0}


def assert_training_disjoint(rows, label):
    """Four-way assert: training ∩ {D_val, D_heldout, old400, sibling300} = ∅."""

    ids = episode_ids(rows)
    surfaces = load_eval_surface_ids()
    report = {}
    for name, surface in surfaces.items():
        overlap = sorted(ids & surface)
        if overlap:
            raise AssertionError({"assert": f"{label}_disjoint_{name}", "overlap_count": len(overlap), "examples": overlap[:20]})
        report[f"{label}_intersect_{name}"] = 0
    return report


def assert_no_leakage(distill_episode_ids, heldout_episode_ids, r_success_eval_ids):
    distill = set(distill_episode_ids)
    heldout = set(heldout_episode_ids)
    r_success = set(r_success_eval_ids)
    overlap_heldout = sorted(distill & heldout)
    overlap_success = sorted(distill & r_success)
    if overlap_heldout or overlap_success:
        raise AssertionError(
            {
                "distill_heldout_overlap": overlap_heldout[:20],
                "distill_r_success_overlap": overlap_success[:20],
                "n_distill_heldout_overlap": len(overlap_heldout),
                "n_distill_r_success_overlap": len(overlap_success),
            }
        )
    return True


def assert_no_selection_leakage(training_episode_ids, selection_episode_ids):
    training = set(training_episode_ids)
    selection = set(selection_episode_ids)
    overlap = sorted(training & selection)
    if overlap:
        raise AssertionError(
            {
                "training_selection_overlap": overlap[:20],
                "n_training_selection_overlap": len(overlap),
            }
        )
    return True
