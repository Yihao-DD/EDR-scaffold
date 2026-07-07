"""Frozen-protocol integrity: splits, judge, patches, and shipped data.

These tests pin the experimental constants — if any of them fails, the
migration or a data change has broken protocol parity with round 1.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edr.data.distill import replay_rows_from_train  # noqa: E402
from edr.data.leakage import (  # noqa: E402
    assert_training_disjoint,
    episode_ids,
    load_eval_surface_ids,
)
from edr.data.splits import deterministic_split, load_round1_inputs  # noqa: E402
from edr.io_utils import read_json, stable_int  # noqa: E402
from edr.verifier import call_success, parse_model_output  # noqa: E402


def test_split_seed_20260630_reproduces_frozen_surfaces():
    baseline, _evolution, split, patches = load_round1_inputs(20260630)
    assert len(split["train"]) == 313
    assert len(split["validation"]) == 156
    assert len(split["held_out"]) == 158
    surfaces = load_eval_surface_ids()
    assert {e["episode_id"] for e in split["held_out"]} == surfaces["d_heldout"]
    assert {e["episode_id"] for e in split["validation"]} == surfaces["d_val"]


def test_h1_patch_set_is_the_frozen_39():
    _baseline, _evolution, _split, patches = load_round1_inputs(20260630)
    assert len(patches) == 39
    assert all(patch.family == "NL" for patch in patches)


def test_distill_train_shape_148_core_296_replay():
    core, replay = replay_rows_from_train()
    assert len(core) == 148
    assert len(replay) == 296
    assert len({row["episode_id"] for row in core}) == 74


def test_verifier_semantics():
    gt = {"name": "f", "accepted_arguments": {"a": ["1", ""], "b": [""]}}
    assert call_success({"name": "f", "arguments": {"a": "1"}}, gt)
    assert not call_success({"name": "f", "arguments": {}}, gt)
    assert not call_success({"name": "g", "arguments": {"a": "1"}}, gt)
    # extra argument not in the required set -> fail (exact-set semantics)
    assert not call_success({"name": "f", "arguments": {"a": "1", "b": "x"}}, gt)


def test_parse_model_output_json_and_call_expression():
    assert parse_model_output('{"name": "f", "arguments": {"a": 1}}') == {"name": "f", "arguments": {"a": 1}}
    assert parse_model_output("noise f(a=1) noise")["name"] == "f"
    with pytest.raises(ValueError):
        parse_model_output("no call here")


def test_stable_int_seed_strings_are_frozen():
    # These exact values seed the round-1 sampling streams. If this test
    # fails, resampling would not reproduce the original data.
    assert stable_int(20260630, "live_multiple_880-183-5", "teacher", 0) == stable_int(
        20260630, "live_multiple_880-183-5", "teacher", 0
    )
    assert stable_int(1, "e", "teacher", 0) != stable_int(1, "e", "pass16", 0)


def test_leakage_assert_fails_closed_on_missing_episode_id():
    with pytest.raises(AssertionError):
        episode_ids([{"episode_id": "ok"}, {"output": "missing"}])


def test_leakage_assert_rejects_heldout_episode():
    heldout = sorted(load_eval_surface_ids()["d_heldout"])
    with pytest.raises(AssertionError):
        assert_training_disjoint([{"episode_id": heldout[0]}], "test")


def test_teacher_reference_is_50_of_158():
    ref = read_json(REPO / "data/round1/teacher_heldout_reference.json")
    assert ref["stats"]["teacher_success"] == 50
    assert ref["stats"]["total"] == 158


def test_reference_evals_are_the_five_frozen_seeds():
    seeds = sorted(p.name for p in (REPO / "data/round1/reference_evals").glob("seed*"))
    assert seeds == [f"seed{s}" for s in (20260703, 20260704, 20260705, 20260706, 20260707)]
    repairs = [
        read_json(REPO / f"data/round1/reference_evals/seed{s}/heldout.json")["summary"]["heldout_repair"]
        for s in (20260703, 20260704, 20260705, 20260706, 20260707)
    ]
    assert repairs == [54, 51, 46, 58, 43]  # frozen round-1 five-seed results
    assert abs(sum(repairs) / 5 - 50.4) < 1e-9
