import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v17_sibling_arena_is_frozen_with_five_zero_intersections():
    arena_path = ROOT / "results" / "v17_sibling_arena.json"
    payload = json.loads(arena_path.read_text(encoding="utf-8"))

    assert payload["probe"] == "v17_sibling_arena"
    assert payload["decision_tree_landing"] == "a_never_touched_parallel_family_success_ge_300"
    assert payload["source"] == "parallel_family_never_touched_base_successes_only"
    assert payload["selected_size"] == 300
    assert len(payload["episode_ids"]) == 300
    assert len(set(payload["episode_ids"])) == 300

    asserts = payload["asserts"]
    assert asserts["arena_intersect_all_training_rows"] == 0
    assert asserts["arena_intersect_old400"] == 0
    assert asserts["arena_intersect_capped106"] == 0
    assert asserts["arena_intersect_d_val"] == 0
    assert asserts["arena_intersect_d_heldout"] == 0
