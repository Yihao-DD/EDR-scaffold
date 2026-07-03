import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_ids(name):
    payload = json.loads((ROOT / "data" / "episode_ids" / name).read_text())
    return set(payload["episode_ids"])


def jsonl_ids(path):
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["episode_id"])
    return ids


def test_rep2_data_counts():
    assert sum(1 for _ in (ROOT / "data" / "distill_main_core.jsonl").open()) == 148
    assert sum(1 for _ in (ROOT / "data" / "distill_main_replay2_capped.jsonl").open()) == 444
    assert len(load_ids("D_val_failures.json")) == 156
    assert len(load_ids("D_heldout_failures.json")) == 158
    assert len(load_ids("old400_success_eval.json")) == 400
    assert len(load_ids("capped106_arena.json")) == 106
    assert len(load_ids("sibling300_arena.json")) == 300


def test_rep2_training_disjoint_from_eval_surfaces():
    train_ids = jsonl_ids(ROOT / "data" / "distill_main_replay2_capped.jsonl")
    for name in [
        "D_val_failures.json",
        "D_heldout_failures.json",
        "old400_success_eval.json",
        "capped106_arena.json",
        "sibling300_arena.json",
    ]:
        assert not (train_ids & load_ids(name)), name
