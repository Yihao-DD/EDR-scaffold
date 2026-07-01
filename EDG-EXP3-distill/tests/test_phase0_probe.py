import pytest

from scripts.phase0_probe import (
    assert_no_leakage,
    dedupe_rows,
    deterministic_sample,
    merge_shard_payloads,
    partition_episode,
    select_regression_sets,
    shard_items,
)


def _record(episode_id, split="multiple", success=False):
    return {
        "episode_id": episode_id,
        "split": split,
        "call_success": success,
    }


def test_partition_episode_labels_sampling_before_scaffold_only():
    assert partition_episode(1, False) == "sampling_rescuable"
    assert partition_episode(2, True) == "sampling_rescuable"
    assert partition_episode(0, True) == "scaffold_only"
    assert partition_episode(0, False) == "neither"


def test_shard_items_are_disjoint_and_cover_input():
    items = [{"episode_id": f"e{i}"} for i in range(11)]
    shards = [shard_items(items, index, 3) for index in range(3)]
    flattened = [item["episode_id"] for shard in shards for item in shard]

    assert set(flattened) == {item["episode_id"] for item in items}
    assert len(flattened) == len(items)
    assert set(item["episode_id"] for item in shards[0]).isdisjoint(item["episode_id"] for item in shards[1])


def test_select_regression_sets_uses_successes_outside_heldout():
    records = [_record(f"s{i}", success=True) for i in range(10)]
    records.extend(_record(f"f{i}", success=False) for i in range(3))
    records.extend(_record(f"simple{i}", split="simple", success=True) for i in range(2))
    split = {"held_out": [{"episode_id": "s0"}, {"episode_id": "f0"}]}

    selected = select_regression_sets(records, split, regression_seed=7, r_success_size=5)

    assert len(selected["r_success_eval"]) == 5
    assert "s0" not in selected["r_success_eval"]
    assert selected["r_other"] == ["simple0", "simple1"]
    assert selected["r_heldout"] == ["f0", "s0"]


def test_assert_no_leakage_raises_on_heldout_or_success_overlap():
    assert assert_no_leakage(["d1"], ["h1"], ["s1"]) is True

    with pytest.raises(AssertionError):
        assert_no_leakage(["d1", "h1"], ["h1"], ["s1"])

    with pytest.raises(AssertionError):
        assert_no_leakage(["d1", "s1"], ["h1"], ["s1"])


def test_merge_shard_payloads_requires_exact_expected_ids():
    payloads = [
        {"records": [{"episode_id": "e1", "value": 1}]},
        {"records": [{"episode_id": "e2", "value": 2}]},
    ]

    merged = merge_shard_payloads(payloads, ["e2", "e1"])

    assert [row["episode_id"] for row in merged] == ["e2", "e1"]

    with pytest.raises(ValueError):
        merge_shard_payloads(payloads, ["e1", "e3"])


def test_dedupe_rows_uses_episode_source_and_output():
    rows = [
        {"episode_id": "e1", "source": "teacher", "output": "{\"name\":\"a\"}"},
        {"episode_id": "e1", "source": "teacher", "output": "{\"name\":\"a\"}"},
        {"episode_id": "e1", "source": "star", "output": "{\"name\":\"a\"}"},
    ]

    deduped = dedupe_rows(rows)

    assert len(deduped) == 2
    assert [row["source"] for row in deduped] == ["teacher", "star"]


def test_deterministic_sample_is_stable_and_bounded():
    rows = [
        {"episode_id": f"e{i}", "source": "teacher", "output": str(i)}
        for i in range(10)
    ]

    first = deterministic_sample(rows, 4, seed=1, label="x")
    second = deterministic_sample(rows, 4, seed=1, label="x")
    all_rows = deterministic_sample(rows, 20, seed=1, label="x")

    assert first == second
    assert len(first) == 4
    assert all_rows == rows
