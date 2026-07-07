"""Contract tests for the Phase 1 arms and the unified launcher.

These close the same blind-spot family the v1.23/v1.25 external reviews found:
commands that render but reference files nobody produces, arms that silently
drift from the locked recipe, and leakage asserts that admit rows without ids.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from launcher.steps import phase1_steps, phase2_steps  # noqa: E402
from phase1.a6_retrieval import BM25, build_query, top_k  # noqa: E402
from phase1.a11_placebo import scramble_text  # noqa: E402
from phase1.common import assert_training_disjoint, load_config, replay_rows_from_capped  # noqa: E402


CONFIG = load_config(REPO / "configs" / "launch.json")


# ---------------------------------------------------------------------------
# DAG integrity: every consumed file has a producer; needs reference real steps
# ---------------------------------------------------------------------------

def _steps_by_id(steps):
    by_id = {step.id: step for step in steps}
    assert len(by_id) == len(steps), "duplicate step ids"
    return by_id


def test_phase1_dag_needs_and_producers():
    steps = phase1_steps(CONFIG)
    by_id = _steps_by_id(steps)
    produced = {path for step in steps for path in step.produces}
    for step in steps:
        for dep in step.needs:
            assert dep in by_id, f"{step.id} needs unknown step {dep}"
        # every ../phase1_outputs dataset consumed by a train/eval step must be produced upstream
        for index, token in enumerate(step.argv):
            if token in ("--dataset", "--train-dataset"):
                target = step.argv[index + 1]
                repo_rel = target[3:] if target.startswith("../") else target
                assert repo_rel in produced, f"{step.id} consumes {repo_rel} with no producer"


def test_phase2_dag_needs_and_adapter_producers():
    steps = phase2_steps(CONFIG)
    by_id = _steps_by_id(steps)
    for step in steps:
        for dep in step.needs:
            assert dep in by_id
    teacher2_evals = [step for step in steps if "--teacher2-loop-output" in step.argv]
    assert len(teacher2_evals) == 1, "teacher-2 forward must run exactly once across the 5 seeds"
    gate2 = by_id["p2.gate2"]
    eval_ids = [step.id for step in steps if step.id.startswith("p2.eval.")]
    assert set(eval_ids).issubset(set(gate2.needs))
    assert "p2.reconcile_m1" in gate2.needs


# ---------------------------------------------------------------------------
# Recipe lock: preregistered hyperparameters only (v1.26 ruling)
# ---------------------------------------------------------------------------

def test_phase1_training_steps_use_locked_recipe():
    steps = phase1_steps(CONFIG)
    trains = [step for step in steps if ".train." in step.id]
    assert len(trains) == 12  # a5 x3 + a7 x3 + a8_25 x3 + a8_50 x3
    for step in trains:
        argv = step.argv
        assert argv[argv.index("--rank") + 1] == "16"
        assert float(argv[argv.index("--lr") + 1]) == 5e-5
        assert argv[argv.index("--epochs") + 1] == "3"
        kl = float(argv[argv.index("--kl-anchor-lambda") + 1])
        if ".a7." in step.id:
            assert kl == 0.0, "A7 must drop the KL anchor together with replay (v1.26)"
        else:
            assert kl == 2.0
        seed = int(argv[argv.index("--seed") + 1])
        assert seed in CONFIG["phase1"]["seeds"]


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def test_a7_a8_builder_shapes(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "phase1.a7_a8_build", "--output-dir", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr
    a7 = [json.loads(line) for line in (tmp_path / "a7" / "distill_a7_noreplay.jsonl").read_text().splitlines()]
    assert len(a7) == 148
    assert all(row.get("source") != "replay_base_success" for row in a7)
    a8_50 = [json.loads(line) for line in (tmp_path / "a8" / "distill_a8_50pct.jsonl").read_text().splitlines()]
    core = [row for row in a8_50 if row.get("source") != "replay_base_success"]
    replay = [row for row in a8_50 if row.get("source") == "replay_base_success"]
    assert len(replay) == 2 * len(core), "A8 must keep the 2:1 replay:core ROW ratio"


def test_a5_builder_fails_closed_on_wrong_shard_kind(tmp_path):
    (tmp_path / "a5").mkdir(parents=True)
    (tmp_path / "a5" / "a5_teacher_all_shard_0.json").write_text(json.dumps({"probe": "wrong_kind", "records": []}))
    rc = subprocess.run(
        [sys.executable, "-m", "phase1.a5_build_unverified", "--output-dir", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert rc.returncode != 0
    assert "a5_shard_kind" in (rc.stderr + rc.stdout)


def test_replay_block_is_the_frozen_296():
    core, replay = replay_rows_from_capped()
    assert len(core) == 148
    assert len(replay) == 296
    assert all(row["source"] == "replay_base_success" for row in replay)


def test_leakage_assert_fails_closed_on_missing_episode_id():
    with pytest.raises(AssertionError):
        assert_training_disjoint([{"episode_id": "x"}, {"output": "no id"}], "test")


def test_leakage_assert_rejects_heldout_episode():
    heldout_ids = json.loads((REPO / "repro_rep2/data/episode_ids/D_heldout_failures.json").read_text())
    ids = heldout_ids.get("episode_ids") if isinstance(heldout_ids, dict) else heldout_ids
    with pytest.raises(AssertionError):
        assert_training_disjoint([{"episode_id": ids[0]}], "test")


# ---------------------------------------------------------------------------
# A6 retrieval determinism
# ---------------------------------------------------------------------------

def test_bm25_deterministic_and_topk_stable():
    docs = ["alpha beta gamma", "beta beta delta", "gamma delta epsilon"]
    first = BM25(docs).scores("beta delta")
    second = BM25(docs).scores("beta delta")
    assert first == second
    assert top_k(first, 2) == top_k(second, 2)


def test_a6_query_uses_names_not_schemas():
    episode = {
        "query": "book a flight",
        "function_pool": [{"name": "book_flight", "parameters": {"huge": "schema"}}, {"name": "cancel_flight"}],
    }
    query = build_query(episode)
    assert "book_flight" in query and "cancel_flight" in query
    assert "schema" not in query, "A6 query must contain function NAMES only (v1.26 spec)"


# ---------------------------------------------------------------------------
# A11 scramble determinism + token multiset preservation
# ---------------------------------------------------------------------------

class _StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def test_a11_scramble_deterministic_and_preserves_tokens():
    import random

    tokenizer = _StubTokenizer()
    text = "patch guidance: use exact parameter names"
    one = scramble_text(tokenizer, text, random.Random(42))
    two = scramble_text(tokenizer, text, random.Random(42))
    assert one == two, "same seed must give the same scramble"
    assert sorted(one) == sorted(text), "scramble must preserve the token multiset"
    assert one != text, "scramble must actually destroy order"


# ---------------------------------------------------------------------------
# v1.26 audit fixes: eval_round2 passes repo-root-correct evaluator paths
# ---------------------------------------------------------------------------

def test_eval_round2_dry_run_passes_explicit_eval_paths(tmp_path):
    rc = subprocess.run(
        [
            sys.executable,
            "round2/eval_round2.py",
            "--m1-adapter",
            "M1",
            "--a2-adapter",
            "A2",
            "--output-prefix",
            str(tmp_path / "m2"),
            "--dry-run",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr
    out = rc.stdout
    assert "--heldout-partition repro_rep2/data/heldout_pass16_partition.json" in out
    assert "--failures EDG-EXP1/results/a2_failures.json" in out
    assert "--exp2-root EDG-EXP2-struct" in out
    assert "--arena repro_rep2/data/v17_sibling_arena.json" in out
    assert "Qwen/Qwen2.5-7B-Instruct" in out
