"""Pipeline contracts: DAG integrity, recipe locks, builder semantics.

Closes the historical blind-spot families: commands that render but consume
files nobody produces, arms drifting from the locked recipe, and asserts that
admit rows without ids.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edr.config import load_config  # noqa: E402
from edr.data.ablations import select_core_episode_uniform  # noqa: E402
from edr.evaluation.placebo import scramble_text  # noqa: E402
from edr.retrieval.patch_retrieval import BM25, build_query, top_k  # noqa: E402
from edr.runner.steps import ablation_steps, round2_steps  # noqa: E402


def _env():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    return env


# ---------------------------------------------------------------------------
# DAG integrity
# ---------------------------------------------------------------------------

def _steps_by_id(steps):
    by_id = {step.id: step for step in steps}
    assert len(by_id) == len(steps), "duplicate step ids"
    return by_id


def test_ablation_dag_needs_and_producers():
    steps = ablation_steps()
    by_id = _steps_by_id(steps)
    produced = {path for step in steps for path in step.produces}
    for step in steps:
        for dep in step.needs:
            assert dep in by_id, f"{step.id} needs unknown step {dep}"
        for index, token in enumerate(step.argv):
            if token in ("--dataset", "--train-dataset") and step.argv[index + 1].startswith("outputs/"):
                target = step.argv[index + 1]
                assert target in produced, f"{step.id} consumes {target} with no producer"


def test_round2_loop_searches_patches_over_f2_not_round1_failures():
    """The round-2 evolve loop must take F2 (M1's residual failures) as input.
    Defaulting to the round-1 failure set would silently re-teach solved
    episodes — the exact class of bug dry-runs cannot catch."""

    steps = _steps_by_id(round2_steps())
    argv = [str(a) for a in steps["r2.loop"].argv]
    assert argv[argv.index("--input") + 1] == "outputs/round2/f2_failures.json"
    assert argv[argv.index("--families") + 1] == "NL"
    assert "--base-adapter-dir" in argv, "raw-M0 patch search is invalid in round 2"


def test_round2_dag_needs_and_teacher2_once():
    steps = round2_steps()
    by_id = _steps_by_id(steps)
    for step in steps:
        for dep in step.needs:
            assert dep in by_id
    teacher2_evals = [step for step in steps if "--teacher2-loop-output" in step.argv]
    assert len(teacher2_evals) == 1, "teacher-2 forward must run exactly once across the 5 seeds"
    gate2 = by_id["r2.gate2"]
    eval_ids = [step.id for step in steps if step.id.startswith("r2.eval.")]
    assert set(eval_ids).issubset(set(gate2.needs))
    assert "r2.reconcile_m1" in gate2.needs


def test_training_steps_use_locked_recipe():
    steps = ablation_steps()
    trains = [step for step in steps if ".train." in step.id]
    assert len(trains) == 12  # a5 x3 + a7 x3 + a8_25 x3 + a8_50 x3
    config = load_config("ablations")
    for step in trains:
        argv = [str(a) for a in step.argv]
        assert argv[argv.index("--rank") + 1] == "16"
        assert float(argv[argv.index("--lr") + 1]) == 5e-5
        assert argv[argv.index("--epochs") + 1] == "3"
        kl = float(argv[argv.index("--kl-anchor-lambda") + 1])
        if ".a7." in step.id:
            assert kl == 0.0, "A7 must drop the KL anchor together with replay"
        else:
            assert kl == 2.0
        assert int(argv[argv.index("--seed") + 1]) in config["seeds"]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def test_a7_a8_builder_shapes(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "edr.data.ablations", "build-a7a8", "--output-dir", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=_env(),
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
        [sys.executable, "-m", "edr.data.ablations", "build-a5", "--output-dir", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert rc.returncode != 0
    assert "a5_shard_kind" in (rc.stderr + rc.stdout)


def test_a5_core_selection_is_episode_uniform_t0_first():
    rows = []
    for episode in ("e1", "e2", "e3"):
        rows.append({"episode_id": episode, "source": "teacher_unverified_t08", "sample_index": 0})
        rows.append({"episode_id": episode, "source": "teacher_unverified_t0", "sample_index": None})
    core = select_core_episode_uniform(rows, 3, seed=1)
    assert len(core) == 3
    assert {row["episode_id"] for row in core} == {"e1", "e2", "e3"}
    assert all(row["source"].endswith("_t0") for row in core), "first pass must take the T=0 row"
    deeper = select_core_episode_uniform(rows, 5, seed=1)
    assert sum(1 for row in deeper if row["source"].endswith("_t0")) == 3
    assert sum(1 for row in deeper if row["source"].endswith("_t08")) == 2
    assert select_core_episode_uniform(rows, 3, seed=1) == core, "deterministic given the same seed"


# ---------------------------------------------------------------------------
# A6 / A11 semantics
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
    assert "schema" not in query, "A6 query must contain function NAMES only"


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
    assert one == two
    assert sorted(one) == sorted(text)
    assert one != text


def test_a11_config_has_three_archived_scramble_seeds():
    seeds = load_config("ablations")["a11"]["scramble_seeds"]
    assert len(seeds) == 3 and len(set(seeds)) == 3


# ---------------------------------------------------------------------------
# Gate 2 classifier semantics
# ---------------------------------------------------------------------------

def test_gate2_classifier_categories():
    from edr.analysis.gate2 import classify

    ids = [f"e{i}" for i in range(100)]
    m1 = {e: 0.0 for e in ids}
    m2_better = [{e: 1.0 for e in ids}]
    result = classify(m1, m2_better, f1_size=74, f2_size=60, forget=0.0)
    assert result["classification"] == "compound"

    m2_same = [dict(m1)]
    result = classify(m1, m2_same, f1_size=74, f2_size=30, forget=0.0)
    assert result["classification"] == "converge"

    result = classify(m1, m2_better, f1_size=74, f2_size=60, forget=0.5)
    assert result["classification"] == "collapse", "forget gate break dominates"
