"""Live scheduler test with fake CPU steps: ordering, resume, failure isolation."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from edr.runner.scheduler import Runner  # noqa: E402
from edr.runner.steps import Step  # noqa: E402

PY = sys.executable


def _touch_step(step_id, path, needs=(), fail=False):
    code = f"import sys; sys.exit(1)" if fail else f"from pathlib import Path; Path(r'{path}').parent.mkdir(parents=True, exist_ok=True); Path(r'{path}').write_text('done')"
    return Step(id=step_id, argv=[PY, "-c", code], needs=list(needs), produces=[str(path)])


def test_scheduler_runs_deps_in_order_and_resumes(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    steps = [
        _touch_step("t.a", a),
        _touch_step("t.b", b, needs=["t.a"]),
    ]
    runner = Runner(steps, {"gpu": {}}, state_path=tmp_path / "state.json", log_dir=tmp_path / "logs")
    counts = runner.run()
    assert counts.get("done") == 2 and not counts.get("failed")
    assert a.exists() and b.exists()

    # resume: rerun skips both (state done + outputs exist)
    runner2 = Runner(steps, {"gpu": {}}, state_path=tmp_path / "state.json", log_dir=tmp_path / "logs")
    counts2 = runner2.run()
    assert counts2.get("done") == 2
    state = runner2.state.data
    assert state["t.a"].get("skipped") or state["t.a"]["status"] == "done"

    # resume with missing output: b's file deleted -> b reruns
    b.unlink()
    runner3 = Runner(steps, {"gpu": {}}, state_path=tmp_path / "state.json", log_dir=tmp_path / "logs")
    runner3.run()
    assert b.exists(), "step with missing output must rerun on resume"


def test_scheduler_failure_blocks_dependents_only(tmp_path):
    bad = tmp_path / "never.txt"
    good = tmp_path / "good.txt"
    steps = [
        _touch_step("t.bad", bad, fail=True),
        _touch_step("t.child", tmp_path / "child.txt", needs=["t.bad"]),
        _touch_step("t.independent", good),
    ]
    runner = Runner(steps, {"gpu": {}}, state_path=tmp_path / "state.json", log_dir=tmp_path / "logs")
    counts = runner.run()
    assert counts.get("failed") == 1
    assert counts.get("blocked") == 1
    assert counts.get("done") == 1
    assert good.exists()
    assert runner.state.record("t.child")["status"] == "blocked"


def test_scheduler_marks_failed_when_output_missing_despite_rc0(tmp_path):
    ghost = tmp_path / "ghost.txt"
    steps = [Step(id="t.ghost", argv=[PY, "-c", "pass"], produces=[str(ghost)])]
    runner = Runner(steps, {"gpu": {}}, state_path=tmp_path / "state.json", log_dir=tmp_path / "logs")
    counts = runner.run()
    assert counts.get("failed") == 1, "exit 0 without declared outputs must be a failure"
