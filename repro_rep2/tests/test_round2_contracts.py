import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "round2"))

from round2.build_replay2 import repeat_to_count
from round2.common import episode_ids


def test_episode_ids_fail_closed_on_missing_id():
    with pytest.raises(AssertionError):
        episode_ids([{"episode_id": "ok"}, {"output": "missing"}])


def test_replay_rows_are_marked_for_kl_anchor():
    rows = repeat_to_count([{"episode_id": "e1", "input": "x", "output": "y"}], 2)
    assert len(rows) == 2
    assert all(row["source"] == "replay_base_success" for row in rows)
    assert all(row["partition"] == "replay" for row in rows)


def test_round2_eval_dry_run_passes_m1_adapter_to_all_eval_surfaces(tmp_path):
    cmd = [
        "python3",
        "round2/eval_round2.py",
        "--m1-adapter",
        "M1_ADAPTER",
        "--a2-adapter",
        "A2_ADAPTER",
        "--train-signature-jsonl",
        "T1_T2.jsonl",
        "--phase0-failures",
        "failures.json",
        "--phase0-exp2-root",
        "exp2",
        "--phase0-s00-input",
        "s00.json",
        "--output-prefix",
        str(tmp_path / "m2"),
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=REPO, check=True, text=True, capture_output=True)
    assert result.stdout.count("--base-adapter-dir M1_ADAPTER") == 3
    assert "--adapter-dir A2_ADAPTER" in result.stdout
    assert "--train-dataset T1_T2.jsonl" in result.stdout
