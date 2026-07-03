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


def test_round2_loop_dry_run_uses_m1_aware_vendor(tmp_path):
    cmd = [
        "python3",
        "round2/loop/evolution_loop_round2.py",
        "--m1-adapter",
        "M1_ADAPTER",
        "--f2",
        "F2.json",
        "--seeds",
        "20260630",
        "--output-dir",
        str(tmp_path / "loop"),
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=REPO, check=True, text=True, capture_output=True)
    assert "round2/loop/evolution_loop_m1.py" in result.stdout
    assert "EDG-EXP2-struct/scripts/evolution_loop.py" not in result.stdout
    assert "--base-adapter-dir M1_ADAPTER" in result.stdout
    assert "--manifest repro_rep2/MANIFEST.json" in result.stdout


def test_vendor_loop_requires_and_merges_m1_adapter():
    text = (REPO / "round2/loop/evolution_loop_m1.py").read_text(encoding="utf-8")
    assert "PeftModel.from_pretrained" in text
    assert "merge_and_unload()" in text
    assert "raw M0 patch search is invalid" in text


def test_train_round2_dry_run_anchors_kl_to_m1(tmp_path):
    train = tmp_path / "t2.jsonl"
    replay = tmp_path / "replay2.jsonl"
    train.write_text('{"episode_id":"e1","input":"x","output":"y"}\n', encoding="utf-8")
    replay.write_text('{"episode_id":"e2","input":"x","output":"y","partition":"replay"}\n', encoding="utf-8")
    cmd = [
        "python3", "round2/train_round2.py",
        "--m1-adapter", "M1_ADAPTER",
        "--train-jsonl", str(train),
        "--replay-jsonl", str(replay),
        "--seed", "20260708",
        "--output-adapter-dir", str(tmp_path / "a2"),
        "--output-json", str(tmp_path / "a2.json"),
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=REPO, check=True, text=True, capture_output=True)
    # KL reference model is M1: A2 trains on top of --base-adapter-dir=M1 with KL lambda 2.
    assert "--base-adapter-dir M1_ADAPTER" in result.stdout
    assert "--kl-anchor-lambda 2" in result.stdout


def test_kl_reference_is_merged_m1_not_raw_m0():
    text = (REPO / "repro_rep2/scripts/lora_phase0.py").read_text(encoding="utf-8")
    # (a) A2 stacks on M1: merge the frozen base adapter BEFORE attaching the trainable LoRA.
    merge_at = text.index("merge_base_adapter(model, args.base_adapter_dir)")
    apply_at = text.index("apply_lora(model, args.rank")
    assert merge_at < apply_at, "M1 must be merged before A2 is attached"
    assert "model = model.merge_and_unload()" in text
    # (b) KL reference is the merged M1: disable_adapter() drops only A2, and metadata records it.
    assert "with model.disable_adapter():" in text
    assert '"kl_anchor_reference": "merged_base_adapter" if args.base_adapter_dir else "raw_base_disable_adapter"' in text


# --- input-file existence: every consumed intermediate must have a producer ---

def test_build_t2_produces_t2_and_t1_t2(tmp_path):
    """build_replay2 consumes t2.jsonl; eval_round2 consumes t1_t2.jsonl. Both must be produced."""
    samples = tmp_path / "teacher2_samples.jsonl"
    samples.write_text(
        '{"episode_id":"mock_ep_1","ast_pass":true,"output":"a","function_name":"fn_a","failure_class":"router"}\n'
        '{"episode_id":"mock_ep_2","ast_pass":true,"output":"b","function_name":"fn_b","failure_class":"validator"}\n',
        encoding="utf-8",
    )
    t2 = tmp_path / "t2.jsonl"
    t1t2 = tmp_path / "t1_t2.jsonl"
    cmd = [
        "python3", "round2/build_t2.py",
        "--teacher2-samples", str(samples),
        "--output", str(t2),
        "--t1t2-output", str(t1t2),
        "--summary-output", str(tmp_path / "t2_summary.json"),
    ]
    subprocess.run(cmd, cwd=REPO, check=True, text=True, capture_output=True)
    assert t2.exists() and t1t2.exists()
    # t1_t2 must include the 74 unique main-core episodes plus the mock T2 episodes.
    t1t2_ids = {__import__("json").loads(line)["episode_id"] for line in t1t2.read_text().splitlines() if line.strip()}
    assert {"mock_ep_1", "mock_ep_2"} <= t1t2_ids
    assert len(t1t2_ids) == 76


def test_build_m1_success_dry_run_produces_replay_source(tmp_path):
    """build_replay2 consumes m1_train_success.jsonl; build_m1_success must produce it."""
    out = tmp_path / "m1_train_success.jsonl"
    cmd = [
        "python3", "round2/build_m1_success.py",
        "--dry-run",
        "--output", str(out),
        "--summary-output", str(tmp_path / "summary.json"),
    ]
    subprocess.run(cmd, cwd=REPO, check=True, text=True, capture_output=True)
    assert out.exists()
    assert sum(1 for line in out.read_text().splitlines() if line.strip()) == 120
