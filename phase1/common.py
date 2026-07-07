#!/usr/bin/env python3
"""Shared helpers for Phase 1 mechanism-isolation scripts.

Phase 1 arms (preregistered in PROJECT_MASTER_PLAN Part IV, recipes locked by
the v1.26 ruling):

- A5  unverified-distilled: A3 pipeline with the AST filter switched off.
- A6  retrieval-patch: BM25 + pinned dense retriever, k in {1,3}, forward only.
- A7  replay-ablation: rep2 recipe minus replay (KL anchor therefore inert).
- A8  data-scale: 25% / 50% episode-level subsets, replay scaled at 2:1.
- A11 placebo-patch: token-scrambled patches, scaffold-on injection, forward only.

All scripts run from the repository root. GPU work is optional at build time:
every script has --dry-run producing the full plan without touching a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = REPO_ROOT / "repro_rep2"
EXP1_FAILURES = REPO_ROOT / "EDG-EXP1" / "results" / "a2_failures.json"
EXP2_ROOT = REPO_ROOT / "EDG-EXP2-struct"
EXP2_RESULT = EXP2_ROOT / "results" / "evolution_main_merged.json"
EPISODE_ID_DIR = REPRO_ROOT / "data" / "episode_ids"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "phase1_outputs"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "launch.json"

# Keep repro_rep2 importable as the `scripts` package used by the frozen
# Phase 0 code (lora_phase0, phase0_probe, evaluators).
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path=None):
    return read_json(path or DEFAULT_CONFIG)


def stable_int(*parts):
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_probe_module():
    from scripts import phase0_probe

    return phase0_probe


def load_inputs(split_seed, failures_path=None, exp2_result_path=None, exp2_root=None):
    """Baseline, split, H1 accepted patches, and the vendored evo module.

    This is the same loading path Phase 0 used (phase0_probe.load_inputs), so
    the split, the patch set, and the judge are byte-identical to rep2.
    """

    probe = load_probe_module()
    return probe.load_inputs(
        str(failures_path or EXP1_FAILURES),
        str(exp2_result_path or EXP2_RESULT),
        str(exp2_root or EXP2_ROOT),
        split_seed,
    )


def load_eval_surface_ids():
    """Frozen evaluation-surface episode ids. Training must be disjoint from all."""

    return {
        "d_val": _episode_id_file(EPISODE_ID_DIR / "D_val_failures.json"),
        "d_heldout": _episode_id_file(EPISODE_ID_DIR / "D_heldout_failures.json"),
        "old400": _episode_id_file(EPISODE_ID_DIR / "old400_success_eval.json"),
        "sibling300": _episode_id_file(EPISODE_ID_DIR / "sibling300_arena.json"),
    }


def _episode_id_file(path):
    payload = read_json(path)
    if isinstance(payload, dict):
        values = payload.get("episode_ids") or payload.get("arena_episode_ids")
    else:
        values = payload
    if values is None:
        raise ValueError(f"{path} does not contain episode ids")
    return {str(item) for item in values}


def assert_training_disjoint(rows, label):
    """Four-way leakage assert: training ∩ {D_val, D_heldout, old400, sibling300} = ∅.

    Fail-closed: rows without an episode_id raise instead of slipping through.
    """

    ids = set()
    missing = []
    for index, row in enumerate(rows):
        episode_id = row.get("episode_id")
        if not episode_id:
            missing.append(index)
            continue
        ids.add(str(episode_id))
    if missing:
        raise AssertionError({"assert": f"{label}_episode_id_present", "missing_count": len(missing), "examples": missing[:20]})
    surfaces = load_eval_surface_ids()
    report = {}
    for name, surface in surfaces.items():
        overlap = sorted(ids & surface)
        if overlap:
            raise AssertionError({"assert": f"{label}_disjoint_{name}", "overlap_count": len(overlap), "examples": overlap[:20]})
        report[f"{label}_intersect_{name}"] = 0
    return report


def replay_rows_from_capped(capped_path=None):
    """The frozen rep2 replay block (296 rows). Reused verbatim by A5/A8."""

    path = capped_path or (REPRO_ROOT / "data" / "distill_main_replay2_capped.jsonl")
    rows = read_jsonl(path)
    replay = [row for row in rows if row.get("source") == "replay_base_success"]
    core = [row for row in rows if row.get("source") != "replay_base_success"]
    if not replay or not core:
        raise AssertionError({"assert": "capped_dataset_shape", "replay": len(replay), "core": len(core)})
    return core, replay


def deterministic_sample(rows, count, seed, label):
    probe = load_probe_module()
    return probe.deterministic_sample(rows, count, seed, label)


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test item cap.")
    parser.add_argument("--dry-run", action="store_true", help="Build/validate everything possible without GPU inference.")


def resolve_model_id(config):
    return os.environ.get("EDR_MODEL_ID") or config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
