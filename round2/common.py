#!/usr/bin/env python3
"""Shared helpers for Phase 2 round-2 handoff scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPRO_ROOT = REPO_ROOT / "repro_rep2"
DEFAULT_DATA_ROOT = DEFAULT_REPRO_ROOT / "data"
DEFAULT_ARTIFACT_ROOT = DEFAULT_REPRO_ROOT / "artifacts"

ROUND2_SEEDS = [20260708, 20260709, 20260710, 20260711, 20260712]
ROUND2_RECIPE = {
    "rank": 16,
    "learning_rate": "5e-5",
    "epochs": 3,
    "replay_ratio": "2:1",
    "kl_lambda": 2.0,
    "kl_anchor": "M1",
    "guards": "non-finite skip + grad clipping",
}


def read_json(path: os.PathLike[str] | str):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: os.PathLike[str] | str, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: os.PathLike[str] | str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: os.PathLike[str] | str, rows: Iterable[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def episode_ids(rows: Iterable[Mapping]) -> set[str]:
    return {str(row["episode_id"]) for row in rows if row.get("episode_id")}


def load_episode_id_file(path: os.PathLike[str] | str) -> set[str]:
    payload = read_json(path)
    if isinstance(payload, dict):
        values = payload.get("episode_ids") or payload.get("arena_episode_ids")
    else:
        values = payload
    if values is None:
        raise ValueError(f"{path} does not contain episode_ids")
    return {str(item) for item in values}


def assert_disjoint(name: str, left: Iterable[str], right: Iterable[str]) -> dict:
    overlap = sorted(set(left) & set(right))
    if overlap:
        raise AssertionError({"assert": name, "overlap_count": len(overlap), "examples": overlap[:20]})
    return {"assert": name, "overlap_count": 0}


def sha256_file(path: os.PathLike[str] | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: os.PathLike[str] | str, root: os.PathLike[str] | str = REPO_ROOT) -> dict:
    path = Path(path)
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repro-root", default=str(DEFAULT_REPRO_ROOT))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-dir", default="round2_outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test item cap.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write plan without GPU inference.")


def slice_limit(rows: Sequence[dict], limit: int | None) -> list[dict]:
    return list(rows[:limit]) if limit is not None else list(rows)
