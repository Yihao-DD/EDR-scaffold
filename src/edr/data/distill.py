"""Distillation-row construction and deterministic sampling.

The row schema, dedupe rule, and hash-sampling labels are preregistered —
sampling seeds derive from the exact strings used here. Ported verbatim.
"""

from __future__ import annotations

import hashlib
import json

from edr.data.splits import build_base_prompt, failure_type, function_name
from edr.io_utils import read_jsonl
from edr.paths import DISTILL_TRAIN


def canonical_call(prediction):
    return {
        "name": prediction.get("name", ""),
        "arguments": prediction.get("arguments") or {},
    }


def call_key(prediction):
    return json.dumps(canonical_call(prediction), ensure_ascii=False, sort_keys=True)


def output_text(prediction):
    return json.dumps(canonical_call(prediction), ensure_ascii=False, sort_keys=True)


def make_distill_row(episode, prompt_template, prediction, source, partition, sample_index=None):
    return {
        "episode_id": episode["episode_id"],
        "source": source,
        "partition": partition,
        "failure_class": failure_type(episode),
        "function_name": function_name(episode),
        "sample_index": sample_index,
        "input": build_base_prompt(episode, prompt_template),
        "output": output_text(prediction),
        "output_call": canonical_call(prediction),
    }


def dedupe_rows(rows):
    result = []
    seen = set()
    for row in rows:
        key = (row["episode_id"], row["source"], row["output"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def deterministic_sample(rows, count, seed, label):
    if count >= len(rows):
        return list(rows)
    keyed = []
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{label}:{row['episode_id']}:{row.get('source')}:{row.get('output')}".encode("utf-8")).hexdigest()
        keyed.append((digest, row))
    return [row for _, row in sorted(keyed, key=lambda item: item[0])[:count]]


def replay_rows_from_train(train_path=None):
    """The frozen round-1 training set split into (core 148, replay 296)."""

    rows = read_jsonl(train_path or DISTILL_TRAIN)
    replay = [row for row in rows if row.get("source") == "replay_base_success"]
    core = [row for row in rows if row.get("source") != "replay_base_success"]
    if not replay or not core:
        raise AssertionError({"assert": "distill_train_shape", "replay": len(replay), "core": len(core)})
    return core, replay
