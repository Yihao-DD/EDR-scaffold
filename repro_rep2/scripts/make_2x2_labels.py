#!/usr/bin/env python3
"""Generate per-arm SS/SN/NS/NN labels for a target evaluation surface.

S = seen in the arm's training set; N = new.
The first position is function-family visibility, the second is failure-class
visibility. For Round 2, call this with the cumulative T1+T2 training rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def family(function_name: str | None) -> str:
    if not function_name:
        return ""
    if "." in function_name:
        return function_name.split(".", 1)[0]
    return function_name.split("_", 1)[0]


def failure_type(row: dict) -> str:
    return str(row.get("failure_class") or row.get("failure_type") or "")


def rows_from_surface(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, dict):
        if "episodes" in payload:
            return payload["episodes"]
        if "records" in payload:
            return payload["records"]
    raise ValueError(f"cannot find records/episodes in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", action="append", required=True, help="Training JSONL; repeat for T1+T2.")
    parser.add_argument("--surface-json", required=True, help="Heldout/pass16/surface JSON with episodes or records.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = []
    for item in args.train_jsonl:
        train_rows.extend(read_jsonl(item))
    seen_families = {family(row.get("function_name") or (row.get("output_call") or {}).get("name")) for row in train_rows}
    seen_failures = {failure_type(row) for row in train_rows}
    seen_families.discard("")
    seen_failures.discard("")

    labels = []
    counts = {"SS": 0, "SN": 0, "NS": 0, "NN": 0}
    for row in rows_from_surface(Path(args.surface_json)):
        f_seen = family(row.get("function_name")) in seen_families
        e_seen = failure_type(row) in seen_failures
        label = ("S" if f_seen else "N") + ("S" if e_seen else "N")
        counts[label] += 1
        labels.append(
            {
                "episode_id": row["episode_id"],
                "function_name": row.get("function_name"),
                "failure_class": failure_type(row),
                "label": label,
            }
        )

    out = {"counts": counts, "n": len(labels), "labels": labels}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output} n={len(labels)} counts={counts}")


if __name__ == "__main__":
    main()
