#!/usr/bin/env python3
"""Idempotent run state. One JSON file, one record per step.

Resume rule: a step is skipped iff its state is `done` AND every file in its
`produces` list exists. A `failed` step is reset to `pending` on the next
launcher invocation (retry-on-resume); its dependents stay blocked until it
succeeds. State transitions are flushed to disk immediately.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class StateStore:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def record(self, step_id):
        return self.data.get(step_id, {"status": "pending"})

    def mark(self, step_id, status, **extra):
        entry = self.data.setdefault(step_id, {})
        entry["status"] = status
        entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        entry.update(extra)
        self.save()

    def reset_failed(self):
        for step_id, entry in self.data.items():
            if entry.get("status") in ("failed", "running"):
                entry["status"] = "pending"
        self.save()

    def is_done(self, step, repo_root):
        entry = self.record(step.id)
        if entry.get("status") != "done":
            return False
        for produced in step.produces:
            if not (Path(repo_root) / produced).exists():
                return False
        return True
