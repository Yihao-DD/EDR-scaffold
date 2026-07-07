#!/usr/bin/env python3
"""Step scheduler: GPU lanes + inline CPU steps, resumable, fail-isolated.

Execution model (docs/gpu_scheduling.md):
- GPU steps run as subprocesses pinned to one whole card via
  CUDA_VISIBLE_DEVICES; concurrency = number of detected lanes.
- CPU steps run inline when their dependencies are done.
- A failed step blocks only its dependents; independent branches continue.
- Every step logs to outputs/logs/<step>.log; state to outputs/state.json.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from edr.runner.gpus import detect_lanes
from edr.runner.state import StateStore
from edr.paths import LOG_DIR, REPO_ROOT, STATE_FILE

POLL_SECONDS = 10


def _env_for(lane):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    if lane is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(lane)
    return env


class Runner:
    def __init__(self, steps, config, state_path=None, log_dir=None):
        self.steps = {step.id: step for step in steps}
        self.order = [step.id for step in steps]
        self.config = config
        self.state = StateStore(state_path or STATE_FILE)
        self.log_dir = Path(log_dir or LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.running = {}  # step_id -> (Popen, lane, log handle)

    # -- status helpers ---------------------------------------------------
    def status_of(self, step_id):
        return self.state.record(step_id).get("status", "pending")

    def deps_done(self, step):
        return all(self.status_of(dep) == "done" for dep in step.needs)

    def deps_failed(self, step):
        return any(self.status_of(dep) in ("failed", "blocked") for dep in step.needs)

    # -- execution ---------------------------------------------------------
    def dry_run(self):
        lanes = detect_lanes(self.config.get("gpu", {}))
        print(f"# plan: {len(self.order)} steps; gpu lanes detected: {lanes or 'none'}")
        for step_id in self.order:
            step = self.steps[step_id]
            marker = "GPU" if step.gpu else "cpu"
            state = "done" if self.state.is_done(step, REPO_ROOT) else self.status_of(step_id)
            needs = f" needs={step.needs}" if step.needs else ""
            print(f"[{marker}] {step_id} ({state}){needs}")
            print(f"      $ {' '.join(step.argv)}")

    def _block_failed_dependents(self):
        for step_id in self.order:
            step = self.steps[step_id]
            if self.status_of(step_id) == "pending" and self.deps_failed(step):
                self.state.mark(step_id, "blocked")

    def _reset_stale_done(self):
        """Resume contract: state 'done' is honored only while every declared
        output still exists; otherwise the step reruns."""

        for step_id in self.order:
            step = self.steps[step_id]
            if self.state.record(step_id).get("status") == "done" and not self.state.is_done(step, REPO_ROOT):
                self.state.mark(step_id, "pending", note="outputs missing on resume; rerunning")
                print(f"[resume] {step_id}: declared outputs missing -> rerunning")

    def run(self):
        self.state.reset_failed()
        self._reset_stale_done()
        lanes = detect_lanes(self.config.get("gpu", {}))
        gpu_steps_present = any(step.gpu for step in self.steps.values())
        if gpu_steps_present and not lanes:
            raise SystemExit(
                "no usable GPU detected (nvidia-smi missing or all cards below min_free_mem_gb) "
                "but the plan contains GPU steps. Fix the environment or run with --dry-run."
            )
        free_lanes = list(lanes)
        print(f"[runner] {len(self.order)} steps, gpu lanes: {lanes or 'none'}")

        while True:
            progressed = self._reap()
            free_lanes = [lane for lane in lanes if lane not in {info[1] for info in self.running.values()}]
            self._block_failed_dependents()
            # launch / execute ready steps
            for step_id in self.order:
                step = self.steps[step_id]
                if self.status_of(step_id) != "pending" or step_id in self.running:
                    continue
                if self.state.is_done(step, REPO_ROOT):
                    self.state.mark(step_id, "done", skipped="resume: outputs already present")
                    progressed = True
                    continue
                if not self.deps_done(step):
                    continue
                if step.gpu:
                    if not free_lanes:
                        continue
                    lane = free_lanes.pop(0)
                    self._launch(step, lane)
                    progressed = True
                else:
                    self._run_inline(step)
                    progressed = True
            if not self.running and not any(
                self.status_of(step_id) == "pending" and not self.deps_failed(self.steps[step_id])
                for step_id in self.order
            ):
                # final pass so dependents of late failures end as 'blocked', not 'pending'
                self._block_failed_dependents()
                break
            if not progressed:
                time.sleep(POLL_SECONDS)

        return self.summary()

    def _launch(self, step, lane):
        log_path = self.log_dir / f"{step.id}.log"
        handle = log_path.open("a", encoding="utf-8")
        handle.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} lane={lane} =====\n$ {' '.join(step.argv)}\n")
        handle.flush()
        proc = subprocess.Popen(
            step.argv,
            cwd=REPO_ROOT,
            env=_env_for(lane),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        self.running[step.id] = (proc, lane, handle)
        self.state.mark(step.id, "running", lane=str(lane), log=str(log_path))
        print(f"[start] {step.id} on gpu {lane} -> {log_path}")

    def _run_inline(self, step):
        log_path = self.log_dir / f"{step.id}.log"
        self.state.mark(step.id, "running", lane="cpu", log=str(log_path))
        print(f"[start] {step.id} (cpu)")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== inline {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n$ {' '.join(step.argv)}\n")
            handle.flush()
            proc = subprocess.run(
                step.argv,
                cwd=REPO_ROOT,
                env=_env_for(None),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        self._finish(step, proc.returncode, log_path)

    def _reap(self):
        progressed = False
        for step_id in list(self.running):
            proc, lane, handle = self.running[step_id]
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            del self.running[step_id]
            self._finish(self.steps[step_id], rc, self.log_dir / f"{step_id}.log")
            progressed = True
        return progressed

    def _finish(self, step, returncode, log_path):
        if returncode == 0:
            missing = [p for p in step.produces if not (REPO_ROOT / p).exists()]
            if missing:
                self.state.mark(step.id, "failed", rc=returncode, missing_outputs=missing)
                print(f"[FAIL] {step.id}: exit 0 but missing outputs {missing}")
                return
            self.state.mark(step.id, "done", rc=0)
            print(f"[done] {step.id}")
        else:
            self.state.mark(step.id, "failed", rc=returncode)
            tail = _tail(log_path)
            print(f"[FAIL] {step.id} rc={returncode}\n--- log tail ---\n{tail}\n----------------")

    def summary(self):
        counts = {}
        failed = []
        for step_id in self.order:
            status = self.status_of(step_id)
            counts[status] = counts.get(status, 0) + 1
            if status in ("failed", "blocked"):
                failed.append((step_id, status))
        print(f"[runner] summary: {counts}")
        for step_id, status in failed:
            print(f"  - {status}: {step_id} (log: outputs/logs/{step_id}.log)")
        return counts


def _tail(path, lines=15):
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError:
        return "(no log)"
