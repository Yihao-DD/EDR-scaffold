#!/usr/bin/env python3
"""GPU lane detection. Design rationale: docs/GPU_SCHEDULING.md.

Policy (v1.26 ruling): auto-detect. Every GPU step gets exactly one whole GPU
via CUDA_VISIBLE_DEVICES; the number of concurrent lanes equals the number of
detected cards with enough free memory. One card -> serial queue; N cards ->
N parallel lanes. No step ever shares a card with another step.
"""

from __future__ import annotations

import subprocess


def detect_lanes(gpu_config):
    """Return the list of usable GPU indices (as strings). Empty = no GPU."""

    min_free_mb = float(gpu_config.get("min_free_mem_gb", 20)) * 1024
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    lanes = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        index, free_mb = parts[0], float(parts[1])
        if free_mb >= min_free_mb:
            lanes.append(index)
    max_lanes = gpu_config.get("max_lanes")
    if max_lanes:
        lanes = lanes[: int(max_lanes)]
    return lanes
