# GPU_SCHEDULING — lane design of the launcher

Ruling (v1.26): auto-detect, one whole card per step. This file is the design
record that ruling asked for.

## Model

1. At launch, `nvidia-smi --query-gpu=index,memory.free` is parsed. Every
   card with ≥ `gpu.min_free_mem_gb` (default 20 GB) free becomes a **lane**.
   `gpu.max_lanes` caps the count if set.
2. Every GPU step is a subprocess started with `CUDA_VISIBLE_DEVICES=<lane>`:
   exactly one whole card, never shared, no intra-step multi-GPU. All model
   code in this repo uses `device_map="auto"`, which then sees only its lane.
3. Concurrency = number of lanes. One card ⇒ a strict serial queue; N cards ⇒
   up to N steps in flight. CPU steps (dataset builders, reports) run inline
   in the launcher process whenever their dependencies are done.
4. The scheduler is dependency-driven, not phase-ordered: e.g. while A5's
   teacher resample occupies lane 0, A7's training can start on lane 1,
   because they share no dependency.
5. If a plan contains GPU steps and zero lanes are detected, the launcher
   refuses to start (no silent CPU fallback — a 7B forward pass on CPU would
   look like a hang).

## Memory expectations per step type

| step type | memory | notes |
|---|---|---|
| Phase-1 LoRA training (QLoRA, M0-based) | ~18–24 GB | 4-bit base + rank-16 LoRA, batch 1, grad-accum 8 |
| Phase-1 forwards (heldout/sibling/A6/A11, QLoRA) | ~12–18 GB | generation only |
| Phase-2 A2 training / M2 eval | **~40–48 GB** | merges M1 into bf16 before attaching/loading A2 (frozen code path prints its own warning). Use a ≥48 GB card |
| A6 dense index encoding | < 4 GB | MiniLM-class embedder; also runs on CPU |

## Operator knobs (configs/launch.json → "gpu")

- `min_free_mem_gb`: raise to exclude busy/small cards from lane detection.
- `max_lanes`: hard cap (e.g. set 1 to force fully serial execution).
- To pin specific cards, set `CUDA_VISIBLE_DEVICES` before `run.py`; detected
  indices are then relative to that mask.

## Failure semantics

- A step OOM-ing fails only itself; the scheduler keeps other lanes running
  and blocks only that step's dependents. Fix (bigger card / fewer lanes) and
  re-run the same `run.py` command — done steps are skipped by the resume
  rule (state `done` + output files exist).
- Two launcher processes must not run simultaneously (single state file; no
  lock is taken). Run one phase command at a time.
- Heterogeneous cards: lanes are whatever passes `min_free_mem_gb`. For
  Phase 2 on a mixed box (e.g. one 24 GB + one 48 GB), either set
  `CUDA_VISIBLE_DEVICES` to the big card for `run.py phase2`, or raise
  `min_free_mem_gb` above the small card's size.
