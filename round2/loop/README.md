# Round 2 NL-Evo Loop Vendor Note

This directory is the Phase 2 handoff slot for the minimal NL-evo loop adapted
from `EDG-EXP2-struct`. The production loop should load the model stack
`M0 + A1`, search patches on the round-2 train failures, and accept patches
only by AST judge improvement on `D_val`.

The wrapper intentionally keeps the policy narrow:

- base model: Qwen2.5-7B-Instruct;
- stack: load base, merge A1 in memory, run H2 search;
- acceptance: validation AST only;
- generation: one NL-evo round using the Phase 1 policy;
- outputs: accepted patch list plus teacher-2 samples for `build_t2.py`.
