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

`evolution_loop_m1.py` is the only valid loop entry. It resolves M1 from
`repro_rep2/MANIFEST.json` unless `--base-adapter-dir` is supplied explicitly,
then loads `base -> PeftModel(A1) -> merge_and_unload()` in memory. Any loop run
on raw M0 is invalid.

Before accepting a rerun, execute:

```bash
python3 round2/loop/no_patch_equivalence_smoke.py \
  --collect-json round2_outputs/f2_failures.json \
  --limit 10
```

The smoke must match `collect_failures.py` raw outputs exactly for the first 10
F2 episodes.
