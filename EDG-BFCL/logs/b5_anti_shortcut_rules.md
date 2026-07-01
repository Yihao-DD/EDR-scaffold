# B5 Anti-Shortcut Rules For Morning Gate

## Scope

This document is a morning review checklist. It implements no gate logic.

## Rules

1. `r_hat` must come only from historical measured net gain and cost. The allocator must not read failure type, function name, error name, query text, or any other semantic field.
2. Every failure episode must receive both router and validator candidate patches. Candidate generation must not prefilter by `failure_class`.
3. Oracle must be empirical: enumerate both components' candidate patches, measure net gain on the validation suite, and choose the measured maximum. Human component labels are not allowed.
4. Online validation must use only BFCL AST local signals. LLM judge calls and cloud judge calls are not allowed in the validation loop.
5. The first gate run must include two ablations:
   - replace `r_hat` with random and constant values;
   - shuffle candidate enumeration order.
6. Gate criteria must be registered before a morning run and not changed after results are visible:
   - tight tier comparison against uniform / random / single-component arms;
   - update-both behavior under loose and tight tiers;
   - allocation changes across loose to tight tiers;
   - both ablations included;
   - gap to empirical oracle recorded.

## Locked Tonight

- patch generation
- utility estimation
- oracle
- repair experiment

Machine result: `results/b5_anti_shortcut_rules.json`.
