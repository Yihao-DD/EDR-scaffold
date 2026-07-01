# A4 Diagnosis And Patch Operator Space

## Scope

This step defines interfaces and allowed patch types only. Diagnosis and patch generation remain locked for Phase B.

## Diagnostic Interface

`gate_scaffold/diagnosis_skeleton.py` defines:

- `DiagnosisRecord`
- `diagnose(failure_episode, trace, harness_layers)`

`diagnose` raises `NotImplementedError("Phase B, human present")`.

## Router Patch Types

- Function alias table: maps intent wording to a function name.
- Function description enhancement: adds external disambiguating metadata.
- Brief function-routing prompt: adds a scoped routing hint.
- Function pool reorder / disambiguation: changes ranking or presentation of candidate functions.

Each patch must be versioned, rollbackable, validation-costed, and able to affect a class of future episodes rather than only one instance.

## Validator Patch Types

- Parameter alias rule.
- Type conversion rule.
- Required-parameter check.
- Retry or parser fallback.
- Schema constraint rule.

Each patch must be versioned, rollbackable, validation-costed, and able to affect a class of future episodes rather than only one instance.

## Patch Interface

`gate_scaffold/patch_skeleton.py` defines:

- `Patch`
- `generate_patches(failure_episode, diagnosis)`

`generate_patches` raises `NotImplementedError("Phase B, human present")`.

## Assert Results

| assert | result |
|---|---|
| diagnostic skeleton imports | T |
| patch skeleton imports | T |
| `diagnose` raises `NotImplementedError` | T |
| `generate_patches` raises `NotImplementedError` | T |

Machine result: `results/a4_a5_a6_scaffold.json`.
