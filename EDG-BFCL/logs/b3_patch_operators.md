# B3 Patch Operators

## Scope

PROBE signal (n=5 patch/operator interfaces). This step documents the allowed operator space and defines locked interfaces only.

## Router Patch Types

- Function alias rule: maps observed intent phrasing to a BFCL function name.
- Function description enhancement: external metadata that clarifies when to choose a function.
- Routing example: external example used by a router component.

Each router patch must be versioned, rollbackable, and able to affect a class of future queries rather than one episode only.

## Validator / Parser Patch Types

- Parameter alias rule: maps query wording to a function parameter name.
- Type conversion rule: converts local values to the expected BFCL parameter type.
- Required-parameter check: detects missing required parameters before validation.
- Default-value rule: external defaulting rule where the function schema permits it.

Each validator patch must be versioned, rollbackable, and able to affect a class of future queries rather than one episode only.

## Locked Interfaces

- `generate_router_patches`
- `generate_validator_patches`
- `apply_patch`
- `rollback_patch`
- `estimate_utility`

## Assert Results

| assert | result |
|---|---|
| all interfaces import | T |
| all interfaces raise `NotImplementedError` when called | T |

Machine result: `results/b3_patch_operators_skeleton.json`.
