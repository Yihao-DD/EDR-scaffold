# A2 Multiple Router Count Below Threshold

## Scope

PROBE signal (n=1253). BFCL Multiple AST natural failure collection with frozen `Qwen/Qwen2.5-7B-Instruct`.

## Multiple Final Counts

| metric | value |
|---|---:|
| total Multiple episodes | 1253 |
| Multiple failures | 627 |
| router_fail | 23 |
| validator_fail | 604 |

## Pre-Registered Gate Marker

The attached spec says to record this file if Multiple `router_fail < 30`.

| criterion | observed |
|---|---:|
| router_fail threshold | 30 |
| observed router_fail | 23 |

## Human Decision Options

- Keep this Multiple run as the main Phase A evidence and decide whether `router_fail=23` is enough for the intended router-vs-validator allocation question.
- Run a smaller or different frozen model to increase function-selection pressure.
- Filter or define a harder Multiple subset before Phase B.

No option is selected here.
