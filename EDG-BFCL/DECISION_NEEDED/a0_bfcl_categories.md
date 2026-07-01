# A0 BFCL Category Boundary

This file records category evidence for morning review. The overnight run connected only categories with one query, one function document, and one ground-truth function call per episode.

## Connected Tonight

| category | file | possible answer file | episodes | observed structure |
|---|---|---|---:|---|
| simple_python | `BFCL_v4_simple_python.json` | yes | 400 | one function doc, one ground-truth call |
| simple_java | `BFCL_v4_simple_java.json` | yes | 100 | one function doc, one ground-truth call |
| simple_javascript | `BFCL_v4_simple_javascript.json` | yes | 50 | one function doc, one ground-truth call |
| live_simple | `BFCL_v4_live_simple.json` | yes | 258 | one function doc, one ground-truth call |

## Not Connected Tonight

| category or family | observed evidence | reason for morning review |
|---|---|---|
| multiple | multiple candidate function docs | not the same one-function-doc structure as simple categories |
| parallel | multiple ground-truth function calls | outside single-call signal split |
| parallel_multiple | multiple functions and multiple calls | outside single-call signal split |
| irrelevance / live_relevance | no `possible_answer` single-call target in package data | relevance detection has different local signal |
| sql / rest / exec / chatable | located in `unused_datasets` or special evaluation structure in package wheel | category policy should be chosen by a human before inclusion |
| multi_turn / memory / web_search | multi-turn or agentic structures | outside tonight's single-turn single-function scope |

## Options For Morning

1. Keep tonight's connected category set: `simple_python`, `simple_java`, `simple_javascript`, `live_simple`.
2. Add selected special categories after writing category-specific local signal checks.
3. Use only non-live simple categories if `live_simple` is considered too different from static simple AST data.

## Evidence Files

- Machine result: `results/a0_bfcl_load.json`
- Human log: `logs/a0.md`
