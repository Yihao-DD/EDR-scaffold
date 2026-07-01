# Three-Family Self-Evolution Main Result

PROBE signal (n_seeds=5). BFCL Multiple natural failures are split train/validation/held-out by deterministic hash per seed.

## Held-Out Pass Rate

| method | overall mean | overall CI95 | router mean | router CI95 | validator mean | validator CI95 |
|---|---:|---:|---:|---:|---:|---:|
| no-evolution | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| NL-evo | 0.2405 | 0.0444 | 0.0900 | 0.1091 | 0.2447 | 0.0450 |
| NL-padded-evo | 0.2114 | 0.0382 | 0.0900 | 0.1091 | 0.2149 | 0.0385 |
| STRUCT-evo | 0.2127 | 0.0413 | 0.0400 | 0.0784 | 0.2175 | 0.0407 |
| oracle | 0.4532 | 0.0186 | 0.1986 | 0.2173 | 0.4595 | 0.0199 |

## Context Cost

| seed | method | accepted patches | mean patch tokens | total patch tokens |
|---:|---|---:|---:|---:|
| 20260630 | NL-evo | 39 | 414.03 | 16147 |
| 20260630 | NL-padded-evo | 36 | 454.42 | 16359 |
| 20260630 | STRUCT-evo | 39 | 513.31 | 20019 |
| 20260631 | NL-evo | 26 | 370.85 | 9642 |
| 20260631 | NL-padded-evo | 27 | 402.74 | 10874 |
| 20260631 | STRUCT-evo | 30 | 409.47 | 12284 |
| 20260632 | NL-evo | 31 | 403.35 | 12504 |
| 20260632 | NL-padded-evo | 32 | 436.97 | 13983 |
| 20260632 | STRUCT-evo | 32 | 427.53 | 13681 |
| 20260633 | NL-evo | 34 | 390.29 | 13270 |
| 20260633 | NL-padded-evo | 36 | 411.19 | 14803 |
| 20260633 | STRUCT-evo | 37 | 439.46 | 16260 |
| 20260634 | NL-evo | 29 | 384.31 | 11145 |
| 20260634 | NL-padded-evo | 29 | 407.28 | 11811 |
| 20260634 | STRUCT-evo | 30 | 467.37 | 14021 |
