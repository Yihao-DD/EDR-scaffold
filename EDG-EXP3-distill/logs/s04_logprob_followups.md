# s04 Log-Prob Follow-Ups

Registered follow-up analyses computed from `results/s04_logprob_probe.json`; no additional GPU scoring was needed.

## H-margin follow-up

| group | n | mean | median |
| --- | ---: | ---: | ---: |
| first_shared_fragile correct logprob | 24 | -1.1728 | -1.0609 |
| non_first_shared correct logprob | 362 | -1.1376 | -1.0821 |
| ep2_union_fragile correct logprob | 63 | -1.2096 | -1.0893 |
| fragile24 margin main_r8_lr2e-5_ep1 | 24 | 0.0356 | 0.0648 |
| fragile24 margin main_r8_lr2e-5_ep2 | 14 | -0.1896 | -0.0553 |
| fragile24 margin star_r8_lr2e-5_ep1 | 24 | 0.0340 | 0.0648 |
| fragile24 margin star_r8_lr2e-5_ep2 | 13 | -0.1758 | -0.0772 |

Margin controls are unavailable for non-fragile episodes because no observed flipped-to output exists for them in the logged grid JSONs; this tests the registered fragile-24 actual-flip margin directly.

## Dose-order prediction

| arm | partition | ep1 transferred | ep2 new | not by ep2 |
| --- | --- | ---: | ---: | ---: |
| main | all_validation_teacher_repaired | 13 / -0.9919 | 24 / -1.1117 | 26 / -1.1517 |
| main | scaffold_only | 7 / -1.0011 | 23 / -1.1184 | 25 / -1.1697 |
| star | all_validation_teacher_repaired | 11 / -0.9960 | 21 / -1.1213 | 31 / -1.1295 |
| star | scaffold_only | 6 / -0.9968 | 19 / -1.1409 | 30 / -1.1438 |

## Scaffold-only transfer success/failure

| arm | grid | transferred | not transferred |
| --- | --- | ---: | ---: |
| main | main_r8_lr2e-5_ep1 | 7 / -1.0011 | 48 / -1.1451 |
| main | main_r8_lr2e-5_ep2 | 28 / -1.0923 | 27 / -1.1625 |
| star | star_r8_lr2e-5_ep1 | 6 / -0.9968 | 49 / -1.1427 |
| star | star_r8_lr2e-5_ep2 | 23 / -1.1093 | 32 / -1.1394 |
