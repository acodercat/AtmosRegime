# GPU hours per RL run (`explore/gpu_hours.py`)

Runs whose endpoint was evaluated. Wall clock summed over the attempts of the run; GPU h = wall × GPUs (DAPO 8; OPD student + teacher).
Warm-start SFT (< 1 h each), evaluation and pass@64 sampling are not included. OPD run logs carry no step at exit; all 12 ended at step 120 (chain gate lines for 10, merge logs of `global_step_120` for P seed 42).

| study | runs | GPU h |
|---|--:|--:|
| DOSE | 4 | 378 |
| DOSE-text | 4 | 249 |
| Ministral DOSE | 4 | 768 |
| OPD | 12 | 200 |
| X | 30 | 2,587 |
| X Qwen3.5-9B seed 43 | 10 | 1,420 |
| X R arm | 4 | 150 |
| **total** | **68** | **5,752** |

| run | study | GPUs | last step logged | exit rc | wall h | GPU h |
|---|---|--:|--:|--:|--:|--:|
| `dose-qwen3-8b-F1-WA-s42` | DOSE | 8 | 120 | 0 | 9.0 | 72 |
| `dose-qwen3-8b-F1-WA-s43` | DOSE | 8 | 120 | 0 | 8.2 | 66 |
| `dose-qwen3-8b-F2-WA-s42` | DOSE | 8 | 120 | 0 | 23.4 | 187 |
| `dose-qwen3-8b-F2-WA-s43` | DOSE | 8 | 120 | 0 | 6.7 | 54 |
| `dtext-qwen3-8b-F1-WA-s42` | DOSE-text | 8 | 120 | 0 | 8.5 | 68 |
| `dtext-qwen3-8b-F1-WA-s43` | DOSE-text | 8 | 120 | 0 | 9.1 | 72 |
| `dtext-qwen3-8b-F2-WA-s42` | DOSE-text | 8 | 120 | 0 | 7.1 | 57 |
| `dtext-qwen3-8b-F2-WA-s43` | DOSE-text | 8 | 120 | 0 | 6.5 | 52 |
| `dose-ministral3-8b-F1-WA-s42` | Ministral DOSE | 8 | 95 | 124 | 24.0 | 192 |
| `dose-ministral3-8b-F1-WA-s43` | Ministral DOSE | 8 | 90 | 124 | 24.0 | 192 |
| `dose-ministral3-8b-F2-WA-s42` | Ministral DOSE | 8 | 90 | 124 | 24.0 | 192 |
| `dose-ministral3-8b-F2-WA-s43` | Ministral DOSE | 8 | 105 | 124 | 24.0 | 192 |
| `opd-qwen3-8b-F1-P-s42` | OPD | 5 | — | 0 | 4.1 | 20 |
| `opd-qwen3-8b-F1-P-s43` | OPD | 5 | — | 0 | 3.5 | 17 |
| `opd-qwen3-8b-F1-Pk2-s42` | OPD | 5 | — | 0 | 2.8 | 14 |
| `opd-qwen3-8b-F1-Pk3-s42` | OPD | 5 | — | 0 | 2.9 | 14 |
| `opd-qwen3-8b-F1-Pshuf-s42` | OPD | 5 | — | 0 | 3.5 | 17 |
| `opd-qwen3-8b-F1-S-s42` | OPD | 5 | — | 0 | 3.4 | 17 |
| `opd-qwen3-8b-F2-P-s42` | OPD | 5 | — | 0 | 3.5 | 18 |
| `opd-qwen3-8b-F2-P-s43` | OPD | 5 | — | 0 | 3.5 | 18 |
| `opd-qwen3-8b-F2-Pk2-s42` | OPD | 5 | — | 0 | 2.9 | 15 |
| `opd-qwen3-8b-F2-Pk3-s42` | OPD | 5 | — | 0 | 2.9 | 15 |
| `opd-qwen3-8b-F2-Pshuf-s42` | OPD | 5 | — | 0 | 3.5 | 18 |
| `opd-qwen3-8b-F2-S-s42` | OPD | 5 | — | 0 | 3.5 | 17 |
| `x-qwen3-8b-F1-A-s42` | X | 8 | 120 | 0 | 6.3 | 50 |
| `x-qwen3-8b-F1-A-s43` | X | 8 | 120 | 0 | 6.4 | 52 |
| `x-qwen3-8b-F1-D-s42` | X | 8 | 120 | 0 | 7.6 | 60 |
| `x-qwen3-8b-F1-D-s43` | X | 8 | 120 | 0 | 7.6 | 61 |
| `x-qwen3-8b-F1-F-s42` | X | 8 | 120 | 0 | 4.8 | 38 |
| `x-qwen3-8b-F1-F-s43` | X | 8 | 120 | 0 | 4.9 | 40 |
| `x-qwen3-8b-F1-WA-s42` | X | 8 | 120 | 0 | 11.6 | 92 |
| `x-qwen3-8b-F1-WA-s43` | X | 8 | 120 | 0 | 9.1 | 73 |
| `x-qwen3-8b-F1-WD-s42` | X | 8 | 120 | 0 | 11.7 | 94 |
| `x-qwen3-8b-F1-WD-s43` | X | 8 | 120 | 0 | 11.2 | 90 |
| `x-qwen3-8b-F2-A-s42` | X | 8 | 120 | 0 | 5.4 | 43 |
| `x-qwen3-8b-F2-A-s43` | X | 8 | 120 | 0 | 5.3 | 42 |
| `x-qwen3-8b-F2-D-s42` | X | 8 | 120 | 0 | 6.2 | 50 |
| `x-qwen3-8b-F2-D-s43` | X | 8 | 120 | 0 | 5.7 | 45 |
| `x-qwen3-8b-F2-F-s42` | X | 8 | 120 | 0 | 3.9 | 31 |
| `x-qwen3-8b-F2-F-s43` | X | 8 | 120 | 0 | 3.8 | 30 |
| `x-qwen3-8b-F2-WA-s42` | X | 8 | 120 | 0 | 9.9 | 79 |
| `x-qwen3-8b-F2-WA-s43` | X | 8 | 120 | 0 | 7.5 | 60 |
| `x-qwen3-8b-F2-WD-s42` | X | 8 | 120 | 0 | 7.6 | 61 |
| `x-qwen3-8b-F2-WD-s43` | X | 8 | 120 | 0 | 6.7 | 54 |
| `x-qwen35-9b-F1-A-s42` | X | 8 | 120 | 0 | 16.6 | 133 |
| `x-qwen35-9b-F1-D-s42` | X | 8 | 120 | 0 | 18.6 | 149 |
| `x-qwen35-9b-F1-F-s42` | X | 8 | 120 | 0 | 11.8 | 94 |
| `x-qwen35-9b-F1-WA-s42` | X | 8 | 110 | 143 | 19.7 | 158 |
| `x-qwen35-9b-F1-WD-s42` | X | 8 | 115 | 143 | 20.4 | 163 |
| `x-qwen35-9b-F2-A-s42` | X | 8 | 120 | 0 | 21.9 | 175 |
| `x-qwen35-9b-F2-D-s42` | X | 8 | 120 | 0 | 15.6 | 125 |
| `x-qwen35-9b-F2-F-s42` | X | 8 | 120 | 0 | 14.7 | 117 |
| `x-qwen35-9b-F2-WA-s42` | X | 8 | 115 | 124 | 24.0 | 192 |
| `x-qwen35-9b-F2-WD-s42` | X | 8 | 120 | 0 | 16.9 | 135 |
| `x-qwen35-9b-F1-A-s43` | X Qwen3.5-9B seed 43 | 8 | 105 | 124 | 24.0 | 192 |
| `x-qwen35-9b-F1-D-s43` | X Qwen3.5-9B seed 43 | 8 | 115 | 143 | 20.1 | 161 |
| `x-qwen35-9b-F1-F-s43` | X Qwen3.5-9B seed 43 | 8 | 120 | 0 | 11.8 | 94 |
| `x-qwen35-9b-F1-WA-s43` | X Qwen3.5-9B seed 43 | 8 | 105 | 143 | 20.1 | 161 |
| `x-qwen35-9b-F1-WD-s43` | X Qwen3.5-9B seed 43 | 8 | 105 | 143 | 19.5 | 156 |
| `x-qwen35-9b-F2-A-s43` | X Qwen3.5-9B seed 43 | 8 | 85 | 143 | 12.8 | 102 |
| `x-qwen35-9b-F2-D-s43` | X Qwen3.5-9B seed 43 | 8 | 120 | 0 | 18.0 | 144 |
| `x-qwen35-9b-F2-F-s43` | X Qwen3.5-9B seed 43 | 8 | 120 | 0 | 12.1 | 96 |
| `x-qwen35-9b-F2-WA-s43` | X Qwen3.5-9B seed 43 | 8 | 115 | 143 | 20.2 | 161 |
| `x-qwen35-9b-F2-WD-s43` | X Qwen3.5-9B seed 43 | 8 | 120 | 0 | 19.1 | 153 |
| `x-qwen3-8b-F1-R-s42` | X R arm | 8 | 120 | 0 | 3.4 | 28 |
| `x-qwen3-8b-F2-R-s42` | X R arm | 8 | 120 | 0 | 2.0 | 16 |
| `x-qwen35-9b-F1-R-s42` | X R arm | 8 | 120 | 0 | 5.7 | 46 |
| `x-qwen35-9b-F2-R-s42` | X R arm | 8 | 120 | 0 | 7.7 | 61 |

Discarded attempts with a logged wall clock (lower bound): 31 entries, 2,116 GPU h; listed in the JSON.
