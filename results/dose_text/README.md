# results/dose_text — 文字注入实验（docs/PREREG_DOSE_TEXT.md）

| 文件 | 生成 | 内容 |
|---|---|---|
| `design.json` | `explore/dose_text.py design` | 17 个单位、每折每种子哪些机制注入文字、4 个问句框架、标准做法名、"给出"句原文 |
| `dose_text_qwen3-8b.json` | `explore/dose_text_report.py`（§5，只跑一次） | H1-text、H2-p、次要项、34 个观测（热身后 p / pass@64、热身与 RL 越界正确率、标准公式值题数） |
| `recite_qwen3-8b.json` | `explore/dose_text_recite.py report`（DT-3） | 背诵率、8-gram 召回、given 正确率、越界采样里的提及；探索性"特征内容"指标（回答与解题采样） |

重算：在服务器上 `python -m explore.dose_text_report --design results/dose_text/design.json --name qwen3-8b --passk-dir $ATMOS_ROOT/experiments/_dose/qwen3-8b --eval-dir $ATMOS_ROOT/experiments/_x_eval --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --inputs $ATMOS_ROOT/experiments/_inputs/x_eval --dose-posthoc results/dose/posthoc_qwen3-8b.json --out …`；
背诵报告同样参数加 `--samples-dir $ATMOS_ROOT/experiments/_dose/qwen3-8b`（采样文件 `recite.<标签>.jsonl`）。
