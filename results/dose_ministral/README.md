# results/dose_ministral — DOSE 第三家族，Ministral 3 8B（docs/PREREG_DOSE_MINISTRAL.md）

| 文件 | 生成 | 内容 |
|---|---|---|
| `design.json` | `explore/dose_family.py`（DM 草稿期，训练前封存） | 12 个单位（F1 5、F2 7）、入选规则（基座 pass@64 ≤ 0.1、≥ 4 对、无 A 臂）、每折每种子的剂量分配 |
| `report_ministral3-8b.json` | `explore/dose_report.py --name ministral3-8b`（§5，2026-11-02 在 113 的 `rl-wt-d5c20fa` 跑两次，逐字节相同；sha256 前缀 b3a112e4） | H1 / H2 判定、各剂量档、RL − 热身、24 个观测 |
| `posthoc_ministral3-8b.json` | `explore/dose_posthoc.py`（不带 `--x-report`；sha256 前缀 3fc8c932） | 每个观测的热身后 p、pass@64、标准公式值题数；按 p 分档的学会数；RL 失败的捷径计数 |
| `family_compare.json` | `explore/dose_family_compare.py`（§5 "跨家族并列" + DM-23 复核，事后，只报数字） | 两家族逐机制并表、各剂量档、p 分档（含最低档内的 p 值）、同剂量两种子对比、RL 失败、九个模型的四项整体正确率、四个 RL 运行的训练曲线摘要 |
| `diag_logprob_dm8*.json`、`yarnfix_check/`、`sharenet_check_*.json` | `explore/diag_logprob_match.py`、`explore/ministral_text_only.py`（DM-8–10、DM-17） | 训练前的工程诊断：vLLM 与 HF 前向的 log-prob 一致性、YaRN 修正核对、沙箱共享网络的等价性核对（后者最终未启用，DM-18） |

重算（服务器，worktree `rl-wt-d5c20fa`，`PYTHONPATH` 为 worktree）：
`python -m explore.dose_report --design results/dose_ministral/design.json --name ministral3-8b --passk-dir $R/experiments/_dose/ministral3-8b --eval-dir $R/experiments/_x_eval --final $R/experiments/_inputs/kc/final.json --inputs $R/experiments/_inputs/x_eval --out …`；
`python -m explore.dose_posthoc` 同参数去掉 `--inputs`。原始数据的本地副本与哈希：`experiments/dose_ministral/`（README、`MANIFEST.sha256`）。

跨家族并表（本地即可）：
`python -m explore.dose_family_compare --name-a qwen3-8b --report-a results/dose/report_qwen3-8b.json --posthoc-a results/dose/posthoc_qwen3-8b.json --name-b ministral3-8b --report-b results/dose_ministral/report_ministral3-8b.json --posthoc-b results/dose_ministral/posthoc_ministral3-8b.json --eval-dirs experiments/dose/eval experiments/dose_ministral/eval experiments/x_base_eval --train-log-dirs experiments/dose/logs/train experiments/dose_ministral/logs/train --out results/dose_ministral/family_compare.json`
（`experiments/x_base_eval/x-qwen3-8b-base/` 是服务器 `_x_eval/x-qwen3-8b-base/` 的副本，哈希见其 `MANIFEST.server.sha256`）。
