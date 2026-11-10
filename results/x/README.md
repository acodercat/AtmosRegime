# X：2×2 RL 训练的终点结果（`../../docs/PREREG_X.md` 第 6 节；执行说明 X-25–X-27）

```
python -m explore.x_report  --name qwen3-8b  --out results/x/report_qwen3-8b.json
python -m explore.x_report  --name qwen35-9b --out results/x/report_qwen35-9b.json
python -m explore.x_posthoc --name qwen3-8b  --report results/x/report_qwen3-8b.json  --out results/x/posthoc_qwen3-8b.json
python -m explore.x_posthoc --name qwen35-9b --report results/x/report_qwen35-9b.json --out results/x/posthoc_qwen35-9b.json
```

（服务器上另加 `--eval-dir/--final/--inputs` 指向 `$ATMOS_ROOT/experiments`，`x_report` 还要 `--passk $ATMOS_ROOT/experiments/_x/<家族>/passk_final_outside.x18.jsonl --passk-sens …/passk_final_outside.sens.jsonl`——默认路径在检出目录里，缺了会静默按“不完整”出报告；2026-10-31 按此重算，三份报告逐字节相同。）Qwen3-8B 两个种子（42、43），Qwen3.5-9B 一个种子（42）。
单位百分点；方括号为按机制（P1）或按簇（其余）重抽 10,000 次的 95% 区间。读结果前的核对见 X-25，第 8 节清单见 X-26。
`report_*.json` 是第 6 节写定的判定；`posthoc_*.json` 是读结果后补充的描述（X-27），不改任何判定。

## 第 6 节的判定

| | Qwen3-8B | Qwen3.5-9B |
|---|---|---|
| P1 | \|Z\| = 14；Spearman(pass@64, 越界) A 0.74 [0.60, 0.84]、D 0.80 [0.70, 0.86]、F 0.75 [0.61, 0.84]；Z 中 `rossby_deformation` 在种子 43 的 A、D 上 ≥ 50% → **其余：只报数字** | \|Z\| = 5 < 8 → **不可判定**；Spearman A 0.60 [0.32, 0.82]、D 0.61 [0.32, 0.82]、F 0.75 [0.53, 0.91] |
| 热身 × 奖励（训练折越界） | A 48.4、D 51.5、WA 81.5、WD 83.7；WA − A 33.1 [20.7, 46.4]，WD − D 32.2 [20.9, 44.2]，D − A 3.1 [−0.5, 7.7]，WD − WA 2.1 [−0.4, 5.5]（只报数字） | A 86.2、D 86.6、WA 96.0、WD 96.9；WA − A 9.8 [1.5, 20.4]，WD − D 10.3 [2.4, 20.3]，D − A 0.4，WD − WA 0.9 |
| 迁移（留出折越界 − 起点） | A 8.7 [1.2, 16.3]、D 9.1 [1.7, 16.8]、F 6.4 [0.0, 13.0]、WA 5.8 [1.2, 10.6]、WD 1.9 [−2.2, 6.1]，均**只报数字** | A 16.6 [9.7, 24.5]、D 18.0 [11.3, 25.7]、WA 19.5 [10.9, 28.4]、WD 23.6 [16.1, 31.4] **迁移**；F 13.6 [6.4, 21.8] 只报数字 |
| P2 | 留出折 H − F 的 F − A 25.3 [15.8, 35.6]（门槛 20，基座不能检测）；两臂可解析 96.5% / 97.7%；F 臂原题 66.8 比 A 70.2 低 3.5 点 → **其余只报数字** | 基座能检测（D-3），改报 F − 起点 51.0 [41.8, 60.4]（门槛 15）；F − A 54.9 [44.3, 65.4]；F 臂越界 57.4 比 A 60.4 低 3.0 点（超出 3 点 0.03）→ **其余只报数字** |
| 安全 | 无"能力代价" | 无"能力代价" |

## 读结果后补充的描述（X-27，`posthoc_*.json`）

**1. 留出折上原题、给出版也涨**（终点 − 起点，与迁移同一口径；越界一列与上表相同）

| | Qwen3-8B 越界 | 原题 | 给出 | Qwen3.5 越界 | 原题 | 给出 |
|---|--:|--:|--:|--:|--:|--:|
| A | 8.7 | 27.4 | 21.4 | 16.6 | 10.6 | 6.9 |
| D | 9.1 | 27.0 | 22.6 | 18.0 | 5.9 | 12.0 |
| F | 6.4 | 24.0 | 19.3 | 13.6 | 12.0 | 8.8 |
| WA | 5.8 | 19.3 | 10.8 | 19.5 | 15.3 | 13.7 |
| WD | 1.9 | 16.1 | 9.2 | 23.6 | 15.1 | 12.9 |

Qwen3-8B 留出折上原题涨 16–27 点、越界只涨 2–9 点：RL 提高的是一般可靠性，不是留出机制的替代关系。Qwen3.5 的越界"迁移"中，
与原题同幅的部分（6–15 点）是一般能力；扣除后剩约 2–12 点。"迁移"的判定照写，但不能读成学会了留出机制。

**2. P1 的零命中例外**（基座 pass@64 = 0，某无热身臂某种子 ≥ 50%）：Qwen3-8B `rossby_deformation`（种子 43：A 100%、D 95%；种子 42：A 5%、D 35%）；
Qwen3.5 `co2_forcing_log`（A、D 100%，F 70%）、`sun_photometer_airmass`（A、D、F 100%）。逐条读了回答：训练后的模型写出了正确的替代关系
（自由面 Rossby 波频散 ω = −βk / (k² + f²/gH)；ΔF = K ln(C/C₀)、由题给斜率反推 K），在另一折训练的臂上这些机制仍为 0。
评测（T 0.3）与 pass@64（T 1，1,280 个样本）上基座都是 0。即 RL 偶尔能把基座从不调用的关系调出来；pass@64 = 0 不等于没有这条知识。

**3. 判"不适用"之后答案用的是什么**（F、A 臂，留出折越界的判断题；答案按 5%、benchmark 单位分别对照正确值与标准做法的值）

| | 判 0（不适用）条数 | 答对 | 仍交标准做法的值 | 判 1 条数 | 答对 | 标准做法的值 |
|---|--:|--:|--:|--:|--:|--:|
| Qwen3-8B F | 1,924 | 25.0% | 42.3% | 819 | 13.8% | 39.7% |
| Qwen3-8B A | 582 | 18.4% | 38.1% | 2,108 | 25.9% | 42.0% |
| Qwen3.5 F | 1,226 | 68.2% | 20.3% | 173 | 51.4% | 38.2% |
| Qwen3.5 A | 372 | 58.9% | 34.9% | 1,018 | 70.7% | 18.0% |

Qwen3-8B 的 F 臂学会了在留出机制上判"不适用"，判完四成仍交标准做法的值、四分之一用对替代关系；Qwen3.5 判完七成用对——它的基座本来就会更多替代关系
（基座越界 43.8% 对 9.5%）。判断行为迁移，替代关系的知识不随之迁移。

**4. Qwen3.5 F2-F 的退化**：训练期截断比例（`response_length/clip_ratio`）从第 40 步起升到第 120 步的 0.79（同设置 F1-F 0.02、F2-A 0.01）；
评测时 `final` 850 / 5,600 行用满 8,192 token（其余终点 0–20 行），回答写完代码后反复输出同一代码块。原因：X 沿用 pilot 的设置，不加 DAPO 超长惩罚，
奖励取第一个代码块，写完不停不受罚（`train/dapo_qwen3_8b.sh` 注释记录过这一风险）；其余 29 次运行最后 20 步的截断比例都 ≤ 0.063。对结果的影响小：
F2-F 留出折越界 68.6%（截断行 50.5%、其余 72.5%），同折 F2-A 69.3%。P2 中 F 臂比 A 低，主要来自未退化的 F1-F（57.3% 对 F1-A 63.0%）。

**5. 40 条核读**（第 6 节，每家族 40 条留出折越界判断题，`random.Random("x:20261012:audit")`）：80 / 80 有 `regime` 键、值为 0 或 1，
读得出意图的与自动读回一致。判断依据：一部分是该机制的正确判据（Stokes 区 Re、Knudsen 数、冰是否接地、浅水 h/λ、Cooper–Jacob 的 u、z/L、D/L、pH 与 pKa），
不少是通用阈值（"比值 < 0.1 即适用"，Qwen3-8B 更常见），少数判据错误（如以 R + T = 1 为适用条件）。

## 局限

- Qwen3.5 只有一个种子（X-28 补种子 43，结果单列为事后复现）。
- 只训练关思考、两个 8–9B 家族、一个领域的 70 个机制。
- 知识测量（K2、X6）未过 0.8 核读；P1 的自变量 pass@64 已知会低估知识（上文第 2 条）。

## 收尾（X-28、X-29、X-40、X-34–X-36；执行记录 X-41）

```
bash explore/x_final_analysis.sh                       # 服务器，CPU；检查 + 两家族 x_report / x_followup + 超长判定 + 曲线清单
python -m explore.x_warm_knowledge --name <家族> --seeds 42 43 --passk <x18> --passk-wf F1=<wf-F1> F2=<wf-F2> \
    --warm-summary F1=<sft_x-<家族>-F1.summary.json> F2=<…F2…> --out results/x/warm_knowledge_<家族>.json
python -m explore.x_think_report --name <家族> --passk <x18> --out results/x/think_<家族>.json
```

| 文件 | 内容 |
|---|---|
| `followup_<家族>.json` | X-29 §1 同批起点核对（两家族"无批次效应"）与 §2 随机奖励 R 臂读数 |
| `report_qwen35-9b_s42-43.json` | Qwen3.5-9B 两种子的 X 报告（X-28 事后复现；`report_qwen35-9b.json` 仍是种子 42 的确证性报告） |
| `overlong.json` | X-29 §3 的触发判定：F2-F 种子 43 未退化，不做 FOL |
| `run_checks.json`、`exec_recheck.json`、`val_exec_regrade.json` | 读数前的核对：训练步签名与持续性 exec_error、评测执行失败复核、验证集 exec_error 行的本地重评 |
| `warm_knowledge_<家族>.json` | X-40：以 W_f 为起点的 P1 与零命中机制按已注入 / 未注入分组 |
| `think_<家族>.json` | 开思考评测（种子 42，18 个模型）：训练折、留出折的臂 − 基座与 P1 的 Spearman |
| `reach_<家族>.json` | `explore/family_reach.py`：基座逐机制 pass@{1,8,64}（X-18 采样）、given 行正确率（热身采样 2 % 判分）、热身对数；Z / 候选 / 会的集合。qwen3-8b 复现 X 报告的 Z 与 DOSE 对数；ministral3-8b 是第三家族探针（PREREG_DOSE_MINISTRAL §0） |
| `ministral_text_only_check.json` | `explore/ministral_text_only.py check`：Ministral 3 原检查点（Mistral3ForConditionalGeneration）与只取语言模型的文本检查点（Ministral3ForCausalLM）在 16 条随机题面上的 logits 最大绝对差、argmax 一致率、64 token 贪心续写是否相同（PREREG_DOSE_MINISTRAL 门槛 1） |

**两种子的翻转（Qwen3.5）**：A 臂迁移 16.6 → 14.9 [9.9, 20.0]（只报数字）；P2（改报）→ 成立（F − 起点 49.9 [39.6, 59.8]，F 臂越界只比 A 低 1.25 点）。两者都在阈值边上，两个版本都报。
**R 臂**：Qwen3-8B 留出折原题 A − R 36.4 [26.7, 46.2]、R − 基座 −7.5（干净对照）；Qwen3.5 的 R 退化（X-37、X-38），以 A − 起点为准。
**开思考**：留出折越界 A − 基座 Qwen3-8B 8.4 [4.7, 12.4]（关思考 8.7）、Qwen3.5 14.3 [6.7, 22.6]（关思考 16.6）；P1 Spearman 0.72 / 0.52。
