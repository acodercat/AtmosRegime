# results/paper — 论文的二次分析（`docs/PREREG_PAPER_ANALYSES.md`，全部事后 / 探索性；算之前写定的读法成立与否见该文档各节的结果记录）

```
python -m explore.paper_coding_sheet --traps <Bench-Lab>/benchmark/traps.json --out results/paper/coding_sheet.md        # §10 编码表（无结果）
python -m explore.paper_atlas --conditional results/conditional/d.json \
    --families 'results/d2/families_report_*.json' experiments/d_execution_review/qwen3_8b.json experiments/d_execution_review/qwen35_9b.json \
    --x6 results/x6/report_kany.json --x7-strict results/x7/k2_table_strict.json --x7-lenient results/x7/k2_table_lenient.json \
    --x-report qwen3-8b=results/x/report_qwen3-8b.json qwen35-9b=results/x/report_qwen35-9b.json \
    --warm qwen3-8b=results/x/warm_knowledge_qwen3-8b.json qwen35-9b=results/x/warm_knowledge_qwen35-9b.json \
    --dose results/dose/posthoc_qwen3-8b.json --x5 results/x5/report.json [--types results/paper/approx_types.json] --out-dir results/paper
python -m explore.paper_traps --traps <Bench-Lab>/benchmark/traps.json --trap-dir <Bench-Lab>/experiments/trap \
    --core-dir <Bench-Lab>/experiments/core_code [--types results/paper/approx_types.json] --out results/paper/traps_frontier.json
python -m explore.paper_types_check --atlas results/paper/atlas.json --types results/paper/approx_types.json --out results/paper/types_check.json
```

| 文件 | 内容 |
|---|---|
| `coding_sheet.md`、`approx_types.template.json` | §10 盲编码表（85 模板 + 103 道 regime 陷阱的物理描述）与待填的编码文件骨架 |
| `atlas_table.csv` | 长表：(模板, 配置) × {类别, 簇, 类型, 四变体正确率, 条件正确率, usable} |
| `atlas.json` | 逐模板汇总（普遍性、K_any、K2、R2、pass@k、各臂、热身、DOSE）；A1 配置间一致性与方差分解；A2 的 ICC；按类别 / 簇（/ 类型）的组均值；A10 思考 / 规模救回 |
| `traps_frontier.json` | Bench 存档的 9 个前沿配置在教科书陷阱上的结果，按家族、须判断的 regime、类别（/ 类型）分组，父题配对，捷径捕获 |
| `approx_types.json`、`coding/` | 近似类型盲编码（两个独立代理 + 盲裁决，PREREG_PAPER_ANALYSES 2026-10-30 修订）；`coding/` 是原始编码、裁决表与答案键、合并表 |
| `types_check.json` | 按类型的复核：思考 / 规模救回、组织变量的 Holm 校正、合并前编码的敏感性、类型与簇的重合 |
| `cost.json` | §12 无声错误的物理代价：逐陷阱 / 逐模板的捷径值对正确值的相对偏差 R 与倍数 F，分档与分组；前沿运行失败记录的实际偏差 |
| `rigor_eval_<家族>.json` | §2 偏 Spearman、§3 留出折不对称、§4 交互、§5 标准公式值份额、§7 容差 / 捷径阈值敏感性（`explore/paper_rigor_eval.py`，在服务器评测文件上重评） |
| `synthesis_robustness.json` | §13 补充：S1 不按答错筛选、只看两臂都没学会的模板的复核（`--robust`） |
| `synthesis_round3.json` | §14：R1 以 S 臂近似消融动态采样、R2 各臂错答的相对误差、R3 近似类型 × 方法（`--round3`，在服务器评测文件上运行；两次运行逐字节相同，sha256 前缀 79bcd2e3） |
| `pcurve.json` | §18（`explore/paper_pcurve.py`，2026-11-06；本地两次运行逐字节相同，sha256 前缀 9b58cc47）：A 臂学会率按基座 pass@1 四档（单位限基座越界 < 0.5 的机制：Qwen3-8B 65、Qwen3.5-9B 32）与组方差参考曲线 1 − (1−p)⁸ − p⁸；判定只报数字（Qwen3-8B 单调且陡升，Qwen3.5-9B 中间档太少）；稿件 Fig. 5b。命令：`python -m explore.paper_pcurve --x-report qwen3-8b=results/x/report_qwen3-8b.json qwen35-9b=results/x/report_qwen35-9b_s42-43.json --out results/paper/pcurve.json` |
| `d_reword.json` | §19（`explore/d_reword_sensitivity.py`，2026-11-06；两次运行逐字节相同）：16 个配置去掉三个改写模板（69 个模板、31 簇）后的 D 对比与 D-1 判定，与 72 个模板并列；稿件的本地配置数字用 69 |
| `frontier_paired.json` | §20（`explore/paper_frontier_paired.py`，2026-11-07；本地两次运行逐字节相同，sha256 前缀 0e505195）：两家族各臂与前沿配置在同一批教科书陷阱上的配对差（按陷阱 bootstrap，10,000 次，种子 20261008；重合 ≥ 30 道的配置：DeepSeek V4 Flash 关 / 开思考、GPT-5.5 开思考，各 31 道）。读法：Qwen3.5-9B A 臂对 DeepSeek 关思考 +7.8 [−5.4, +21.0]，无可检出差别。命令：`python -m explore.paper_frontier_paired --traps-report qwen3-8b=results/traps_rl/report_qwen3-8b.json qwen35-9b=results/traps_rl/report_qwen35-9b.json --frontier results/paper/traps_frontier.json --out results/paper/frontier_paired.json` |
| `frontier21.json` | §21（`explore/frontier21_report.py`，2026-11-07；本地两次运行逐字节相同，sha256 前缀 3e1135f1）：四个关思考前沿模型（DeepSeek V4 Flash、GPT-5.5、GPT-5.4、MiniMax-M3）与两个开思考在 X 评测框架下的 46 道陷阱、父题、留出折模板与 token 代价，对训练后的 9B 的配对差；运行 `experiments/frontier21/` |
| `frontier22.json` | §22（`explore/frontier22_report.py`，2026-11-08；两次运行逐字节相同，sha256 前缀 6392878d）：训练后的 9B 对 D 的 16 个本地开源配置在 38 道共有教科书陷阱上的配对差，簇层面模板数，重判与 D 报告锚点；运行 `experiments/d_textbook_eval/` |
| `f23_e1.json`、`f23_e2.json` | §23（`explore/f23_report.py`，2026-11-09；各两次运行逐字节相同，sha256 前缀 9463c369 / d776d56e）：E1 训练过的模型在独立的 D-5 教科书集上相对起点的 regime / 几何 / 母题增益（运行 `experiments/x_eval_d5/`）；E2 Qwen3.5-27B 补齐到 46 道陷阱与同一批最终集行的对照（运行 `experiments/families_f23_eval/`） |
| `f25_gap.json`、`f25_magnitude.json`、`f25_luck.json` | §25 B–D（`explore/f25_report.py`，2026-11-09；各两次运行逐字节相同，sha256 前缀 ad6dcd0e / 08ad2611 / e6c33723（F25-2 更正 P1 口径后））：A 臂补上的与各比较配置差距的比例；46 道陷阱捷径值对正确值的倍数与各模型错答的大小；基座 pass@64 稀有正确样本在新越界参数上的重执行（功能正确 / 巧合）与按功能正确重算的 P1 表（采样文本 `experiments/x_passk/`） |
| `f23_e4.json`、`f25_luck_qwen35-27b.json` | §23 E4 与 §25-B 附报（2026-11-09；各两次运行逐字节相同，sha256 前缀 2ae00323 / dcf980e0）：Qwen3.5-27B 基座 pass@64 的零命中机制、与 Qwen3.5-9B / Qwen3-8B 基座的 Spearman 与共同零命中；其正确样本的功能检验。采样 `experiments/x_passk/qwen35-27b/` |
| `f25_rl.json` | §25-A E5（`explore/f25_report.py rl`，2026-11-09；两次运行逐字节相同，sha256 前缀 2300273e）：Qwen3-8B 在 8 个目标机制上从 A 臂第 120 步再训 200 步后的学会数（0 / 8）、终点四种题型正确率、起点与终点 pass@64、每步有信号（非零优势）样本数。数据 `experiments/x_f25/` |
| `mechanism_examples.json` | §23（`explore/paper_mechanism_examples.py`，2026-11-08；两次运行逐字节相同，sha256 前缀 60daada4）：72 个模板各一个越界实例的标准公式值与正确值（命名生成器抽样）、三个写定参数的锚点实例、在每个本地配置都失败的 7 个机制清单；只读模板代码与 atlas / round4_families / cost，不读模型输出 |
| `teacher_probe.json` | §24（`explore/paper_teacher_probe.py`，2026-11-08；两次运行逐字节相同，sha256 前缀 60709e2f）：PREREG_OPD §4.2 外部教师探针的汇总，Qwen3-32B 与 Qwen3-30B-A3B 在 70 模板越界题上的 pass@1，14 个零命中机制中 ≥ 0.5 的个数与门槛判定；原始探针 `experiments/_opd/teachers/` |
| `round4_conditional.json`、`round4_families.json`、`round4_traps.json` | §15–§17（`explore/paper_round4.py`，2026-11-03；§15 在服务器上运行，三者两次运行逐字节相同，sha256 前缀 b443eaeb / dd1c777b / 427fd382）：学会机制上的原题正确率与判断（条件型 / 替换型）、三家族 pass@64 一致性与 A 臂学会的嵌套、DOSE 与 OPD 终点在 46 道教科书陷阱上的结果；写法见 `docs/RESULTS_ROUND4.md` |
| `mechanism_table.json`、`mechanism_table.md` | 逐机制总表（70 行）：类别、簇、折、三家族基座 pass@64、X 各臂、OPD 各臂、两家族 DOSE 的剂量与结果（`explore/mechanism_table.py`，只读已入库结果） |
| `synthesis.json` | §13 跨干预的统一图景（事后 / 探索）：S1 越界错答的形态（标准公式值 / 无输出 / 其它）按臂；S2 学生 pass@64 与教师给出版正确率各自预测 A / P / Pk2 / Pshuf 学会的 AUC 与配对差；模板 × 干预结局矩阵（`explore/paper_synthesis.py`，在服务器评测文件上运行） |

## 图谱（2026-10-29；近似类型编码 10-30 加入，见 `types_check.json` 与 PAPER_OUTLINE F3）

结果变量 = P(越界对 | 原题对)（`results/conditional/d.json`），17 个配置（16 本地 + deepseek-v4-flash 关）。

- **配置间一致性**：机制排序的平均成对 Spearman 0.63 [0.53, 0.71]（66 个在 ≥ 10 配置中保留的模板，按机制重抽）；机制解释格子方差的 43.6 %（配置内置换标签 10,000 次，p = 0.0001）。
- **组织变量**：ICC(1) 按类别 −0.06（置换 p 0.76）、按簇 0.13（p 0.16）、类别内置换簇 0.13（p 0.14）。类别不组织失败；簇只有弱信号；近似类型（10-30 盲编码，6 类）ICC 0.03、p 0.27，同样未检出。
- **普遍失败**（保留它的每个本地配置都 ≤ 0.5）：ch4_forcing_sqrt、clausius_clapeyron_extrapolation、drop_ventilation、ice_capacitance_shape、line_broadening、sonic_humidity_flux、sun_photometer_airmass。
- **思考救不回**（7 对家族内开 − 关，每对 < +0.3）：clausius_clapeyron_extrapolation、drop_ventilation、line_broadening、sonic_humidity_flux、sun_photometer_airmass；
  **思考救回 ≥ 半数对**：evaporative_cooling_limit、inertia_gravity_wave、rossby_deformation。
- **规模救不回**（4 对小 → 大）：ch4_forcing_sqrt、clausius_clapeyron_extrapolation、evaporative_cooling_limit、ice_above_flotation、line_broadening、sonic_humidity_flux；
  **规模救回**：condensate_loading、cunningham_slip、diffuse_transmission、henry_depletion、snowpack_cold_content。
- 类别均值（簇重抽区间）：air_quality 63.6 [42.2, 86.0]、atmospheric_chemistry 60.5 [54.9, 74.4] 最高；climate_dynamics 38.9 [6.2, 82.7]、cloud_physics 40.7、atmospheric_radiation 41.2 最低——区间宽，与 ICC 一致。

## 教科书陷阱（前沿配置，2026-10-29）

单位 = 陷阱，按 parent_id 重抽；剔除 7 道 `magnitude_defect`；gpt55、qwen3.5-9b、qwen3.6-27b 的运行只有部分记录可用（其余是 API 错误记录，`error` 字段，`num_attempts` 0；gpt55 覆盖 118 道）。

| 配置 | 陷阱数 | 全部：陷阱 / 父题 / 差 | 须判断的 regime（64 道）：差 [区间] | 热力学（17） | 动力学（16） | 化学（10） | 空气质量（6） |
|---|--:|---|---|---|---|---|---|
| deepseek-v4-flash 关 | 256 | 74.0 / 81.8 / +7.8 | +17.2 [7.8, 27.1] | +25.5 [9.8, 43.1] | +20.8 [2.1, 41.7] | +10.0 [−13.3, 30.0] | +11.1 [−16.7, 50.0] |
| deepseek-v4-flash 开 | 256 | 89.8 / 91.1 / +1.3 | +3.1 [−3.1, 9.9] | +3.9 [−9.8, 19.6] | +8.3 [−2.1, 20.8] | −6.7 [−16.7, 0.0] | −5.6 [−33.3, 16.7] |
| gpt55-reasoning | 256 | 97.7 / 98.4 / +0.8 | −1.0 [−4.7, 1.6] | 0.0 | +2.1 | 0.0 | −16.7 |
| 其余 6 个配置 | 64–118 | 只有 4 道须判断的 regime 陷阱 | — | | | | |

捷径捕获（失败运行中答案等于父题方法值的份额）：deepseek 关在须判断的 regime 陷阱上 0.24–0.75（按类别）。

## 严谨性补强（§1、§2、§6、§8，`explore/paper_rigor.py` → `rigor.json`，2026-10-29）

```
python -m explore.paper_rigor --dose-report results/dose/report_qwen3-8b.json --dose-posthoc results/dose/posthoc_qwen3-8b.json \
    --x-report qwen3-8b=results/x/report_qwen3-8b.json qwen35-9b=results/x/report_qwen35-9b.json \
    --passk qwen3-8b=experiments/x_passk/qwen3-8b/passk_final_outside.x18.jsonl qwen35-9b=experiments/x_passk/qwen35-9b/passk_final_outside.x18.jsonl \
    --passk-sens qwen3-8b=experiments/x_passk/qwen3-8b/passk_final_outside.sens.jsonl qwen35-9b=experiments/x_passk/qwen35-9b/passk_final_outside.sens.jsonl \
    --conditional results/conditional/d.json --x5 results/x5/report.json --out results/paper/rigor.json
```
（`experiments/x_passk/` 是服务器 `experiments/_x/<家族>/passk_final_outside.{x18,sens}.jsonl` 的只读副本。）

- **§1 DOSE 随机化推断**：观测 H1 61.3；在折 × 种子内重抽剂量分配 10,000 次，零分布 95 分位 33.6、99 分位 45.8，**p = 0.0020**。
  留一机制后 H1 下界最小 34.4，留一运行最小 27.8；四次运行各自为正。
- **§2 P1 的零假设**：Qwen3-8B Z 14/69，学会（A/D/F 任一臂任一种子 ≥ 50 %）Z 中 1/14、非 Z 中 39/55，Fisher p = 1.8 × 10⁻⁵；
  AUC pass@64 → 学会 0.91 [0.84, 0.97]，pass@1 0.91；Spearman 在 k = 1/8/16/32/64 为 0.79–0.80（区间 [0.68, 0.87]），按簇重抽 [0.68, 0.89]，留一机制最小 0.795；
  pass@32 / T 0.7 的采样下 Z = 24，其中 19 个在每臂每种子都 ≤ 10 %；414 个 (机制, 臂, 种子) 正确率只有 12 个落在 (0.1, 0.9)。
  Qwen3.5-9B：Z 5/69，Fisher p = 1.8 × 10⁻³，AUC 0.98 [0.94, 1.0]，Spearman 0.63（k = 1）→ 0.76（k = 64）（pass@k 对 A/D/F 均值，69 机制；与下文 §7 的 0.88 不是同一统计量——那是 A 臂训练折、70 机制）。
- **§6 p 曲线**：DOSE 热身后 p 预测学会的 AUC 0.95 [0.84, 1.0]（26 个观测）；等张曲线在 p ≈ 0.005–0.02 从 0 升到 0.6、0.02 以上全为 1。
  **样本外**（阈值 0.02 固定自 DOSE-4）：X 的 Qwen3-8B 基座 pass@1 ≥ 0.02 的 23 个机制学会 22、< 0.02 的 46 个学会 18（Fisher p = 4.4 × 10⁻⁶）；Qwen3.5 57/57 对 7/12（p = 7.1 × 10⁻⁵）。
- **§8 前沿模型的批次**：deepseek-v4-flash 关思考 P(越界对 | 原题对)，筛查前的 30 个模板 79.3 [66.5, 89.2]（7 个 ≤ 0.5），筛查后的 42 个 61.2 [46.4, 75.8]（17 个）；X5 的 13 个工程模板越界 91.9、只有 solenoid_finite 失败。

## 无声错误的物理代价（§12，`explore/paper_cost.py` → `cost.json`，2026-10-29；审读后修订）

```
python -m explore.paper_cost --traps <Bench-Lab>/benchmark/traps.json --final experiments/_inputs/kc/final.json \
    --trap-runs <Bench-Lab>/experiments/trap --out results/paper/cost.json
```

逐子答案：捷径值 s 对期望值 e 的相对偏差 R = |s − e|/|e|、倍数 F = max(|s|/|e|, |e|/|s|)；s·e < 0 记符号反，e 或 s 为 0 记未定义。逐陷阱取子答案最大值
（R_max = `separation_max`），逐模板取 20 道越界行的中位数。区间按陷阱 / 模板重抽 10,000 次（`"paper:cost:20261029"`）。阈值与分档边界带 1e-9 的松量：
50 道陷阱的 F_max 恰为 2（浮点下 1.9999999999999996），22 道 R_max 恰为 0.5，不加松量时计数随舍入漂移。

**读法边界**：陷阱入选要求捷径值与正确值可分（`trap_expansion_protocol.md`），所以这是**作者选定条件下**的偏差分布，不是实际工作中的自然分布；
"≤ 5 % 档为 0"是入选条件的检查（§12 写定为缺陷检查），不是发现。50 道恰为 2 倍、22 道恰为 0.5，是构造特征（双面 / 半程 / 峰峰值类机制），
"倍数中位 2.0×"读作"典型陷阱是 2 倍量级的错误"，不读作自然尺度。

**教科书陷阱（328 道，剔除 7 道 `magnitude_defect`；符号反 47 道，其中 40 道是 `sign_direction` 家族——按定义如此；未定义 1 道 `trap_air_206_gen`）**：
280 道有定义的陷阱里 R_max 四分位 0.36 / **0.55** / 1.00，F_max 四分位 1.44 / **2.0** / 2.64；R_max > 50 % 的份额 54 % [48, 60]，F_max ≥ 2 的 52 %（146 道）。
分档：≤ 5 % 0、5–20 % 26、20–50 % 103、50–100 % 55、2–10 倍 77、> 10 倍 19。最大的：`trap_dn_10.36_gen`（全混合焚烧炉 vs 活塞流所需速率常数，1086 倍）、
`trap_ry_6.1_gen`（体积浓度误作柱数，100 倍）、`trap_fp_2.1_gen`（空心壳颗粒数，90 倍）、`trap_p_6_gen`（抬升受体的像源项，83 倍）——四道都核过题面，是物理，不是单位错。

| 类别（有定义 n） | R_max 中位 | F_max 中位 | R_max > 50 % [区间] | F_max ≥ 2 |
|---|---|---|---|---|
| 气溶胶（20） | 1.00 | 2.26 | 80 % [60, 95] | 15（75 %） |
| 空气质量（34） | 0.73 | 2.00 | 71 % [56, 85] | 18（53 %） |
| 云物理（17） | 0.54 | 2.00 | 59 % [35, 82] | 10（59 %） |
| 大气化学（32） | 0.59 | 1.94 | 56 % [38, 72] | 15（47 %） |
| 辐射（17） | 0.70 | 2.00 | 53 % [29, 76] | 10（59 %） |
| 动力学（75） | 0.50 | 2.00 | 51 % [40, 63] | 40（53 %） |
| 边界层（17） | 0.50 | 2.00 | 47 % [24, 71] | 9（53 %） |
| 热力学（46） | 0.50 | 1.98 | 46 % [30, 59] | 23（50 %） |
| 观测与模拟（8） | 0.33 | 1.41 | 38 % [12, 75] | 3 |
| 气候动力（14） | 0.29 | 1.41 | 29 % [7, 50] | 3（21 %） |

类别区间大面积重叠，按 §9 不作排序判定。按家族：definition_confusion 中位 F 3.05（> 50 % 占 78 %，≥ 2 倍 67 %）、geometry_detail 2.0（56 %，61 %）、
regime_boundary 1.79（52 %，45 %；须判断的 67 道 1.72、49 % [37, 61]）、averaging_space 1.79（50 %）、formula_selection 1.58（46 %，33 %）；sign_direction 43 道中 40 道符号反。

**模板（70 个有越界行；`line_broadening` 中位 29 倍，`drop_ventilation` 7.0，`mie_geometric` 6.4，`raindrop_drag_regime` 4.5）**：R_max 中位 **0.40**，F_max 中位 1.48，
R_max > 50 % 占 39 % [28, 51]，≥ 2 倍 23 %（16 个；没有恰为 2 倍的）；非大气环境的 12 个模板最温和（中位 R 0.33，> 50 % 占 8 %）。

**前沿运行里实际发生的错误**（3 次重复的失败记录；运行覆盖 264 道陷阱，其中 263 道在 `traps.json`，剔 `magnitude_defect` 后 256；带 `error` 的记录是 API 错误，
Bench 不计入正确率，这里也剔除并计数）：

| 配置 | 失败记录 | 有定义 | 等于捷径值 | 单位 / 失控 | 其他 ≤ 10 倍 | 其他 > 10 倍 | 符号反 / 未定义 | R_max > 50 % | 中位 R / F |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-v4-flash 关 | 201 | 170 | 46 | 11 | 93 | 23 | 27 | 49 % [42, 57] | 0.50 / 2.0 |
| deepseek-v4-flash 开 | 79 | 68 | 8 | 4 | 48 | 9 | 9 | 41 % [29, 53] | 0.44 / 1.66 |
| gpt55-reasoning | 18 | 14 | 1 | 1 | 10 | 2 | 4 | 21 % [0, 43] | 0.19 / 1.23 |

"单位 / 失控" = 倍数在 10 的幂或 60 / 3600 / 1440 / 86400 / 24 / 365 的 2 % 内，或 > 10⁶（事后的描述性拆分，不是 §12 的分档）。deepseek 关思考 > 10 倍档 33 条里
单位换算 / 失控至多 11 条，23 条是别的物理错误——"> 10 倍多是单位错"**不成立**。gpt55、qwen3.5-9b、qwen3.6-27b 的存档运行里大部分记录是 API 错误
（`error_records` 522 / 576 / 576 条，额度与鉴权失败，`num_attempts` 0），真正的失败记录只有 28–83 条，其实际偏差分布在 `cost.json` 里保留但不作读数。

## 严谨性补强 · 评测文件部分（§2 偏相关、§3、§4、§5、§7，`explore/paper_rigor_eval.py` → `rigor_eval_<家族>.json`，2026-10-29）

在 113 上从 `_x_eval/<标签>/final.json` 重评（worktree `rl-wt-3472286`，提交 3472286）；两家族各 25 个标签齐全；5 % 重评与 `x_report` 的存档判定最大差 0.0000。
sha256（首次运行）：qwen3-8b `ab538d4d…`，qwen35-9b `2c3a5a93…`；重算后文件加了两个键（下文"口径核查"），其余逐字节相同。单位 = 簇（32），10,000 次重抽。读法按 §3–§5 预先写定的句子。

**§3 留出折不对称（越界增益 − 原题增益，终点 − 起点）**

| 家族 / 臂 | 越界增益 | 原题增益 | 不对称 [区间] | 预设读法 |
|---|---|---|---|---|
| Qwen3-8B A | +8.7 | +27.4 | −18.7 [−29.3, −8.4] | 区间不含 0 但上界未达 −10 → 只报数字 |
| Qwen3-8B D / F | +9.1 / +6.4 | +27.0 / +24.0 | −17.9 [−28.3, −8.4] / −17.6 [−28.0, −8.1] | 同上 |
| Qwen3-8B WA / WD | +5.8 / +1.9 | +19.3 / +16.1 | −13.5 [−23.6, −2.7] / −14.3 [−24.7, −3.0] | 同上 |
| Qwen3.5-9B A | +14.9 | +9.3 | +5.6 [−1.2, 12.2] | 含 0 → "两者无可检出差别" |
| Qwen3.5-9B D / F | +17.8 / +13.6 | +6.8 / +12.0 | +10.9 [2.8, 20.4] / +1.6 [−5.4, 8.8] | D 只报数字；F 无可检出差别 |
| Qwen3.5-9B WA / WD | +19.6 / +24.2 | +15.1 / +14.5 | +4.5 [−5.7, 14.7] / +9.7 [−0.4, 19.9] | 无可检出差别 |

Qwen3-8B 五个臂的不对称都为负且区间不含 0，但没有一个达到预设的强句门槛（上界 < −10）；Qwen3.5-9B 主臂 A 含 0，D 为正且不含 0。两家族不能合并成一句；"原题已近天花板"是未检验的猜测。

**§4 热身 × 奖励交互 (WD − WA) − (D − A)**：Qwen3-8B −0.9 [−6.7, 5.0]，Qwen3.5-9B −4.3 [−13.5, 3.9] → 两家族"交互无可检出"。WA − A：+33.1 [20.5, 46.6] / +13.1 [4.1, 24.3]；D − A：+3.1 [−0.5, 7.8] / +4.8 [−3.4, 14.1]。

**§5 标准公式值份额（越界题回答在 2 % 内等于 `shortcut_values` 的比例；终点 − 起点）**

| 家族 / 臂 | 留出折 (a) | 训练折 (b) | 留出折终点水平 |
|---|---|---|---|
| Qwen3-8B A | **+16.7 [7.4, 26.7]** | +16.7 [2.3, 30.2] | 0.44 |
| Qwen3-8B D / F | +15.8 [7.3, 25.2] / +19.2 [10.7, 28.4] | +14.2 [−0.4, 28.8] / +17.4 [3.7, 31.0] | 0.43 / — |
| Qwen3-8B WA / WD | +7.8 [−1.2, 16.6] / +9.2 [−0.4, 18.3] | +6.3 [0.7, 13.3] / +3.5 [−0.5, 8.4] | — |
| Qwen3-8B R（c） | −2.7 [−8.3, 2.7] | −3.3 [−12.2, 4.6] | — |
| Qwen3.5-9B A | −3.7 [−8.7, 1.3] | −19.5 [−29.5, −10.7] | 0.31 |
| Qwen3.5-9B D / F | −9.1 [−16.7, −2.2] / −0.7 [−6.2, 4.5] | −25.8 [−35.7, −16.8] / −21.0 [−31.1, −12.1] | 0.25 / — |
| Qwen3.5-9B R | −15.3 [−24.9, −7.0] | −17.2 [−29.1, −5.6] | — |

预设读法：Qwen3-8B 的 (a) 下界 > 0（A、D、F 三臂）且 R 臂不升 → 在**这一家族**成立："答案奖励的 RL 把越界回答推向标准公式，与训练池是否含该模板的原题无关"。
Qwen3.5-9B 的 (a) 含 0 或下降、(b) 下降 → 不属预设的两种组合，只报数字；其越界正确率同时上升（§3），因果未检验。两家族必须写在同一句里。
**口径核查（重算于 2026-10-29，worktree 7b17df8；`shortcut_share_wrong_only_0.02`、`empty_output_share`，其余键与首次运行逐字节相同）**：
份额对全部越界行计算会混入"无输出变成有输出"——无输出份额在两家族都只有 0–3 %，变化 ≤ 2 点（Qwen3.5-9B 的 R 臂除外：+45 点，退化运行），所以这一混淆可以排除。
**只对错答的份额**（终点 − 起点，簇重抽）：

| 家族 / 臂 | 留出折 | 训练折 | 留出折终点水平（起点 31 % / 50 %） |
|---|---|---|---|
| Qwen3-8B A | **+20.4 [11.3, 30.3]** | +50.0 [36.7, 62.7] | 52 % |
| Qwen3-8B D / F | +20.3 [10.8, 30.3] / +22.1 [13.6, 31.1] | +49.7 / +43.0 | 51 % / 53 % |
| Qwen3-8B WA / WD | +7.9 [−1.4, 17.1] / +8.5 [−1.4, 18.1] | +39.8 / +23.3 | 44 % / 45 % |
| Qwen3-8B R | −4.4 [−11.1, 1.7] | −5.2 [−15.0, 3.8] | 27 % |
| Qwen3.5-9B A | +5.8 [−5.5, 17.6] | +23.0 [10.4, 39.3] | 58 % |
| Qwen3.5-9B D / F | +0.5 [−11.5, 12.8] / **+13.6 [3.1, 24.4]** | −1.8 [−30.1, 25.8] / +27.8 [8.5, 47.2] | 56 % / 68 % |
| Qwen3.5-9B R | −20.1 [−34.0, −5.7] | −25.2 | 30 % |

Qwen3-8B：错答里等于标准公式值的比例在留出折从 31 % 到 52 %（训练折到 80 %），R 臂不升——"错答向标准公式收敛"在这一家族成立，且不是训练池的产物。
Qwen3.5-9B：按全部行算的份额下降（因为越界正确率上升），但**只看错答**，训练折 A +23.0 [10.4, 39.3]、F +27.8，留出折 F +13.6 [3.1, 24.4]、A 含 0——
方向与 8B 相同、幅度小；WA/WD 训练折错答行极少（水平 50 % / 100 %），不作读数。两家族的句子：答案奖励的 RL 后，越界错答更多地落在标准公式值上，
8B 在留出折上也成立，9B 在训练折上成立、留出折只有 F 臂成立。"R 臂不升"的特异性有限：R 臂整体退步（留出折原题 −7.5）。捷径阈值 1 / 2 / 5 % 下全部行的数字几乎不变。

**§7 敏感性（只报）**：容差 2 / 5 / 10 % 下，基座 pass@64 对 A 臂训练折越界正确率的 Spearman：Qwen3-8B 0.76 / 0.80 / 0.80，Qwen3.5-9B 0.74 / 0.88 / 0.88；
A 臂训练折越界均值 0.48 / 0.51 / 0.51 与 0.84 / 0.86 / 0.86。S1（基座越界错答等于捷径值）在 1 / 2 / 5 % 下 0.29 / 0.30 / 0.31 与 0.41 / 0.42 / 0.42。

**§2 偏 Spearman**：基座 pass@64 对 A 臂越界正确率、控制基座"给出替代关系"正确率后：Qwen3-8B 0.81，Qwen3.5-9B 0.88（70 机制）；"给出"正确率本身与越界的相关只有 0.07 / 0.20，
与 pass@64 的相关 0.25 / 0.12——可达性预测的不是"被告知时会不会用"。

## 跨干预的统一图景（§13，`explore/paper_synthesis.py` → `synthesis.json`，2026-11-01）

    cd $ATMOS_REPO && python -m explore.paper_synthesis --eval-dir $ATMOS_ROOT/experiments/_x_eval \
        --final $ATMOS_ROOT/experiments/_inputs/kc/final.json --inputs $ATMOS_ROOT/experiments/_inputs/x_eval --out results/paper/synthesis.json
    python -m explore.paper_synthesis --selftest

113 `rl-wt-31c5a4d` 上运行两次，输出逐字节相同（sha256 前缀 8fefa0d6）；S1 各臂的标准公式份额与 `results/opd/report*_qwen3-8b.json` 的 `shortcut_wrong_only_0.02` 一致。
