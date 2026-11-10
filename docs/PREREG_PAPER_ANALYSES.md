# 论文分析说明：写在算之前的二次分析（2026-10-29）

写于 2026-10-29，在 `results/paper/` 下任何数字被计算之前；此时 DOSE-text（`PREREG_DOSE_TEXT.md`）的 RL 正在训练，其任何单位结果、p、正确率都未读过。
本文一经提交不改；追加只能是带日期的说明。

**性质声明。** 下列分析用的数据全部已经看过（X-41、DOSE-4 已读过每一个数字）。它们**不是**对原假设的确证检验：P1 "只报数字"、P2 "只报数字"、
DOSE H1 "成立"、H2 "不可检验" 这些判定一字不改。本文只是把每项分析的统计量、重抽单位、种子、次数和**每种结果对应的读法句**在计算之前固定下来，
使得看到数字之后不再有任何选择。所有输出由 `explore/paper_rigor.py`、`explore/paper_atlas.py`、`explore/paper_traps.py` 生成到 `results/paper/`，
每个数字附带来源文件与键。

通用约定（沿用 `explore/x_report.py`）：种子先在机制内平均再重抽；按簇重抽（`atmoscoder_rl/coverage_clusters.units`，32 单元）或按机制重抽，10,000 次；
区间为 2.5 / 97.5 百分位；Spearman 按机制重抽（`spearman_ci`）。本文的随机数种子串统一为 `"paper:<项目>:20261029"`。
除非下面明文写了读法句，任何结果只报数字，不写判定词。

## 1. DOSE 的随机化推断（`results/dose/report_qwen3-8b.json` 的 34 个观测）

- **检验**：在每折 × 种子内，按 `explore/dose.assign` 的同一规则（每档 ⌊n/4⌋ 次，余数档无放回抽取，随机配给单位）重新分配剂量，
  其余全部保持（每个观测的 RL 正确率不变），重算 H1 统计量（剂量 ≥ 1 均值 − 剂量 0 均值），10,000 次，`random.Random("paper:dose-rand:20261029")`；
  p = 重抽统计量 ≥ 观测值 61.3 的份额（含相等）。
- **读法**：p < 0.01 → "剂量效应不能归因于分配本身"；0.01 ≤ p < 0.05 → "与分配随机性相容性弱"；p ≥ 0.05 → "只报数字"。
- **稳健性**（只报）：留一机制、留一运行（折 × 种子）后 H1 按机制重抽区间下界的最小值；以运行为单位的符号检验 4/4（p = 0.0625）照写，并说明为何随机化检验是合适的推断。

## 2. P1 的零假设与敏感性（`results/x/report_<家族>.json` `P1.per_mechanism`、`P1.sensitivity_pass32_T0.7`）

- **"学会"**：A、D、F 任一臂、任一种子在训练折越界题上 ≥ 50 %。
- **Fisher 精确检验**：Z（基座 pass@64 = 0）对非 Z × 学会 / 未学会，双侧；两家族各报。
- **AUC**：基座 pass@64、基座 pass@1 分别预测"学会"的 AUC，按机制重抽区间。
- **k 的敏感性**：Spearman(pass@k, 三臂两种子均值的越界正确率)，k ∈ {1, 8, 16, 32, 64}（同一 64 样本的无偏估计）；`sensitivity_pass32_T0.7` 下 |Z| 与"仍 ≤ 10 %"的个数并列 13/14。
- **难度混杂**：控制基座给出版正确率（`accuracy_by_variant` 的 given，按机制）的偏 Spearman；留一机制的 Spearman 最小值；按簇重抽的第二区间。
- **50 % 截断**：报 (机制, 臂, 种子) 的越界正确率落在 (0.1, 0.9) 的个数。
- 读法：无判定词。Fisher p 与 AUC 作为 P1 Spearman 的伴随数字报出。

## 3. 留出折的不对称：越界增益 − 原题增益（`x_report.transfer` 的口径）

- 每家族每臂（A、D、F、WA、WD）：留出折越界（终点 − 起点）减留出折原题（终点 − 起点），同一簇重抽得一个区间；起点为该臂的起点（A/D/F 基座，WA/WD 为 W_f）。
- **读法**（A 臂为主）：区间上界 < −10 点 → "越界增益低于一般增益至少 10 点"；区间含 0 → "两者无可检出差别"；其他 → 只报数字。

## 4. 热身 × 奖励的交互

- (WD − WA) − (D − A)，训练折越界，簇重抽区间（`warm_x_reward` 的四个臂均值的同一单元）。读法：无判定词；区间含 0 则写"交互无可检出"。

## 5. "未学会的机制被统一成标准公式值"是否是训练池的产物

- **统计量**：标准公式值份额 = 越界题中回答在 2 % 内等于 `shortcut_values` 的比例（`explore/dose_posthoc.close`），按机制；
  (a) **留出折**（训练池中没有该模板任何行）：各臂终点 − 起点（簇重抽区间），两家族；(b) 训练折同口径并列；(c) R 臂（Qwen3-8B）终点 − 基座；
  (d) DOSE 的热身模型对 RL 终点（已有，`posthoc` 的 `shortcut_warm/rl`）。
- **读法**：若留出折 (a) 的区间下界 > 0 且 R 臂不升（区间含 0 或 < 0）→ "答案奖励的 RL 把越界回答推向标准公式，与训练池是否含该模板的原题无关"；
  若 (a) 区间含 0 而 (b) 训练折上升 → "统一是训练池含同模板原题行的产物，不是 RL 的一般性质"；其他组合只报数字。论文措辞按此结果定。

## 6. DOSE 的 p 曲线与样本外检验

- **AUC**：热身后 p（`posthoc` 的 `warm_p`）预测"学会"（RL ≥ 50 %）的 AUC，26 个有剂量观测，按机制重抽；另报等张回归曲线（只报）。
- **样本外**：阈值 0.02 取自 DOSE-4、本文固定；在 X 的 69 个机制上（基座 pass@1 ≥ 0.02 对 < 0.02）× 学会（§2 定义）做 Fisher，两家族。
  读法：p < 0.01 → "0.02 的阈值在 X 上样本外可复现"；否则只报数字。DOSE-text 的检验另按其预注册。

## 7. 容差与捷径匹配的敏感性（只报，不判定）

- 从存档 `details` 以 `atmoscoder_rl.grading.verify(..., tolerance=)` 重评，容差 {2, 5, 10} %，报 D 的模板差距（16 配置）与 P1 的 Spearman；
  捷径匹配阈值 {1, 2, 5} %，报 S1 与 §5。写明：若干模板的越界区按"替代关系在 5 % 内正确"设计（`atmoscoder_rl/templates/MECHANISMS.md`），2 % 不是公平的检查，10 % 是稳健性的方向。

## 8. 前沿模型差距的批次拆分

- 模板首次提交日期（`git log --diff-filter=A`）：2026-09-26/27 共 30 个为筛查前（E0 于 10-01/02 用它们筛查 deepseek），10-01 之后的 42 个为筛查后。
  本文固定此标签。分别报 deepseek-v4-flash 关思考的原题 − 越界（按簇重抽），与 X5 的 13 个工程模板并列。读法：无判定词。

## 9. 分组分析的多重性

- 族：14 个类别、32 个簇、≤ 8 个近似类型（§10）。任何组间比较若报 p，按族做 Holm 校正；图谱与各组均值只报数字与区间，不写判定词。
  类别合并规则：soil、environmental_chemistry、hydrology、oceanography 合并为"非大气环境"；少于 4 个模板的类别不做组间比较。

## 10. 近似类型的盲编码

- 编码表 `results/paper/coding_sheet.md` 只含每个模板 / 陷阱的物理描述（模板文档的 "Parent's method / Where it fails" 段与标准做法 → 替代关系一句；
  陷阱的 `mechanism` 文本），不含任何结果；由用户（未看过逐机制结果）填 `results/paper/approx_types.json`，其 sha256 在计算任何按类型的数字之前追加到本文。
- 备选类型：线性化 / 局部斜率外推；几何理想化（平面平行、平地、无限延伸）；忽略过程 / 缺项；固定储库 / 稳态 / 过量反应物；无量纲数越界；
  参考态 / 基准面 / 边界条件；边界与像源 / 有限域；仪器约定。n < 5 的类型并入最近的类型（并入规则由编码人写在文件里）。
- 报告规则：类型 ≥ 5 个模板且跨 ≥ 3 个簇才报均值与区间；思考 / 规模救回的对比要求类型内 ≥ 3 个簇两臂都有。

## 11. 图谱与陷阱分析的口径（`paper_atlas.py`、`paper_traps.py`；只报数字）

- 主结果变量：P(越界对 | 原题对)（`results/conditional/d.json`，17 配置）；失败 = ≤ 0.5；普遍性分 = 失败的配置数（只计该模板被保留的配置）。
- 配置间一致性：机制排序的平均成对 Spearman（按机制重抽）；置换方差分解（配置内打乱机制标签 10,000 次，`"paper:atlas-perm:20261029"`）。
- 组织变量：机制难度（14 本地配置的 rec_outside 均值）在类别 / 簇 / 类型下的 ICC(1) 与置换零分布；类别内置换簇标签得 ICC(簇 | 类别)。
- 陷阱：Bench-Lab `experiments/trap/<模型>.run{1,2,3}.json` 配 `experiments/core_code/` 的父题（`control_id`），剔除 `trap_dn_9.27_gen` 与无 `details` 的记录；
  单位 = 陷阱，按 `parent_id` 重抽；只用 `review.measures_regime_judgement == true` 的 regime 陷阱做"须判断"组；每格 ≥ 6 道才报区间；
  捷径捕获 = 失败且每个子答案与 `shortcut_values` 在 2 % 内。

## 12. 无声错误的物理代价（2026-10-29 追加，写于计算之前；`explore/paper_cost.py` → `results/paper/cost.json`；只报数字）

- 问题：标准公式在越界条件下给出的值离正确值多远——领域读者要的"错多大"，与"错多少道"分开报。
- 数据：Bench-Lab `benchmark/traps.json` 全部 335 道（`review.magnitude_defect` 的 7 道另列，主表含它们也报不含它们的数）；72 个模板在
  `experiments/_inputs/kc/final.json` 的 20 道越界行；两者都带 `shortcut_values` 与 `sub_answers`。
- 逐子答案：捷径值 s、期望值 e（多个可接受值取第一个）。相对偏差 R = |s − e| / |e|；倍数 F = max(|s|/|e|, |e|/|s|)。s·e < 0 记"符号反"，
  e = 0 或 s = 0 记"未定义"，两者只计数不入分布。逐陷阱取各子答案的最大值（R_max 即 `separation_max` 的定义）；逐模板取 20 行 R_max、F_max 的中位数。
- 分布：四分位数与分档计数——≤ 5 %（在判分容差内）、5–20 %、20–50 %、50–100 %、2–10 倍、> 10 倍、符号反；总体、按类别（§11 的合并规则）、
  按 `trap_type`、regime 陷阱按 `measures_regime_judgement`；模板按类别。区间按陷阱 / 模板重抽 10,000 次，`random.Random("paper:cost:20261029")`。
- 实际发生的错误：前沿运行（`experiments/trap/<模型>.run{1,2,3}.json`，deepseek-v4-flash 关 / 开思考、gpt55-reasoning）每条失败记录的
  实际答案对期望值的 R_max 与 F_max（同一分档），按配置、按类别；另报失败记录中 R_max > 50 % 的份额与等于捷径值（2 % 内）的份额。
- 读法：只报分布与计数，不写判定词；"≤ 5 %"档的陷阱若存在，说明该陷阱在 5 % 容差下无法区分捷径与正确做法，须在陷阱表里标出并从捕获统计中剔除后复算。

## 2026-10-30 修订：编码人改为盲的独立代理（写于任何编码之前）

- 用户委托执行编码。执行者（主会话）已看过逐机制结果，不能作盲编码人；改为两个新起的独立代理 C1、C2，各自只读编码表
  `results/paper/coding_sheet.md`（sha256 前 16 位 `c09e2fac06eeac8a`）的一份副本，放在仓库外的空目录里，不读仓库、结果或对话上下文，互不知道对方的编码。
- 一致性：85 个模板与 103 道陷阱分别报 C1–C2 的原始一致率与 Cohen κ（合并前的 8 类）。
- 裁决：不一致的行交给第三个同样盲的代理 C3，只给物理描述与 C1、C2 的两个选项（不说谁选了哪个），在两者中选一。
  一致的行直接采用。
- 合并：裁决后若某类模板数 < 5，由 C3 按"物理上最近的类型"写出合并规则，写进 `approx_types.json` 的 `merges`。
- 最终文件 `results/paper/approx_types.json` 与 C1、C2 的原始编码一并入库，其 sha256 在计算任何按类型的数字之前追加到本文。第 10 节其余规定不变。

## 2026-10-30 编码结果入库（写于任何按类型的数字计算之前）

- C1–C2 一致性（合并前 8 类）：模板 78/85（91.8 %），Cohen κ 0.90；陷阱 93/103（90.3 %），κ 0.88。C1、C2 是同一模型的两个实例，
  一致性可能高于两名独立的人类编码人。两人各自报告约 45–48 行"难以决定"，陷阱的难点主要是编码表只给了一句机制或题目首句。
- C3 裁决 17 行不一致（选项随机排序，不知来源；随机种子 `paper:adjudicate:20261030`），A 9 次、B 8 次。
- 合并（C3 写定）：边界 / 像源 / 有限域（4 个模板）→ 几何理想化；仪器约定（3 个模板）→ 参考态 / 基准面 / 边界条件。
- 合并后模板：忽略过程 24、线性化 17、几何理想化 15、无量纲数越界 11、参考态 10、固定储库 8；陷阱：38 / 11 / 11 / 13 / 16 / 14（同序）。
- `results/paper/approx_types.json` sha256 `c85157d7bded249914107d70db136e28888fc7545bdf556e3c9fcbd0dfee1fbd`；
  C1、C2 原始编码、裁决表与答案键、合并表在 `results/paper/coding/`。

## 13. 跨干预的统一图景（2026-11-01 追加，写于计算之前；`explore/paper_synthesis.py` → `results/paper/synthesis.json`；事后 / 探索性）

- **前提**：X、DOSE、OPD（含 OPD-13 的 Pk2、Pshuf）的结果都已看过；已看到的数字有两处与本节相关——OPD 报告的次要块里越界错答的标准公式份额
  （A 74.5 %、P 16.1 %、Pk2 15.9 %、Pshuf 33.0 %）与 F14 的"教师给出版 ≥ 0.35 学会 7 / 9，≤ 0.18 学会 0 / 5"。本节不是确证检验，只把统计量、单位、种子与读法在算之前固定，
  结论在台账里标探索级。不读 Ministral 的任何结果。家族：Qwen3-8B。重抽按模板 10,000 次，种子串 `random.Random("paper:synthesis:<项>:20261101")`。
- **S1 错误形态**：臂 A（种子 42 / 43 平均）、P、S、Pk3、Pk2、Pshuf（种子 42）与基座；每个模板在其训练折的越界行上，只看答错的行（5 % 容差，`units="benchmark"`），
  分三类——等于标准公式值（每个子答案与 `shortcut_values` 在 2 % 内，同 `paper_rigor_eval.shortcut_share`）、无可判输出（无代码、崩溃或无返回）、其它数值；
  每类取模板内份额，再对模板平均。原计划的第四类"Pshuf 错答等于正确替代关系的答案"不做：给出版行的期望值就是越界行的正确答案，等于它即为答对；
  供体模板的关系套到本题会得出的值，现有数据算不出。对比：A − P、Pshuf − P、A − 基座、P − 基座的标准公式份额差（只用两边都有错答的模板）。
  读法：A − P 区间下界 > 0 → "RL 让答错的越界题收拢到标准公式值，特权蒸馏不会"；否则只报数字。
- **S2 谁的可达性限制谁**：单位 = 基座越界正确率 < 0.5 的模板。预测量：学生的基座 pass@64（`results/x/report_qwen3-8b.json` 的 `P1.per_mechanism`）；
  教师的给出版正确率（`results/opd/main_set.json` 的 `per_template[*].given_acc`，即基座在给出版题上的采样正确率——P 的教师就是看着给出版的基座）。
  结局：A、P、Pk2、Pshuf 各自"学会"= 训练折越界正确率 ≥ 0.5（同 P1）。统计量：每个结局下两个预测量各自的 AUC（`paper_rigor.auc`），
  以及按模板配对重抽的 AUC 差。读法：A 下"学生 − 教师"差的下界 > 0，且 P 下"教师 − 学生"差的下界 > 0 → "每种方法受其信号源的可达性限制"；
  否则只报数字。Pk2、Pshuf 只报数字（Pshuf 是负对照：教师给的是错关系）。一个结局下学会或没学会的模板为 0 时，AUC 不可算，记"不可算"。
- **附表（只报数字）**：模板 × 干预的结局矩阵——基座越界、学生 pass@64、教师给出版、A / D / F（两种子平均，X 报告）、DOSE 观测（`results/dose/posthoc_qwen3-8b.json`，
  有剂量的单位）、P / S / Pk3 / Pk2 / Pshuf；行附 `approx_types.json` 的近似类型（sha256 `c85157d7…`）与类别。

### 13 补充（2026-11-01，看过 S1 结果之后加的复核，事后）

- S1 只看答错的行，RL 已学会的模板错答少，剩下的错答可能集中在难题上，从而抬高标准公式份额。两项复核（`explore/paper_synthesis.py --robust`，
  读已入库的 `synthesis.json` → `results/paper/synthesis_robustness.json`，种子串同 §13）：(1) 不按答错筛选——每个模板全部越界行中给出标准公式值的份额
  （= 错答中的份额 ×（1 − 正确率）），69 个模板上 A − P；(2) 只看 A、P 都没学会（训练折越界 < 0.5）的模板，错答中的份额 A − P。只报数字，用来判断 S1 的读法是否依赖筛选。

## 14. 第三轮挖掘（2026-11-01 追加，写于计算之前；`explore/paper_synthesis.py --round3`；事后 / 探索性，只报数字）

- **已看到的**：`synthesis.json` 的矩阵里，A 学会的 33 个模板上 OPD 各臂保住的个数（P 17、Pk2 13、Pk3 10、Pshuf 6、S 4）；S 臂日志的蒸馏损失约 0.001、
  `distillation_loss_coef` 1.0、`use_task_rewards` True（策略梯度损失在算，`verl/trainer/distillation/losses.py:distillation_ppo_loss`）。
  OPD 各臂与 A 的写定差别是关闭动态采样（PREREG_OPD"与 A 的两处必要差别"(i)）。因此 S ≈ "同一 RL、不做动态采样、外加一个几乎为零的 KL"。
- **R1 动态采样的近似消融**：单位 = A 学会的模板（基座越界 < 0.5 且 A ≥ 0.5）。报 A、S、P、Pk3、Pk2、Pshuf 的平均训练折越界正确率与"学会"个数；A − S 的区间；
  P 在其中保住与没保住的模板上教师给出版正确率的中位数与 AUC。读法只报数字；写明 S 与 A 还差 GPU 划分（4 卡学生）与一个几乎为零的蒸馏项，不能当成干净的消融。
  并据此在 F16 注明："A 会 P 不会"的一侧不能归因于教师——OPD 管线里的奖励学习被关闭动态采样削弱了。
- **R2 错答离真值多远**：对 S1 的同一批错答（训练折越界行、答错且有数值输出），逐行 R = max 子答案 |给出值 − 期望值| / |期望值|（期望值为 0 的子答案不计），
  逐模板取中位数，再报各臂的模板中位数分布（四分位）与 R > 0.5 的模板份额；臂：基座、A、P、S、Pk2、Pshuf。用来检验 F15 的"错得分散"是否意味着偏得更多。
- **R3 近似类型 × 方法**：`approx_types.json`（合并后 6 类）× {A、P、DOSE 有剂量观测} 的"学会"计数与比例（单位 = 基座越界 < 0.5 的模板）；只报数字，
  不做检验（每类 4–15 个，F3 已报类型对难度无可检出效应）。

### 13–14 结果记录（2026-11-01；数字只报不判，除 §13 写定的两句读法）

- **§13**（`synthesis.json`，113 `rl-wt-31c5a4d`，两次运行逐字节相同，sha256 前缀 8fefa0d6）：S1 越界错答中等于标准公式值——基座 29.8 %、A 74.5 %、P 16.1 %、S 33.7 %、
  Pk3 28.7 %、Pk2 15.9 %、Pshuf 33.0 %；A − P 57.4 [44.5, 69.8]（n = 36）→ 读法成立："RL 让答错的越界题收拢到标准公式值，特权蒸馏不会"。
  S2（65 个模板）：A 学生 AUC 0.888 / 教师 0.512，差 +0.376 [0.223, 0.534]；P 0.621 / 0.860，差 −0.239 [−0.411, −0.070] → 读法成立："每种方法受其信号源的可达性限制"。
  Pk2 −0.111 [−0.311, 0.091]、Pshuf +0.277 [0.095, 0.477]，只报数字。
- **§13 补充**（`synthesis_robustness.json`）：不按答错筛选 A 42.4 % 对 P 6.2 %，差 36.2 [25.4, 47.4]（69 个模板）；两臂都没学会的 19 个模板上 73.5 [60.3, 85.8]。S1 的读法不依赖筛选。
- **§14**（`synthesis_round3.json`，113 `rl-wt-f0122b6`，两次运行逐字节相同，sha256 前缀 79bcd2e3）：
  R1 在 A 学会的 33 个模板上 A 33（0.95）、P 17、Pk2 13、Pk3 10、Pshuf 6、S 4（0.17），A − S 77.7 [68.3, 86.5]；P 保住的教师给出版中位数 0.84、没保住的 0.18（AUC 0.88）。
  S 日志：`distillation_loss_coef` 1.0、蒸馏损失约 0.001、策略梯度损失非零——差别主要是关闭了动态采样（不是干净消融：还差 4 卡学生与该 KL 项）。
  据此 §13 S2 的"RL 会而 P 不会"一侧不归因于教师；台账 F16 已注明，F17 记为探索级只报数字。
  R2 错答相对误差的模板中位数：基座 0.89、A 0.46、P 1.00（四分位 0.59–4.92）、S 0.97、Pk2 1.00、Pshuf 0.82；中位数 > 0.5 的模板份额 72 / 49 / 80 / 72 / 80 / 65 %。
  P、Pk2 的中位数恰在 1.0 附近，提示大量错答接近 0 或差一个 2 倍因子，形态未细查。
  R3 学会 / 模板数（基座越界 < 0.5）：几何理想化 A 2/8、P 5/9、DOSE 有剂量 1/4；固定储库 A 6/7、P 3/7、DOSE 0/1；线性化 A 6/12、P 6/12、DOSE 3/3；
  忽略过程 A 9/20、P 6/20、DOSE 5/5；参考态 A 6/10、P 6/10、DOSE 1/2；无量纲数越界 A 4/8、P 4/8、DOSE 1/1。

## 15. 学会的是什么：条件判断还是公式替换（2026-11-03 追加，写于计算之前；`explore/paper_round4.py --conditional` → `results/paper/round4_conditional.json`；事后 / 探索性）

- **已看到的**：§6 台账 F1–F18 的全部数字；X 各臂 `accuracy_by_variant`（按标签的 inside 聚合）、P2（按簇的 H / F）；OPD `inside_training_fold`（70 模板聚合）；
  `rigor_eval` §3 的留出簇不对称。**没看过**的：任何终点逐机制的 inside 行正确率；DOSE / OPD / Ministral 终点的 judge.json。
- **单位**：机制 × 干预终点，训练折。干预与起点：X A 臂（Qwen3-8B 与 Qwen3.5-9B，两折 × 两种子，起点 = 基座）；DOSE RL 终点（Qwen3-8B、Ministral 3 8B，起点 = 该种子的热身模型 `sft-dose-…`）；
  OPD P / S / Pk3 / Pk2 / Pshuf（Qwen3-8B，起点 = 基座，与 opd_report 一致）。种子先平均（`x_report.avg`），折合并。
- **每格四个量**（逐机制，`x_report.Family`）：outside 正确率、inside 正确率（`accuracy(label, "inside")`，走 `final_regrade.json` 的重判缓存，与 x_report / opd_report 同口径）、
  H = outside 行判"不成立"的份额（`flag0(label, "outside")`）、F = inside 行判"不成立"的份额（`flag0(label, "inside")`，judge 评分 `d3_report.score` 同 P2）。
- **"学会"** = 终点 outside ≥ 0.5 且起点 outside < 0.5（同 P1、§14 R1）。
- **S15-a 原题代价**：学会的机制上 Δinside = inside(终点) − inside(起点)：按干预报均值与机制重抽区间（10,000 次，`random.Random("paper:round4:<干预>:20261103")`）；
  逐机制三类（边界事先定）：**条件型** Δinside ≥ −0.10；**替换型** Δinside < −0.25；其余**中间**。报每种干预三类的个数与名单。对照：同一干预下没学会机制的 Δinside（均值与区间）。
- **S15-b 顺带学会判断**：学会的机制上 Δ(H − F) = (H − F)(终点) − (H − F)(起点)，按干预报均值与区间；没学会机制的同一量作对照；X 的 F 臂（训练判断）的 P2 数字只引用、不重算。
- **读法（写定）**：S15-a 某干预下替换型个数 ≥ 条件型 → "该干预学到的主要是无条件的公式替换"；条件型占多数且 Δinside 区间下界 > −0.10 → "学会越界答案没有以原题为代价"；否则只报数字。
  S15-b 学会机制的 Δ(H − F) 区间下界 > 0 → "学会替代关系顺带提高了失效判断"；否则只报数字。两家族 A 臂分别写，不合成。
- 附（只报）：每个标签的 test_id（64 道）与四项整体正确率（`family_compare.overall_accuracy` 的做法）。
- **执行**：X 各臂的 final / judge 只在服务器 `_x_eval/`，judge 评分要跑沙箱 → 在 113 从提交后的 worktree 跑两次、逐字节相同；DOSE / OPD / Ministral 的子集本地也跑一次核对对应键。

## 16. 难在机制还是模型：三家族可达性与学会的一致性（2026-11-03 追加，写于计算之前；`--families` → `results/paper/round4_families.json`；事后，只报数字加两句写定读法）

- **已看到的**：三家族各自的基座 pass@64（`results/x/report_qwen3-8b.json`、`report_qwen35-9b.json` 的 `P1.per_mechanism`，`results/x/reach_ministral3-8b.json`）、F3 的机制方差（D 的 17 个配置）、
  F7 / F18。没看过的：三家族 pass@64 并在一起的任何统计。
- **S16-a**：69 个共同机制（`plume_mixing_lid` 按 `P1_EXCLUDED` 排除），三对家族的 Spearman（`x_report.spearman_ci` 的口径，按机制重抽；种子 `"paper:round4:families:<对>:20261103"`）；
  三家族 pass@64 都为 0 的机制、恰有一家为 0 的机制（按家族列出）；每家族 pass@64 与 D 图谱机制难度 `atlas.per_template[t]["mean_rec_outside_local"]` 的 Spearman。
- **S16-b**：A 臂"学会"（两种子均值 ≥ 0.5，`P1.per_mechanism` 的 `A_s42 / A_s43`）在 Qwen3-8B 与 Qwen3.5-9B 之间的 2×2 表、Cohen κ 与机制重抽区间；只一家学会的机制列出两家的 pass@64。
- **S16-c**：DOSE 两家族有剂量的观测合并（Qwen 26 + Ministral 18）：热身后 p 预测学会的 AUC（`paper_rigor.auc_boot`，单位 = 机制，一个机制两个观测取学会比例与 p 均值——写明），分家族与合并；
  合并的 p 三档表；按家族分层的"有剂量 − 无剂量"差（各家族内差按观测数加权，机制重抽）。只报数字，不改两家的 H1 判定。
- **读法（写定）**：S16-a 三对 ρ 的区间下界都 > 0.3 → "可达性首先是机制的属性，其次才是模型的"；S16-b κ 下界 > 0.4 → "两家族学会的是同一批机制"；否则只报数字。S16-c 只报数字。

## 17. 注入的知识到不到教科书（2026-11-03 追加，写于计算之前；`--traps` → `results/paper/round4_traps.json`；事后，只报数字加一句写定读法）

- **已看到的**：`results/traps_rl/` 的 X 各臂、`results/opd/report_qwen3-8b.json` 的 OPD P / S / Pk3 textbook；没看过的：DOSE 两家族（热身与 RL）、Ministral 基座、OPD Pk2 / Pshuf 在 46 道 regime 陷阱上的数。
- **口径**：与 `explore/traps_rl_report.py` 相同——46 道 `regime_arm` 陷阱与父题，判定用运行器的 passed；每标签陷阱正确率、父题正确率、父题对时陷阱对、失败行的捷径捕获（2 %）；
  相对起点的配对增益按陷阱重抽 10,000 次（`random.Random("paper:round4:traps:<标签组>:20261103")`）。起点：DOSE RL = 同种子热身模型；DOSE 热身与 Ministral RL 的起点 = 该家族基座；OPD = 基座。
- **报**：每个标签与每组（种子平均、折合并）的数；DOSE RL − X A（同家族同折，`results/traps_rl/report_qwen3-8b.json` 的 per_trap 直接引用）的配对差；Ministral 基座 → 热身 → RL；OPD 五臂并列（P / S / Pk3 引用已有报告的口径重算一次以对齐种子）。
- **读法（写定）**：Qwen DOSE RL − A 的区间含 0 → "注入的模板知识没有额外迁移到教科书陷阱"（F11 的扩展）；下界 > 0 → 只报数字并标"单家族、46 道"。

### 15–17 结果记录（2026-11-03；全文与表在 `docs/RESULTS_ROUND4.md`，台账 F19–F21）

- **§15**（`round4_conditional.json`，113 `rl-wt-443141b`，两次运行逐字节相同，sha256 前缀 b443eaeb）：读法 a 在全部干预下成立（替换型 0 个；学会机制 Δinside Qwen A +58.0 [46.1, 69.2]、Qwen3.5 A +23.8 [14.0, 34.2]、
  Qwen DOSE +51.4、Ministral DOSE +56.1、OPD P +20.3、Pk2 +40.7）；读法 b 不成立且方向相反（Δ(H−F) Qwen A −9.5 [−19.5, −0.3]、Qwen3.5 A −15.2 [−27.3, −4.2]，DOSE 含 0，Pk2 +10.1 [0.2, 20.4] 为单臂单种子）。
  复核：训练题库含训练折模板的 inside 行（分布内）；judge 解析率起点与终点 0.98–1.00。
- **§16**（`round4_families.json`，本地两次相同，dd1c777b）：ρ 0.78 / 0.53 / 0.47，最低下界 0.25 → 读法不成立，只报数字；κ 0.23 [0.09, 0.40] → 读法不成立，只报数字（学会集合嵌套 37 ⊂ 62）；
  S16-c 合并 AUC 0.92 [0.81, 0.99]，分层差 +56.7 [24.7, 81.8]。
- **§17**（`round4_traps.json`，本地两次相同，427fd382）：Qwen DOSE RL − A +2.2 [−2.2, 6.5] → 读法成立；复核时追加对 WA 臂的差 +3.8 [−2.2, 9.8]（脚本同日加入，只报数字）。

## 18. 学会率随起点可达性的分档形状（2026-11-06 追加，写于计算之前；`explore/paper_pcurve.py` → `results/paper/pcurve.json`；事后，只报数字加一句写定读法；供稿件 Fig. 5b）

- **已看到的**：DOSE-4 记录里 Qwen3-8B A 臂按基座 pass@1 分档的学会率（6/32、10/14、7/9、14/14）与 `results/dose/posthoc_qwen3-8b.json` 的 `x_a_arm_by_base_p`；
  没看过的：Qwen3.5-9B 的分档、按 pass@64 而非 pass@1 的分档、分档的区间。
- **口径**：单位是 X 的机制（两家族各 69 个，`results/x/report_<family>.json` 的 `P1.per_mechanism`）。起点可达性 p = 基座 pass@1（20 道越界行 × 64 次采样的平均单次正确率）。
  单位限于基座越界正确率 < 0.5 的机制（RL 有东西可学；基座越界正确率取 `results/paper/round4_conditional.json` 的 `outside_start`，两家族都有），
  学会 = A 臂两种子终点越界正确率均值 ≥ 0.5（与 §13 / F7 的"学会"相同；Qwen3.5-9B 用 s42–43 报告）。分档固定为 [0, 0.005)、[0.005, 0.02)、[0.02, 0.1)、[0.1, 1]。
  （2026-11-06 自检后、算真实数据前的修订：原稿把"基座越界 < 0.5"写成学会的条件而非入选条件，合成数据自检暴露了这一点。）
  每档报 n、学会数、学会率与 Clopper–Pearson 95 % 区间。理论参考曲线 g(p) = 1 − (1 − p)⁸ − p⁸：一组 8 次采样里"有对有错"的概率，即一步里该题产生非零优势的概率（DAPO 动态采样最多补采 10 批，所以这是每步信号量的下界）；
  在 p 的网格上给出，不拟合、不缩放。
- **报**：两家族各四档的表；p 的中位数按档；基座 pass@1 = 0 的机制数与其中学会的数。
- **读法（写定）**：若两家族里学会率都随档单调不降、且 [0.02, 0.1) 档的下界高于 [0, 0.005) 档的上界 → "学会率在 p 为百分之几处陡升，与组方差曲线同形"；否则只报数字。

### 18 结果记录（2026-11-06；`results/paper/pcurve.json`，本地两次运行逐字节相同，sha256 前缀 9b58cc47）

- Qwen3-8B（65 个基座越界 < 0.5 的机制）：6/32 = 0.19 [0.07, 0.36]、10/14 = 0.71 [0.42, 0.92]、7/9 = 0.78 [0.40, 0.97]、10/10 = 1.00 [0.69, 1.00]；pass@1 = 0 的 14 个学会 1 个。单调，且第三档下界 0.40 > 第一档上界 0.36。
- Qwen3.5-9B（32 个）：4/9 = 0.44 [0.14, 0.79]、2/3 = 0.67、2/4 = 0.50、16/16 = 1.00 [0.79, 1.00]；pass@1 = 0 的 5 个学会 2 个。不单调（中间两档各只有 3–4 个机制）。
- **判定：只报数字**（Qwen3.5-9B 未过）。稿件 Fig. 5b 画两家族的分档与曲线，图注写明 Qwen3-8B 同形、Qwen3.5-9B 的中间档样本太少。

## 19. D 对三个改写模板的敏感性（2026-11-06 追加，写于计算之前；`explore/d_reword_sensitivity.py` → `results/paper/d_reword.json`；事后，只报数字加一句写定读法）

- **为什么**：训练研究对 `line_broadening`、`drop_ventilation`、`clausius_clapeyron_extrapolation` 三个模板在所有版本里用改写题面（`PREREG_REWORD.md`，`results/e0/README.md` 措辞复核），D 用的是原措辞（其输入构建没有改写步）；
  `PAPER_METHODS.md` 的 open points 要求在写 D 的数字前核清。
- **已看到的**：`results/d_execution/summary.md` 的 16 行（全部 72 个模板）；为确认数据可用，打印过这三个模板在 16 个配置里的逐模板正确率（原题几乎都 ≥ 0.85、越界全为 0；`line_broadening` 自成一簇，另两个分属 `diffusional_growth`、`radiative_linearisation`）。
  去掉它们之后的对比与判定没有算过。
- **数据**：16 个配置的逐模板正确率——12 个取 `results/d2/families_report_*.json`，Qwen3-8B 关、Qwen3.5-9B 关、Gemma4-E4B 关 / 开取 `experiments/d_execution_review/{qwen3_8b,qwen35_9b,gemma_e4b}.json`
  （与 `results/d_execution/completed_configurations.json` 记录的 sha256 一致；gemma_e4b 于 2026-11-06 自服务器取回）。
- **口径**：完全沿用 `atmoscoder_rl.diagnosis.template_summary` 的聚合（簇内平均全部模板、簇等权；usable 门 = 至少 15 个簇有可用模板）与 `mean_ci`（种子 20261008，10,000 次）；
  只把三个模板从模板集合里去掉（`line_broadening` 簇随之消失，簇数 32 → 31）。先用全部 72 个模板复算，必须与 `families_report` / 复核文件里存的 `all.contrasts` 逐位相同，才算去掉后的数。
  D-1 判定按 `diagnosis.verdict` 重算：教科书一侧不变，只换模板对比与 usable 门。
- **报**：每个配置原题 − 越界、给出 − 越界、只提示 − 越界在 72 / 69 模板下的均值与区间、差值；usable 簇数；D-1 判定是否改变。
- **读法（写定）**：若 16 个配置的 D-1 判定都不变，且原题 − 越界的变化绝对值都 ≤ 3 个百分点 → "D 的数字不依赖这三个模板的原措辞，正文沿用 72 个模板"；否则列出改变的配置，正文报 69 个模板的数并注明。

## 20. 训练后的小模型与前沿模型在同一批教科书陷阱上的配对比较（2026-11-07 追加，写于计算区间之前；`explore/paper_frontier_paired.py` → `results/paper/frontier_paired.json`；事后，只报数字加一句写定读法）

- **为什么**：正文 §3 把 Qwen3.5-9B 训练后在 46 道陷阱上的 54.9 % 与 DeepSeek V4 Flash（关思考）在其中 31 道上的 52.7 % 并列，只能写 "close to"，因为两者不在同一组题上。`traps_frontier.json` 有前沿配置逐陷阱的结果，可以在重合的陷阱上配对。
- **已看到的**：本日在会话里按同 31 道陷阱算过点估计（Qwen3.5-9B A 臂 60.5 %、基座 41.9 %；DeepSeek 关思考 52.7 %、开思考 87.1 %、GPT-5.5 开思考 97.8 %；Qwen3-8B A 臂 21.8 %），以及各外部配置与 46 道的重合数（三个配置 31 道，其余 4–9 道）。没有算过任何区间。
- **数据**：`results/traps_rl/report_<family>.json` 各臂 `per_trap`（逐陷阱正确率 = 该臂全部终点的均值，A/D/F/WA/WD 四个终点，base 一个）；`results/paper/traps_frontier.json` 各配置 `per_trap`（三次运行的均值）。
- **口径**：单位是陷阱。对每个家族 × 臂 × 外部配置，取两者都有记录的陷阱；重合 < 30 道的配置不报（与 `traps_rl_report` 的 `frontier_on_same_traps` 同门槛）。统计量 = 逐陷阱（臂 − 外部配置）的均值；区间为按陷阱重抽的 95 % 百分位 bootstrap，10,000 次，种子 20261008。同时报两者在重合陷阱上的水平。
- **报**：两个家族、臂 base / A / D / F / WA / WD / R，对每个过门槛的外部配置：n、臂水平、外部水平、配对差与区间。
- **读法（写定，只对 Qwen3.5-9B A 臂对 DeepSeek V4 Flash 关思考这一组）**：差的区间下界 > 0 → "训练后的 9B 在同一批陷阱上超过关思考的前沿模型"；区间含 0 → "两者无可检出差别"（正文用 matches）；上界 < 0 → "仍低于"。开思考的配置只报数字；正文无论结果都要同时给出开思考的水平。其余家族 × 臂只报数字。
- **结果（2026-11-07，两次运行逐字节相同，sha256 前缀 0e505195）**：过门槛的外部配置三个（DeepSeek V4 Flash 关 / 开思考、GPT-5.5 开思考，各与 46 道重合 31 道）。Qwen3.5-9B A 臂 60.5 % 对 DeepSeek 关思考 52.7 %，配对差 +7.8 [−5.4, +21.0] → 读法「两者无可检出差别」；对开思考 87.1 % 为 −26.6 [−40.3, −13.4]，对 GPT-5.5 开思考 97.8 % 为 −37.4 [−51.3, −24.2]；基座 41.9 %（−10.8 [−32.3, +10.8]）。Qwen3-8B A 臂 21.8 %，低于全部三个配置（对关思考 −30.9 [−45.4, −16.1]）。

## 21. 训练后的 9B 与多个前沿模型在同一评测框架下的比较：教科书陷阱、父题、留出折模板（2026-11-07 追加，写于任何新调用之前；`explore/frontier21_report.py` → `results/paper/frontier21.json`；事后，只报数字加写定读法）

- **为什么**：§20 的配对比较只有一个关思考的前沿配置（DeepSeek V4 Flash）、只覆盖 46 道陷阱里的 31 道，而且前沿一侧的数来自 AtmosCoder-Bench 的评测框架，训练后的 9B 来自本仓库的 X 评测框架。要把"训练后的 9B 关思考达到前沿模型关思考的水平"写进正文，需要（i）同一框架、同一提示词、同一采样设置；（ii）全部 46 道陷阱和它们的 46 道父题；（iii）不止一个前沿模型；（iv）陷阱之外，在 9B 没训过的留出折模板上也比一次。
- **已看到的**：§20 的结果（31 道上 9B A 臂 60.5 %、DeepSeek 关 52.7 %、开 87.1 %、GPT-5.5 开 97.8 %）；AtmosCoder-Bench 框架下各前沿配置在 46 道父题上的多数票正确率（DeepSeek 关 69.6、GPT-5.5 关 84.8、Kimi K2.6 82.6、Qwen3.5-397B 87.0、Gemini 3.1 Pro 93.5；开思考更高）；
  `results/traps_rl/report_qwen35-9b.json` 各臂在 46 道上的水平（A 臂陷阱 54.9、父题 64.7；基座 34.8 / 58.7）；DeepSeek 关思考在**诊断集**（72 模板，另一次参数抽样）上的越界正确率 65.3 %、开思考 88.4 %（`results/d2/frontier_*.json`）；
  9B A 臂留出折越界 60.4 %（`results/x/report_qwen35-9b.json` transfer）。本节的新调用一次都没有发出；本框架下没有任何前沿模型在 46 道陷阱、父题或最终集上的结果。
- **模型（按可用性事先定下，结果无论如何全部报告）**：关思考四个——DeepSeek V4 Flash（用户 2026-11-07 提供的 gateway.example 网关，`thinking.type = disabled`，探测返回 reasoning_tokens 0）、GPT-5.5（external-api-platform，`reasoning_effort = none`，探测 reasoning_tokens 0）、GPT-5.4（同，`none`）、MiniMax-M3（同，`thinking.type = disabled`，reasoning_tokens 0）。
  开思考两个，只报数字——DeepSeek V4 Flash（api1，`reasoning_effort = high`，探测返回 reasoning_content）、GPT-5.5（`reasoning_effort = medium`，与 AtmosCoder-Bench 的 gpt55-reasoning 一致）。
  不可用、不跑：Qwen3.5-397B（用户说明跑不了）、DeepSeek V4 Pro、Kimi、Claude、GLM、Grok（网关无通道或账户欠费，2026-11-07 探测）。models.toml 节 `frontier21-*`。
- **评测框架**（与 X 的终点评测完全相同，`explore/x_eval.sh`）：`atmoscoder_rl.evaluation.runner`，code 协议，`prompts.SYSTEM` 与 X 相同，温度 0.3，最大输出 8,192（开思考 28,672），容差 5 %，最多 5 次内容尝试，每行 1 个样本；API 错误行在第二遍重跑，仍为错误的行不进分母并报个数。
  输入：`experiments/_inputs/x_eval/textbook.json`（92 行 = 46 道陷阱 + 46 道父题，manifest sha256 65c24635…）三次重复（`--run 1/2/3`，六个配置都跑）；`experiments/_inputs/x_eval/final_think.json`（最终集的 inside / outside 行，70 模板 × 20 对 × 2 = 2,800 行，与 X 评测的 `final.json` 同一批行）一次，只跑四个关思考配置。输出 `experiments/frontier21/<配置>/`。
- **口径**：
  - S21-a 陷阱与父题：单位是陷阱（46；父题与其陷阱配对）。前沿配置逐陷阱正确率 = 三次重复的均值（与 `traps_rl_report` 的 `frontier_on_same_traps` 相同）；9B A 臂逐陷阱正确率 = 四个终点的均值（`results/traps_rl/report_qwen35-9b.json` `arms.A.per_trap`，不重算）。每个前沿配置报：陷阱水平、父题水平、陷阱 − 父题，以及配对差（A − 前沿）在陷阱与父题上各一个；区间为按陷阱重抽的 95 % 百分位 bootstrap，10,000 次，`random.Random("paper:frontier21:<项>:20261107")`。同样给出基座 9B、Qwen3-8B A 臂对每个前沿配置的差，只报数字。
  - S21-b 留出折模板：单位是簇（70 模板的 32 簇，簇内模板等权、簇等权，与 X 的 transfer 统计量一致）。9B A 臂逐模板的留出折越界 / 原题正确率 = 该模板所在折之外的那一折训练出的 A 终点在最终集上的正确率，两种子平均（服务器 `_x_eval/<label>/final.json`，`explore.x_report.Family` 的口径）；前沿配置逐模板正确率来自本节的 `final.json`。报每个关思考配置的越界水平、原题水平、原题 − 越界，以及簇层面的配对差（A 留出折 − 前沿）在越界与原题上各一个，区间同上。
  - S21-c 代价：每个配置在 92 行上的 completion tokens 中位数与均值，与 9B A 臂四个终点的同一量并列（记录里的 `usage`），只报数字。
- **读法（写定；只对关思考配置的 S21-a 陷阱差与 S21-b 越界差）**：区间下界 > 0 → 该配置上"训练后的 9B 超过"；区间含 0 → "无可检出差别"（正文用 matches）；上界 < 0 → "仍低于"。正文凡写到此比较，必须同时给出：同一配置的父题差（预期 9B 低于前沿）、开思考配置的水平、以及 9B 是在同类机制的模板上训练过而前沿模型没有。四个关思考配置逐个写判定，不合并、不挑选。

### 21 结果记录（2026-11-07；`results/paper/frontier21.json`，本地两次运行逐字节相同，sha256 前缀 3e1135f1；运行数据 `experiments/frontier21/`，输入哈希 `INPUTS.sha256`）

- **完整性**：六个配置的 textbook 三次重复都是 92/92 行、0 个 API 错误行；四个关思考配置的 final 都是 2,800/2,800 行、0 错误。关思考配置的全部 9,228 条记录 reasoning_tokens 均为 0，开思考配置全部 > 0。锚点：9B A 臂留出折越界（簇加权）0.5867 与 `results/x/report_qwen35-9b_s42-43.json` transfer.A.held_out 逐位相同。
- **S21-a 陷阱（46 道）**，9B A 臂 54.9 %、父题 64.7 %：

  | 配置 | 陷阱 | 父题 | 陷阱 − 父题 | A − 配置（陷阱） | A − 配置（父题） | 判定 |
  |---|--:|--:|--:|--:|--:|---|
  | DeepSeek V4 Flash 关 | 56.5 [45.7, 67.4] | 69.6 [57.2, 81.2] | −13.0 [−25.4, −0.7] | −1.6 [−12.7, +9.2] | −4.9 [−15.2, +4.9] | **无可检出差别** |
  | GPT-5.5 关 | 78.3 [67.4, 88.4] | 79.0 [68.1, 89.1] | −0.7 [−12.3, +10.9] | −23.4 [−33.0, −14.3] | −14.3 [−24.6, −4.3] | 仍低于 |
  | GPT-5.4 关 | 71.0 [58.7, 82.6] | 71.7 [59.4, 83.3] | −0.7 [−11.6, +10.1] | −16.1 [−26.4, −6.3] | −7.1 [−16.8, +2.7] | 仍低于 |
  | MiniMax-M3 关 | 14.5 [8.0, 21.7] | 31.9 [21.0, 43.5] | −17.4 [−29.7, −5.1] | +40.4 [+28.6, +52.5] | +32.8 [+21.6, +44.4] | 超过 |
  | DeepSeek V4 Flash 开 | 97.1 [93.5, 100] | 91.3 [84.1, 97.1] | +5.8 [+1.4, +10.9] | −42.2 [−53.6, −31.2] | −26.6 [−39.1, −14.9] | 只报数字 |
  | GPT-5.5 开 | 98.6 [96.4, 100] | 92.8 [85.5, 98.6] | +5.8 [0.0, +13.0] | −43.7 [−55.3, −32.6] | −28.1 [−41.8, −14.3] | 只报数字 |

  基座 9B 对各关思考配置（陷阱）：DeepSeek 关 −21.7 [−36.2, −7.2]、GPT-5.5 关 −43.5、GPT-5.4 关 −36.2、MiniMax +20.3；Qwen3-8B A 臂：−39.7、−61.4、−54.2、+2.4 [−6.5, +11.6]。
  同一框架下 DeepSeek 关在 §20 的 31 道上为 61.3 %（Bench 框架 52.7 %），9B A 臂 60.5 %：§20 的"无可检出差别"在本框架、全部 46 道上复现。
- **S21-b 留出折模板（32 簇）**，9B A 臂留出折越界 58.7 [46.3, 70.6]、原题 93.8 [90.4, 96.8]：

  | 配置 | 越界 | 原题 | 原题 − 越界 | A 留出 − 配置（越界） | A 留出 − 配置（原题） | 判定 |
  |---|--:|--:|--:|--:|--:|---|
  | DeepSeek V4 Flash 关 | 66.5 [54.5, 77.9] | 93.0 [87.4, 97.5] | 26.5 [14.4, 39.0] | −7.8 [−14.2, −1.0] | +0.9 [−4.9, +6.8] | 仍低于 |
  | GPT-5.5 关 | 87.1 [79.5, 93.8] | 99.6 [98.9, 100] | 12.4 [5.8, 20.3] | −28.5 [−38.0, −19.6] | −5.7 [−9.2, −2.8] | 仍低于 |
  | GPT-5.4 关 | 78.6 [68.0, 87.9] | 98.5 [96.9, 99.6] | 19.8 [10.3, 30.3] | −19.9 [−28.5, −12.5] | −4.6 [−8.3, −1.7] | 仍低于 |
  | MiniMax-M3 关 | 26.5 [17.9, 36.0] | 58.7 [48.4, 69.3] | 32.2 [21.3, 43.2] | +32.2 [+22.3, +42.3] | +35.1 [+25.4, +44.8] | 超过 |

  DeepSeek 关在最终集上的越界 66.5 % 与诊断集（另一次抽样）的 65.3 % 一致。
- **S21-c 代价**（textbook 92 行的 completion tokens，中位数 / 均值）：9B A 臂 1,822 / 2,262；DeepSeek 关 552 / 894、开 8,253 / 13,594；GPT-5.5 关 379 / 418、开 1,475 / 2,003；GPT-5.4 关 407 / 439；MiniMax 328 / 418。训练后的 9B 关思考写的程序比前沿模型关思考长 3–5 倍。
- **读法结论**：9B 训练后关思考与 DeepSeek V4 Flash 关思考在 46 道陷阱上无可检出差别（§20 复现），在留出折模板上低 8 点；对 GPT-5.5 / GPT-5.4 关思考在两处都仍低于；开思考的前沿模型在陷阱上 97–99 %。
  **前沿模型自身的越界缺陷**：DeepSeek 关在陷阱上 −13 点、模板上 −26.5 点；GPT-5.5 关在模板上 −12.4 [−5.8, −20.3]，但在陷阱上 −0.7 [−12.3, +10.9]——没有可检出的陷阱缺陷。注意 46 道陷阱的入选见证里 gpt55-reasoning 占 42 道、deepseek-v4-pro-reasoning 39 道（`certification.witnesses`）：陷阱集按 GPT-5.5 推理模式能解来筛过，GPT 家族在其上的数字带选择偏差，模板上的数字没有这一层。
- **正文可写的句子**（每一部分都有数据）：训练把 Qwen3.5-9B 关思考在 46 道教科书陷阱上从 34.8 % 提到 54.9 %，达到 DeepSeek V4 Flash 关思考（56.5 %）的水平，但低于 GPT-5.5 / 5.4 关思考（78 / 71 %）和任何开思考的前沿模型（97–99 %）；父题上 9B 仍低于全部前沿模型。不可写"小模型超过前沿模型"。

## 22. 训练后的 9B 与 D 的 16 个本地开源配置在同一批教科书陷阱上的比较（2026-11-08 追加，写于读任何 D 配置在这些陷阱上的正确率之前；`explore/frontier22_report.py` → `results/paper/frontier22.json`；事后，只报数字加写定读法）

- **为什么**：§21 把"训练后的 9B 关思考达到大模型水平"限定到了 DeepSeek V4 Flash 关思考一个配置。论文的 D 研究已经用**同一个 runner、同样的温度 0.3、同样的 8,192 / 28,672 输出上限**（`explore/d_launch.py` 写入 `models_v2.toml` 的 `d-v2-*` 节）把 16 个本地开源配置——Qwen3-8B、Qwen3.5-9B、Qwen3.5-27B、Gemma 4 E4B、Gemma 4 31B、Granite 4.2 8B、Granite 4.2 30B 各关 / 开思考，Ministral 3 8B / 14B 关——在 `experiments/_inputs/families_v4/textbook.json`（398 行）上各跑了三次。那 398 行里有 46 道 regime 陷阱中的 **38 道**（题面与 `sub_answers` 与 X 的 `textbook.json` 逐字相同）和全部 46 道父题。所以"训练后的 9B 关思考，比它大 3 倍的开源模型（含开思考）是否仍在它之上"可以不跑任何新实验、在同一框架下回答。不在 D 集里的 8 道：`trap_3.4_regime`、`trap_5.7_regime`、`trap_7.6_regime`、`trap_air_202_regime`、`trap_dn_4.4_regime`、`trap_p_40_regime`、`trap_p_55_regime`、`trap_ry_7.7_regime`。
- **已看到的**：D-1 / D-2 的全部已入库数（`results/d_execution/summary.md`、`results/d2/`）：每个配置的模板对照（原题 − 越界等，簇层面）、教科书"regime − 几何"配对差（38 道 regime 对 72 道几何，三次重复）、各家族的父题正确率与条件识别率，以及 `families_report_*.json` 里每个配置 32 个簇的 inside / outside / given / flag 正确率；§21 的全部结果；`results/traps_rl/` 里 9B 各臂与基座在 46 道上的逐陷阱正确率。**没看过的**：任何 D 配置在这 38 道 regime 陷阱上的绝对正确率、在这 38 道父题上的正确率，以及 9B A 臂限定到这 38 道的水平。本节在读这些数之前写定。
- **数据与取舍规则（事先定）**：从服务器 `$ATMOS_ROOT/experiments/d_*/outputs/families_context_v2/<配置>.textbook.run{1,2,3}.json` 拷回（sha256 逐文件核对，存 `experiments/d_textbook_eval/`）。一个配置若在两个启动组里都有文件（`qwen3-8b.think-on`：`d_bridge_ready_20261010` 与 `d_qwen3_8b_on_restart`；`qwen3.5-27b.think-off` run1：`d_qwen35_27b_dp4_eager` 与 `d_qwen35_27b_114_20261011`），只用三次重复都满 398 行的那一组；两组都满时用重启组（restart / 后启动的），并在结果记录里报另一组在 38 道上的数作为核对；`d_qwen35_27b_114_20261011` 是 `synthetic_probe`，不用。`qwen3.5-9b.think-off` 的文件在 `d_context_v2_20261010`。每条记录以 runner 存下的 `passed` 为准，API 错误行（`attempts` 为空）不进分母并报个数；另外把每条最终答复用 `explore.x_report.regrade_response`（与 §21 核对 frontier 用的同一函数，runner 的 5 % / benchmark 单位规则）重判一遍，报与存储判定的不一致数；若有不一致，以重判为准并同样重判 9B 各终点与 §21 的 frontier 记录，使全部配置同一判卷器。
- **口径**：
  - S22-a 陷阱与父题：单位是陷阱（38；父题与其陷阱配对）。D 配置逐陷阱正确率 = 三次重复的均值；9B A 臂 = `results/traps_rl/report_qwen35-9b.json` `arms.A.per_trap` 限定到 38 道（四个终点均值，不重算），基座 9B、Qwen3-8B A 臂同理；§21 的六个前沿配置也限定到这 38 道并列入同一张表（只报数字）。每个 D 配置报：陷阱水平、父题水平、陷阱 − 父题、配对差（A − 配置）在陷阱与父题上各一个；区间为按陷阱重抽的 95 % 百分位 bootstrap，10,000 次，`random.Random("paper:frontier22:<项>:20261108")`。
  - S22-b 模板（只报数字，带口径差异）：单位是簇（32）。D 配置的簇层面越界 / 原题正确率直接取 `results/d2/families_report_*.json`（E4B、Qwen3.5-9B 关、Qwen3-8B 关取 `experiments/d_execution_review/*.json`）的 `templates.all.clusters`——那是 D 自己的参数抽样（`families_v4/templates.json`，72 模板，一次运行），不是最终集的行；9B A 臂留出折的簇层面值与 §21 相同（`frontier21_report.heldout_9b`）。报配对差（A 留出 − 配置）在越界与原题上各一个，区间同上。因为两侧的行不同、模板数 72 对 70，这一部分不作判定。
  - 锚点：D 的 `qwen3.5-9b.think-off` 与 X 的基座 `x-qwen35-9b-base` 是同一个模型、同样设置的两次独立运行，两者在 38 道上的陷阱与父题正确率并列只报；从拷回文件重算的 `qwen3.5-27b.think-off` 全部父题正确率须等于 `families_report_qwen3.5-27b.json` 的 `parent_accuracy_unique_parents`（0.8074），regime 对数 114 = 38 × 3——证明拷回的文件就是 D 报告用的文件。
- **读法（写定；只对 S22-a 的陷阱差，16 个配置逐个判，关思考与开思考都判，不合并、不挑选）**：区间下界 > 0 → 该配置上"训练后的 9B（关思考）超过"；区间含 0 → "无可检出差别"；上界 < 0 → "仍低于"。正文凡写到此比较，必须同时给出：同一配置的父题差、9B 是在同类机制的模板上训练过而被比较的模型没有、这是 46 道里的 38 道、以及 §21 的前沿结果（GPT-5.5 / 5.4 关思考与全部开思考前沿模型仍在 9B 之上）。"小模型超过大模型"只能写到被比较配置的名字和规模，不能写成对前沿模型成立。

### 22 结果记录（2026-11-08；`results/paper/frontier22.json`，本地两次运行逐字节相同，sha256 前缀 6392878d；运行数据 `experiments/d_textbook_eval/`，服务器哈希 `MANIFEST.server.sha256`）

- **完整性与锚点**：16 个配置各三次 398/398 行，0 个 API 错误行；20,000 余条答复重判与存储判定 0 条不一致（判卷器与 §21 相同）；所选文件与 D-2 报告所用的 48 个文件逐一 sha256 相同，取舍与写定规则一致；Qwen3.5-27B 关唯一父题正确率重算 0.80737 与已提交报告逐位相同，regime 对数 114。D 的 Qwen3.5-9B 关（三次）在 38 道上 42.1，X 的同一基座单次运行 36.8，只报。
- **S22-a（38 道共有陷阱）**，9B A 臂陷阱 59.9、父题 67.1：

  | 配置 | 陷阱 | 父题 | A − 配置（陷阱） | 判定 |
  |---|--:|--:|--:|---|
  | Qwen3.5-27B 关 | 69.3 | 74.6 | −9.4 [−22.4, +3.1] | 无可检出差别 |
  | Gemma 4 31B 关 | 71.1 | 77.2 | −11.2 [−24.6, +2.4] | 无可检出差别 |
  | Granite 4.2 30B 开 | 54.4 | 64.9 | +5.5 [−7.5, +18.0] | 无可检出差别 |
  | Qwen3.5-27B 开 | 73.7 | 81.6 | −13.8 [−25.2, −1.8] | 仍低于 |
  | Gemma 4 31B 开 | 93.9 | 89.5 | −34.0 [−44.5, −23.9] | 仍低于 |
  | Granite 4.2 30B 关 | 17.5 | 33.3 | +42.3 [+28.7, +55.5] | 超过 |
  | Qwen3.5-9B 关 / 开 | 42.1 / 49.1 | 52.6 / 63.2 | +17.8 [+6.1, +29.6] / +10.7 [+1.5, +19.7] | 超过 / 超过 |
  | Granite 4.2 8B 关 / 开 | 12.3 / 43.0 | 25.4 / 55.3 | +47.6 / +16.9 [+2.2, +32.0] | 超过 / 超过 |
  | Qwen3-8B 关 / 开 | 8.8 / 28.9 | 27.2 / 54.4 | +51.1 / +30.9 | 超过 / 超过 |
  | Gemma 4 E4B 关 / 开 | 21.9 / 25.4 | 41.2 / 41.2 | +37.9 / +34.4 | 超过 / 超过 |
  | Ministral 3 8B / 14B 关 | 14.9 / 24.6 | 31.6 / 37.7 | +45.0 / +35.3 | 超过 / 超过 |

  同 38 道上的 §21 前沿参照：DeepSeek Flash 关 61.4（−1.5 [−12.9, +10.3]）、GPT-5.4 关 75.4、GPT-5.5 关 80.7、开思考 96.5–98.2。
- **S22-b（簇层面，D 自己的抽样，只报数字）**：9B 留出折越界 58.7 对 Qwen3.5-27B 关 67.2（−8.5 [−14.5, −3.2]）、Gemma 31B 关 64.8（−6.1 [−12.3, −0.4]）、Granite 30B 开 50.4（+8.3 [+0.8, +15.7]）、27B 开 71.7（−13.0）、31B 开 80.9（−22.2）；其余配置都在 9B 之下。
- **读法结论**：训练后的 9B 关思考与 Qwen3.5-27B、Gemma 4 31B 关思考无可检出差别（点估计低 9–11 点，38 道的功效所限，正文写 on par / not distinguishable，不写 outperforms），超过全部 ≤ 14B 的开源配置与 Granite 4.2 30B 两种模式，仍低于 27B / 31B 开思考与 GPT-5.5 / 5.4 关思考。Granite 30B 关的 17.5 是该模型在本框架下自身偏弱（父题 33），不作卖点。正文可写：after training, Qwen3.5-9B with thinking off performs on par with open models three times its size (Qwen3.5-27B, Gemma 4 31B) and with DeepSeek V4 Flash, all with thinking off, on textbook regime problems.

## 23. 机制实例值与"在每个本地配置都失败"的机制清单（2026-11-08 追加，写于计算之前；`explore/paper_mechanism_examples.py` → `results/paper/mechanism_examples.json`；描述性，只报数字）

- **为什么**：稿件要给领域读者两样具体的东西：(i) 哪些机制在每个本地配置上都失败，连同它们的物理内容；(ii) 每个机制一个具体越界实例，标准公式给的数与正确的数并列，让读者看到错多大。两者都不需要新的模型调用。
- **已看到的**：`results/paper/atlas.json` `per_template` 的 `universality_local` / `configs_kept_local` / `deepseek_off`（§13 已入库，`explore/paper_atlas.py` 的 `FAIL = 0.5`、`MIN_KEPT = 10`）；`results/paper/round4_families.json` `S16a.zero_in_all`（三个机制：ch4_forcing_sqrt、clausius_clapeyron_extrapolation、co2_forcing_log）；`results/paper/cost.json` `templates.per_template.F_max`（§12，最终集 20 行的最大比值，70 模板）。本节不读任何模型输出，只读模板代码和上述已入库文件。
- **口径**：
  - S23-a 全失败清单：模板 t 入选当且仅当 `configs_kept_local ≥ 10` 且 `universality_local == configs_kept_local`，即在每个能定义"原题对时越界也对"概率的本地配置上，该概率 ≤ 0.5（与 `paper_atlas.py` 打印的 "universal failures" 同一规则）。报个数、名单、每个的 `configs_kept_local` 与 `deepseek_off`。不排序以外不作解读；正文点名时须写明"在每个能定义该指标的本地配置上"。
  - S23-b 代表实例：对 `ALL_TEMPLATES` 的 72 个模板各取一对参数，`templates.base.pair(t, random.Random(f"paper:examples:{name}:20261108"))`（模板自己的可采纳性规则：inside 差 ≤ 2 %，outside 差 ≥ 15 %）。报 outside 参数、`shortcut(p_out)`、`correct(p_out)`、单位、比值 F = max(s/c, c/s) 与相对偏差 |s − c|/|c|；并列 cost.json 的 `F_max`（70 模板；两个训练前排除的模板无此值）。
  - S23-c 锚点实例（参数写定，用模板函数直接算，不抽样）：`co2_forcing_log` 取 dc = 280 ppm（CO₂ 加倍，模板 docstring 的锚点）；`line_broadening` 取 p = 0.4 hPa 与 0.1 hPa（越界范围两端）；`sun_photometer_airmass` 取 θ = 87°（越界范围上端）。报 shortcut、correct、F。
- **读法**：无。只报数字与清单；正文使用的句子限于"标准公式给 X，正确值 Y"和"在每个能定义该指标的本地配置上都失败的机制有 N 个：……"。
- **程序**：`--selftest`（锚点实例对模板 docstring 的已知值：CO₂ 加倍 3.71 W m⁻² 对线性 5.35；line_broadening 0.1–0.4 hPa 的比值在 docstring 所述 19–85 之间；清单规则对一组合成的 per_template 数据）；两次运行逐字节相同后入库。

### 23 结果记录（2026-11-08；`results/paper/mechanism_examples.json`，本地两次运行逐字节相同，sha256 前缀 60daada4；selftest 通过）

- **S23-a**：7 个模板在每个能定义该指标的本地配置上都失败（保留配置数 / DeepSeek 关思考的条件越界正确率）：ch4_forcing_sqrt（16 / 0）、clausius_clapeyron_extrapolation（16 / 0）、drop_ventilation（13 / 0）、ice_capacitance_shape（10 / 0.29）、line_broadening（16 / 0）、sonic_humidity_flux（16 / 0）、sun_photometer_airmass（14 / 0）。与 `paper_atlas.py` 打印的清单一致。
- **S23-b**：72 个模板各一个实例已入库；`line_broadening` 的抽样实例（p = 0.2 hPa）比值 43，高于最终集 20 行的 F_max 29（两者并列报）。
- **S23-c 锚点**：CO₂ 加倍（dc = 280 ppm）线性 5.348 对对数 3.707 W m⁻²（比值 1.44）；line_broadening p = 0.4 / 0.1 hPa 比值 21.3 / 83.7；sun_photometer_airmass θ = 87° 平面平行大气质量给光学厚度 0.178 对 Kasten–Young 0.224（低 21 %）。
- 正文用法：§1 点名 7 个机制与 3 个全模型零采样机制；CO₂ 加倍与 θ = 87° 两个锚点数进正文；Table 1 加"越界实例：标准值 / 正确值"一列（S23-b 的抽样实例）；Supplementary Table 1 以 ‡ 标出 7 个全失败机制。稿件此前写 line_broadening "computed at 1 hPa" 有误，模板越界范围为 0.1–0.4 hPa，随本次一并改正。

## 24. 外部教师探针的汇总（2026-11-08 追加，写于计算之前；`explore/paper_teacher_probe.py` → `results/paper/teacher_probe.json`；只报数字）

- **为什么**：PREREG_OPD §4.2 规定，外部大教师臂 T 只在探针过门槛时才跑：候选教师在 14 个 Z 机制（Qwen3-8B 基座 pass@64 = 0）的越界题上 pass@1 ≥ 0.5 的机制数 ≥ 6。探针已跑、结果记在 PREREG_OPD 的执行记录里，T 臂按规则未跑。稿件要在 Methods 写明这一点并引用探针数字，按仓库规则数字须来自 `results/`，所以把探针汇总进结果文件。
- **已看到的**：PREREG_OPD 执行记录中的汇总（Qwen3-32B 70 模板均值 pass@1 0.214，Z 机制上 ≥ 0.5 为 0 个，最高 0.28；Qwen3-30B-A3B 均值 0.314，Z 机制上 ≥ 0.5 为 1 个）。本节只是把这些数从原始探针文件重算并入库，不产生新的读法。
- **数据**：`experiments/_opd/teachers/passk_outside_n8.{qwen3-32b,qwen3-30b-a3b-2507}.jsonl`，每行一道越界题（70 模板 × 20 对），字段 `n`（8 次采样）、`c`（正确次数）。Z 机制取 `results/x/report_qwen3-8b.json` 的 `P1.Z`。
- **口径**：逐模板 pass@1 = 该模板 20 行 c/n 的均值；全模板均值 = 70 个模板 pass@1 的均值；Z 机制上 ≥ 0.5 的个数与名单；Z 机制上的最大 pass@1；门槛（≥ 6）是否通过。与 PREREG_OPD §4.2 的定义一致，不加其它统计量。
- **读法**：无；只报数字。正文用法限于 Methods 的一句事实陈述（两个候选教师各有几个 Z 机制过 0.5、T 臂因此未跑）和讨论里"同家族更大的模型也不会这些机制"的半句。
- **程序**：`--selftest`（合成数据核对 pass@1、门槛判定与 Z 过滤）；两次运行逐字节相同后入库。

### 24 结果记录（2026-11-08；`results/paper/teacher_probe.json`，本地两次运行逐字节相同，sha256 前缀 60709e2f；selftest 通过）

- Qwen3-32B：1,400 行、70 模板；全模板均值 pass@1 0.211；14 个 Z 机制上 ≥ 0.5 的为 0 个，最高 0.28（snowpack_cold_content）。
- Qwen3-30B-A3B-Instruct-2507：均值 0.314；Z 机制上 ≥ 0.5 的为 1 个（displacement_height_canopy 0.61），其次 snowpack_cold_content 0.48。
- 两者都未过门槛（≥ 6），与 PREREG_OPD 执行记录一致。一处口径差：执行记录写 Qwen3-32B 均值 0.214，本节按"先逐模板取均值、再对 70 模板取均值"得 0.211；执行记录当时很可能是对 1,400 行直接取均值。Z 机制上的个数与最大值两处记录完全一致，稿件只引用这两项。

## 23. 冻结后的补充运行：独立陷阱上的迁移、比较模型补齐、OPD 控制臂第二种子、27B 可达性探针（2026-11-08 追加，写于任何一次本节运行之前；用户 2026-11-08 解除 11-01 的实验冻结，授权本节四项；事后，读法事先写定，结果不论方向全部报告）

- **为什么**：审稿最可能追问的四处——(i) 训练增益是否只在 46 道 X 陷阱上（与训练模板同批机制?），在一批独立的 regime 陷阱上是否重复、是否只是一般性变好；(ii) §22 的开源对照只覆盖 38 道、模板对照用的是不同抽样；(iii) OPD 的 P − S、P − Pshuf 只有单种子的对照臂；(iv) 训练的模型只有 8–9B，可达性屏障在更大模型上是否仍在。
- **已看到的**：§1–§22 全部结果；D-5 的 16 个配置在 207 行上的结果（`results/d5/`）。**没看过**：任何训练过的模型（X / DOSE / OPD 终点、热身模型）在 D-5 集上的任何回答；D 的配置在 8 道缺失陷阱与最终集上的回答；OPD S / Pshuf 种子 43；Qwen3.5-27B 在最终集越界行上的任何采样。
- **E1 独立陷阱上的迁移**：D-5 集（`experiments/_inputs/d5/textbook.json`，207 行，sha256 `cc3d59e2…`：54 道 regime 陷阱、56 道几何陷阱、97 个母题；与 X 的 46 道 regime 陷阱无共同陷阱，母题共 8 个）。对服务器 `_x_eval/` 下有评测记录、且所用检查点仍在本机的全部 Qwen 标签（X 两家族基座、热身、各臂终点含保留的种子 43；DOSE 热身与 RL；DOSE-text；OPD 全部终点），用 X 的评测框架（`explore/f23_eval_queue.sh`：x_eval.sh 的服务参数与 runner 设置，关思考，每行 1 次）评一次，输出 `experiments/_x_eval_d5/<标签>/textbook_d5.json`。检查点已删或不在 113 / 114 的标签记为缺失并列名，不补训。Ministral 标签不评（权重在 047，服务参数不同）。
  - **口径**（`explore/f23_report.py`）：单位是陷阱。臂的逐陷阱正确率 = 现存终点（折 × 种子）的平均；起点：A / D / F / R 为基座，WA / WD 为两折热身的平均。S23-a：regime 陷阱上臂 − 起点；S23-b：(regime 增益) − (几何增益)，两族各自独立重抽（同 D-5）；母题增益只报。区间 10,000 次百分位 bootstrap，`random.Random("paper:f23:<项>:20261108")`。
  - **读法（只对两家族的 A 臂，各自写）**：S23-a 下界 > 0 →"增益在独立的 regime 陷阱上重复"；含 0 →"独立陷阱上未检出增益"；上界 < 0 →"下降"。S23-b 下界 > 0 →"增益集中在 regime 陷阱"；含 0 →"与几何陷阱的增益无可检出差别（一般性提升）"；上界 < 0 →"几何陷阱增益更大"。其余臂、DOSE、OPD 只报数字（同 F11 / §17 的对比：DOSE RL − WA、OPD 各臂 − A）。
  - **污染核对**：W 起点臂与 DOSE 热身的 SFT 数据含 RAFT 背景行（`$ATMOS_ROOT/data/sft_raft.parquet`，取自 benchmark 训练侧），D-5 的母题都在训练侧。运行前统计 RAFT 背景里出现的 D-5 母题与陷阱 id；这些臂另报"去掉母题在 RAFT 背景中的陷阱"的数字。A / D / F / R 臂的训练数据只有模板行，不受影响。
- **E2 比较模型补齐**：D 的 Qwen3.5-27B、Gemma 4 31B 关 / 开思考四个配置，用 D 自己的服务参数与 runner（`explore/f23_d_serve.sh`，`results/d5/serving.json`、`models_v2.toml` d5-* 节），在 (a) X 教科书集里 D 集没有的 8 道陷阱与其 8 个母题（`explore/build_f23_inputs.py` → `experiments/_inputs/f23/textbook_missing.json`，sha256 `2b400d77…`）上各跑三次，(b) 关思考两个配置在最终集 inside / outside 2,800 行（`final_think.json`，与 §21 前沿、9B 留出折同一批行）上各一次。输出 `experiments/families_f23/families_context_v2/`。
  - **口径与读法**：(a) 与 D 的 38 道合并成 46 道，S22-a 的口径与三档读法原样重算这四个配置（46 道代替 38 道，结果另存，不改 §22 的记录）；(b) 簇层面越界 / 原题正确率与 9B A 臂留出折的配对差，同 §21 S21-b 的口径与三档读法。其余 12 个 D 配置不补跑（权重在 047 或不在被比较之列）。
- **E3 OPD 控制臂第二种子**：S 与 Pshuf 两折各加种子 43（`train/opd/opd_chain.sh`，RUNS = S / Pshuf × F1 / F2 × 43；链门槛、合并与评测同 OPD-8 / OPD-13），F1 在 114、F2 在 113。报告 `explore/opd_report_ext.py` 两种调用：原判定口径（种子 42，不改 OPD-16 的判定），与 S / Pshuf / P 都取两种子机制内平均的口径（只报数字）。读法：两种子平均的 P − S、P − Pshuf 在主检验集上下界 > 0 → "H-OPD2 / H-OPD6 在对照臂第二种子上复现"；否则写"未复现"并给数字。PREREG_OPD 以 OPD-18 指向本节。
- **E4 27B 可达性探针**：Qwen3.5-27B 基座关思考，最终集越界行每行 64 次、T 1.0（`explore/x_passk.sh`，X-18 的设置，TP 4），输出 `experiments/_x/qwen35-27b/passk_final_outside.x18.jsonl`。只报数字：零命中机制数、与 Qwen3.5-9B / Qwen3-8B 基座 pass@64 的 Spearman（按机制重抽）、两者都零命中的机制。是否用它训练 27B（约一周 GPU）由用户看过探针后决定，本节不启动。
- **顺序与资源**：113 / 114 各 8 卡。先 E1（两台）与 E2（114 GPU 4–7）、E4（114 GPU 0–3，E1 在 114 的队列之后）；GPU 空出后 E3 两条链。全部只用已提交脚本，在服务器的独立 worktree（本节提交）中运行，不更新共享检出。

### 23 执行记录 F23-1（2026-11-08 20:55；写于读任何 §23 结果之前）

- **资源**：113 拒绝本机密钥（`Permission denied (publickey,password)`，20:40 起反复重试），后台每 10 分钟重试；114 的 GPU 0–3 被用户自己的服务占用（`/path/to/local/serve`，Qwen3.8-27B，11-03 起），不碰。本节只有 114 的 GPU 4–7 可用。
- **代码**：§23 提交 68717cd 与链脚本 5b26a3c 以 git bundle 传到服务器，在 `$ATMOS_ROOT/scratch/rl-wt-5b26a3c`（独立 worktree）运行，共享检出的工作区未动。输入哈希与第 23 节一致（D-5 `cc3d59e2…`、缺失 16 行 `2b400d77…`、最终集 `576a2c86…`）。
- **E1 的标签**：114 上检查点仍在的只有 Qwen3-8B 基座、热身 F1、A / D / F / R / WA / WD 第一折种子 42，Qwen3.5-9B 基座、热身 F2、R 第二折，OPD P 第一折种子 42；其余（Qwen3.5-9B 的 A / D / F / WA / WD、Qwen3-8B 第二折与种子 43、DOSE、DOSE-text、其余 OPD）的检查点在 113 的本地盘或已删，113 恢复前缺失。因此 E1 的写定读法目前只能对 **Qwen3-8B 的 A 臂（单折单种子终点）** 作出；Qwen3.5-9B 的 A 臂记为"缺数据"，113 恢复后补评再判。
- **RAFT 背景核对**（`data/sft_raft.parquet` sha256 前缀 8d3ab3b6，1,060 行；母题 id 或其数值变体 `<母题>_vN` 计入）：D-5 的 54 道 regime 陷阱中 33 道、56 道几何陷阱中 32 道的母题出现在 RAFT 背景里；陷阱本身 0 道。X 的 46 道陷阱 0 道。WA / WD 与热身模型按写定口径另报去掉这些陷阱的数字。
- **E3**：OPD 需要 4 + 1 张卡做训练、另 3 张评测，114 只有 4 张空卡，113 不可达，暂不能跑；113 恢复后按第 23 节启动。
- **链**：`explore/f23_chain.sh`，GPUS 4,5,6,7，顺序 E1（12 个标签）→ E2（27B / 31B 关思考：缺失 16 行 × 3、最终集 × 1；开思考：缺失 16 行 × 3）→ E4。日志 `$ATMOS_ROOT/logs/f23/chain-host-D.log`。

### 23 结果记录 F23-2（2026-11-09 05:15；E1、E2 完成，E3 未跑，E4 运行中）

- **执行**：链 21:26 起在 114 GPU 4–7 运行（`rl-wt-8226e34`；第一次启动时 vLLM 服务 27 分钟才起来，被队列的 30 分钟上限杀掉，改为 3 小时上限后重启，没有任何评测结果受影响）。
  E1 12 个标签全部 207/207、0 个 API 错误行；E2 Qwen3.5-27B 关 / 开在缺失 16 行上各三次 16/16，关思考最终集 2,800/2,800。Gemma 4 31B 的权重已不在 114（`/path/to/local/models/` 只剩 E4B 与 Ministral），
  该模型受访问限制、服务器无 Hugging Face 登录，无法按 D 固定的版本 842da37 重新下载，Gemma 两个配置的 E2 未跑。113 整夜拒绝密钥，E3 未跑。
  报告各两次运行逐字节相同：`results/paper/f23_e1.json`（sha256 前缀 9463c369）、`f23_e2.json`（d776d56e）。
- **E1（D-5 集，单位陷阱；Qwen3-8B 只有第一折种子 42 的终点）**：

  | 臂 − 起点 | regime（54） | 几何（56） | regime − 几何 | 母题 |
  |---|--:|--:|--:|--:|
  | A − 基座（9.3 → 29.6） | +20.4 [+7.4, +33.3] | +21.4 [+10.7, +33.9] | −1.1 [−19.0, +17.0] | +40.2 [+30.9, +50.5] |
  | D − 基座 | +24.1 [+13.0, +35.2] | +17.9 [+5.4, +30.4] | +6.2 [−10.3, +22.7] | +38.1 |
  | F − 基座 | +18.5 [+7.4, +29.6] | +14.3 [+3.6, +26.8] | +4.2 [−12.2, +20.6] | +36.1 |
  | R（随机奖励）− 基座 | −5.6 [−14.8, +1.9] | +1.8 | −7.3 | −8.2 |
  | WA − 热身 / WD − 热身 | +18.5 / +16.7 | +16.1 / +12.5 | +2.4 / +4.2 | +38.1 / +35.1 |
  | 热身 − 基座 | −3.7 | +1.8 | −5.5 | +8.2 |
  | OPD P − 基座 | +5.6 [−1.9, +14.8] | +1.8 | +3.8 | +2.1 |

  Qwen3.5-9B 只有基座、热身与 R 臂（A 臂在 113），R − 基座 regime −7.4、热身 − 基座 0.0。去掉母题在 RAFT 背景中的陷阱后，Qwen3-8B 起点与 WA / WD 在剩余 21 道 regime 陷阱上都是 0–5 %：
  RAFT 背景就是基座自己做对的题，母题在其中的恰是较容易的陷阱，这一核对没有发现污染迹象，只说明剩下的是难题。
  **读法（写定）**：Qwen3-8B A 臂 S23-a "增益在独立的 regime 陷阱上重复"，S23-b "与几何陷阱的增益无可检出差别（一般性提升）"；Qwen3.5-9B A 臂"缺数据"。
  即：模板训练让 Qwen3-8B 在独立的教科书陷阱上提高约 20 点，但几何陷阱同样提高、母题提高 40 点，增益是在这批教科书题上的普遍提升，不是 regime 判断的专门提升；随机奖励不提高。
- **E2（46 道，S22-a 口径；最终集 32 簇，S21-b 口径）**，9B A 臂陷阱 54.9 %：

  | 配置 | 陷阱 | 母题 | A − 配置（陷阱） | 判定 | 模板越界：9B 留出 − 配置 | 判定 |
  |---|--:|--:|--:|---|--:|---|
  | Qwen3.5-27B 关 | 67.4 | 72.8 | −12.5 [−24.5, −0.9] | 仍低于 | −10.2 [−16.1, −4.7]（27B 越界 68.9、原题 97.8） | 仍低于 |
  | Qwen3.5-27B 开 | 73.9 | 79.7 | −19.0 [−31.0, −7.2] | 仍低于 | — | — |

  §22 在 38 道上的"无可检出差别"（−9.4 [−22.4, +3.1]）补齐到 46 道后变为"仍低于"：补上的 8 道中 27B 做对得多。**正文"与三倍大的开源模型相当"对 Qwen3.5-27B 不再成立**；Gemma 4 31B 只有 38 道的结果（−11.2 [−24.6, +2.4]），无法补齐。
  与 DeepSeek V4 Flash 关思考的相当（46 道，−1.6 [−12.7, +9.2]）不受影响。
- **E4**：05:00 时 1,400 行中 33 行（脚本固定 4 路并发，TP 4），预计约 17 小时完成；完成后另记。

## 25. 审稿防线四项：延长且集中的 RL、低可达机制上偶发正确的功能检验、捷径错误的物理量级、差距闭合比例（2026-11-09 追加，写于本节任何计算与运行之前；用户 2026-11-09 授权；事后，读法事先写定，结果不论方向全部报告）

- **为什么**：(i) ProRL（NeurIPS 2025）与 arXiv 2606.15455 主张"RL 训练够久就能越过基座的可达边界"，P1 的"pass@64 = 0 的机制学不会"可被质疑为训练太短（A 臂 120 步、每题 8 个样本）；
  (ii) arXiv 2510.08325 指出数值题上的 pass@k 会把蒙对算成可达，P1 / DOSE 的可达性度量（pass@1、pass@64）若被偶然命中抬高，"可达程度决定 RL 成败"就不干净；
  (iii) 领域读者问的是错了多少——捷径错误在物理量上有多大；(iv) "与三倍大的模型相当"在 46 道上不成立（F23-2），改报"训练补上了与各比较模型差距的多少"。
- **已看到的**：§1–§24 与 F23-1 / F23-2 的全部结果；两家族 `P1.per_mechanism`（基座 pass@k、A 臂两种子终点越界正确率）。**没看过**：基座 pass@64 采样文本里任何一个正确样本的代码；
  任何陷阱的捷径值与正确值之比的分布；任何"差距闭合比例"；任何超过 120 步或每题多于 8 个样本的 A 臂训练。

### 25-A 延长且集中的 RL（E5，GPU；Qwen3-8B 第一折种子 42）

- **目标机制**：Qwen3-8B 第一折训练模板里，基座 pass@64 ≤ 0.1 且 A 臂两种子终点越界正确率都 < 0.5 的 8 个：diffuse_transmission、displacement_height_canopy、freezing_point_depth、line_broadening、
  saturated_lapse（以上 pass@64 = 0）、henry_depletion、unconfined_dupuit、surface_multiple_reflection（pass@64 0.05–0.1）。名单由 `results/x/report_qwen3-8b.json` 机械取出，写在此处后不改。
- **起点**：A 臂第一折种子 42 的第 120 步合并模型（114 本地 `/path/to/local/x/merged/x-qwen3-8b-F1-A-s42/global_step_120`）——即"在标准 RL 之后继续训练"。
- **题库**：`data-x/F1-A/train.parquet` 中这 8 个模板的全部行（每模板 inside / outside / given 各 20，共 480 行，与 A 臂题库中这些行逐字节相同；`explore/f25_data.py` 过滤，不改任何行）。
  验证集：同目录 `train_probe.parquet` 中这 8 个模板的行（76 行），第 0 步与每 20 步、T 0.3，只画曲线，不用于选步数。
- **设置**：与 A 臂相同（`train/dapo_qwen3_8b.sh`：DAPO clip 0.2 / 0.28、lr 1e-6、无 KL、无熵奖励、关思考、回答上限 3,072、奖励 regime 文件同 A 臂），只改四处：
  每步 32 题 × **32 个样本**（A 臂 128 × 8，每步样本数相同，每题样本数 4 倍）、mini-batch 8（每步 4 次更新，同 A 臂）、**关闭动态采样过滤**（全对或全错的题组优势为 0、不产生梯度，
  只是不再为凑批重采；集中题库下开着过滤会因凑不满批而停滞）、4 卡（114 GPU 4–7）。最多 **200 步**或墙钟 **36 小时**，先到为准；每 20 步存档，保留最近 1 个；一次基础设施崩溃后同种子续跑一次。
  按此设置，每个目标行在延长训练中被采样约 6,400 × 32 / 480 ≈ 430 次，约为 A 臂（120 × 128 × 8 / 2,880 ≈ 43 次）的 10 倍。
- **冒烟门槛**（开跑前，同设置 2 步）：训练正常结束、rollout 与 actor 的 log-prob 一致（`training/rollout_actor_probs_pearson` ≥ 0.95 若有记录）、奖励在 [0, 1]、无沙箱错误风暴；任一不过则修正后重做冒烟，不开正式运行。
- **评测**：(a) 终点合并模型在 `final.json` 这 8 个模板的全部 640 行（inside / outside / given / flag 各 160）上，用 X 的评测框架（`x_eval.sh` 的服务参数与 runner 设置，T 0.3，每行 1 次）；
  (b) 终点与起点（A 臂第 120 步）在这 8 个模板的 160 个 outside 行上各做 pass@64（`explore/x_passk.sh` 同设置，T 1.0，输入为过滤后的 160 行）。
- **主统计量**：8 个目标机制中终点 outside 正确率 ≥ 0.5（"学会"，同 F7）的个数。**读法（写定）**：0–1 个 →"约十倍的 RL 采样集中在这些机制上仍学不会：P1 的边界不是训练时长造成的"；
  ≥ 4 个 →"延长、集中的 RL 学会了标准 RL 学不会的机制：可达性边界部分是训练量的产物"（正文第三幕需改写）；2–3 个 → 只报数字。
- **次要（只报数字）**：pass@64 = 0 的目标机制个数（基座 → 起点 → 终点）；终点 inside 正确率相对起点的变化；验证曲线上目标 outside 行的正确率随步数。

### 25-B 低可达机制上的偶发正确：功能检验（CPU）

- **数据**：两家族基座 pass@64 的采样文本（`passk_final_outside.x18.texts.jsonl`，X-18，outside 行，每行 64 个样本）与逐样本判定（同名 `.jsonl` 的 `passed`）。
- **组**：按机制的基座 pass@1——**稀有组** 0 < pass@1 < 0.02（DOSE 的 0.02 阈值以下且至少命中一次），**常见组** pass@1 ≥ 0.1。稀有组取全部正确样本；
  常见组每行至多 3 个正确样本，按 `random.Random("paper:f25:common:<行 id>")` 抽取。
- **功能检验**（`explore/f25_report.py luck`）：从样本代码（`engine.extract_code`）的 `solve()` 签名读出参数默认值（字面量与常数算式），与该行模板参数 `params` 按值配对
  （默认值 = 参数值 × 10^k，k 为整数，相对误差 ≤ 1e-6，且一一对应）。从该模板自己的 `base` 与 `trigger(…, "outside", …)` 抽 3 组新的越界参数（种子 `random.Random("paper:f25:luck:<行 id>:<样本序号>")`），
  只替换配对上的参数（未配对的保持原值），以同样的 10^k 换算成样本代码的参数，在沙箱（`sandbox.run_solver_sandboxed`）中调用样本的 `solve()`，用评分器（`grading.verify`，5 %、严格单位）对照模板 `correct` 的新值。
  三组都对 →"功能正确"；任一组不对（含报错）→"巧合"；模板的 trigger 参数有未配对的 →"不可检验"。
- **报**：每组、每家族：样本数、可检验比例、功能正确比例（按机制重抽的区间，10,000 次，`random.Random("paper:f25:<项>:20261109")`）；稀有 − 常见之差及区间；
  只算功能正确样本时 pass@64 > 0 的机制数变化（两家族），与 P1 的 Fisher 检验与 AUC 是否改变方向。
- **读法（写定，按家族各自）**：稀有组功能正确比例 ≥ 0.8 且（稀有 − 常见）下界 > −0.15 →"低可达机制上的偶发正确是真解，不是数值巧合：pass@k 没有高估可达性"；
  稀有组 < 0.5 →"偶发正确多为巧合，可达性须按功能正确重算"（并给重算结果）；其余只报数字。Qwen3.5-27B 的采样（§23 E4）完成后同法附报，只报数字。

### 25-C 捷径错误的物理量级（CPU，描述性）

- **陷阱层面**：X 的 46 道 regime 陷阱（`experiments/_inputs/x_eval/textbook.json`），每道的捷径值（`shortcut_values`）与正确值（`sub_answers`，同单位）之比 f = 捷径 / 正确（逐子问，取 |log f| 最大的子问）。
  报 |log10 f| 的中位数与四分位，倍数 ≥ 1.5、≥ 2、≥ 10 的陷阱数，符号相反的陷阱数；按类别列出，并给每类一个带单位的实例（数值直接取自数据文件）。
- **模型层面**：两家族基座与 A 臂终点（`experiments/x_textbook/eval`）、§21 的六个前沿配置（`experiments/frontier21`），在陷阱上**可评分的错答**里：
  落在捷径值 2 % 内的份额（同 `traps_rl_report` 口径）；错答与正确值之比（经引擎的单位换算到同一量纲；绝对温度另报开尔文差；不可换算者计数后剔除）的 |log10| 中位数、偏差 ≥ 2 倍的份额。
- 只报数字；正文据此写一句"套错公式时，错得有多远"。

### 25-D 差距闭合比例（CPU，描述性）

- **单位**：陷阱。比较配置：§21 六个（46 道全覆盖）与 Qwen3.5-27B 关 / 开（§23 E2 合并后的 46 道）。家族：Qwen3.5-9B（基座 → A 臂四终点均值）与 Qwen3-8B（同）。
- **统计量**：G = Σ(A − 基座) / Σ(配置 − 基座)，按陷阱重抽 10,000 次（`random.Random("paper:f25:gap:<配置>:<家族>:20261109")`），每次重算比值；只在配置 − 基座的均值 ≥ 0.05 时报 G。同时报剩余差距 配置 − A 的均值与区间。
- 只报数字；正文用"RL 补上了与某配置差距的 G"替代"与某配置相当"，其中 G 的区间不含 1 时不得写"追平"。

### 25 结果记录 F25-1（2026-11-09 11:20；B、C、D 完成，A 等 114 的 GPU 4–7）

- **执行**：`explore/f25_report.py`（提交 2aa29c7 / f94769f：第一次运行后补上 §25-B 点名的 AUC，并让不在 P1 的 plume_mixing_lid 只进样本层面的份额）。各两次运行逐字节相同：
  `results/paper/f25_gap.json`（ad6dcd0e）、`f25_magnitude.json`（08ad2611）、`f25_luck.json`（a470fa0f）。采样文本 `experiments/x_passk/<家族>/passk_final_outside.x18.texts.jsonl` 与服务器哈希一致（`MANIFEST.server.sha256`）。
- **25-B 偶发正确**（单位样本，区间按机制重抽）：

  | 家族 | 组 | 样本 | 机制 | 可检验 | 功能正确 |
  |---|---|--:|--:|--:|--:|
  | Qwen3-8B | 稀有（0 < pass@1 < 0.02） | 251 | 32 | 96 % | **49.6 % [29.3, 67.7]** |
  | Qwen3-8B | 常见（pass@1 ≥ 0.1） | 819 | 14 | 97 % | 94.6 % [86.3, 99.3] |
  | Qwen3.5-9B | 稀有 | 68 | 8 | 87 % | 76.3 % [45.8, 91.9] |
  | Qwen3.5-9B | 常见 | 3,178 | 53 | 95 % | 99.0 % [98.4, 99.5] |

  稀有 − 常见：Qwen3-8B −45.0 [−66.0, −25.1]，Qwen3.5-9B −22.8 [−54.1, −7.2]。
  **检验本身的对照**：Qwen3-8B 稀有组里全为巧合的机制，Qwen3.5-9B 在同一模板上的正确样本几乎全部功能正确（anelastic_continuity 0/18 对 59/59，sea_ice_insulation 3/25 对 59/60，frictional_three_force_balance 1/15 对 54/55）；
  人工核对一例（anelastic_continuity，8B 样本 24）：代码把标准公式值乘上空气密度（≈ 1.26 kg m⁻³，量纲不通）而恰好落在正确值 110.6 mm s⁻¹ 的 5 % 内。
  **读法（写定）**：Qwen3-8B "偶发正确多为巧合，可达性须按功能正确重算"；Qwen3.5-9B "只报数字"。
  **重算**：Qwen3-8B 有 9 个机制的 pass@64 > 0 全部来自巧合（anelastic_continuity、capillary_rise_gravity、cunningham_slip、deposition_brownian、ice_capacitance_shape、inertia_gravity_wave、raindrop_drag_regime、spherical_slant_path、unconfined_dupuit），
  它们 A 臂都没学会。按功能正确重算后 P1 的 2 × 2 表从 36 / 19 / 1 / 13（Fisher p 1.3 × 10⁻⁴）变为 36 / 10 / 1 / 22（p 2.0 × 10⁻⁹），基座 pass@1 预测学会的 AUC 0.895 → 0.932；
  Qwen3.5-9B 1 个机制（drop_ventilation）变为零，p 6.1 × 10⁻³ → 5.6 × 10⁻⁴，AUC 0.959 → 0.961。即：去掉数值巧合后，"可达性决定 RL 能否学会"更强，不是更弱。
- **25-C 物理量级**（46 道陷阱）：捷径值 / 正确值的倍数中位数 1.68（|log10| 四分位 0.10–0.48），≥ 1.5 倍 27 道、≥ 2 倍 19 道、≥ 10 倍 7 道，符号相反 1 道。
  各类别中位实例：大气化学 3.0 年对 63.4 年（21 倍）、辐射 77 K 对 235 K、空气质量 5.8 × 10⁻⁵ 对 2.0 × 10⁻⁵ g m⁻³、动力学 46 对 69 分钟、热力学 4.8 对 3.3 °C km⁻¹、云物理 0.31 % 对 0.52 %（正确 对 捷径）。
  错答（可评分）：落在捷径值 2 % 内的份额 Qwen3.5-9B 基座 23 % → A 臂 34 %，Qwen3-8B 9 % → 21 %；前沿关思考 18–38 %（GPT-5.5 33 %，DeepSeek 38 %）。
  错答对正确值的倍数中位数：Qwen3.5-9B A 臂 1.28、DeepSeek / GPT-5.5 关思考 1.33 / 1.41（偏差 ≥ 2 倍的份额 24–35 %）；Qwen3-8B 基座 43 倍（RL 后 2.7 倍）。
  注意：绝对温度答案的倍数按开尔文计，偏小（如 250 K 对 314 K 记 1.26 倍），正文举例用差值。
- **25-D 差距闭合比例**（46 道，G = A 臂增益 / 配置相对基座的差距）：

  | 配置 | Qwen3.5-9B G | 剩余差距（配置 − A） | Qwen3-8B G |
  |---|--:|--:|--:|
  | DeepSeek V4 Flash 关 | 92 % [33, 168] | +1.6 [−9.2, +12.7] | 27 % |
  | Qwen3.5-27B 关 | 62 % [19, 98] | +12.5 [+0.7, +24.6] | 23 % |
  | Qwen3.5-27B 开 | 51 % [16, 79] | +19.0 | 21 % |
  | GPT-5.4 关 | 56 % [17, 82] | +16.1 | 21 % |
  | GPT-5.5 关 | 46 % [13, 68] | +23.4 | 19 % |
  | DeepSeek V4 Flash 开 / GPT-5.5 开 | 32 % / 32 % | +42.2 / +43.7 | 16 % / 15 % |

  MiniMax-M3 关（14.5 %）低于两家族的 A 臂，G 不报。按写定规则：只有 DeepSeek 关思考的 G 区间含 1，可写"追平"；其余写"补上了差距的约一半（27B 关 62 %、GPT-5.5 关 46 %）"。

### 25 更正与审查记录 F25-2（2026-11-09 12:00；写于 E5 任何运行之前）

- **25-B 口径更正**：F25-1 的 2 × 2 表与 AUC 把"学会"定义为 A 臂两种子均值 ≥ 0.5，与 P1（`rigor.json`："A / D / F 任一臂、任一种子 ≥ 0.5"）不同，本节写的"P1 的 Fisher 与 AUC"应按 P1 的定义。
  改正后（提交 a305f66，原始表锚定 `rigor.json` 并核对一致；两次运行逐字节相同，`f25_luck.json` sha256 前缀 e6c33723）：
  Qwen3-8B 原始 39 / 16 / 1 / 13（Fisher p 1.8 × 10⁻⁵，AUC 0.913，与 rigor 相同）→ 按功能正确重算 38 / 8 / 2 / 21（p 2.8 × 10⁻⁹，AUC 0.925）；
  Qwen3.5-9B 62 / 2 / 2 / 3（p 1.8 × 10⁻³，AUC 0.978）→ 62 / 1 / 2 / 4（p 8.5 × 10⁻⁵，AUC 0.981）。F25-1 中按旧定义给出的表、p 与 AUC 作废，功能正确比例与读法不变。
  **例外**：9 个可达性全来自巧合的 Qwen3-8B 机制里，`raindrop_drag_regime` 在 6 次运行中有 1 次学会（D 臂种子 43，越界 0.70；其 9 个基座正确样本全为巧合），其余 8 个在全部 6 次运行都 ≤ 0.05。
- **25-B 读法的边界**：Qwen3-8B 稀有组功能正确 49.6 %，刚低于写定门槛 0.5，区间 [29.3, 67.7] 跨过 0.5；"多为巧合"按写定规则成立，但正文应写"约一半"，不写"多数"。
- **25-C 与 §12 重叠**：§12（`cost.json`）已在 328 道陷阱上报过捷径值对正确值的倍数（全部中位 2.0，regime_boundary 1.79）；25-C 的 46 道（中位 1.68）是其子集上的同一量，正文引 §12 的全集数，25-C 只作 46 道训练评测集上的补充。审查时才发现重叠，写 §25 之前没有核对 §12。
- **25-A 文字更正**：验证集为过滤后 `train_probe.parquet` 的 **69** 行（§25-A 写成 76，为 8 个模板 inside / outside 计数的误加），服务器上过滤输出的 sha256 前缀 4f617c19。其余设置不变。
- **25-A 设计复核**（不改设置）：关闭动态采样过滤后，全对 / 全错题组的 token 仍计入 token-mean 的分母，梯度按"有信号 token 的比例"缩小；Adam 的更新对梯度的整体缩放近似不变，所以这不等于更小的学习率，但步与步之间的缩放波动会进入二阶矩。
  报告时另给每步有信号（非零优势）样本数与目标 outside 行上的有信号样本累计数，作为"训练实际给了多少信号"的直接量。

### 23 结果记录 F23-3（2026-11-09 17:40；E4 完成）

- **执行**：114 GPU 4–7，`rl-wt-8226e34`。第一次尝试 08:21 在 378 行后 vLLM 报 `RPC call to sample_tokens timed out`，其余 1,022 行被记为连接错误；按 runner 处理基础设施错误的做法，备份原文件（`passk_final_outside.x18.jsonl.crash-20261109T0821.bak`），只保留 378 行可评分结果，08:34 以同样设置续跑，17:12 完成。
  最终 1,400 行、每行 64 个样本、0 个错误行；采样文本与结果逐行对应（无重复、无缺失）。副本 `experiments/x_passk/qwen35-27b/`，与服务器哈希一致。报告 `results/paper/f23_e4.json`（两次运行逐字节相同，sha256 前缀 2ae00323）。
- **E4（只报数字）**：Qwen3.5-27B 基座关思考在 70 个机制越界行上的平均 pass@64 0.897（Qwen3.5-9B 0.854、Qwen3-8B 0.436）；零命中 5 个：ch4_forcing_sqrt、clausius_clapeyron_extrapolation、ice_above_flotation、line_broadening、sonic_humidity_flux（9B 5 个、8B 14 个）。
  与 Qwen3.5-9B 的 Spearman 0.76 [0.55, 0.90]（69 个共同机制），两者都为零 3 个（ch4_forcing_sqrt、clausius_clapeyron_extrapolation、sonic_humidity_flux）；与 Qwen3-8B 0.43 [0.24, 0.57]，两者都为零 4 个。ch4_forcing_sqrt 与 clausius_clapeyron_extrapolation 在三个模型上都为零。
- **§25-B 附报**（`results/paper/f25_luck_qwen35-27b.json`，两次相同，dcf980e0）：稀有组只有 2 个正确样本（co2_forcing_log、sun_photometer_airmass 各 1），都功能正确；常见组 99.3 % [98.0, 99.9]（3,660 个样本、61 个机制）。
- 是否训练 27B 由用户决定（§23 写定）；按 11-09 的封闭清单，不再新增运行。

### 25 结果记录 F25-3（2026-11-09 19:40；A 完成）

- **执行**：114 GPU 0–3（§25-A 写的是 GPU 4–7；E4 占着 4–7，用户 11-09 停了自己在 0–3 上的服务让给 E5，设置不变），worktree `rl-wt-e7d69ec`。冒烟通过（rollout / actor log-prob Pearson 0.998、显存 48 GB、奖励 0.56 / 0.72、截断 0.2–0.5 %）。
  正式运行一次尝试，14:29 起，200 步全部完成（墙钟 258 分钟，未触 36 小时上限，未续跑）；步 200 存档后 DataLoader 子进程在退出时被杀，进程返回 0，不影响存档。第 200 步合并后按 §25-A 评测。
  每步 rollout 全部转存（`/path/to/local/x/rollouts/f25-qwen3-8b-F1-A-s42-ext/`，200 个文件，留在服务器）。副本 `experiments/x_f25/`，与服务器哈希一致。
  报告 `results/paper/f25_rl.json`（本地两次运行逐字节相同，sha256 前缀 2300273e）；有信号样本数 `f25_report signal`（3b6fb8d）在服务器跑两次，相同（a8fd4c5f）。
- **主统计量**：8 个目标机制中终点 outside 正确率 ≥ 0.5 的 **0 个**。按写定读法：**"约十倍的 RL 采样集中在这些机制上仍学不会：P1 的边界不是训练时长造成的。"**
  终点 outside：7 个为 0，saturated_lapse 0.05（A 臂种子 42 第 120 步 0.10）。终点 inside、given 8 个机制全为 1.00，flag 7 个为 0、saturated_lapse 0.05。
- **次要（只报数字）**：pass@64 = 0 的目标机制个数 基座 5 → 起点 6 → 终点 7（saturated_lapse 0.35 → 0.20，unconfined_dupuit 0.05 → 0，其余起点终点都为 0）。
  inside：A 臂第 120 步两种子均值 0.975–1.00（`round4_conditional.json`），终点 1.00，无下降。验证曲线（69 行，T 0.3）：第 0 步 0.507，第 20 步 0.478，第 40–200 步 0.522（36 / 69），平。
- **有信号样本（F25-2）**：204,800 个训练样本中 7,488 个（3.7 %）优势非零；outside 行 68,160 个中 3,264 个（4.8 %），**全部来自 saturated_lapse**（8,512 个中 3,264）。
  其余 7 个机制每个约 8,500 个 outside 样本，**有信号样本为 0**：200 步里这些行没有一个样本拿到奖励，策略梯度在这些行上恒为零。按 50 步分段，全部有信号样本 2,816 / 1,920 / 1,440 / 1,312，outside 960 / 1,056 / 736 / 512。
  唯一有信号的 saturated_lapse 也没有学会（outside 0.10 → 0.05，pass@64 0.35 → 0.20）。
- 限度：单模型、单折、单种子；"约十倍"按 §25-A 的每行采样次数计（约 430 次对 A 臂约 43 次）。按 11-09 封闭清单，不做 9B 的 E5 或第二种子。
