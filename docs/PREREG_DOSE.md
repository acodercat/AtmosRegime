# 预注册 DOSE：注入剂量梯度——RL 能学会的，是否由热身后能否采到决定

写于 2026-10-27，在任何本文所列的数据构建、训练、采样或评测之前。本文一经提交不改；执行上的改动只能以带日期的修订追加在末尾，
且须在看到相关结果之前。

## 0. 写定时已知的（全部来自 PREREG_X，已提交或已报告）

- Qwen3-8B 的 X 实验全部结果：基座 pass@64（X-18）、A / D / F / WA / WD 两种子的 `final.json` 正确率、W_f 的正确率与 pass@64（§4）、
  热身构建的每模板对数（`data/sft_x-qwen3-8b-F{1,2}.summary.json`）。
- X-40（探索性、事后）：W_f pass@64 = 0 的机制 WA 后仍为 0、> 0 的机制 WA 后几乎全为 100 %；Spearman(W_f pass@64, WA) = 0.81。
  被注入的零命中机制上，W_f 本身已达 78 %，RL 再加约 13 点。注入非随机，不能作因果读。
- 本实验就是为把 X-40 的相关性变成随机化的检验。单位的选定只用上述已知数（§2），剂量随机分配，热身模型、RL 终点均未训练过。

## 1. 问题与预测

在单靠 RL（A 臂）学不会的机制上，用随机分配的剂量（每机制注入 0、1、2、4 对已验证的正确程序）做热身，再做与 WA 完全相同的 RL：

- **H1（注入是因）**：剂量 ≥ 1 的观测，RL 后越界正确率高于剂量 0 的观测。
- **H2（RL 只锐化）**：在剂量 ≥ 1 的观测中，RL 能否学会由热身后的模型能否采到正确答案（热身后 pass@64 > 0）决定，而不是由剂量本身决定。

## 2. 单位（`results/dose/design.json`，`explore/dose.py design`）

家族 1（Qwen3-8B），两折各自训练折的机制（模板），排除 `plume_mixing_lid`（同 P1），同时满足：

1. 基座 pass@64 ≤ 0.10（X-18 采样，即 `results/x/report_qwen3-8b.json` 的 P1 `per_mechanism`）；
2. A 臂越界正确率两种子均值 < 0.5（单靠 RL 学不会）；
3. W_f 热身构建中该模板至少有 4 对（最高剂量可行）。

共 17 个，A 两种子均为 0：

- F1（7）：diffuse_transmission、displacement_height_canopy、freezing_point_depth、henry_depletion、line_broadening、
  surface_multiple_reflection、unconfined_dupuit；
- F2（10）：ch4_forcing_sqrt、clausius_clapeyron_extrapolation、co2_forcing_log、cunningham_slip、doppler_aliasing、
  inertia_gravity_wave、plume_ground_reflection、snowpack_cold_content、spherical_slant_path、sun_photometer_airmass。

被规则排除而值得说明的：A 已学会的 rossby_deformation（0.53）、pseudo_first_order（1.0）、constant_pressure_expansion（1.0）、
gravity_wave_doppler_shift（0.95）；对数不足 4 的 drop_ventilation（2）、ice_capacitance_shape（2）；以及无对的机制。

## 3. 剂量与随机分配

- 档位 0、1、2、4 对。一对 = 一条 given 行的第一个两侧都对的基座程序，作越界题的 SFT 行及其原题孪生（`explore/x_data.warm_rows`，不改）。
  "全剂量"（18–20 对）不再训练，已有的 WA 作参照（这些机制 WA 全为 100 %）。
- 每折、每个 RL 种子（42、43）独立分配：`random.Random("dose:<折>:<种子>")`，每档 ⌊n/4⌋ 次，余数档无放回抽取，再随机配给单位
  （`explore/dose.assign`）。剂量为 d 的机制从其可用对中取 d 对：`random.Random("dose-pairs:<折>:<种子>:<模板>")`，无放回。
- 分配结果（`design.json`，sha256 `baa11d85…`）：

| 折 | 种子 | 0 对 | 1 对 | 2 对 | 4 对 |
|---|---|---|---|---|---|
| F1 | 42 | diffuse_transmission | surface_multiple_reflection、unconfined_dupuit | displacement_height_canopy、henry_depletion | freezing_point_depth、line_broadening |
| F1 | 43 | line_broadening、surface_multiple_reflection | diffuse_transmission、displacement_height_canopy | freezing_point_depth、henry_depletion | unconfined_dupuit |
| F2 | 42 | ch4_forcing_sqrt、doppler_aliasing、inertia_gravity_wave | spherical_slant_path、sun_photometer_airmass | co2_forcing_log、cunningham_slip、plume_ground_reflection | clausius_clapeyron_extrapolation、snowpack_cold_content |
| F2 | 43 | ch4_forcing_sqrt、spherical_slant_path | clausius_clapeyron_extrapolation、cunningham_slip、inertia_gravity_wave | doppler_aliasing、snowpack_cold_content | co2_forcing_log、plume_ground_reflection、sun_photometer_airmass |

  观测共 34 个：0 档 8、1 档 9、2 档 9、4 档 8。

## 4. 训练与评测

- **热身数据**（`explore/dose.py build`）：该折 W_f 的同一采样文件（`experiments/_x/qwen3-8b/warm_samples_F{1,2}.jsonl`，sha256
  `9db251bd…` / `2e45b6c2…`），同一验证；只把 17 个单位的对按剂量减少，其余模板的对、顺序与 RAFT 背景行（全部 1,060 行，`sft_raft.parquet`
  `8d3ab3b6…`，同一抽样）与 W_f 逐行相同。**开跑前核对**：`--full`（不减任何对）重建的 parquet 必须与 W_f 的 parquet（F1 `a00be8a3…`，
  F2 `f45ab1fb…`）逐行相同，否则不训练。
- **热身 SFT**：同 W_f（`train/sft_qwen3_8b.sh`，1 轮，lr 5e-6，SFT 种子 42）。行数减少，SFT 步数约 53–54（W_f 为 F1 60、F2 64），照报。
- **RL**：从该种子的热身模型出发，与 WA 相同（`data-x/F{1,2}-A` 题库，regime 奖励即这些行上的答案奖励，DAPO，120 步，
  PREREG_X X-4 的 24 小时上限与 X-10 的续跑规则），合并最后检查点。训练代码用共享 checkout（与 X 的全部运行相同的提交 4791584；
  训练、奖励、数据构建代码与本文提交时的 HEAD 无差别）。
- **测量**：每个热身模型：`final.json` 越界行 pass@64（`explore/x_passk.sh`，同 X-18 / §4），`explore/x_eval.sh` 全部评测；
  每个 RL 终点：`explore/x_eval.sh`。关思考，T 0.3，与 X 相同；正确率用 `x_report` 的重评。
- **标签**：热身 `sft-dose-qwen3-8b-<折>-s<种子>`，RL `dose-qwen3-8b-<折>-WA-s<种子>`。
- **排程**（`explore/dose_chain.sh`）：047 跑 F1、113 跑 F2，每台先 `STAGE=prep`（重建核对、两个热身模型、评测与 pass@64），
  检查热身数据量、SFT 步数与评测完整后，再 `STAGE=rl`（两个种子依次、各约 11–12 小时，再评测终点）。约 400 GPU 小时。

## 5. 判定（`explore/dose_report.py`，提交于任何训练之前）

观测 = (单位机制, 种子)。区间按机制重抽（每机制带它的两个观测），10,000 次，`random.Random("dose:20261027")`；某次重抽缺少比较的一组时丢弃该次。

- **H1**：剂量 ≥ 1 观测的 RL 后越界正确率均值 − 剂量 0 观测的均值。**成立**：差 ≥ 30 点且区间下界 > 0；否则不成立。
- **H2**：剂量 ≥ 1 的观测按热身后 pass@64 = 0 / > 0 分两组，"学会" = RL 后越界正确率 ≥ 50 %。**成立**：pass@64 = 0 组学会的比例 ≤ 20 %
  且 pass@64 > 0 组 ≥ 80 %；任一组少于 3 个观测则**不可检验**（例如 1 对就让所有机制都能采到），只报 H1 与剂量曲线。
- **次要，只报数字**：每档的热身正确率、热身 pass@64 > 0 的个数、RL 后正确率与 ≥ 50 % 的个数；RL − 热身（全部剂量 ≥ 1，及
  "热身 < 50 % 但 pass@64 > 0" 的观测）；剂量 0 的观测 RL 后对同种子 A（溢出效应）、热身对基座；已有 WA 在单位上的均值；
  非单位训练折机制上新 RL 终点对同种子 WA 之差（检查新热身模型在其余机制上与 W_f 相当）。
- 任一标签不完整（`x_report` 的 `missing()`），判定记为"不完整"，不替补。

## 6. 局限（写定时已知）

- 单位由已看过的结果选出（A 学不会、基座几乎采不到）；随机化只在单位之间分配剂量，因此结论只适用于这类机制。
- 注入的是正确程序本身，H1 成立首先说明"SFT 把解法教会了"；H2 才是对"RL 只锐化"的检验。只注入知识文本的版本不在本文内。
- 每个单位只有两个观测（两个种子各一个剂量）；17 个机制，H2 的两组大小事先不知。
- 剂量减少使 SFT 步数比 W_f 少约 10–17 %；非单位机制上的新 RL 对 WA 之差（§5 次要）用于看这一差别的影响。
- 只有 Qwen3-8B，关思考。

## 2026-10-27 执行记录 DOSE-1：prep 核对、RL 启动、F1 第一次尝试的端口冲突（写于任何剂量模型的结果被读之前）

- **重建核对**：`--full` 重建的 parquet 与 W_f 逐字节相同（两折）；链内再核一次，通过。
- **剂量数据**（独立于构建脚本的检查）：四个 parquet 都是 W_f 的保序子序列；删掉的行全部属于单位模板（F1 s42/s43 206/214 行、
  F2 s42/s43 358/352 行，等于对数 × 2）；每个单位保留 2 × 剂量行，非单位模板与 W_f 相同，RAFT 1,060 行相同，每对的原题孪生都在。
- **热身 SFT**：步数 F1 54 / 53、F2 53 / 53（按行数预计）；lr、调度、种子、轮数、batch、长度、起点与 W_f 的日志配置相同。
- **热身评测与 pass@64**：四个模型 final 5,600、judge 2,800、test_id 64、textbook 92 行全齐，无 API 错误；服务的模型、机器、提交正确；
  pass@64 每个 1,400 行、n = 64、无错误；沙箱无启动失败类，执行超时 7–16 次 / 89,600 次采样。只核完整性，未读任何正确率或 pass@64 数值。
- **RL 配置**：verl 打印的完整配置与 WA 种子 42 对比（两折，895 个键值），只有实验名、检查点与验证目录、起点模型不同。
- **F1 种子 42 第一次尝试**（047，20:30）在第 0 步失败：vLLM 引擎初始化时端口 32791 已被占用（`DistNetworkError ... code: -98`），
  属基础设施故障，按 X-10 同种子续跑一次。失败后 `ray stop` 未清掉该尝试的 9 个 vLLM 进程（EngineCore / Worker / resource_tracker，
  进程组 902762，父进程已为 1），占着 GPU 0、3、6，链在 `wait_free` 等待；21:0x 按 PID 逐个终止这 9 个进程后第二次尝试开始，
  第一次尝试没有检查点，等同于从头重跑；等待时间不计入 24 小时上限（计时从 `wait_free` 之后开始）。

## 2026-10-28 执行说明 DOSE-2：两项探索性分析（写于任何剂量模型的正确率或 pass@64 被读之前）

此时 F1 种子 42 的 RL 已跑完，其余三次在跑；所有热身模型与 RL 终点的正确率、pass@64 均未读过（DOSE-1 起只核完整性与
截断比例、回答长度）。§5 的判定、门槛、单位、剂量一概不变；以下两项只报数字，不设判定。

- **剂量档内的 H2**：§5 的 H2 把剂量 ≥ 1 的观测混在一起按热身后 pass@64 分组，而 pass@64 > 0 本身多出现在高剂量，
  所以即使 H2 "成立"，也无法区分"能否采到"与"剂量大小"哪个起作用。补报：每个剂量档（1、2、4）内，pass@64 = 0 / > 0
  两组各自的观测数与学会数（RL 后越界正确率 ≥ 50 %）；以及两组学会比例之差（> 0 组 − = 0 组），只在两组都有观测的档内计算，
  按档的观测数加权平均，区间同 §5（按机制重抽，`dose:20261027`，10,000 次，无可算档的重抽丢弃）。没有任何档两组都有时报"无"。
- **每次训练内的 H1**：34 个观测来自 4 次训练（2 折 × 2 种子，SFT 种子同为 42），按机制重抽的区间不含"整次训练"的波动。
  补报 4 次训练各自的 RL(剂量 ≥ 1) − RL(剂量 0)，看方向是否一致；每次训练的剂量 0 观测只有 1–3 个，只作参考。
- 代码：`explore/dose_report.py` 的 `within_dose_stat` 与 `exploratory`（输出 `exploratory` 块，打印行以 `[DOSE-2]` 开头）；
  `explore/dose_selftest.py` 新增 4 项，故意改错（不加权、档内缺组、剂量 0 定义）均被检出。最终报告用含本提交的新工作树运行，
  训练用的 `rl-wt-93c5c32` 不动。

## 2026-10-28 执行记录 DOSE-3：113 变慢，F2 种子 43 改到 047（写于任何剂量模型的正确率或 pass@64 被读之前）

- **现象**：F2 种子 42（113）从约 03:00 起每步从约 8 分钟变为 0.7–1.5 小时（第 108–115 步），回答长度与截断比例不变，多出的时间全在
  生成阶段。113 上一个内核 netns 回收线程（`kworker ... +netns`）自约 03:04 起处于不可中断等待，同时数十个沙箱 bwrap 进程卡在
  `rtnetlink_rcv_msg`（新建网络命名空间），评分调用由每小时约 18 万次降到数千次。单独测一次 `bwrap --unshare-net` 需 1.4–4 秒，
  不建网络命名空间时 0.02 秒。这是机器内核的状态，不是训练代码或配置的问题；无 root 权限，无法在本机修复。
- **评分未受污染**：04:00 后约 21 万次评分调用中，因等待超过 60 秒墙钟而判失败的 1 次，另有 192 次第一次超时、按 X-24 重跑成功
  （`logs/x/reward_diag`，只读沙箱计数，未读分数）。047 上同期出现 X-16 所述的 bwrap "Too many open files" 启动失败，按 X-16 退回
  无 bwrap 的隔离层执行，评分照常；F1 种子 43 无评分因此失败，速度正常。
- **处理**（用户 2026-10-28 选定）：F2 种子 42 在 113 继续，预计在 24 小时上限内跑满 120 步（若到上限，按 X-4 合并最后检查点并照报步数）。
  在 113 的 `checkpoints/dose-qwen3-8b-F2-WA-s43/` 预先放 `STOPPED`，使 113 的链跳过种子 43、只评测种子 42 终点后结束，不终止任何进程。
  种子 43 的热身模型 `sft-dose-qwen3-8b-F2-s43/global_step_53` 复制到 047（逐文件 sha256 与源核对），F1 的链在 047 完成后，
  在 047 用同一提交的 `dose_chain.sh`（STAGE=rl，FOLD=F2；047 存储里为种子 42 放 `STOPPED`，注明已在 113 训练）训练种子 43 并评测其终点。
  配置、代码、种子、步数、上限与 WA 和其余剂量运行相同，只换机器（同为 8×H800）；24 小时上限从 047 上的启动起算。

## 2026-10-29 执行记录 DOSE-4：执行事实、§5 判定、事后读数（写于全部结果读过之后；不改 §5 任何判定）

**执行事实**

- 四次 RL 都跑满 120 步、未触及 24 小时上限、无看门狗与崩溃：F1 s42 542 分钟（第一次尝试见 DOSE-1）、F1 s43 491、F2 s42 1,403（113 变慢，DOSE-3）、
  F2 s43 402（047）。四个热身 SFT 1 轮、batch 32，步数 F1 54 / 53、F2 53 / 53（F1 s42 多 8 行：1,734 ÷ 32）。
- F2 s43 在 047 启动时发现 `dose_chain.sh` 在训练前检查**两个**种子的热身模型都已合并在本机，而 047 没有 s42 的热身模型；
  为通过该检查在 047 放了一个空占位目录 `merged/sft-dose-qwen3-8b-F2-s42/global_step_0/`（只含一个说明文件）。s42 的训练由 `STOPPED` 标记跳过
  （日志 "no checkpoint (trained on 113 …)"），终点评测只评 s43（"s42 has no merged model"）。未改任何代码与数据；DOSE-3 未写到这一点。
- **评分沙箱**：047 上的两次 RL（F1 s43、F2 s43）开跑约 2–3 小时后 bwrap 因 "Too many open files" 启动失败，按 X-16 退回无 bwrap 的隔离层评分：
  全程 64 % / 67 % 的评分调用走了退回路径（1,173,207 / 1,842,584；983,908 / 1,467,536），后几小时约 90 %。113 上的 F2 s42 与 F1 s42 无退回。
  退回路径的评分正确性在 X-16 时已验证；H1、H2 都是同一次运行内的机制间对比，两组评分方式相同。每次 RL 评分 140–180 万次，重跑后仍失败的至多 1 次。
- **完整性**：8 个标签 EVALDONE、基础设施错误 0；4 个 RL 终点都在 `global_step_120`；4 个 pass@64 文件各 1,400 行、id 不重复、每行 n = 64；
  `serving.json` 记录的模型路径与主机正确。报告在 113 上从新工作树 `rl-wt-5ae2c08`（HEAD，含 DOSE-2 代码）运行一次，`dose_selftest` 在该工作树通过。
  报告与打印输出存 `results/dose/report_qwen3-8b.json`；原始评测、pass@64、热身数据与全部日志已同步到本地 `experiments/dose/`（`MANIFEST.sha256`）。

**§5 判定**

- **H1 成立**：RL(剂量 ≥ 1) − RL(剂量 0) = 61.3 [38.5, 82.9]（n 26 对 8）。四次训练各自为正：66.7、40.0、57.1、74.4（DOSE-2）。
- **H2 不可检验**：剂量 ≥ 1 的观测中热身后 pass@64 = 0 的只有 1 个（henry_depletion s43），少于 3。可读的那一组不符合 H2 的说法：
  pass@64 > 0 的 25 个观测学会 16 个（64 %，H2 要求 ≥ 80 %）。DOSE-2 的档内差 50.0 [12.5, 77.8] 同样只基于那 1 个观测，无信息量。
- **次要**：按档（0 / 1 / 2 / 4）热身 0.0 / 3.3 / 15.0 / 50.6，RL 0.0 / 55.0 / 44.4 / 87.5，学会 0 / 5 / 4 / 7 个（n 8 / 9 / 9 / 8）；
  RL − 热身（剂量 ≥ 1）39.4 [21.4, 59.0]，"热身 < 50 % 且 pass@64 > 0" 的 20 个观测 49.0 [26.1, 71.6]；剂量 0 的 RL 对同种子 A 0.0 [0.0, 0.0]、
  热身对基座 0.0；非单位训练折机制上新 RL 对 WA 0.0 [−0.7, 1.0]。

**事后读数（`explore/dose_posthoc.py`，`results/dose/posthoc_qwen3-8b.json`；只报数字，分档边界是看过结果后定的）**

- RL 后的越界正确率几乎只有 0 与约 100 % 两种（34 个观测中只有 1 个为 0.95）；H1 的 61.3 点即"学会的比例"。
- **每次采样答对的概率 p**（热身模型在 20 道越界题上各 64 次采样，共 1,280 次）比"pass@64 是否 > 0"分得开得多。剂量 ≥ 1 的 26 个观测：
  p < 0.005 学会 1 / 8，0.005–0.02 学会 3 / 6，**p ≥ 0.02 学会 12 / 12**；剂量 0 的 8 个观测 p 都 < 0.02，全部未学会。
  同一机制在不同剂量下一致：co2_forcing_log 剂量 2 时 p 0.004 未学会、剂量 4 时 0.073 学会；clausius_clapeyron 剂量 1 时 0.016 未学会、剂量 4 时 0.48 学会。
  用 X 的 69 个机制交叉看（基座 pass@1 对 A 臂，两种子均值 ≥ 0.5）：p < 0.005 学会 6 / 32，0.005–0.02 学会 10 / 14，0.02–0.1 学会 7 / 9，≥ 0.1 学会 14 / 14——
  方向相同、更有噪声（低 p 偶有学会，即 P1 的例外；高 p 也有失败，如 log_wind_stability p 0.047 而 A 为 0）。
  DAPO 每题采 8 次、按组过滤全同分的组（`train/dapo_qwen3_8b.sh`）：p = 0.003 时一组至少一次答对的概率约 2 %，p = 0.02 时约 15 %。
- **三个在任何剂量下都未学会的机制**（displacement_height_canopy、henry_depletion、plume_ground_reflection）：注入的热身行正确且明确
  （分别写出 d = 2/3 h、p0 / (1 + H R T L)、像源项），但注入后 p 仍 ≤ 0.006（plume 剂量 4 时 0.003）。它们不是另一类机制，而是剂量 → p 曲线更陡的机制；
  全剂量（18–20 对）的 WA 在这三个机制上都为 100 %（X-40）。
- **未学会的机制上 RL 把回答统一成标准公式的值**：18 个未学会的观测（10 个有剂量、8 个剂量 0），RL 终点在 20 道越界题上**全部 20 / 20** 给出标准公式的值
  （2 % 以内）；热身模型在同一些题上标准公式值的个数为 0–20（中位数 14），其余是别的错误。最极端的 spherical_slant_path：热身 1 / 20 → RL 20 / 20。
  同一模板的原题在 RL 题库里，标准公式在原题上得分，越界题随之被"标准化"。
- **教科书迁移不可读**：17 个单位模板中只有 2 个的来源陷阱在终点评测的 92 道教科书题里（cunningham_slip ← trap_air170_cunningham；
  spherical_slant_path ← trap_7.6_regime、trap_p_55_regime）。cunningham_slip 在剂量 2 / 1 下模板越界题学会（20 / 20、19 / 20），
  其来源陷阱在 8 个模型上全部失败；该陷阱还要求自行写出 Stokes 形式（模板里给出），两个模型都把 k1 r² 写成 k1 r，失败原因混杂。n = 1，不作读数。
- **第二家族**：Qwen3.5-9B 满足单位规则前两条（基座 pass@64 ≤ 0.10、A < 0.5）的机制只有 5 个（全在 F2），未计第三条（对数 ≥ 4）；
  按同一设计做第二家族的 DOSE 不可行。
