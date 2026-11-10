# 预注册 DOSE-text：只注入知识的文字——RL 能把一句话变成能力吗

写于 2026-10-29，在本文所列的任何数据构建、训练、采样或评测之前。本文一经提交不改；执行上的改动只能以带日期的修订追加在末尾，
且须在看到相关结果之前。前置实验与全部已知结果：`PREREG_DOSE.md`（含 DOSE-4）、`PREREG_X.md`（含 X-41）。

## 0. 写定时已知的

- DOSE（Qwen3-8B，17 个单靠 RL 学不会的机制，随机剂量 0 / 1 / 2 / 4 对**正确程序**）：H1 成立（有剂量 61.3 对无剂量 0.0）；H2 不可检验。
- DOSE-4 的事后读数：热身后每次采样答对的概率 p（20 道越界题 × 64 次采样）几乎决定 RL 成败——剂量 ≥ 1 的观测中 p ≥ 0.02 学会 12 / 12，
  0.005–0.02 学会 3 / 6，< 0.005 学会 1 / 8；剂量 0 的 8 个观测 p 都 < 0.02、全部未学会。未学会的机制 RL 终点在每道越界题上都给出标准公式的值。
  X 的 69 个机制（基座 p 对 A 臂）方向相同、更有噪声。这些分档边界是看过 DOSE 结果后定的。
- DOSE 注入的是解题程序，H1 分不开"给了知识"与"教了解法"（PREREG_DOSE §6）。

## 1. 问题与预测

把注入物从程序换成**一段陈述替代关系的文字**（不含任何程序、任何题目），其余与 DOSE 相同。

- **H1-text（注入是因）**：注入了文字的机制，RL 后越界正确率高于同一机制未注入时。
- **H2-p（可达性决定 RL）**：注入文字后，RL 能否学会由热身后的 p 决定——p ≥ 0.02 的观测学会，p < 0.005 的不学会。阈值取自 DOSE-4，本实验是样本外检验。
- 两个方向都有结论：文字能把 p 推到 0.02 以上并让 RL 学会，则 RL 把知识变成能力；文字推不动 p、RL 不学会，则知识的文字形式进不了采样分布，RL 需要可执行的示范。

## 2. 单位

与 DOSE 相同的 17 个机制（`results/dose/design.json` 的 `units`），同两折：F1 7 个、F2 10 个。

## 3. 注入物与分配

- **文字**：该模板"给出替代"版本（`atmoscoder_rl/templates/<机制>.py` 的 `text()` 中 `given` 字符串）**逐字**取出（只合并空白），不写新内容。
  它陈述失效条件与替代关系（例如 displacement_height_canopy："Because the mast is less than two canopy heights above the ground, the profile is
  referenced not to the ground but to an effective origin raised into the canopy: use u(z) = (u*/κ) ln((z − d)/z0), where the zero-plane displacement
  height d is two-thirds of the canopy height."）。
- **SFT 行**：每个注入的机制 4 行，同一答案、4 个固定问法，system 与热身行相同（`atmoscoder_rl/prompts.SYSTEM`）；user 为下列之一，`{S}` 为该机制的标准做法名
  （下表，本文写定）；assistant 为上述文字。不经 `user_prompt()` 的代码规则包装，没有 `solve()`。
  1. "In atmospheric-science calculations, the standard approach is {S}. Under what condition does it stop being valid, and what relation should be used instead?"
  2. "A problem is usually solved with {S}. When does this fail, and what replaces it?"
  3. "State the condition under which {S} is not valid, and give the relation that applies in that case."
  4. "{S}: what is its range of validity, and what should replace it outside that range?"

| 机制 | `{S}` |
|---|---|
| ch4_forcing_sqrt | the linear local-slope approximation for the CH4 radiative forcing |
| clausius_clapeyron_extrapolation | linear extrapolation of the saturation vapour pressure with its local rate of change |
| co2_forcing_log | the linear local-slope approximation for the CO2 radiative forcing |
| cunningham_slip | the Stokes settling speed of a small particle |
| diffuse_transmission | Beer's-law beam transmittance applied to a diffuse flux |
| displacement_height_canopy | the neutral logarithmic wind profile referenced to the ground |
| doppler_aliasing | taking the Doppler radial velocity reported by the radar as the true velocity |
| freezing_point_depth | the freezing point of sea water from its salinity alone |
| henry_depletion | Henry's law with an unchanged gas reservoir |
| inertia_gravity_wave | the non-rotating gravity-wave dispersion relation |
| line_broadening | the pressure-scaled Lorentz half-width |
| plume_ground_reflection | the Gaussian plume formula without a ground |
| snowpack_cold_content | melt computed as absorbed energy divided by the latent heat of fusion |
| spherical_slant_path | the plane-parallel slant path H sec(theta) |
| sun_photometer_airmass | the plane-parallel air mass sec(theta) |
| surface_multiple_reflection | the single-pass transmitted flux absorbed by the surface |
| unconfined_dupuit | the Thiem drawdown formula with a fixed transmissivity |

- **分配（交叉设计）**：每个机制恰在一个种子里注入、另一个种子里不注入。每折 `random.Random("dose-text:<折>")` 打乱单位，前 ⌈n/2⌉ 个在种子 42 注入、其余在种子 43 注入：

| 折 | 种子 42 注入（种子 43 不注入） | 种子 43 注入（种子 42 不注入） |
|---|---|---|
| F1 | diffuse_transmission、freezing_point_depth、henry_depletion、surface_multiple_reflection | displacement_height_canopy、line_broadening、unconfined_dupuit |
| F2 | ch4_forcing_sqrt、cunningham_slip、doppler_aliasing、inertia_gravity_wave、plume_ground_reflection | clausius_clapeyron_extrapolation、co2_forcing_log、snowpack_cold_content、spherical_slant_path、sun_photometer_airmass |

  观测 34 个：注入 17（F1 s42 4、s43 3；F2 s42 5、s43 5），未注入 17。每个机制一对机制内对照。

## 4. 训练与评测

- **热身数据**（`explore/dose_text.py build`）：从该折 W_f 的同一采样文件出发（同 PREREG_DOSE §4），**17 个单位的程序对全部去掉**（相当于 DOSE 每个单位剂量 0），
  非单位模板的对、顺序与 RAFT 背景行（1,060 行）与 W_f 逐行相同；注入机制的 4 行文字放在该机制原来第一对所在的位置。
  **开跑前核对**：`--full` 重建的 parquet 必须与 W_f 的 parquet 逐行相同，否则不训练。
- **热身 SFT、RL、测量、标签、排程**：与 PREREG_DOSE §4 逐项相同（SFT 1 轮 lr 5e-6 种子 42；RL 从该种子热身模型出发、与 WA 相同、120 步、X-4 上限、X-10 续跑；
  pass@64 与 `x_eval.sh` 全部评测；关思考）。标签前缀 `dtext`：热身 `sft-dtext-qwen3-8b-<折>-s<种子>`，RL `dtext-qwen3-8b-<折>-WA-s<种子>`；
  数据 `data/sft_dtext-qwen3-8b-<折>-s<种子>.parquet`。`explore/dose_chain.sh` 加 `PREFIX` 与 `BUILDER` 两个变量（默认 `dose`、`explore.dose`，默认行为逐条不变）。
  约 400 GPU 小时。
- **RL 前的格式门（T 的教训；每个热身模型，评测完成后、RL 之前）**：(a) `final.json` 5 次尝试后仍不可评分的行 ≤ 5 %（DOSE 的四个热身模型为 1.6–2.3 %）；
  (b) 非单位训练折机制的原题正确率不低于同折同种子的 DOSE 热身模型 5 点以上。任一不满足：不做 RL，报告"文字行破坏了代码格式"，本实验到此为止。

## 5. 判定（`explore/dose_text_report.py`，提交于任何训练之前）

观测 = (单位机制, 种子)。区间按机制重抽（每机制带它的两个观测），10,000 次，`random.Random("dose-text:20261029")`。"学会" = RL 后越界正确率 ≥ 50 %。

- **H1-text**：17 个机制各自的 d = RL(注入的种子) − RL(未注入的种子)，均值。**成立**：均值 ≥ 30 点且区间下界 > 0；否则不成立。
  另报与 DOSE H1 同口径的非配对差（注入观测均值 − 未注入观测均值）。
- **H2-p**：17 个注入观测按热身后 p 分三组（< 0.005、0.005–0.02、≥ 0.02）。**成立**：≥ 0.02 组学会比例 ≥ 80 % **且** < 0.005 组 ≤ 20 %；
  任一组少于 3 个观测则该组**不可检验**，只报另一组与中间组的数字。未注入观测同样分组报数字。
- **次要，只报数字**：每个观测的热身后 p、pass@64、热身与 RL 越界正确率、热身与 RL 在越界题上给出标准公式值的题数（DOSE-4 的指标，本文预先写定）；
  注入观测的 p 对同机制未注入观测的 p；注入观测的 p 与 DOSE 同机制剂量 1 观测的 p 并列（描述）；RL − 热身；未注入观测对 DOSE 的剂量 0 观测与同种子 A 臂；
  非单位训练折机制上新 RL 终点对同种子 WA 之差。
- 任一标签不完整，判定记为"不完整"，不替补。

## 6. 局限（写定时已知）

- 文字是模板的"给出"句，部分符号在句外无定义（如 surface_multiple_reflection 的 T、S、α、R）；它是"题内提示"的原文，不是教科书式的完整陈述。
- 4 行同一答案，相当于把一句话过 4 遍；没有测剂量。
- 文字行是代码 SFT 语料里的新行型，格式门只能在热身评测后检出，不能事先排除。
- 单位沿用 DOSE（由已看过的结果选出）；H2-p 的阈值来自 DOSE-4；一个家族、关思考；每个机制两个观测来自两个不同的 RL 种子。
- 与 DOSE 相比，本实验的未注入观测与 DOSE 的剂量 0 观测同构，但热身模型不同（本实验所有单位都无程序对）。

## 2026-10-29 执行记录 DT-1：prep 核对与格式门（写于任何单位机制的正确率或 p 被读之前）

- **代码与启动**：`cf11bd2`（`explore/dose_text.py`、`dose_text_report.py`、`dose_text_selftest.py`、`dose_chain.sh` 的 `PREFIX` / `BUILDER` / `DESIGN`）；
  自测通过，故意改错（H2 最小组数、H2 忽略低 p 组、H1 门槛、配对差符号、删掉文字行）均被检出；默认设置下 `dose_chain.sh` 的空跑与改前逐条相同
  （唯一差别是重建核对的输出文件名多了前缀）。工作树 `scratch/rl-wt-cf11bd2`；047 跑 F1、113 跑 F2，各 `STAGE=prep` 启动一次（12:54、12:57）。
- **重建核对**：`--full` 重建的 parquet 与 W_f 逐行相同（两折）。四份热身数据与启动前在 scratch 的独立构建逐字节相同
  （F1 s42 `85ff7ec0…`、s43 `3bf77637…`，F2 s42 `d1b673fa…`、s43 `44c3da6d…`）；17 个单位的程序对全为 0，注入机制各 4 行文字（F1 4 / 3、F2 5 / 5 个机制），
  非单位模板的对与 W_f 相同，RAFT 1,060 行。抽读了一条文字行（system 同热身行，user 为问法 1 加标准做法名，assistant 为"给出"句原文）。
- **热身 SFT**：步数 F1 53 / 53、F2 52 / 52（行数 1,722 / 1,718、1,688 / 1,688，÷ 32）。
- **热身评测与 pass@64**：四个模型 EVALDONE，`serving.json` 的模型路径正确；pass@64 各 1,400 行、id 不重复、n = 64、无错误。
- **格式门（§4，只用非单位机制与不可评分行数，单位机制的正确率与 p 未读）**：
  (a) 不可评分行 F1 s42 2.55 %、s43 1.54 %，F2 s42 1.71 %、s43 2.12 %（门槛 5 %）；(b) 非单位训练折机制原题正确率对同折同种子的 DOSE 热身模型
  F1 80.7 对 82.3、82.5 对 84.8，F2 64.2 对 67.1、63.2 对 63.5（差 −0.3 到 −2.9 点，门槛 −5）。四个都通过，进入 RL。

## 2026-10-29 执行记录 DT-1：prep 核对与格式门（写于任何单位机制的越界正确率或 p 被读之前；RL 尚未启动）

- **代码**：预注册 dc628f5；构建、判定、自测 cf11bd2（自测 21 项通过，故意改错 H2 最小组数、H2 忽略低 p 组、H1 门槛、配对差符号、删掉文字行均被检出；
  `dose_chain.sh` 默认设置下的空跑与改前逐条相同，只有重建核对文件名多了前缀）；格式门脚本 1cf9dc3。链在 047（F1）与 113（F2）从 `rl-wt-cf11bd2` 运行，
  12:54 / 12:57 启动，14:21 两边 prep 结束。启动前 113 的 `bwrap --unshare-net` 实测 0.01 秒（DOSE-3 的内核问题已消失）。
- **重建核对**：两折 `--full` 重建的 parquet 与 W_f 逐行相同（启动前在 scratch 做过一次，链内再做一次，都通过）。
- **数据**：四份 parquet 与启动前 review 构建的逐字节相同（sha256 `85ff7ec0…`、`3bf77637…`、`d1b673fa…`、`44c3da6d…`）；17 个单位的程序对全部为 0；
  注入机制各 4 行文字（F1 s42 4 个、s43 3 个；F2 各 5 个），放在该机制原第一对的位置；非单位模板与 RAFT 1,060 行与 W_f 相同。
  行数 1,722 / 1,718 / 1,688 / 1,688；SFT 步数 F1 53 / 53、F2 52 / 52（= 行数 ÷ 32）；SFT 配置同 W_f。实际看过一条文字行：system 同热身行，user 为问法 1 加标准做法名，assistant 为"给出"句原文。
- **完整性**：四个热身模型 final 5,600、judge 2,800、test_id 64、textbook 92 行全齐（EVALDONE）；pass@64 各 1,400 行、id 不重复、n = 64、无错误。
- **格式门（§4，`explore/dose_text_gate.py`）**：不可评分行 2.55 % / 1.54 % / 1.71 % / 2.12 %（门槛 5 %）；非单位训练折机制原题正确率对同折同种子的 DOSE 热身模型
  −1.6 / −2.3 / −3.0 / −0.3 点（门槛 −5 点）。**通过**，RL 可以开始。文字行带来的原题正确率小幅下降（1–3 点）在读数时注明。
- 只读过上述数字；单位机制的越界正确率、p、pass@64 一概未读。

## 2026-10-30 执行记录 DT-2：RL 完成、报告（§5 的读数，只跑一次）

- **运行**：047 F1 s42 / s43 各 120 步（rc 0；508 / 543 min；平均截断率 0.032 / 0.008），113 F2 s42 / s43（05:59 DONE）；训练后审查无 OOM、无质量异常，
  退出时的 DataLoader 回溯为收尾噪音。终点评测 8 个标签全部完整（final 5,600、judge 2,800、test_id 64、textbook 92），基础设施错误 0。
  报告 09:49–09:51 于 114 从 `rl-wt-c5619b8` 运行 `explore.dose_text_report`（脚本与 DT-1 的 cf11bd2 逐字相同），输出 `results/dose_text/dose_text_qwen3-8b.json`。
- **H1-text：不成立**。17 机制配对差 RL(注入种子) − RL(未注入种子) = 6.5 点 [−9.0, 20.0]；非配对差 6.5 [−10.6, 24.1]。上界 20 低于 DOSE H1 的下界 38.5。
- **H2-p：不可检验**。注入观测按热身后 p：< 0.005 14 个学会 2、0.005–0.02 2 个学会 1、≥ 0.02 **1 个**（surface_multiple_reflection s42，未学会）。
  未注入观测：< 0.005 14 个学会 2、0.005–0.02 3 个学会 0、≥ 0.02 0 个。
- **次要**：注入 / 未注入观测热身越界正确率 1.2 / 0.0，RL 后 17.6 / 11.2，RL ≥ 50 % 3 / 2；RL − 热身（注入）16.5 [−1.2, 35.0]；
  未注入 RL − 同种子 A 11.2 [0.0, 27.6]；非单位训练折机制新 RL − WA 0.4 [−0.5, 1.6]。注入 p 对同机制未注入 p：17 个机制里基本相同（中位数都 ≈ 0.002）；
  注入 p 对 DOSE 剂量 1（一对程序）的 p，9 个可比机制：中位数 0.0039 对 0.0102，≥ 0.02 的个数 1 对 3，7 个程序对更高。
  学会的 5 个观测：doppler_aliasing s42（文字，p 0.0195）、sun_photometer_airmass s43（文字，p 0）、snowpack_cold_content s43（文字，0.0016）、
  snowpack_cold_content s42（无文字，0）、cunningham_slip s43（无文字，0.0023）。34 个观测里 29 个 RL 后越界 20 题全答标准公式值，学会的 5 个降到 0（同 DOSE-4）。
- **读法**：把"给出"句（含替代关系的公式）作为 4 行问答注入，既没有抬高热身后采到正确做法的概率，也没有让 RL 学会；对照 DOSE 的一对程序（61.3 点），
  **提示句形式的文字不产生可达性**。p 曲线低端复现（p < 0.005 学会 4 / 28，DOSE-4 为 1 / 8），高端空着。局限：文字行的训练目标是复述该句，不是应用它（§6）；
  p = 0 只意味着 p < 1/64，未注入也有 2 / 17 学会；一个家族、关思考。记入 `docs/PAPER_OUTLINE.md` 台账 F13（只报数字）。

## 2026-10-30 执行记录 DT-3：背诵探针（追加项；写于任何采样之前，用户 2026-10-30 同意）

DT-2 的零结果有两种读法：文字根本没训进去，或训进去了但不转成程序。本探针只报数字，区分这两种；不改 §5 的判定。
- **(a) 背诵**：四个热身模型各自对所在折的 17 个单位回答训练时的 4 个问句（`explore.dose_text.FRAMES` 填入标准做法名，系统提示同训练行），
  每句 16 次、T 1.0、top_p 1.0、关思考、至多 1,024 token（`explore/dose_text_recite.sh` + `explore.dose_text_recite sample`）。
  "背出" = 空白归一后的"给出"句是回答的子串；另报该句词级 8-gram 在回答中出现的份额（召回）。对照 = 同机制未注入的那个种子的热身模型。
- **(b) 句子在题面里**：热身模型在单位机制 given 行上的正确率（已有评测，`x_report` 同口径重评），并列基座。
- **(c) 提而不用**：热身模型越界 64 次采样（`passk_*.texts.jsonl`）里 8-gram 召回 ≥ 0.5 的份额，按答对 / 答错拆。
- **写定的描述句**："能背" = 注入观测背出率 ≥ 50 %（对照 < 5 %）的个数；"句子在题面里会用" = given 正确率 ≥ 50 % 的观测数；
  "提而不用" = 注入观测中越界采样提到份额 ≥ 10 % 且热身越界正确率 < 10 % 的个数。注入 − 对照的差按机制重抽 10,000 次，`random.Random("dose-text-recite:20261030")`。
  不设成立 / 不成立；写入台账时为探索级。
- **算力**：4 个模型各 68 次请求（17 × 4 句 × n = 16），每模型几分钟；F1 的两个热身模型在 047，F2 的在 113（GPU 4–7）。
- **DT-3 结果（2026-10-30 10:05–10:35；采样 047 GPU 0–1 / 113 GPU 4–5，报告 `results/dose_text/recite_qwen3-8b.json`，脚本 `58bdc74`）**：
  4 个模型 136 条记录全齐（F1 每模型 7 单位 × 4 句、F2 10 × 4，各 16 次）。
  - **(a) 背诵：0 %**。34 个观测没有一次回答包含该句（8-gram 召回也是 0）；回答是千字级的自由陈述（平均 3,200–3,800 字符，每模型 237–265 / 448–640 次在 1,024 token 截断），
    不是训练行的那句话。"能背" 0 / 17。
  - **(b) 句子在题面里**：热身模型 given 行正确率注入 78.2 %、对照 71.5 %、基座 75.3 %；≥ 50 % 的注入观测 13 / 17（基座 28 / 34）。句子放在题面里能用，与 D 系列一致。
  - **(c) 提而不用**：越界采样里 8-gram 召回 ≥ 0.5 的份额 0 %（代码里不复述句子，指标失效），"提而不用" 0 / 17。
  - **探索性补充（在看到 (a) 为 0 之后加的指标，事后）**：回答里出现该句特征数字 / 人名（如 5.35、Kasten、Nyquist）的份额，注入 60.7 % 对对照 46.2 %，
    差 14.5 [2.6, 29.1]（13 个有特征 token 的机制）；升幅 ≥ 20 点的 3 个：co2_forcing_log 79.7 对 29.7、doppler_aliasing 56.2 对 10.9、sun_photometer_airmass 78.1 对 3.1
    （snowpack_cold_content 14.1 对 0）。同一指标放到越界解题采样上：注入 26.4 % 对对照 23.5 %——sun_photometer 的解题采样里 Kasten 只出现 0.3 %、co2 的 5.35 为 0 %、doppler 11.7 %。
    DT-2 里注入后 RL 学会的 3 个观测（doppler s42、sun_photometer s43、snowpack s43）都落在这 4 个"内容进去了"的机制里；co2 进去了但没学会。
  - **读法**：4 行问答没有让模型背出句子，但在 4 / 13 个机制上改变了它*陈述*这个关系的方式；这种陈述几乎不进入解题采样，p 也没升；RL 学会的 3 个注入观测恰是内容进去的机制
    ——陈述性吸收可能是 RL 学会的前提之一，但 n = 4、事后，只能作线索。零结果的解释因此偏向"训进去了一点、不转成程序"，而非"根本没训进去"；
    更强的文字形式（更多行、带推导的陈述）仍是开放问题。
