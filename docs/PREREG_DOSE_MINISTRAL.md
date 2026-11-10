# 预注册 DOSE-M：第三家族 Ministral 3 8B 上的注入剂量梯度复现

写于 2026-10-30，在该家族任何热身数据构建、SFT、RL 或剂量模型评测之前。**定稿 2026-10-30 11:50**：草稿内容为提交 `cf5c4e2`，审核由 Claude 代做（核对设计文件、
DRY 运行每条命令、判定脚本；修正了训练脚本来源、p 曲线脚本的 X 报告依赖、§5 一处措辞），用户同意定稿。定稿后不改，执行上的改动只以带日期的记录追加在末尾，
且须在看到相关结果之前。探针（§0）已按用户 2026-10-29 的批准完成，是本文的输入，不是本文的结果。门槛 1（冒烟）、2（RAFT）在定稿前完成并记于"门槛结果"；
门槛 3、4、5 在定稿后逐条写入执行记录。

## 0. 写定时已知的

- **家族 1 的 DOSE（`docs/PREREG_DOSE.md`，DOSE-4）**：H1 成立（61.3 [38.5, 82.9]）；H2 不可检验（热身后 pass@64 = 0 的只有 1 个观测）；事后：热身后每次采样答对的概率 p ≥ 0.02 的
  12 / 12 学会、p < 0.005 的 1 / 8；未学会的机制 RL 把回答统一成标准公式值；第二家族 Qwen3.5-9B 只有 5 个候选，不可行。
- **第三家族探针（`explore/family_reach.py`，`results/x/reach_ministral3-8b.json`；模型 Ministral-3-8B-Instruct-2512-BF16，Mistral3ForConditionalGeneration，
  vLLM 0.24 可服务，关思考无开关）**：`final.json` 1,400 条越界行 × 64（T 1.0，`explore/x_passk.sh`）与 1,400 条 given 行 × 8 的热身采样（`explore/x_prep.sh`），全部无错误行。
  70 模板 pass@1 均值 0.122、pass@64 均值 0.597（Qwen3-8B：0.090 / 0.434）；**Z（pass@64 = 0）10 个，其中 8 个与 Qwen3-8B 的 14 个 Z 重合**
  （ch4_forcing_sqrt、clausius_clapeyron_extrapolation、co2_forcing_log、drop_ventilation、ice_above_flotation、line_broadening、rossby_deformation、snowpack_cold_content）；
  候选（pass@64 ≤ 0.10）17 个，其中热身对数 ≥ 4 且非 plume_mixing_lid 的 12 个。同一脚本在 Qwen3-8B 上复现 X 报告的 14 个 Z 与 DOSE 设计文件的全部对数（自检）。
- 用户 2026-10-29 定的筛查规则：真候选 ≥ 10 → 做。满足。

## 1. 问题与预测

家族 1 的 H1 在一个不同厂商、不同预训练语料的家族上是否复现：在基座几乎采不到正确做法的机制上，随机分配 0 / 1 / 2 / 4 对正确程序做热身再做同样的 RL，
剂量 ≥ 1 的观测 RL 后越界正确率是否高于剂量 0。**预测**：H1 复现（差 ≥ 30 点、下界 > 0）。同时报 p 曲线（家族 1 的事后读数 p ≥ 0.02 → 学会，在本家族是样本外检验）。

## 2. 单位（`results/dose_ministral/design.json`，`explore/dose_family.py design`；reach 文件 sha256 记于其中）

训练折机制，排除 plume_mixing_lid（同 P1），满足：(1) 基座 pass@64 ≤ 0.10；(3) 热身对数 ≥ 4。**与家族 1 的差别**：家族 1 的条件 (2)"A 臂两种子均值 < 0.5"
在本家族无法施加——没有 A 臂（跑 A 两折两种子约 140 GPU 小时，只为筛单位）。本设计里剂量 0 的观测就是每个单位上"只做 RL"的对照（家族 1 里剂量 0 的 RL 对同种子 A 为 0.0 [0.0, 0.0]）；
若某单位剂量 0 也学会，它会把 H1 的差拉低，是保守方向。

共 12 个：F1（5）diffuse_transmission、freezing_point_depth、henry_depletion、line_broadening、rossby_deformation；
F2（7）ch4_forcing_sqrt、clausius_clapeyron_extrapolation、co2_forcing_log、doppler_aliasing、inertia_gravity_wave、snowpack_cold_content、sun_photometer_airmass。
其中 11 个也是家族 1 的单位（家族 1 独有 6 个：displacement_height_canopy、surface_multiple_reflection、unconfined_dupuit、cunningham_slip、plume_ground_reflection、spherical_slant_path，
本家族基座 pass@64 都 > 0.10；本家族独有 rossby_deformation——家族 1 里 A 已学会 0.53）。对数不足 4 的候选：drop_ventilation 0、ice_above_flotation 1、raindrop_drag_regime 0、saturated_lapse 0。

## 3. 剂量与随机分配（同 PREREG_DOSE §3，`explore.dose.assign` 不改）

| 折 | 种子 | 0 对 | 1 对 | 2 对 | 4 对 |
|---|---|---|---|---|---|
| F1 | 42 | henry_depletion | diffuse_transmission | line_broadening | freezing_point_depth、rossby_deformation |
| F1 | 43 | henry_depletion | diffuse_transmission | freezing_point_depth、rossby_deformation | line_broadening |
| F2 | 42 | ch4_forcing_sqrt、doppler_aliasing | inertia_gravity_wave | snowpack_cold_content、sun_photometer_airmass | clausius_clapeyron_extrapolation、co2_forcing_log |
| F2 | 43 | ch4_forcing_sqrt、doppler_aliasing | co2_forcing_log、inertia_gravity_wave | snowpack_cold_content | clausius_clapeyron_extrapolation、sun_photometer_airmass |

观测 24 个：0 档 6、1 档 5、2 档 6、4 档 7。henry_depletion、ch4_forcing_sqrt、doppler_aliasing 两个种子都抽到 0（随机结果，不重抽）；
这三个机制因此只提供剂量 0 的信息。

## 4. 训练与评测（与家族 1 相同处不重复，只写差别）

- **RAFT 背景**：本家族基座自己的已验证解答（X-10 规则），`explore/x10_raft.sh`（`SECTION=local-raft-plain`：Mistral 分词器的 vLLM 服务拒绝 `chat_template_kwargs`），
  1,140 题 × 8、T 1.0、每题至多 2 条 → `data/raft_ministral3-8b/sft_raft.parquet`。
- **热身数据**：`explore.dose build` 不改，`--raft` 指向上述文件；没有 W_f，`--full` 重建只运行、不与任何文件比对（`dose_chain.sh` 的 `EXPECT=`）。
  全剂量热身模型不训练（家族 1 有 WA 作参照，本家族没有，§5 次要项相应缺一项）。
- **模型**：`Ministral-3-8B-Instruct-2512-BF16-text`——官方检查点只取语言模型另存为 `Ministral3ForCausalLM`（`explore/ministral_text_only.py convert`；视觉塔从不参与文本计算，
  16 条题面 logits 逐位相同），`tokenizer_class` 改为 `PreTrainedTokenizerFast`（transformers 5.9 的 Llama 专用类对该字节级 tokenizer.json 丢空白；
  修后在 verl 加载器上编码与 `tokenizers` 库逐 id 相同、解码逐字回环）。探针（§0）用官方检查点经 vLLM 采样，语言模型与分词路径等价，数字沿用。
- **热身 SFT**：`train/sft_qwen3_8b.sh`，1 轮、lr 5e-6、SFT 种子 42；**本家族的第一个热身 SFT 兼作 SFT 烟测**（verl 的 SFT 训练器要能加载 `Ministral3ForCausalLM`、
  `sft_dataset.py` 的模板一致性检查要通过）；任一失败即停，修后记录再继续。
- **RL**：`train/dapo_qwen3_8b.sh` 同 WA，`MAX_MODEL_LEN=5120`（Ministral 宣称 262k 上下文，vLLM 按此分配 KV 会溢出；5,120 = 最长题面 + 3,072 回复的上限，
  对 Qwen 不设此项）；RL 的 DAPO 烟测（2 步）**先于本文定稿执行**，结果见执行记录（第 5 次通过）。链的训练脚本从本 worktree 运行（`dose_chain.sh` 的 `TRAIN_REPO=$WT`；
  共享检出 4791584 的 `dapo_qwen3_8b.sh` 不认识 `MAX_MODEL_LEN`），`MAX_MODEL_LEN` 与 `ATMOS_NO_CHAT_TEMPLATE_KWARGS=1` 在启动链时导出、由子进程继承。RL 单次上限 24 h（链默认）。
- **评测**：`explore/x_eval.sh` 与 `x_passk.sh`，环境变量 `ATMOS_NO_CHAT_TEMPLATE_KWARGS=1`（客户端去掉 `chat_template_kwargs`，其余请求参数不变）；
  基座四项评测 `x-ministral3-8b-base` 用文本检查点评出（§5 的基座列）。
- **标签**：热身 `sft-dose-ministral3-8b-<折>-s<种子>`，RL `dose-ministral3-8b-<折>-WA-s<种子>`；`explore/dose_chain.sh` 以 `NAME=ministral3-8b`、`MODEL=<文本检查点>` 运行，
  `DESIGN=results/dose_ministral/design.json`、`RAFT=…/raft_ministral3-8b/sft_raft.parquet`、`EXPECT=`、`TRAIN_REPO=$WT`、`X_STORE=/path/to/local/dose`。
- **排程**：每折 prep（两个热身 SFT + 评测 + pass@64）约 4 小时；每折 rl 两个种子各约 9–11 小时 + 评测。约 350 GPU 小时。机器：114 专用；113 在 OPD 链结束后可并行跑第二折。

## 5. 判定（`explore/dose_report.py` 不改，`--name ministral3-8b`，`random.Random("dose:20261027")` 同家族 1）

- **H1**：同家族 1（差 ≥ 30 点且下界 > 0）。
- **H2**：同家族 1 的规则与不可检验条件。
- **样本外的 p 规则（新增，只报数字，不作判定）**：剂量 ≥ 1 的观测按热身后 p（`explore/dose_posthoc.py` 不带 `--x-report`——本家族无 X 报告，其 A 臂两项不报；
  64 次采样的答对比例）分 p < 0.005 / 0.005–0.02 / ≥ 0.02 三档，报每档学会的个数；家族 1 事后得到的"p ≥ 0.02 全学会、p < 0.005 几乎不学会"在本家族若复现，才可写成跨家族的规律。
- **次要**：同家族 1，脚本不改；无 WA 与 A 的项（已有 WA 在单位上的均值、新 RL 对 WA 之差、剂量 0 对 A）为空、不报。
- **跨家族并列**：11 个共同单位上两家族的剂量—学会表并列，只报数字。

## 6. 局限（写定时已知）

- 单位没有 A 臂筛选；剂量 0 观测承担"只做 RL"的对照，但每个单位只有 0–2 个剂量 0 观测。
- Ministral 3 的热身与 RL 走文本检查点（语言模型抽出、分词器类改为通用）、无思考开关的模板、Le Chat 默认系统提示，与 Qwen 不同；烟测只证明"能跑"，不能排除训练动态差异。
  冒烟五次才通过（视觉塔、上下文长度、分词器），每一次都是工程层面的、在训练前被门槛拦住。
- 12 个单位、24 个观测，比家族 1 少（17 / 34）。
- 本家族的 p 规则是家族 1 事后读数的样本外检验，但 p 的分档边界取自家族 1。

## 执行前门槛（定稿后、训练前，逐条写入执行记录）

1. DAPO 烟测（MAX_MODEL_LEN=5120，2 步）通过：奖励与回复长度有限、无模板错误；每步耗时推算 120 步在 24 小时内。
2. RAFT 采样完成：通过验证的题数、条数（家族 1：1,060 行）。
3. 第一个热身 SFT 跑完并合并；`sft_dataset.py` 的一致性检查通过；热身模型评测与 pass@64 完整。
4. 热身数据量核对：每个单位保留的对数 = 剂量；非单位模板的对数 = 全量。
5. 基座四项评测 `x-ministral3-8b-base` 完整（只核行数，不读数字）。

## 门槛结果（定稿前已得，草稿阶段追加）

- **RAFT 采样（门槛 2）完成，2026-10-30 04:49–05:46，114 GPU 4–6**：1,140 题 × 8、T 1.0、`local-raft-plain`；至少一条通过验证的题 702 / 1,140（61.6 %），
  保留 1,232 条（每题至多 2）→ `data/raft_ministral3-8b/sft_raft.parquet` 1,232 行。对照：Qwen3-8B 596 题 / 1,060 行，Qwen3.5-9B 960 题 / 1,832 行。
  热身数据里 RAFT 行数取 min(2 × 模板行数, RAFT 池)，与家族 1 同一规则；按实际比例报告。
- **DAPO 冒烟（门槛 1）第一次失败，2026-10-30 07:53–09:15，114 GPU 0–3**：verl 的 FSDP 训练端整体加载 `Mistral3ForConditionalGeneration`（含 Pixtral 视觉塔，8.92 B 参数），
  而其 vLLM 采样端只建纯文本模型，第一次权重同步在 `vision_tower.patch_conv.weight` 上 KeyError（vLLM worker 08:10 死亡，主进程挂住，09:15 按 PID 停止；0 步）。
  04:35 排队、07:53 起跑，浪费约 1.4 h × 4 GPU。**修法**：`explore/ministral_text_only.py convert` 只取语言模型，另存为 `Ministral3ForCausalLM` 检查点
  （`/path/to/local/models/Ministral-3-8B-Instruct-2512-BF16-text`；transformers 5.9 与 vLLM 0.24 都注册该架构；分词器、聊天模板、生成配置逐字节复制，
  `tokenizer_config.json` 只去掉 `processor_class`）。本研究只发文本，视觉塔从未参与计算；`check` 在 16 条随机题面（`random.Random("ministral-text-only:20261030")`）上
  比对两个检查点的 logits 与 64 token 贪心续写，结果追加于下。探针（§0）用的是原检查点经 vLLM 服务，等价性成立则其数字直接沿用；此后本家族的热身 SFT、RL 与评测一律用文本检查点。
- **文本检查点的等价性（`results/x/ministral_text_only_check.json`，09:20–09:21，114 GPU 4）**：保留 309 个张量（15.81 GiB）、丢弃 222 个（`vision_tower.*`、`multi_modal_projector.*`）；
  16 条随机题面（每条 688–864 个 token）的全部位置 logits 最大绝对差 **0.0**、argmax 一致率 1.000；64 token 贪心续写 14 / 16 逐字相同，
  2 条（cooper_jacob_early_time_08_inside、lognormal_mass_01_flag）在增量解码中分叉——提示 logits 逐位相同，分叉只能来自两个包装类生成路径的 bf16 数值差在近似平局处翻转。
  判定：语言模型权重与前向完全一致，探针数字沿用。**冒烟第二次失败（09:21–09:27）**：重启时漏传 `MAX_MODEL_LEN=5120`，vLLM 按 262,144 分配 KV 缓存溢出（34.0 > 29.85 GiB），0 步；
  09:31 带该变量第四次启动（文本检查点，114 GPU 0–3）。
- **冒烟第三次失败（09:31–11:15，114 GPU 0–3，第 4 次启动）**：vLLM 与模型都正常，但 0 步——初始验证 64 条全部 `exec_error`、奖励 0，DAPO 的动态采样因此无限重采。
  根因：verl 用 transformers 5.9 的 `AutoTokenizer` 加载该检查点时按 `tokenizer_config.json` 的 `LlamaTokenizerFast` 走 Llama 专用类，它对这个字节级（tekken）
  `tokenizer.json` 编码时丢空白（"def solve" → "def","s","olve"）、解码时把 Ġ / Ċ 原样输出，代码无法运行；探针阶段的 vLLM 服务不经这条路径，数字不受影响。
  **修法**：`ministral_text_only.py convert` 把 `tokenizer_class` 改为通用的 `PreTrainedTokenizerFast`（直接按 `tokenizer.json` 编解码）；在 verl 自己的加载器上验证
  编码与 `tokenizers` 库逐 id 相同、解码逐字回环、聊天模板单个 BOS；`check` 加入同样的断言。重新转换检查点后第 5 次冒烟。
  教训：上一次的等价性检验只比了两个检查点彼此一致，没有比它们与真实分词是否一致。
- **门槛 1 通过（第 5 次冒烟，11:24–11:33，114 GPU 0–3，文本检查点 + 修正后的分词器类 + `MAX_MODEL_LEN=5120`）**：退出 0、2 步；初始验证（test_id，T 0.3）奖励 0.391；
  第 1 / 2 步训练奖励 0.406 / 0.516、回复长度 614 / 596、截断率 0.27 / 0.22（家族 2 的冒烟为 0.31）、梯度范数 22.5 / 21.9、每步 59 / 46 s（家族 2 冒烟 183 / 49 s；
  家族 2 正式跑 120 步约 9 h）→ 120 步推算 < 24 h。退出时一条 DataLoader 回溯为收尾噪音。等价性检验（分词断言版）同样通过：logits 最大差 0.0、argmax 一致率 0.99999994、贪心 14 / 16。
- **基座四项评测**：探针阶段只做了 pass@64 与热身采样，`x-ministral3-8b-base` 的 final / judge / test_id / textbook 尚未评测（§5 次要项与 `dose_report` 的基座列需要），
  11:36 用文本检查点在 114 GPU 0–3 启动（`ATMOS_NO_CHAT_TEMPLATE_KWARGS=1`），结果不读，只核完整性。
- **冒烟小结**：五次启动——047 第 1 次（旧，记于 §0 之前）、114 第 2 次视觉塔、第 3 次漏传 `MAX_MODEL_LEN`、第 4 次分词器、第 5 次通过。浪费约 1.4 + 0.1 + 1.7 h × 4 GPU。
  剩余门槛（定稿后）：第一个热身 SFT 兼 SFT 烟测（门槛 3）、热身数据量核对（门槛 4）。
- **DRY 运行（11:41，114 `rl-wt-1fe1cff`）**：`dose_chain.sh` 在 Ministral 设置下打印的每条命令核对无误：热身数据从本设计与 Ministral RAFT 构建、SFT 以文本检查点为起点、
  训练脚本从本 worktree 运行（`TRAIN_REPO`）、评测与 pass@64 用本 worktree 的脚本、RL 数据 `data-x/F1-A`（与家族 1、2 相同的题面池）。冒烟的回复上限是 1,024 token
  （`dapo_qwen3_8b.sh` 的 SMOKE 设置），截断率 0.27 对应该上限；正式跑为 3,072。`dose_posthoc.py` 的 `--x-report` 改为可选（本家族无 X 报告）。


## 2026-10-30 执行记录 DM-1：prep F1 第一次启动在重建核对处失败（链脚本的空变量处理），修后重启

- 12:24 基座四项评测完成（`x-ministral3-8b-base`：final 5,600、judge 2,800、test_id 64、textbook 92，门槛 5 通过，数字未读）；prep F1 随即启动，
  12:34 `--full` 重建核对失败：`dose_chain.sh` 用 `${EXPECT:-默认}` 取默认值，空字符串 `EXPECT=` 也被替换成家族 1 的 W_f 路径 `data/sft_x-ministral3-8b-F1.parquet`（不存在）。
  §4 写的 `EXPECT=` 语义没变，改为 `${EXPECT-默认}`（只在未设置时取默认）；DRY 运行时该参数在行尾被截断，没有看到。只耗 CPU 约 10 分钟，无 GPU 浪费。
  同一时刻 11:45 的第一个排队启动器也起了一个链，被文件锁拦下（"not starting"），只有一个链在跑。修后从新 worktree 重启 prep F1。

## 2026-10-30 执行记录 DM-2：第二折改在 047 跑（排程变更；不改设计、臂、种子、步数或判定；写于任何 Ministral 结果之前）

- 用户 15:10 把 047 交还本项目使用。§4 的排程原为"114 专用；113 在 OPD 链结束后可并行跑第二折"；OPD 链约 10-31 07:00 才结束，
  故第二折（prep + rl 两个种子）改在 047 上跑，与 114 上第一折的 rl 并行。配方与第一折完全相同（同一 worktree `rl-wt-f8785fb`、同样的环境变量与参数，`FOLD=F2`）。
- 047 的文本检查点从 114 复制（`/path/to/local/models/Ministral-3-8B-Instruct-2512-BF16-text`，逐文件 sha256 与 114 核对后才启动）。
- 链脚本的锁按主机名，两台机器各跑一条链不冲突；两折的数据、热身模型与评测标签互不重名。

## 2026-10-31 执行记录 DM-3：第一折 prep 完成、门槛 3–5 通过；第一折 RL 两个种子分到 114 与 113 并行（写于任何 Ministral 正确率或 pass@64 数字被读之前）

- **prep F1（114，`rl-wt-f8785fb`）**：10-30 12:40 起，W_f 重建核对 12:49 通过；两个种子的热身 SFT 各 59 步、退出 0，13:10 / 13:23 合并；
  评测与 pass@64 10-31 05:00 完成，`DOSE PREP ministral3-8b F1 DONE (rc 0)`。pass@64 用时约 15.5 小时（原估约 4 小时：每个热身模型只用 1 卡，Ministral 生成比 Qwen 慢）。
- **门槛 3**：SFT 日志 17 处 `Ministral3ForCausalLM`（文本检查点，无视觉塔），无 `ValueError` / 回溯（`sft_dataset.py` 的一致性检查只以报错表示失败）；
  两个热身模型评测完整（final 5,600、judge 2,800、test_id 64、textbook 92，0 错误行，EVALDONE）；pass@64 两个文件各 1,400 行（PASSKDONE）。只核行数，未读数字。
- **门槛 4**：按 `results/dose_ministral/design.json` 核对 `build-*.log` 的每模板对数：F1 5 个单位，s42 剂量 0/1/2/4 各 1/1/1/2 个、s43 各 1/1/2/1 个，
  每个单位保留的对数 = 剂量，21 个非单位模板 = W_f 全量，0 处不符；模板行 680 / 676，RAFT 行 1,232 / 1,232。
- **排程变更（不改设计、臂、种子、步数、上限或判定）**：§4 原为一条链依次跑两个种子（约 20 小时）。改为 114 只跑 F1 种子 42（`SEEDS=42`，现在启动），
  F1 种子 43 在 113 的 OPD 链结束后（约 07:30）在 113 跑（`SEEDS=43`，同一 worktree 与环境变量）：热身模型 `sft-dose-ministral3-8b-F1-s43/global_step_59`
  从 114 复制到 113 同一路径，逐文件 sha256 核对后才启动；113 起链前确认 114 的链不含种子 43。若 113 起不来，114 跑完种子 42 后接着跑种子 43。
  两个终点各在自己的机器上由链评测（同一 `x_eval.sh`）；第二折（047）prep 完成后同样核门槛再起 RL。

## 2026-10-31 执行记录 DM-4：第一折 RL 两个种子因训练端与 rollout 端的 log-prob 严重不一致而停止（不重试；写于任何 Ministral 正确率被读之前）

- **事实**：F1 种子 42（114，05:02 起）与种子 43（113，07:19 起）的训练日志里，verl 记录的 rollout（vLLM）与训练端（FSDP / Hugging Face）对同一批采样 token 的概率严重不一致，
  从第 1 步就存在并逐步恶化。种子 42：第 1 步 `training/rollout_actor_probs_pearson_corr` 0.571、`rollout_corr/training_ppl` 127.4 对 `rollout_ppl` 11.3；
  第 10 步 0.143 / 127,458 / 3.3；第 48 步 0.024 / 4,157,842 / 3.5。种子 43：第 5 步 0.326 / 19,699 / 6.7。同一指标在家族 1 DOSE 运行里是
  0.999 / 1.18 / 1.18；`actor/pg_clipfrac` 0.34（家族 1 0.0004）、`actor/grad_norm` 15–90（家族 1 0.34）、`actor/entropy` 第 1 步 4.4（家族 1 0.18）。
  训练端算出的策略比率因此没有意义，这两次 RL 不是 §2 的 WA 处理，结果作废。
- **处理**：07:50 先在两个检查点目录写 `STOPPED`（链不再重试），再按进程组停掉两台机器的启动器 + 链与训练进程组（PID 核对后 `kill -TERM -<pgid>`），
  两台 GPU 全部释放。约 2.8 + 0.5 小时 × 8 卡作废。第二折（047）的 prep 继续（pass@64 只用 vLLM，不受影响），**RL 不启动**，直到原因查清、
  修正方案写入新的执行记录。
- **门槛为何没拦住**：门槛 1 的冒烟（DM 定稿前，2 步）只看了奖励、回复长度、截断率与模板错误；冒烟日志第 1 步梯度范数 22.5（家族 1 约 0.3）
  已是同一信号，rollout / 训练一致性指标没有被列为门槛。今后冒烟门槛增加：`training/rollout_actor_probs_pearson_corr` ≥ 0.99、
  `rollout_corr/training_ppl` 与 `rollout_ppl` 之比 ≤ 1.05。
- **诊断**：`explore/diag_logprob_match.py`（716abd9）在 114 单卡上比较 vLLM 贪心生成的逐 token log-prob 与 Hugging Face 前向（单条、按 verl 无填充方式打包）；
  结果与原因另记。候选原因：transformers 5.9 的 `Ministral3ForCausalLM` 与 vLLM 0.24 在 YaRN（factor 16，原长 16,384）/ `llama_4_scaling_beta` 上实现不同，
  或 verl 无填充打包对该模型的位置 / 注意力处理不同。

## 2026-10-31 执行记录 DM-5：原因查明——verl 的无填充打包（`use_remove_padding`）对 Ministral 3 的训练端前向是错的；热身 SFT 同样受影响；全部 Ministral prep 作废，等用户决定

- **诊断 1（`explore/diag_logprob_match.py` 2b19421，114 单卡，F1 s42 热身模型，4 道训练题 × 256 个贪心 token）**：vLLM 0.24 与 Hugging Face（transformers 5.9，
  flash_attention_2）逐条前向的 token log-prob 基本一致——困惑度 1.085 对 1.071，平均绝对差 0.037 nats，最大 0.90；同样四条按 verl 方式打包成一行（位置编号每条重置）后，
  Hugging Face 自己的结果就变了（单条 vs 打包平均差 0.034、最大 1.40），本应为 0。模型权重、YaRN / `llama_4_scaling` 不是原因。
- **诊断 2（2 步冒烟，同一热身模型，4 卡，只差一个开关）**：`use_remove_padding=True`（113，对照）第 1 步 pearson 0.553、training_ppl 87.2 对 rollout_ppl 3.10、
  pg_clipfrac 0.098、梯度范数 60；**`use_remove_padding=False`（114）第 1 步 pearson 0.983、3.79 对 3.51、pg_clipfrac 0.00005、梯度范数 2.8**，第 2 步 0.983、4.36 对 4.02。
  关掉无填充打包后不一致基本消失；剩下的差（困惑度比 1.08，pearson 0.983，未达 DM-4 写的 0.99 / 1.05）与诊断 1 里单条前向本身 vLLM 对 HF 的差同量级，是两套实现的数值差。
- **热身 SFT 也受影响**：`sft_qwen3_8b.sh` 的 SFT 同样 `use_remove_padding=True`、`pad_mode no_padding`（打包）。Ministral 热身 SFT 的训练损失 1.17 → 0.7–1.0（第 1–50 步），
  家族 1 的同一步骤是 0.30 → 0.11–0.16；SFT 数据是模型自己的通过样本，正确前向下损失应很低。即热身模型是在错误的前向下优化的，剂量注入的效果不可信。
  因此 DM-3 的 prep（两折热身模型、评测、pass@64）全部作废，不只是 RL。
- **处理**：047 的第二折 prep（pass@64 剩种子 42）停止；不重试、不重启任何 Ministral 训练。家族 1、2（Qwen3 / Qwen3.5）不受影响：其 pearson 0.999、SFT 损失正常。
- **待用户决定（建议，不执行）**：在 `sft_qwen3_8b.sh` 与 `dapo_qwen3_8b.sh` 加一个默认不变的开关（`RMPAD`，默认 True，Ministral 设 False），从热身 SFT 起重跑两折 prep 与 RL；
  旧的热身模型、评测与 pass@64 目录改名保留（不删）以免链跳过；冒烟门槛按诊断 1 的数值底线定为 pearson ≥ 0.97、困惑度比 ≤ 1.10（此阈值在看到 0.983 / 1.08 之后才定，事后，需写明）。
  估计：prep 每折约 16 小时（关打包更慢，pass@64 仍是瓶颈）+ RL 每种子约 6–8 小时，三台机器并行约 1.5 天。

## 2026-10-31 执行记录 DM-6：用户决定重跑（方案 A）；SFT 受影响程度的更正；修正方案、门槛与排程（写于任何 Ministral 正确率或 pass@64 数字被读之前）

- **用户决定（11:10）**：重跑 Ministral，三台机器（113、114、047）。
- **对 DM-5 的更正**：DM-5 以"SFT 损失 0.7–1.0（家族 1 0.11–0.16）"为证说热身 SFT 是在错误前向下优化的。用同一数据、同一种子、只关打包重跑 F1 s42 的热身 SFT（113，
  `RMPAD=False`，`smoke-sft-ministral-rmpadFalse`）逐步比较：第 1 步 1.172 → 0.648，此后每步只低 0.02–0.05（如第 2–5 步 1.046 / 0.909 / 0.777 / 0.719 →
  0.887 / 0.839 / 0.721 / 0.654）。损失水平约 0.8 是 Ministral 自身的：它的 RAFT 样本在 T 1.0 下采得，困惑度高（rollout_ppl 3–11）。所以打包对 SFT 的影响小而系统，
  远小于对 RL 的影响（训练困惑度 127 对 11）；DM-5"剂量注入效果不可信"的说法过强。仍按用户决定从热身 SFT 起重跑，使每一步训练都在正确的前向下进行。
- **修正**：`train/sft_qwen3_8b.sh` 与 `train/dapo_qwen3_8b.sh` 新增 `RMPAD`（17eced7，默认 True，其它家族不变）；Ministral 全部 SFT 与 RL 设 `RMPAD=False`。
  其余配方与 DM-3 完全相同；worktree `rl-wt-17eced7` 与 `rl-wt-f8785fb` 只差这两个文件的这一处（`git diff --stat` 已核）。
- **门槛 1 补充（事后定，写明）**：RL 一致性门槛定为第 1–2 步 `training/rollout_actor_probs_pearson_corr` ≥ 0.97、`rollout_corr/training_ppl / rollout_ppl` ≤ 1.10。
  DM-4 写的 0.99 / 1.05 是在测量前提出的；关打包冒烟得 0.983 / 1.08，未达；新阈值在看到该数之后定，依据是诊断 1 中不经 verl 的单条前向 vLLM 对 HF 本身的差
  （平均 0.037 nats / token）。每次 RL 起跑后核对前 2 步，不过即停。
- **旧产物**：两折热身模型、评测、pass@64、两次作废 RL 的检查点改名加后缀 `.rmpad-void-20261031` 保留（不删，不进任何报告），以免链按"已完成"跳过。
  训练数据（`sft_dose-ministral3-8b-F*-s*.parquet`）由同一确定性构建产生，沿用；W_f 重建核对的 `.ok` 沿用。
- **排程（执行顺序变化，不改设计）**：114 跑第一折 prep、047 跑第二折 prep（两个种子的热身 SFT、评测、pass@64）；113 在第一折两个热身模型合并后
  从 114 复制（逐文件 sha256 核对）并跑第一折两个种子的 RL（`SEEDS="42 43"`）。RL 只需要热身模型，不读 pass@64，故第一折 RL 不等 pass@64。
  两折 prep 完成后，114 跑第二折种子 42、047 跑第二折种子 43 的 RL（各 `SEEDS=` 一个种子）。终点各在训练机上由链评测。

## 2026-10-31 执行记录 DM-7：第一折 RL 在第 0 步显存溢出（关打包后的补齐），减小微批次 token 预算后重启（写于任何 Ministral 结果之前）

- **事实**：113 上第一折两个种子的 RL（`RMPAD=False`，11:40 起）各两次尝试都在第 0 步的 `compute_log_prob` 里显存溢出（rc 1，单次申请 33–37 GiB），
  链按规则写 `STOPPED`（crash rc 1）后 12:05 结束，无检查点、无任何训练步。原因：关掉无填充打包后，每个微批次按最长行补齐，Ministral 的 131,072 词表
  logits 在每卡 24,576 token 的预算下放不下；冒烟（回复上限 1,024）没有触发。约 25 分钟 × 8 卡作废。
- **修正**：`train/dapo_qwen3_8b.sh` 新增 `PPO_MAX_TOKENS`（d29645b，默认 24,576 不变；所有 log-prob 前向的预算由它插值继承），Ministral 设 8,192。
  只改显存占用：同一 mini-batch 内按 token 平均，梯度相同，与 OPD-3 的同类处理一致。启动器改用 worktree `rl-wt-d29645b`（与 17eced7 相比，运行相关的代码只差
  `dapo_qwen3_8b.sh` 这一处）；第二折 prep 链（17eced7）不受影响。
- **处理**：113 上两个作废的检查点目录改名加后缀 `.oom-20261031` 保留；重启第一折 RL（`SEEDS="42 43"`），起跑后照 DM-6 核前 2 步一致性门槛。

## 2026-10-31 执行记录 DM-8：第一折种子 42 的 RL 在第 32 步停止——关打包后剩下的训练端 / rollout 端不一致不是"数值差"，而是比其它家族大约 100 倍，并随训练放大（写于任何 Ministral 正确率被读之前）

- **事实**（`python -m explore.audit_rollout_consistency --steps $ATMOS_ROOT/logs/dose/dose-ministral3-8b-F1-WA-s42.attempt1.nohup`）：
  第 1–2 步 pearson 0.990 / 0.988、困惑度比 1.080 / 1.070，过了 DM-6 的门槛（≥ 0.97 / ≤ 1.10）。此后：第 11–19 步 pearson 0.966 → 0.930（第 17 步）→ 0.957，
  低于同一门槛；困惑度比第 12 步 1.116、第 16 步 1.223、第 24 步 1.581。`rollout_corr/kl`（每 token）第 1 步 0.057，第 18 步 0.157；
  梯度范数 1.17（第 1 步）→ 6.6（第 17 步）→ 28.5（第 32 步）；`actor/pg_clipfrac` 0.0007 → 0.19；第 18 步起熵 0.57 → 2.2、平均回复长度 380 → 1,480 token、
  截断率 0.1 % → 41 %（第 28 步）。同一脚本对家族 1 的 `dose-qwen3-8b-F1-WA-s42`：pearson 0.999、比 1.000、kl 0.0002–0.0005、梯度范数约 0.35，全程不变。
  `--logs $ATMOS_ROOT/logs` 扫全部 106 次运行：除 Ministral 外 `kl_max` 全部 < 0.001（中位 0.0006）；关打包的 Ministral 冒烟 0.070，本次 0.05–0.17。
- **对 DM-6 的更正**：DM-6 把关打包后的 0.983 / 1.08 当作"两套实现的数值差"，并据此把门槛放宽到 0.97 / 1.10。拿其它家族一比（kl 约 100 倍），这个读法不成立：
  剩下的不一致本身就是缺陷，只是比打包小一个量级；而且它随策略变化放大，并在第 18 步后伴随熵与长度失控。DM-6 门槛只看前 2 步，因此没拦住。
  这次 RL 不是 §2 的 WA 处理，结果作废，不评测、不进任何报告。约 1.5 小时 × 8 卡作废。检查点（第 5–30 步）保留，只作诊断用。
- **处理**：13:56 写 `STOPPED`，按进程组停掉 113 的启动器 + 链（843550）与训练（858887），GPU 全部释放、无残留进程。种子 43 未起跑。
  **任何 Ministral RL 都不再启动**（含第二折，114 / 047），直到原因查清并写入新的执行记录。两折 prep 的评测与 pass@64 继续（只用 vLLM）；
  热身 SFT 是在 Hugging Face 前向下训练的，若查明是 HF 一侧错，prep 是否作废在下一条记录里定。
- **诊断（下一步）**：`explore/diag_logprob_match.py` 加 `--temperature 1.0`（按 RL 方式采样）、`--ref-attn eager --ref-dtype float32`（float32 eager HF 前向作参照）
  与按回复位置分段的差，在 113 单卡上对第一折种子 42 的热身模型跑；先看 vLLM 与 bf16 HF 哪一侧偏离 float32 参照、偏离是否随位置增长。
- **门槛今后改为**：每步 `rollout_corr/kl` ≤ 0.002（其它家族全部运行的最大值 0.0009 的约 2 倍；事后定，写明）且全程监控，不只看前 2 步。

## 2026-10-31 执行记录 DM-9：原因查明——vLLM 读文本检查点（HF 格式配置）时多乘了 YaRN 注意力幅度系数 1.277；Hugging Face 一侧正确；作废范围与修正方案（等用户决定；写于任何 Ministral 正确率被读之前）

- **诊断**（`explore/diag_logprob_match.py` da6d5bd，113 GPU 4，第一折种子 42 热身模型，8 道训练题，T 1.0 采样至多 2,048 token，float32 eager HF 作参照；
  `results/dose_ministral/diag_logprob_dm8.json`）：bf16 HF 对参照平均差 0.011 nats / token，vLLM 对参照 0.234（最大 5.5），vLLM 一侧的负对数似然系统地低 0.072。
  错的是 vLLM，不是训练端。
- **原因**：官方 `params.json` 的 YaRN 设置是 `apply_scale: False`（不乘注意力幅度系数）；HF 格式 `config.json` 用 `mscale = mscale_all_dim = 1` 表达同一件事，
  transformers 5.9 据此得 attention_factor 1。vLLM 0.24 读 HF 格式时（`rotary_embedding/__init__.py` 的 `yarn` 分支）不读 `mscale` / `mscale_all_dim`，
  `apply_yarn_scaling` 缺省 True，于是 cos / sin 乘 `yarn_get_mscale(16)` = 1 + 0.1 ln 16 = 1.277，注意力 logits 放大 1.63 倍。
  vLLM 只有在目录里同时有 `consolidated*.safetensors` 与 `params.json` 时才按 Mistral 格式读（`apply_scale` 映射为 `apply_yarn_scaling`）；
  官方目录满足，`explore/ministral_text_only.py convert` 生成的 `-text` 目录不含这两个文件，所以走了 HF 格式。
- **修正验证**（同一热身模型，`config.json` 的 `rope_parameters` 加 `"apply_yarn_scaling": false`，权重软链接；`results/dose_ministral/diag_logprob_dm8_yarnfix.json`）：
  vLLM 对参照平均差 0.234 → 0.0091，与 bf16 HF 对参照（0.0093）同量级；按位置分段 0.006 / 0.011 / 0.010，不随位置增长。
  未修正时 vLLM 采到的回复在正确模型下困惑度 26.1，修正后 3.7。transformers 对这个键只警告（"Unrecognized keys"），HF 前向不变（参照对 HF 0.011 → 0.0093）。
- **作废范围**（凡经 vLLM 以 `-text` 目录生成的东西）：基座四项评测 `x-ministral3-8b-base`（10-30 12:24）；两折热身模型的评测与 pass@64（10-31 的第二轮，14:10 停止）；
  第一折种子 42 的 RL（DM-8）；以及更早已作废的各轮。**不受影响**：§0 探针与设计（`results/x/reach_ministral3-8b.json`、`design.json`，用官方目录）；
  RAFT 采样（10-30 04:49–05:46，`-text` 目录 09:30 才生成）；热身 SFT 的训练（只用 HF 前向，正确），即四个热身模型的权重本身可用；
  D 系列的 Ministral 3 8B / 14B（官方目录，Mistral 格式；其 106 次运行外的另一套服务，不在 rollout 审计内，但按上面的格式判定规则不受影响）。
- **处理**：14:10 按进程组停掉 114 与 047 的两条 prep 链（启动器 + 链 + 两个 vLLM 服务；PID 核对后 `kill -TERM -<pgid>`），两台 GPU 全部释放；
  047 的 `/path/to/local/serve` 未碰。约 2.7 小时 × 4 卡（两台各 2 卡）评测的是错误模型。113 上的 OPD-13 冒烟（Qwen3-8B，不涉及 YaRN）继续。
- **修正方案（建议，不执行，等用户决定）**：
  1. `explore/ministral_text_only.py` 加一个 `fix-yarn` 子命令：给文本检查点与四个热身模型的 `config.json` 加 `"apply_yarn_scaling": false`（原文件改名 `.pre-yarnfix` 保留），
     并用 `diag_logprob_match.py` 在每个目录上核 vLLM 对 HF 平均差 ≤ 0.02。
  2. 链与 RL 合并后的终点模型同样检查该键（verl 合并时若丢键则补上，再核一次），RL 门槛按 DM-8：每步 `rollout_corr/kl` ≤ 0.002。
  3. 重跑：基座四项评测；两折热身评测与 pass@64（热身权重沿用，不重训）；四个 RL 种子。预计 prep 评测每折约 15 小时（两台并行），RL 每种子约 6 小时，基座评测约 2 小时。

## 2026-10-31 执行记录 DM-10：用户批准 DM-9 的修正（"按你的推荐做"）；配置已修正并逐目录核对通过；作废产物改名；重跑排程（写于任何 Ministral 正确率被读之前）

- **修正**（`explore/ministral_text_only.py fix-yarn`，44d3b64；`--params` 为官方 `params.json` 的副本，sha256 前缀 81b8377f，断言 `yarn.apply_scale` 为 false）：
  三台机器上文本检查点与热身模型的 `config.json` 加 `rope_parameters.apply_yarn_scaling: false`，原文件存为 `config.json.pre-yarnfix`。共 9 个目录：
  114 的文本检查点、F1 s42 / s43、F2 s42（副本）；047 的文本检查点、F2 s42 / s43；113 的文本检查点、F1 s42 / s43（副本）。修正后文本检查点的配置
  sha256 前缀 2ed76e6a、热身模型 4a65d0bd，三台一致。每个目录都核过：transformers 的 AutoConfig 读入再保存后这个键仍在，RL 检查点与合并后的终点会继承它。
- **逐目录核对**（`diag_logprob_match.py`，T 1.0，8 道训练题，至多 2,048 token，float32 eager 参照；`results/dose_ministral/yarnfix_check/*.json`）：
  vLLM 对 bf16 HF 的平均差——文本检查点 0.0109；F1 s42 0.0095、s43 0.0110；F2 s42 0.0093、s43 0.0111（门槛 ≤ 0.02），最大 0.18–0.30；
  float32 参照对 bf16 HF 0.0093–0.0108，同量级；按位置分段 0.006–0.012，不随位置增长；两侧困惑度一致到小数点后两位。
  F1 与 F2 的文本检查点核对是两台机器分别跑的（同一模型、两折 A 池前 8 题相同、vLLM 种子相同），两份输出逐字节相同。
- **作废产物改名**（后缀 `.yarn-void-20261031`，保留不删，不进任何报告）：`_x_eval` 下的 `x-ministral3-8b-base` 与四个热身标签；四个热身模型的 pass@64 文件（含 texts）；
  对应的 x_eval、server、passk 日志；`$ATMOS_ROOT/logs/dose-ministral3-8b-F1-WA-s4{2,3}.log` 与各次 attempt 日志（追加写，含作废轮次）；
  113 上第一折种子 42 的 RL 检查点目录。共 33 + 1 项。
- **排程**（不改设计；RL 不读 pass@64，同 DM-6）：先 RL——114 跑第一折、047 跑第二折（`launch-ministral-v2.sh`，`rl-wt-d29645b`，`SEEDS="42 43"` 顺序跑，
  终点由链评测）。每个种子起跑后核第 1–2 步、之后每小时核全程：`rollout_corr/kl` ≤ 0.002（DM-8）、pearson ≥ 0.99。RL 完成后两台各跑本折 prep
  （热身模型已合并，链跳过 SFT，只做评测与 pass@64）；基座四项评测在有空卡时补（`x_eval.sh`，`LABEL=x-ministral3-8b-base`，同 10-30 的命令）。

## 2026-10-31 执行记录 DM-11：按用户要求删除全部作废产物

- 用户 18:00 要求删除作废结果，以免污染最终结果。删除了所有带 `.rmpad-void-20261031`、`.yarn-void-20261031`、`.oom-20261031` 后缀的产物：
  共享盘 45 项（`_x_eval` 下的作废评测、作废的 pass@64 文件、对应日志与 attempt 日志），113 / 114 / 047 本地盘 15 项（作废的热身模型、RL 检查点），
  以及 113 上 DM-9 诊断用的软链接目录 `/path/to/local/diag-yarnfix`。逐项记录在 `$ATMOS_ROOT/logs/cleanup-2026-10-31.log`。
- 因此 DM-4 到 DM-8 引用的作废运行逐步日志已不存在，那些记录里的数字不能再从日志重算；诊断结论另有已提交文件
  （`results/dose_ministral/diag_logprob_dm8*.json`、`yarnfix_check/`），不受影响。保留的有：有效的四个热身模型、文本检查点（含 `config.json.pre-yarnfix`）、
  正在进行的 RL，以及门槛 1 的冒烟日志（只证明流程能跑，不是结果）。

## 2026-11-01 执行记录 DM-12：047 第二折种子 42 在第 81 步停顿约 35 分钟后自行恢复（系统负载，不是数值问题）

- **事实**：第 76–80 步每步生成 470–805 秒（114 同期约 350 秒）；第 81 步 23:36 起 128 条生成 0 条完成，持续到约 00:10；其间 047 的 sshd 不响应（外部与内网都在握手阶段超时），
  训练日志每分钟照常写。00:12 恢复登录：15 分钟平均负载 190 → 100、内存可用 1.7 TB、GPU 8 卡 100 %；3,024 个 root 进程，63 个 bwrap 处于 D 状态，
  内核 `kworker/...+netns` 处于 D 状态；00:11 生成重新推进（完成 17 / 128，0 失败）。读法：奖励沙箱（bwrap `--unshare-all`，每次执行新建网络命名空间）的创建快于内核的
  命名空间回收，积压拖住整机。第 1–80 步一致性不受影响（kl ≤ 0.00027、pearson ≥ 0.9997）。
- **处理**：不停、不重启（训练未崩，数值正常）；链的看门狗（bwrap > 300 或负载 > 1000）未触发。若复发或在种子 43 前仍有积压，种子 43 起跑前降低沙箱并发
  （`ATMOSCODER_SANDBOX_CONCURRENCY`，运行参数，不改设计）并记录。

## 2026-11-01 执行记录 DM-13：两个种子 43 改到 113 跑（排程变化，不改设计；写于任何 Ministral 结果之前）

- **为什么**：两台的种子 42 都在变慢——每步生成 114 从约 300 秒升到 720 秒、047 升到约 800 秒（DM-12 的 bwrap / 网络命名空间积压，00:40 时 047 仍有 61 个 bwrap 处于 D 状态）；
  按现速度种子 42 约 09:00–10:00 训完，各机再接种子 43 要到 11-02 下午。113 的 OPD-13 链约 04:00 结束后空闲，机器较"干净"。
- **做法**（同 DM-6 的拆分方式）：00:50 在 114 的 `dose-ministral3-8b-F1-WA-s43`、047 的 `dose-ministral3-8b-F2-WA-s43` 检查点目录写 `STOPPED`（"split: run on 113 (DM-13)"），
  两条链训完种子 42 后跳过种子 43、只评测种子 42 的终点。113 上 `$ATMOS_ROOT/scratch/s43-113.sh`：从 047 复制 F2 s43 热身模型（逐文件 sha256 核对、确认含
  `apply_yarn_scaling: false`；F1 s43 的副本已在 113 并已修正），等 `OPD CHAIN DONE` 且无评测进程后，依次跑 F1、F2 种子 43（`launch-ministral-v2.sh`，`SEEDS=43`，
  链日志 `rl-ministral3-8b-F{1,2}-s43.113.log`），终点由链评测。114 / 047 训完种子 42 后接本折 prep。
- **24 小时上限**：各种子各自计时（链的 `WALL_SECONDS`），种子 42 按现速度在上限内；若触及上限，按 X-4 规则以最后检查点为终点并记录。

## 2026-11-01 执行记录 DM-14：变慢的主因是 DAPO 动态采样（训练奖励升高），不是故障；排程改为种子 43 留在原机、113 跑基座评测与两折 prep（取代 DM-13 的排程）

- **原因更正**：DM-12 把变慢归因于沙箱的网络命名空间积压。bwrap 确实卡在 `rtnetlink_rcv_msg` / `rtnl_lock`（114，02:00），但 113 上同样有 57 个 bwrap 处于 D 状态而负载只有 15，
  这只是叠加因素。主因是动态采样：`filter_groups` 丢弃整组同分的题，奖励升高后（第一折 0.51 → 0.85，第二折 0.43 → 0.81）整组全对的题变多，每步要生成多批才凑满 128 组；
  `max_num_gen_batches=10` 被 verl 内置的 V1 ReplayBuffer 忽略（日志原文 "num_gen_batches=10 is ignored by the built-in V1 ReplayBuffer"），所以没有上限。
  第二折第 84 步生成用了 4,127 秒。配方与家族 1 相同，不改。
- **后果**：两个种子 42 按现速度会在 24 小时上限（11-01 16:22）前到不了 120 步，按 X-4 规则以最后检查点为终点并如实记录终点步数；种子 43 同样以上限为界。
- **排程（取代 DM-13）**：02:45 停掉 113 上的 `s43-113.sh`（尚未启动任何训练），撤掉 114 / 047 上种子 43 的 split 标记——两个种子 43 仍在原机、种子 42 之后顺序跑。
  113 上 `$ATMOS_ROOT/scratch/prep-113.sh`：F2 s42 热身模型从 114 复制（114 的副本在 DM-6 时与 047 逐文件核过 sha256、DM-10 修正过；02:53 复制并核对通过），
  五个模型目录都核过 `apply_yarn_scaling: false`；等 `OPD CHAIN DONE` 后依次跑基座四项评测（与 10-30 同一命令）、第一折 prep、第二折 prep（热身模型已合并，只做评测与 pass@64）。
  F2 s43 的 113 副本（DM-13 复制）供第二折 prep 使用。

## 2026-11-01 执行记录 DM-15：基座评测与第一折 prep 完成、门槛 3–5 通过；第二折 prep 因锁的时序冲突晚 15 分钟启动（写于任何 Ministral 正确率或 pass@64 数字被读之前）

- **基座四项评测**（113，04:19–04:56，`rl-wt-b0da125`，与 10-30 同一命令，修正后的文本检查点）：final 5,600、judge 2,800、test_id 64、textbook 92，`EVALDONE`。
- **第一折 prep**（113，04:56–13:14，`launch-ministral-v2.sh`；两个热身模型已合并，链跳过 SFT）：两个热身模型评测各 5,600 / 2,800 / 64 / 92、0 错误行；
  `serving.json` 指向修正后的 `sft-dose-ministral3-8b-F1-s4{2,3}/global_step_59`；服务日志解析为 `Ministral3ForCausalLM`，无模板 / 分词器类 ValueError；
  pass@64 文件各 1,400 行、每行 n = 64，`PASSKDONE`。pass@64 日志末尾的沙箱统计里有 "fail: ValueError: could not convert string to float"（s43 13 次）等，
  是模型所写解答的运行错误（与 NameError、TypeError 并列），不是基础设施错误。修正后 pass@64 每个模型约 6–8 小时（作废轮次的错误模型约 13 小时）。
- **第二折 prep**：`prep-113.sh` 在 13:14:06 第一折链退出的同一秒启动第二折，链的锁（`flock`）尚未释放，第二折按规则拒绝启动（"another dose chain is running here"）。
  13:28 核对：无进程持锁、GPU 全空；13:29 以同一启动器重启（失败的日志改名 `prep-ministral3-8b-F2.yarnfix.attempt1-locked.log` 保留），两个热身模型跳过 SFT，进入评测与 pass@64。

## 2026-11-01 执行记录 DM-16：变慢的根因是 114 / 047 上新建网络命名空间变慢（主机状态），DM-14 的归因更正；奖励未被污染；种子 43 的判断规则（写于任何 Ministral 结果之前）

- **测量**（14:40，每台 5 次 `bwrap --unshare-all ... /bin/true`，对照 `--share-net`）：113 0.06 秒 / 0.02 秒；114 **82.7 秒** / 0.58 秒；047 **48.9 秒** / 0.29 秒。
  即在 114、047 上每新建一个网络命名空间要 10–16 秒（113 约 0.01 秒），不新建则不受影响。训练中每小时沙箱调用数：114 从 77,521（02 时）降到 10,747（14 时），
  047 从 37,298 降到约 14,000；第 94 步 GPU 大多空闲，128 条请求都在等奖励沙箱。DM-14 "主因是动态采样"更正为：主因是这两台主机上命名空间创建随运行累积变慢
  （内核 `rtnl_lock`，DM-12），动态采样只是让每步的沙箱调用更多。
- **奖励没有被污染**（`$ATMOS_ROOT/logs/x/reward_diag/` 全部记录求和）：第一折 169 万次调用中墙钟超时 528 次、全部在 X-24 的重跑中恢复，未恢复 1 次；
  第二折 125 万次中 195 次、全部恢复，未恢复 0 次；对照 `dose-qwen3-8b-F1-WA-s42` 145 万次中未恢复 1 次。慢只影响速度，不影响判分。
- **种子 43 的判断规则**（写于测量之前）：种子 42 在 16:22 到上限后、种子 43 开始采样前，在每台重测 5 次 `bwrap --unshare-all`。< 5 秒 → 种子 43 在原机照常跑；
  ≥ 5 秒 → 写 `STOPPED` 并按进程组停掉该机的种子 43，待 113 第二折 prep 完成后在 113 上依次跑（113 测量为 0.06 秒），并记录。
  不改沙箱的网络隔离（`--share-net` 会改变奖励环境，与家族 1 不同）。

## 2026-11-01 执行记录 DM-17：用户选方案 A——种子 43 的沙箱改为共享主机网络命名空间（不改判分）；等价性验证通过；排程（写于任何 Ministral 结果之前）

- **用户决定（15:30）**：DM-16 列出的四个方案中选 A：给沙箱加默认关闭的开关，只对 Ministral 的两个种子 43 打开，部署前做等价性验证、任一项不过即退回"原机照跑"。
- **改动**（be6d67d）：`atmoscoder_rl/sandbox.py` 新增 `ATMOSCODER_SANDBOX_SHARE_NET`（默认 0，其它家族与种子 42 不变）；为 1 时 bwrap 加 `--share-net`，不再为每次执行新建网络命名空间；
  文件系统隔离不变。联网仍被下一层挡住：审计钩子拒绝一切 `socket.` / `urllib.` / `http.client.` 事件以及子进程、`ctypes`，`RLIMIT_NPROC` 为 1。
  `isolation()` 新增 `share_net` 与 `network_blocked_by_audit_hook`，训练开始时记入日志。与种子 42 所用 `rl-wt-d29645b` 相比，训练相关代码（`train/`、`atmoscoder_rl/`、
  `dose_chain.sh`、`x_eval.sh`、`x_passk.sh`、`dose.py`、设计文件）只差这一个文件（`git diff --stat` 已核）。
- **等价性验证**（113，`explore/sandbox_sharenet_check.py`，`results/dose_ministral/sharenet_check_{0,1}.json`）：从第一折种子 42 热身模型的 pass@64 样本中每题抽 3 条
  （`random.Random("sharenet:<id>")`，共 4,200 条），用 pass@64 同一判分路径重判：开关关 907 / 4,200 通过、与存档判定 0 处不同；开关开 907 / 4,200、0 处不同。
  两个探针（连 socket；按绝对路径读答案文件）在两种设置下都得 0。`atmoscoder_rl.check_reward` 在开关打开时全部通过（读答案文件两种路径都被挡、沙箱与引擎在 11 个金标解答上一致、
  并发上限、失败计数、bwrap 启动失败的回退）。三项都过，部署。
- **排程**：15:40 在 114 / 047 的种子 43 检查点目录写 hold 标记（`STOPPED`："hold: relaunch with share-net sandbox"），两条现有链在种子 42 到上限、合并、评测终点后结束，
  不以旧设置起种子 43。`$ATMOS_ROOT/scratch/s43-sharenet.sh <折>`（114 第一折、047 第二折）等链的 `DONE` 且无 `dose_chain.sh` 进程后，核对标记内容、删除标记，
  以 `launch-ministral-v3.sh`（`rl-wt-be6d67d`、`ATMOSCODER_SANDBOX_SHARE_NET=1`，其余设置同 v2）跑 `SEEDS=43`，链日志 `rl-ministral3-8b-<折>-s43.sharenet.log`。
- **报告时须写明**：种子 42 以旧沙箱设置在 24 小时上限处结束（终点约第 85–90 步），种子 43 以新设置运行；判分函数在两种设置下逐条相同（上述验证），差别只在训练吞吐。

## 2026-11-01 执行记录 DM-18：两个种子 42 到上限收尾；第二折终点评测完成、种子 43 起跑；更正 DM-17——共享网络开关实际未生效，种子 43 与其它全部运行用同一沙箱

- **上限与合并**（链日志 `rl-ministral3-8b-F{1,2}.yarnfix.log`）：第一折种子 42 16:22:43 触发 24 小时上限（rc 124），按 PID 停掉进程，16:27:08 **在第 95 步合并**；
  第二折种子 42 16:22:52 触发，16:26:00 **在第 90 步合并**。两条链都按 hold 标记跳过种子 43（"no checkpoint"）后进入终点评测。按 X-4 规则，终点即这两个检查点。
  两个合并日志里都有 transformers 对 `apply_yarn_scaling` 的"Unrecognized keys"警告（即检查点配置带着该键）。
- **第二折（047）**：16:56 核对——终点 `global_step_90/config.json` 的 `apply_yarn_scaling` 为 false，无 ppid = 1 的 vLLM 孤儿进程，等待脚本在。
  终点评测 17:48 完成：final 5,600、judge 2,800、test_id 64、textbook 92，0 错误行，`serving.json` 指向 `global_step_90`（数字未读）。17:49 等待脚本删除 hold 标记、
  以 `launch-ministral-v3.sh` 起种子 43；18:28 第 14 步，pearson ≥ 0.9996、kl ≤ 0.00026，每步约 138 秒（生成约 72 秒）。
- **第一折（114）**：16:27 后无法 ssh（外部与内网都超时），共享盘上无任何新写入；终点评测的推理服务日志停在 16:30（架构 `Ministral3ForCausalLM` 已解析，未见权重加载）。
  终点配置与孤儿进程待 114 恢复后核对；已通知用户。终点模型在 114 本地盘，不能移到别机评测。
- **更正 DM-17**：18:29 在 047 上读正在运行的 50 个 bwrap 进程的命令行，全部是 `--unshare-all`、**没有** `--share-net`。原因：奖励函数由
  `reward.custom_reward_function.path="$ATMOS_REPO/atmoscoder_rl/verl_reward_regime.py"` 加载，`PYTHONPATH` 由 `env.sh` 置为共享 checkout
  （`/path/to/AtmosCoder-RL`，4791584，10-23），其 `sandbox.py` 没有该开关；`TRAIN_REPO` 只决定训练脚本从哪里运行。
  因此 be6d67d 的开关从未被加载，**种子 43 与种子 42、家族 1 / 2 的全部 RL 用同一份奖励与沙箱代码**；DM-17 的等价性验证本身成立但与运行无关。
  047 上新建网络命名空间的 bwrap 18:29 测得 0.02 秒（种子 42 停下后内核积压已清空），所以种子 43 当前很快。决定不改、不重启：环境与全部运行一致最干净；
  若主机再次积压变慢，按上限规则以最后检查点为终点并记录。报告里不需要写"种子 43 换了沙箱设置"。

## 2026-11-01 执行记录 DM-19：114 的 Lustre 客户端卡住约 2.5 小时后自行恢复；第一折种子 42 的终点核对通过、评测继续

- **114 核对**（18:55 恢复登录）：终点 `dose-ministral3-8b-F1-WA-s42/global_step_95/config.json` 的 `apply_yarn_scaling` 为 false；无 ppid = 1 的 vLLM 孤儿进程；
  种子 43 的等待脚本（3919419）在。
- **卡住的原因**：终点评测的推理服务 16:28 启动后，三个 `VLLM::Worker` 处于 D 状态，`wchan` 为 `wait_on_page_bit_common` / `cl_sync_io_wait`（Lustre 客户端 I/O 等待）——
  读共享盘 `/path/to/shared` 时被挂住；同期 114 的 sshd 不响应、15 分钟负载约 198，GPU 0–3 只占约 529 MB（权重未载入）。这次不是 DM-12 / DM-16 的沙箱命名空间问题（bwrap 0 个）。
  19:04 推理服务开始正常应答，评测继续；未做任何干预。训练与合并在卡住之前已完成，不受影响。

## 2026-11-01 执行记录 DM-20：第一折种子 42 终点评测完成；第二折 prep 完成、门槛 3–5 通过；两个种子 43 都在跑（写于任何 Ministral 正确率或 pass@64 数字被读之前）

- **第一折种子 42**：114 的 Lustre 卡顿恢复后（DM-19），终点评测 20:15 完成：final 5,600、judge 2,800、test_id 64、textbook 92，0 错误行，`serving.json` 指向 `global_step_95`。
  链 `DONE (rc 0)` 后等待脚本删除 hold 标记，20:15:50 起第一折种子 43（`launch-ministral-v3.sh`；奖励与沙箱同全部运行，DM-18）。启动约 13 分钟（114 读共享盘仍慢），
  21:15 第 11 步，pearson ≥ 0.9998、kl ≤ 0.00023，每步约 175 秒。
- **第二折 prep**（113，13:29–21:05）：两个热身模型评测各 5,600 / 2,800 / 64 / 92、0 错误行；`serving.json` 指向 `sft-dose-ministral3-8b-F2-s4{2,3}/global_step_65`；
  服务日志解析为 `Ministral3ForCausalLM`，无 ValueError；pass@64 各 1,400 行、每行 n = 64，`PASSKDONE`。门槛 3–5 通过。核对无进程打开后删除 113 上这两个热身模型的副本
  （原件在 047，已核存在），记入 `cleanup-2026-11-01.log`。
- **第二折种子 43**：21:15 第 58 步，pearson ≥ 0.9996、kl ≤ 0.0003，每步约 310 秒（随奖励升高变慢）。
- 剩余：两个种子 43 跑到 120 步或 24 小时上限（第二折 11-02 17:50、第一折 11-02 20:16），各自评测终点；之后按 §5 出报告。

## 2026-11-02 执行记录 DM-21：第二折种子 43 到上限、在第 105 步合并并评测完成；单种子重启使链在合并后退出，评测改由补跑脚本完成；第一折种子 43 看门狗中断后续跑（写于任何种子 43 正确率数字被读之前）

- **第二折种子 43（047）**：11-02 17:50:14 触发 24 小时上限（rc 124），按 PID 停掉进程；**终点第 106 步记录、第 105 步检查点**，墙钟累计 1,440 分钟，平均截断率 0.030。
  18:31 合并为 `global_step_105`，`config.json` 的 `apply_yarn_scaling` 为 false，无 ppid = 1 的 vLLM 孤儿进程。全程 106 步 kl ≤ 0.0003、pearson ≥ 0.9993，OOM / Traceback 0，
  奖励日志中 "timed out" 0 次。
- **链在合并后退出**：种子 43 是用 `scratch/s43-sharenet.sh` 以 `SEEDS=43` 重启的，而 `dose_chain.sh` 的终点评测读 `$2`（第二个种子），在 `set -u` 下报
  `line 186: $2: unbound variable` 退出，未起评测。训练与合并不受影响。补救：`scratch/s43-evalfix.sh <折>` 等本机种子 43 链退出、核合并终点的 yarn 键与孤儿进程后，
  以 `launch-ministral-v2.sh`（`rl-wt-d29645b`，与两个种子 42 的评测同一份代码）和 `SEEDS="42 43"` 重跑 rl 阶段：两个种子都已合并，`train()` 跳过；
  `x_eval.sh` 的 check 跳过已完成的种子 42 评测，只评种子 43。第二折 18:36 手动按此启动。
- **一份日志被清空**：ssh 重试循环把启动命令发了两次；第二次被 flock 拒绝（"another dose chain is running here"），但它的 `>` 重定向已截断
  `logs/dose/rl-ministral3-8b-F2-s43.evalfix.log` 的前段。评测本身的日志 `logs/x_eval/dose-ministral3-8b-F2-WA-s43.log` 完整。此后启动先用精确 `ps` 匹配检查再启动。
- **第二折终点评测** 19:32 完成（`DONE (rc 0)`）：final 5,600、judge 2,800、test_id 64、textbook 92，0 错误行，`serving.json` 指向 `global_step_105`，commit d29645b（数字未读）。
- **第一折种子 43（114）**：11-02 18:45:44 训练看门狗触发（bwrap 101 个、负载 1,046，超过阈值 1,000），按进程组停掉第一次尝试，停在第 90 步检查点（记录到第 92 步），
  墙钟累计 1,352 分钟、平均截断率 0.070。链按 X-10 续跑规则 18:48:52 从第 90 步起第二次尝试，剩余 5,259 秒（约 20:16 到上限）。前 92 步 kl ≤ 0.0003、pearson ≥ 0.9993，OOM 0。
  合并后同样会因 `SEEDS=43` 退出；`s43-evalfix.sh F1`（114，PID 469797）在等，将自动补跑评测。
- **变慢的原因**同 DM-16：两台主机上每个 bwrap 新建网络命名空间约 10–11 秒，种子 43 后段每步 6,000–10,700 秒。

## 2026-11-02 执行记录 DM-22：第一折种子 43 到上限、终点第 90 步，评测完成；四个终点全部评测完（写于任何种子 43 正确率数字被读入报告之前）

- **上限与合并**：第二次尝试（DM-21）11-02 20:16:45 触发 24 小时上限（rc 124），按 PID 停掉进程；这次尝试未记录任何新步，**终点为第 90 步检查点**
  （共记录 92 步，平均截断率 0.070）。20:27:40 合并为 `global_step_90`，`apply_yarn_scaling` 为 false，无 ppid = 1 的 vLLM 孤儿进程。
  全程 92 步 kl ≤ 0.0003、pearson ≥ 0.9993，OOM 0，两次尝试的奖励日志中 "timed out" 0 次。
- **评测**：链合并后同 DM-21 因 `SEEDS=43` 退出；`s43-evalfix.sh F1` 20:27:55 按其检查启动 v2 评测，22:07:05 `DONE (rc 0)`：final 5,600、judge 2,800、test_id 64、textbook 92，
  0 错误行，`serving.json` 指向 `global_step_90`，commit d29645b。巡检时评测日志尾行带出了 final 的汇总正确率，未用于任何决定。评测后 114 上无本人进程、GPU 空闲。
- **终点汇总**：第一折 s42 第 95 步、s43 第 90 步；第二折 s42 第 90 步、s43 第 105 步。四个终点均为 24 小时上限截断（X-4），均未到 120 步。下一步按 §5 出报告。

## 2026-11-03 执行记录 DM-23：报告与判定（H1 不成立、H2 不可检验、p 规则上半句复现）；复核发现；结果入库

**报告**（`results/dose_ministral/report_ministral3-8b.json` b3a112e4、`posthoc_ministral3-8b.json` 3fc8c932；2026-11-02 在 113 的 worktree `rl-wt-d5c20fa` 连跑两次，JSON 与打印输出逐字节相同；
原始数据本地副本 `experiments/dose_ministral/`，290 个文件与服务器 sha256 逐一核对）。24 个观测齐全，无基础设施错误。

- **H1 不成立**：RL（剂量 ≥ 1）− RL（剂量 0）= +50.0 点，[−20.0, 94.4]（门槛 30 点且下界 > 0）。分折分种子：F1-s42 +100、F1-s43 +50、F2-s42 +30、F2-s43 +50，全正。
  区间宽的原因：剂量 0 只有 3 个机制、6 个观测（ch4_forcing_sqrt、henry_depletion、doppler_aliasing，三者两个种子都是剂量 0），其中 doppler_aliasing 两个种子都学会。
- **H2 不可检验**：热身后 pass@64 = 0 的观测 0 个（规则允许基座 pass@64 ≤ 0.1，RAFT 热身又把 p 推高）；pass@64 > 0 的 18 个学会 15 个。
- **各剂量**：剂量 0 学会 2/6（RL 33.3 %）、1 4/5（80.0）、2 4/6（66.7）、4 7/7（100）。RL − 热身（加示例）+49.2 [26.9, 72.8]；热身低于 50 % 的子集 +67.5 [42.8, 88.1]。
- **p 规则（§5，只报数字）**：p ≥ 0.02 学会 10/10（家族 1 12/12），0.005–0.02 2/2（3/6），< 0.005 3/6（1/8）。上半句"p ≥ 0.02 全学会"复现，可写成跨家族的规律；
  下半句"p < 0.005 几乎不学会"不复现，不写。最低档内学会的 3 个 p = 0.0023 / 0.0031 / 0.0047，未学会的 3 个 0.0023 / 0.0023 / 0.0047——p 在这一档内不区分成败。
- **RL 失败 7 个观测（3 个加了示例）**：终点在 20/20 道越界题上全给标准公式值，热身时为 1 / 1 / 5 / 5 / 6 / 13 / 20 题（家族 1 18/18 同样）。
- **跨家族并表**（§5 最后一项；`explore/dose_family_compare.py` → `family_compare.json` f66dbb15，本地跑两次逐字节相同）：共同机制 11 个。两家族都给过示例的 8 个机制里 7 个在两家族都学会，
  只有 diffuse_transmission 一家族（Qwen）没学会；henry_depletion 在 Qwen 给 2 例两个种子都不会、在 Ministral 剂量 0 不会。两家族剂量 0 都学会 0/8 对 2/6。

**复核发现（事后，只报数字）**

1. **单例剂量是高方差干预。** diffuse_transmission 两个种子都是剂量 1，注入的两道示例（`kc:train:20260930_diffuse_transmission_20 / _07`）参数相近
   （τ 0.374 / 0.331，都在评测越界行 τ 0.30–0.49 内）、答案文本几乎相同（都是模型自己的"给出"版解答，`F0 * exp(-1.66 * tau)` 加一句"diffusivity approximation"），
   热身后 p 却是 0.5016 对 0.0023（642 对 3 / 1,280），热身正确率 0.95 对 0.00，RL 1.00 对 0.00。差别只在示例身份与其它单位的剂量（F1 两种子的热身数据 680 对 676 行）。
   同剂量的另三对差 2–3 倍（clausius 0.18 / 0.11、inertia 0.005 / 0.007、snowpack 0.07 / 0.19）。单例，不作结论。
2. **通用热身可自带火种。** doppler_aliasing 基座 pass@64 0/20 行（设计表），不含本单位示例的 RAFT 热身后 p 0.0109 / 0.0023（14 / 3 命中），RL 两个种子都到 1.00；
   第一折的热身模型（训练数据完全不含第二折模板）在 doppler 上也有 8 / 14 命中。所以本家族"剂量 0"不是干净的无火种对照。Qwen 的剂量 0 doppler（p 0）未学会。
3. **热身伤留出、RL 修回**（整体正确率，右 / 可判，`family_compare.json` overall_accuracy）：Ministral 基座 → 热身 → RL：final 42.3 → 45.7–49.5 → 63.2–73.9；
   test_id（64 行）45.3 → 34.4–40.6 → 46.9–56.2；textbook（92 道）25.0 → 17.4–20.7 → 29.3–31.5。Qwen：final 29.0 → 43.2–46.5 → 63.2–67.6；test_id 34.4 → 31.2–40.6 → 53.1–57.8；
   textbook 16.3 → 9.8–16.3 → 26.1–32.6。两家族同向；test_id 64 行的区间约 ±12 点。
4. **训练动态**：Ministral 训练奖励 0.43–0.51 → 0.81–0.85（最后 10 步均值 0.81–0.84，已平台），截断率 0.03–0.07，回答长度 630–780 → 1,170–1,310 token；
   Qwen 0.52–0.56 → 0.63–0.76，截断率 0.01–0.02。Ministral 的热身后 p 中位数 0.048（Qwen 0.015），基座整体也更强（final 42.3 对 29.0）。
5. **执行无偏离**：四个终点被 24 小时上限截断（第 95 / 90 / 90 / 105 步，计划 120），奖励已平台；一致性门全程通过；评测代码同一份（基座评测 worktree b0da125 与 d29645b 的评测代码无差别，其间只有 `train/dapo` 的 `PPO_MAX_TOKENS` 开关）；
   沙箱与奖励代码与全部 RL 相同；奖励超时 0。

**设计教训（写进局限，不再补实验）**：随机分配剂量时应保证每个机制两个种子里至少一次剂量 0、一次 ≥ 1，否则小家族没有机制内翻转（本家族 0 个）；
入选规则允许 pass@64 ≤ 0.1 加 RAFT 热身，H2 的 0 组注定为空，两家族都不可检验；剂量 0 对照应排除通用热身能自带火种的机制，或改以"热身后 p"为协变量。

**入库**：`results/dose_ministral/`（README、report、posthoc、family_compare）；PAPER_OUTLINE §2 第四幕、§3、§5、§6（F8、F15 更新，新增 F18）；`results/README.md`、`docs/README.md`。
