# 预注册：在线策略蒸馏能否越过可达性屏障（OPD）

**状态：2026-10-29 用户审阅通过（"review下，没问题就可以"），同日定稿；此后只追加带日期的执行记录，不改定稿内容。** 开训前仍须过第 4 节的门槛。
**2026-11-03 注**：已完成，记录到 OPD-17；跑过的臂 P（两种子）、S、Pk3、Pk2、Pshuf，T 臂与第二家族未跑（OPD-4、PAPER_OUTLINE §5 冻结）。
已看过的结果：X 全部、DOSE 全部、`results/paper/` 全部；**未看过**：任何 OPD 运行、Qwen3-32B 在模板上的任何回答（第 4 节的探针在定稿后才跑）。

## 1. 问题

X 的 P1 与 DOSE 表明：Qwen3-8B 基座 64 次采样都写不出替代关系的机制（Z，14 个），答案奖励的 RL 学不会；注入 1–4 个正确示例即可。
两种解释：(a) **没有知识**——模型里没有替代关系；(b) **信号太稀**——知识在（D：题面给出替代关系即可做对），但 0/1 奖励在采不到正确做法时没有梯度。
OPD 把奖励换成教师对学生每个 token 的稠密打分，并能在**学生没采到的 token** 上给信号（教师 top-K 前向 KL），因此能把两种解释分开。

## 2. 臂（Qwen3-8B，关思考；两折 F1 / F2，种子 42；主臂 P 另跑种子 43）

| 臂 | 教师 | 教师看到的题面 | 蒸馏损失 | 其余 |
|---|---|---|---|---|
| A | — | — | — | 已有（`x-qwen3-8b-F*-A-s42/43`），不重跑 |
| **P** | Qwen3-8B 基座（冻结） | 越界行：其 `given` 孪生的题面（多一句替代关系）；其他行：学生题面 | `forward_kl_topk_tail`，K = 32，GKD（`use_policy_gradient: false`） | 叠加 A 的答案奖励（`use_task_rewards: true`，系数 1.0） |
| S | 同上 | 一律学生题面（无特权） | 同 P | 同 P——控制"对冻结基座的 KL 正则"本身 |
| Pk3 | 同 P | 同 P | `k3`（只在学生采到的 token 上） | 同 P |
| T | 外部教师：Qwen3-32B 或 Qwen3-30B-A3B-Instruct-2507（两者 `tokenizer.json` 与学生逐字节相同；第 4 节探针选会得多的那个） | 学生题面 | 同 P | 同 P；**只在探针过门槛时跑** |

**第二家族（复现，条件性）**：学生 Qwen3.5-9B，臂 P2（特权自教师）与 T2（教师 Qwen3.8-27B：常规词表 248,044 个 token 与合并规则和 Qwen3.5-9B 完全相同，只多 7 个音频 / TTS 特殊 token，代码回答中不出现）。Qwen3.5-9B 的 Z 只有 5 个，主检验集放宽为"基座 pass@64 ≤ 0.10 且 A 两种子均值 < 0.5"（8 个）；只报数字，不作判定。T2 同样先探针（门槛同 §4.2，按放宽后的集合算）。Qwen3.5 的教师与学生都要带 X-12 的 GDN Triton 内核（教师 vLLM 的 `engine_kwargs` 同样设置）。Qwen3.8-27B 只从共享盘读权重，不使用 114。

**与 A 的两处必要差别（写定）**：(i) **关闭动态采样**（`algorithm.filter_groups.enable=False`，`gen_batch_size = train_batch_size = 128`）：DAPO 会剔除 8 次 rollout 奖励全相同的题组，而 Z 机制的越界题正是全 0 的组——不关掉，蒸馏信号永远到不了要检验的题。这些组在 A 里对策略梯度本来就贡献零优势，所以差别只在每步的题组构成；S 臂与 P 同设置，P − S 不受影响，P − A 含此差别，读法里注明。(ii) **GPU 划分**：教师占独立池（8B 自教师 1 卡 TP1；32B / 30B-A3B 2 卡 TP2），学生用 4 卡（verl 按 DP 整除切分 1,024 条序列，6 卡不整除）；全局 batch、mini-batch、rollout 数不变，每步墙钟约为 A 的两倍，OPD 臂的墙钟上限定为 **36 小时**（X-4 的其余规则不变）。除此之外一切与 A 相同：`data-x/F*-A` 的同一 train.parquet（P 只给 580 条越界行加 `teacher_prompt`，`train/opd/build_opd_data.py`）、
DAPO 设置、120 步、24 小时上限（X-4）、续跑规则（X-10）、沙箱退回（X-16）、评测（`explore/x_eval.sh`，同一 `final.json` 与 textbook 集）。
实现：`train/opd/patch_verl_privileged.py` 在**独立的 verl 工作树**（同一提交 5b79827b）上打补丁，经 `PYTHONPATH` 优先加载；不改共享 `$R/verl`。
教师在单独的 GPU 池上（8B：1 卡 TP 1；32B：2 卡 TP 2），学生用其余卡；全局 batch（128 题 × 8）不变，只改每卡分摊。

## 3. 假设与读法（单位 = 机制；区间按机制重抽 10,000 次，`random.Random("opd:<项>:20261029")`）

主结局：Z 机制（`results/x/report_qwen3-8b.json` 的 P1，基座 pass@64 = 0）在训练折越界题上的正确率，两折合并、该臂全部终点平均。
主检验集 = Z ∩ **特权教师会**：基座在该模板 `given` 行上的正确率 ≥ 0.5，取自 W_f 的热身采样（`experiments/_x/qwen3-8b/warm_samples_F*.jsonl`，每行 8 次、T 1.0，
已存档、未按此口径读过）；第 4 节第 3 条在开训前算出并记录这一集合。D 的存档只覆盖 9 个 Z 机制（given ≥ 0.5 的 6 个），不作主口径。
H-OPD1–3 在主检验集上判定；全部 14 个 Z 机制并列报告；主检验集少于 5 个机制时 H-OPD1–3 只报数字。

- **H-OPD1（主）**：P − A ≥ 30 点且区间下界 > 0 → "稠密的特权信号让 RL 学不会的机制学会：缺的不是知识而是信号"（支持解释 b）。
  若 P − A 区间含 0 或点估计 < 10 → "即使教师在每个 token 上给出知道替代关系时的分布，模型也学不会：屏障不只是信号稀疏"（支持解释 a）。其他只报数字。
- **H-OPD2**：P − S ≥ 30 点且下界 > 0 → P 的增益来自特权信息，不是 KL 正则。H-OPD1 成立而 H-OPD2 不成立时，H-OPD1 的读法改为"只报数字"。
- **H-OPD3**：Pk3 − A 的区间含 0 或点估计 < 10 → "只在学生采到的 token 上打分的蒸馏同样被可达性限制"；P − Pk3 ≥ 20 点 → "越过屏障靠的是对未采样 token 的信号"。
- **H-OPD4（T，过门槛才检验）**：按探针把 Z 机制分为教师会（所选教师越界 pass@1 ≥ 0.5）/ 不会；学会（训练折越界 ≥ 0.5）的比例之差 ≥ 0.5 且 Fisher p < 0.05 →
  "OPD 只能传递教师自己有的知识"。

次要（只报数字）：留出折越界与原题的增益（`x_report.transfer` 口径）；全部 70 机制的 P − A；只对错答的标准公式值份额（`paper_rigor_eval` 口径，看 OPD 后是否也"更一致地错"）；
原题正确率下降 > 5 点即报告；46 道教科书 regime 陷阱（`traps_rl_report` 口径）。P 的种子 43 只用于报告 H-OPD1 的跨种子一致性，不改判定。

## 4. 开跑前的门槛（定稿后执行，结果记入执行记录后才开训）

1. **冒烟**：P 与 T 各 2 步（F1），检查：补丁的模板自检通过（`ATMOS_OPD_PRIVILEGED=1`，`extra_info` 缺失或前 2,048 条样本无 `teacher_prompt` 即报错）；教师 logprob 张量形状与学生一致；越界行与非越界行的蒸馏损失都有限；在一个越界样本上手算——同一回答在 given 题面下的教师 logprob 与学生题面下的不同（否则特权未生效）；学生 4 卡时每步耗时，推算 120 步在 36 小时内。任一不过即停，修后重做。
2. **教师探针**（只在 T 前）：Qwen3-32B 与 Qwen3-30B-A3B-Instruct-2507 关思考各在 `final.json` 越界行上 n = 8、T 1.0（`explore/x_passk.sh`，`TP=2`），逐机制 pass@1。
   选 14 个 Z 机制中 pass@1 ≥ 0.5 更多的那个作教师；门槛：≥ 6 个才跑 T；两个都不过则 T 不跑，H-OPD4 记为不可检验。探针结果与选择写入执行记录后再开 T。
3. 特权自教师的"会"：从 W_f 热身采样算每个 Z 机制 given 行的基座正确率（2 % 判分同 X），写入执行记录，定出主检验集；不另跑模型。

## 5. 算力与顺序

P、S、Pk3 各两折 = 6 次，P 种子 43 两折 = 2 次，T 两折 = 2 次（条件性），第二家族 P2 / T2 各两折 = 4 次（条件性）；每次约 20–30 小时 × 5–6 卡，共约 1,000–1,600 GPU 小时。
顺序：P(F1, F2) → S → Pk3 → P s43 → T → 第二家族。P 两折完成后先报 H-OPD1 的数字，再决定是否继续（不改判定，只决定是否花后面的算力）。
在 DOSE-text 终点评测完成、Ministral 探针结束之后启动。

## 6. 已知局限（写进讨论）

特权教师就是学生的冻结基座，P 的结论限于"知识以题面陈述的形式存在于同一模型中"；given 题面多出的句子由我们撰写；
主检验一个家族、8B、关思考（第二家族只报数字）；教师知识由一次 n = 8 的探针判定；OPD 臂关闭了动态采样并用 4 卡训练，与 A 不是逐项相同。

## 2026-10-29 执行记录 OPD-1：门槛项、教师探针、损失模式替换（写于任何 OPD 训练步之前）

- **特权检查（§4.1）通过**：`train/opd/check_privilege.py`，Qwen3-8B 经 vLLM 服务（114 GPU 7），F1 特权数据的前 6 条越界行，各配其 given 行的首条热身采样程序：
  同一程序在 given 题面下的对数概率比学生题面高 38.5–45.2 nats（中位 +40.9），6/6 同向；`experiments/_opd/privilege_check_F1.json`。
- **数据**：`train/opd/build_opd_data.py` 由 `data-x/F{1,2}-A` 生成 `data-opd/F{1,2}-{priv,self}`；F1-priv 2,880 行、580 条越界行带 `teacher_prompt`，非越界行 0 条；
  F1 源 `train.parquet` sha256 `f75fe411…`，输出 `d09c0ee7…`。
- **补丁树**：`$R/scratch/verl-opd` = verl 5b79827b 的工作树 + `patch_verl_privileged.py`；`PYTHONPATH` 首位时 `verl.experimental.agent_loop.agent_loop` 从该树导入（核对过）。
- **教师探针（§4.2）**：Qwen3-32B 关思考，`final.json` 1,400 条越界行 × 8 次、T 1.0、无截断（`experiments/_opd/teachers/passk_outside_n8.qwen3-32b.jsonl`）：
  70 模板均值 pass@1 0.214（Qwen3-8B 基座 0.090；69 机制上 32B 高于 8B 的 47 个、低于 3 个，秩相关 0.82——探针本身可信）。
  **14 个 Z 机制上 pass@1 ≥ 0.5 的为 0 个**（> 0 的 2 个：snowpack_cold_content 0.28、displacement_height_canopy 0.006）→ Qwen3-32B 不过门槛。
  Qwen3-30B-A3B-Instruct-2507 的探针在跑；两者都不过则 T 臂不跑、H-OPD4 记为不可检验。
- **损失模式替换**：本仓库锁定的 verl（5b79827b）只支持 `forward_kl_topk`（无 `_tail` 尾桶变体，冒烟在配置阶段报 "Unsupported loss mode"）。
  P、S、T 臂改用 `forward_kl_topk`（教师 top-32 的截断前向 KL，负的逐 token 值截为 0，verl 文档所述），K = 32 不变；这是实现约束，不改任何假设或读法。
- **与 A 的差别核对**：冒烟配置打印确认 `filter_groups.enable=False`、`gen_batch_size = train_batch_size`、学生 4 卡 + 教师 1 卡后，再记录。

## 2026-10-29 执行记录 OPD-2：冒烟通过，P 臂 F1 种子 42 启动（114）

- **冒烟**（`train/opd/run_opd.sh`，SMOKE=1，F1-priv，GPU 0–3 学生 + 7 教师，2 步 × 16 题 × 4 次）六次尝试，前五次的失败都是实现问题、不是设计问题：
  (1) 锁定的 verl 没有 `forward_kl_topk_tail`（OPD-1 已记）；(2) top-K 损失不接受 `strategy=fsdp2`——补丁把 fsdp2 并入 fsdp 分支（与 veomni 同理，
  `patch_verl_privileged.py`，损失只读 `student_logits`）；(3) 失败尝试留下的孤儿 vLLM 引擎占着 GPU 7，第三次的教师引擎起不来——按 PID 清理后正常；
  (4)(5) 通过但 worker 的日志行不进驱动日志，无法证明特权生效——改为文件诊断（`ATMOS_OPD_DIAG`）。
- **第六次冒烟通过，门槛项（§4.1）逐条**：补丁树被所有 Ray worker 导入（诊断记录的 `module` 路径是 `scratch/verl-opd`）；`ATMOS_OPD_PRIVILEGED=1` 到达 worker；
  每个样本的 kwargs 含 `extra_info`（dict），越界行带 `teacher_prompt`；4 个 worker 记录了首个特权样本（教师题面 342 / 493 / 384 / 381 个 token，学生题面 293 / 361 / 344 / 320，
  差即替代关系那句话）；模板自检未报错（重渲染的学生题面与 rollout 的 prompt_ids 相同）；蒸馏损失有限（2.70 → 1.56，overlap_ratio 0.94 → 0.95），
  答案奖励 0.16 → 0.28，策略损失非零；特权差异由 `check_privilege.py` 另证（OPD-1）。冒烟每步 35 / 15 s（含首步编译）。
- **与 A 的差别核对**：有效配置打印 `filter_groups.enable=False`、`gen_batch_size=16`（正式 128 = train_batch）、`trainer.n_gpus_per_node=4`、
  `distillation.n_gpus_per_node=1`、`loss_mode=forward_kl_topk`、`use_task_rewards=True`、`use_policy_gradient=False`。
- **P 臂 F1 种子 42 正式启动**：114，GPU 0–3 + 7，`opd-qwen3-8b-F1-P-s42`，120 步、36 小时上限、每 5 步存档；数据 `data-opd/F1-priv/train.parquet`
  sha256 前 16 位记于启动输出（`logs/opd/run-P-F1-s42.log`）。F2 等 047 / 113 的 DOSE-text 结束后启动。
- **教师探针**：Qwen3-30B-A3B-Instruct-2507 在跑；Qwen3-32B 已不过门槛（OPD-1）。

## 2026-10-29 执行记录 OPD-3：P 臂 F1 第一次尝试在第 1 步更新时显存溢出，改小微批后重启

- **现象**：`opd-qwen3-8b-F1-P-s42` 第一次尝试（23:18 启动）在第 1 步的 `update_actor` 反向传播中 CUDA OOM（GPU 0 已用 70.1 GiB，再申请 12.5 GiB）；
  rollout 与教师打分已完成。原因：top-K 蒸馏损失要保留整个词表的 logits（151,936 × 每微批 token 数），学生只有 4 卡时 FSDP 分片本身已占约 32 GiB，
  `ppo_max_token_len_per_gpu=24,576` 的微批放不下；冒烟的批太小没暴露。
- **处理**：OPD 臂的 `ppo_max_token_len_per_gpu` 定为 **8,192**（`PPO_MAX_TOKENS`，`run_opd.sh`）。只改微批切分（动态批按 token 数切），mini-batch 32 题 × 8、
  优化器步数、token-mean 聚合都不变；是实现约束，不改设计。第一次尝试无检查点、无训练步，按 X-10 不计入续跑次数；24/36 小时上限从第二次启动起算。
- 清理核对：失败后无孤儿 vLLM 进程（只剩 Ministral 采样服务与 30B-A3B 探针）。

## 2026-10-29 执行记录 OPD-4：两个外部教师都不过门槛，T 臂不跑

- Qwen3-30B-A3B-Instruct-2507 关思考，同一探针（1,400 条越界行 × 8，T 1.0，`experiments/_opd/teachers/passk_outside_n8.qwen3-30b-a3b-2507.jsonl`）：70 模板均值 pass@1 0.314
  （Qwen3-32B 0.214，Qwen3-8B 基座 0.090）。**14 个 Z 机制上 pass@1 ≥ 0.5 的只有 1 个**（displacement_height_canopy 0.61；snowpack_cold_content 0.48、plume_ground_reflection 0.09、
  saturated_lapse 0.04，其余 10 个为 0）。Qwen3-32B 为 0 个（OPD-1）。
- 按 §4.2 的门槛（≥ 6 个）：**T 臂不跑，H-OPD4 记为不可检验。** 附带观察（只报数字）：与 Qwen3-8B 共词表的两个更大的 Qwen3 模型，在 8B 够不着的 14 个机制里分别只会 0 个和 1 个，
  虽然它们在全部 70 个模板上的越界正确率是 8B 的 2.4–3.5 倍——8B 够不着的机制在同一家族更大的模型里也几乎没有。
- 第二家族的 T2（Qwen3.8-27B → Qwen3.5-9B）仍按 §2 的条件先探针，不受此影响。

## 2026-10-30 执行记录 OPD-5：P 臂 F1 第二次尝试因教师张量对齐错位作废，补丁 v2 后第三次启动

- **训练后审查（第二次尝试，23:36–02:55，120 步，退出 0）**：特权生效（5 个 worker 都收到 `teacher_prompt`，特权样本约 3,180 / 15,350 ≈ 21 %），
  但蒸馏指标在物理上不可能：第 1 步 `teacher_mass` 均值 1.02、`teacher_mass_max` 32、单 token 蒸馏损失最大 1476.5；第 4–5 步梯度范数 833 → 1300、
  回复长度 350 → 8 个 token、奖励 0.28 → 0.003（塌陷）；第 20 步起奖励回到 0.16–0.30、长度 190–250，但 `teacher_mass` ≈ `student_mass` ≈ 3 持续到第 120 步。
  top-32 概率质量 > 1 只能来自占位行。**该模型作废，不合并、不评测**；检查点与日志改名为 `*.attempt2-v1` 留档（114 `/path/to/local/opd/checkpoints/`，92 GB，待授权删除）。
- **根因**：verl 的教师行是移位对齐的——`vllm_rollout/utils.py` `extract_prompt_logprobs` 丢掉第 0 位、末尾补一行哑行（id 0、logprob 0），第 i 行是第 i+1 个 token 的
  top-k；损失读学生序列的第 prompt_len−1 … L−2 行（`workers/utils/padding.py` 的左移一位切片）。补丁 v1 的重排保留教师第 teacher_prompt_len 行起的行，
  丢掉了第 teacher_prompt_len−1 行（第一个回复 token 的教师分布），损失在每个特权样本的第一个回复 token 处读到占位行（32 个 pad、概率各 1）：
  loss_max = −32·log q(pad) ≈ 1476，学生被推向在首 token 输出 pad（= EOS）→ 第 5 步回复塌到 8 个 token。非特权样本不经重排，不受影响。
- **冒烟为何没拦住**：第六次冒烟（OPD-2）日志第 1 步已有 `loss_max 1476.5`、`teacher_mass_max 32`，审查只看了诊断文件、奖励与平均损失。
  从此冒烟门槛加入：`teacher_mass` / `student_mass` 均值与最大 ≤ 1.001、`distillation/loss_max` < 100、梯度范数与回复长度不突变。
- **修复（`61a236f`）**：`patch_verl_privileged.py` v2——保留教师第 teacher_prompt_len−1 行起（回复长度 + 1 行，含哑行），前补 prompt_len−1 个占位行；
  运行时断言保留行（哑行除外）的 top-k 质量 ≤ 1.001，否则报错；脚本拒绝在 v1 补丁树上叠加（须先 `git checkout` 还原）。`train/opd/test_realign.py`
  按 verl 的切片约定做单元测试：v2 读到的正是教师对各回复 token 的分布、无占位行、哑行留在末尾；v1 复现首 token 读到占位行（质量 = K）；质量守卫触发。
  114 的 `scratch/verl-opd` 还原后重打 v2（`git diff --stat`：agent_loop.py +83，losses.py ±1），单元测试通过。
- **v2 冒烟（03:19–03:45，同 OPD-2 设置）过门槛**：`teacher_mass` 0.99987 / 0.99988、`teacher_mass_max` 1.0000005、`student_mass` 0.9992 / 0.9996、
  `loss_max` 38.9 / 34.1、蒸馏损失 0.172 → 0.094、overlap 0.942 → 0.952、梯度范数 9.8 / 5.8（v1 为 36）、奖励 0.14 → 0.34、回复长度 325 / 318；
  4 个 worker 记录首个特权样本；无质量守卫报错。退出时的 `tabulate` / DataLoader 回溯是 torch atexit 收尾噪音（第二次尝试同样出现、训练完整）。
- **第三次启动**：03:47，114，GPU 0–3 + 7，`opd-qwen3-8b-F1-P-s42`（新日志；worktree `rl-wt-61a236f`）。按 X-10，第二次尝试是实现错误造成的作废，
  不是续跑；36 小时上限自第三次启动起算。浪费：约 3.2 h × 5 GPU。守望在第 3 步核对上述门槛，不过即按 PID 停止。
- **对设计的影响**：无——S、Pk3 臂的 `teacher_prompt` 为空、不经重排；T 臂不跑。第二家族若跑 P2 用同一 v2 补丁。

## 2026-10-30 执行记录 OPD-6：主检验集（§4 第 3 条，补记）与报告脚本（写于任何 OPD 终点评测之前）

- **偏离说明**：§4 第 3 条要求开训前算出主检验集，OPD-1 漏记；本记录在 P 臂 F1 第三次启动（03:47）之后、任何 OPD 终点评测之前补算。
  集合只由存档的热身采样（X-10 前采、sha256 记于结果文件）与 2 % 判分决定，与 OPD 训练无关，不影响样本外性质。
- **算法**：`explore/opd_main_set.py`——W_f 热身采样（`experiments/_x/qwen3-8b/warm_samples_F{1,2}.jsonl`，每条 given 行 8 次、T 1.0、题面无提示）
  每个程序按 given 行自身答案 2 % 判分（`atmoscoder_rl.reward.grade`，strict），模板 given 正确率 = 各行 8 次通过份额的均值；1,400 条 given 行全部有样本。
- **结果（`results/opd/main_set.json`）**：14 个 Z 机制的 given 正确率——

  | 机制 | 折 | given 正确率 | pass@8 | 会 |
  |---|---|---|---|---|
  | diffuse_transmission | F1 | 1.000 | 1.00 | ✓ |
  | ch4_forcing_sqrt | F2 | 0.894 | 1.00 | ✓ |
  | co2_forcing_log | F2 | 0.887 | 0.95 | ✓ |
  | sun_photometer_airmass | F2 | 0.812 | 1.00 | ✓ |
  | freezing_point_depth | F1 | 0.800 | 1.00 | ✓ |
  | displacement_height_canopy | F1 | 0.787 | 1.00 | ✓ |
  | plume_ground_reflection | F2 | 0.731 | 1.00 | ✓ |
  | snowpack_cold_content | F2 | 0.431 | 1.00 | |
  | clausius_clapeyron_extrapolation | F2 | 0.350 | 0.70 | |
  | rossby_deformation | F1 | 0.181 | 0.55 | |
  | line_broadening | F1 | 0.037 | 0.30 | |
  | drop_ventilation | F2 | 0.013 | 0.10 | |
  | saturated_lapse | F1 | 0.006 | 0.05 | |
  | ice_above_flotation | F2 | 0.000 | 0.00 | |

  **主检验集 = 7 个**（F1 3、F2 4）≥ 5 → H-OPD1–3 可判定。其余 7 个 Z 机制即使题面给出替代关系基座也多半写不对（4 个 pass@8 ≤ 0.3）——
  特权教师在这些机制上同样"不会"，是"知识缺失"而非"信号稀疏"的直接证据，报告时与主检验集并列。
- **报告脚本**：`explore/opd_report.py`（在任何 OPD 终点评测之前提交）按 §3 固定：主结局与三组对比（P − A、P − S、Pk3 − A、P − Pk3）在主检验集 / Z / 全部模板上，
  机制重抽 10,000 次、`random.Random("opd:<项>:20261029")`；判定规则、"只报数字"条件（主检验集 < 5 或对比两臂任一终点未评测完）；次要项按 §3 所列。
  OPD 终点标签 `opd-qwen3-8b-<折>-<臂>-s<种子>`，用 `explore/x_eval.sh` 评入同一目录。先用 A 臂已有评测空跑一次核对代码路径（P 缺失时各项只报 A）。

## 2026-10-30 执行记录 OPD-7：P 臂 F1 第三次尝试训练后审查、合并与评测启动；P 臂 F2 启动（写于任何 OPD 终点评测结果之前）

- **P 臂 F1 种子 42 第三次尝试（114，GPU 0–3 + 7，03:47–07:53，训练 3 h 56 min，120 步、约 118 s/步，退出 0）**：
  第 3 步门槛（OPD-5 新定）通过——`teacher_mass_max` 1.0000006、`distillation/loss_max` 28.1、梯度范数 1.21、训练奖励 0.312、回复长度 345。
  全程 120 步 `teacher_mass_max` ≤ 1.0000007、`student_mass_max` ≤ 1.0000004（无占位行）；`loss_max` 第 1 步 40.8 → 第 69–114 步 11–18；蒸馏损失 0.092 → 0.016–0.021；
  梯度范数第 1 步 4.5、此后 0.50–0.75；训练奖励（T 1.0）0.285 → 0.43–0.50（第 91–114 步）；回复长度 328–383、截断率 0.001；
  训练探针（`train_probe`，T 0.3，每 20 步）0.238 → 0.402 → 0.484 → 0.523 → 0.555 → 0.578 → 0.547。无质量守卫报错；退出时一条 DataLoader 回溯同 OPD-5 所记的收尾噪音
  （`global_step_120` 完整、包装脚本退出 0）。**判定：训练过程正常，模型合格。**
- **合并与评测**：08:01 合并 `global_step_120`（`verl.model_merger`，同 `x_chain.sh`）→ 114 `/path/to/local/opd/merged/opd-qwen3-8b-F1-P-s42/global_step_120`；
  随后 `explore/x_eval.sh`，标签 `opd-qwen3-8b-F1-P-s42`，114 GPU 4–7（GPU 0–3 同时跑第三家族的 DAPO 冒烟，与 OPD 无关）。评测结果未看。
- **P 臂 F2 种子 42（113，GPU 0–3 + 7，06:05 启动，worktree `rl-wt-9adf8f7`，同一 `scratch/verl-opd` v2 补丁树）**：第 3 步门槛通过——`teacher_mass_max` 1.0000006、
  `loss_max` 36.3、梯度范数 0.95、训练奖励 0.327、回复长度 374；第 63 步（08:05）`loss_max` 21.6、梯度范数 0.62、奖励 0.366、长度 376。约 2 min/步，预计 10:00 前后结束，
  之后同样审查 → 合并 → 评测。两折 P 评测齐后先跑 `explore/opd_report.py` 报数，再按 §5 决定 S / Pk3 / 种子 43。

## 2026-10-30 执行记录 OPD-8：P 臂两折评测完成、§3 读数（只跑一次）、S 臂启动

- **P 臂 F2 种子 42 训练后审查（113，06:05–09:38，120 步，退出 0）**：全程 `teacher_mass_max` ≤ 1.0000007、`student_mass_max` ≤ 1.0000004；`loss_max` 45.4 → 15–20；
  蒸馏损失 0.081 → 0.024；梯度范数 3.65 → 0.5–0.8；训练奖励 0.309 → 0.43；回复长度 345–375、截断率 0.001；探针（T 0.3）0.223 → 0.477。合格。
  09:49 合并（优化器分片删除）、评测 `opd-qwen3-8b-F2-P-s42`（113 GPU 0–3），10:32 完整（5,600 / 2,800 / 64 / 92 行，0 错误行）。
- **报告（`explore/opd_report.py`，10:40，113 `rl-wt-58bdc74`，脚本与 OPD-6 的 fadd574 逐字相同 → `results/opd/report_qwen3-8b.json`）**：
  主检验集 7 个机制——A 0.0 [0.0, 0.0]、**P 70.7 [39.3, 96.4]**；P − A 70.7 [40.0, 96.4] → **H-OPD1 支持 b**：稠密的特权信号让 RL 学不会的机制学会，缺的是信号不是知识。
  Z 14 个：A 4.3 [0.0, 12.3]、P 48.6 [25.4, 71.8]，P − A 44.3 [17.3, 69.1]。全部 70 模板：A 51.2、P 49.8，P − A −1.4 [−14.0, 11.1]。
  H-OPD2（P − S）、H-OPD3（Pk3）待 S / Pk3；H-OPD4 不可检验（OPD-4）。
- **次要**：留出簇迁移 P 越界 0.2 [−2.5, 2.9]、原题 0.5 [−1.7, 2.8]（无迁移）；训练折原题终点 − 基座 A +53.6、**P +11.8 [7.6, 16.4]**（P 对原题的提升远小于 A：
  无策略梯度，蒸馏损失主导）；错答落在标准公式值上的份额（2 %）终点 − 基座 A +42.4 [28.4, 55.9]、**P −11.6 [−23.5, 0.1]**（P 没有 F9 的捷径收敛）；
  test_id P 34.4 / 32.8（基座 34.4）；P 只有一个种子。
- **逐机制（Z，越界正确率；given = 教师在 given 行的正确率，OPD-6）**：

  | 机制 | given | 基座 | A s42 / s43 | P s42 |
  |---|---|---|---|---|
  | diffuse_transmission | 1.00 | 0 | 0 / 0 | **1.00** |
  | ch4_forcing_sqrt | 0.89 | 0 | 0 / 0 | 0.00 |
  | co2_forcing_log | 0.89 | 0 | 0 / 0 | **1.00** |
  | sun_photometer_airmass | 0.81 | 0 | 0 / 0 | **0.85** |
  | freezing_point_depth | 0.80 | 0 | 0 / 0 | **0.95** |
  | displacement_height_canopy | 0.79 | 0 | 0 / 0 | **0.95** |
  | plume_ground_reflection | 0.73 | 0 | 0 / 0 | 0.20 |
  | snowpack_cold_content | 0.43 | 0 | 0 / 0 | **0.85** |
  | clausius_clapeyron_extrapolation | 0.35 | 0 | 0 / 0 | **0.85** |
  | rossby_deformation | 0.18 | 0 | 0.05 / 1.00 | 0.00 |
  | line_broadening | 0.04 | 0 | 0 / 0 | 0.00 |
  | drop_ventilation | 0.01 | 0 | 0 / 0 | 0.00 |
  | saturated_lapse | 0.01 | 0 | 0.10 / 0.05 | 0.15 |
  | ice_above_flotation | 0.00 | 0 | 0 / 0 | 0.00 |

  教师 given ≥ 0.35 的 9 个机制 P 学会 7 个（ch4 0、plume 0.20 例外）；given ≤ 0.18 的 5 个一个也没学会（§3 的预测"只有教师会的机制被学会"成立，例外 ch4）。
- **已见数字的说明**：报告前（09:06 核对评测完整性时）日志过滤误匹配到 P F1 四个评测文件的总正确率（final 42.1 %、judge 37.3 %、test_id 34.4 %、textbook 14.1 %）；
  非主检验集读数、未据此做任何决定，报告脚本未改。
- **S 臂 F1 种子 42 启动**：113，GPU 0–3 + 7，10:47，worktree `rl-wt-58bdc74`（`run_opd.sh`、`dapo_qwen3_8b.sh` 与 61a236f 相同），同一 v2 补丁树（`git diff --stat` 83 / 2 行）。
  审查项新增：S 的教师与学生同模型同题面，**第 1 步蒸馏损失应 ≈ 0**（verl 行对齐的端到端检验）；若不是，停止并查。Pk3 F1 等 114 空出后启动；047 已交还用户，不再使用。

## 2026-10-30 执行记录 OPD-9：S 臂两折训练后审查、评测完成、H-OPD2 读数；Pk3 F1 启动

- **S 臂训练后审查（113 `opd_chain.sh`，worktree `rl-wt-b9a1987`）**：F1 10:47–14:12、F2 14:13–17:40，各 120 步、退出 0，链的门槛（rc 0 / 120 步 / mass ≤ 1.001 / loss_max < 100）均 PASS：
  `teacher_mass_max` 1.0000006 / 1.0000007，`loss_max` 全程最大 16.5 / 23.7。**OPD-8 新增的行对齐检验通过**：第 1 步蒸馏损失 0.00061 / 0.00059（教师 = 学生、同题面时应 ≈ 0），
  第 120 步 0.0015 / 0.0016；梯度范数 0.40 / 0.48；训练奖励 0.296 → 0.41 左右；回复长度 344–355。退出时的回溯是 torch dynamo 退出噪音（同 OPD-5）。
  两折合并、评测完整（`opd-qwen3-8b-F{1,2}-S-s42`：5,600 / 2,800 / 64 / 92 行，0 错误行）；合并后删除两折 `model_world_size_*.pt` 分片（各 31 GB，合并模型 16.4 GB 已核）。
- **报告（`explore/opd_report.py`，19:50，113 `rl-wt-58bdc74`，脚本与 fadd574 逐字相同；`--seeds 42 43 --opd-seeds 42`，同 OPD-8 命令 → `results/opd/report_qwen3-8b.json` 覆盖）**：
  与 OPD-8 的报告逐键比对，P / A 的所有数字不变，只新增 S。
  主检验集 7 个：S 0.0 [0.0, 0.0]；**P − S 70.7 [40.0, 96.4] → H-OPD2 成立：P 的增益来自特权信息，不是 KL 正则**；H-OPD1 维持"支持 b"。
  Z 14 个：S 0.0，P − S 48.6 [25.7, 71.4]。全部 70 模板（训练折越界）：S 14.1 [7.8, 21.1]，P − S 35.7 [26.9, 45.0]。
- **S 的次要项**：训练折原题终点 − 基座 +9.7 [6.3, 13.4]（P +11.8）；留出簇迁移越界 0.3 [−1.5, 3.1]、原题 0.4 [−2.5, 3.1]（无迁移）；
  错答落在标准公式值上的份额（2 %）终点 − 基座 +3.6 [1.5, 5.8]（P −11.6，A +42.4）；test_id F1 39.1、F2 34.4（基座 34.4）。
- **读法注**：S 的教师就是起点学生本身，蒸馏损失全程 ≈ 0.001，S 的作用相当于"关动态采样、无策略梯度、KL 到起点"——它控制的是 OPD 管线与正则本身，
  不是"同等强度但无特权的稠密信号"；H-OPD2 的结论按 §3 写定的读法只说"增益来自特权信息而不是 KL 正则"，不外推。
- **Pk3 F1 种子 42 启动**：113，17:41（OPD-8 原计划放 114；114 被 Ministral 占用，按链顺序在 113 接续），19:40 时第 83 步。

## 2026-10-30 执行记录 OPD-10：Pk3 F1 被链的门槛误判 FAIL（k3 无"质量"指标），人工审查后按链的同一命令合并、评测（写于任何 Pk3 评测之前）

- **事实**：`opd_chain.sh` 20:32 判 `opd-qwen3-8b-F1-Pk3-s42` 门槛 FAIL（`rc 0 steps 120 mass_max ? loss_max 10.0`），未合并、未评测，链转去训练 Pk3 F2。
  原因是门槛脚本的缺陷：OPD-5 的质量门槛 `teacher/student_mass_max ≤ 1.001` 只对 top-K 损失有意义；Pk3 用 `k3`（只在学生采到的 token 上），
  日志里没有质量项，脚本把缺失当成 9 → FAIL。Pk3 的设计（§1 表）本来就不产生该指标。
- **人工审查（训练日志全程）**：退出 0、120 步；`distillation/loss_max` 每步恰为 10.0 = verl `k3` 估计量的截断上限（[−10, 10]），即每步都有少数 token 撞截断，
  这是 k3 的固有行为，不是数值失控；`loss_min` ≈ −1e−7；平均蒸馏损失 0.056 → 0.038；梯度范数 0.80–0.96；训练奖励 0.293 → 0.446；
  回复长度 314–350、截断率 0.001；训练探针（T 0.3）第 60 步 0.434、第 120 步 0.457；特权 / 普通行计数正常；退出回溯是同 OPD-5 的 DataLoader 收尾噪音。判定：合格。
- **处理**：不重训；用链里逐字相同的命令（`verl.model_merger merge --backend fsdp`，删优化器分片；`explore/x_eval.sh`，GPU 4–6，worktree `rl-wt-b9a1987`）
  人工合并并评测 Pk3 F1。Pk3 F2 会遇到同一误判，训练结束后同样人工审查（rc 0、120 步、loss_max ≤ 10 的截断值、无发散），合格则同样处理。
  P 种子 43 两折用 top-K 损失，链的门槛照常适用。链脚本不改（它在跑）。

## 2026-10-31 执行记录 OPD-11：Pk3 两折评测完成、H-OPD3 读数

- **Pk3 F2 种子 42**：20:32–23:29，120 步、退出 0；链门槛同 OPD-10 误判 FAIL（`mass_max ?`）。人工审查（00:40）按 OPD-10 写定的条件：`loss_max` 每步 10.0（k3 截断值）、
  平均蒸馏损失 0.047 → 0.040、梯度范数 0.65–0.91、训练奖励 0.296 起、训练探针第 60 / 120 步 0.273 / 0.383，无发散 → 合格。00:41 合并（16.4 GB 已核，删分片）、
  评测 01:30 完整。Pk3 F1 评测 21:31 完整。两折均 5,600 / 2,800 / 64 / 92 行，0 错误行。
- **报告（01:45，同 OPD-9 命令与 worktree → `results/opd/report_qwen3-8b.json` 覆盖）**：与 OPD-9 报告逐键比对，非 Pk3 的数字全部不变；四组对比全部齐（missing: none）。
  主检验集 7 个：Pk3 1.4 [0.0, 4.3]；**Pk3 − A 1.4 [0.0, 4.3] → "只在学生采到的 token 上打分的蒸馏同样被可达性限制"**；
  **P − Pk3 69.3 [39.3, 95.0]（≥ 20）→ "越过屏障靠的是对未采样 token 的信号"**。H-OPD3 两个读法均成立。
  Z 14 个：Pk3 10.7 [0.7, 26.8]，Pk3 − A 6.4 [−8.0, 23.6]，P − Pk3 37.9 [15.7, 61.1]。全部 70 模板：Pk3 24.0，Pk3 − A −27.2 [−37.6, −17.0]，P − Pk3 25.8 [17.6, 34.4]。
- **Pk3 次要项**：训练折原题终点 − 基座 +17.4 [11.9, 23.5]；留出簇迁移越界 0.7 [−1.1, 2.6]、原题 0.2 [−2.7, 3.0]；捷径份额终点 − 基座 −1.2 [−8.7, 6.0]；test_id 35.9 / 34.4（基座 34.4）。
- **读法注**：Pk3 与 P 只差损失（§1 表），但这一差别包含两件事——只在采到的 token 上打分，以及 `k3` 估计量本身（含 [−10, 10] 截断，每步都有 token 撞上）。
  P − Pk3 归因于"未采样 token 上的信号"是 §3 写定的读法；截断削弱信号的另一种解释本记录不能排除，论文里与读法并列写出。P、S、Pk3 都只有种子 42。

## 2026-10-31 执行记录 OPD-12：P 种子 43 两折完成、跨种子一致性（§5）；OPD 链结束

- **P 种子 43**：F1 23:29–02:57、F2 02:58–06:31，各 120 步、退出 0，链门槛 PASS（`mass_max` 1.0000008 / 1.0000007，`loss_max` 38.4 / 35.6）；
  第 120 步蒸馏损失 0.013 / 0.023、梯度范数 0.49 / 0.52、训练探针 0.598 / 0.531（种子 42：0.555 / 0.477）。两折合并（删分片）、评测完整（5,600 / 2,800 / 64 / 92 行，0 错误行）。
  `OPD CHAIN DONE` 07:18。
- **跨种子一致性（`opd_report.py --opd-seeds 42 43`，07:35，同 worktree → `results/opd/report_qwen3-8b_seeds42-43.json`，另存，不覆盖判定用报告）**：
  主检验集 P 种子 42 70.7 [40.7, 96.4]、种子 43 70.7 [40.7, 95.0]，逐机制平均绝对差 4.3 点；≥ 50 % 的机制两个种子完全相同
  （co2、diffuse、displacement、freezing、sun_photometer；ch4 两种子都是 0，plume 0.20 / 0.25）。H-OPD1 跨种子一致。
- **注意**：`--opd-seeds 42 43` 对所有 OPD 臂生效，S / Pk3 没有种子 43，该次运行因此把 H-OPD2/3 降为"只报数字"；这是调用口径，不是结果变化。
  §3 写定种子 43 只用于 H-OPD1 的一致性，判定以 OPD-11 的种子 42 报告（`report_qwen3-8b.json`）为准。
- OPD 全部计划内的 run 完成（P、S、Pk3 种子 42 各两折，P 种子 43 两折）；T 臂不跑（OPD-4）。113 交给 Ministral 第一折种子 43（PREREG_DOSE_MINISTRAL DM-3）。

## 2026-10-31 补充 OPD-13：两个追加臂 Pk2、Pshuf（写于两者任何训练之前；不改 H-OPD1–4 的任何判定）

- **为什么**：审查 OPD-11 时两处读法有替代解释。(1) H-OPD3 把 P − Pk3 归因于"未采样 token 上的信号"。但 verl 对每 token 的蒸馏损失截断在 10（`loss_max_clamp`，
  各臂都有），Pk3 的 `k3` 内部也截在 ±10；Pk3 里损失最大的恰是"学生采了错 token、特权教师给它很低概率"的位置，截断把这些最有信息的 token 的梯度清零。
  "截断让 Pk3 失效"这一解释 OPD-11 无法排除。(2) H-OPD1/2 说增益来自特权信息；S 去掉了特权，但没有控制"教师题面多了一句话"本身。
- **更正 OPD-11 的一处描述**：OPD-11 说截断是 Pk3 与 P 的差别之一。实际上逐 token 的 `loss_max_clamp = 10` 对 P 同样生效（日志里的 `loss_max` 是截断前的值，P 为 16–45），
  两臂差在截断落在哪些 token 上，而不是有没有截断。各 OPD 臂都叠加答案奖励（`use_task_rewards: true`，§1 表），审查中提出的"蒸馏 + 答案奖励"联合臂因此就是 P，不另跑。
- **Pk2**：与 Pk3 只差损失——`k2` = ½(log p_学生 − log p_教师)²，只在学生采到的 token 上，`loss_max_clamp = null`（`train/opd/run_opd.sh`，99e2978）。
  无任何逐 token 截断；全局梯度裁剪 1.0 照旧。门槛：rc 0、120 步、蒸馏损失无 NaN / inf；`loss_max` 只记录不设门槛（k2 无上界，设计如此）。
  训练后人工审查梯度范数与训练奖励；若训练发散（奖励崩到 < 0.1 或损失变 NaN），H-OPD5 记"不可判定：无截断的 k2 本身不稳定"，不作 H-OPD3 的结论。
- **Pshuf**：与 P 只差教师题面——越界行插入的不是本模板的替代关系，而是**另一个模板**的那一句（同一位置；模板按 `random.Random("opd:shuffle-hint:<折>")`
  的错位排列配对，无一保留自己的句子）。数据 `data-opd/F1-privshuf`（2,880 行，580 行特权，29 个模板，sha256 前缀 5535a428）、`F2-privshuf`（3,600 / 820 / 41，bb7c4890）；
  `build_opd_data.py --shuffle-hint`。核对：580 / 580 行恰为一处插入、学生题面与 `F1-priv` 逐字相同、system 相同、0 行插入本模板的句子。门槛同 P（OPD-5）。
- **假设与读法**（`explore/opd_report_ext.py`，99e2978，与本记录同时提交；单位、结局、重抽同 §3；需主检验集 ≥ 5 且两臂两折都评完）：
  - **H-OPD5**：Pk2 − A 区间含 0 或点估计 < 10，且 P − Pk2 ≥ 20 → "去掉截断后只在采样 token 上蒸馏仍学不会：H-OPD3 的读法不依赖截断"；
    Pk2 − A ≥ 30 且下界 > 0 → "截断是 Pk3 学不会的原因"，撤回 H-OPD3 的"未采样 token"读法、改为只报数字；其余只报数字。
  - **H-OPD6**：P − Pshuf ≥ 30 且下界 > 0 → "增益需要正确的替代关系，不是教师题面里多了任何一句话"；Pshuf − A ≥ 30 且下界 > 0 →
    "多一句话本身就有大部分作用"，H-OPD1 的读法收窄为"特权题面"、只报数字；其余只报数字。
  - 次要（只报数字）：留出迁移、训练折原题、捷径份额、test_id、截断率。干跑（无新臂评测）输出"只报数字"，P / A 数字与 `report_qwen3-8b.json` 相同。
- **排程**：种子 42、两折，`opd_chain.sh`（RUNS="Pk2:F1:42 Pk2:F2:42 Pshuf:F1:42 Pshuf:F2:42"），113，在 Ministral 第一折 RL 链结束后自动启动；
  每次约 3.5 小时 + 评测，共约 15 小时，约 11-01 夜间完成。每次训练结束按门槛审查并追加执行记录。

## 2026-10-31 执行记录 OPD-14：Pk2、Pshuf 冒烟通过；链提前启动（Ministral 第一折 RL 已停，PREREG_DOSE_MINISTRAL DM-8）

- **冒烟**（113，`rl-wt-99e2978`，各 2 步，第一折，GPU 0–3 学生 + 7 教师；`python -m explore.audit_rollout_consistency --steps $ATMOS_ROOT/logs/opd/smoke13-<臂>.nohup`）：
  - Pk2（14:01–14:10，rc 0）：`loss_mode k2`、`loss_max_clamp None`、数据 `data-opd/F1-priv`；pearson 0.999 / 0.998，`rollout_corr/kl` 0.0005 / 0.0008，
    蒸馏损失 0.734 / 0.234、`loss_max` 447 / 178（无截断，设计如此，不设门槛），无 NaN / inf；梯度范数 6.6 / 3.4（P 冒烟 9.8 / 5.8；全局裁剪 1.0）。
  - Pshuf（14:10–14:16，rc 0）：`loss_mode forward_kl_topk`、数据 `data-opd/F1-privshuf`；pearson 0.998 / 0.998，kl 0.0007 / 0.0005，
    teacher / student mass 最大 1.0000005 / 1.0000004，`loss_max` 10.5 / 17.5（< 100），蒸馏损失 0.018 / 0.023；梯度范数 2.1 / 2.4。
  - 两次的特权诊断文件各有 6 个 `first_privileged` 事件（> 0；首条 `shallow_water_wave_speed_15_outside`，教师 358 token 对学生 293）。
  - 两个日志里各有一处 Traceback：进程退出时 torch dynamo 打印编译耗时失败（`import tabulate`），不在训练路径上。
- **排程变化**：OPD-13 写的是"Ministral 第一折 RL 链结束后自动启动"；该链已于 13:56 停止（DM-8），113 空出，链即时启动，四个 run 与门槛不变。

## 2026-10-31 执行记录 OPD-15：按用户要求删除作废的 P 臂第二次尝试（OPD-5）的日志

- 删除 `$ATMOS_ROOT/logs/opd-qwen3-8b-F1-P-s42.attempt2-v1.log`、`logs/opd/run-P-F1-s42.attempt2-v1.log`、`logs/opd/diag-opd-qwen3-8b-F1-P-s42.attempt2-v1.jsonl`
  （OPD-5 作废的运行；记入 `cleanup-2026-10-31.log`）。该运行从未评测，不在任何报告里。
- Pk2 第一折（OPD-13）训练结束：链门槛 PASS（rc 0、120 步、蒸馏损失无 NaN / inf，`loss_max` 872.7 只记录）；全程 pearson ≥ 0.998、`rollout_corr/kl` ≤ 0.0007、
  梯度范数 ≤ 3.2，训练奖励 0.29 → 0.49（未发散）。17:29 合并（单文件 `model.safetensors` 16.4 GB、399 个张量，评测服务已加载），分片已删；评测进行中。
- 18:40 补充：Pk2 第一折评测完整（`_x_eval/opd-qwen3-8b-F1-Pk2-s42`：final 5,600、judge 2,800、test_id 64、textbook 92，0 错误行，`EVALDONE`；数字未读）。
  Pk2 第二折 17:29 起训练，第 44 步：pearson ≥ 0.998、kl ≤ 0.0007、奖励 0.30 → 0.38；梯度范数第 43 步出现一次 9.85（第 44 步 1.58；第一折最大 3.2），
  k2 无截断下的单步尖峰，全局裁剪 1.0，记录不设门槛，继续观察。
- 20:40 补充：Pk2 第二折训练结束，链门槛 PASS（rc 0、120 步、无 NaN / inf，`loss_max` 1,562.9 只记录）；全程 pearson ≥ 0.998、kl ≤ 0.0007、
  梯度范数最大 9.85（仅第 43 步），训练奖励 0.30 → 0.44（最低 0.29，未发散）。20:28 合并（399 个张量、16.4 GB，核过后删分片），评测进行中；
  Pshuf 第一折 20:28 起训练。
- 21:40 补充：Pk2 第二折评测完整（final 5,600、judge 2,800、test_id 64、textbook 92，`EVALDONE`；数字未读）——Pk2 两折齐。
  Pshuf 第一折第 37 步：pearson ≥ 0.998、kl ≤ 0.0007，teacher / student mass 最大 1.000001 / 1.0000004，`loss_max` 最大 31.9（< 100），奖励 0.30 → 0.37。
- 03:30 补充：Pk2 第二折评测完整（见上）。Pshuf 第一折：链门槛 PASS（23:57；rc 0、120 步、mass 最大 1.000001、`loss_max` 31.97 < 100）；全程 pearson ≥ 0.998、
  kl ≤ 0.0007、梯度范数 ≤ 1.25，奖励 0.29 → 0.43；合并（399 个张量、16.4 GB）核过后删分片；评测完整（5,600 / 2,800 / 64 / 92，`EVALDONE`；数字未读）。
  Pshuf 第二折第 119 步。

## 2026-11-01 执行记录 OPD-16：Pk2、Pshuf 两折评测齐，`opd_report_ext.py` 报告；H-OPD5 → 撤回 H-OPD3 的"未采样 token"读法；H-OPD6 → 增益需要正确的替代关系

- **运行**：四个 run 都过链门槛（OPD-15），全程 pearson ≥ 0.998、kl ≤ 0.0007；评测全部完整（5,600 / 2,800 / 64 / 92），0 基础设施错误行。Pshuf 第二折 03:30 门槛 PASS
  （mass 最大 1.000001、`loss_max` 33.8），合并后删分片（10:54）；`OPD CHAIN DONE` 04:18。
- **报告**（10:55，113 `rl-wt-d1a6d40`；`opd_report_ext.py` / `opd_report.py` / `x_report.py` / `main_set.json` / `report_qwen3-8b.json` 与 99e2978 逐字相同；
  `--seeds 42 43 --opd-seeds 42`，其余参数同 OPD-12 → `results/opd/report_ext_qwen3-8b.json`）。主检验集 7 个机制，无缺失，四组对比都完整：
  - 主检验集（训练折越界正确率，点）：A 0.0、P 70.7 [39.3, 96.4]、Pk2 51.4 [14.3, 85.7]、Pshuf 0.0 [0, 0]。
    Pk2 − A 51.4 [14.3, 85.7]；P − Pk2 19.3 [−2.9, 48.6]；Pshuf − A 0.0；P − Pshuf 70.7 [40.7, 96.4]。
    逐机制 Pk2：co2 1.0、displacement 1.0、sun_photometer 1.0、freezing 0.6，diffuse 0（P 1.0）、plume 0（P 0.2）、ch4 0（P 0）；Pshuf 七个全 0。
  - Z（14）：P 48.6、Pk2 32.5 [11.4, 55.7]、Pshuf 0.0；P − Pk2 16.1 [1.8, 34.3]；P − Pshuf 48.6 [25.7, 71.4]。
  - 全部 70 模板：A 51.2、P 49.8、Pk2 37.1、Pshuf 14.6；Pk2 − A −14.0 [−26.5, −1.2]；Pshuf − A −36.6 [−46.6, −27.1]。
  - 训练折原题（终点 − 基座）：A +53.6、P +11.8、Pk2 +20.3、Pshuf +9.1。
- **判定（按 OPD-13 写定的规则，机械执行）**：
  - **H-OPD5**：Pk2 − A ≥ 30 且下界 > 0 → "截断是 Pk3 学不会的原因：撤回 H-OPD3 的'未采样 token'读法，改为只报数字"。即只在采样 token 上蒸馏、去掉逐 token 截断后，
    7 个机制学会 4 个；OPD-11 的 H-OPD3 读法（越过屏障靠未采样 token 上的信号）不再成立，H-OPD3 改为只报数字。注意 Pk2 与 Pk3 同时差损失形式（k2 对 k3）与截断，
    本设计不能把作用分给两者中的哪一个，只能说"k3 + 截断"这一组合是 Pk3 失败的原因。
  - **H-OPD6**：P − Pshuf ≥ 30 且下界 > 0 → "增益需要正确的替代关系，不是教师题面里多了任何一句话"。
- **次要（只报数字）**：给教师错误的替代关系（Pshuf）不只学不会，还把全部模板的越界正确率从 A 的 51.2 拉到 14.6（−36.6），比无截断 k2 的代价（−14.0）大得多；
  P 本身 −1.4（OPD-11）。test_id 各终点 0.31–0.38（基座 0.34）。

## 2026-11-01 执行记录 OPD-17：更正 OPD-13 关于逐 token 截断的描述（结果与判定不变）

- **事实**（审查同步前核对源码与日志）：verl 的 `DistillationLossConfig.loss_max_clamp` 数据类默认 10.0，但训练实际读取的 yaml（`verl/trainer/config/distillation/distillation.yaml`）
  默认是 `null`；六个臂（P、S、Pk3、Pk2、Pshuf 及 P 种子 43）的训练日志里打印的生效配置都是 `loss_max_clamp: None`。`losses.py` 只在不为 None 时截断。
  即**没有任何臂做过全局逐 token 截断**；P / Pshuf 日志里的 `loss_max`（16–45 / 32–34）就是实际损失，不是"截断前的值"。
- **Pk3 的截断来自 k3 本身**：`core_algos.kl_penalty_forward` 的 k3 先把对数比截到 ±20，再把 k3 值截到 [−10, 10]（`torch.clamp(kld, min=-10, max=10)`）。
  学生采到、而特权教师给很低概率的 token 上 k3 ≈ (log p_学生 − log p_教师) − 1，超过 10 后被截住、梯度为 0——这正是 OPD-13 设想的机制，只是截断在 k3 内部。
  Pk3 的 `loss_max` 恰为 10.0 与此一致。
- **因此**：OPD-13 写的"verl 对每 token 的蒸馏损失截断在 10（各臂都有）"与"更正 OPD-11：截断对 P 同样生效"是错的，撤回；OPD-11 原句"截断是 Pk3 与 P 的差别之一"是对的。
  Pk2 的 `loss_max_clamp=null` 覆盖等于默认值、不起作用；Pk2 与 Pk3 的真实差别是损失形式（无截断的 k2 对内部截断的 k3）。OPD-16 的数字与 H-OPD5 / H-OPD6 的判定不变；
  H-OPD5 的读法相应写成"Pk3 学不会来自 k3（含其内部截断），不是因为只用了采样 token"，与 OPD-16 已注明的"损失形式与截断分不开"一致。

## 2026-11-08 补充 OPD-18：S 与 Pshuf 的种子 43（写于两者任何训练之前）

用户 2026-11-08 解除实验冻结。S、Pshuf 两折各加种子 43，设置、链门槛、合并与评测同 OPD-8 / OPD-13；设计、读法与报告口径写在
`PREREG_PAPER_ANALYSES.md` §23 E3。OPD-16 的判定不改；两种子平均的数字另存另报。
