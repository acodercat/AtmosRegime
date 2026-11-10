# 数据集与训练 / 测试划分

一页总览：项目里有哪些题，哪些用来训练，哪些只用来测试。细节的出处：每个文件是什么见
[`../data/README.md`](../data/README.md)，来源见 [`../data/PROVENANCE.md`](../data/PROVENANCE.md)，
冻结的划分见 [`../splits/MANIFEST.md`](../splits/MANIFEST.md)，分析组的定义见 [`DESIGN.md`](DESIGN.md)。

**数字是 2026-09-26 的。** 数据重新导入或陷阱集变动后会变，重新数一遍用：

```bash
uv run python -c "from atmoscoder_rl.datasets import PROBLEM_SETS as S; [print(n, len(f())) for n, f in S.items()]"
uv run python -m atmoscoder_rl.check_corpus          # 陷阱母题落在训练集内外各多少
```

## 一、有哪些数据集

都在 `data/`，按名字登记在 `atmoscoder_rl.datasets.PROBLEM_SETS`（`--set <名字>` 用的就是它）。

| 数据集 | 题数 | 是什么 | 用途 |
|---|--:|---|---|
| **core** | 436 | 13 本教材的核心题（母题） | 训练 / 测试划分只作用在这 436 道上 |
| unseen_book | 12 | 第 14 本教材（Lamb & Verlinde）的题 | 不在任何划分里，永不训练；测从没见过的教材 |
| **traps** | 335 | 陷阱题：母题改一个条件，让原解法失效 | 测试用，与母题配对比较 |
| traps_regime_twins | 31 | 另一批 regime 陷阱，其母题在 traps 里已有一道陷阱 | 只按类别单独分析；不能与 traps 合并统计（母题会被数两次） |
| traps_extended | 18 | 非大气领域的 regime 陷阱（以土力学为主） | 辅助证据 |
| variants_numeric | 1,730 | 数值变体：母题换一组输入数值（每道可变体的母题 5 个） | 训练集母题的变体用于训练 |
| variants_paraphrase | 2,180 | 改写变体：换说法，数值和答案不变（每道母题 5 个） | 测表述鲁棒性 |
| cross_domain | 131 | 水文、土壤、环境化学、生态四个非大气领域 | 测跨领域迁移，永不训练 |
| scaffolding_original / _stripped | 169 + 169 | 同一批题的「给提示 / 去掉提示」两版 | 测对题面提示的依赖 |

`problems`（783 条）是运行时拼出来的主语料：core 436 + unseen_book 12 + traps 335。

此外，**72 个 regime 模板**（32 个簇；`atmoscoder_rl/templates/`，簇表 `atmoscoder_rl/coverage_clusters.py`）是出题程序，
需要时现生成「公式成立 / 失效 / 失效但给出替代 / 只提示不成立」四个版本的题。它们不是存好的数据；划分见下文第四节。

## 二、训练 / 测试怎么分

冻结于 2026-09-02（`splits/splits.json`，种子 20260902，按「难度 × 类别」分层）。之后所有结果都以这份划分为准，
不重新生成。

**规则**
1. 划分单位是**母题**，不是单道题。母题的数值变体、改写变体和陷阱都跟着母题走，否则解法会通过变体泄漏到测试里。
2. **只有 `train`（及其前缀子集）能提供训练数据**，其余全部只测不训。
3. 训练之后，「436 题总体准确率」没有意义了，必须按划分报告。

**436 道核心题的去向**

| 划分 | 母题 | 角色 | 怎么选的 |
|---|--:|---|---|
| **train** | 190 | 训练 | 能做数值变体的母题；每道带 5 个变体，共 **1,140 条训练样本**（`datasets.train_families()`） |
| test_id | 64 | 测试 | 同分布：同样可变体、同样分层，没参与训练 |
| test_core_only | 63 | 测试 | 做不了数值变体、因此本来就无法训练的母题 |
| test_book | 56 | 测试 | 两本整本留出的教材（Andrews《大气物理导论》、Jacob《大气化学导论》），测跨教材迁移 |
| probe_trap | 67 | 探针 | 最初 67 道陷阱的母题，作陷阱的对照 |

这 5 组覆盖全部 436 道，唯一重叠是 probe_trap 与 test_book 共有 4 道母题。另有：

- **probe_scaffolding**（77）：留作测试的「给提示 / 去提示」题对，两种条件都测。
- **probe_cross_domain**（131）：即 cross_domain，永不训练。
- **train_50 / train_100**：train 的前 50 / 100 道母题，用于「数据量 vs 效果」曲线。

## 三、陷阱题与训练集的关系

335 道陷阱中，**130 道的母题在 train 里**（模型训练时见过母题，不能用来测训练对陷阱的影响），
**205 道的母题在测试 / 探针划分里**（probe_trap 63、test_id 50、test_core_only 49、test_book 43）。

所以报告用的对照组只取「母题不在 train」、且没有题面缺陷（`magnitude_defect`）的陷阱，由 `traps.arm_ids` 定义：

| 对照组 | 题数 | 测什么 |
|---|--:|---|
| regime | 46 | 条件越过公式适用范围，题面不提示（研究的主测量） |
| regime_given | 21 | 同上，但题面给出了新公式（内部对照：物理相同，去掉「自己识别」这一步） |
| geometry | 72 | 多一个几何细节、方法不变（类别对照） |
| regime_xdom | 14 | 非大气领域的 regime 陷阱（辅助证据） |

母题在 train 里的 46 道 regime 陷阱不进上面的对照组（`traps.targeted_regime_ids`）。原计划拿它们做「定向训练」，
该方案已被取代、没有运行（`PROPOSAL.md` §4）。

## 四、注意

1. **模板题的划分**：训练试点用 `results/pilot/split.json`（16 个单位中 10 个训练、6 个留出，种子 20260930）；
   T 实验用 32 个簇的两折划分（`PREREG_MAIN.md`）。划分都在看任何结果之前按簇定并提交。
   **已用过的模板种子不能再用于新的确认性实验**：20260924（试点）、20260927（预注册筛选）、20260928（预注册 RL 实验）；
   另有 1、777、901、12345、4242 用于试运行、盲解和审阅（生成的题只在临时目录里，没有进仓库）；20260929 用于第一步 b
   （已取消）；**预留**：20260930（训练试点的训练题）、20260931（训练试点的检验题）。
2. **`splits/baselines.json` 有两项已过时**（`splits/MANIFEST.md` 写明）：cross_domain 是补全单位之前算的；
   trap_gap 混用了新旧两批陷阱。引用训练前基线时避开这两项。
