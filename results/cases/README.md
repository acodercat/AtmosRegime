# 高价值案例（论文素材；规则选出、逐字摘录，2026-10-29）

```
python -m explore.case_mining --bench <Bench-Lab> --dose-eval experiments/dose/eval --dose-sft experiments/dose/sft_data \
    --x-textbook experiments/x_textbook --final experiments/_inputs/kc/final.json --dose-posthoc results/dose/posthoc_qwen3-8b.json \
    --out results/cases/candidates.json
```

`candidates.json` 是十条规则（脚本 docstring 列出：A/A2/B/C/D/E/F/G/G2/H）在存档运行上的全部命中，每条带来源 id 与原文；本文从中挑出
写论文可直接引用的案例，每例注明来自哪条规则、哪个运行。**案例只作说明，不作证据**：规则是看过结果之后写的，数量不构成率，
相应的率在 `results/paper/`、`results/traps_rl/`、`results/dose/`。摘录只删 docstring 与空行，不改一字。候选总数：
A 8、C 8（与 A 同 8 道）、A2 1、B 1、H 10、E 3、D 4、F 4、G 2、G2 4。

## 一、前沿模型"会做原题、条件一变就照套公式"（规则 A：deepseek-v4-flash 关思考，3 次运行里 ≥ 2 次错且等于捷径值，父题 ≥ 2 次对）

规则 A 命中 8 道；其中 `trap_13.5_gen` 带 Bench 评审的 `magnitude_defect` 标记（CO₂ 设为 2.8e-4 ppmv，地球大气从未有过；`results/paper/` 的陷阱分析都剔除它），表中保留但不作案例，可用的 7 道。7 道在开思考下都在 ≥ 2/3 次运行中做对（规则 C；holton_28、air_63、ca_5.3 各有一次失败）。注意：错答等于捷径值是规则 A 的**入选条件**，不是观察结果；观察结果是这些题存在。

| 陷阱 | 类别 | 越界机制 | 关思考答案 | 捷径值 | 正确值 | 失败/捕获 |
|---|---|---|---|---|---|---|
| trap_holton_28_gen | 动力学 | 两球 Coriolis 偏转：要错开的是球心距减直径，关思考按半程直线算 | 3.98e-6 m/s | 3.98e-6 | 2.98e-6 | 3/3 |
| trap_7.1_gen | 观测 | 3000 km 高的薄层，临边路径质量比不能再用 z ≪ a 近似 | 71.4 | 71.4 | 86.6 | 3/3 |
| ~~trap_13.5_gen~~（magnitude_defect，不用） | 化学 | CO₂ 极稀时雨水 pH 受水自电离下限约束 | pH 8.73 | 8.73 | 7.00 | 2/3 |
| trap_air_63_gen | 热力学 | 100 km 深柱里重力随高度平方反比减小 | 1.16e-4 kPa | 1.16e-4 | 1.43e-4 | 3/3 |
| trap_air_91_gen | 辐射 | 热的半透明体：探测到的出射辐射含自身发射项 | 0.2/0.3/0.5 | 同 | 0.04/0.14/0.82 | 3/3 |
| trap_ca_5.3_gen | 空气质量 | 层流无混合的 ESP：收集是线性的，不是 Deutsch–Anderson 指数律 | 95 941 ft² | 95 941 | 20 625 | 3/3 |
| trap_p_114_gen | 热力学 | 气旋沿陆架以 U 移动：逆气压响应被 Proudman 共振放大 | 0.60 m | 0.60 | 3.23 | 2/3 |
| trap_snp_30_gen | 化学 | NO₂ 与 O₃ 等量且封闭：二级动力学，不是赝一级 | 32 243 s | 32 243 | 55 402 | 2/3 |

**案例 1 · trap_air_63_gen（热力学）**。题目明说"取地球半径 R_E"，关思考的代码把 R_E 放进参数表却从未使用——沿用等温大气指数律：

```python
def solve(P0=101.325, a=0.0342, RE=6371e3, z=100e3, T1=250.0, T2=300.0):
    # For an isothermal atmosphere, pressure decreases exponentially:
    # P(z) = P0 * exp(-z / H)   ...   So H = T / a
    H1 = T1 / a; H2 = T2 / a
    P1 = P0 * math.exp(-z / H1); P2 = P0 * math.exp(-z / H2)
```

开思考直接写出球面形式 `P = P0 * exp(-(a/T) * R_E*z/(R_E+z))`，注释 "geopotential height correction"。
同一模型、同一题、同一组参数：差别只在是否被允许想。

**案例 2 · trap_ca_5.3_gen（空气质量）**。题目用整整一句话描述层流无混合（"laminar, unmixed plug flow, with no transverse turbulent mixing and
particles initially uniformly distributed"），关思考的代码把这句话抄进注释，然后套了它所否定的公式：

```python
    # Deutsch-Anderson equation for laminar, unmixed plug flow:
    # efficiency = 1 - exp(- (A * w) / Q)
    A_total = - (flow_rate_ft3_per_s / drift_velocity_ft_per_s) * math.log(1 - efficiency)
```

Deutsch–Anderson 的指数律正是**充分横向混合**的推论；层流下效率 = w·A/Q，99 % 只需 20 625 ft²，而不是 95 941 ft²（4.65 倍）。

**案例 3 · trap_holton_28_gen（动力学，Holton 习题）**。关思考用 `deflection = Ω sinφ (d/2)²/v`（每球走半程、直线偏转）；开思考用
`V = f (d² − r_min²)/(4 r_min)`，即把"刚好错过"写成最小球心距等于直径。前者 3.98e-6，后者 2.98e-6 m/s；差别是 4/3 倍，三次运行全部落在捷径值上。

**案例 4 · trap_snp_30_gen（化学）**。NO₂、O₃ 都是 50 ppb、封闭气团：关思考三次里两次算 `tau = 1/(k·[O3])`（赝一级，O₃ 视为过量；第三次做对），
开思考改成二级 `tau = (e − 1)/(k·[NO2])`——e 折寿命在二级反应里不再是 1/(k c₀)。

## 二、推理模型"知道却仍照套"（规则 B：开思考错且等于捷径，推理文本含有效性词；规则 A2：gpt55-reasoning 的失败）

**案例 5 · trap_p_108_gen（动力学，龙卷风拉伸）** — 规则 B 唯一命中，`deepseek-v4-flash-reasoning.run1`，推理 7 139 字符。
推理里**两种算法都算了**：相对涡度拉伸给 69.1 min，绝对涡度 (ζ+f) 拉伸给 46.0 min，然后写道：

> "But is there any effect of Coriolis? Usually, conservation of potential vorticity includes absolute vorticity. … If we use absolute
> vorticity, it's about 46 minutes. Which one is more likely? I recall a similar problem: … The answer is about 69 minutes. I think I've
> seen something like that. Thus, we'll go with relative vorticity."

最终代码 `t = T * math.log(ζ_final / ζ0)`，f 作为参数传入但不用。正确答案 46.0 min（位涡守恒拉伸的是绝对涡度）。
这是"用记忆中的教科书答案覆盖自己的物理推理"的逐字记录；也是 Qwen3-8B 任何 RL 终点都没学会的 4 道之一（规则 G2）。

**案例 6 · trap_5.3_gen（动力学，内重力波）** — 规则 A2：gpt55-reasoning 在 ≥ 2/3 次运行中错的只有 5 道（至少错一次 9 道），这一道 2/3 次错，不等于捷径值。
核查后这是**答案约定的分歧**：三个主配置的全部 9 次运行子答案 4 都给 81.65 m，恰是标准答案 163.3 m 的一半——教科书取峰峰值位移 2u₀/ω，模型取振幅 u₀/ω。
不作为证据，只说明前沿推理模型的残余失败需要逐例核查。

## 三、注入一粒种子后 RL 学会（规则 D：同一单元，剂量 0 的种子 RL 为 0，剂量 ≥ 1 的种子 RL ≥ 0.95）

DOSE 的交叉设计给了同一机制在两个种子里的对照（`results/dose/posthoc_qwen3-8b.json`）：

| 机制 | 折 | 剂量（种子） | 热身后 p（剂量种子 / 零种子） | 零种子 RL 的做法 | 剂量种子 RL 的做法 |
|---|---|---|---|---|---|
| line_broadening | F1 | 4（s42）/ 0（s43） | 0.049 / 0 | `HWHM_final = HWHM_initial * (P_final/P_initial)` 纯压力标度 | 加 Doppler 半宽，Voigt 合成 |
| doppler_aliasing | F2 | 2（s43）/ 0（s42） | 0.349 / 0 | "the reported velocity is the actual velocity" | Nyquist 速度 `vn = λ·PRF/4`，折叠到 [−vn, vn] |
| inertia_gravity_wave | F2 | 1（s43）/ 0（s42） | 0.0047 / 0 | `T = 2π λx/(N λz)` 无旋转 | `ω² = f² + N² k²/(k²+m²)` 含 Coriolis |
| surface_multiple_reflection | F1 | 1（s42）/ 0（s43） | 0.033 / 0.016 | `S·T·(1−α)` 单程 | `S·T·(1−α)/(1−α·R)` 多次反射级数 |

**案例 7 · inertia_gravity_wave（剂量 1）**。热身后同一种子的模型已经算出 Coriolis 参数 `f = 2Ω sinφ`，却把它丢掉，周期写成
`T = 2π/(N·k·m)`（量纲都不对）；RL 之后代码把 f 放进色散关系。一个注入示例把 p 从 0 提到 0.0047（64 次采样里不到 1 次），RL 便把它放大到 20/20。
零剂量种子的 RL 终点是 `2π λx/(N λz)`，即无旋转内波周期——正是设计的捷径。

**案例 8 · doppler_aliasing（剂量 2）**。零剂量种子的 RL 代码注释："the reported radial velocity is the actual velocity since the problem states that
the radar reports its velocity estimate without further processing"——把题干里"不做去折叠处理"读成"不发生折叠"。剂量种子的 RL 正确区分了两者。

## 四、注入了正确程序仍学不会（规则 E：剂量 ≥ 2 而两种子 RL 都为 0）

| 机制 | 折 / 种子 / 剂量 | 热身后 p | 热身捷径行 / RL 捷径行（20 行） | RL 终点的做法 |
|---|---|---|---|---|
| henry_depletion | F1 s42 d2 | 0.0008 | 13 / 20 | 平 Henry 定律，不扣除溶入液相的气量 |
| plume_ground_reflection | F2 s43 d4 | 0.0031 | 2 / 20 | 高斯烟羽**无像源项** |
| displacement_height_canopy | F1 s42 d2 | 0.0031 | 20 / 20 | `ln(z/z0)`，无零平面位移 |

**案例 9 · plume_ground_reflection（剂量 4）**。训练集里有 4 份逐字的正确程序（含 `exp(-(z+H)²/2σz²)` 像源项），RL 终点仍写：

```python
    # C = (Q / (u * 2 * pi * sigma_y * sigma_z)) * exp(-((z - H)^2) / (2 * sigma_z^2))
    # Since the receptor height z is equal to the stack's effective height H, (z - H) = 0, so exp(0) = 1
    concentration = (Q) / (u * 2 * math.pi * sigma_y * sigma_z)
```

热身后 20 行里只有 2 行是捷径值（其余各式各样），RL 后 20/20 全是捷径值：**RL 没有学到正确做法，而是把多样的错误统一成了标准公式**。
这三例符合 DOSE-4 的 p 规则（p ≤ 0.006 → 未学会），是 H2（热身后 pass@64 > 0 即可学会）的反例；也是 DOSE-text 正在检验的对象（文本能否把 p 抬过阈值）。

## 五、RL 的"统一"效应（规则 F：热身捷径 ≤ 2/20，RL 终点捷径 20/20）

4 命中：inertia_gravity_wave（F2 s42 d0）、plume_ground_reflection（F2 s43 d4）、spherical_slant_path（F2 s42 d1；F2 s43 d0）。

**案例 10 · spherical_slant_path**。热身后两行的答案是 587 与 369 km（s42）、6.0 与 14 515 km（s43）——四次四样；RL 后全部是 `h/cos(z)`
平面平行斜程，注释逐字为：

> "# However, we must account for the curvature of the Earth if the height is significant compared to Earth's radius.
> # But since the height of the layer (6 km) is much smaller than Earth's radius (6371 km), the curvature effect is negligible"

题目的天顶角接近 90°，此时 h ≪ R 不是判据（应看 h/(R cos²z)）；模型说出了正确的判据名，用了错误的判据。这与 `results/traps_rl/README.md`
第 4 条（失败形态向父题方法收敛）和 `PREREG_PAPER_ANALYSES` §5 的"自信地错"检验是同一现象的个案面。

## 六、代码注释里"知道"替代概念（规则 H：DOSE 模型关思考，答案等于捷径值，代码提到替代概念关键词）

10 命中、7 个机制。最清楚的四条（标签 → 注释原文 → 实际做法）：

- cunningham_slip，`sft-dose-qwen3-8b-F2-s42`："# Calculate the Knudsen number" → 算了 Kn，然后用纯 Stokes 速度；s43 更进一步：
  `(1 + 0.15 * knudsen**2)`——编了一个不存在的修正式。
- doppler_aliasing，`dose-qwen3-8b-F2-WA-s42`："# affected by the Doppler shift limitations (e.g., Nyquist frequency)." → 报真实速度。
- diffuse_transmission，`dose-qwen3-8b-F1-WA-s42`："# Since the radiation is isotropic over the upward hemisphere, the flux decreases exponentially"
  → 用了 `exp(−τ)` 而非 2E₃(τ)（或 1.66 倍扩散因子）。
- unconfined_dupuit，`sft-dose-qwen3-8b-F1-s42`："# Calculate the drawdown using the Dupuit–Thiem equation" → 用的是承压 Thiem（水头线性），不是 Dupuit 的 h² 形式。

## 七、训练后学会 / 没学会的教科书陷阱（规则 G / G2：Qwen3-8B 基座错，A 臂 4 终点全对 / A 臂 4 终点全错且都给捷径值）

**学会（2 道，都是化学）**：
- trap_6.2_gen（反应 Gibbs 能，HNO₃ 分压 1e-13 bar）：基座写出了 `ΔG = ΔG° + RT ln Q` 与正确的 Q，却把 kJ 的 ΔG° 与 J 的 RT ln Q 直接相加（答 −74.1）；
  A 臂把每个活度逐项写出（a_NO3 = a_H2O = a_OH = 1，a_HNO3 = p/p°）并统一单位，−2.10 kJ/mol；捷径 72.1 kJ/mol（只看 ΔG°）基座与 A 臂都没有落入。
- trap_fp_6.1_gen（10 Torr 低压腔里的 ppb → 分子数密度）：基座用 1 atm 下的 `2.69e10 molecules cm⁻³ per ppb` 常数（答 1.55 min，另有单位错）；A 臂从 `P/(RT)·N_A` 按 10 Torr 算，911.9 min；捷径值 13.9 min。
  两道的共同点是 A 臂把"记住的数 / 记住的式"换成了按题给条件逐项计算，与 `results/traps_rl/README.md` 的化学陷阱增益（7 道 +39 到 +50 点）同向。

**没学会（4 道；A 臂 4 终点全部给捷径值）**：trap_air_226_gen（球面上的涡度）、trap_air_301_gen（凝结水负荷）、trap_dn_4.4_regime（颗粒相对气体的沉降速度）、trap_p_108_gen（案例 5）。
- **案例 11 · trap_air_301_gen** 值得单列：它的机制 condensate_loading 正是 X 的 72 个模板之一，Qwen3-8B 在模板上 pass@1 0.35，A/D/F 六个终点全部 1.0；
  但同一机制的教科书陷阱（Stull 的雷暴上升气流题，含 `−g·r_l` 项）在 22 个训练后 / 热身终点里 21 个答 19.14–19.36 m/s（捷径值 2 % 内，不扣负荷；1 个无输出），基座答 1800（单位错），正确值 16.07
  （统计自 `experiments/x_textbook/eval/*/textbook.json`，不在 `candidates.json` 里）。
  即**模板上学会了"要扣凝结水负荷"，没有迁移到措辞不同的教科书题**——这是 `PREREG_PAPER_ANALYSES` "模板 ↔ 陷阱同机制交叉验证"（第二期）的一个现成对子，也是对第三节"学会"含义的边界说明。
- trap_air_226_gen 的 A 臂代码注释长达十行，讨论"flat Earth vs spherical"，最后仍写 `ζ = 2v/r`；与规则 H 同型。

## 用法与边界

- 引用时写明运行标签与 id（表中给出），数值可在对应 `experiments/` 文件里核对；规则定义在 `explore/case_mining.py` docstring。
- 规则 A/B/C 只看了 deepseek-v4-flash 与 gpt55-reasoning 的 Bench 存档（3 次运行），规则 D–H 只看了 Qwen3-8B 的 DOSE 与 X 运行；
  Qwen3.5-9B 没有进入案例库。
- "知道却照套"（B、H）的判据是关键词，命中不等于模型真的理解了替代概念；只用于说明，率不从这里取。
- 本目录由看过全部结果后写的规则生成，不改任何预注册判定。
