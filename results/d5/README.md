# D-5 教科书 regime 陷阱的独立重复（`../../docs/PREREG_MAIN.md` 修订 D-5、D-5a–c）

```
python -m explore.d5_report --input $ATMOS_ROOT/experiments/_inputs/d5/textbook.json \
  --run-dir $ATMOS_ROOT/experiments/families_d5/families_context_v2 \
  --d-input $ATMOS_ROOT/experiments/_inputs/families_v4/textbook.json \
  --d-run-dir $ATMOS_ROOT/experiments/_reports_20261012/d2_runs --out results/d5/report.json
```

- 输入：D-5 207 行，sha256 `cc3d59e2…`（与 D-5 §2 一致）；新陷阱 regime:recognise 54 道、geometry_detail 56 道，与 D 的陷阱无重叠。
- 运行：D 的 16 个配置 × 3 次，各沿用其 D 运行的服务与采样设置（`serving.json`），48 个文件各 207 条、无错误记录，全部配置完成。
- 新旧合并用的 D 运行是 `_reports_20261012/d2_runs`（D 报告 `results/d2/report.json` 所读的 48 个文件，各 398 行、输入 sha256 `710be63c…`）；
  用它重算的 D 合并值与 `results/d2/report.json` 相同（11.1 [−0.4, 22.8]）。
- 单位百分点；每道陷阱的配对差 = 母题对 − 陷阱对（3 次平均），配置间平均后，两族各自按陷阱有放回重抽 10,000 次，`random.Random("d5-pool:20261013")`。

## 主结果 D5-1

regime:recognise − geometry_detail，16 个配置合并：**12.7 [4.4, 21.3]**，按写定的读法 **"在独立陷阱上重复出现"**。

两族各自的母题 − 陷阱差（描述）：regime:recognise 15.8，geometry_detail 3.1——效应来自 regime 陷阱，几何细节陷阱几乎不掉分。

## 次要结果（只报数字）

| | regime − geometry |
|---|--:|
| 关思考（9 个配置） | 12.6 [3.5, 22.0] |
| 开思考（7 个配置） | 12.9 [4.1, 21.8] |
| 新旧合并 92 + 128 道（事后决定：加题时已知旧结果） | 12.3 [5.5, 19.3] |

| 配置 | regime − geometry |
|---|--:|
| qwen3-8b 关 | 12.0 [−3.6, 27.7] |
| qwen3-8b 开 | 19.2 [3.7, 35.0] |
| qwen3.5-9b 关 | 20.9 [5.6, 36.8] |
| qwen3.5-9b 开 | 17.4 [4.6, 30.2] |
| qwen3.5-27b 关 | 5.1 [−7.5, 17.4] |
| qwen3.5-27b 开 | 7.4 [−3.0, 17.7] |
| gemma4-e4b 关 | 22.2 [5.7, 38.8] |
| gemma4-e4b 开 | 13.1 [−4.4, 30.7] |
| gemma4-31b 关 | 5.2 [−6.2, 16.9] |
| gemma4-31b 开 | 9.1 [1.9, 17.1] |
| granite4.2-8b 关 | 8.2 [−5.6, 22.2] |
| granite4.2-8b 开 | 9.3 [−5.9, 24.9] |
| granite4.2-30b 关 | 12.5 [−4.0, 29.7] |
| granite4.2-30b 开 | 14.7 [1.3, 28.6] |
| ministral3-8b 关 | 9.1 [−6.6, 25.4] |
| ministral3-14b 关 | 18.0 [2.3, 33.8] |

16 个点估计全部为正；单个配置的区间宽（约 ±15 点），只作描述。Qwen3.5-27B 与 Gemma 4 31B 的差最小（5–9 点），但同样较大的 Granite 4.2 30B 是 12–15 点，不能读成"随规模减小"；
合并估计里开 / 关思考之间没有差别。

## 局限（D-5 §5）

- 新的 regime 陷阱集中在热力学与动力学，与测试侧一样经过"推理模型能做对"的入选筛选。
- 双生陷阱与其几何陷阱共用母题，两族独立重抽忽略这一正相关，区间偏宽（保守方向）。
- D-1 的判定不变；本结果只说明 S4 的方向在一批互不重叠的陷阱上再次出现，且区间不再含 0。
