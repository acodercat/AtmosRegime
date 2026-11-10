# 训练曲线（逐步指标）

每次训练一份 CSV：第一列 `step`，其余是该运行在控制台日志里记录的全部逐步指标。由 `python -m explore.export_curves --logs $ATMOS_ROOT/logs --runs <运行名> --out-dir results/curves`
从服务器日志导出；原始日志与 TensorBoard 事件文件在 `$ATMOS_ROOT/logs/<运行名>.log`、`$ATMOS_ROOT/logs/tensorboard/<运行名>/`。

画图常用的列：
- RL：`critic/score/mean`（训练奖励）、`response_length/mean`、`actor/entropy`、`actor/ppo_kl`、`actor/pg_clipfrac`、`training/filter_groups/evicted_samples`、`timing_s/step`；
  验证集上的分项在 `val-aux/<数据源>/<指标>/mean@1`（例如 `answer`、`judge/correct`、`cf/right`）。
- SFT：`train/loss`、`train/grad_norm`、`train/lr`。

| 文件 | 是什么 |
|---|---|
| `dapo-qwen3-8b`、`seed1-3`、`ctrl-random` | 主训练 run 2、三种子、随机奖励对照（`results/FINAL.md`） |

## PREREG_X 的运行（`x-<家族>-<折>-<臂>-s<种子>`、热身 `sft-x-<家族>-<折>`）

验证集是修订 X-5 的训练侧探针（每 20 步，256 行），只用于画图。X-16 重跑时共享日志 `logs/<运行>.log` 里新旧运行的行混在一起（X-26），
所以这些曲线不从它读，而按清单 `x_sources.json` 读每次运行自己的文件：

```
python -m explore.x_curve_sources --logs $ATMOS_ROOT/logs --out x_sources.json          # 在服务器上，列出每个文件选用或剔除的理由
python -m explore.x_val_regrade --points x-qwen3-8b-F2-F-s43:40 x-qwen3-8b-F1-A-s43:40 \
    --controls x-qwen3-8b-F2-F-s43:20 x-qwen3-8b-F1-A-s43:60 --root $ATMOS_ROOT --out x_val_regraded.json
python -m explore.export_curves --logs $ATMOS_ROOT/logs --manifest x_sources.json --val-override x_val_regraded.json --out-dir results/curves
```

- RL 运行读 X-16 开始（10-17 15:00）之后写出的 attempt 日志；X-16 保留的 Qwen3-8B F2 种子 42 的 A、D、F 读其全部日志。被剔除的只有
  Qwen3.5 F1-A、F1-D 种子 42 两个被取代运行留下的 attempt2。提前结束的 Qwen3.5 F1-WA、F1-WD、F2-WA 种子 42 止于第 110、113、116 步，最后的验证点是第 100 步。
  Qwen3-8B F2-D-s43 的第 0–9 步来自 attempt1、第 10–120 步来自续跑的 attempt2（续跑时在第 10 步多验证一次）。
- 热身：Qwen3.5 F2 的 nohup 属于 X-16 未用的 113 那次 SFT（X-22），用的是 114 那次，取 `logs/sft-x-qwen35-9b-F2.log` 的第一段；其余三个读各自的 nohup。
- X-24 的两个受污染验证点（Qwen3-8B F2-F-s43、F1-A-s43 第 40 步）换成重评值，`val_regraded` 列为 1。`x_val_regrade` 先核两件事才写出：
  逐行存值的均值与日志一致（四个点都一致）、对照点重评后 0 行变化；受污染点变化 59 行与 8 行（与 X-24 相同），验证分数 0.417 → 0.541、0.539 → 0.563。
- 解析：只取名字含字母的键（同一控制台行上混入的时间戳 `19:21` 不是指标），`np.int64(…)` 的整数指标也读入；旧运行的 CSV 是改动前导出的，含少量这种数字列。
- Qwen3.5 F2-F 种子 42 的 `response_length/clip_ratio` 从第 40 步起升到第 120 步的 0.79（X-27 的退化）。种子 43 与 R 臂跑完后照此重新生成清单再导出。

2026-10-29（X-41）：种子 43 与 R 臂结束后在 113 上重新生成清单（48 次运行：两家族 × 两折 × 五臂 × 两种子中的 40 次、4 个 R 臂、4 个 SFT）并按同一命令全部重新导出；
原有的 34 个 CSV 与重导出的逐字节相同，新增 14 个（Qwen3.5 种子 43 十个、R 臂四个）。验证点重评文件与 10-23 的相同。
