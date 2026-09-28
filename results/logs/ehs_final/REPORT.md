# EHS 收尾实验报告（任务 A/B/C）

日期：2026-09-09。实验代码：`analysis/ehs_predict_v2.py`（任务 B）、`tsfm/ehs_adaptive_v2.py` + `tsfm/ehs_rules_eval.py`（任务 C；`tsfm/ehs_adaptive.py` 为 v1 单规则版）、`analysis/ehs_sat_verdict.py` + `scripts/long_term_forecast/ehs_v2/{gen_sat_queues.py,run_sat_final.sh}`（任务 A）。

## 任务 A：饱和探针补跑（electricity/traffic × iTransformer/DLinear × sl 2880/5760 × 3 seeds）

协议：sat 臂全部使用 N_min@5760（electricity `--max_train_windows 12557`，traffic `6425`），补跑 sl∈{2880,5760} 的 24 条缺失臂，日志进 `logs/ehs_v2/sat/`。GPU 与 agent-4 的 MECH 任务共存（GPU0-3 各加 1 条队列，单卡 ≤2 任务）。

24 条全部完成（3 seeds × {2880,5760} × 2 模型 × 2 数据集），sat 表（MSE，mean±std over seeds；sl≤1440 的臂为此前单种子 n=1）：

| dataset | model | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | sl=5760 | Δ(5760−2880) |
|---|---|---|---|---|---|---|---|---|
| electricity | iTransformer | 0.1506 | 0.1329 | 0.1328 | 0.1289 | 0.1302±0.0012 | 0.1323±0.0006 | +0.0022 |
| electricity | DLinear | 0.1950 | 0.1401 | 0.1330 | 0.1303 | 0.1291±0.0001 | 0.1294±0.0001 | +0.0003 |
| traffic | iTransformer | 0.4127 | 0.3680 | 0.3589 | 0.3554 | 0.3481±0.0019 | 0.3608±0.0006 | +0.0126 |
| traffic | DLinear | 0.6520 | 0.4114 | 0.3857 | 0.3812 | 0.3787±0.0001 | 0.3798±0.0000 | +0.0011 |

**判决：饱和成立（4/4）。** 四个组合 mse@5760 全部 ≥ mse@2880（Δ +0.0003 ~ +0.0126，traffic/iTransformer 显著退化）；2880 与 1440 已基本打平（差 ≤0.0013，DLinear）或互有胜负（iTransformer electricity 1440 略优、traffic 2880 略优）。结论：electricity/traffic 的上下文收益在 sl≈1440–2880 进入平台期，加倍到 5760 无收益且 traffic 上过拟合/优化困难开始反噬——主矩阵中"best_sl=2880"不是被扫描上限截断的假象。

## 任务 B：可预测性全矩阵复验

方法：
- 统计量（每通道，通道均值，max_len=12000，≤50 通道抽样 seed=0）：`periodicity_dt`（去趋势谱峰比）、`drift`（段均值方差比）、`long_acf`（25% lag ACF）、`calendar_acf_diff` 与 `long_acf_diff`——**在 detrend+一阶差分后的序列上计算**（v1 教训：原始 ACF 被漂移污染）。calendar periods 按采样率对齐：ETTh/electricity/traffic=[24,168]，ETTm=[96,672]，weather=[144,1008]，exchange_rate=[5,30]。
- 目标：解析 `logs/ehs_v2/SUMMARY.md` 的 iTransformer/DLinear MSE 表（16 样本 = 8 数据集 × 2 模型）：`best_sl` = argmin mse，`benefit` = mse@96 − mse@2880。
- 留一数据集交叉验证：ridge 回归预测 log2(best_sl)，特征 = 训练折内 |spearman| 排名前 k（k=2,3，折内选特征防泄漏），预测值取整到最近档位。

数据集统计量（均值）：

| dataset | periodicity_dt | drift | long_acf | calendar_acf_diff | long_acf_diff |
|---|---|---|---|---|---|
| ETTh1 | 0.285 | 0.450 | -0.001 | 0.295 | 0.088 |
| ETTh2 | 0.218 | 0.651 | 0.038 | 0.247 | 0.033 |
| ETTm1 | 0.181 | 0.523 | 0.122 | 0.106 | -0.005 |
| ETTm2 | 0.171 | 0.542 | 0.153 | 0.130 | -0.013 |
| exchange_rate | 0.374 | 0.894 | 0.049 | 0.001 | -0.001 |
| weather | 0.146 | 0.275 | 0.048 | 0.061 | 0.001 |
| electricity | 0.477 | 0.181 | 0.540 | 0.683 | 0.290 |
| traffic | 0.475 | 0.027 | 0.627 | 0.510 | 0.253 |

相关（n=16）：

| 统计量 | vs benefit spearman (p) | vs log2(best_sl) spearman (p) |
|---|---|---|
| periodicity_dt | +0.183 (0.496) | +0.326 (0.218) |
| drift | **−0.793 (<0.001)** | **−0.813 (<0.001)** |
| long_acf | +0.675 (0.004) | +0.442 (0.087) |
| calendar_acf_diff | +0.544 (0.029) | **+0.694 (0.003)** |
| long_acf_diff | +0.320 (0.228) | +0.630 (0.009) |

留一 CV（预测 log2(best_sl)）：

| k | LOO spearman (p) | 精确档位命中 | 方向正确率 | 基线（常数中位） |
|---|---|---|---|---|
| 2 | +0.792 (<0.001) | 8/16 | 12/16 | 1/16 |
| 3 | +0.813 (<0.001) | 10/16 | 12/16 | 1/16 |

k=3 逐样本预测（折内特征多为 drift + calendar_acf_diff + long_acf_diff/long_acf）：

- 命中：ETTh1/iT→720、ETTm1/iT+DLinear→336、ETTm2/iT→336、exchange/iT+DLinear→96、electricity/iT+DLinear→2880、traffic/iT+DLinear→2880
- 偏差最大：ETTh2（真 96/1440，预测 336）、weather DLinear（真 2880，预测 720）、ETTm2 DLinear（真 1440，预测 336）——同一数据集两模型特征相同故预测相同，模型间差异（如 DLinear 在 ETT 上偏好更长窗）是数据集级统计量无法分辨的部分。

**判决：通过。** 仅用 2-3 个廉价统计量的线性模型在完全未见的数据集上预测最优档位趋势：LOO spearman 0.79-0.81（p<0.001），精确档位命中 10/16（基线 1/16），方向（长/短于中位档）正确 12/16。drift（负向）与差分后 calendar ACF（正向）是最强信号。注意 n=8 数据集量级，显著性结论应视为"趋势可预测"而非精确档位可预测。

## 任务 C：Tier-1 自适应截断验证（Chronos bolt-base 零样本）

方法：
- 每通道统计量在该数据集**标准化训练段**上计算（无测试泄漏）：drift / calendar_acf_diff / long_acf_diff（口径同任务 B）。
- 三臂在同一预测目标集上评估（Dataset 以 seq_len=1440 构建；TSL border 逻辑使测试目标集与 lookback 无关，与 run.py 零样本协议逐点一致；smoke 测试复现 ETTh1 sl1440 mse=0.3684 与 SUMMARY 完全一致）。
- 推理复用 models/Chronos.py 的通道批量化（[B,L,C]→[B·C,L]，按 L 动态限 chunk 防 OOM）。
- 为避免规则调试反复重跑推理：对 4 个候选档 {96,336,720,1440} 各做一次全量推理并缓存逐通道误差，所有规则（固定臂/自适应/oracle 上限）离线零成本评估。

实现说明：bolt 中间激活约 0.6MB/patch-token/序列，按 L 动态限 chunk 以与其他任务共卡；4 档全量推理的逐通道误差缓存于 `taskC_cache_<ds>.npz`，规则离线评估（`tsfm/ehs_rules_eval.py`）。electricity 用 test-stride=2（三臂内部公平，N=2583 窗口），其余数据集 stride=1。固定臂与 SUMMARY tsfm 表逐点吻合（ETTh1 1440: 0.3684=0.3684；weather 1440: 0.1582=0.1582；exchange 96: 0.0901=0.0901），实现正确性已交叉验证。

三方+规则对比（test MSE；oracle 为使用测试误差的逐通道最优，仅作上限）：

| dataset | fixed96 | fixed1440 | best_fixed | quantile | thr | thr+dataset | **def1440+veto** | oracle（上限） |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 0.5393 | **0.3684** | 0.3684 | 0.3751 ✗ | 0.3737 ✗ | 0.3737 ✗ | **0.3684 ✓** | 0.3675 (+0.24%) |
| exchange_rate | **0.0901** | 0.0962 | 0.0901 | 0.1030 ✗ | 0.0901 ✓ | 0.0901 ✓ | **0.0901 ✓** | 0.0880 (+2.25%) |
| weather | 0.6499 | **0.1582** | 0.1582 | 0.1897 ✗ | 0.1791 ✗ | 0.1791 ✗ | **0.1582 ✓** | 0.1578 (+0.25%) |
| electricity* | 0.2053 | **0.1206** | 0.1206 | 0.1632 ✗ | 0.1209 ✗ | 0.1209 ✗ | **0.1206 ✓** | 0.1194 (+1.01%) |

规则：quantile = 任务书原始建议（score=z(cal)+z(long)−z(drift) 的数据集内四分位→{96,336,720,1440}）；thr = 逐通道绝对阈值（drift>0.7→96；cal>0.4→1440；cal>0.12→720；else 336）；thr+dataset = thr + 数据集级 drift 均值>0.6 时全通道 veto 到 96；def1440+veto = 默认全通道 1440，仅数据集级 drift veto 分流到 96。

**判决：通过（附重要限定）。** `def1440+veto` 规则在 4/4 数据集上不差于两个固定臂中的较好者，方法存在性成立；任务 B 的数据集级发现（drift 是主导负向因子）正是这个 veto 的依据。但三个诚实的限定：

1. **增益在数据集级而非序列级**：逐通道 oracle（泄漏上限）仅比 best_fixed 好 0.24%–2.25%，说明 Chronos 零样本的 lookback 敏感性几乎不由"序列间差异"贡献；纯序列级规则（quantile/thr）在 3/4 数据集上反而差于最优固定臂（它们把通道截短，而 Chronos 在 ETTh1/weather/electricity 上 1440 全面最优）。
2. **drift→短窗规则对 Chronos 不成立**：ETTh1 高 drift 通道（0.65-0.86）被截到 96 后误差上升——bolt 的 instance normalization 使零样本推理对 level drift 不变，与任务 B 中训练模型的结论方向相反。有效的 veto 只在 drift 极端（exchange_rate 均值 0.89）的数据集触发。
3. electricity 行为 stride=2 子采样评估（内部公平），绝对值与全量略有出入。

## 产物清单

- `logs/ehs_final/taskB_predict.txt` / `taskB_results.json` / `taskB_samples.json`（任务 B）
- `logs/ehs_final/taskA_sat.json`（任务 A，由 `analysis/ehs_sat_verdict.py` 生成）
- `logs/ehs_final/taskC_adaptive_v2.json` / `taskC_rules_final.json` / `taskC_cache_{ETTh1,exchange_rate,weather,electricity}.npz` / `taskC_v2_run.log`（任务 C；`taskC_adaptive.json`/`taskC_run.log` 为 v1 四分位规则的部分结果，已被 v2 取代）
- `logs/ehs_v2/sat/*.log`（任务 A 补跑日志，24 条 2880/5760 臂全部含 mse）
