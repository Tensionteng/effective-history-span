# EHS x GIFT-Eval：零样本规模扩展测量报告

> 2026-09-16。目的：把 EHS 两轴测量从 8 数据集 × 训练模型扩到 GIFT-Eval 规模 × Chronos-Bolt 零样本（Significance 4→5 的关键证据）。
> **判决：两轴模式在 GIFT 规模零样本设置下不保持**——原方向相关全部消失，唯一稳定的显著相关是反向的。详见 §5。

## 1. 数据获取（hf-mirror.com）

- `Salesforce/GiftEvalPretrain` 在镜像上 401（gated），**`Salesforce/GiftEval` 可达**（HF 官方直连不通，全程走 hf-mirror）。
- 下载全部 56 个 `data-*.arrow`（`logs/gift_ext/download.py`，8 并发，0 失败，1.1GB → `data/raw/`）。`Monash-University/monash_tsf` 未动用（不需要）。
- `logs/gift_ext/inventory.csv`：56 个 config 的 freq / 序列数 / 长度 / NaN 率清点。

## 2. 任务选择与统一缓存

GIFT-Eval 官方协议（`SalesforceAIResearch/gift-eval` 的 `src/gift_eval/data.py`）：
- `prediction_length` = SHORT term（×1）× 频率基准 {M:12, W:8, D:30, H:48, T:48, S:60}（M4 另有映射，本集合无 M4 入选）；
- `windows` = min(20, max(1, ceil(0.1 × min_series_len / P)))；test = 末尾 `windows × P` 步，按 P 滚动；
- 多变量按 `MultivariateToUnivariate` 拆成单变量序列（Chronos 本就单变量）。

**入选规则**：`min_series_len ≥ 2048 + windows×P`（保证 L≤2048 各臂上下文完整；L=2880 臂 bolt 内部左截断到 context_length=2048，已注明），+ 1 个标注任务（hierarchical_sales/D：min input 1615，仅 2880 臂被截）。
每任务确定性采样 ≤64 条序列（seed 0）→ `data/cache/<task>.parquet` + `tasks.json`。

**最终 25 个任务**（频率 × 领域双覆盖）：

| 频率 | 任务 |
|---|---|
| 10S | bizitobs_application, bizitobs_service |
| 5T | LOOP_SEATTLE/5T, bitbrains_fast_storage, bitbrains_rnd, bizitobs_l2c/5T |
| 15T | SZ_TAXI, ett1, ett2, electricity |
| 10T | jena_weather, solar |
| H | LOOP_SEATTLE, M_DENSE, bizitobs_l2c, ett1, ett2, electricity, jena_weather, kdd_cup_2018 |
| D | saugeenday, us_births, hierarchical_sales(capped) |
| W | saugeenday |

领域：energy 8 / cloud 5 / transport 4 / nature 4 / demo 1 / sales 1。
**排除记录**（31 个 config，原因均为序列不够长，见 inventory.csv）：m4_*（6 个，min_len 19–4227）、car_parts(M,51)、covid_deaths(D,212)、hospital(M,84)、restaurant(D,67)、temperature_rain(D,725)、ett*/D,/W、solar/D,/W、electricity/D,/W、us_births/M,/W、saugeenday/M、bitbrains*/H、SZ_TAXI/H、LOOP_SEATTLE/D、M_DENSE/D、kdd/D、hierarchical_sales/W、bizitobs_l2c 无（已入选）。
**注意**：低频宏观/财经类（M4、exchange 类）全部因长度被排除——GIFT 长序列子集天然偏向周期性物理/IT 过程，这是外推时的选择偏差（见 §5 解释边界）。

## 3. 扫描（`scan.py`，GPU 0/1 各一进程）

- Chronos-Bolt base（本地缓存，HF_HUB_OFFLINE=1），bf16；每任务 × L∈{96,336,720,1440,2880→2048}。
- 上下文 = 窗口输入的末尾 min(L,2048) 步；NaN 输入由 bolt 原生 mask；NaN target 步在所有指标中剔除（with_missing 口径：只在观测值上评测）。
- 每任务 64 序列 × ≤20 窗 × 5 L ≈ 全集合 12 万条序列级预测，GPU 总耗时 ~4 分钟（`scan_gpu0/1.log`）。
- 指标（`scan_summary.csv`，125 行）：**CRPS**（GIFT 口径 mean weighted quantile loss，quantile 0.1–0.9，per-item 2ΣQL/Σ|y| 后取均值）、**MASE**（seasonal_period=1，gluonts/GIFT 口径）、**nMSE**（seasonal-naive 归一 MSE，季节 = 日历周期）、MSE/MAE。分位数预测全量存 `scan/<task>.npz` 供离线重算（`recompute.py`）。

## 4. 两轴统计量（`stats.py` → `stats.csv`）

严格复用 `analysis/ehs_stats.py` / `ehs_predict_v2.py` 口径，只用 train 段（series[:−W×P]，≤12000 步，无测试泄漏；统计前线性插值 NaN）：
- **轴1 drift**：20 段均值方差 / 总方差；
- **轴2 calendar_acf_diff**：日历 lag 处 ACF（去趋势+一阶差分后）；**long_acf_diff**：25% lag ACF（同处理）；
- 辅助：periodicity_dt（去趋势谱峰比）、long_acf（原序列 25% lag）。
日历 periods 映射：H[24,168]、15T[96,672]、10T[144,1008]、5T[288,2016]、10S[360]、D[7,30]、W[4,52]。

## 5. 结果与判决

### 5.1 逐任务最优 L / 长上下文收益（CRPS）

- best_L 分布：**96×0**, 336×3, 720×6, 1440×8, 2880×7；EHS@1%容差（收益<1% 的最小 L，抗 argmin 噪声）：336×4, 720×9, 1440×11, 2880×1。
- **长上下文收益 (CRPS@96−CRPS@2880)/CRPS@96 > 0：23/25**；bootstrap（1000 次 item 重采样）显著为正 **19/25**（p>0.95），显著为负 **0/25**。仅 bitbrains_fast_storage（−5.6%, p=0.33）与 solar_H（−0.3%, p=0.45）为负且不显著。
- 即：**对零样本 bolt，"漂移 → 长上下文有害"的害通道几乎从不出现**——收益变异被压缩到"几乎都是正"的区间。

### 5.2 Spearman 相关（n=25；`correlations.csv` 含全部子集）

| 统计量 | 目标 | 旧结果(8ds×2模型) | GIFT-25 | 判决 |
|---|---|---|---|---|
| drift | rel_benefit CRPS | −0.813 (p=0.001) | **−0.197 (p=0.345)** | 方向保持，显著性消失 |
| calendar_acf_diff | rel_benefit CRPS | +0.694 (p=0.004) | **−0.175 (p=0.402)** | **符号翻转** |
| long_acf_diff | rel_benefit CRPS | +0.630 (p=0.012) | **−0.249 (p=0.230)** | **符号翻转** |
| drift | log2 best_L CRPS | （旧：漂移→短 L） | **+0.490 (p=0.013)*** | **反向显著** |
| calendar_acf_diff | log2 EHS@1% CRPS | （旧：周期→长 L） | −0.384 (p=0.058)；去 capped −0.425 (p=0.038)* | **反向** |
| long_acf_diff | rel_benefit nMSE | — | **−0.483 (p=0.014)***（去 capped −0.474*；n≥20 子集 −0.300 ns） | **反向且最稳** |

- 显著单元格跨指标/子集不互相复现（drift→best_L +0.49 在 EHS@1% 目标上塌缩到 +0.07 ns——argmin 在收益平台上的噪声所致）；唯一跨子集稳定的是 **long_acf_diff 与收益的负相关**。
- n_series≥20 子集（n=15，更稳的任务级估计）：所有相关 |ρ|≤0.36、p>0.19，无一显著。

**判决：两轴模式（漂移轴驱动害、周期/长记忆轴驱动益）在 GIFT 规模零样本设置下不保持。**
- 害通道消失：漂移不再显著惩罚长上下文（23/25 正收益，0 显著负收益）；
- 益通道反转：周期/长记忆越强，长上下文边际收益反而越小（唯一稳定显著的方向）。机制解释候选：对通用基础模型，强周期序列用少量几个周期上下文即可外推（seasonal-naive 已很强），长历史无新增信息；而训练模型的旧定律里周期性是"长上下文安全/有用"的条件。两个世界（训练 vs 零样本基础模型）的 EHS 规律不同——这本身就是结论。
- 旧结果中 drift→短 L 的核心案例是 exchange_rate（金融日频），该类序列在 GIFT 长序列子集中不存在（长度过滤全灭）——两轴定律可能在"漂移主导金融型"序列上仍成立，但 GIFT 无法检验该象限。

### 5.3 两轴分布图（`fig_two_axes.png`）

- 左：drift × calendar_acf_diff 散点（色=EHS@1%，点大小=相对收益）。高漂移区（ETT 系，drift 0.46–0.66）不再对应短 EHS；高周期区（electricity_H 0.71、solar_H 0.68、M_DENSE 0.65）EHS 反而中长（720–1440）。
- 右：逐任务主导轴（z(drift)−z(calendar)）：**漂移主导 15 / 周期主导 10**（漂移主导：ETT×4、jena×2、kdd、SZ_TAXI、LOOP_SEATTLE/5T、bitbrains×2、bizitobs_l2c/5T、saugeenday/D、us_births、hierarchical_sales；周期主导：electricity×2、LOOP_SEATTLE/H、M_DENSE、bizitobs×3、solar×2、saugeenday/W）。
- 主导轴与 EHS 无线性关系（上述相关全部不显著）——"漂移主导→短 EHS、周期主导→长 EHS"的旧映射在零样本下不成立。

## 6. 口径与边界（诚实披露）

1. L=2880 臂 = bolt context cap 2048（左截断，与 e1 相同口径）；hierarchical_sales/D 的 2880 臂再截到 ≤1615（已标注 capped，去它结论不变）。
2. 每任务序列采样 ≤64（seed 0）；us_births/saugeenday 仅 1 序列、bizitobs_application 2 序列——靠 ≤20 个滚动窗撑 item 数，任务级指标噪声大，n_series≥20 子集分析已单列。
3. CRPS/MASE 聚合为 per-item 后取均值（GIFT/glunots 口径）；nMSE 季节 = 日历周期。点预测 = 9 分位数均值（MSE/MAE/nMSE），MASE 用 0.5 分位数。
4. 统计量 max_len=12000 与旧工作一致；这意味着对超长序列（electricity/15T 14 万步）只测头部 1.2 万步的漂移/周期。
5. 预训练泄漏：Chronos-Bolt 预训练语料未知，GIFT-Eval 官方也用它做零样本基准；本测量与官方口径一致，不做额外泄漏控制。
6. jena_weather 裸目录 = jena_weather/10T 重复，只用后者。electricity/15T 的两个 cache-*.arrow 为冗余缓存文件，未下载。

## 7. 产物清单（logs/gift_ext/）

| 路径 | 内容 |
|---|---|
| `data/raw/` (1.1GB, 56 arrow) + `tree.json` + `download.py/.log` | 原始数据（hf-mirror 镜像） |
| `inventory.csv` + `inventory.py` | 56 config 清点（freq/长度/NaN/协议参数） |
| `data/cache/*.parquet` (25) + `tasks.json` + `build_cache.py` | 统一缓存（task/series_id/freq/start/target） |
| `scan/*.npz` (25) + `scan.py` + `scan_gpu0/1.log` | 全部量化预测（5L×9分位×≤20窗×≤64序列） |
| `scan_summary.csv` + `recompute.py` | 125 行指标（CRPS/MASE/MSE/MAE/nMSE） |
| `stats.csv` + `stats.py` | 两轴统计量（train 段） |
| `analysis.csv` + `correlations.csv` + `analyze.py` | 合并表 + 全部 Spearman |
| `fig_two_axes.png` | 两轴散点 + 主导轴分布 |
