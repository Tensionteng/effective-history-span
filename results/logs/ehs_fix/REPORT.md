# EHS must-fix 实验包（E3/E1/E4/E2-lite）

日期：2026-09-10~11。对应审稿意见 `ccfa-review-reports/ehs-review.md` 的 must-fix 清单（E3=Q4，E1=Q2/C4，E4=方法章预算实验，E2-lite=horizon 稳健性）。脚本：`analysis/e3_val_select.py`、`analysis/e3_val_select_chronos.py`、`analysis/e3_pacf_aic.py`、`analysis/e3_verdict.py`（E3 新增，其余复用既有）。

## E3：validation 选 L vs 统计量预测 span vs PACF/AIC

### E3.1 trained 臂（复用 logs/ehs_v2 日志，零训练）

协议：解析每条日志的逐 epoch "Vali Loss"（early-stopping 选择依据）取最小值作为该臂的验证误差；对每 (dataset, model) 取 **验证误差最小的 sl**（mean over seeds；另有 per-seed 各自选再平均的变体），与"测试集 oracle 最优 sl"对比测试 MSE。28 个 (dataset×model) 行全量解析成功（含缺 seed 的臂按可用 seed 平均，与 SUMMARY 口径一致）。

结果（完整表：`logs/ehs_fix/e3_val_select_trained.md` / `.json`）：

- **validation 选择离 oracle 很近但不完美**：mean-based 精确命中 16/28，平均 regret +0.0036 MSE（平均相对 +1.24%，中位 0.00%）；per-seed 选择平均 regret +0.0037，几乎相同。
- 失败模式集中在两类：(a) 长窗臂过拟合在验证期尚未暴露（traffic/iTransformer：val 选 720，oracle 2880，+5.1%；weather/TimesNet +8.6%）；(b) ETT 上验证集对相邻档位分辨力弱（ETTh1/iTransformer val 选 96 vs oracle 720，+2.0%）。
- 注意（诚实限定）：TSLib 的 val 窗口数随 sl 变化，各 sl 的验证 MSE 在略微不同的目标集上计算——这正是"标准 HPO 实践"本身的样子，未做额外校正。

### E3.2 Chronos 零样本臂（taskC 缓存切分，零推理）

协议：Chronos 零样本无训练、无 canonical val 用法；将 `taskC_cache_<ds>.npz` 的逐窗口误差**按时间序前 50% 作选择集、后 50% 作评估集**（模拟 val→test 时序关系，零重推理成本）。electricity 为既有 stride=2 子采样缓存（内部公平）。候选 L∈{96,336,720,1440}。

| dataset | val-选 sl | oracle sl | 命中 | mse(val-选) | mse(规则 def1440+veto) | mse(oracle) | regret val | regret 规则 |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 1440 | 720 | n | 0.4205 | 0.4205 (@1440) | 0.4192 | +0.0013 | +0.0013 |
| exchange_rate | 96 | 96 | Y | 0.0949 | 0.0949 (@96) | 0.0949 | +0.0000 | +0.0000 |
| weather | 1440 | 1440 | Y | 0.1474 | 0.1474 (@1440) | 0.1474 | +0.0000 | +0.0000 |
| electricity | 1440 | 720 | n | 0.1203 | 0.1203 (@1440) | 0.1202 | +0.0002 | +0.0002 |

逐通道变体（次级）：val 逐通道选择 mse {ETTh1 0.4399, exchange 0.1039, weather 0.1488, electricity 0.1202}，逐通道 oracle 上限 {0.4172, 0.0890, 0.1474, 0.1181}——逐通道选择在 2/4 数据集上反而差于数据集级（选择噪声），支持任务 C"增益在数据集级"的结论。

**关键对照：统计量规则（def1440+veto）在 4/4 数据集上与 validation 选择选到完全相同的 sl**（各档误差面平坦，1440 与 720 差距 <0.4%），即零样本侧统计量规则 ≡ validation 选择，成本为零。

### E3.3 PACF/AIC 经典基线（statsmodels，8 数据集）

协议：训练段原始序列（另有一阶差分变体），每数据集 ≤32 通道（seed=0 抽样，口径同任务 B），ACF(fft, maxlag=2880) → Levinson-Durbin 同时得 PACF 与 AR(p) 创新方差；PACF 建议阶 = 95% 带外最大显著滞后；AIC 建议阶 = argmin N·log σ²(p)+2p（同 R `ar(aic=TRUE)` 的 Yule-Walker 构造）。数据集级取通道中位数，映射到候选档 {96,336,720,1440,2880}（log2 最近；脚本 `analysis/e3_pacf_aic.py`，输出 `e3_pacf_aic.json`）。

| dataset | PACF 阶(中位) level/diff | →档 | AIC 阶(中位) level/diff | →档 | 实测 best_sl (iT/DLinear) |
|---|---|---|---|---|---|
| ETTh1 | 2796 / 2812 | 2880 | 222 / 221 | 336 | 720 / 1440 |
| ETTh2 | 2766 / 2765 | 2880 | 266 / 265 | 336 | 96 / 1440 |
| ETTm1 | 2874 / 2873 | 2880 | 598 / 675 | 720 | 336 / 336 |
| ETTm2 | 2876 / 2875 | 2880 | 677 / 676 | 720 | 336 / 1440 |
| exchange_rate | 2281 / 2445 | 2880 | 10 / 16 | 96 | 96 / 96 |
| weather | 2872 / 2871 | 2880 | 169 / 297 | 96 / 336 | 336 / 2880 |
| electricity | 2857 / 2857 | 2880 | 516 / 510 | 720 | 2880 / 2880 |
| traffic | 2854 / 2856 | 2880 | 506 / 505 | 720 | 2880 / 2880 |

- **PACF"最后显著滞后"在这批数据上退化**：8/8 数据集（level 与 diff 皆然）显著滞后延伸到 2880 上限 → 永远建议 2880。命中 5/16 全靠"碰巧 oracle=2880"的格，平均相对 regret **+11.8%**。周期性强序列的 PACF 长尾是已知现象，该经典准则对此任务基本无分辨力。
- **AIC 阶有结构但系统性偏短**：exchange→96 ✓、ETTm→720、ETTh→336、electricity/traffic→720。它度量的是"短程线性可预测性"，无法表达非线性模型从长周期上下文（168h/周周期）与分布外稳健性中获得的收益 → 命中 0/16，平均相对 regret **+3.5%**。

### E3 判决（Q4：budget 规则 vs validation 选择固定窗）

16 格（8 数据集 × {iTransformer, DLinear}）上四种选择器的测试 MSE regret（完整表 `logs/ehs_fix/e3_verdict.md` / `.json`）：

| 选择器 | 精确命中 | 平均 regret | 平均相对 | 中位相对 | 成本 |
|---|---|---|---|---|---|
| **统计量预测 span（任务B k=3 LOO）** | **8/16** | **+0.0022** | +1.16% | **+0.25%** | ≈0（3 个廉价统计量） |
| validation 选择（标准 HPO） | 6/16 | +0.0032 | +1.04% | +0.39% | 5 档全量训练×seed |
| AIC 阶 → 档 | 0/16 | +0.0081 | +3.52% | +3.00% | 低 |
| PACF 显著滞后 → 档 | 5/16（退化，恒选 2880） | +0.0327 | +11.79% | +10.28% | 低 |

**判决：统计量预测 span 不劣于（且略优于）validation 选择，显著优于 PACF/AIC 经典基线。** 具体：stat 与 val 的平均相对 regret 打平（1.16% vs 1.04%，差异远小于 seed 噪声量级），stat 精确命中更多（8 vs 6）、中位 regret 更低；而 stat 的成本是 3 个统计量 vs 5×3 次完整训练。Chronos 零样本侧统计量规则与 val 选择逐格相同。PACF 退化、AIC 系统性偏短，两者均不能替代也不优于统计量回归器。审稿 Q4"budget 规则 vs validation 选择固定窗"：不塌。

## E1：Chronos-Bolt 宽数据零样本补全（electricity/traffic × L∈{96..2880}）

协议：与任务 C 逐点一致——测试集按 max(L)=2880 构建（TSL border 逻辑使测试目标集与 lookback 无关，已由任务 C 对 SUMMARY 的逐点吻合验证），逐通道批量化推理（`tsfm/e1_wide_zero_shot.py`，复用 `tsfm/ehs_adaptive.py` 的 `batched_predict`，按 L 动态限 chunk）；**electricity 与 traffic 均用 test-stride=2**（electricity N=2583 窗口、C=321；traffic N=1707 窗口、C=862；绝对值与全量略有出入，各 L 间内部公平）。electricity 的 L≤1440 直接复用 `taskC_cache_electricity.npz`（同协议、确定性零样本，值与任务 C 完全一致），只补跑 L=2880；traffic 五档全新跑（窗口分 6 片跨 GPU 并行，`analysis/e1_merge.py` 合并）。

**重要披露（context 上限）**：chronos-bolt-base 的 `context_length=2048`；`ChronosBoltPipeline.predict` 对超长输入无条件左截断（代码路径确认，且数值验证：同一批序列 L=2880 与 L=2048 输入的输出逐位相同，max|Δ|=0.0）。因此 **L=2880 臂实际度量的是有效上下文 2048**。

零样本测试 MSE（MAE）：

| dataset | L=96 | L=336 | L=720 | L=1440 | L=2880(→2048) | best L |
|---|---|---|---|---|---|---|
| electricity | 0.2053 (0.2601) | 0.1253 (0.2140) | 0.1211 (0.2100) | **0.1206** (0.2089) | 0.1208 (0.2092)* | 1440 |
| traffic | 1.2247 (0.3468) | 0.3952 (0.2481) | 0.3759 (0.2413) | 0.3717 (0.2402) | **0.3715** (0.2400) | 2880 |

（*electricity L=2880 与 L=1440 差 <0.2%：截断后有效 token 相同，差异仅来自更宽输入窗口下实例归一化/分片的数值路径。）

**判决：长 EHS 在零样本下于两个宽数据集上复现。** traffic 零样本 MSE 随 L 单调降至最长档（96→2880：1.2247→0.3715，−70%），best L=2880；electricity 96→1440 大幅下降（0.2053→0.1206，−41%）后进入平台（1440 与 2880 差 <0.2%），best L=1440。两者 best sl 均 ≥1440，与训练模型的结论（iTransformer/DLinear 两数据集 best_sl=2880，饱和于 1440–2880）方向一致。结合既有 6 数据集（ETT×4/exchange/weather，L≤1440），零样本 8/8 数据集覆盖了训练侧观察到的全部三种 lookback 类型（短优 exchange、居中 ETT/weather、长优 electricity/traffic）。绝对精度参照：Chronos traffic@2880 0.3715 vs 训练 iTransformer 0.3406；electricity@1440 0.1206 vs 训练 iTransformer 0.1307（Chronos 更低——见下重叠披露，electricity 可能不是严格零样本）。

**预训练语料重叠披露（公开信息检索结论，2026-09-11）**：
- 原始 Chronos（T5 系列）语料公开：论文 §5.2 + 附录 Table 3（[arXiv:2403.07815v3](https://arxiv.org/html/2403.07815v3)）列出 28 个训练数据集；其中 **Electricity (Hourly)（321 序列 × 26304 小时，与 TSLib electricity 同源同尺寸）在语料内**（Benchmark I，in-domain）；ETT (Hourly/15min)、Exchange Rate、Traffic、Weather 均不在（Benchmark II，零样本评测集）。
- Chronos-Bolt 语料**未逐数据集公开**：官方模型卡/AWS 博客仅称"nearly 100 billion time series observations"（[AWS ML 博客, 2024-12-02](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)）；同一博客声称 Bolt 在 27 个 Benchmark II 数据集上"no prior exposure during training"——**traffic、ETT、exchange、weather 据此可确认为 Bolt 的严格零样本**；**electricity 不在该 27 集中，无法排除其在 Bolt 语料内（且在原始 Chronos 语料内的强先验）**。
- 对论文的含义：electricity 的零样本数字须带"疑似 in-domain"限定（traffic 的长 EHS 零样本复现是干净的；electricity 若被污染，方向上是让零样本数字偏优而非伪造"长窗有益"的形状——形状由 traffic 独立支持）。

## E4：输入端预算臂（PatchTSTGated，sl=2880 + --input_budget）

实现：`run.py` 新增 `--input_budget`；`data_provider/data_factory.py` 新增 `InputBudgetWrapper`——在数据进入模型前把每个输入窗口中早于最近 b 步的部分置零（数据集缩放单位下；train/val/test 三分裂同口径，wrapper 复制后改写、不污染底层数据；单测验证通过）。臂配置：`--attn_variant none`（= 原版 PatchTST，SDPA 数值等价已经机制实验 smoke 验证），sl=2880，无窗口上限（与 exp3 对照臂同协议），seed {2021,2022}；超参为官方 PatchTST 配置（exchange/electricity 与 exp3 完全相同；ETTh1 el1/nh2、weather el2/nh4）。b 取数据集级预算：exchange=96、ETTh1=720、electricity=1440、weather=336。对照：同配置 sl=2880 无预算臂 + 原生 sl=b 臂（exchange/electricity 引用 exp3 既有 none 臂，ETTh1/weather 为本次新跑同配置臂）。16 条新臂全部完成（`logs/ehs_fix/e4/*.log`，汇总 `analysis/e4_verdict.py` → `e4_verdict.json/.md`）。

测试 MSE（mean±std，seed 2021/2022）：

| dataset | b | **预算臂 sl=2880+b** | 无预算 sl=2880 | 原生 sl=b | recmask sl=2880（参考） |
|---|---|---|---|---|---|
| exchange_rate | 96 | **0.1013±0.0126** | 0.3989±0.0237 | 0.0950±0.0013 | 0.2625±0.0600 |
| ETTh1 | 720 | **0.4269±0.0030** | 0.6228±0.0796 | 0.4204±0.0068 | - |
| weather | 336 | **0.1602±0.0083** | 0.1817±0.0160 | 0.1528±0.0002 | - |
| electricity | 1440 | 0.1283±0.0004 | 0.1301±0.0013 | 0.1291±0.0004 | 0.1269±0.0000 |

**判决：输入端预算把 sl=2880 的退化恢复到接近原生最优臂——成立（3 个存在退化的数据集上 gap 回收 74–98%）。**
- exchange：无预算长窗退化 4.2×（0.0950→0.3989）；预算臂回收 **97.9%** 的 gap（0.1013 vs 原生 0.0950），且远优于 recmask（0.2625）——硬预算（置零）比注意力端近端掩码更有效。
- ETTh1：回收 **96.8%**（0.4269 vs 原生 0.4204；无预算 0.6228）。
- weather：回收 **74.4%**（0.1602 vs 原生 0.1528；无预算 0.1817）。
- electricity：长窗本无退化（数据集级 b=1440 是"长 EHS"侧），预算臂 0.1283 与无预算 0.1301、原生 0.1291 在噪声内打平——预算在不需要时无害。
- 结论：数据集级预测预算 b（可由 E3 的廉价统计量给出）作为输入端操作，足以消除"长窗有害"数据集的退化，且不需要任何注意力结构改动；与 recmask 的对照（exchange 0.101 vs 0.263）说明"让模型看不到旧数据"优于"让注意力只看近端"。

## E2-lite：horizon 稳健性（iTransformer，pred_len=192）

协议：exchange/electricity/ETTh1 × L∈{96,336,1440} × seed {2021,2022}，全部启用 `--max_train_windows = N_min(sl=1440, pred=192)`（exchange 3680、electricity 16781、ETTh1 7009——由实际数据集对象验证；各臂日志 "train window subsample" 行确认一致生效），equal-data-budget 口径与 H=96 主矩阵相同。18 臂全部完成（`logs/ehs_fix/e2lite/*.log`，汇总 `analysis/e2lite_verdict.py` → `e2lite_verdict.json/.md`）。

测试 MSE（mean±std）与排序对照：

| dataset | L=96 | L=336 | L=1440 | best L (H=192) | H=96 同档网格最优 | 排序一致 |
|---|---|---|---|---|---|---|
| exchange_rate | **0.1784±0.0003** | 0.2181±0.0077 | 0.4015±0.0097 | 96 | 96 | Y（96<336<1440 完全一致，退化更陡） |
| electricity | 0.1646±0.0006 | 0.1544±0.0003 | **0.1483±0.0005** | 1440 | 1440 | Y（1440<336<96 完全一致；H=96 全网格最优 2880=0.1307 与 1440=0.1312 本为近平局） |
| ETTh1 | 0.4488±0.0010 | **0.4402±0.0004** | 0.4483±0.0129 | 336 | 96 | n（但全局差 <2.1%：H=192 三档 0.4402–0.4488，H=96 全网格 0.3915–0.4527 中 96/720/1440 亦在 ~2.5% 带内） |

**判决：最优 L 的数据集级格局对 horizon 稳健——成立（按极性而非逐档名次）。** 两个极性数据集精确复现：exchange 短优（且 H=192 下长窗退化更陡，96→1440 退化 2.25× vs H=96 的 1.81×）、electricity 长优（96→1440 −9.9%）。ETTh1 在两个 horizon 下都是"平/居中"数据集（档间差 ≤2.5%），带内名次在 96/336/720 间摆动属噪声级重排，不改变"ETTh1 无极性强偏好"的定性结论。H=96 的三分类（短优/平/长优）在 H=192 完整保持。

## E7：BOCPD 证伪加固（多重校正 / 去季节变体 / 非 BOCPD 漂移量）

对应评审"证伪是否 BOCPD 操作化特有"的质疑，三条交叉验证。脚本 `analysis/e7_bocpd_robust.py`（复用 `analysis/bocpd_ehs.py` 的 BOCPD 与数据管线），产物 `logs/ehs_fix/e7/`。

### E7.1 hazard 扫描的多重比较校正

协议：对 `logs/mechanism/exp4/bocpd_summary.md` 的 5 hazard × 2 模型 Spearman 表（10 检验；表中 E[r] 整数化后各列秩与未舍入值一致，ρ=-0.667 已逐位复现）用**精确置换 p**（n=8，8! 全枚举、并列带权重）替代原表的 t 近似 p，再做 Bonferroni/Holm（全族 m=10 与每模型侧 m=5）。全表 `e7_1_multcorr.{md,json}`。

头条格（iTransformer, H=1/50）：ρ=-0.667，**精确置换 p=0.0798**（原表 t 近似 0.028 在 n=8 下系统性偏低——同批其余 9 格同样如此）；Holm m=10 → **0.798**，Bonferroni m=10 → 0.798；即便按每模型侧 m=5，Holm → **0.399**。

**判决：校正后不显著——且在未校正的精确检验下本就不显著。** 论文 §4.1 的证伪不能引用该格 p 值；它成立的形式是模式证据：(i) 负相关集中在 electricity/traffic 反例上；(ii) iTransformer 侧 H≥1/250 的 4/5 hazard 方向一致为负；(iii) hazard→1/1000 时 ρ 扫过零翻正（+0.111）——"理论锚"的预测本身随 hazard 选择变号，进一步说明该 BOCPD 操作化钉不住 EHS。建议论文把该格的 p≈0.03 表述降级为方向性描述（附精确 p 与本校正表）。

### E7.2 去季节单轴变体

协议：同 exp4 训练段/通道子采样/标准化管线，先做两种去季节（日历周期：ETTh 24、ETTm 96、exchange 7、weather 144、electricity/traffic 24）：(a) phase=减按相位分组的均值；(b) diff=同 lag 季节差分。重跑全部 5 hazard 的 BOCPD E[r]（16 个并行 worker，`deseas-worker`/`deseas-merge`；产物 `e7_2_deseas.{md,json}`，worker 日志 `e7/workers/`）。

electricity/traffic 的 E[r] 秩（1=最短，共 8 数据集）vs 实测最优 lookback（两模型均并列最长 2880）：

| 变体 | electricity E[r] 秩（5 hazard） | traffic E[r] 秩（5 hazard） | iTr ρ（5 hazard） |
|---|---|---|---|
| 原始 | [2,3,4,5,7] | [1,1,2,3,4] | -0.667 ~ +0.111 |
| phase 去季节 | [4,4,4,4,4] | [1,1,1,1,1] | -0.593 ~ -0.544（全负） |
| diff 去季节 | [3,3,4,4,4] | [4,4,3,3,3] | -0.469（全部相同） |

**判决：方向不反转，无部分撤回信号；证伪对季节混杂更牢。** (i) 周期成分消失后 traffic 的 E[r] 在相位去季节下仍恒为全样本**最短**（rank 1 @ 全部 5 hazard），而其实测最优 lookback 仍并列最长 2880——反例在去掉季节混杂后更干净（phase 变体 ρ 全 hazard 为负）；(ii) electricity 的 E[r] 升至中段（rank 4）但 lookback 仍最长，秩相关方向不变；(iii) diff 变体整体 E[r] 变短（过差分引入 MA 结构的已知效应），ρ 减弱但仍为负。如实记录：E[r] 幅度本身对去季节处理敏感（H=1/1000 时 electricity 372→155）——BOCPD E[r] 统计量随操作化选择漂移，这与论文"该锚在操作化层面脆弱"的论点一致而非矛盾。

### E7.3 非 BOCPD 漂移量交叉验证

协议：同 exp4 数据管线（训练段、≤24 linspace 通道、标准化），计算与 changepoint 模型无关的漂移度量：滚动相邻窗**均值差**（w∈{96,336}）、滚动相邻窗**对称化高斯 KL**（窗方差下限 1e-3 标准化单位，mean/median over t；第一版 1e-6 下限被近退化窗主导，已修正重跑）、论文既有 20 段均值方差 drift index（参照，非独立）。与实测最优 lookback 做 Spearman（精确置换 p + Holm 校正；产物 `e7_3_drift.{md,json}`）。

| 度量 | iTransformer ρ (p_exact) | DLinear ρ (p_exact) |
|---|---|---|
| meandiff w96 | -0.791 (0.025) | -0.325 (0.436) |
| meandiff w336 | -0.865 (0.008) | -0.526 (0.191) |
| KL w96 median | -0.840 (0.013) | -0.601 (0.129) |
| KL w336 median | -0.902 (0.005) | -0.501 (0.218) |
| drift index 20seg（参照） | -0.803 (0.021) | -0.801 (0.023) |

Holm 校正（每模型侧 m=7）：iTr 侧 KL-w336-med adj 0.033 显著、meandiff-w336 adj 0.050 临界、其余不显著；全族 m=14 最佳 adj 0.067。DLinear 侧校正后均不显著，但 7 度量中 6 个方向为负。

**判决："漂移越狠→最优跨度越短"独立于 BOCPD 框架成立（iTransformer 侧强、DLinear 侧方向一致）。** 两个完全独立于 changepoint 模型的度量族（均值差、KL）在 iTr 侧全部负相关（ρ∈[-0.90,-0.79]），最强格 Holm 校正后仍显著。即：被证伪的只是"后验 run length"这一特定操作化；漂移-跨度负相关本身在非 BOCPD 度量下稳健——论文"漂移轴有效、BOCPD 锚无效"的两轴叙事不受影响且更精确。

## E8：触发器合成演示（判决：否决，建议从 fig4/§6 撤下）

对应评审"fig4/§6 的 accumulated-evidence changepoint trigger 支路：演示或删除"。脚本 `analysis/e8_trigger_demo.py`，数据 `tsfm/shock_data.py` ShockTS（与附录 G 探针同生成器：长 512、0–3 个冲击、4 类型、含 ground-truth 位置与无冲击反事实）；val seed 777（n=1024，只用于调阈值），test seed 999（n=2048，与附录 G 同 seed 同规模）。

### 设置

- **预测器**：在线 AR(K=48)+截距 的指数加权最小二乘（EWRLS，折扣充分统计量 + ridge 1.0），1-step 与 direct 24-step 两个 horizon；这正是论文 §6 输入端 decay 的受控对应物——"预算/decay"=估计窗口的指数遗忘，"重置"=触发点处把充分统计量清零（硬）或乘 0.1（软）。
- **基座 decay**：μ_law=exp(−1/N*)，N*=65.63·λ^(−1/3)=459（λ=1.5/512，该生成器的冲击率，常数取自 logs/power_law 校准）——即论文 auto 规则在此生成器上的取值；对照 μ=0.99（更强遗忘）与 μ=1（不遗忘）。
- **触发器（累积证据）**：对基座预测器的 1-step 标准化创新（因果 EMA-RMS 归一）跑三路 CUSUM——快符号路（allowance 0.5）、慢符号路（0.15/0.25，针对 trend 级持续偏置）、方差路（ê²，allowance 1.0）——任一路超阈触发，16 步冷却。阈值在 val 上按 F1 网格搜索后冻结。另设 pointwise-surprise 基线（|ê|>τ）与 fast-only 消融。
- **命中口径**：触发时刻落在冲击 [−16,+48] 窗口记命中（附录 G 用 ±1 patch=16；累积证据允许延迟，放宽到 +48）；报告 precision/recall/F1、平均延迟、分类型 recall、无冲击序列的误报率。协议披露：两 pass——触发器从基座臂（A）的创新流计算，D 臂在这些触发点重置（非 D 自身创新的闭环）。

### 检测指标（test，阈值 hf=5、慢路/方差路被 val F1 关闭）

P=0.528，R=0.399，F1=0.454；平均延迟 13.6 步（中位 9.5）；误报 0.48 次/无冲击序列。分类型 recall：level 0.50、vol 0.76、trend 0.08、freq 0.31。对照 pointwise 基线（F1=0.400）：level 0.35、vol 0.83、trend 0.08、freq 0.11。**累积证据相对点态惊奇的真实增益：freq recall ×2.9（0.11→0.31）、level +0.15；trend 无任何增益（0.08 vs 0.08）**。慢符号路能把 trend recall 推到 0.40–0.73，但 precision 同步崩到 ≤0.20（val F1 拒绝）——附录 G"点态惊奇看不到 trend/freq，累积证据可以"的动机句只对 freq 成立，对 trend 在任何可用工作点不成立。

### 预测演示（test MSE；overall=全部目标点，post=冲击后 48 步内，calm=无冲击序列，stable=有冲击序列中距冲击 ≥96 步）

1-step（24-step 同号，全表 `e8_trigger_demo_v2.{md,json}`）：

| 臂 | overall | post-shock | calm | stable |
|---|---|---|---|---|
| A const μ_law | 0.4298 | 0.6345 | 0.3004 | 0.3630 |
| C const μ=1 | **0.4206** | **0.6181** | 0.2999 | 0.3581 |
| B const μ=0.99 | 0.4772 | 0.7159 | 0.3152 | 0.3922 |
| D trigger 硬重置 | 0.5456 (+26.9%) | 0.7524 (+18.6%) | 0.3235 (+7.7%) | 0.4535 |
| E oracle 硬重置 | 0.6647 (+54.7%) | 1.1582 (+82.5%) | 0.3004 | 0.3980 |
| F oracle 软重置 ×0.1 | 0.5489 (+27.7%) | 0.9053 (+42.7%) | 0.3004 | 0.3877 |
| G trigger 软重置 ×0.1 | 0.5004 (+16.4%) | 0.7129 (+12.3%) | 0.3079 | 0.4108 |

（括号 = 相对 A。v1 无截距变体数字几乎相同：D +27.4%、E +56.7%，见 `e8_trigger_demo.json`。）

### 判决：否决（veto）

**触发器在该受控环境的所有配置下都是负增益，且负增益不依赖触发质量**：连 oracle（真实冲击时刻）硬重置都把 post-shock MSE 推高 83%（overall +55%），软重置 ×0.1 仍 +28%（overall）。机理清楚：ShockTS 的冲击是分量级的（level/vol/trend/freq 只改一个分量，谐波结构与 AR 系数持续），冲击前的数据对未变分量仍有高度预测价值——丢弃/降权它的代价（硬重置另有 ~K 步冷启动）超过消除陈旧偏置的收益；stable 区 E 仍差于 A（0.398 vs 0.363）说明损失来自遗忘持续结构，而非瞬态失配。常数臂排序（μ=1 最优、μ=0.99 最差）也与论文 §6 自己的披露一致（真实数据上 N* 饱和到下限，auto≈universal safe decay）。

对论文的动作建议：(i) 从 fig4 与 §6 撤下 accumulated-evidence trigger 支路（§6 末句"An optional accumulated-evidence trigger ... resets the budget"删除）；(ii) 附录 G 的动机句"pointwise surprise cannot see trend- and frequency-level changes → accumulated-evidence trigger"需限定：本演示中累积证据只把 freq recall 提升 ~3 倍，trend 在可用精度下仍不可见（≤0.08），且即使可检测，重置预算也无预测增益。局限披露：EWRLS 在线预测器≠论文的训练式神经预测器（但它是 reset-adaptive filtering 的标准试验台，也是"输入端预算重置"最直接的受控对应物）；两 pass 触发协议；生成器冲击为分量级——全参数重抽的 regime switch 可能更有利于重置，但那已不是本文附录探针所用的冲击模型。

## E6：ehs_v2 主矩阵覆盖补齐（PatchTST/TimesNet × electricity/traffic 长窗臂）

对应缺口：`logs/ehs_v2/SUMMARY.md` 中 PatchTST/TimesNet 在两个宽数据集上的 sl∈{720,1440,2880} 臂此前全部因 OOM/超时缺失（原 bs16/32 在多作业共享 GPU 下 OOM，失败日志中可见他进程占 54–70GB），"PatchTST/TimesNet 在 electricity/traffic 停在短窗"的 best-sl 行实际建立在 sl≤336（traffic PatchTST 甚至只有 sl=96 一个 seed）上。本批在独占 GPU（4-7，单卡单任务，PYTORCH_CUDA_ALLOC_CONF=expandable_segments）下补齐 42 臂（36 条长窗臂 + traffic PatchTST sl336 整行缺失的 3 seed + PatchTST/TimesNet sl336 缺 seed 补齐 5 条）。

协议：超参与主矩阵同数据集口径一致（`scripts/long_term_forecast/ehs_v2/fill_cmds.txt`），`--max_train_windows` 同臂（electricity 15437 / traffic 9305）；bs 降级防 OOM：electricity PatchTST 16→8、traffic PatchTST 4（sl2880 臂 4→2，单卡 80GB 下 bs4 仍 OOM：PatchTST traffic sl2880 的注意力得分矩阵 [B·862, 8, 360, 360] 单张约 14GB，bs2 峰值 65GB）、TimesNet electricity 32→8、TimesNet traffic 32→8；lr 不变。OOM→减半重试由调度器自动执行并记录（`logs/ehs_v2/fill_scheduler*.log`；另有两条臂因调度进程被杀后由孤儿进程续跑完成，一条（electricity PatchTST sl2880 s2023）在清重误杀后手动 bs4 重跑）。**披露**：新旧臂 batch size 不同（原 96/336 臂 bs16/32，新长窗臂 bs8/4/2），数值在 bs 层面不完全对齐；但同一 (model,dataset) 行内长窗臂间口径一致，best-sl 判决基于行内比较。

测试 MSE（mean±std，seed {2021,2022,2023}，[n=k] 标记不足 3 seed；汇总 `logs/ehs_fix/e6_fill_verdict.{json,md}`，主矩阵 `SUMMARY.md` 已同步重生成）：

| row | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl（补齐前 → 后） |
|---|---|---|---|---|---|---|
| electricity/PatchTST | 0.1816±0.0001[n=2] | 0.1371±0.0007 | 0.1352±0.0007 | **0.1290±0.0007** | 0.1397±0.0102[n=2] | 336 → **1440** |
| electricity/TimesNet | **0.1687±0.0000[n=1]** | 0.1821±0.0027 | 0.1848±0.0037 | 0.1974±0.0065 | 0.2178±0.0037 | 96 → 96 |
| traffic/PatchTST | 0.4683±0.0000[n=1] | 0.3809±0.0008 | 0.3664±0.0007 | **0.3628±0.0004** | **0.3625±0.0024** | 96 → **1440≈2880**（平台，差 0.1%） |
| traffic/TimesNet | **0.5845±0.0000[n=1]** | 0.6024±0.0066 | 0.6172±0.0042 | 0.6167±0.0033 | 0.6586±0.0060 | 96 → 96 |

MAE 同向：electricity/PatchTST best 1440（0.2255），traffic/PatchTST best 720（0.2591，与 1440 的 0.2601 在噪声内），两个 TimesNet 行 best 仍 96。

**判决：TimesNet"长窗全差"成立；PatchTST 的"短窗最优"是覆盖缺口假象，真实行为是数据集依赖——"模型族门控"表述需修订。**

- **TimesNet 结构性失败（不变）**：两个宽数据集上 MSE 随 lookback 单调恶化（electricity 96→2880：0.1687→0.2178，+29%；traffic：0.5845→0.6586，+13%），与 seed 无关。period-fold 卷积对 321/862 通道的强制混合在长窗下不可用，门控成立。
- **PatchTST 长窗可用（反转）**：electricity 336→1440 改善 0.1371→0.1290（−5.9%，96→1440 −29%）；traffic 96→1440 改善 0.4683→0.3628（−22.5%），2880 与 1440 打平（n=3 后 0.3625 vs 0.3628，差 0.1%，属平台）。方向与 iTransformer/DLinear 在这两个数据集上的长 EHS（best=2880，饱和于 1440–2880）一致；exp3 无窗口上限臂佐证同型行为（electricity PatchTST sl2880 uncapped 0.1301±0.0013 ≈ 原生 sl=1440 的 0.1291，无退化）。PatchTST 在漂移数据上仍失效（exchange 96→2880 退化 4.65×：0.0871→0.4053）——失效由数据集属性（漂移 vs 周期）而非模型族决定。
- **修订后的"模型族门控"**：TimesNet（period-fold 卷积强制混合）长窗结构性失败；PatchTST（channel-independent 注意力）与 iTransformer/DLinear 同型——在漂移数据（exchange）上变平失效，在周期数据（electricity/traffic）上长窗可用（best=1440，2880 平台）。`tab:bestl` 修订：electricity PatchTST 336→1440、traffic PatchTST 96→1440（2880 并列平台）；TimesNet 两行维持 96。

## 产物清单（logs/ehs_fix/）

- E3：`e3_val_select_trained.{json,md}`、`e3_val_select_chronos.{json,md}`、`e3_pacf_aic.json`、`e3_verdict.{json,md}`；脚本 `analysis/e3_{val_select,val_select_chronos,pacf_aic,verdict}.py`
- E1：`e1_traffic_shard{0..5}of6.npz`、`e1_electricity_shard{0,1}of2.npz`、`e1_merged.json`、运行日志 `e1_run_*.log`；脚本 `tsfm/e1_wide_zero_shot.py`、`analysis/e1_merge.py`（`e1_traffic_shard0of100.npz` 为早期冒烟测试残留，未计入合并）
- E4：`e4/*.log`（16 臂）、`e4_scheduler.log`、`e4_verdict.{json,md}`；脚本 `scripts/long_term_forecast/ehs_fix/run_e4.py`、`analysis/e4_verdict.py`；代码改动 `run.py`（--input_budget）、`data_provider/data_factory.py`（InputBudgetWrapper）
- E2-lite：`e2lite/*.log`（18 臂）、`e2lite_scheduler.log`、`e2lite_verdict.{json,md}`；脚本 `scripts/long_term_forecast/ehs_fix/run_e2lite.py`、`analysis/e2lite_verdict.py`
- E7：`e7/e7_1_multcorr.{json,md}`、`e7/e7_2_deseas.{json,md}`、`e7/e7_3_drift.{json,md}`、`e7/deseas_{phase,diff}_*.json`（16 worker 分片）、`e7/workers/*.log`；脚本 `analysis/e7_bocpd_robust.py`（复用 `analysis/bocpd_ehs.py`）
- E8：`e8/e8_trigger_demo.json`（v1：双路 CUSUM、无截距 EWRLS）、`e8/e8_trigger_demo_v2.{json,md}`（v2 主结果：三路 CUSUM、截距增广、软重置臂）、`e8/run*.log`；脚本 `analysis/e8_trigger_demo.py`（复用 `tsfm/shock_data.py`）
- E6：`e6_fill_verdict.{json,md}`（判决表）、补齐臂日志 `logs/ehs_v2/{electricity,traffic}_{PatchTST,TimesNet}_sl{336..2880}_s{2021..2023}.log`、调度日志 `logs/ehs_v2/fill_scheduler*.log`、`SUMMARY.md` 已重生成；脚本 `scripts/long_term_forecast/ehs_v2/fill_scheduler.py`（GPU 调度 + OOM 减半重试 + flock 队列）、`fill_cmds.txt`（42 臂）、`analysis/e6_fill_verdict.py`

## E3-ext（2026-09-16 追加）：validation 选择器扩到 6 新数据集（零训练）

对应 review-v4 "最高 ROI 改进机会"。协议同 E3.1，作用在 logs/ehs_v2_ext 的 180 条日志
（6 新数据集 × {iTransformer, DLinear} = 12 格）；统计量侧用 `analysis/ehs_predict_v3.py`
的 n=28 LOO 岭回归器（k=3；对新数据集即严格样本外预测）。脚本
`analysis/e3_val_select_ext.py`，产物 `logs/ehs_v2_ext/e3_ext_verdict.{json,md}`，
详节见 `logs/ehs_v2_ext/REPORT.md` §4。

| 选择器（新 12 格） | 精确命中 | 平均 regret | 平均相对 | 中位相对 |
|---|---|---|---|---|
| validation（标准 HPO） | 8/12 | +0.0027 | +0.80% | +0.00% |
| 统计量预测（n=28 LOO） | 7/12 | +0.0049 | +1.79% | +0.00% |

合并 28 格（旧 14 + exchange_rate 2 + 新 12，stat 侧统一 n=28 LOO）：val 16/28
（+0.94%）、stat 15/28（+1.83%）；新 12 格配对差 stat−val = +0.99pt ± 0.94pt（SEM，
t≈1.05，不显著），中位差 0。

**判决：n=28 口径下"统计量打平 validation"保持（弱化版）**——统计上不可区分、成本
为零的结论成立，但逐点方向由旧口径"stat 略优"翻转为"stat 略逊"；失误集中于居中-oracle
格（ercot-iT +6.3%、pedestrian-iT +8.1%），n=28 回归器受 2880 边界样本（12/28）影响
偏好长档。

**顺带的完整性勘误**：原 `analysis/e3_val_select.py` 文件名正则 `[A-Za-z0-9]+` 不匹配
exchange_rate 的下划线 → 旧 E3 trained 臂实为 14 格（7 数据集 × 2 模型），且
`e3_verdict.py` 汇总时分母硬编码 16。exchange 两格补上后 val/stat 均命中 oracle=96
（regret 0），旧判决方向不变；本次 pooled-28 已按正确口径重算。旧产物未改动。

## E9（2026-09-16 追加）：decay 真实数据重标定（判决：负结果）

对应 review-v4"唯一可能 7.5+ 的实验"。做法：drift→λ 改 log-log 外推（替代端点钳制），
c 由真实网格最优重拟合（c_i=N\*_opt·λ_i^{1/3}，几何平均；weather 最优由新 mini-grid
μ∈{0.996,0.998}×2 seeds 钉定在 N\*≈250–287）。c_real=124.4（c_i 散布 37.8–349.3，
9.2×）。验证：4 数据集 × iTransformer × seeds{2021,2022}（autorc 臂，无下界）。

结果（详节与新臂日志见 `logs/decay/REPORT.md`"真实重标定"节，`logs/decay/recalib.json`）：
μ 逐数据集全不同（0.997079/0.997651/0.998194/0.998478）——辨别力机械恢复；但
"不差于 none" 3/4：exchange 0.3745 vs none 0.3211（**FAIL +16.6%**，N\*=342 落入
instance-norm 中毒区）、ETTh1 0.4276 ≤ 0.4527、weather 0.1776 ≤ 0.2244、electricity
0.1299 ≤ 0.1307。与普适默认（floor, N\*=L/3）对比：四格平均 0.2774 vs 0.2386，floor
仅 weather 一局落败（0.1927 vs 0.1776）。根因：真实 iTransformer 最优遗忘强度与合成
幂律**反号**（低 drift 的 electricity 要 N\*=200，高 drift 的 exchange 要 N\*=960），
单一 c 无解。**判决：普适安全默认仍最优，重标定记为负结果**——论文 §6
"auto≈universal safe decay" 的披露获得直接实验支撑。

产物：`analysis/decay_recalib.py`、`scripts/long_term_forecast/decay/run_recalib.py`、
`logs/decay/{weather_iTransformer_mu0p996,mu0p998}_sl2880_s{2021,2022}.log`、
`logs/decay/*_autorc_*.log`（8 臂）、`logs/decay/recalib.json`；代码改动 `run.py` /
`data_provider/data_factory.py`（新增 --decay_c/--decay_extrapolate/--decay_floor，
默认行为不变，decay_unittest 5/5 PASS）。
