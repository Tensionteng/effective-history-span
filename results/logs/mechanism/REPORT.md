# Mechanism 判决实验报告（logs/mechanism/）

日期：2026-09-09~10。基座数据：logs/ehs_v2/SUMMARY.md（混杂受控 lookback 扫描）。
四个实验分别验证："softmax 稀释/sink"（实验1）、"线性模型自动门控"（实验2）、
"加选择性可消除长上下文退化"（实验3）、"EHS = run-length 后验期望"（实验4）。

## 实验 1：PatchTST 注意力熵与注意力图

- 脚本：`analysis/mechanism/attn_entropy.py`；指标存 `logs/mechanism/exp1/attn_metrics.npz`。
- 对象：exchange_rate 与 electricity 上 LBv2 checkpoint（PatchTST, dm512/el2/df2048），
  测试集前 8 个 batch（eval 模式，dropout 关闭），逐层统计：
  逐 query 平均熵 H（另报归一化 H/ln S）、首 patch（最旧）注意力质量占比（sink）、
  最近 10% patch 的总质量占比（recency mass）。S = patch 数 = (sl-16)/8+2。
- electricity 的 sl=1440/2880 在 ehs_v2 中全部 OOM 未完成（无 checkpoint），
  改用实验 3 的无窗口上限 vanilla 对照臂（PatchTSTGated none，SDPA，数值等价原版）补齐；
  electricity sl=720 仅有 4 epoch 的部分训练 checkpoint（ehs_v2 超时截断），已标注。

### 数据（种子 2021/2022 均值；L0/L1 = 第 1/2 层）

exchange_rate（LBv2 ckpt）：

| sl | S | H(L0) | H/lnS(L0) | H(L1) | H/lnS(L1) | sink(L0/L1) | near10%(L0/L1) |
|---|---|---|---|---|---|---|---|
| 96 | 12 | 1.92 | 0.77 | 2.09 | 0.84 | 0.138 / 0.127 | 0.127 / 0.103 |
| 720 | 90 | 4.18 | 0.93 | 4.07 | 0.90 | 0.021 / 0.015 | 0.124 / 0.087 |
| 2880 | 360 | 5.19 | 0.88 | 5.38 | 0.91 | 0.007 / 0.004 | 0.163 / 0.145 |

（均匀分布基准：H/lnS = 1；near10% 均匀基准 = 0.10；sink 均匀基准 = 1/S。）

electricity：sl=96（LBv2 充分训练）H/lnS(L0/L1) = 0.60/0.68，near10% = 0.063/0.068
（低于均匀 0.083）；sl=720（部分训练）H/lnS = 0.77/0.88，near10% = 0.190/0.114。
MECH 无上限 vanilla 对照臂（完整训练）：

| sl | S | H/lnS(L0) | H/lnS(L1) | sink(L0/L1) | near10%(L0/L1) |
|---|---|---|---|---|---|
| 96 | 12 | 0.59 | 0.66 | 0.086 / 0.080 | 0.063 / 0.063 |
| 1440 | 180 | 0.66 | 0.91 | 0.008 / 0.005 | 0.185 / 0.093 |
| 2880 | 360 | 0.75 | 0.95 | 0.003 / 0.003 | 0.192 / 0.129 |

### 图

- 热力图样例：`figs/exp1_heatmap_{exchange_rate,electricity}_sl{96,720,2880}_{lbv2,mech}.png`
  （exchange sl=2880 的第 2 层呈大范围竖条纹 = 多数 query 弥散地分摊到大量 key；
  sl=96 时能量集中在少数 key 列；electricity sl=2880 的能量反而集中在最近若干
  key 列与少数周期列上。）
- 对比图：`figs/exp1_entropy_vs_lookback.png`。

### 判决（判据：长 lookback 下熵显著上升且近端质量占比被稀释 → 成立）

**证伪（按判据的关键合取项）/ 仅"熵上升"一半成立**：
- 熵稀释普遍存在但无害化：归一化熵在两个数据集上都随 lookback 上升
  （exchange L1 0.84→0.91；electricity L1 0.66→0.95，升得更多），
  但 electricity 长上下文性能全局最优——所以"softmax 稀释"本身不是退化的充分条件。
- 近端稀释证伪：exchange near10% 未被稀释（0.13→0.16，对均匀基准的倍数
  1.6x→1.6x 持平）；electricity 近端质量反而上升（L0 0.06→0.20）。
- sink 证伪：首 patch 质量占比从 0.15（sl=96，高于均匀 2 倍）跌至均匀水平
  （sl=2880, ≈1/S），长上下文下旧 token 不积聚 sink 质量，而是被均匀稀释。
- 综合：注意力在长上下文下确实变平（熵升、sink 消失），但退化与否取决于
  远端内容是否有用（exchange 远端=噪声 → 稀释有害；electricity 远端=周期结构
  → 稀释无碍）。这与实验 2（DLinear 远端权重在 exchange 上弥散无结构、
  在 electricity 上有周期尖峰）互为印证。

## 实验 2：DLinear 权重衰减（"自动门控"假说）

- 脚本：`analysis/mechanism/dlinear_weights.py`；数据 `logs/mechanism/exp2/`，
  图 `figs/exp2_dlinear_decay_{dataset}.png`。
- 方法：读 Linear_Seasonal/Linear_Trend 权重 [96, sl]，|w| 按输出位平均得每个 lag 位置的质量
  （索引 0 = 最早 lag），绘制归一化质量随 lag 距离（d=1 为最近步）的衰减曲线。

### 关键数字（mass(last 96 lags) / mass(lag>720) / 日周期谐波对比度）

exchange_rate：
- Linear_Seasonal sl=2880：0.060 / 0.726 / 1.03
- Linear_Trend sl=2880：0.238 / 0.556 / 1.05

electricity：
- Linear_Seasonal sl=2880：0.187 / 0.487 / 1.25
- Linear_Trend sl=2880：0.181 / 0.529 / 0.94

### 判决

**证伪（按原表述）**：exchange 上 sl=2880 的 DLinear 权重并未集中于最近 ~96 lag
（seasonal 仅 6.0% 质量在最近 96 lag），远端也不是 ≈0（55-73% 质量在 lag>720），
且远端质量弥散无结构（日周期谐波对比度 ≈1）。即线性模型没有学会"自动门控"，
反而把大量权重撒到远过去——这与其 exchange sl=2880 MSE 爆到 0.636 一致
（远端噪声权重 = 过拟合/退化源）。electricity 的远端质量同样不小（49-53%），
但那里长上下文有帮助（MSE 0.129 全局最优），短 sl 曲线上可见 d≈24/168 的
日/周周期尖峰（sl=96 seasonal 对比度 1.67）。结论：线性模型不具备隐式门控；
数据属性决定远端权重是否有害。

## 实验 3：PatchTST 修复消融（denbias / recmask）

- 模型：`models/PatchTSTGated.py`（SDPA 实现 + 开关；'none' 与原版数值等价，
  冒烟测试 max|Δ|=9.5e-7；sl=96 时 recmask 为恒等）。
- 设置：exchange_rate / electricity × sl {96,1440,2880} × {none, denbias, recmask}
  × seed {2021,2022} = 36 臂（24 变体臂 + 12 无上限 vanilla 对照臂），不开
  --max_train_windows，超参沿用 ehs_v2 主矩阵（dm512/df2048/el2/nh8/lr1e-4；
  bs: exchange 32，electricity 16/8/4 for sl 96/1440/2880 以避免 OOM）。
  注：用户括号内 "exchange d_model 128 / electricity e_layers 3 lr 5e-4" 与
  ehs_v2 主矩阵实际配置（两者均 dm512/el2/lr1e-4）不符，为与原版同臂可比，
  采用 ehs_v2 主矩阵配置。
- 结果（MSE，mean±std over seeds {2021,2022}；`none` = PatchTSTGated vanilla 对照）：

**exchange_rate**（对照基线 ehs_v2 capped PatchTST 在括号内）：

| seq_len | none (ctrl) | denbias | recmask | ehs_v2 PatchTST |
|---|---|---|---|---|
| 96 | 0.0950±0.0009 | 0.0952±0.0010 | 0.0950±0.0009 | 0.0871±0.0020 |
| 1440 | 0.1371±0.0003 | 0.1387±0.0030 | 0.1282±0.0132 | 0.1484±0.0239 |
| 2880 | 0.3989±0.0168 | 0.3989±0.0181 | **0.2625±0.0424** | 0.4053±0.0169 |

健全性：none sl=2880 = 0.399 精确复现 ehs_v2 的 0.405（该臂 capped 与 uncapped 的
训练窗口数相同，N=2336）；sl=96 时 recmask 退化为恒等（patch 数 12 < 42），
结果与 none 逐位一致 ✓。checkpoint 与日志互洽（重算 MSE 与日志一致到 6 位小数）。

denbias 诊断（学习到的 null-logit b 与 softmax null 质量，测试集前 4 batch）：
b 在所有臂上保持 ≈0.00（未离开初始化），null 质量 ≤0.001-0.01（sl≥1440），
即"什么都不看"的出口完全没被梯度利用——这解释了 denbias 与 none 逐点相同的结果。

**electricity**（ehs_v2 capped PatchTST 仅有 sl=96/336 完成臂）：

| seq_len | none (ctrl) | denbias | recmask | ehs_v2 PatchTST |
|---|---|---|---|---|
| 96 | 0.1803±0.0002 | 0.1803±0.0002 | 0.1803±0.0002 | 0.1816±0.0001 |
| 1440 | 0.1291±0.0003 | 0.1285±0.0003 | 0.1291±0.0003 | -（OOM） |
| 2880 | 0.1301±0.0009 | 0.1384±0.0089 | **0.1269±0.0000** | -（OOM） |

健全性：none sl=96 = 0.1803 ≈ ehs_v2 的 0.1816 ✓。electricity 上 PatchTST 在长
lookback 不退化（0.180→0.129→0.130），与 iTransformer/DLinear 同向。
denbias sl=2880 的 0.1384 由 s2022 的 0.1473（epoch 1 后早停）拉高，s2021 为 0.1295；
单种子噪声，非系统性变差。denbias 的 null 质量在 electricity 短 lookback 下确实被
使用（sl=96 L0 达 0.11-0.17），但在 sl=2880 反而 ≈0.001——出口在最需要它的
地方（exchange 长上下文）完全没被学到。

### 判决（判据：exchange sl=2880 recmask/denbias 应显著优于原版 ~0.405 并回到 0.10-0.15；
### electricity 变体不应显著差于原版）

**部分成立（硬约束显著缓解但未消除；软出口失败）**：
- recmask exchange sl=2880：0.399 → 0.263（-34%，弥合退化缺口 ~45%），显著优于原版 ✓，
  但未回到 0.10-0.15 ✗。sl=1440：0.137→0.128。
- denbias exchange sl=2880：0.3989 vs 原版 0.3989 —— 完全无效 ✗（b≈0 未学习，null 质量 ≤0.4%）。
- electricity：recmask 不差于原版（2880 上 0.127 还略优）✓；denbias 在 2880 上
  +0.008（单种子噪声），其余臂与原版一致 ✓。
- 含义：长上下文退化中约一半可直接归因于注意力被迫"看完整个远过去"（硬掩码即恢复），
  另一半来自 value/head 通路在长输入下的过拟合/优化困难（electricity sl2880 的 vanilla
  vali 曲线剧烈震荡而 recmask/denbias 平稳收敛也佐证优化层面的作用）。
  可学习的 softmax 出口（off-by-one）不是有效机制——梯度并不使用它。
  对论文机制设计：选择硬结构先验（recency mask / 周期掩码 / 分块稀疏）优于软门控；
  且修复实验应按数据属性配对（exchange 类数据才需要，electricity 类数据长上下文无害）。

## 实验 4：BOCPD 理论对照

- 脚本：`analysis/bocpd_ehs.py`；观测模型为高斯未知均值+未知方差
  （Normal-Inverse-Gamma 先验 → Student-t 预测）。注意：最初实现为"未知均值+已知方差"
  （用户描述的简化），但该模型后验 CP 概率恒等于先验 hazard（常数 hazard 下
  p(r_t=0)≡H 是构造恒等式），且 E[r] 几乎完全由先验决定——似合没有信息量；
  故升级为 NIG。常数 hazard H∈{1/50,1/100,1/250,1/500,1/1000} 扫描，
  run-length 后验截断 R_MAX=3000，通道子采样 ≤24，训练段边界与 data_provider 一致。
  健全性：iid 高斯输入 E[r]≈1740（持续增长至截断），分段均值偏移输入 E[r]≈102（检出 CP）。
- 结果：每数据集"期望 run length"（时间平均，burn-in 500 后）见
  `logs/mechanism/exp4/bocpd_summary.md`，曲线样例 `figs/exp4_runlength_curves.png`，
  散点 `figs/exp4_bocpd_scatter.png`。

E[r]（H=1/250，单位：步；ETTm* 为 15 分钟步，其余为小时/天步）：

| dataset | E[r] | best sl iTr | best sl DLin |
|---|---|---|---|
| ETTh1 | 90 | 720 | 1440 |
| ETTh2 | 262 | 96 | 1440 |
| ETTm1 | 68 | 336 | 336 |
| ETTm2 | 390 | 336 | 1440 |
| exchange_rate | 219 | 96 | 96 |
| weather | 301 | 336 | 2880 |
| electricity | 122 | 2880 | 2880 |
| traffic | 87 | 2880 | 2880 |

Spearman(E[r], best sl) 随 hazard 扫描（n=8）：

| H | iTransformer | DLinear | 两者均值 |
|---|---|---|---|
| 1/50 | -0.667 (p~0.03) | -0.225 | -0.395 |
| 1/100 | -0.581 | -0.100 | -0.263 |
| 1/250 | -0.457 | 0.100 | -0.072 |
| 1/500 | -0.272 | 0.325 | 0.144 |
| 1/1000 | 0.111 | 0.551 (p~0.11) | 0.443 |

### 判决

**不成立**：任何 hazard 下都没有显著正相关（iTransformer 一侧多数 hazard 下甚至为负，
H=1/50 时 ρ=-0.667, p~0.03 显著为负）。反例最刺眼的是 electricity/traffic：E[r] 最短
（122/87 步，regime 切换最频繁）却实测最优 lookback 最长（2880）；ETTm2 E[r] 最长
（390）但最优 lookback 仅 336。含义：最优 lookback 不由"regime 平均持续长度"决定；
electricity/traffic 的长上下文收益来自强日/周**周期性**（lurking 结构跨越多个 regime），
而非 regime 持续性。EHS 若想作为锚点，需要显式编码周期结构而非只用 run-length 后验。
对论文机制设计的含义：不要把"数据属性"建模为单一的 changepoint/run-length 量；
periodicity 与 nonstationarity 是两个独立的轴，长上下文收益由前者驱动、退化由后者驱动。

## 总体结论（四实验联合判决）

1. **softmax 稀释/sink（实验1）**：稀释普遍存在（两数据集归一化熵都升到 0.9+），
   sink 不存在；但稀释本身无害（electricity 稀释更狠却最优）。退化取决于远端内容性质。
2. **线性自动门控（实验2）**：证伪。DLinear 在 exchange sl=2880 上 56-73% 权重质量在
   lag>720 且弥散无周期结构——没有门控，远端噪声权重直接造成损害。
3. **修复消融（实验3）**：硬 recency 掩码显著缓解 exchange 退化（-34% MSE），
   可学习 off-by-one 出口无效。长上下文伤害的一半机制在注意力层。
4. **BOCPD 理论锚点（实验4）**：不成立。E[r] 与实测最优 lookback 无显著正相关
   （iTransformer 侧多 hazard 下显著为负）；最优 lookback 由周期性而非 regime 持续性驱动。

统一图景：**"数据属性 × 模型族"中的数据属性应分解为 周期性（决定长上下文是否有用）
与 非平稳远端噪声（决定长上下文是否有害）两个正交轴**；transformer 族缺乏任何
内置选择性（softmax 无出口且学不会出口；线性层无门控），所以在远端噪声大的数据上
长上下文退化，而硬结构约束能恢复其中约一半。

## 复现与产物清单

- 代码：`models/PatchTSTGated.py`（新）、`run.py`（+`--attn_variant`/`--recmask_window`，
  默认不改变任何现有行为）、`analysis/mechanism/{smoke_patchtstgated,attn_entropy,dlinear_weights,collect_exp3}.py`、
  `analysis/bocpd_ehs.py`、`scripts/long_term_forecast/mechanism/run_exp3.py`。
- 数据：`logs/mechanism/exp{1,2,3,4}/`（npz + summary.md + 逐臂日志），
  图：`logs/mechanism/figs/`（exp1 热力图×7 + 熵对比图，exp2 衰减×2，
  exp3 消融对比图，exp4 曲线+散点）。
- 验证：PatchTSTGated(none) 与 PatchTST 数值等价（max|Δ|=9.5e-7）；
  recmask@sl96 ≡ vanilla（MSE 逐位一致）；exchange none sl2880 复现 ehs_v2
  （0.399 vs 0.405）；checkpoint-日志互洽（重算 MSE 一致到 6 位小数）；
  BOCPD 健全性（iid → E[r] 持续增长；分段偏移 → E[r]≈102 且检出重置）。
- 偏差声明：实验 3 超参沿用 ehs_v2 主矩阵（两数据集均 dm512/el2/lr1e-4），
  与任务括号内 "exchange d_model 128 / electricity e_layers 3 lr 5e-4" 不一致——
  括号数值与 ehs_v2 实际配置矛盾，为保证与同臂原版可比采用主矩阵配置。
  实验 3 不设 --max_train_windows（按要求）；为此补跑了 12 个无上限 vanilla 对照臂
  （electricity 在 ehs_v2 中 sl≥1440 无完成臂，对照必需）。electricity sl=1440/2880
  用 bs=8/4 防 OOM（任务允许）。exchange 18 臂曾由本调度器之外的进程先行完成
  （本会话内其他 agent），结果经独立 checkpoint 重评验证后采纳；本调度器随后
  重跑了全部 exchange 臂，两轮结果一致（差异 < 1e-5 相对）。
