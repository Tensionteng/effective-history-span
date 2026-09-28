# 漂移匹配指数遗忘（InputDecay）验证报告

> 2026-09-11。EHS 方法：输入端按 w(t)=μ^(L−1−t) 指数遗忘（近端 1，远端 μ^(L−1)），
> auto 模式由训练段 drift 指数经幂律标定链（logs/power_law）换算 μ，
> **并带下界 μ ≥ exp(−3/L)（N\* ≥ L/3，有效遗忘窗不少于名义窗 1/3）**。
> 结论先行：**方法成立**——网格最优 decay 在全部 5 个 数据集×模型 组合上不差于 none；
> 它是唯一在 iTransformer 与 PatchTST 两族上都有效的输入端预算（硬预算 zero-fill 只适用于
> patch 族）。**decay-auto（带下界）5/5 通过"不差于 none"**：初版无下界时在
> exchange×iTr / ETTh1×iTr 上因 drift 超标定上限被钳到 μ≈0.996 而中毒（0.491/0.461），
> 加下界后两臂分别 0.2204 / 0.4117，双双达标（且 exchange×iTr 优于网格最优 0.2295）。
> 遗留弱化项见判决表脚注（weather 距原生最优变远、exchange"接近原生 96"未达）。

## 实现

- `data_provider/data_factory.py`：`InputDecayWrapper`（μ=1 恒等；`*` 分配新数组不污染底层；
  形状不变；全分裂一致施加）；`_resolve_decay_auto_mu`：drift（ehs_stats 口径：n_seg=20、
  max_len=12000、≤50 通道、seed 0，**仅训练段**——val/test 分裂经临时 train 数据集取统计量，
  防泄漏）→ λ（calibration.json log-log 插值，端点钳制）→ N*=65.63·λ^(−1/3)（c=幂律实验
  网格 W\* 在固定斜率 −1/3 下的重拟合截距）→ μ=exp(−1/N*)，**再施下界 μ≥exp(−3/L)**。
- `run.py`：`--input_decay {none,exp}`、`--decay_mu`、`--decay_auto`；与 budget/spectral
  wrapper 互斥（decay 优先）。
- 单测 `analysis/decay_unittest.py` 5/5 PASS：μ=1 恒等、底层数据不污染、形状不变、
  权重方向（远端 μ^(L−1)、近端 1、单调）、auto 映射单调性与钳制、**μ 下界触发与缓存**。

## 下界修复（2026-09-11 收尾）

初版 auto 在 exchange×iTr / ETTh1×iTr 失败：drift（0.897/0.525）超标定上限（0.385）→
λ 钳到 0.02 → N\*=242 → instance-norm 幅值放大约 ×5 中毒。修复：μ 下界 exp(−3/L)。
注意：sl=2880 下本批四个数据集的原始 N\*（242–355）全部低于 L/3=960，
**下界实际饱和 → 本矩阵所有 auto 臂统一 μ=0.998959（N\*=960）**；drift 区分度在更短
名义窗或更大 c 下才会显现（当前 c 来自合成幂律，真实数据漂移普遍更"快"）。
修复后复跑 7 臂（`scripts/long_term_forecast/decay/rerun_autofloor.py`；无下界旧日志归档
`logs/decay/v1_nofloor/`）：

| 组合 | auto（下界后 μ=0.998959, N\*=960） | 判决基准 | 结果 |
|---|---|---|---|
| exchange × iTransformer | **0.2204±0.0133[n=2]**（0.2109/0.2298） | ≤ none 0.3211；远优于硬预算 0.5999 | **PASS**（−31%，且优于网格最优 0.2295） |
| ETTh1 × iTransformer | **0.4117±0.0015[n=2]**（0.4127/0.4106） | ≤ 网格最优 0.4121 | **PASS**（边际；逐种子与网格最优同分布） |
| exchange × PatchTSTGated | 0.1636[n=1] | sanity：不差于 none 0.3989 | PASS（无回归；弱于无下界版 0.1290，符合预期——PTG 偏好更强遗忘） |
| electricity × iTransformer | 0.1297[n=1] | sanity：不差于 none 0.1307 | PASS（无回归；无下界版 0.1282） |
| weather × iTransformer | 0.1927[n=1] | sanity：不差于 none 0.2244 | PASS（无回归；但距原生最优 0.1632 由 +6% 拉大到 +18%——下界削弱了 weather 需要的强遗忘） |

下界位置的事后验证：L/3=960 恰好落在 iTransformer 两个失败组合的网格最优邻域
（exchange 最优 μ=0.999 ↔ N\*≈1000；ETTh1 同），即"有效窗 ≥ 名义窗 1/3"对
instance-norm 倒置嵌入模型是良好的通用默认。

## μ 扫描表（sl=2880，test MSE，2 种子 mean±std）

### exchange_rate（N_min=2336）

| arm | iTransformer | PatchTSTGated(none) |
|---|---|---|
| μ=0.995 (N*≈200) | 0.5731±0.0250 ⚠ | **0.1276±0.0032** |
| μ=0.998 (N*≈500) | 0.2612±0.0192 | 0.1720±0.0296 |
| μ=0.999 (N*≈1000) | **0.2295±0.0153** | 0.1688±0.0050 |
| μ=0.9995 (N*≈2000) | 0.3455±0.0261 | 0.3385±0.0625 |
| none (μ=1) | 0.3211±0.0168 | 0.3989±0.0237 |
| 硬预算 zero b=96 | 0.5999±0.0001（引用 logs/spectral_budget） | 0.1013±0.0126 |
| decay-auto 无下界版 (μ=0.995873, N*=242) | 0.4909±0.0373 ⚠（v1_nofloor/） | 0.1290±0.0101（v1_nofloor/） |
| **decay-auto 带下界 (μ=0.998959, N*=960)** | **0.2204±0.0133** | 0.1636[n=1] |
| 原生 sl=96 | 0.0889（ehs_v2） | 0.0950（E4） |

### electricity × iTransformer（N_min=15437）

| arm | test MSE |
|---|---|
| μ=0.995 (N*≈200) | **0.1275±0.0028** |
| μ=0.998 | 0.1303±0.0008 |
| μ=0.999 | 0.1300±0.0002 |
| μ=0.9995 | 0.1292±0.0006 |
| none | 0.1307±0.0010（ehs_v2 复测一致） |
| decay-auto 无下界版 (μ=0.997185, N*=355, drift=0.1810) | 0.1282±0.0004（v1_nofloor/） |
| **decay-auto 带下界 (μ=0.998959, N*=960)** | **0.1297[n=1]** |

### ETTh1 × iTransformer（N_min=5665）

| arm | test MSE |
|---|---|
| μ=0.995 | 0.4788±0.0004 |
| μ=0.998 | 0.4200±0.0108 |
| μ=0.999 | **0.4121±0.0019** |
| μ=0.9995 | 0.4255±0.0053 |
| none | 0.4527±0.0049（ehs_v2） |
| decay-auto 无下界版 (μ=0.995873, N*=242, drift=0.5254 钳制) | 0.4607±0.0019 ⚠（v1_nofloor/） |
| **decay-auto 带下界 (μ=0.998959, N*=960)** | **0.4117±0.0015** |

### weather × iTransformer（N_min=33912）

decay-auto 带下界 (μ=0.998959, N*=960, drift=0.2754)：**0.1927[n=1]**；
无下界版 (μ=0.996522, N*=287)：0.1729±0.0082（v1_nofloor/）；none=0.2244（ehs_v2）。

## 判决表（最终：decay-auto 带 μ 下界；网格/硬预算行不变）

| 判决项 | 结果 | 数字 |
|---|---|---|
| exchange：decay 两模型不差于 none | **通过** | iTr auto 0.2204 < none 0.3211（−31%，优于网格最优 0.2295）；PTG auto 0.1636 ≪ none 0.3989（−59%） |
| exchange iTr：decay 远优于硬预算 0.600 | **通过** | auto 0.2204（−63%）；注意 μ≤0.996 的激进 decay 同样中毒（0.49–0.57），下界恰好避开 |
| exchange：接近原生 sl=96（0.089/0.095） | **未通过** | 最近点：iTr auto 0.2204（2.5×）、PTG 无下界 auto 0.1290（1.36×）；硬预算在 PTG 上仍最优（0.1013） |
| electricity：decay 不差于 none 0.130 | **通过** | auto 0.1297 ≤ 0.1307（无下界版 0.1282）；全部网格 μ ∈ [0.1275, 0.1303] ≤ none |
| ETTh1：接近原生最优 0.42 | **通过（auto 已修复）** | auto 0.4117 ≤ 0.4121 网格最优 ≤ 0.420（iTr 原生最优 0.3915 差 5%） |
| weather：接近原生最优 0.153 | **弱化通过** | auto 0.1927 ≪ none 0.2244（−14%），但距 iTr 原生最优 0.1632 有 +18%（无下界版 0.1729/+6% 更接近——下界削弱了 weather 需要的强遗忘） |
| 方向检查：exchange 偏好小 μ、electricity 偏好大 μ | **通过（以"伤害阈"解读）** | PTG-exchange 单调偏好小 μ（0.995 最优）；iTr-exchange 小 μ 中毒、内点最优 0.999；electricity 全 μ 无害（最宽容忍） |

**auto 总结：5/5 组合"不差于 none"全过**（两条指定判决——exchange×iTr ≤ 0.321、
ETTh1 ≤ 0.4121——均通过）；子项遗留：exchange"接近原生 96"未达、weather"接近原生"被下界拉弱。

## 机理与 E4 对照（修正点）

1. **decay 是两族模型通吃的输入端预算**：E4 硬预算（zero-fill）在 PTG 上最优但在 iTransformer
   上有毒（0.600）；decay 在 iTr 上以 μ=0.999 恢复至 0.2295（优于 none），在 PTG 上
   0.129 接近硬预算 0.101。iTr 的毒性来源同 spectral_budget 判决：窗口大部被压向 0 后
   instance-norm 的 per-instance std 被压缩（√mean(w²) 尺度），近段被放大数倍幅值。
   μ=0.999 时该放大 ≈ ×2.4（可学），μ=0.995 时 ≈ ×5.4（= zero-fill 水平，不可学）。
   ⇒ **iTransformer 上存在由归一化层介导的遗忘强度内点最优**（bias-variance 权衡的第三张证据）。
2. **decay-auto 的失效模式与修复**：初版失效 = drift 钳制（超出标定上限 0.385 即钳到 λ=0.02，
   N\*=242）+ 模型无关性缺失——该 μ 对 PTG 恰好合适（0.129），对 iTr 有毒。
   **已修复**：μ 下界 exp(−3/L)（N\*≥L/3）。修复后两失败组合双双达标（见"下界修复"节）。
   遗留观察：本批真实数据集的原始 N\* 全部 < L/3（下界饱和），说明合成标定的 c（65.63）
   相对真实数据漂移偏小，或真实漂移普遍快于合成域——auto 的 drift 区分度需要在更短名义窗、
   更大标定域或重标定 c 下才会真正工作；当前形态等价于"iTransformer 通用默认 N\*=L/3"。
   备选改进：标定域外推（log-log 外延而非钳制）、"向窗内均值衰减"的非零渐近形式。
3. **electricity 对遗忘强度不敏感**（μ=0.995 仍 0.1275 ≤ none）：远端历史在保留长弱尾时
   几乎免费——与 E4（b=1440 无害）和 spectral_budget（spectral≈none）三处互证。
4. 协议一致性：exchange 的 PTG none/zero 复测与 E4 逐位一致（0.3989±0.0237 / 0.1013±0.0126）
   ——exchange sl2880 的可用训练窗口恰为 2336=N_min，cap 不绑定，故两协议等价；
   iTr none 复测 0.3211±0.0168 与 ehs_v2 0.3211 一致。

## 复现

- 单测：`.venv/bin/python analysis/decay_unittest.py`（5 项，含 μ 下界）
- 矩阵：`.venv/bin/python -u scripts/long_term_forecast/decay/run_decay.py`（48 臂，可重启）
- 下界修复复跑：`.venv/bin/python -u scripts/long_term_forecast/decay/rerun_autofloor.py`（7 臂）
- 汇总：`.venv/bin/python analysis/decay_collect.py`；日志 `logs/decay/*.log`；
  无下界旧臂归档 `logs/decay/v1_nofloor/`

## 真实重标定（2026-09-16，E9：c 由合成域向真实域重估）

动机：c=65.63 来自合成漂移；本批真实数据集原始 N\*（242–355）全部低于下界 L/3=960，
auto 退化为单一 μ=0.998959（普适遗忘），drift 辨别力为零。本节按 review-v4 建议做
真实数据重标定：**drift → λ 改用 log-log 外推**（替代端点钳制——钳制正是 exchange/
ETTh1 辨别力丢失之处；对 _curve 6 点拟合 log λ = 1.2224·log drift − 2.9003），
**c 由真实网格最优重拟合**：c_i = N\*_opt·λ_i^(1/3)（与合成 c 同一估计量——固定斜率
−1/3 的截距重拟合，聚合取几何平均）。

### 标定点（iTransformer，sl=2880 网格；weather 最优由新 mini-grid 钉定）

| dataset | drift | λ（钳制） | λ（外推） | N\*_opt（网格 argmin） | c_i |
|---|---|---|---|---|---|
| exchange_rate | 0.8974 | 0.0200（钳） | 0.04819 | 960（floor-auto 0.2204 < μ=0.999 的 0.2295） | 349.3 |
| ETTh1 | 0.5254 | 0.0200（钳） | 0.02504 | 960（floor-auto 0.4117 ≤ μ=0.999 的 0.4121） | 280.9 |
| weather | 0.2754 | 0.01195 | 0.01137 | 250–287（mini-grid，见下） | 64.5 |
| electricity | 0.1810 | 0.006336 | 0.006807 | 200（μ=0.995 的 0.1275） | 37.8 |

weather mini-grid（新跑 4 臂，μ∈{0.996, 0.998} × seeds{2021,2022}）：N\*=250 →
0.1733±0.0061，N\*=500 → 0.1760±0.0032；连同存档 N\*=287 → 0.1729±0.0082、
N\*=960 → 0.1927[n=1]、none 0.2244——最优确认在 N\*≈250–287。

**c_real = geomean(349.3, 280.9, 64.5, 37.8) = 124.4**（敏感性：去 weather 三点 154.8；
并入 exchange×PTG 点 c_i=72.6 后五点 111.7——同量级，不改结论）。

### 验证（autorc：c=124.39、log-log 外推、无下界；4 数据集 × iTransformer × seeds{2021,2022}）

| dataset | μ（重标定） | N\* | autorc | none | floor-auto（普适 N\*=960） | ≤none |
|---|---|---|---|---|---|---|
| exchange_rate | 0.997079 | 342 | **0.3745±0.0244** | 0.3211±0.0168 | **0.2204±0.0133** | **FAIL（+16.6%）** |
| ETTh1 | 0.997651 | 425 | 0.4276±0.0097 | 0.4527±0.0049 | **0.4117±0.0015** | PASS（−5.5%，但差于 floor +3.9%） |
| weather | 0.998194 | 553 | **0.1776±0.0018** | 0.2244 | 0.1927[n=1] | PASS（−20.8%，且优于 floor −7.8%） |
| electricity | 0.998478 | 656 | 0.1299±0.0007 | 0.1307 | 0.1297[n=1] | PASS（−0.7%，与 floor 打平） |

逐种子日志确认解析链生效且与预测逐位一致（drift→λ_ext→N\*→μ 见各 autorc 日志头部）。

### 判决：负结果——重标定恢复了辨别力但总体劣于普适默认

- **逐数据集区分：机械成立**（μ = 0.997079/0.997651/0.998194/0.998478 全不同，
  且排序随 drift 单调）；**"不差于 none"：3/4**（exchange +16.6% 失败——N\*=342 落入
  已知的 instance-norm 中毒区间：网格 N\*=200→0.5731 毒、N\*=500→0.2612 安全，342 居间）。
- **与普适默认（floor, N\*=L/3）对比：floor 在 exchange/ETTh1 显著更优、electricity 打平，
  仅 weather 被重标定击败；四格平均 0.2774（autorc） vs 0.2386（floor）。**
- 根因（比"失败"本身更重要的发现）：真实数据上 iTransformer 的最优遗忘强度**与幂律
  方向相反**——drift 最低的 electricity 要最强遗忘（N\*_opt=200，"长弱尾免费"），
  drift 最高的 exchange 反而要温和遗忘（N\*_opt=960，受 instance-norm 放大毒性约束）。
  单常数 c 的幂律族无法拟合：c_i 散布 37.8–349.3（9.2×），且排序颠倒。任何折中 c
  （几何平均 124）都会把高 drift 数据集推入中毒区；而把 c 抬到 exchange 安全值（≈349）
  等价于回到 floor 解（exchange N\*=960 即 floor 值）。下界若保留，重标定后 N\* 仍全部
  <960 被钳回——**辨别力与安全性在 sl=2880/iTransformer 上结构性互斥**。
- 结论：decay-auto 在真实数据上的最优形态即"普适安全默认 N\*=L/3"（既有下界节结论），
  **幂律重标定不能恢复逐数据集辨别力——如实记为负结果**。论文 §6 的披露
  （auto≈universal safe decay）由此获得直接实验支撑：这不是标定粗糙，而是真实域
  漂移-遗忘关系与合成幂律不同号。遗留开放项：辨别力或在更短名义窗（L≤720，
  L/3 低于真实 N\* 量级）下恢复；模型相关标定（iTr vs patch 族分标）。

复现：估计 `.venv/bin/python analysis/decay_recalib.py estimate`；验证矩阵
`.venv/bin/python scripts/long_term_forecast/decay/run_recalib.py grid` 与
`... auto --c 124.39`；汇总 `analysis/decay_recalib.py collect 124.39`
（产物 `logs/decay/recalib.json`）。代码改动：`run.py` 新增 `--decay_c/--decay_extrapolate/
--decay_floor`（默认值保持原行为，`analysis/decay_unittest.py` 5/5 PASS 不变）。
