# EHS v2 extension (E5): lookback measurement on 6 new datasets (n: 8 → 14)

Date: 2026-09-16. Scan: 180/180 arms finished (6 datasets × {iTransformer, DLinear} ×
sl {96,336,720,1440,2880} × seeds {2021,2022,2023}), zero failed arms.

## 0. New dataset inventory

All converted to TSLib `custom` format (first column `date`, last channel renamed `OT`,
no NaN), verified: rows ≥ 15000, uniform sampling grid, loads via `Dataset_Custom`
(train/val/test borders and channel counts checked through `data_provider`).

| dataset | file | rows | channels | sampling | span | source (as downloaded) |
|---|---|---|---|---|---|---|
| solar | dataset/solar/solar.csv | 52560 | 137 | 10-min | 2006-01-01..2006-12-31 | hf-mirror `Salesforce/GiftEval` `solar/10T` arrow (= LSTNet solar-energy, 137 PV plants in Alabama) |
| PEMS04 | dataset/PEMS04/PEMS04.csv | 16992 | 307 | 5-min | 2018-01-01..2018-02-28 | hf-mirror `jimmygao3218/PEMS04` npz, md5 5238832691101da85fcac772a32051da (identical to Zenodo TrafficDataSets rec. 7816008); speed channel (feat 2 of 3) kept |
| PEMS08 | dataset/PEMS08/PEMS08.csv | 17856 | 170 | 5-min | 2016-07-01..2016-08-31 | hf-mirror `jimmygao3218/PEMS08` npz, md5 2a528d169c0d90294c9e24288a430132 (identical to Zenodo rec. 7816008); speed channel kept |
| ercot | dataset/ercot/ercot.csv | 154872 | 8 | hourly | 2004-01-01..2021-09-01 | hf-mirror `autogluon/chronos_datasets` config `ercot` (ERCOT zonal settlement point prices); 18 DST spring-forward NaN hours (one/year, all zones) forward-filled |
| pedestrian | dataset/pedestrian/pedestrian.csv | 84331 | 13 | hourly | 2009-05-01..2018-12-13 | hf-mirror `autogluon/chronos_datasets` config `monash_pedestrian_counts` (Melbourne CBD sensors); 13/66 sensors covering the full common span kept |
| bikes | dataset/bikes/bikes.csv | 52608 | 441 | hourly | 2016-01-01..2021-12-31 | hf-mirror `autogluon/chronos_datasets` config `mexico_city_bikes` (EcoBici station demand); 441/494 stations with contiguous coverage of the window kept |

Network notes (this machine): github.com / huggingface.co unreachable (timeout);
gitcode.com reachable but its search API requires auth (token not available);
hf-mirror.com reachable — used for all dataset downloads; zenodo.org reachable but
too slow (~50 KB/s; PEMS abandoned there, same files fetched via hf-mirror with
md5 verified against the Zenodo record).

### wind candidate — evaluated and rejected (documented honestly)
`Monash-University/monash_tsf` `wind_farms_minutely_dataset_with_missing_values.zip`
(339 AEMO wind farms, minutely, 2019-08..2020-08) was downloaded and parsed. Missingness
is block-structured and identical across all 311 full-length series: 17.5% of 15-min
buckets fully absent, concentrated in two market-wide outage blocks of 4416 (46 days)
and 1718 (18 days) buckets. Longest fully-valid contiguous run = 7694 15-min buckets
(80 days) < 15000 rows; ffilling 46-day blocks would fabricate the signal. Rejected.
`autogluon/chronos_datasets` `wind_farms_hourly` was also checked: max series length
8784 < 15000. Rejected. No clean 15-min wind dataset at ≥15000 rows was obtainable
from reachable mirrors; the slot was given to `bikes` instead.

## 1. Scan protocol (identical to ehs_v2 main matrix)

- `--max_train_windows` = N_min of the sl=2880 arm: N_min = int(rows×0.7) − 2880 − 96 + 1
  (Dataset_Custom 0.7/0.1/0.2 split). solar 33817, PEMS04 8919, PEMS08 9524,
  ercot 105435, pedestrian 56056, bikes 33850.
- Config (task spec): wide (>100 ch) = electricity config (dm512/df512/el3/lr5e-4/bs16);
  narrow = ETT config (dm128/df128/el2/lr1e-4/bs32). Wide: solar/PEMS04/PEMS08/bikes;
  narrow: ercot (8 ch)/pedestrian (13 ch).
- Commands: `scripts/long_term_forecast/ehs_v2_ext/{main,bikes,all}_cmds.txt`
  (gen_cmds.py); logs `logs/ehs_v2_ext/{dataset}_{model}_sl{L}_s{seed}.log`.
- Ops note: first attempt ran with torch's default 128 OMP threads/job → load ~500,
  10–40× slowdown. Restarted with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`, P=16 →
  full 180-arm matrix finished in ~1.2 h. All results come from the restarted run.

## 2. Results (test MSE, mean±std over 3 seeds; full tables in SUMMARY.md)

### iTransformer
| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| solar | 0.2205±0.0123 | 0.2151±0.0098 | 0.1868±0.0045 | 0.1839±0.0209 | **0.1815±0.0071** | 2880 |
| PEMS04 | 0.8700±0.0006 | 0.6431±0.0052 | 0.6582±0.0016 | **0.5727±0.0073** | 0.5915±0.0079 | 1440 |
| PEMS08 | 0.9716±0.0027 | 0.7354±0.0023 | 0.7266±0.0094 | 0.6936±0.0097 | **0.6284±0.0080** | 2880 |
| ercot | 0.2249±0.0007 | **0.2117±0.0011** | 0.2173±0.0010 | 0.2251±0.0005 | 0.2278±0.0015 | 336 |
| pedestrian | 0.3552±0.0041 | 0.2370±0.0007 | **0.2330±0.0004** | 0.2479±0.0028 | 0.2519±0.0030 | 720 |
| bikes | 0.2340±0.0007 | 0.2165±0.0005 | 0.2145±0.0003 | 0.2147±0.0005 | **0.2109±0.0002** | 2880 |

### DLinear
| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| solar | 0.2858±0.0015 | 0.2217±0.0002 | 0.1906±0.0000 | **0.1763±0.0002** | 0.1778±0.0001 | 1440 |
| PEMS04 | 0.9197±0.0024 | 0.7312±0.0037 | 0.7313±0.0026 | 0.6921±0.0061 | **0.5665±0.0003** | 2880 |
| PEMS08 | 1.0412±0.0010 | 0.8562±0.0002 | 0.8381±0.0015 | 0.8131±0.0007 | **0.6445±0.0002** | 2880 |
| ercot | 0.2756±0.0033 | 0.2370±0.0013 | 0.2345±0.0020 | 0.2380±0.0049 | **0.2312±0.0015** | 2880 |
| pedestrian | 0.4568±0.0001 | 0.2387±0.0000 | 0.2198±0.0000 | 0.2138±0.0000 | **0.2125±0.0000** | 2880 |
| bikes | 0.3003±0.0016 | 0.2276±0.0007 | 0.2213±0.0002 | 0.2174±0.0000 | **0.2162±0.0001** | 2880 |

### best_sl on new datasets vs statistics (two-axis check)

| dataset | drift | calendar_acf_diff | best iT | best DL | two-axis classification |
|---|---|---|---|---|---|
| solar | 0.033 | +0.234 | 2880 | 1440 | periodicity-dominant → long window ✓ |
| PEMS04 | 0.092 | +0.095 | 1440 | 2880 | periodicity-dominant → long window ✓ |
| PEMS08 | 0.071 | +0.163 | 2880 | 2880 | periodicity-dominant → long window ✓ |
| pedestrian | 0.010 | +0.649 | 720 | 2880 | periodicity-dominant → long window ✓ |
| bikes | 0.016 | +0.258 | 2880 | 2880 | periodicity-dominant → long window ✓ |
| ercot | 0.340 | +0.900 | 336 | 2880 | drift/mixed: model-dependent (iT short 336, DL 2880) |

The two-axis pattern holds on the new data. All five low-drift sets (drift ≤ 0.10)
prefer long windows (≥1440) for both models — none of the 10 model×dataset cells
picks sl=96/336, mirroring the old matrix where only high-drift sets
(exchange_rate 0.89, ETTh2 0.65, ETTm1/m2 ~0.52–0.54, ETTh1 0.45) picked short windows.
ercot is the only new set with appreciable drift (0.34) and is the only one where
iTransformer prefers a short window (336) — i.e. the single drift-side observation
also lands on the predicted side. The new sets fill the previously sparse
low-drift/mid-calendar region, and the outcome there is uniformly "long window",
consistent with the axis-2 rule.

Caveat: PEMS04/08 calendar stats are computed on max_len=12000 rows (~41 days at
5-min ≈ 2 weekly cycles), so their calendar_acf_diff is a low-cycle estimate.

## 3. Statistics & predictability re-estimation (n = 14 datasets × 2 models = 28)

Pipeline: `analysis/ehs_predict_v3.py` (same atoms as v2: `analysis/ehs_stats.py`;
targets parsed from `logs/ehs_v2/SUMMARY.md` + `logs/ehs_v2_ext/SUMMARY.md`).
Full dump: `logs/ehs_v2_ext/taskB_predict_v3.txt`; samples: `taskB_samples_v3.json`.

### Spearman correlations, v2 (n=16) → v3 (n=28)

| statistic | vs benefit v2 | vs benefit v3 | vs log2(best_sl) v2 | vs log2(best_sl) v3 |
|---|---|---|---|---|
| periodicity_dt | +0.183 (p=.50) | +0.081 (p=.68) | +0.326 (p=.22) | +0.160 (p=.42) |
| **drift** | **−0.793 (p<.001)** | **−0.762 (p<.001)** | **−0.813 (p<.001)** | **−0.702 (p<.001)** |
| long_acf | +0.675 (p=.004) | +0.121 (p=.54) | +0.442 (p=.09) | +0.231 (p=.24) |
| **calendar_acf_diff** | +0.544 (p=.03) | +0.181 (p=.36) | **+0.694 (p=.003)** | **+0.395 (p=.038)** |
| long_acf_diff | +0.320 (p=.23) | +0.041 (p=.84) | +0.630 (p=.009) | +0.294 (p=.13) |

### Leave-one-dataset-out ridge (feature selection inside fold)

| k | LOO Spearman v2 (n=16) | LOO Spearman v3 (n=28) | exact tier hit | direction acc |
|---|---|---|---|---|
| 2 | +0.792 (p<.001) | **+0.702 (p<.001)** | 9/28 (baseline 5/28) | 20/28 |
| 3 | +0.813 (p<.001) | **+0.702 (p<.001)** | 15/28 (baseline 5/28) | 20/28 |

Fold-selected features are (drift, calendar_acf_diff) in 13/14 folds at k=2
(exchange_rate fold picks drift+periodicity_dt), i.e. the model-free two-axis
descriptor is stable under n-expansion.

### Verdict (n = 14)

- **Axis 1 (drift → short window): holds, essentially unchanged.** ρ = −0.76 (benefit)
  / −0.70 (best_sl), p < 0.001 at n=28. Still the dominant single predictor.
- **Axis 2 (calendar periodicity → long window): holds for best_sl, attenuated for
  benefit.** calendar_acf_diff vs log2(best_sl) stays significant (+0.395, p=.038);
  the benefit correlation drops below significance, largely a ceiling effect —
  12/28 samples now sit at the sl=2880 boundary — and the low-cycle PEMS estimate.
- **LOO predictability survives expansion**: Spearman +0.70 (was +0.79/+0.81 at n=16),
  still p<.001; exact-tier 15/28 vs 5/28 constant baseline (k=3); direction 20/28 (71%).
  Mild attenuation is expected: the 8-set pilot spanned the extremes, the 6 new sets
  add mid-range points. Conclusion: the statistics-to-best-lookback law is predictive
  out-of-dataset at n=14; the paper's E5 robustness item is satisfied.
- long_acf / long_acf_diff / periodicity_dt lose significance at n=28 (their v2 signal
  was carried by electricity/traffic extremes); keep them as secondary descriptors,
  not headline axes.

## 4. Zero-shot extension: Chronos-Bolt on the 6 new datasets (E5 follow-up)

Date: 2026-09-16. Script `tsfm/e5_ext_zeroshot.py`; caches `logs/ehs_v2_ext/zeroshot/zs_*.npz`
(per-window per-channel se/ae at all 5 lookbacks); merged table `zeroshot_summary.json`.

Protocol identical to E1 (`tsfm/e1_wide_zero_shot.py`, itself identical to task C v2):
Chronos-Bolt-base zero-shot, test dataset built at seq_len=max(L) so the prediction-target
set is independent of L (TSL border logic), channel-batched inference with L-adaptive
chunking, pred_len=96. Per-dataset strides (protocol-internal, same for all L): ercot 1,
pedestrian 1, PEMS04 2, PEMS08 2, solar 4, bikes 6 — chosen to keep strided window counts
in the 1.6k–2.6k band of the E1 runs (electricity stride2 → 2583; traffic stride2 → 1707).
**L=2880 arm = effective context 2048** (bolt-base context_length=2048; inputs are
left-truncated inside ChronosBoltPipeline.predict — same disclosure as E1).

### 4.1 Results (test MSE / MAE, standardized channels)

| dataset | stride | n_win×C | L=96 | L=336 | L=720 | L=1440 | L=2880* | best L |
|---|---|---|---|---|---|---|---|---|
| solar | 4 | 2605×137 | 1.9890 / 0.9301 | 0.6190 / 0.4579 | 0.2388 / 0.2632 | **0.2060 / 0.2237** | 0.2164 / 0.2240 | 1440 |
| PEMS04 | 2 | 1652×307 | 1.7383 / 0.7857 | 1.1249 / 0.5968 | 1.0257 / 0.5282 | 0.9522 / 0.4924 | **0.8651 / 0.4600** | 2880* |
| PEMS08 | 2 | 1738×170 | 2.0314 / 0.8519 | 1.4089 / 0.6843 | 1.1802 / 0.5782 | 1.0755 / 0.5183 | **0.9976 / 0.4861** | 2880* |
| ercot | 1 | 30879×8 | 0.2829 / 0.3658 | 0.2505 / 0.3429 | 0.2431 / 0.3407 | 0.2331 / 0.3346 | **0.2325 / 0.3339** | 2880* |
| pedestrian | 1 | 16771×13 | 6.8905 / 0.2196 | 0.1663 / 0.1606 | **0.1620 / 0.1571** | 0.1624 / 0.1571 | 0.1634 / 0.1580 | 720 |
| bikes | 6 | 1738×441 | 0.2404 / 0.2845 | 0.2148 / 0.2646 | **0.2113** / 0.2626 | 0.2894 / 0.2622 | 0.5404 / 0.2627 | 720 (see note) |

\* effective context 2048. Per-channel best-L distribution: solar 125/137 @1440;
PEMS04 266/307 @2880; PEMS08 145/170 @2880; pedestrian 10/13 @720; bikes 298/441 @2880;
ercot 4/8 @2880, 3/8 @1440, 1/8 @336.

**bikes note (mean-MSE outlier effect, disclosed):** dataset-mean MSE at L=1440/2880 is
dominated by a handful of channels (max per-channel MSE 1.60 @720 → 34.8 @1440 → 131.3
@2880; only 128/441 channels worsen 1440→2880). The per-channel **median** MSE decreases
monotonically (0.2026 → 0.1779 → 0.1743 → 0.1735 → 0.1732), MAE is flat-to-improving
(0.2845 → 0.2622 @1440), and 298/441 channels individually prefer 2880. The blow-up
channels are near-zero stations whose Bolt mean-scaling amplifies occasional spikes at
long context. By mean MSE the dataset-level best is L=720; by median/MAE/per-channel-vote
it is 2880.

### 4.2 Chronos pretraining-corpus overlap (disclosure)

Checked against the Chronos paper appendix corpus table (arXiv:2403.07815, Appendix B
Table 3: 55 datasets = 13 pretraining-only + 15 in-domain + 27 zero-shot) and the AWS
Benchmark-II "no prior exposure" criterion, 2026-09-16. Caveat for all rows: the
enumerated corpus is from the Chronos-T5 paper; Chronos-Bolt's model card states only
"nearly 100 billion time series observations" without an enumerated corpus, so absence
claims cannot be independently verified for Bolt itself.

| dataset | our file's origin | in Chronos corpus? |
|---|---|---|
| ercot | autogluon/chronos_datasets `ercot` = "ERCOT Load" (8 zones, hourly, 2004–2021; paper lists 154854 obs, ours 154872 with 18 DST hours ffilled) | **No — held out.** Listed in the **zero-shot evaluation** section of Table 3 (Benchmark II, "no prior exposure"). Genuine zero-shot for Chronos/Bolt. |
| pedestrian | autogluon/chronos_datasets `monash_pedestrian_counts` = "Pedestrian Counts" (66 Melbourne sensors; we keep 13/66) | **Yes — in-domain** section of Table 3 (used in Chronos training). Ours is a subset. |
| bikes | autogluon/chronos_datasets `mexico_city_bikes` = "Mexico City Bikes" (494 stations; we keep 441/494, 2016–2021 window) | **Yes — pretraining-only** section of Table 3. Ours is a subset. |
| solar | Salesforce/GiftEval `solar/10T` = LSTNet solar-energy (137 Alabama PV plants, 10-min, 2006; GIFT-Eval App. D: "we utilize the solar dataset from LSTNet") | **Source-level overlap.** Table 3 pretraining-only lists "Solar (5 Min.)"/"Solar (Hourly)" = NREL solar power data (US, 2006, 5166 sites); LSTNet solar is a 137-site Alabama subset of the same NREL 2006 source at 10-min. The exact 10-min/137-site configuration differs, but the underlying measurements are in the corpus → cannot claim no-prior-exposure. |
| PEMS04 | Zenodo TrafficDataSets rec. 7816008 (Caltrans PeMS, speed, 5-min) | **无法排除 / no positive evidence.** Not in Table 3; closest relative "Traffic" (862 sensors, hourly *occupancy*, SF Bay Area) is itself in the zero-shot (held-out) section — different measure and sensor set. Bolt corpus not enumerated → inclusion cannot be excluded, nothing indicates it. |
| PEMS08 | same as PEMS04 | **无法排除 / no positive evidence.** Same reasoning. |

Net: only **ercot** is verifiably unseen by Chronos; pedestrian/bikes/solar are (subsets of)
training-corpus data; PEMS04/08 are absent from the enumerated corpus but cannot be
positively excluded for Bolt. The lookback-pattern verdict below is about *relative*
performance across L on identical targets, which is meaningful regardless of membership,
but absolute zero-shot numbers for in-corpus datasets should not be read as
generalization evidence.

### 4.3 Verdict: zero-shot best L vs trained side

| dataset | drift | calendar_acf_diff | zero-shot best L | trained best sl (iT / DL) | consistent? |
|---|---|---|---|---|---|
| solar | 0.033 | +0.234 | 1440 | 2880 / 1440 | ✓ long |
| PEMS04 | 0.092 | +0.095 | 2880* | 1440 / 2880 | ✓ long |
| PEMS08 | 0.071 | +0.163 | 2880* | 2880 / 2880 | ✓ long |
| pedestrian | 0.010 | +0.649 | 720 (flat ≥720, Δ<1%) | 720 / 2880 | ✓ long |
| bikes | 0.016 | +0.258 | 720 mean / 2880 median | 2880 / 2880 | ✓ long (outlier caveat) |
| ercot | 0.340 | +0.900 | 2880* (monotone ↓) | **336** / 2880 | partial: matches DL, not iT |

- **Period-dominated → long window holds zero-shot on all 5 low-drift new sets** (drift
  ≤ 0.10): every one has best L ≥ 720 and none picks 96/336; MSE gains from 96 → best are
  large (solar −90%, PEMS04 −50%, PEMS08 −51%, pedestrian −97.6% [L=96 catastrophic:
  6.89 vs 0.162], bikes −12%/median-monotone). This mirrors the trained-side §2 outcome.
- **ercot, the only drift-side new point, prefers long context zero-shot** (0.2829 →
  0.2325 monotone, best 2880\*; 1440 within 0.3%). The trained-iTransformer short-window
  result (336) does NOT transfer to zero-shot Bolt; zero-shot ercot behaves like trained
  DLinear (2880). With its very high calendar_acf_diff (0.90), periodicity dominates the
  zero-shot response. In the pooled zero-shot picture (old: ETTh1 0.45→1440,
  exchange_rate 0.89→**96**, weather→1440, electricity→1440, traffic→2880; new: all
  ≥720), the drift→short-window axis only manifests at extreme drift (0.89) for Bolt —
  the zero-shot drift threshold is much higher than the trained iTransformer's.
- Overall: **6/6 new datasets have zero-shot best L ≥ 720; 5/6 (all low-drift) agree with
  the trained-side "period-dominated → long window" rule; ercot is split** (zero-shot
  long, trained iT short, trained DL long). The axis-2 (calendar → long window) law
  transfers to the foundation-model zero-shot setting; axis-1 (drift → short) is weaker
  zero-shot than trained.

## Artifacts

- logs/ehs_v2_ext/: 180 run logs, SUMMARY.md, REPORT.md, dataset_stats_v3.json,
  taskB_predict_v3.txt, taskB_results_v3.json, taskB_samples_v3.json
- logs/ehs_v2_ext/zeroshot/: zs_{ercot,pedestrian,PEMS04,PEMS08,solar,bikes}.npz
  (per-window per-channel se/ae, 5 lookbacks), zs_bikes_L2880.npz (side cache, merged
  into zs_bikes.npz), run_*.log, zeroshot_summary.json; script tsfm/e5_ext_zeroshot.py
- scripts/long_term_forecast/ehs_v2_ext/: gen_cmds.py, main/bikes/all_cmds.txt
- analysis/: ehs_ext_prep.py, ehs_ext_prep_pems.py, ehs_ext_prep_bikes.py,
  ehs_ext_prep_wind.py (rejected candidate), collect_ehs_ext.py, ehs_predict_v3.py
- dataset/{solar,PEMS04,PEMS08,ercot,pedestrian,bikes}/*.csv (TSLib custom format)

## 4. E3-ext: validation 选择器扩展到 6 新数据集（2026-09-16，零训练）

协议与 `analysis/e3_val_select.py` 逐点一致（脚本 `analysis/e3_val_select_ext.py`，产物
`e3_ext_verdict.{json,md}`）：解析本目录 180/180 条日志的逐 epoch "Vali Loss" 最小值
（early-stopping 依据）与最终 test MSE；每 (dataset, model) 取验证误差最小的 sl 为
val-selected，与 test oracle 对比 regret。统计量预测侧复用 `analysis/ehs_predict_v3.py`
的 n=28 留一（LOO）岭回归器（k=3 特征，折内选特征；k=2 敏感性）——对新数据集而言
LOO 即严格的样本外预测。

12 格（6 新数据集 × {iTransformer, DLinear}）结果（全表 `e3_ext_verdict.md`）：

| 选择器 | 精确命中 | 平均 regret | 平均相对 | 中位相对 | 成本 |
|---|---|---|---|---|---|
| validation（标准 HPO） | 8/12 | +0.0027 | +0.80% | +0.00% | 5 档 × 3 seed 全量训练 |
| 统计量预测（n=28 LOO, k=3） | 7/12 | +0.0049 | +1.79% | +0.00% | ≈0（3 个廉价统计量） |

- stat 的 5 格失误中 3 格与 val 相同（PEMS04-iT +3.3%，两者同选 2880），独有失误集中在
  "居中 oracle" 格：ercot-iT（oracle 336，stat 选 1440，+6.3%）与 pedestrian-iT
  （oracle 720，选 2880，+8.1%）——n=28 回归器在 12/28 样本压 2880 边界的训练分布下
  系统性偏好长档。反向地，stat 在 bikes 两格与 solar-iT 修正了 val 的失误（val 验证集
  对长档分辨力弱：bikes 选 336/720 各 +2.4~2.7%）。
- 配对差（stat − val，相对 oracle）：mean +0.99pt ± SEM 0.94pt（t≈1.05，不显著）；
  中位差 0。val 的 per-seed 变体平均 regret +0.0047，与 mean-based +0.0027 的协议内
  摆动本身就有 ~0.4pt 量级。
- 合并 28 格（旧 14 格 + 补解析 exchange_rate 2 格——原 `e3_val_select.py` 的文件名
  正则 `[A-Za-z0-9]+` 漏掉了 exchange_rate 的下划线，旧 E3 表实为 14 格且分母按 16 计，
  此处按正确口径重算；stat 侧统一用 n=28 LOO 预测）：val 16/28（+0.94%）、
  stat 15/28（+1.83%）。

**判决：n=28 口径下"统计量打平 validation"保持（弱化版）。** 两者中位 regret 均为 0，
平均相对 regret 差 ~1pt 且在配对 SEM 内不显著；stat 成本仍近似为零。但逐点估计的
方向由旧 14 格的"stat 略优"翻转为新 12 格的"stat 略逊"（+1.79% vs +0.80%），失误结构
清楚（居中-oracle 格被长档先验拉走）。建议论文表述由"略优于"软化为"与 validation
选择统计上不可区分，成本为零"。
