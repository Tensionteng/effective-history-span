# E3-ext: validation selector vs statistic predictor (n=28 LOO) vs oracle
# 6 new datasets x {iTransformer, DLinear} = 12 cells; logs logs/ehs_v2_ext/*.log
# parsed 180/180 logs (3 seeds x 5 sl per cell)

| dataset | model | oracle sl | val sl | stat sl (k=3) | mse(oracle) | regret val | rel% | regret stat | rel% |
|---|---|---|---|---|---|---|---|---|---|---|
| PEMS04 | DLinear | 2880 | 2880 | 1440 | 0.5665 | +0.0000 | +0.00 | +0.1256 | +22.17 |
| PEMS04 | iTransformer | 1440 | 2880 | 1440 | 0.5727 | +0.0189 | +3.29 | +0.0000 | +0.00 |
| PEMS08 | DLinear | 2880 | 2880 | 1440 | 0.6445 | +0.0000 | +0.00 | +0.1686 | +26.16 |
| PEMS08 | iTransformer | 2880 | 2880 | 1440 | 0.6284 | +0.0000 | +0.00 | +0.0651 | +10.37 |
| bikes | DLinear | 2880 | 720 | 2880 | 0.2162 | +0.0051 | +2.36 | +0.0000 | +0.00 |
| bikes | iTransformer | 2880 | 336 | 2880 | 0.2109 | +0.0056 | +2.66 | +0.0000 | +0.00 |
| ercot | DLinear | 2880 | 2880 | 1440 | 0.2312 | +0.0000 | +0.00 | +0.0067 | +2.92 |
| ercot | iTransformer | 336 | 336 | 1440 | 0.2117 | +0.0000 | +0.00 | +0.0134 | +6.33 |
| pedestrian | DLinear | 2880 | 2880 | 2880 | 0.2125 | +0.0000 | +0.00 | +0.0000 | +0.00 |
| pedestrian | iTransformer | 720 | 720 | 2880 | 0.2330 | +0.0000 | +0.00 | +0.0188 | +8.08 |
| solar | DLinear | 1440 | 1440 | 2880 | 0.1763 | +0.0000 | +0.00 | +0.0015 | +0.87 |
| solar | iTransformer | 2880 | 1440 | 2880 | 0.1815 | +0.0024 | +1.30 | +0.0000 | +0.00 |

new-12 | val     : hit 8/12, mean regret +0.0027, mean rel +0.80%, median rel +0.00%
new-12 | stat k=3: hit 5/12, mean regret +0.0333, mean rel +6.41%, median rel +1.89%
new-12 | stat k=2: hit 5/12, mean regret +0.0333, mean rel +6.41%, median rel +1.89%
new-12 | per-seed val: mean regret +0.0047

pooled 28 cells (old = 16 from logs/ehs_fix/e3_val_select_trained.json (e3_val_select.py regex now keeps exchange_rate); stat side re-derived under the n=28 LOO regressor):
pooled-28 | val     : hit 16/28, mean regret +0.0029, mean rel +0.92%, median rel +0.00%
pooled-28 | stat k=3: hit 8/28, mean regret +0.0186, mean rel +5.33%, median rel +2.30%
