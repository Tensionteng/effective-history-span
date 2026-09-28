# Confounder-controlled lookback sweep (iTransformer, pred_len=96)

All seq_len arms of a dataset trained on the SAME number of windows
(--max_train_windows = N_min, the window count of the seq_len=2880 arm;
evenly-spaced deterministic subsampling in data_provider/data_loader.py).
val/test splits untouched. model_id = LBctrl_{dataset}_{seq_len}.

## N_min (training windows per arm, after subsampling)

| dataset | N_min |
|---|---|
| ETTh1 | 5665 |
| ETTm1 | 31585 |
| exchange_rate | 2336 |
| weather | 33912 |
| electricity | 15437 |
| traffic | 9305 |

### MSE (test)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best |
|---|---|---|---|---|---|---|
| ETTh1 | 0.4003 | 0.4004 | 0.3947 | 0.3970 | 0.4576 | sl=720 |
| ETTm1 | 0.3432 | 0.3047 | 0.3139 | 0.3409 | 0.3419 | sl=336 |
| exchange_rate | 0.0890 | 0.1080 | 0.1255 | 0.1583 | 0.3330 | sl=96 |
| weather | 0.1756 | 0.1631 | 0.1719 | 0.1989 | 0.2247 | sl=336 |
| electricity | 0.1487 | 0.1339 | 0.1320 | 0.1314 | 0.1300 | sl=2880 |
| traffic | 0.4073 | 0.3639 | 0.3586 | 0.3476 | 0.3403 | sl=2880 |

### MAE (test)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best |
|---|---|---|---|---|---|---|
| ETTh1 | 0.4139 | 0.4196 | 0.4263 | 0.4300 | 0.4712 | sl=96 |
| ETTm1 | 0.3775 | 0.3582 | 0.3690 | 0.3803 | 0.3883 | sl=336 |
| exchange_rate | 0.2099 | 0.2350 | 0.2593 | 0.2943 | 0.4494 | sl=96 |
| weather | 0.2155 | 0.2118 | 0.2240 | 0.2538 | 0.2801 | sl=336 |
| electricity | 0.2405 | 0.2298 | 0.2286 | 0.2280 | 0.2287 | sl=1440 |
| traffic | 0.2779 | 0.2636 | 0.2618 | 0.2590 | 0.2560 | sl=2880 |
