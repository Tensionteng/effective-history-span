| dataset | model | oracle | val_sl | val hit | val rel% | aic grid | aic hit | pacf grid | pacf rel% |
|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | iTransformer | 720 | 96 | n | +1.99 | 336 | n | 2880 | +16.7 |
| ETTh1 | DLinear | 1440 | 336 | n | +3.14 | 336 | n | 2880 | +6.0 |
| ETTh2 | iTransformer | 96 | 96 | Y | +0.00 | 336 | n | 2880 | +49.1 |
| ETTh2 | DLinear | 720 | 720 | Y | +0.00 | 336 | n | 2880 | +47.0 |
| ETTm1 | iTransformer | 336 | 336 | Y | +0.00 | 720 | n | 2880 | +12.3 |
| ETTm1 | DLinear | 336 | 1440 | n | +2.65 | 720 | n | 2880 | +3.4 |
| ETTm2 | iTransformer | 336 | 720 | n | +2.95 | 720 | n | 2880 | +10.1 |
| ETTm2 | DLinear | 2880 | 1440 | n | +0.09 | 720 | n | 2880 | +0.0 |
| exchange_rate | iTransformer | 96 | 96 | Y | +0.00 | 96 | Y | 2880 | +255.6 |
| exchange_rate | DLinear | 96 | 96 | Y | +0.00 | 96 | Y | 2880 | +486.3 |
| weather | iTransformer | 336 | 336 | Y | +0.00 | 96 | n | 2880 | +37.8 |
| weather | DLinear | 2880 | 1440 | n | +0.24 | 96 | n | 2880 | +0.0 |
| electricity | iTransformer | 2880 | 1440 | n | +0.07 | 720 | n | 2880 | +0.0 |
| electricity | DLinear | 2880 | 2880 | Y | +0.00 | 720 | n | 2880 | +0.0 |
| traffic | iTransformer | 2880 | 720 | n | +5.06 | 720 | n | 2880 | +0.0 |
| traffic | DLinear | 2880 | 2880 | Y | +0.00 | 720 | n | 2880 | +0.0 |

{
  "val": {
    "exact_hits": 8,
    "n_cells": 16,
    "mean_rel_regret_pct": 1.01,
    "median_rel_regret_pct": 0.07
  },
  "aic": {
    "exact_hits": 2,
    "n_cells": 16,
    "mean_rel_regret_pct": 3.49,
    "median_rel_regret_pct": 2.91
  },
  "pacf": {
    "exact_hits": 6,
    "n_cells": 16,
    "mean_rel_regret_pct": 57.77,
    "median_rel_regret_pct": 10.09
  }
}
