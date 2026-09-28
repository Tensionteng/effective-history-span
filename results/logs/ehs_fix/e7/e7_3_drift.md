# E7.3 BOCPD-free drift statistics vs measured best seq_len

train split, <=24 linspace channels, standardized (same data as exp4).
meandiff_w: mean |mean diff| of adjacent w-windows; kl_w: symmetrized
Gaussian KL of adjacent w-windows (window variance floored at 1e-3,
standardized units; mean and median over t). All changepoint-model-free.

| dataset | meandiff w96 | meandiff w336 | KL w96 mean/med | KL w336 mean/med | drift_idx(20seg) | best iTr | best DLin |
|---|---|---|---|---|---|---|---|
| ETTh1 | 0.2468 | 0.2783 | 0.451/0.130 | 0.456/0.092 | 0.5254 | 720 | 1440 |
| ETTh2 | 0.2819 | 0.3519 | 11.972/0.197 | 30.538/0.301 | 0.5983 | 96 | 1440 |
| ETTm1 | 0.2674 | 0.2526 | 3.464/0.270 | 0.477/0.141 | 0.5250 | 336 | 336 |
| ETTm2 | 0.1957 | 0.2760 | 6.327/0.145 | 10.595/0.188 | 0.5994 | 336 | 1440 |
| exchange_rate | 0.2808 | 0.5739 | 6.552/1.831 | 7.213/2.990 | 0.8974 | 96 | 96 |
| weather | 0.3387 | 0.3177 | 62.743/0.748 | 45.232/0.344 | 0.3661 | 336 | 2880 |
| electricity | 0.1666 | 0.1751 | 1.523/0.020 | 0.872/0.019 | 0.1974 | 2880 | 2880 |
| traffic | 0.1906 | 0.0842 | 0.403/0.044 | 0.169/0.010 | 0.0222 | 2880 | 2880 |

- meandiff_w96 vs best sl [iTransformer]: rho=-0.791 (p_exact~0.025, Holm m=7 0.086, Holm m=14 0.236)
- meandiff_w96 vs best sl [DLinear]: rho=-0.325 (p_exact~0.436, Holm m=7 1.000, Holm m=14 1.000)
- meandiff_w336 vs best sl [iTransformer]: rho=-0.865 (p_exact~0.008, Holm m=7 0.050, Holm m=14 0.108)
- meandiff_w336 vs best sl [DLinear]: rho=-0.526 (p_exact~0.191, Holm m=7 0.955, Holm m=14 0.955)
- kl_w96 vs best sl [iTransformer]: rho=-0.803 (p_exact~0.021, Holm m=7 0.086, Holm m=14 0.236)
- kl_w96 vs best sl [DLinear]: rho=-0.200 (p_exact~0.652, Holm m=7 1.000, Holm m=14 1.000)
- kl_w96_med vs best sl [iTransformer]: rho=-0.840 (p_exact~0.013, Holm m=7 0.065, Holm m=14 0.157)
- kl_w96_med vs best sl [DLinear]: rho=-0.601 (p_exact~0.129, Holm m=7 0.771, Holm m=14 0.771)
- kl_w336 vs best sl [iTransformer]: rho=-0.655 (p_exact~0.087, Holm m=7 0.087, Holm m=14 0.608)
- kl_w336 vs best sl [DLinear]: rho=0.000 (p_exact~1.000, Holm m=7 1.000, Holm m=14 1.000)
- kl_w336_med vs best sl [iTransformer]: rho=-0.902 (p_exact~0.005, Holm m=7 0.033, Holm m=14 0.067)
- kl_w336_med vs best sl [DLinear]: rho=-0.501 (p_exact~0.218, Holm m=7 0.955, Holm m=14 0.955)
- drift_index_20seg vs best sl [iTransformer]: rho=-0.803 (p_exact~0.021, Holm m=7 0.086, Holm m=14 0.236)
- drift_index_20seg vs best sl [DLinear]: rho=-0.801 (p_exact~0.023, Holm m=7 0.163, Holm m=14 0.236)

Expectation under the EHS drift story: higher drift -> shorter best lookback -> NEGATIVE rho supports the story independent of BOCPD.
