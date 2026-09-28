| dataset | model | val-selected sl | oracle sl | hit | mse(val-sel) | mse(oracle) | regret | rel.% |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | DLinear | 336 | 1440 | n | 0.3801 | 0.3686 | +0.0116 | +3.14 |
| ETTh1 | PatchTST | 96 | 336 | n | 0.3862 | 0.3849 | +0.0014 | +0.35 |
| ETTh1 | TimesNet | 96 | 96 | Y | 0.4173 | 0.4173 | +0.0000 | +0.00 |
| ETTh1 | iTransformer | 96 | 720 | n | 0.3993 | 0.3915 | +0.0078 | +1.99 |
| ETTh2 | DLinear | 720 | 720 | Y | 0.3081 | 0.3081 | +0.0000 | +0.00 |
| ETTh2 | PatchTST | 336 | 96 | n | 0.3137 | 0.2939 | +0.0199 | +6.75 |
| ETTh2 | TimesNet | 96 | 96 | Y | 0.3337 | 0.3337 | +0.0000 | +0.00 |
| ETTh2 | iTransformer | 96 | 96 | Y | 0.2988 | 0.2988 | +0.0000 | +0.00 |
| ETTm1 | DLinear | 1440 | 336 | n | 0.3083 | 0.3003 | +0.0080 | +2.65 |
| ETTm1 | PatchTST | 336 | 336 | Y | 0.2911 | 0.2911 | +0.0000 | +0.00 |
| ETTm1 | TimesNet | 96 | 96 | Y | 0.3326 | 0.3326 | +0.0000 | +0.00 |
| ETTm1 | iTransformer | 336 | 336 | Y | 0.3039 | 0.3039 | +0.0000 | +0.00 |
| ETTm2 | DLinear | 1440 | 2880 | n | 0.1621 | 0.1619 | +0.0002 | +0.09 |
| ETTm2 | PatchTST | 336 | 336 | Y | 0.1801 | 0.1801 | +0.0000 | +0.00 |
| ETTm2 | TimesNet | 96 | 96 | Y | 0.1862 | 0.1862 | +0.0000 | +0.00 |
| ETTm2 | iTransformer | 720 | 336 | n | 0.1788 | 0.1737 | +0.0051 | +2.95 |
| electricity | DLinear | 2880 | 2880 | Y | 0.1291 | 0.1291 | +0.0000 | +0.00 |
| electricity | PatchTST | 1440 | 1440 | Y | 0.1290 | 0.1290 | +0.0000 | +0.00 |
| electricity | TimesNet | 96 | 96 | Y | 0.1676 | 0.1676 | +0.0000 | +0.00 |
| electricity | iTransformer | 1440 | 2880 | n | 0.1309 | 0.1308 | +0.0001 | +0.07 |
| exchange_rate | DLinear | 96 | 96 | Y | 0.1078 | 0.1078 | +0.0000 | +0.00 |
| exchange_rate | PatchTST | 96 | 96 | Y | 0.0871 | 0.0871 | +0.0000 | +0.00 |
| exchange_rate | TimesNet | 96 | 96 | Y | 0.1122 | 0.1122 | +0.0000 | +0.00 |
| exchange_rate | iTransformer | 96 | 96 | Y | 0.0889 | 0.0889 | +0.0000 | +0.00 |
| traffic | DLinear | 2880 | 2880 | Y | 0.3781 | 0.3781 | +0.0000 | +0.00 |
| traffic | PatchTST | 2880 | 2880 | Y | 0.3625 | 0.3625 | +0.0000 | +0.00 |
| traffic | TimesNet | 336 | 96 | n | 0.6024 | 0.5900 | +0.0124 | +2.10 |
| traffic | iTransformer | 720 | 2880 | n | 0.3580 | 0.3408 | +0.0172 | +5.06 |
| weather | DLinear | 1440 | 2880 | n | 0.1677 | 0.1673 | +0.0004 | +0.24 |
| weather | PatchTST | 1440 | 720 | n | 0.1569 | 0.1539 | +0.0031 | +1.99 |
| weather | TimesNet | 336 | 336 | Y | 0.1664 | 0.1664 | +0.0000 | +0.00 |
| weather | iTransformer | 336 | 336 | Y | 0.1632 | 0.1632 | +0.0000 | +0.00 |

mean-based: exact-hit 20/32, mean regret 0.0027, mean rel 0.86%, median rel 0.00%
per-seed : mean regret 0.0036 (per-seed selection, n rows with per-seed data: 32)
