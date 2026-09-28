# E7.2 deseasonalized BOCPD re-run (E[r], train split, same protocol as exp4)

periods: ETTh 24, ETTm 96, exchange 7, weather 144, electricity/traffic 24.
phase = subtract per-phase means; diff = seasonal difference at the same lag.

## original

| dataset | H=0.02 | H=0.01 | H=0.004 | H=0.002 | H=0.001 | best sl iTr | best sl DLin |
|---|---|---|---|---|---|---|---|
| ETTh1 | 65 | 76 | 90 | 102 | 114 | 720 | 1440 |
| ETTh2 | 234 | 248 | 262 | 271 | 279 | 96 | 1440 |
| ETTm1 | 60 | 64 | 68 | 71 | 74 | 336 | 336 |
| ETTm2 | 363 | 377 | 390 | 399 | 407 | 336 | 1440 |
| exchange_rate | 210 | 215 | 219 | 221 | 223 | 96 | 96 |
| weather | 291 | 297 | 301 | 303 | 305 | 336 | 2880 |
| electricity | 53 | 68 | 122 | 234 | 372 | 2880 | 2880 |
| traffic | 48 | 59 | 87 | 146 | 238 | 2880 | 2880 |

- H=0.02 [original]: iTransformer rho=-0.667 (p_exact~0.080)  DLinear rho=-0.225 (p_exact~0.596)  
- H=0.01 [original]: iTransformer rho=-0.581 (p_exact~0.140)  DLinear rho=-0.100 (p_exact~0.845)  
- H=0.004 [original]: iTransformer rho=-0.457 (p_exact~0.260)  DLinear rho=0.100 (p_exact~0.845)  
- H=0.002 [original]: iTransformer rho=-0.272 (p_exact~0.507)  DLinear rho=0.325 (p_exact~0.436)  
- H=0.001 [original]: iTransformer rho=0.111 (p_exact~0.800)  DLinear rho=0.551 (p_exact~0.177)  

  electricity: E[r] rank among 8 (1=shortest) per hazard = [2, 3, 4, 5, 7], best sl = 2880/2880 (tied longest)
  traffic: E[r] rank among 8 (1=shortest) per hazard = [1, 1, 2, 3, 4], best sl = 2880/2880 (tied longest)

## phase

| dataset | H=0.02 | H=0.01 | H=0.004 | H=0.002 | H=0.001 | best sl iTr | best sl DLin |
|---|---|---|---|---|---|---|---|
| ETTh1 | 71 | 80 | 93 | 103 | 113 | 720 | 1440 |
| ETTh2 | 245 | 255 | 265 | 271 | 276 | 96 | 1440 |
| ETTm1 | 73 | 77 | 82 | 86 | 89 | 336 | 336 |
| ETTm2 | 427 | 440 | 453 | 461 | 468 | 336 | 1440 |
| exchange_rate | 210 | 215 | 219 | 221 | 223 | 96 | 96 |
| weather | 292 | 297 | 302 | 305 | 307 | 336 | 2880 |
| electricity | 88 | 103 | 124 | 140 | 155 | 2880 | 2880 |
| traffic | 50 | 57 | 67 | 76 | 84 | 2880 | 2880 |

- H=0.02 [phase]: iTransformer rho=-0.593 (p_exact~0.132)  DLinear rho=-0.075 (p_exact~0.877)  
- H=0.01 [phase]: iTransformer rho=-0.544 (p_exact~0.174)  DLinear rho=-0.025 (p_exact~0.980)  
- H=0.004 [phase]: iTransformer rho=-0.544 (p_exact~0.174)  DLinear rho=-0.025 (p_exact~0.980)  
- H=0.002 [phase]: iTransformer rho=-0.544 (p_exact~0.174)  DLinear rho=-0.025 (p_exact~0.980)  
- H=0.001 [phase]: iTransformer rho=-0.544 (p_exact~0.174)  DLinear rho=-0.025 (p_exact~0.980)  

  electricity: E[r] rank among 8 (1=shortest) per hazard = [4, 4, 4, 4, 4], best sl = 2880/2880 (tied longest)
  traffic: E[r] rank among 8 (1=shortest) per hazard = [1, 1, 1, 1, 1], best sl = 2880/2880 (tied longest)

## diff

| dataset | H=0.02 | H=0.01 | H=0.004 | H=0.002 | H=0.001 | best sl iTr | best sl DLin |
|---|---|---|---|---|---|---|---|
| ETTh1 | 22 | 26 | 31 | 36 | 43 | 720 | 1440 |
| ETTh2 | 75 | 86 | 102 | 116 | 130 | 96 | 1440 |
| ETTm1 | 35 | 39 | 42 | 45 | 47 | 336 | 336 |
| ETTm2 | 121 | 129 | 137 | 143 | 149 | 336 | 1440 |
| exchange_rate | 96 | 100 | 107 | 112 | 119 | 96 | 96 |
| weather | 287 | 292 | 296 | 299 | 300 | 336 | 2880 |
| electricity | 43 | 50 | 59 | 67 | 75 | 2880 | 2880 |
| traffic | 49 | 52 | 56 | 58 | 61 | 2880 | 2880 |

- H=0.02 [diff]: iTransformer rho=-0.469 (p_exact~0.246)  DLinear rho=0.125 (p_exact~0.782)  
- H=0.01 [diff]: iTransformer rho=-0.469 (p_exact~0.246)  DLinear rho=0.125 (p_exact~0.782)  
- H=0.004 [diff]: iTransformer rho=-0.469 (p_exact~0.246)  DLinear rho=0.125 (p_exact~0.782)  
- H=0.002 [diff]: iTransformer rho=-0.469 (p_exact~0.246)  DLinear rho=0.200 (p_exact~0.652)  
- H=0.001 [diff]: iTransformer rho=-0.469 (p_exact~0.246)  DLinear rho=0.200 (p_exact~0.652)  

  electricity: E[r] rank among 8 (1=shortest) per hazard = [3, 3, 4, 4, 4], best sl = 2880/2880 (tied longest)
  traffic: E[r] rank among 8 (1=shortest) per hazard = [4, 4, 3, 3, 3], best sl = 2880/2880 (tied longest)

