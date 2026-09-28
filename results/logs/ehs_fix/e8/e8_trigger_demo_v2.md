# E8 trigger demo (synthetic regime-switch)

mu_law=0.997822 (N*=459 from the paper power law, c=65.63, lam=1.5/512); AR(K=48)+intercept EWRLS, ridge=1.0; val seed 777 (n=1024, threshold tuning), test seed 999 (n=2048, held out).
Hit window [-16,+48], refractory 16; target grid [88,512).

## Detection (test; 3-path CUSUM hf=5 ks=0.15 hs=1e+09 hv=1e+09 frozen from val F1=0.454)

- precision=0.528, recall=0.399, F1=0.454 (tp=1248, fp=1125, fn=1816; triggers=2373, shocks=3064)
- delay: mean=13.6 steps, median=9.5
- per-type recall: level=0.499 (n=721), vol=0.763 (n=760), trend=0.082 (n=783), freq=0.305 (n=800)
- false alarms per shock-free series: 0.477 (n=511)

- fast-only CUSUM ablation: P=0.528 R=0.399 F1=0.454; per-type recall: level=0.499, vol=0.763, trend=0.082, freq=0.305
- pointwise-surprise baseline: P=0.496 R=0.335 F1=0.400; per-type recall: level=0.347, vol=0.828, trend=0.078, freq=0.107

## Prediction MSE (per-series mean then series mean)

### 1-step

| arm | overall | post-shock | calm | stable |
|---|---|---|---|---|
| A_const_mu_law | 0.4298 | 0.6345 | 0.3004 | 0.3630 |
| B_const_mu0.99 | 0.4772 | 0.7159 | 0.3152 | 0.3922 |
| C_const_mu1.0 | 0.4206 | 0.6181 | 0.2999 | 0.3581 |
| D_trigger_reset | 0.5456 | 0.7524 | 0.3235 | 0.4535 |
| E_oracle_reset | 0.6647 | 1.1582 | 0.3004 | 0.3980 |
| F_oracle_soft0.1 | 0.5489 | 0.9053 | 0.3004 | 0.3877 |
| G_trigger_soft0.1 | 0.5004 | 0.7129 | 0.3079 | 0.4108 |

- trigger vs const(mu_law): overall -26.94%, post-shock -18.58%, calm cost +7.66%
- oracle(hard) vs const(mu_law): overall -54.65%, post-shock -82.53%
- trigger overall 0.5456 vs best constant-arm overall 0.4206
- oracle(soft0.1) vs const: overall -27.72%, post-shock -42.67%
- trigger(soft0.1) vs const: overall -16.43%, post-shock -12.34%

### 24-step direct

| arm | overall | post-shock | calm | stable |
|---|---|---|---|---|
| A_const_mu_law | 1.1967 | 1.4818 | 0.8574 | 0.9869 |
| B_const_mu0.99 | 1.3082 | 1.5902 | 0.8969 | 1.0670 |
| C_const_mu1.0 | 1.1819 | 1.4663 | 0.8562 | 0.9782 |
| D_trigger_reset | 1.5779 | 1.7480 | 0.9510 | 1.3189 |
| E_oracle_reset | 1.8973 | 1.8721 | 0.8574 | 1.3343 |
| F_oracle_soft0.1 | 1.5227 | 1.6878 | 0.8574 | 1.1655 |
| G_trigger_soft0.1 | 1.4131 | 1.6643 | 0.8765 | 1.1861 |

- trigger vs const(mu_law): overall -31.86%, post-shock -17.97%, calm cost +10.91%
- oracle(hard) vs const(mu_law): overall -58.55%, post-shock -26.35%
- trigger overall 1.5779 vs best constant-arm overall 1.1819
- oracle(soft0.1) vs const: overall -27.24%, post-shock -13.90%
- trigger(soft0.1) vs const: overall -18.09%, post-shock -12.32%

