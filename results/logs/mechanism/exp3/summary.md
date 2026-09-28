# Exp3: PatchTSTGated repair ablation (uncapped windows; pred_len=96)

MSE mean±std over seeds {2021,2022}. `none` = PatchTSTGated vanilla (SDPA).

## exchange_rate

| seq_len | none (ctrl) | denbias | recmask | ehs_v2 PatchTST (capped) |
|---|---|---|---|---|
| 96 | 0.0950±0.0009[n=2] | 0.0952±0.0010[n=2] | 0.0950±0.0009[n=2] | 0.0871±0.0020 |
| 1440 | 0.1371±0.0003[n=2] | 0.1387±0.0030[n=2] | 0.1282±0.0132[n=2] | 0.1484±0.0239 |
| 2880 | 0.3989±0.0168[n=2] | 0.3989±0.0181[n=2] | 0.2625±0.0424[n=2] | 0.4053±0.0169 |

## electricity

| seq_len | none (ctrl) | denbias | recmask | ehs_v2 PatchTST (capped) |
|---|---|---|---|---|
| 96 | 0.1803±0.0002[n=2] | 0.1803±0.0002[n=2] | 0.1803±0.0002[n=2] | 0.1816±0.0001 |
| 1440 | 0.1291±0.0003[n=2] | 0.1285±0.0003[n=2] | 0.1291±0.0003[n=2] | - |
| 2880 | 0.1301±0.0009[n=2] | 0.1384±0.0089[n=2] | 0.1269±0.0000[n=2] | - |


## denbias learned null-logit b and null mass

- exchange_rate sl=96 s2021: L0: mean b=0.00, L1: mean b=0.00; null mass: L0: 0.050, L1: 0.062
- exchange_rate sl=96 s2022: L0: mean b=0.00, L1: mean b=0.00; null mass: L0: 0.059, L1: 0.099
- exchange_rate sl=1440 s2021: L0: mean b=-0.00, L1: mean b=-0.00; null mass: L0: 0.003, L1: 0.003
- exchange_rate sl=1440 s2022: L0: mean b=-0.00, L1: mean b=-0.00; null mass: L0: 0.004, L1: 0.004
- exchange_rate sl=2880 s2021: L0: mean b=-0.00, L1: mean b=-0.00; null mass: L0: 0.002, L1: 0.001
- exchange_rate sl=2880 s2022: L0: mean b=-0.00, L1: mean b=-0.00; null mass: L0: 0.002, L1: 0.001
- electricity sl=96 s2021: L0: mean b=0.01, L1: mean b=0.03; null mass: L0: 0.113, L1: 0.060
- electricity sl=96 s2022: L0: mean b=0.02, L1: mean b=0.01; null mass: L0: 0.170, L1: 0.019
- electricity sl=1440 s2021: L0: mean b=0.00, L1: mean b=-0.05; null mass: L0: 0.001, L1: 0.122
- electricity sl=1440 s2022: L0: mean b=-0.00, L1: mean b=-0.01; null mass: L0: 0.001, L1: 0.125
- electricity sl=2880 s2021: L0: mean b=0.01, L1: mean b=-0.02; null mass: L0: 0.001, L1: 0.001
- electricity sl=2880 s2022: L0: mean b=0.00, L1: mean b=-0.01; null mass: L0: 0.001, L1: 0.003
