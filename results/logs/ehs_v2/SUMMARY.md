# EHS v2: confounder-controlled lookback (all arms use --max_train_windows = N_min of the sl=2880 arm)

Cells: mean±std test MSE over seeds {2021,2022,2023}; **bold** = best seq_len of the row.
`[n=k]` marks cells computed from fewer than 3 finished seeds (time-boxed cut).

### Model: iTransformer (MSE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | 0.3993±0.0007 | 0.4015±0.0008 | **0.3915±0.0023** | 0.4003±0.0023 | 0.4569±0.0072 | 720 |
| ETTh2 | **0.2988±0.0000** | 0.3020±0.0027 | 0.3049±0.0039 | 0.3250±0.0032 | 0.4454±0.0155 | 96 |
| ETTm1 | 0.3424±0.0006 | **0.3039±0.0011** | 0.3130±0.0013 | 0.3363±0.0043 | 0.3414±0.0007 | 336 |
| ETTm2 | 0.1853±0.0008 | **0.1737±0.0009** | 0.1788±0.0014 | 0.1873±0.0041 | 0.1912±0.0005 | 336 |
| electricity | 0.1487±0.0001 | 0.1329±0.0009 | 0.1343±0.0017 | 0.1309±0.0005 | **0.1308±0.0006** | 2880 |
| exchange_rate | **0.0889±0.0001** | 0.1096±0.0012 | 0.1234±0.0015 | 0.1664±0.0076 | 0.3162±0.0120 | 96 |
| traffic | 0.4063±0.0008 | 0.3646±0.0005 | 0.3580±0.0007 | 0.3481±0.0005 | **0.3408±0.0004** | 2880 |
| weather | 0.1751±0.0004 | **0.1632±0.0020** | 0.1771±0.0062 | 0.1953±0.0030 | 0.2248±0.0007 | 336 |

### Model: DLinear (MSE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | 0.4054±0.0009 | 0.3801±0.0003 | 0.3694±0.0003 | **0.3686±0.0002** | 0.3907±0.0017 | 1440 |
| ETTh2 | 0.3572±0.0035 | 0.3144±0.0017 | **0.3081±0.0043** | 0.3132±0.0109 | 0.4530±0.0429 | 720 |
| ETTm1 | 0.3457±0.0002 | **0.3003±0.0003** | 0.3096±0.0023 | 0.3083±0.0003 | 0.3104±0.0002 | 336 |
| ETTm2 | 0.1919±0.0019 | 0.1694±0.0011 | 0.1634±0.0003 | 0.1621±0.0016 | **0.1619±0.0010** | 2880 |
| electricity | 0.1948±0.0000 | 0.1401±0.0001 | 0.1329±0.0000 | 0.1302±0.0000 | **0.1291±0.0000** | 2880 |
| exchange_rate | **0.1078±0.0021** | 0.1172±0.0014 | 0.1419±0.0048 | 0.1687±0.0216 | 0.6322±0.0101 | 96 |
| traffic | 0.6501±0.0001 | 0.4106±0.0000 | 0.3853±0.0001 | 0.3805±0.0000 | **0.3781±0.0001** | 2880 |
| weather | 0.1961±0.0006 | 0.1753±0.0006 | 0.1695±0.0009 | 0.1677±0.0002 | **0.1673±0.0013** | 2880 |

### Model: PatchTST (MSE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | 0.3862±0.0028 | **0.3849±0.0046** | 0.4017±0.0045 | 0.4567±0.0164 | 0.6228±0.0274 | 336 |
| ETTh2 | **0.2939±0.0016** | 0.3137±0.0057 | 0.3183±0.0045 | 0.3498±0.0179 | 0.5066±0.0524 | 96 |
| ETTm1 | 0.3266±0.0047 | **0.2911±0.0014** | 0.3049±0.0044 | 0.3389±0.0046 | 0.3880±0.0156 | 336 |
| ETTm2 | 0.1838±0.0027 | **0.1801±0.0015** | 0.1851±0.0017 | 0.1911±0.0022 | 0.1934±0.0023 | 336 |
| electricity | 0.1816±0.0001 | 0.1371±0.0007 | 0.1352±0.0007 | **0.1290±0.0007** | 0.1416±0.0088 | 1440 |
| exchange_rate | **0.0871±0.0020** | 0.1048±0.0125 | 0.0924±0.0030 | 0.1385±0.0240 | 0.3804±0.0378 | 96 |
| traffic | 0.4680±0.0003 | 0.3809±0.0008 | 0.3664±0.0007 | 0.3628±0.0004 | **0.3625±0.0024** | 2880 |
| weather | 0.1739±0.0006 | 0.1548±0.0020 | **0.1539±0.0023** | 0.1569±0.0046 | 0.1817±0.0079 | 720 |

### Model: TimesNet (MSE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | **0.4173±0.0164** | 0.4261±0.0045 | 0.4356±0.0102 | 0.5736±0.0592 | 1.0904±0.1031 | 96 |
| ETTh2 | **0.3337±0.0022** | 0.3741±0.0250 | 0.3862±0.0072 | 0.3948±0.0191 | 0.9686±0.1439 | 96 |
| ETTm1 | **0.3326±0.0030** | 0.3459±0.0152 | 0.3744±0.0359 | 0.3471±0.0107 | 0.4293±0.0129 | 96 |
| ETTm2 | **0.1862±0.0015** | 0.1879±0.0040 | 0.2130±0.0087 | 0.2392±0.0068 | 0.2116±0.0191 | 96 |
| electricity | **0.1676±0.0011** | 0.1821±0.0027 | 0.1848±0.0037 | 0.1974±0.0065 | 0.2178±0.0037 | 96 |
| exchange_rate | **0.1122±0.0022** | 0.1643±0.0033 | 0.2919±0.0173 | 0.6894±0.2596 | 2.0699±0.3994 | 96 |
| traffic | **0.5900±0.0046** | 0.6024±0.0066 | 0.6172±0.0042 | 0.6167±0.0033 | 0.6586±0.0060 | 96 |
| weather | 0.1750±0.0034 | **0.1664±0.0035** | 0.1737±0.0037 | 0.1840±0.0033 | 0.1918±0.0047 | 336 |

### Best seq_len summary (by mean test MSE)

| dataset | iTransformer | DLinear | PatchTST | TimesNet |
|---|---|---|---|---|
| ETTh1 | 720 | 1440 | 336 | 96 |
| ETTh2 | 96 | 720 | 96 | 96 |
| ETTm1 | 336 | 336 | 336 | 96 |
| ETTm2 | 336 | 2880 | 336 | 96 |
| electricity | 2880 | 2880 | 1440 | 96 |
| exchange_rate | 96 | 96 | 96 | 96 |
| traffic | 2880 | 2880 | 2880 | 96 |
| weather | 336 | 2880 | 720 | 336 |

### Model: iTransformer (MAE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | **0.4131±0.0005** | 0.4205±0.0007 | 0.4233±0.0022 | 0.4315±0.0011 | 0.4708±0.0040 | 96 |
| ETTh2 | **0.3491±0.0004** | 0.3591±0.0016 | 0.3614±0.0025 | 0.3841±0.0021 | 0.4799±0.0094 | 96 |
| ETTm1 | 0.3769±0.0004 | **0.3576±0.0007** | 0.3673±0.0013 | 0.3804±0.0005 | 0.3887±0.0012 | 336 |
| ETTm2 | 0.2718±0.0014 | **0.2665±0.0016** | 0.2734±0.0012 | 0.2819±0.0035 | 0.2884±0.0007 | 336 |
| electricity | 0.2404±0.0002 | 0.2290±0.0010 | 0.2311±0.0019 | **0.2273±0.0006** | 0.2288±0.0003 | 1440 |
| exchange_rate | **0.2100±0.0001** | 0.2369±0.0014 | 0.2568±0.0017 | 0.3022±0.0071 | 0.4342±0.0107 | 96 |
| traffic | 0.2774±0.0005 | 0.2642±0.0004 | 0.2620±0.0002 | 0.2587±0.0002 | **0.2561±0.0002** | 2880 |
| weather | 0.2158±0.0002 | **0.2127±0.0022** | 0.2269±0.0044 | 0.2493±0.0034 | 0.2796±0.0011 | 336 |

### Model: DLinear (MAE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | 0.4188±0.0006 | 0.4035±0.0003 | **0.4003±0.0003** | 0.4009±0.0002 | 0.4246±0.0023 | 720 |
| ETTh2 | 0.4086±0.0022 | 0.3762±0.0013 | **0.3722±0.0040** | 0.3771±0.0083 | 0.4704±0.0260 | 720 |
| ETTm1 | 0.3722±0.0001 | **0.3445±0.0010** | 0.3528±0.0024 | 0.3541±0.0007 | 0.3579±0.0004 | 336 |
| ETTm2 | 0.2900±0.0029 | 0.2652±0.0015 | **0.2575±0.0011** | 0.2581±0.0015 | 0.2590±0.0013 | 720 |
| electricity | 0.2780±0.0002 | 0.2375±0.0002 | 0.2299±0.0001 | 0.2272±0.0001 | **0.2270±0.0001** | 2880 |
| exchange_rate | **0.2465±0.0025** | 0.2605±0.0021 | 0.2906±0.0058 | 0.3170±0.0218 | 0.6385±0.0032 | 96 |
| traffic | 0.3983±0.0001 | 0.2836±0.0000 | 0.2701±0.0000 | 0.2680±0.0001 | **0.2674±0.0002** | 2880 |
| weather | 0.2561±0.0017 | 0.2371±0.0007 | 0.2306±0.0024 | **0.2275±0.0014** | 0.2287±0.0013 | 1440 |

### Model: PatchTST (MAE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | **0.4010±0.0013** | 0.4071±0.0036 | 0.4208±0.0026 | 0.4700±0.0122 | 0.5725±0.0142 | 96 |
| ETTh2 | **0.3454±0.0014** | 0.3650±0.0025 | 0.3782±0.0021 | 0.4023±0.0098 | 0.5101±0.0263 | 96 |
| ETTm1 | 0.3660±0.0022 | **0.3443±0.0009** | 0.3608±0.0027 | 0.3847±0.0031 | 0.4171±0.0092 | 336 |
| ETTm2 | **0.2658±0.0015** | 0.2726±0.0005 | 0.2756±0.0006 | 0.2838±0.0019 | 0.2895±0.0019 | 96 |
| electricity | 0.2739±0.0002 | 0.2400±0.0009 | 0.2372±0.0017 | **0.2255±0.0027** | 0.2472±0.0156 | 1440 |
| exchange_rate | **0.2056±0.0022** | 0.2315±0.0136 | 0.2164±0.0037 | 0.2721±0.0194 | 0.4641±0.0212 | 96 |
| traffic | 0.3036±0.0006 | 0.2715±0.0017 | **0.2591±0.0019** | 0.2601±0.0019 | 0.2598±0.0026 | 720 |
| weather | 0.2152±0.0002 | **0.2032±0.0022** | 0.2085±0.0022 | 0.2176±0.0052 | 0.2439±0.0107 | 336 |

### Model: TimesNet (MAE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | best sl |
|---|---|---|---|---|---|---|
| ETTh1 | **0.4309±0.0105** | 0.4412±0.0019 | 0.4548±0.0059 | 0.5458±0.0392 | 0.8141±0.0529 | 96 |
| ETTh2 | **0.3716±0.0014** | 0.4169±0.0172 | 0.4197±0.0043 | 0.4349±0.0086 | 0.6950±0.0525 | 96 |
| ETTm1 | **0.3738±0.0014** | 0.3794±0.0086 | 0.3964±0.0136 | 0.3881±0.0037 | 0.4309±0.0045 | 96 |
| ETTm2 | **0.2660±0.0020** | 0.2743±0.0026 | 0.2978±0.0061 | 0.3256±0.0061 | 0.3041±0.0158 | 96 |
| electricity | **0.2715±0.0011** | 0.2862±0.0017 | 0.2889±0.0032 | 0.3005±0.0045 | 0.3186±0.0023 | 96 |
| exchange_rate | **0.2420±0.0012** | 0.3035±0.0017 | 0.4088±0.0110 | 0.6005±0.0826 | 1.0340±0.0642 | 96 |
| traffic | **0.3179±0.0024** | 0.3219±0.0005 | 0.3304±0.0075 | 0.3279±0.0035 | 0.3519±0.0069 | 96 |
| weather | 0.2240±0.0022 | **0.2234±0.0053** | 0.2317±0.0045 | 0.2471±0.0037 | 0.2610±0.0038 | 336 |

## Saturation probe (electricity/traffic, N_min recomputed at sl=5760; MSE)

### Model: iTransformer (MSE, N_min@5760)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | sl=5760 | best sl |
|---|---|---|---|---|---|---|---|
| electricity | 0.1506±0.0000[n=1] | 0.1329±0.0000[n=1] | 0.1328±0.0000[n=1] | **0.1289±0.0000[n=1]** | 0.1302±0.0012 | 0.1323±0.0006 | 1440 |
| traffic | 0.4127±0.0000[n=1] | 0.3680±0.0000[n=1] | 0.3589±0.0000[n=1] | 0.3554±0.0000[n=1] | **0.3481±0.0019** | 0.3608±0.0006 | 2880 |

### Model: DLinear (MSE, N_min@5760)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 | sl=5760 | best sl |
|---|---|---|---|---|---|---|---|
| electricity | 0.1950±0.0000[n=1] | 0.1401±0.0000[n=1] | 0.1330±0.0000[n=1] | 0.1303±0.0000[n=1] | **0.1291±0.0001** | 0.1294±0.0001 | 2880 |
| traffic | 0.6520±0.0000[n=1] | 0.4114±0.0000[n=1] | 0.3857±0.0000[n=1] | 0.3812±0.0000[n=1] | **0.3787±0.0001** | 0.3798±0.0000 | 2880 |

## TSFM zero-shot (Chronos bolt-base, single run; MSE / MAE)

| dataset | sl=96 | sl=336 | sl=720 | sl=1440 | sl=2880 |
|---|---|---|---|---|---|
| ETTh1 | 0.5387/0.4114 | 0.3890/0.3869 | 0.3691/0.3787 | 0.3684/0.3764 | - |
| ETTh2 | 0.3825/0.3604 | 0.2856/0.3341 | 0.2865/0.3309 | 0.2745/0.3200 | - |
| ETTm1 | 1.0144/0.5723 | 0.3395/0.3427 | 0.3208/0.3337 | 0.3112/0.3271 | - |
| ETTm2 | 0.3031/0.3241 | 0.1828/0.2551 | 0.1748/0.2484 | 0.1640/0.2410 | - |
| electricity | - | - | - | - | - |
| exchange_rate | 0.0901/0.2058 | 0.1080/0.2263 | 0.1019/0.2239 | 0.0962/0.2167 | - |
| traffic | - | - | - | - | - |
| weather | 0.6508/0.2939 | 0.1813/0.2231 | 0.1691/0.2096 | 0.1582/0.1977 | - |

## Missing/failed arms

- sat: electricity iTransformer sl=96 seed=2022 — timeout-cut (killed mid-run)
- sat: electricity iTransformer sl=96 seed=2023 — timeout-cut (never started)
- sat: electricity iTransformer sl=336 seed=2022 — timeout-cut (killed mid-run)
- sat: electricity iTransformer sl=336 seed=2023 — timeout-cut (never started)
- sat: electricity iTransformer sl=720 seed=2022 — timeout-cut (never started)
- sat: electricity iTransformer sl=720 seed=2023 — timeout-cut (never started)
- sat: electricity iTransformer sl=1440 seed=2022 — timeout-cut (never started)
- sat: electricity iTransformer sl=1440 seed=2023 — timeout-cut (never started)
- sat: electricity DLinear sl=96 seed=2022 — timeout-cut (killed mid-run)
- sat: electricity DLinear sl=96 seed=2023 — timeout-cut (never started)
- sat: electricity DLinear sl=336 seed=2022 — timeout-cut (killed mid-run)
- sat: electricity DLinear sl=336 seed=2023 — timeout-cut (never started)
- sat: electricity DLinear sl=720 seed=2022 — timeout-cut (never started)
- sat: electricity DLinear sl=720 seed=2023 — timeout-cut (never started)
- sat: electricity DLinear sl=1440 seed=2022 — timeout-cut (never started)
- sat: electricity DLinear sl=1440 seed=2023 — timeout-cut (never started)
- sat: traffic iTransformer sl=96 seed=2022 — timeout-cut (killed mid-run)
- sat: traffic iTransformer sl=96 seed=2023 — timeout-cut (never started)
- sat: traffic iTransformer sl=336 seed=2022 — timeout-cut (killed mid-run)
- sat: traffic iTransformer sl=336 seed=2023 — timeout-cut (never started)
- sat: traffic iTransformer sl=720 seed=2022 — timeout-cut (never started)
- sat: traffic iTransformer sl=720 seed=2023 — timeout-cut (never started)
- sat: traffic iTransformer sl=1440 seed=2022 — timeout-cut (never started)
- sat: traffic iTransformer sl=1440 seed=2023 — timeout-cut (never started)
- sat: traffic DLinear sl=96 seed=2022 — timeout-cut (killed mid-run)
- sat: traffic DLinear sl=96 seed=2023 — timeout-cut (never started)
- sat: traffic DLinear sl=336 seed=2022 — timeout-cut (killed mid-run)
- sat: traffic DLinear sl=336 seed=2023 — timeout-cut (never started)
- sat: traffic DLinear sl=720 seed=2022 — timeout-cut (never started)
- sat: traffic DLinear sl=720 seed=2023 — timeout-cut (never started)
- sat: traffic DLinear sl=1440 seed=2022 — timeout-cut (never started)
- sat: traffic DLinear sl=1440 seed=2023 — timeout-cut (never started)
- tsfm: ETTh1 Chronos sl=2880 — timeout-cut (never started)
- tsfm: ETTh2 Chronos sl=2880 — timeout-cut (never started)
- tsfm: ETTm1 Chronos sl=2880 — timeout-cut (never started)
- tsfm: ETTm2 Chronos sl=2880 — timeout-cut (never started)
- tsfm: electricity Chronos sl=96 — timeout-cut (killed mid-run)
- tsfm: electricity Chronos sl=336 — timeout-cut (killed mid-run)
- tsfm: electricity Chronos sl=720 — timeout-cut (killed mid-run)
- tsfm: electricity Chronos sl=1440 — timeout-cut (killed mid-run)
- tsfm: electricity Chronos sl=2880 — timeout-cut (never started)
- tsfm: exchange_rate Chronos sl=2880 — timeout-cut (never started)
- tsfm: traffic Chronos sl=96 — timeout-cut (killed mid-run)
- tsfm: traffic Chronos sl=336 — timeout-cut (killed mid-run)
- tsfm: traffic Chronos sl=720 — timeout-cut (killed mid-run)
- tsfm: traffic Chronos sl=1440 — timeout-cut (killed mid-run)
- tsfm: traffic Chronos sl=2880 — timeout-cut (never started)
- tsfm: weather Chronos sl=2880 — timeout-cut (never started)
