# E7.1 multiple-comparison correction of the exp4 BOCPD hazard scan

Exact permutation p (n=8, 8! enumeration, ties with multiplicity);
t-approx p as printed in logs/mechanism/exp4/bocpd_summary.md for continuity.

| H | model | rho | p_exact | p_tapprox | Holm (m=10) | Bonf (m=10) | Holm (m=5) | Bonf (m=5) |
|---|---|---|---|---|---|---|---|---|
| 0.02 | iTransformer | -0.667 | 0.0798 | 0.028 | 0.7179 | 0.7976 | 0.3988 | 0.3988 |
| 0.02 | DLinear | 0.000 | 1.0000 | 1.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.01 | iTransformer | -0.581 | 0.1405 | 0.081 | 1.0000 | 1.0000 | 0.5619 | 0.7024 |
| 0.01 | DLinear | 0.114 | 0.7857 | 0.778 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.004 | iTransformer | -0.457 | 0.2595 | 0.208 | 1.0000 | 1.0000 | 0.7786 | 1.0000 |
| 0.004 | DLinear | 0.292 | 0.4869 | 0.455 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.002 | iTransformer | -0.272 | 0.5071 | 0.489 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.002 | DLinear | 0.495 | 0.2119 | 0.163 | 1.0000 | 1.0000 | 0.8476 | 1.0000 |
| 0.001 | iTransformer | 0.111 | 0.8000 | 0.784 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.001 | DLinear | 0.723 | 0.0560 | 0.010 | 0.5595 | 0.5595 | 0.2798 | 0.2798 |

headline cell (iTransformer, H=1/50): rho=-0.667, p_exact=0.0798 (t-approx 0.028), Holm m=10 -> 0.7179, Bonf m=10 -> 0.7976, Holm m=5 -> 0.3988, Bonf m=5 -> 0.3988
significant at 0.05 after m=10 Holm: False; after m=5 Holm: False; uncorrected exact: False
