#!/usr/bin/env python
"""E2-lite verdict: iTransformer pred_len=192 lookback ranking vs H=96 ranking.

Parses logs/ehs_fix/e2lite/<ds>_iTransformer_pl192_sl<L>_s<seed>.log,
builds mean test MSE per (dataset, L), and compares the best-L ranking with
the H=96 iTransformer row of logs/ehs_v2/SUMMARY.md (hardcoded below from the
SUMMARY table, pred_len=96, equal-budget protocol).
Outputs logs/ehs_fix/e2lite_verdict.json + .md
"""
import json
import os
import re

SLS = [96, 336, 1440]
# H=96 iTransformer (SUMMARY.md, mean test MSE): dataset -> {sl: mse} for sl in grid
H96 = {
    'ETTh1': {96: 0.3993, 336: 0.4015, 720: 0.3915, 1440: 0.3997, 2880: 0.4527},
    'exchange_rate': {96: 0.0889, 336: 0.1096, 720: 0.1234, 1440: 0.1613, 2880: 0.3211},
    'electricity': {96: 0.1486, 336: 0.1329, 720: 0.1340, 1440: 0.1312, 2880: 0.1307},
}
re_mse = re.compile(r'^mse:([\d.eE+-]+), mae:([\d.eE+-]+)', re.M)


def main():
    data = {}
    for ds in ['ETTh1', 'exchange_rate', 'electricity']:
        data[ds] = {}
        for sl in SLS:
            mses = []
            for seed in [2021, 2022]:
                log = f'logs/ehs_fix/e2lite/{ds}_iTransformer_pl192_sl{sl}_s{seed}.log'
                if not os.path.exists(log):
                    continue
                m = re_mse.findall(open(log, errors='ignore').read())
                if m:
                    mses.append(float(m[-1][0]))
            if mses:
                mu = sum(mses) / len(mses)
                sd = (sum((x - mu) ** 2 for x in mses) / max(1, len(mses) - 1)) ** 0.5 if len(mses) > 1 else 0.0
                data[ds][sl] = (mu, sd, len(mses))

    rows = []
    for ds, arms in data.items():
        if not arms:
            continue
        order192 = sorted(arms, key=lambda s: arms[s][0])
        order96 = sorted(H96[ds], key=lambda s: H96[ds][s])
        rows.append({
            'dataset': ds,
            'mse192': {str(sl): f'{arms[sl][0]:.4f}±{arms[sl][1]:.4f}[n={arms[sl][2]}]' for sl in arms},
            'best192': order192[0], 'rank192': order192,
            'best96': order96[0], 'rank96_full': order96,
            'rank96_restricted': [s for s in order96 if s in SLS],
            'rank_match_restricted': order192 == [s for s in order96 if s in SLS],
        })

    with open('logs/ehs_fix/e2lite_verdict.json', 'w') as f:
        json.dump(rows, f, indent=2)
    lines = ['| dataset | mse L=96 (H=192) | mse L=336 (H=192) | mse L=1440 (H=192) | best L (H=192) | best L (H=96, restricted grid) | ranking match |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['mse192'].get('96', '-')} | {r['mse192'].get('336', '-')} | "
                     f"{r['mse192'].get('1440', '-')} | {r['best192']} | {r['rank96_restricted'][0]} | "
                     f"{'Y' if r['rank_match_restricted'] else 'n'} |")
    with open('logs/ehs_fix/e2lite_verdict.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    for r in rows:
        print(r['dataset'], 'rank192:', r['rank192'], '| rank96 restricted:', r['rank96_restricted'],
              '| rank96 full:', r['rank96_full'])


if __name__ == '__main__':
    main()
