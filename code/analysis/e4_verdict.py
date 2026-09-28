#!/usr/bin/env python
"""E4 verdict: input-budget arm vs no-budget sl=2880 vs native sl=b (PatchTSTGated, attn none).

Sources:
  logs/ehs_fix/e4/<ds>_{b<budget>|nb}_sl<sl>_s<seed>.log      (new runs)
  logs/mechanism/exp3/<ds>_{none|denbias|recmask}_sl<sl>_s<seed>.log  (existing controls)
Outputs logs/ehs_fix/e4_verdict.json + .md
"""
import json
import os
import re
from collections import defaultdict

BUDGET = {'exchange_rate': 96, 'ETTh1': 720, 'electricity': 1440, 'weather': 336}
re_mse = re.compile(r'^mse:([\d.eE+-]+), mae:([\d.eE+-]+)', re.M)


def parse(log):
    with open(log, errors='ignore') as f:
        txt = f.read()
    m = re_mse.findall(txt)
    return (float(m[-1][0]), float(m[-1][1])) if m else (None, None)


def ms(d):
    xs = [v[0] for v in d.values()]
    if not xs:
        return None
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return mu, sd, len(xs)


def main():
    data = defaultdict(dict)  # (name, tag, sl) -> {seed: (mse, mae)}
    for fn in sorted(os.listdir('logs/ehs_fix/e4')):
        m = re.match(r'^([A-Za-z0-9_]+)_(b\d+|nb)_sl(\d+)_s(\d+)\.log$', fn)
        if not m:
            continue
        mse, mae = parse(os.path.join('logs/ehs_fix/e4', fn))
        if mse is not None:
            data[(m.group(1), m.group(2), int(m.group(3)))][int(m.group(4))] = (mse, mae)
    for fn in sorted(os.listdir('logs/mechanism/exp3')):
        m = re.match(r'^([A-Za-z0-9_]+)_(none|denbias|recmask)_sl(\d+)_s(\d+)\.log$', fn)
        if not m:
            continue
        mse, mae = parse(os.path.join('logs/mechanism/exp3', fn))
        if mse is not None:
            data[(m.group(1), m.group(2), int(m.group(3)))][int(m.group(4))] = (mse, mae)

    rows = []
    for name, b in BUDGET.items():
        def get(tag, sl):
            r = ms(data.get((name, tag, sl), {}))
            return r
        row = {
            'dataset': name, 'budget': b,
            'budget_arm_sl2880': get(f'b{b}', 2880),
            'no_budget_sl2880': get('nb', 2880) or get('none', 2880),
            'native_sl_b': get('nb', b) or get('none', b),
            'recmask_sl2880': get('recmask', 2880),
            'denbias_sl2880': get('denbias', 2880),
        }
        rows.append(row)

    with open('logs/ehs_fix/e4_verdict.json', 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    def fmt(r):
        if r is None:
            return '-'
        mu, sd, n = r
        return f'{mu:.4f}±{sd:.4f}[n={n}]'

    lines = ['| dataset | b | budget sl=2880 | no-budget sl=2880 | native sl=b | recmask sl=2880 | denbias sl=2880 |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['budget']} | {fmt(r['budget_arm_sl2880'])} | "
                     f"{fmt(r['no_budget_sl2880'])} | {fmt(r['native_sl_b'])} | "
                     f"{fmt(r['recmask_sl2880'])} | {fmt(r['denbias_sl2880'])} |")
    with open('logs/ehs_fix/e4_verdict.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
