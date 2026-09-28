"""Task A verdict: collect logs/ehs_v2/sat results and decide whether the
2880 -> 5760 extension is saturated for electricity/traffic.

Reads every sat log, aggregates test MSE over seeds per (dataset, model, sl),
then compares sl=2880 vs sl=5760 (same N_min@5760 protocol):
  saturated   := mse@5760 >= mse@2880 (no gain from doubling)
Outputs logs/ehs_final/taskA_sat.json and prints a table.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.collect_ehs import last_metric

SAT_DIR = 'logs/ehs_v2/sat'
NAME_RE = re.compile(r'(.+?)_(iTransformer|DLinear)_sl(\d+)_s(\d+)\.log$')
OUT = 'logs/ehs_final/taskA_sat.json'


def main():
    rec = {}
    for fn in sorted(os.listdir(SAT_DIR)):
        m = NAME_RE.match(fn)
        if not m:
            continue
        ds, model, sl, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        v = last_metric(os.path.join(SAT_DIR, fn))
        if v is not None:
            rec[(ds, model, sl, seed)] = v[0]

    table = {}
    for ds in sorted({k[0] for k in rec}):
        for model in sorted({k[1] for k in rec if k[0] == ds}):
            for sl in sorted({k[2] for k in rec if k[0] == ds and k[1] == model}):
                vals = [v for (d, mo, s, sd), v in rec.items()
                        if d == ds and mo == model and s == sl]
                table.setdefault(ds, {}).setdefault(model, {})[sl] = {
                    'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}

    lines = []
    verdict = {}
    for ds, models in table.items():
        for model, sls in models.items():
            row = '  '.join(f"sl{sl}={c['mean']:.4f}±{c['std']:.4f}[n={c['n']}]"
                            for sl, c in sorted(sls.items()))
            lines.append(f'{ds:12} {model:12} {row}')
            if 2880 in sls and 5760 in sls:
                d = sls[5760]['mean'] - sls[2880]['mean']
                sat = d >= 0
                verdict[f'{ds}/{model}'] = {
                    'mse_2880': sls[2880], 'mse_5760': sls[5760],
                    'delta_5760_minus_2880': float(d), 'saturated': bool(sat)}
                lines.append(f'    -> 5760-2880 delta={d:+.4f} -> '
                             f'{"SATURATED (no gain beyond 2880)" if sat else "NOT saturated (5760 still improves)"}')
            else:
                verdict[f'{ds}/{model}'] = {'incomplete': True,
                                            'have': sorted(sls)}
                lines.append('    -> incomplete (missing 2880 or 5760)')
    for l in lines:
        print(l)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'table': table, 'verdict': verdict}, f, indent=1)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
