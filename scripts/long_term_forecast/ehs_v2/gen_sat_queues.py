#!/usr/bin/env python
"""Generate per-GPU serial queues for the 24 missing saturation arms
(sl in {2880,5760}, seeds 2021-2023, electricity/traffic x iTransformer/DLinear).

Each queue is pinned to one GPU (<=1 extra job per GPU, coexisting with the
MECH jobs of agent-4). Longest jobs (traffic, sl=5760, iTransformer) are dealt
first round-robin so queue makespans stay balanced. A command is skipped at
run time if its log already contains an 'mse:' line.
"""
import re
import sys

SRC = 'scripts/long_term_forecast/ehs_v2/remaining_cmds.txt'
GPUS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['0', '1', '2', '3'])]

cmds = []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.search(r'LBsat_(\w+?)_(iTransformer|DLinear)_(\d+)_s(\d+)', line)
        if m and m.group(3) in ('2880', '5760'):
            cmds.append(line)

assert len(cmds) == 24, f'expected 24 sat arms, got {len(cmds)}'


def cost_key(c):
    m = re.search(r'LBsat_(\w+?)_(\w+?)_(\d+)_s(\d+)', c)
    ds, model, sl = m.group(1), m.group(2), int(m.group(3))
    return ((ds == 'traffic'), sl, model == 'iTransformer')


cmds.sort(key=cost_key, reverse=True)

queues = [[] for _ in GPUS]
for i, c in enumerate(cmds):
    queues[i % len(GPUS)].append(c)

for gpu, q in zip(GPUS, queues):
    path = f'scripts/long_term_forecast/ehs_v2/sat_queue_gpu{gpu}.sh'
    with open(path, 'w') as f:
        f.write('#!/bin/bash\nset -u\ncd .\n')
        for c in q:
            log = re.search(r'> (\S+\.log)', c).group(1)
            c2 = re.sub(r'CUDA_VISIBLE_DEVICES=\d+', f'CUDA_VISIBLE_DEVICES={gpu}', c)
            f.write(f'if grep -q "mse:" "{log}" 2>/dev/null; then echo "SKIP {log}"; else {c2}; fi\n')
    print(f'{path}: {len(q)} jobs')
