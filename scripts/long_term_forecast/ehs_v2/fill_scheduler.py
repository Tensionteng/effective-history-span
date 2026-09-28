#!/usr/bin/env python
"""ehs_v2 E6 coverage-fill scheduler.

Runs a queue of training commands on GPUs 4-7, at most one job per GPU.
On CUDA OOM the job is retried with halved batch size (floor 2), each retry
overwriting the same log; all events go to the scheduler log.
Restart-safe: jobs whose log already contains a final 'mse:' line are skipped.

usage: .venv/bin/python scripts/long_term_forecast/ehs_v2/fill_scheduler.py \
         scripts/long_term_forecast/ehs_v2/fill_cmds.txt logs/ehs_v2/fill_scheduler.log
"""
import fcntl
import os
import re
import subprocess
import sys
import threading
import time

GPUS = [int(x) for x in os.environ.get('FILL_GPUS', '4,5,6,7').split(',')]
QFILE = sys.argv[1]
SCHED_LOG = sys.argv[2] if len(sys.argv) > 2 else 'logs/ehs_v2/fill_scheduler.log'

_lock = threading.Lock()


def log(msg):
    line = '%s %s' % (time.strftime('%F %T'), msg)
    with _lock:
        print(line, flush=True)
        with open(SCHED_LOG, 'a') as f:
            f.write(line + '\n')


def logpath(cmd):
    m = re.search(r'>\s*(\S+)', cmd)
    return m.group(1) if m else None


def status(logf):
    try:
        with open(logf, errors='ignore') as f:
            txt = f.read()
    except OSError:
        return 'norun'
    if re.search(r'^mse:', txt, re.M):
        return 'ok'
    if 'OutOfMemoryError' in txt or 'CUDA out of memory' in txt:
        return 'oom'
    return 'fail'


def halve_bs(cmd):
    m = re.search(r'--batch_size (\d+)', cmd)
    bs = int(m.group(1))
    if bs <= 2:
        return None
    return cmd.replace(m.group(0), '--batch_size %d' % (bs // 2))


def pop_next():
    """Atomically pop the first unfinished command from QFILE.

    Marks taken lines with '#TAKEN ' and finished ones with '#DONE_SKIPPED '
    so multiple scheduler processes (one per GPU) can share one queue file.
    """
    with open(QFILE + '.lock', 'a') as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        with open(QFILE) as f:
            lines = f.readlines()
        picked = None
        for i, l in enumerate(lines):
            s = l.strip()
            if not s or s.startswith('#'):
                continue
            if status(logpath(s)) == 'ok':
                lines[i] = '#DONE_SKIPPED ' + l
                continue
            lines[i] = '#TAKEN ' + l
            picked = s
            break
        with open(QFILE, 'w') as f:
            f.writelines(lines)
        return picked


def worker(gpu):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    while True:
        cmd = pop_next()
        if cmd is None:
            return
        name = logpath(cmd)
        log('GPU%d START %s' % (gpu, name))
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            r = subprocess.run(cmd, shell=True, env=env, cwd=os.getcwd())
            st = status(name)
            log('GPU%d DONE(%s) attempt=%d rc=%d %.1fmin %s'
                % (gpu, st, attempt, r.returncode, (time.time() - t0) / 60, name))
            if st == 'ok':
                break
            if st == 'oom':
                nc = halve_bs(cmd)
                if nc:
                    bs = re.search(r'--batch_size (\d+)', nc).group(1)
                    log('GPU%d OOM -> retry with --batch_size %s: %s' % (gpu, bs, name))
                    cmd = nc
                    continue
            break


def main():
    n = sum(1 for l in open(QFILE)
            if l.strip() and not l.lstrip().startswith('#'))
    log('scheduler start: %d queued jobs on GPUs %s' % (n, GPUS))
    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log('ALL DONE')


if __name__ == '__main__':
    main()
