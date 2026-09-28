#!/bin/bash
# Task A: run the 4 per-GPU saturation queues in parallel, then regenerate SUMMARY.
set -u
cd /mnt/jd/users/tengshiyuan.1/codes/Time-Series-Library
.venv/bin/python scripts/long_term_forecast/ehs_v2/gen_sat_queues.py "${1:-0,1,2,3}"
pids=()
for q in scripts/long_term_forecast/ehs_v2/sat_queue_gpu*.sh; do
  bash "$q" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "ALL SAT QUEUES DONE"
