# Effective History Span (EHS)

Code, configurations, run logs, and analysis pipeline for the paper
**"How Far Back Should We Look? Measuring, Explaining, and Exploiting the Effective History Span of Time Series Forecasting"** (ICLR 2027 submission, anonymous).

The repository is organized so that every number in the paper can be traced to a script and a log.

## Layout

```
code/            runnable code: TSLib fork (run.py, data_provider, models, layers, exp, utils)
                 + analysis/ (statistics, predictability, selectors, BOCPD, power law, decay, figures)
                 + tsfm/     (Chronos-Bolt zero-shot evaluation, latent-action probes)
scripts/         run manifests and schedulers (gen_cmds.py emits every launch command verbatim)
results/logs/    per-run training logs, aggregation summaries, and verdict files (the run ledger)
paper/           anonymous manuscript source
```

## Evidence index (paper artifact -> source)

| Paper artifact | Generator script | Source data |
| --- | --- | --- |
| Table 1 (main sweep) | `code/analysis/collect_ehs.py` | `results/logs/ehs_v2/*.log` -> `SUMMARY.md` |
| Table 2 (best lookback per family) | `code/analysis/collect_ehs.py` + `code/analysis/e6_fill_verdict.py` | same |
| Figure 1 (sweep) | `code/analysis/make_figs.py` (`fig1`) | `results/logs/ehs_v2/SUMMARY.md` |
| Figure 2 (predictability) | `code/analysis/make_figs.py` (`fig2`) | `results/logs/ehs_final/taskB_*_train.*` |
| Figure 3 (attention) | `code/analysis/make_figs.py` (`fig3`) | `results/logs/mechanism/exp1/attn_metrics.npz` |
| Section 3 zero-shot sweep | `code/tsfm/` + `results/logs/ehs_fix/e1_merged.json` | stride-2 caches (release asset) |
| Section 4.1 BOCPD falsification | `code/analysis/bocpd_ehs.py`, `code/analysis/e7_bocpd_robust.py` | `results/logs/ehs_fix/e7/` |
| Section 4.3 power law | `code/analysis/power_law_gen*.py`, `power_law_collect*.py` | `results/logs/power_law/` |
| Section 5 predictability | `code/analysis/ehs_predict_v2.py` / `ehs_predict_v3.py` (`--convention train`) | `results/logs/ehs_final/`, `results/logs/ehs_v2_ext/` |
| Section 5 selectors (val/PACF/AIC) | `code/analysis/e3_val_select*.py`, `e3_pacf_aic.py`, `e3_verdict.py`, `e3_val_select_ext.py` | `results/logs/ehs_fix/` |
| Section 6 budget / decay | `code/analysis/e4_verdict.py`, `code/analysis/decay_*.py` | `results/logs/ehs_fix/e4*`, `results/logs/decay/` |
| Appendix tables | `code/analysis/make_figs.py` (`write_appendix_tables`) | as above |
| Extended suite (6 datasets) | `code/analysis/ehs_ext_prep*.py` (data), same sweep pipeline | `results/logs/ehs_v2_ext/` |
| GIFT-Eval study | `results/logs/gift_ext/*.py` | `results/logs/gift_ext/*.csv`, `REPORT.md` |

## Reproducing the pipeline

1. Install the environment (Python 3.10, torch 2.x; see `code/requirements` of the base TSLib).
2. Datasets: public benchmarks (ETT, electricity, traffic, weather, exchange_rate) from the
   standard TSLib links; the six extension datasets are built by `code/analysis/ehs_ext_prep*.py`.
3. Training: every run command is emitted by
   `scripts/long_term_forecast/ehs_v2/gen_cmds.py` into `main_cmds.txt` (480 runs),
   `sat_cmds.txt` (saturation probe), and the fill manifests. `fill_scheduler.py` runs a
   manifest one job per GPU with automatic OOM retry.
4. Aggregation and analysis: `code/analysis/collect_ehs.py`, then the evidence-index scripts above.
   Statistics for the predictability analysis use the leakage-free train-split convention
   (`--convention train`).

## Batch-size note

All 480 main-matrix runs carry three seeds at the per-dataset configurations of Appendix A,
except three wide-data PatchTST cells whose official batch does not fit a single 80GB GPU at
long lookbacks (PatchTST is channel-independent, so memory scales with channel count):
`electricity sl1440` (batch 16->8), `electricity sl2880` (16->4), `traffic sl2880` (4->2).
Batch size is uniform across the three seeds of each affected cell. Earlier reduced-batch
rounds are preserved under `results/logs/ehs_v2/reduced_batch_backup/` for provenance.

## Missing-arms ledger

The main matrix (8 datasets x 4 families x 5 lookbacks x 3 seeds = 480 runs) is complete.
The initial snapshot missed 135 runs for execution-environment reasons (46 GPU OOM under
co-placement, 65 time-box cuts before launch, 24 killed mid-run); all were re-run to
completion. See `results/logs/ehs_v2/fill91_scheduler*.log` and Appendix C of the paper.

## Data exclusions (size)

Per-window zero-shot caches (`*.npz`, including `taskC_cache_*`, `zs_*.npz`, e1 shards) and the
GIFT-Eval downloaded corpora are not tracked in the repository; the dataset-level aggregates
derived from them are preserved in the JSON/CSV verdict files listed in the evidence index.

## License

MIT (base framework: THUML Time-Series-Library; see LICENSE).
