# Held-out DCE Matched-Sample Audit

## Why $N$ differs

- **Panel A (Pure-DCE):** $N=3690$ — every alternative-level row in the held-out 20% respondent test split.
- **Panel B (all methods):** $N=809$ — the subset of those rows whose `(wait, eff, se)` triple **exactly matches** an attribute combination in `bdt_eval_grid_static.csv` / Qwen unconstrained LLM outputs.
- **Unmatched:** 2881 rows (78.1%) lack an LLM probability because the DCE design includes attribute levels outside the 64-state grid.

### DCE vs. 64-state grid

| Attribute | DCE (dce_encoded) | 64-state grid (LLM) |
|-----------|-------------------|---------------------|
| WaitTime | 0, 1, 2, 3, 6 | 0, 2, 4, 6 |
| Efficacy | 0, 0.5, 0.7, 0.95 | 0.3, 0.5, 0.7, 0.9 |
| SideEffects | 0, 1, 2, 3 | 0, 1, 2, 3 |

Examples of unmatched DCE rows: `eff=0` or `0.95`, `wait=1` or `3`, or combinations that never appear in the grid. No nearest-neighbor imputation was used.

## Files used

- `/Users/cary/.openclaw/workspace/results/heldout_dce_predictions.csv`
- `analysis_output/dce_encoded.csv` (source DCE, via prior run)
- `llm_parsed_outputs_qwen72b_unconstrained.csv` (LLM merge)

## Panel B matched-sample comparisons (Static-BDT, $\lambda=0.25$)

- Static-BDT minus Unconstrained LLM (log loss): **-0.111490**
- Static-BDT minus Unconstrained LLM (Brier): **-0.053253**
- Static-BDT minus Pure-DCE (log loss): **+0.015267**
- Static-BDT minus Pure-DCE (Brier): **+0.008138**

Negative log-loss / Brier deltas vs. Unconstrained LLM mean Static-BDT is better on that metric.
Negative deltas vs. Pure-DCE mean Static-BDT beats the train-only DCE frontier on the matched subsample.
