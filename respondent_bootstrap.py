#!/usr/bin/env python3
"""Respondent-level cluster bootstrap for held-out DCE predictions.
Resamples 205 respondents with replacement → computes metrics on all their obs.
2,000 reps. Output: point estimates + 95% CIs.
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress

SEED = 42
N_BOOT = 2000
ALPHA = 0.05

def compute_metrics(y_true, y_pred, eps=1e-15):
    yp = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    yt = np.asarray(y_true, dtype=float)
    ll = -float(np.mean(yt * np.log(yp) + (1 - yt) * np.log(1 - yp)))
    brier = float(np.mean((yp - yt) ** 2))
    n = len(yt)
    if n < 3:
        return ll, brier, np.nan, np.nan
    slope, intercept, *_ = linregress(yp, yt)
    return ll, brier, float(intercept), float(slope)

# Load
d = pd.read_csv('results/heldout_dce_predictions.csv')
matched = d[d['matched_llm_state'].notna()].copy()
resp_ids = matched['respondent_id'].unique().tolist()
n_resp = len(resp_ids)
rng = np.random.default_rng(SEED)

# Precompute point estimates
y_true = matched['y'].values
pred_cols = {'Pure-DCE (λ=0)': 'p_dce',
             'Static-BDT (λ=0.25)': 'p_bdt',
             'Unconstrained LLM (λ=1)': 'p_llm'}
method_names = list(pred_cols.keys())

print('=== Point Estimates (N=809 matched) ===')
print(f'{"Method":>25}  {"Log Loss":>9}  {"Brier":>8}  {"Cal Int":>8}  {"Cal Slp":>8}')
print('-' * 65)
for name, col in pred_cols.items():
    ll, br, ci, cs = compute_metrics(y_true, matched[col].values)
    print(f'{name:>25}  {ll:>9.6f}  {br:>8.6f}  {ci:>8.4f}  {cs:>8.4f}')
print()

# Bootstrap storage
boot_stats = {name: {'ll': [], 'brier': [], 'cal_int': [], 'cal_slp': []}
              for name in method_names}

# Group data by respondent for fast lookup
resp_groups = {rid: grp for rid, grp in matched.groupby('respondent_id')}

for b in range(N_BOOT):
    # Resample respondents
    sampled = rng.choice(resp_ids, size=n_resp, replace=True)
    boot_df = pd.concat([resp_groups[rid] for rid in sampled], ignore_index=True)
    yb = boot_df['y'].values
    for name, col in pred_cols.items():
        ll, br, ci, cs = compute_metrics(yb, boot_df[col].values)
        boot_stats[name]['ll'].append(ll)
        boot_stats[name]['brier'].append(br)
        boot_stats[name]['cal_int'].append(ci)
        boot_stats[name]['cal_slp'].append(cs)

    if (b + 1) % 500 == 0:
        print(f'  bootstrap {b+1}/{N_BOOT}')

print()

# Summary
print(f'=== Respondent-Level Bootstrap (N={N_BOOT} reps, {n_resp} respondents) ===')
print(f'{"Method":>25}  {"Metric":>10}  {"Estimate":>9}  {"95% CI":>18}')
print('-' * 65)
for name in method_names:
    for metric, label in [('ll', 'Log Loss'), ('brier', 'Brier'),
                          ('cal_int', 'Cal Int'), ('cal_slp', 'Cal Slp')]:
        arr = np.array(boot_stats[name][metric])
        arr = arr[~np.isnan(arr)]
        if len(arr) < 100:
            continue
        est = float(np.median(arr))
        lo = float(np.percentile(arr, 100 * ALPHA / 2))
        hi = float(np.percentile(arr, 100 * (1 - ALPHA / 2)))
        print(f'{name:>25}  {label:>10}  {est:>9.6f}  [{lo:>7.6f}, {hi:>7.6f}]')
    print()
