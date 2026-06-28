#!/usr/bin/env python3
"""Covariate balance check: matched vs unmatched held-out subsets"""
import pandas as pd
import numpy as np
from scipy import stats

# ── Load data ──
full = pd.read_csv(
    '/Users/cary/Dev/behavioral-digital-twins/results/heldout_dce_predictions.csv'
)
full['is_matched'] = full['matched_llm_state'].notna().astype(int)

n_matched   = full['is_matched'].sum()
n_unmatched = (full['is_matched'] == 0).sum()
print(f"Matched:   {n_matched:>5} observations ({100*n_matched/len(full):.1f}%)")
print(f"Unmatched: {n_unmatched:>5} observations ({100*n_unmatched/len(full):.1f}%)")
print(f"Total:     {len(full):>5} observations")
print()

# ── Covariates to check ──
covariates = [
    ('wait',    'WaitTime (months)',      'continuous'),
    ('eff',     'Efficacy',               'continuous'),
    ('se',      'SideEffects',            'continuous'),
    ('cash',    'Cash Incentive (CNY)',   'continuous'),
    ('y',       'Choice (y=1)',           'continuous'),  # binary but t-test is fine for balance
]

# ── Respondent-level aggregation ──
# Also check per-respondent means (since observations are nested)
resp_agg = full.groupby(['respondent_id', 'is_matched']).agg(
    mean_wait=('wait', 'mean'),
    mean_eff=('eff', 'mean'),
    mean_se=('se', 'mean'),
    mean_cash=('cash', 'mean'),
    mean_y=('y', 'mean'),
    n_obs=('y', 'count')
).reset_index()

print("=" * 100)
print("BALANCE TABLE: Observation-Level")
print("=" * 100)

rows = []
for col, label, kind in covariates:
    m = full.loc[full['is_matched'] == 1, col]
    u = full.loc[full['is_matched'] == 0, col]

    mean_m = m.mean()
    sd_m   = m.std()
    mean_u = u.mean()
    sd_u   = u.std()

    # SMD
    pooled_sd = np.sqrt((sd_m**2 + sd_u**2) / 2)
    smd = (mean_m - mean_u) / pooled_sd if pooled_sd > 0 else 0.0

    # t-test
    if kind == 'continuous':
        if m.nunique() == 2 and u.nunique() == 2 and set(m.unique()) == set(u.unique()):
            # Binary variable -> chi-square
            ct = pd.crosstab(full['is_matched'], full[col])
            _, p, _, _ = stats.chi2_contingency(ct)
            p_str = f"{p:.4f}"
        else:
            t, p = stats.ttest_ind(m, u, equal_var=False)
            p_str = f"{p:.4f}"
    else:
        t, p = stats.ttest_ind(m, u, equal_var=False)
        p_str = f"{p:.4f}"

    rows.append((label, mean_m, sd_m, mean_u, sd_u, smd, p_str))

# Print table
print(f"{'Covariate':>30} | {'Matched Mean':>12} {'Matched SD':>10} | {'Unmatched Mean':>14} {'Unmatched SD':>10} | {'SMD':>8} | {'p-value':>8}")
print("-" * 100)
for label, mm, sm, mu, su, smd, p in rows:
    bal = " ✓" if abs(smd) < 0.1 else " ✗"
    print(f"{label:>30} | {mm:>12.4f} {sm:>10.4f} | {mu:>14.4f} {su:>10.4f} | {smd:>+8.4f}{bal} | {p:>8}")

print()
print("=" * 100)
print("BALANCE TABLE: Respondent-Level (mean per respondent)")
print("=" * 100)

resp_covariates = [
    ('mean_wait', 'WaitTime (months)'),
    ('mean_eff',  'Efficacy'),
    ('mean_se',   'SideEffects'),
    ('mean_cash', 'Cash Incentive (CNY)'),
    ('mean_y',    'Choice Rate'),
    ('n_obs',     'N observations'),
]

rows_r = []
for col, label in resp_covariates:
    m = resp_agg.loc[resp_agg['is_matched'] == 1, col]
    u = resp_agg.loc[resp_agg['is_matched'] == 0, col]

    mean_m = m.mean()
    sd_m   = m.std()
    mean_u = u.mean()
    sd_u   = u.std()

    pooled_sd = np.sqrt((sd_m**2 + sd_u**2) / 2)
    smd = (mean_m - mean_u) / pooled_sd if pooled_sd > 0 else 0.0

    t, p = stats.ttest_ind(m, u, equal_var=False)
    rows_r.append((label, mean_m, sd_m, mean_u, sd_u, smd, p))

print(f"{'Covariate':>30} | {'Matched Mean':>12} {'Matched SD':>10} | {'Unmatched Mean':>14} {'Unmatched SD':>10} | {'SMD':>8} | {'p-value':>8}")
print("-" * 100)
for label, mm, sm, mu, su, smd, p in rows_r:
    bal = " ✓" if abs(smd) < 0.1 else " ✗"
    print(f"{label:>30} | {mm:>12.4f} {sm:>10.4f} | {mu:>14.4f} {su:>10.4f} | {smd:>+8.4f}{bal} | {p:>8}")

print()
print(f"Matched respondents:   {resp_agg[resp_agg['is_matched']==1]['respondent_id'].nunique()}")
print(f"Unmatched respondents: {resp_agg[resp_agg['is_matched']==0]['respondent_id'].nunique()}")

# ── Persistent effect: do matched respondents have different wait-time profiles? ──
print()
print("=" * 100)
print("ATTRIBUTE LEVEL DISTRIBUTION (showing share of each attribute level)")
print("=" * 100)

for attr, label in [('wait', 'WaitTime'), ('eff', 'Efficacy'), ('se', 'SideEffects')]:
    print(f"\n--- {label} ---")
    ct = pd.crosstab(full['is_matched'], full[attr], normalize='index')
    ct.columns = [f"{c}" for c in ct.columns]
    print(ct.to_string(float_format="%.3f"))

# ── Binary outcome prevalence ──
print("\n--- Cash Incentive ---")
# Cash is continuous but show means across matched/unmatched at each level
for cash_lev in sorted(full['cash'].dropna().unique()):
    m = full[(full['is_matched']==1) & (full['cash']==cash_lev)].shape[0]
    u = full[(full['is_matched']==0) & (full['cash']==cash_lev)].shape[0]
    print(f"  Cash={cash_lev:>6.0f}:  matched={m:>4d}  unmatched={u:>4d}")
