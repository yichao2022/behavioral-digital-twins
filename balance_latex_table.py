#!/usr/bin/env python3
"""Generate LaTeX balance table"""
import pandas as pd
import numpy as np
from scipy import stats

full = pd.read_csv(
    '/Users/cary/Dev/behavioral-digital-twins/results/heldout_dce_predictions.csv'
)
full['is_matched'] = full['matched_llm_state'].notna().astype(int)

covariates = [
    ('wait',    'WaitTime (months)'),
    ('eff',     'Efficacy (1-specificity)'),
    ('se',      'SideEffect burden'),
    ('cash',    'Cash incentive (CNY)'),
    ('y',       'Choice indicator ($y=1$)'),
]

# ── Observation-level balance ──
print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{Covariate balance: matched vs. unmatched held-out subsets (observation-level)}")
print("\\label{tab:balance}")
print("\\begin{tabular}{lrrrrrr}")
print("\\toprule")
print("Covariate & \\multicolumn{2}{c}{Matched} & \\multicolumn{2}{c}{Unmatched} & \\multicolumn{1}{c}{SMD} & \\\\")
print("& Mean & SD & Mean & SD & & $p$-value\\\\")
print("\\midrule")

for col, label in covariates:
    m = full.loc[full['is_matched'] == 1, col]
    u = full.loc[full['is_matched'] == 0, col]

    mean_m = m.mean()
    sd_m   = m.std()
    mean_u = u.mean()
    sd_u   = u.std()

    pooled_sd = np.sqrt((sd_m**2 + sd_u**2) / 2)
    smd = (mean_m - mean_u) / pooled_sd if pooled_sd > 0 else 0.0

    t, p = stats.ttest_ind(m, u, equal_var=False)
    p_str = f"{p:.4f}"

    print(f"  {label} & ${mean_m:.4f}$ & ${sd_m:.4f}$ & ${mean_u:.4f}$ & ${sd_u:.4f}$ & ${smd:+.4f}$ & ${p_str}$\\\\")

print("\\midrule")
print("\\multicolumn{7}{l}{\\emph{Observation-level: matched $N=809$, unmatched $N=2,\\!881$}}\\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print()
print("%" * 80)
print()

# ── Respondent-level ──
resp_agg = full.groupby(['respondent_id', 'is_matched']).agg(
    mean_wait=('wait', 'mean'),
    mean_eff=('eff', 'mean'),
    mean_se=('se', 'mean'),
    mean_cash=('cash', 'mean'),
    mean_y=('y', 'mean'),
    n_obs=('y', 'count')
).reset_index()

print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{Covariate balance: matched vs. unmatched held-out subsets (respondent-level means)}")
print("\\label{tab:balance_respondent}")
print("\\begin{tabular}{lrrrrrr}")
print("\\toprule")
print("Covariate & \\multicolumn{2}{c}{Matched} & \\multicolumn{2}{c}{Unmatched} & \\multicolumn{1}{c}{SMD} & \\\\")
print("& Mean & SD & Mean & SD & & $p$-value\\\\")
print("\\midrule")

for col, label in [
    ('mean_wait', 'WaitTime (months)'),
    ('mean_eff',  'Efficacy'),
    ('mean_se',   'SideEffects'),
    ('mean_cash', 'Cash incentive (CNY)'),
    ('mean_y',    'Choice rate'),
    ('n_obs',     'N observations'),
]:
    m = resp_agg.loc[resp_agg['is_matched'] == 1, col]
    u = resp_agg.loc[resp_agg['is_matched'] == 0, col]

    mean_m = m.mean()
    sd_m   = m.std()
    mean_u = u.mean()
    sd_u   = u.std()

    pooled_sd = np.sqrt((sd_m**2 + sd_u**2) / 2)
    smd = (mean_m - mean_u) / pooled_sd if pooled_sd > 0 else 0.0

    t, p = stats.ttest_ind(m, u, equal_var=False)
    p_str = f"{p:.2e}"

    print(f"  {label} & ${mean_m:.4f}$ & ${sd_m:.4f}$ & ${mean_u:.4f}$ & ${sd_u:.4f}$ & ${smd:+.4f}$ & ${p_str}$\\\\")

print("\\midrule")
print("\\multicolumn{7}{l}{\\emph{Respondent-level: matched $N=205$, unmatched $N=205$ (all respondents appear in both)}}\\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
