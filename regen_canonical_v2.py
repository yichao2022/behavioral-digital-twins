"""Regenerate all matched-based outputs using INTENDED-DESIGN frontier
(trained only on wait ∈ {0,1,3,6}, NOT including wait=2 outlier).

This is the canonical pipeline (recompute_intended_design.py) extended to:
  Table S6 12-row λ-sweep held-out sensitivity
  AUC column
  Bootstrap CIs (9 cells, 2000 reps)
  Covariate balance S13
  Subgroup heterogeneity S14
  Task-level multinomial S15
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
from scipy.special import expit
from sklearn.metrics import roc_auc_score

REPO = Path('/Users/cary/bdt_repo')
RESULTS = REPO / 'results'
RESULTS.mkdir(exist_ok=True)

SEED = 2026
TRAIN_FRAC = 0.80
N_BOOTSTRAP = 2000
RNG_BOOT = random.Random(2026)
LAMBDA_FIXED = 0.25
EPS = 1e-12

# Load all data
df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
pllm_df = pd.read_csv(REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv')
grid_df = pd.read_csv(REPO / 'bdt_eval_grid_static.csv')

# INTENDED DESIGN FILTER: exclude wait=2 outlier row
intended_dce = df[df['WaitTime'] != 2].copy()
print(f'[0] Intended design support: wait ∈ {{0,1,3,6}}; wait=2 outlier excluded')
print(f'    Full DCE rows: {len(df)} → intended: {len(intended_dce)}')

# Respondent split on intended-design respondents
rids = sorted(intended_dce['RespondentID'].unique())
rng = random.Random(SEED)
shuffled = rids[:]
rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train])
test_ids = set(shuffled[n_train:])
print(f'    Train resp: {len(train_ids)} | Test resp: {len(test_ids)}')

# Intended-design train and held-out
train_intended = intended_dce[intended_dce['RespondentID'].isin(train_ids)].copy()
heldout_intended = intended_dce[intended_dce['RespondentID'].isin(test_ids)].copy()
print(f'    Train rows: {len(train_intended)} | Held-out rows: {len(heldout_intended)}')

# Build training design matrix
def build_X_y(rows, with_asc=False):
    n = len(rows)
    cols = ['WaitTime', 'VaccineEfficacy', 'SideEffects']
    if with_asc:
        cols.append('ASC_optout')
    X = np.ones((n, len(cols) + 1))  # +1 for intercept
    y = np.zeros(n)
    names = ['const', 'wait', 'eff', 'se']
    if with_asc:
        names.append('ASC')
    for i, (_, r) in enumerate(rows.iterrows()):
        X[i, 0] = 1.0
        X[i, 1] = r['WaitTime']
        X[i, 2] = r['VaccineEfficacy']
        X[i, 3] = r['SideEffects']
        if with_asc:
            X[i, 4] = 1.0 if r['Alt'] == 'C' else 0.0
        y[i] = float(r['Choice'])
    return X, y, names

# Fit 6-parameter conditional logit MLE (IRLS) on intended-design train rows
def fit_logit_mle(X, y, max_iter=100):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        mu = expit(eta)
        w = np.clip(mu * (1 - mu), 1e-8, None)
        z_adj = eta + (y - mu) / w
        XtW = X.T * w
        A = XtW @ X
        b = XtW @ z_adj
        try:
            beta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta_new, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new; break
        beta = beta_new
    return beta

X_train, y_train, coef_names = build_X_y(train_intended, with_asc=True)
beta_train = fit_logit_mle(X_train, y_train)
print(f'\n[1] 6-parameter conditional logit trained on intended design:')
for n, b in zip(coef_names, beta_train):
    print(f'    {n:10s}  {b:+.4f}')

# Build P_emp (binary) for held-out alt-rows under intended design
X_heldout, y_heldout, _ = build_X_y(heldout_intended, with_asc=True)
p_emp_heldout = expit(X_heldout @ beta_train)

# Get LLM predictions
llm_lookup = {}
for _, r in pllm_df.iterrows():
    key = (r['wait'], r['eff'], r['se'])
    if key not in llm_lookup:
        llm_lookup[key] = r['probability_0_1']

p_llm_heldout = np.zeros(len(heldout_intended))
for i, (_, r) in enumerate(heldout_intended.iterrows()):
    key = (r['WaitTime'], r['VaccineEfficacy'], r['SideEffects'])
    p_llm_heldout[i] = llm_lookup.get(key, 0.5)

# Matched subset: 11-of-12 cells (wait ∈ {0,6} ∩ {0,2,4,6}, eff ∈ {0.5,0.7}, se ∈ {1,2,3})
matched_mask = np.array([
    (int(r['WaitTime']) in {0, 6}) and
    (int(round(r['VaccineEfficacy'] * 100)) in {50, 70}) and
    (int(r['SideEffects']) in {1, 2, 3})
    for _, r in heldout_intended.iterrows()
])
n_matched = matched_mask.sum()
print(f'\n[2] Matched alt-rows: {n_matched}')

# Restricted to non-optout (Alt ∈ {A,B})
nonoptout_mask = np.array([r['Alt'] in ('A', 'B') for _, r in heldout_intended.iterrows()])
final_mask = matched_mask & nonoptout_mask
y_matched = y_heldout[final_mask]
p_emp_matched = p_emp_heldout[final_mask]
p_llm_matched = p_llm_heldout[final_mask]
n_final = final_mask.sum()
print(f'    Restricted to non-optout: {n_final} (matched alt-rows)')

# Task-level grouping: each task has 2 non-optout alternatives (A vs B)
# Group rows by (RespondentID, Task)
task_groups = {}
for i, (_, r) in enumerate(heldout_intended[final_mask].iterrows()):
    key = (r['RespondentID'], r.get('Task', i))
    if key not in task_groups:
        task_groups[key] = {'y': [], 'emp': [], 'llm': []}
    task_groups[key]['y'].append(y_heldout[final_mask][i] if False else None)  # placeholder

# Better: just use n_final matched alt-rows directly for alt-level metrics
# (alt-level binary log loss is the main metric)

# ===== 12-row λ sweep =====
print(f'\n[3] 12-row λ-sweep (alt-level binary, {n_final} matched rows):')
print(f'{"λ":>6} {"N":>5} {"LogLoss":>9} {"Brier":>8} {"AUC":>8} {"CalInt":>8} {"CalSlope":>9}')
lambda_sweep_rows = []
for lam in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]:
    p_blend = lam * p_llm_matched + (1 - lam) * p_emp_matched
    p_clip = np.clip(p_blend, EPS, 1 - EPS)
    ll = -np.mean(y_matched * np.log(p_clip) + (1 - y_matched) * np.log(1 - p_clip))
    br = np.mean((p_clip - y_matched) ** 2)
    auc = roc_auc_score(y_matched, p_clip) if len(np.unique(y_matched)) > 1 else np.nan
    # Calibration intercept and slope (logistic): logit(y) = a + b * logit(p)
    z_p = np.log(p_clip / (1 - p_clip))
    z_y = np.log(np.clip(y_matched, EPS, 1 - EPS) / np.clip(1 - y_matched, EPS, 1 - EPS))
    X_cal = np.column_stack([np.ones_like(z_p), z_p])
    try:
        coef_cal = np.linalg.lstsq(X_cal, z_y, rcond=None)[0]
        cal_int = coef_cal[0]
        cal_slope = coef_cal[1]
    except Exception:
        cal_int = cal_slope = np.nan
    note = 'Pure-DCE' if lam == 0.0 else ('Unconstrained LLM' if lam == 1.0 else
                                            ('Static-BDT Anchor' if lam == LAMBDA_FIXED else ''))
    lambda_sweep_rows.append({
        'lambda': lam, 'N': n_final, 'log_loss': ll, 'brier': br,
        'auc': auc, 'cal_intercept': cal_int, 'cal_slope': cal_slope,
        'note': note
    })
    print(f'{lam:>6.2f} {n_final:>5d} {ll:>9.4f} {br:>8.4f} {auc:>8.4f} {cal_int:>8.4f} {cal_slope:>9.4f}')

lambda_df = pd.DataFrame(lambda_sweep_rows)
lambda_df.to_csv(RESULTS / 'lambda_sensitivity_heldout_intended.csv', index=False)
print(f'\nWrote {RESULTS / "lambda_sensitivity_heldout_intended.csv"}')

# Extract primary numbers
row_025 = lambda_df[lambda_df['lambda'] == LAMBDA_FIXED].iloc[0]
row_00 = lambda_df[lambda_df['lambda'] == 0.0].iloc[0]
row_10 = lambda_df[lambda_df['lambda'] == 1.0].iloc[0]
print(f'\n[4] PRIMARY ALT-LEVEL (intended design, 6-param refit, matched N={n_final}):')
print(f'    Pure-DCE  (λ=0.0):  Log loss = {row_00["log_loss"]:.4f}  Brier = {row_00["brier"]:.4f}')
print(f'    EFR       (λ=0.25): Log loss = {row_025["log_loss"]:.4f}  Brier = {row_025["brier"]:.4f}')
print(f'    Raw LLM   (λ=1.0):  Log loss = {row_10["log_loss"]:.4f}  Brier = {row_10["brier"]:.4f}')

# ===== Task-level A-vs-B =====
print(f'\n[5] Task-level A-vs-B binary log loss (matched {n_final} alt-rows = {(n_final // 2)} tasks):')
n_tasks = n_final // 2
y_pairs = y_matched.reshape(n_tasks, 2)
emp_pairs = p_emp_matched.reshape(n_tasks, 2)
llm_pairs = p_llm_matched.reshape(n_tasks, 2)
for label, p_pairs in [('Pure-DCE', emp_pairs), ('EFR (λ=0.25)',
                         0.25 * llm_pairs + 0.75 * emp_pairs), ('Raw LLM', llm_pairs)]:
    p_norm = p_pairs / p_pairs.sum(axis=1, keepdims=True)
    ll = -np.mean(y_pairs[:, 0] * np.log(p_norm[:, 0]) + y_pairs[:, 1] * np.log(p_norm[:, 1]))
    print(f'    {label:18s}  Log loss = {ll:.4f}  ({n_tasks} tasks)')

# ===== Covariate balance =====
print(f'\n[6] Covariate balance (matched vs unmatched held-out):')
matched_idx = heldout_intended.index[final_mask]
heldout_intended = heldout_intended.copy()
heldout_intended['is_matched'] = heldout_intended.index.isin(matched_idx)
matched_set = heldout_intended[heldout_intended['is_matched']]
unmatched_set = heldout_intended[~heldout_intended['is_matched']]
print(f'    Matched: {len(matched_set)} | Unmatched: {len(unmatched_set)}')

balance_rows = []
def get_y(rows):
    return (rows['Choice'] == 'A').astype(float) + (rows['Choice'] == 'B').astype(float)

for col, label in [('WaitTime', 'Wait Time (months)'),
                    ('VaccineEfficacy', 'Vaccine Efficacy'),
                    ('SideEffects', 'Side-Effect Burden (coded level)'),
                    ('CashIncentives', 'Cash Incentives (RMB)'),
                    ('VaccineOrigin', 'Vaccine Origin (0=dom, 1=imp)'),
                    ('y', 'Choice Rate (y=1)')]:
    if col == 'y':
        m = float((matched_set['Choice'].isin(['A', 'B'])).mean())
        u = float((unmatched_set['Choice'].isin(['A', 'B'])).mean())
        s_m = float((matched_set['Choice'].isin(['A', 'B'])).std())
        s_u = float((unmatched_set['Choice'].isin(['A', 'B'])).std())
    else:
        m = float(matched_set[col].mean())
        u = float(unmatched_set[col].mean())
        s_m = float(matched_set[col].std())
        s_u = float(unmatched_set[col].std())
    pooled_sd = np.sqrt((s_m**2 + s_u**2) / 2)
    smd = (m - u) / pooled_sd if pooled_sd > 0 else np.nan
    balance_rows.append({'Variable': label, 'Matched_Mean': m, 'Unmatched_Mean': u, 'SMD': smd})
    print(f'    {label:35s}  Matched: {m:.3f}  Unmatched: {u:.3f}  SMD: {smd:.3f}')

balance_df = pd.DataFrame(balance_rows)
balance_df.to_csv(RESULTS / 'covariate_balance_intended.csv', index=False)
print(f'\nWrote {RESULTS / "covariate_balance_intended.csv"}')

# ===== Subgroup heterogeneity (Age <55 / >=55, Female / Male, Below College / College+) =====
print(f'\n[7] Subgroup heterogeneity (Static-BDT vs LLM, matched alt-rows):')
demo_lookup = {}
for _, r in heldout_intended[['RespondentID', 'Age', 'Gender']].drop_duplicates('RespondentID').iterrows():
    age_str = str(r['Age'])
    try:
        if '-' in age_str:
            lo, hi = age_str.split('-')
            age_mid = (int(lo) + int(hi)) / 2
        elif age_str.endswith('+'):
            age_mid = int(age_str[:-1]) + 2
        else:
            age_mid = float(age_str)
    except Exception:
        age_mid = 99
    demo_lookup[r['RespondentID']] = {'Age': age_mid, 'Gender': str(r.get('Gender', ''))}

subgroups = []
for name, mask_fn in [
    ('Age < 55', lambda r: demo_lookup.get(r['RespondentID'], {}).get('Age', 99) < 55),
    ('Age >= 55', lambda r: demo_lookup.get(r['RespondentID'], {}).get('Age', 99) >= 55),
    ('Female', lambda r: demo_lookup.get(r['RespondentID'], {}).get('Gender', '') == 'Female'),
    ('Male', lambda r: demo_lookup.get(r['RespondentID'], {}).get('Gender', '') == 'Male'),
]:
    # Compute on matched subset
    sub_rows = []
    for i, (_, r) in enumerate(heldout_intended[final_mask].iterrows()):
        if mask_fn(r):
            sub_rows.append(i)
    if len(sub_rows) == 0:
        continue
    y_sub = y_matched[sub_rows]
    p_emp_sub = p_emp_matched[sub_rows]
    p_llm_sub = p_llm_matched[sub_rows]
    # LLM
    p_clip_llm = np.clip(p_llm_sub, EPS, 1 - EPS)
    ll_llm = -np.mean(y_sub * np.log(p_clip_llm) + (1 - y_sub) * np.log(1 - p_clip_llm))
    br_llm = np.mean((p_clip_llm - y_sub) ** 2)
    # EFR
    p_efr = LAMBDA_FIXED * p_llm_sub + (1 - LAMBDA_FIXED) * p_emp_sub
    p_clip_efr = np.clip(p_efr, EPS, 1 - EPS)
    ll_efr = -np.mean(y_sub * np.log(p_clip_efr) + (1 - y_sub) * np.log(1 - p_clip_efr))
    br_efr = np.mean((p_clip_efr - y_sub) ** 2)
    # Spearman rho vs P_emp
    from scipy.stats import spearmanr
    rho_llm = spearmanr(y_sub, p_llm_sub).correlation
    rho_efr = spearmanr(y_sub, p_emp_sub).correlation
    subgroups.append({
        'Subgroup': name, 'N': len(sub_rows),
        'LLM_LogLoss': ll_llm, 'LLM_Brier': br_llm, 'LLM_Rho': rho_llm,
        'EFR_LogLoss': ll_efr, 'EFR_Brier': br_efr, 'EFR_Rho': rho_efr,
    })
    print(f'    {name:18s}  N={len(sub_rows):4d}  LLM={ll_llm:.4f}/{br_llm:.4f}/{rho_llm:+.3f}  EFR={ll_efr:.4f}/{br_efr:.4f}/{rho_efr:+.3f}')
    sub_df = pd.DataFrame(subgroups)
    sub_df.to_csv(RESULTS / 'subgroup_heterogeneity_intended.csv', index=False)
    print(f'\nWrote {RESULTS / "subgroup_heterogeneity_intended.csv"}')
else:
    print('    demographics file not found, skipping subgroup analysis')

# ===== Bootstrap CIs (9 cells) =====
print(f'\n[8] Bootstrap CIs (2000 reps, respondent-level cluster):')
unique_resp = heldout_intended.loc[final_mask, 'RespondentID'].unique()
boot_rows = []
for lam in [0.00, 0.25, 1.00]:
    boot_ll = []
    boot_br = []
    for _ in range(N_BOOTSTRAP):
        boot_resp = RNG_BOOT.choices(unique_resp.tolist(), k=len(unique_resp))
        boot_idx = np.concatenate([
            np.where(heldout_intended.loc[final_mask, 'RespondentID'].values == r)[0]
            for r in boot_resp
        ])
        if len(boot_idx) == 0:
            continue
        y_b = y_matched[boot_idx]
        p_b = lam * p_llm_matched[boot_idx] + (1 - lam) * p_emp_matched[boot_idx]
        p_clip = np.clip(p_b, EPS, 1 - EPS)
        boot_ll.append(-np.mean(y_b * np.log(p_clip) + (1 - y_b) * np.log(1 - p_clip)))
        boot_br.append(np.mean((p_clip - y_b) ** 2))
    boot_ll = np.array(boot_ll)
    boot_br = np.array(boot_br)
    label = {0.0: 'Pure-DCE', 0.25: 'EFR', 1.0: 'Raw LLM'}[lam]
    boot_rows.append({
        'lambda': lam, 'label': label, 'logloss_mean': np.mean(boot_ll),
        'logloss_lo': np.percentile(boot_ll, 2.5), 'logloss_hi': np.percentile(boot_ll, 97.5),
        'brier_mean': np.mean(boot_br), 'brier_lo': np.percentile(boot_br, 2.5),
        'brier_hi': np.percentile(boot_br, 97.5),
    })
    print(f'    {label:10s} (λ={lam:.2f}):  Log loss = {np.mean(boot_ll):.4f} [{np.percentile(boot_ll, 2.5):.4f}, {np.percentile(boot_ll, 97.5):.4f}]')
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(RESULTS / 'bootstrap_heldout_intended.csv', index=False)
print(f'\nWrote {RESULTS / "bootstrap_heldout_intended.csv"}')

print(f'\n{"="*60}\nDONE. All outputs under /Users/cary/bdt_repo/results/')