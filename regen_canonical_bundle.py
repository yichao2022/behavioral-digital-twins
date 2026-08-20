"""
Dependency sweep: regenerate all matched-based outputs under canonical
(intended design + seed=2026 + 813 rows / 78 tasks / 11 of 12 cells).

Outputs:
  results/lambda_sensitivity_heldout_intended.csv   (12-row lambda sweep, alt-level binary)
  results/lambda_sensitivity_heldout_intended.json   (same, plus AUC + CI)
  results/bootstrap_heldout_intended.csv             (Figure 1 respondent-cluster bootstrap)
  results/covariate_balance_intended.csv             (Table S13)
  results/subgroup_heterogeneity_intended.csv        (Table S14)
"""
import pandas as pd
import numpy as np
import random
from scipy.special import expit
from pathlib import Path

SEED = 2026
TRAIN_FRAC = 0.80
REPO = Path('/Users/cary/bdt_repo')
RES = REPO / 'results'

# ----------------------------------------------------------------------
# 1. Load canonical coef + split data
# ----------------------------------------------------------------------
coef = pd.read_csv(RES / 'heldout_dce_frontier_coefficients.csv')
b = dict(zip(coef['term'], coef['coef']))
beta_const = b['const']
beta_wait = b['wait']
beta_eff = b['eff']
beta_se = b['se']
beta_cash = b['cash']

def u(row):
    return (beta_const
            + beta_wait * row['WaitTime']
            + beta_eff * row['VaccineEfficacy']
            + beta_se * row['SideEffects']
            + beta_cash * row['CashIncentives'])

df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED); shuffled = rids[:]; rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train])
test_ids = set(shuffled[n_train:])
heldout = df[df['RespondentID'].isin(test_ids)].copy()

# Intended design: wait {0,1,3,6} (excludes wait=2 anomaly)
heldout_int = heldout[heldout['WaitTime'].isin({0,1,3,6})].copy()

# Non-optout subset
nonopt = heldout_int[
    (heldout_int['CashIncentives']>0) &
    (heldout_int['VaccineEfficacy']>0) &
    (heldout_int['SideEffects']>0)
].copy()

# Canonical literal overlap on the three focal attributes
# DCE non-optout wait ∩ grid wait {0,2,4,6} = {0,6}
# DCE non-optout eff ∩ grid eff {0.3,0.5,0.7,0.9} = {0.5,0.7}
# DCE non-optout se ∩ grid se {0,1,2,3} = {1,2,3}
matched = nonopt[
    nonopt['WaitTime'].isin({0, 6}) &
    nonopt['VaccineEfficacy'].isin({0.5, 0.7}) &
    nonopt['SideEffects'].isin({1.0, 2.0, 3.0})
].copy()

cells = matched[['WaitTime','VaccineEfficacy','SideEffects']].drop_duplicates()
print(f'Matched alt-rows: {len(matched)}; Distinct cells: {len(cells)} (of 12 candidate)')
tasks = matched.groupby(['RespondentID','Choiceset']).filter(lambda g: len(g)==2)
n_tasks = tasks.groupby(['RespondentID','Choiceset']).ngroups
print(f'2-alt tasks: {n_tasks}')

# Unmatched held-out alt-rows
unmatched = heldout_int[~heldout_int.index.isin(matched.index)].copy()
print(f'Unmatched held-out alt-rows (intended design): {len(unmatched)}')

# LLM probabilities
pllm_df = pd.read_csv(REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv')
pllm_state = pllm_df.groupby('state')['probability_0_1'].mean().to_dict()
grid = pd.read_csv(REPO / 'bdt_eval_grid_static.csv')

def state_of(row):
    matches = grid[(grid['wait']==row['WaitTime']) &
                   (grid['eff']==row['VaccineEfficacy']) &
                   (grid['se']==row['SideEffects'])]
    return matches.iloc[0]['state'] if len(matches) else None

matched['state'] = matched.apply(state_of, axis=1)
matched['p_llm'] = matched['state'].map(pllm_state)
matched_llm = matched.dropna(subset=['p_llm']).copy()

# Pure-DCE predictions on matched subset
matched['p_dce'] = matched.apply(lambda r: expit(u(r)), axis=1)
matched_llm['p_dce'] = matched_llm.apply(lambda r: expit(u(r)), axis=1)

# ----------------------------------------------------------------------
# 2. Table S6: 12-row lambda sweep on matched alt-rows (binary)
# ----------------------------------------------------------------------
def log_loss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def brier(y, p):
    return np.mean((y-p)**2)

from sklearn.metrics import roc_auc_score

y_match = matched_llm['Choice'].values
p_dce = matched_llm['p_dce'].values
p_llm = matched_llm['p_llm'].values

sweep_rows = []
LAMBDA_MAIN = 0.25
LAMBDAS = [0.00, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
for lam in LAMBDAS:
    p = lam * p_llm + (1-lam) * p_dce
    ll = log_loss(y_match, p)
    br = brier(y_match, p)
    try:
        auc = roc_auc_score(y_match, p)
    except Exception:
        auc = float('nan')
    note = ''
    if abs(lam) < 1e-9: note = 'Pure-DCE'
    elif abs(lam-1) < 1e-9: note = 'Unconstrained LLM'
    elif abs(lam-LAMBDA_MAIN) < 1e-9: note = 'Main'
    # Compute calibration intercept/slope via logistic regression on y ~ logit(p)
    eps = 1e-7
    p_clip = np.clip(p, eps, 1-eps)
    logit_p = np.log(p_clip / (1-p_clip))
    from sklearn.linear_model import LogisticRegression
    Xc = logit_p.reshape(-1,1)
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=200).fit(Xc, y_match)
        cal_int = lr.intercept_[0]
        cal_slope = lr.coef_[0][0]
    except Exception:
        cal_int = float('nan'); cal_slope = float('nan')
    sweep_rows.append({
        'lambda': f'{lam:.2f}',
        'N': len(matched_llm),
        'log_loss': f'{ll:.4f}',
        'brier': f'{br:.4f}',
        'AUC': f'{auc:.4f}' if not np.isnan(auc) else 'NA',
        'cal_int': f'{cal_int:.4f}',
        'cal_slope': f'{cal_slope:.4f}',
        'note': note,
    })
sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(RES / 'lambda_sensitivity_heldout_intended.csv', index=False)
print(f'\nWrote {RES / "lambda_sensitivity_heldout_intended.csv"}')
print(sweep_df.to_string(index=False))

# ----------------------------------------------------------------------
# 3. Figure 1: respondent-cluster bootstrap CI
# ----------------------------------------------------------------------
N_BOOT = 2000
matched_llm_bootstrap = matched_llm.copy()
# Pure-DCE point estimate
p_dce_pt = matched_llm_bootstrap['p_dce'].values
y_pt = matched_llm_bootstrap['Choice'].values
ll_dce_pt = log_loss(y_pt, p_dce_pt)
br_dce_pt = brier(y_pt, p_dce_pt)
try:
    auc_dce_pt = roc_auc_score(y_pt, p_dce_pt)
except Exception:
    auc_dce_pt = float('nan')

# Static-BDT (λ=0.25) point estimate
LAMBDA = 0.25
p_bdt_pt = LAMBDA * matched_llm_bootstrap['p_llm'].values + (1-LAMBDA) * p_dce_pt
ll_bdt_pt = log_loss(y_pt, p_bdt_pt)
br_bdt_pt = brier(y_pt, p_bdt_pt)
try:
    auc_bdt_pt = roc_auc_score(y_pt, p_bdt_pt)
except Exception:
    auc_bdt_pt = float('nan')

# Raw LLM point estimate
p_llm_pt = matched_llm_bootstrap['p_llm'].values
ll_llm_pt = log_loss(y_pt, p_llm_pt)
br_llm_pt = brier(y_pt, p_llm_pt)
try:
    auc_llm_pt = roc_auc_score(y_pt, p_llm_pt)
except Exception:
    auc_llm_pt = float('nan')

# Cluster bootstrap
unique_resp = sorted(matched_llm_bootstrap['RespondentID'].unique())
resp_to_idx = {r: [] for r in unique_resp}
for i, r in enumerate(matched_llm_bootstrap['RespondentID'].values):
    resp_to_idx[r].append(i)

boot_results = {'Pure-DCE': {'log_loss': [], 'brier': [], 'AUC': []},
                'Static-BDT': {'log_loss': [], 'brier': [], 'AUC': []},
                'LLM': {'log_loss': [], 'brier': [], 'AUC': []}}

np.random.seed(2026)
for b in range(N_BOOT):
    # Sample respondents with replacement
    sampled = np.random.choice(unique_resp, size=len(unique_resp), replace=True)
    idx = []
    for r in sampled:
        idx.extend(resp_to_idx[r])
    idx = np.array(idx)
    y_b = y_pt[idx]
    p_dce_b = p_dce_pt[idx]
    p_llm_b = p_llm_pt[idx]
    p_bdt_b = p_bdt_pt[idx]
    try:
        boot_results['Pure-DCE']['log_loss'].append(log_loss(y_b, p_dce_b))
        boot_results['Pure-DCE']['brier'].append(brier(y_b, p_dce_b))
        boot_results['Pure-DCE']['AUC'].append(roc_auc_score(y_b, p_dce_b))
        boot_results['Static-BDT']['log_loss'].append(log_loss(y_b, p_bdt_b))
        boot_results['Static-BDT']['brier'].append(brier(y_b, p_bdt_b))
        boot_results['Static-BDT']['AUC'].append(roc_auc_score(y_b, p_bdt_b))
        boot_results['LLM']['log_loss'].append(log_loss(y_b, p_llm_b))
        boot_results['LLM']['brier'].append(brier(y_b, p_llm_b))
        boot_results['LLM']['AUC'].append(roc_auc_score(y_b, p_llm_b))
    except Exception:
        continue

boot_rows = []
for model in ['Pure-DCE', 'Static-BDT', 'LLM']:
    for metric in ['log_loss', 'brier', 'AUC']:
        arr = np.array(boot_results[model][metric])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        pt = {'log_loss': ll_dce_pt if model=='Pure-DCE' else (ll_bdt_pt if model=='Static-BDT' else ll_llm_pt),
              'brier': br_dce_pt if model=='Pure-DCE' else (br_bdt_pt if model=='Static-BDT' else br_llm_pt),
              'AUC': auc_dce_pt if model=='Pure-DCE' else (auc_bdt_pt if model=='Static-BDT' else auc_llm_pt)}[metric]
        boot_rows.append({
            'model': model, 'metric': metric,
            'point': f'{pt:.4f}',
            'CI_low': f'{lo:.4f}', 'CI_high': f'{hi:.4f}',
            'n_boot': len(arr),
        })
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(RES / 'bootstrap_heldout_intended.csv', index=False)
print(f'\nWrote {RES / "bootstrap_heldout_intended.csv"}')
print(boot_df.to_string(index=False))

# ----------------------------------------------------------------------
# 4. Table S13: covariate balance matched (813) vs unmatched
# ----------------------------------------------------------------------
matched_ids = set(matched_llm_bootstrap['RespondentID'].unique())
heldout_int = heldout_int.copy()
heldout_int['is_matched'] = heldout_int.index.isin(matched_llm_bootstrap.index)

cols = ['WaitTime','VaccineEfficacy','SideEffects','CashIncentives','VaccineOrigin','Choice']
balance_rows = []
for col in cols:
    m_mean = heldout_int.loc[heldout_int['is_matched'], col].mean()
    u_mean = heldout_int.loc[~heldout_int['is_matched'], col].mean()
    m_std = heldout_int.loc[heldout_int['is_matched'], col].std()
    u_std = heldout_int.loc[~heldout_int['is_matched'], col].std()
    pooled = np.sqrt((m_std**2 + u_std**2) / 2)
    smd = (m_mean - u_mean) / pooled if pooled > 0 else 0
    balance_rows.append({
        'attribute': col,
        'matched_mean': f'{m_mean:.4f}',
        'unmatched_mean': f'{u_mean:.4f}',
        'SMD': f'{smd:.4f}',
        'matched_N': int(heldout_int['is_matched'].sum()),
        'unmatched_N': int((~heldout_int['is_matched']).sum()),
    })
balance_df = pd.DataFrame(balance_rows)
balance_df.to_csv(RES / 'covariate_balance_intended.csv', index=False)
print(f'\nWrote {RES / "covariate_balance_intended.csv"}')
print(balance_df.to_string(index=False))

# ----------------------------------------------------------------------
# 5. Table S14: subgroup heterogeneity (Static-BDT vs LLM, on 813-row subset)
# ----------------------------------------------------------------------
sub_rows = []
for stratum_name, stratum_mask in [
    ('All', matched_llm_bootstrap.index.notna()),
    ('Age < 55', matched_llm_bootstrap['Age'].isin(['18-24','25-34','35-44','45-54'])),
    ('Age >= 55', matched_llm_bootstrap['Age'].isin(['55-65','65+'])),
    ('Female', matched_llm_bootstrap['Gender']=='F'),
    ('Male', matched_llm_bootstrap['Gender']=='M'),
    ('College or Univ', matched_llm_bootstrap['Education']=='College or University'),
    ('Below College', matched_llm_bootstrap['Education']!='College or University'),
]:
    sub = matched_llm_bootstrap.loc[stratum_mask]
    if len(sub) < 10:
        continue
    y_s = sub['Choice'].values
    p_dce_s = sub['p_dce'].values
    p_llm_s = sub['p_llm'].values
    p_bdt_s = LAMBDA * p_llm_s + (1-LAMBDA) * p_dce_s
    try:
        from scipy.stats import spearmanr
        rho_bdt, _ = spearmanr(p_bdt_s, p_dce_s)
        rho_llm, _ = spearmanr(p_llm_s, p_dce_s)
    except Exception:
        rho_bdt = float('nan'); rho_llm = float('nan')
    sub_rows.append({
        'stratum': stratum_name,
        'N': len(sub),
        'BDT_log_loss': f'{log_loss(y_s, p_bdt_s):.4f}',
        'BDT_brier': f'{brier(y_s, p_bdt_s):.4f}',
        'BDT_rho_Pemp': f'{rho_bdt:.4f}' if not np.isnan(rho_bdt) else 'NA',
        'LLM_log_loss': f'{log_loss(y_s, p_llm_s):.4f}',
        'LLM_brier': f'{brier(y_s, p_llm_s):.4f}',
        'LLM_rho_Pemp': f'{rho_llm:.4f}' if not np.isnan(rho_llm) else 'NA',
    })
subgroup_df = pd.DataFrame(sub_rows)
subgroup_df.to_csv(RES / 'subgroup_heterogeneity_intended.csv', index=False)
print(f'\nWrote {RES / "subgroup_heterogeneity_intended.csv"}')
print(subgroup_df.to_string(index=False))

# ----------------------------------------------------------------------
# 6. Summary
# ----------------------------------------------------------------------
print('\n' + '='*72)
print('SUMMARY — canonical bundle (intended design + seed=2026)')
print('='*72)
print(f'Matched alt-rows: {len(matched_llm)}')
print(f'Distinct cells: {len(cells)} of 12 candidate')
print(f'2-alt tasks: {n_tasks}')
print(f'Unmatched held-out alt-rows: {len(unmatched)}')
print(f'\\nAlt-level log loss:')
print(f'  Pure-DCE:       {ll_dce_pt:.4f}')
print(f'  Static-BDT (λ=0.25): {ll_bdt_pt:.4f}')
print(f'  LLM:             {ll_llm_pt:.4f}')
print(f'\\nAlt-level Brier:')
print(f'  Pure-DCE:       {br_dce_pt:.4f}')
print(f'  Static-BDT:      {br_bdt_pt:.4f}')
print(f'  LLM:             {br_llm_pt:.4f}')
print(f'\\nAUC:')
print(f'  Pure-DCE:       {auc_dce_pt:.4f}')
print(f'  Static-BDT:      {auc_bdt_pt:.4f}')
print(f'  LLM:             {auc_llm_pt:.4f}')