"""Bootstrap CIs + subgroup heterogeneity, canonical intended-design frontier.

Same pipeline as recompute_intended_design.py / lambda_sweep_intended.py:
- drop wait=2 BEFORE split, seed=2026, 80/20
- 5-param logit refit on intended train non-optout (const, wait, eff, se, cash, origin)
- matched alt-rows N=813, wait {0,6} × eff {0.5,0.7} × se {1,2,3}

Outputs:
- results/bootstrap_heldout_intended.csv (9 rows: 3 models × logloss/brier/AUC, 2000 reps)
- results/subgroup_heterogeneity_intended.csv
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

SEED = 2026
TRAIN_FRAC = 0.80
REPO = Path('/Users/cary/bdt_repo')
EPS = 1e-12
N_BOOT = 2000
LAMBDA = 0.25
RNG = random.Random(2026)

df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
intended = df[df['WaitTime'] != 2].copy()
rids = sorted(intended['RespondentID'].unique())
rng = random.Random(SEED); shuffled = rids[:]; rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train]); test_ids = set(shuffled[n_train:])
heldout = intended[intended['RespondentID'].isin(test_ids)].copy()

nonopt = heldout[(heldout['CashIncentives']>0) & (heldout['VaccineEfficacy']>0) & (heldout['SideEffects']>0)].copy()
matched = nonopt[nonopt['WaitTime'].isin({0,6}) & nonopt['VaccineEfficacy'].isin({0.5,0.7}) & nonopt['SideEffects'].isin({1.0,2.0,3.0})].copy()

train_dce = intended[(intended['RespondentID'].isin(train_ids)) & (intended['CashIncentives']>0) & (intended['VaccineEfficacy']>0) & (intended['SideEffects']>0)].copy()
Xtr = np.column_stack([np.ones(len(train_dce)), train_dce['WaitTime'], train_dce['VaccineEfficacy'], train_dce['SideEffects'], train_dce['CashIncentives'], train_dce['VaccineOrigin']])
ytr = train_dce['Choice'].values

def fit_irls(X, y, max_iter=100):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta; mu = expit(eta); w = np.clip(mu*(1-mu), 1e-8, None)
        z = eta + (y - mu) / w
        try:
            beta_new = np.linalg.solve(X.T @ (X * w[:,None]), X.T @ (z * w))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new; break
        beta = beta_new
    return beta

beta_train = fit_irls(Xtr, ytr)

def X_of(sub):
    return np.column_stack([np.ones(len(sub)), sub['WaitTime'], sub['VaccineEfficacy'], sub['SideEffects'], sub['CashIncentives'], sub['VaccineOrigin']])

# LLM lookup
pllm = pd.read_csv(REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv')
pllm_by_state = pllm.groupby('state')['probability_0_1'].mean().to_dict()
grid = pd.read_csv(REPO / 'bdt_eval_grid_static.csv')
def state_of(row):
    m = grid[(grid['wait']==row['WaitTime']) & (grid['eff']==row['VaccineEfficacy']) & (grid['se']==row['SideEffects'])]
    return None if len(m)==0 else m.iloc[0]['state']
matched = matched.copy()
matched['state'] = matched.apply(state_of, axis=1)
matched['p_llm'] = matched['state'].map(pllm_by_state)
matched = matched.dropna(subset=['p_llm'])
y = matched['Choice'].values
p_dce = expit(X_of(matched) @ beta_train)
p_llm = matched['p_llm'].values

def log_loss(y, p): p = np.clip(p, EPS, 1-EPS); return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
def brier(y, p): return np.mean((y-p)**2)

# ---- bootstrap CIs ----
print('[bootstrap] 2000 reps respondent-cluster...')
resp_ids = matched['RespondentID'].values
unique_resp = np.unique(resp_ids)
idx_by_resp = {r: np.where(resp_ids == r)[0] for r in unique_resp}
boot_rows = []
for lam, label in [(0.00,'Pure-DCE'), (0.25,'Static-BDT'), (1.00,'LLM')]:
    p = lam*p_llm + (1-lam)*p_dce
    pt_ll, pt_br, pt_auc = log_loss(y,p), brier(y,p), roc_auc_score(y,p)
    boot_ll, boot_br, boot_auc = [], [], []
    for _ in range(N_BOOT):
        chosen = RNG.choices(unique_resp.tolist(), k=len(unique_resp))
        idx = np.concatenate([idx_by_resp[r] for r in chosen])
        yb, pb = y[idx], p[idx]
        boot_ll.append(log_loss(yb, pb))
        boot_br.append(brier(yb, pb))
        boot_auc.append(roc_auc_score(yb, pb))
    boot_rows.append({'model': label, 'metric': 'log_loss', 'point': pt_ll,
                 'CI_low': np.percentile(boot_ll,2.5), 'CI_high': np.percentile(boot_ll,97.5), 'n_boot': N_BOOT})
    boot_rows.append({'model': label, 'metric': 'brier', 'point': pt_br,
                 'CI_low': np.percentile(boot_br,2.5), 'CI_high': np.percentile(boot_br,97.5), 'n_boot': N_BOOT})
    boot_rows.append({'model': label, 'metric': 'AUC', 'point': pt_auc,
                 'CI_low': np.percentile(boot_auc,2.5), 'CI_high': np.percentile(boot_auc,97.5), 'n_boot': N_BOOT})
    print(f'  {label:10s} ll={pt_ll:.4f} [{np.percentile(boot_ll,2.5):.4f},{np.percentile(boot_ll,97.5):.4f}] br={pt_br:.4f} auc={pt_auc:.4f}')
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(REPO/'results/bootstrap_heldout_intended.csv', index=False)
print('Wrote bootstrap_heldout_intended.csv')

# ---- subgroup ----
print('[subgroup]')
demo = heldout[['RespondentID','Age','Gender','Education']].drop_duplicates('RespondentID').copy()
def age_mid(a):
    s = str(a)
    try:
        if '-' in s:
            lo, hi = s.split('-'); return (int(lo)+int(hi))/2
        if s.endswith('+'): return int(s[:-1])+2
        return float(s)
    except Exception: return 99
demo['age_mid'] = demo['Age'].apply(age_mid)
demo_lookup = demo.set_index('RespondentID').to_dict('index')
g = matched.merge(demo[['RespondentID','age_mid','Gender','Education']], on='RespondentID', how='left', suffixes=('','_demo'))
g = g.drop(columns=['Gender_demo'], errors='ignore')
sub_rows = []
def add_sub(name, mask):
    sub = g[mask]
    if len(sub) == 0: return
    ys, pd_, pl = sub['Choice'].values, expit(X_of(sub) @ beta_train), sub['p_llm'].values
    pefr = LAMBDA*pl + (1-LAMBDA)*pd_
    sub_rows.append({'stratum': name, 'N': len(sub),
        'BDT_log_loss': log_loss(ys, pefr), 'BDT_brier': brier(ys, pefr),
        'BDT_rho_Pemp': spearmanr(pefr, pd_).correlation,
        'LLM_log_loss': log_loss(ys, pl), 'LLM_brier': brier(ys, pl),
        'LLM_rho_Pemp': spearmanr(pl, pd_).correlation})
    print(f'  {name:18s} N={len(sub):4d} BDT={log_loss(ys,pefr):.4f}/{brier(ys,pefr):.4f}/{spearmanr(pefr,pd_).correlation:+.3f} LLM={log_loss(ys,pl):.4f}/{brier(ys,pl):.4f}/{spearmanr(pl,pd_).correlation:+.3f}')
add_sub('All', np.ones(len(g), dtype=bool))
add_sub('Age < 55', g['age_mid'] < 55)
add_sub('Age >= 55', g['age_mid'] >= 55)
add_sub('Female', g['Gender'] == 'F')
add_sub('Male', g['Gender'] == 'M')
add_sub('College or Univ', g['Education'].astype(str).str.contains('College|Univ', na=False))
add_sub('Below College', ~g['Education'].astype(str).str.contains('College|Univ', na=False))
sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv(REPO/'results/subgroup_heterogeneity_intended.csv', index=False)
print('Wrote subgroup_heterogeneity_intended.csv')
