"""λ-sweep for held-out DCE predictive validation (canonical intended-design).

Exact same pipeline as recompute_intended_design.py:
- drop wait=2 BEFORE respondent split (intended design, wait ∈ {0,1,3,6})
- seed=2026, train_frac=0.80, 5-param logit refit on intended train non-optout
- matched alt-rows: wait {0,6} × eff {0.5,0.7} × se {1,2,3}, 11-of-12 cells, N=813
- LLM: mean probability_0_1 per grid state (Qwen unconstrained)

Output: results/lambda_sensitivity_heldout_intended.csv (12 rows, λ 0.00..1.00)
"""
import pandas as pd
import numpy as np
import random
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from pathlib import Path

SEED = 2026
TRAIN_FRAC = 0.80
REPO = Path('/Users/cary/bdt_repo')
EPS = 1e-12
LAMBDAS = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]

df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
# 0) INTENDED DESIGN FILTER BEFORE SPLIT
intended = df[df['WaitTime'] != 2].copy()
rids = sorted(intended['RespondentID'].unique())
rng = random.Random(SEED); shuffled = rids[:]; rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train]); test_ids = set(shuffled[n_train:])
heldout = intended[intended['RespondentID'].isin(test_ids)].copy()

# 1) matched subset
nonopt = heldout[(heldout['CashIncentives']>0) & (heldout['VaccineEfficacy']>0) & (heldout['SideEffects']>0)].copy()
matched = nonopt[nonopt['WaitTime'].isin({0,6}) & nonopt['VaccineEfficacy'].isin({0.5,0.7}) & nonopt['SideEffects'].isin({1.0,2.0,3.0})].copy()

# 2) refit 5-param logit on intended train non-optout
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

# 3) predictions on matched
Xm = np.column_stack([np.ones(len(matched)), matched['WaitTime'], matched['VaccineEfficacy'], matched['SideEffects'], matched['CashIncentives'], matched['VaccineOrigin']])
ym = matched['Choice'].values
p_dce = expit(Xm @ beta_train)

# 4) LLM p per grid state
pllm = pd.read_csv(REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv')
pllm_by_state = pllm.groupby('state')['probability_0_1'].mean().to_dict()
grid = pd.read_csv(REPO / 'bdt_eval_grid_static.csv')
def state_of(row):
    m = grid[(grid['wait']==row['WaitTime']) & (grid['eff']==row['VaccineEfficacy']) & (grid['se']==row['SideEffects'])]
    return None if len(m)==0 else m.iloc[0]['state']
matched = matched.copy()
matched['state'] = matched.apply(state_of, axis=1)
matched['p_llm'] = matched['state'].map(pllm_by_state)
print('matched rows:', len(matched), '| LLM unmatched:', matched['p_llm'].isna().sum())
matched = matched.dropna(subset=['p_llm'])
ym = matched['Choice'].values
p_llm = matched['p_llm'].values
p_dce = expit(np.column_stack([np.ones(len(matched)), matched['WaitTime'], matched['VaccineEfficacy'], matched['SideEffects'], matched['CashIncentives'], matched['VaccineOrigin']]) @ beta_train)

def log_loss(y, p): p = np.clip(p, EPS, 1-EPS); return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
def brier(y, p): return np.mean((y-p)**2)

rows = []
for lam in LAMBDAS:
    p = lam*p_llm + (1-lam)*p_dce
    ll = log_loss(ym, p); br = brier(ym, p)
    auc = roc_auc_score(ym, p) if len(np.unique(ym)) > 1 else np.nan
    zp = np.log(np.clip(p, EPS, 1-EPS)/(1-np.clip(p, EPS, 1-EPS)))
    from sklearn.linear_model import LogisticRegression
    Xc = zp.reshape(-1,1)
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=200).fit(Xc, ym)
        ci, cs = lr.intercept_[0], lr.coef_[0][0]
    except Exception:
        ci = cs = np.nan
    note = 'Pure-DCE' if lam == 0.0 else ('Unconstrained LLM' if lam == 1.0 else ('Main' if lam == 0.25 else ''))
    rows.append({'lambda': lam, 'N': len(matched), 'log_loss': ll, 'brier': br, 'AUC': auc, 'cal_int': ci, 'cal_slope': cs, 'note': note})
    print(f'{lam:>6.2f} {len(matched):>5d} {ll:>9.4f} {br:>8.4f} {auc:>8.4f} {ci:>8.4f} {cs:>9.4f} {note}')

out = pd.DataFrame(rows)
out.to_csv(REPO / 'results/lambda_sensitivity_heldout_intended.csv', index=False)
print('\nWrote results/lambda_sensitivity_heldout_intended.csv')
