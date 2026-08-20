"""Regenerate 64-state canonical bundle from CLEAN 6-param clogit (wait=2 excluded).

Provenance fix (Round 3):
- Main Table 1's 64-state frontier was generated from 5-param MXL (mxl_coefs.csv),
  contradicting the paper's "corrected 6-parameter grouped/conditional logit" claim.
- The correct 6-param formula is P = sigmoid(U(x) - beta_ASC), U = b_w*wait + b_e*eff + b_s*se
  (VO=0, cash=0 on the grid), which reproduces the stored P_static_6 to 1e-16.
- This script rebuilds the bundle from the clean 6-param fit (wait=2 excluded).

Outputs (results/):
- canonical_6param_fit_clean.csv            (coefs, wait=2 excluded)
- bdt_eval_grid_static_6param_clean.csv     (grid with P_static_6_clean)
- canonical_main_table1_clean.csv           (3 models x Raw/EFR full metrics)
- lambda_sensitivity_frontier_clean.csv     (Qwen 12-row lambda-sweep)
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from pathlib import Path
import os

REPO = Path('/Users/cary/bdt_repo')
RESULTS = REPO / 'results'

def fit_clogit(df):
    dce = df.copy()
    dce['alt_idx'] = dce['Alt'].map({'C': 0, 'A': 1, 'B': 2})
    dce['set_id'] = dce['RespondentID'].astype(str) + '_' + dce['Choiceset'].astype(str)
    sets = sorted(dce['set_id'].unique()); N = len(sets)
    X = np.zeros((N, 3, 6)); y = np.full(N, -1, dtype=int)
    sid = {s: i for i, s in enumerate(sets)}
    for _, r in dce.iterrows():
        s = sid[r['set_id']]; j = r['alt_idx']
        X[s, j, 0] = r['VaccineOrigin']; X[s, j, 1] = r['WaitTime']; X[s, j, 2] = r['VaccineEfficacy']
        X[s, j, 3] = r['SideEffects']; X[s, j, 4] = r['CashIncentives']; X[s, j, 5] = 1 if r['Alt'] == 'C' else 0
        if r['Choice'] == 1: y[s] = j
    def nll(b):
        u = X @ b; u -= u.max(axis=1, keepdims=True)
        p = np.exp(u) / np.exp(u).sum(axis=1, keepdims=True)
        return -np.log(p[np.arange(N), y] + 1e-12).sum()
    return minimize(nll, np.zeros(6), method='BFGS', options={'maxiter': 2000}).x

df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
b_clean = fit_clogit(df[df['WaitTime'] != 2])
names = ['VaccineOrigin', 'WaitTime', 'VaccineEfficacy', 'SideEffects', 'CashIncentives', 'ASC_optout']
pd.DataFrame({'term': names, 'coef': b_clean}).to_csv(RESULTS / 'canonical_6param_fit_clean.csv', index=False)
print('coefs (clean, wait=2 excluded):', dict(zip(names, np.round(b_clean, 4))))

grid = pd.read_csv(REPO / 'bdt_eval_grid_static_6param.csv')
w = grid['wait'].astype(float).values; e = grid['eff'].astype(float).values; s = grid['se'].astype(float).values
# Correct formula: P = sigmoid(U - ASC), U = b_w*w + b_e*e + b_s*s  (VO=0, cash=0)
p_ref = expit(b_clean[1]*w + b_clean[2]*e + b_clean[3]*s - b_clean[5])
grid_clean = grid.copy()
grid_clean['P_static_6_clean'] = p_ref
grid_clean.to_csv(RESULTS / 'bdt_eval_grid_static_6param_clean.csv', index=False)
print(f'P_static_6_clean range: {p_ref.min():.4f} - {p_ref.max():.4f}')

def mvr(p, axis='wait'):
    v = tot = 0
    for i in range(len(p)):
        for j in range(len(p)):
            if axis == 'wait' and w[j] > w[i] and e[j] == e[i] and s[j] == s[i]:
                tot += 1
                if p[j] > p[i]: v += 1
            elif axis == 'se' and s[j] > s[i] and w[j] == w[i] and e[j] == e[i]:
                tot += 1
                if p[j] > p[i]: v += 1
    return v, tot

def full_metrics(p_llm, lam):
    pi = lam*p_llm + (1-lam)*p_ref
    mse = float(np.mean((pi-p_ref)**2)); mae = float(np.mean(np.abs(pi-p_ref)))
    vw, tw = mvr(pi, 'wait'); vs, ts = mvr(pi, 'se')
    rho = float(spearmanr(pi, p_ref).correlation)
    return dict(MSE=mse, MAE=mae, MVR_W=vw, MVR_W_tot=tw, MVR_SE=vs, MVR_SE_tot=ts, rho=rho)

files = {
    'Qwen2.5-72B-Instruct': REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv',
    'DeepSeek V4': REPO / 'llm_parsed_outputs_deepseek_unconstrained.csv',
    'MiroThinker-1.7-mini': REPO / 'llm_parsed_outputs_mirothinker_unconstrained.csv',
}
rows = []
print('\n=== Table 1 (clean 6-param, correct P formula) ===')
for model, path in files.items():
    if not os.path.exists(path):
        print(f'MISSING {model}: {path}'); continue
    pllm = pd.read_csv(path)
    if 'parse_success' in pllm.columns:
        pllm = pllm[pllm['parse_success'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    p_llm = pllm.groupby('state')['probability_0_1'].mean().sort_index().values
    print(f'\n{model}:')
    for tag, lam in [('Raw', 1.0), ('EFR', 0.25)]:
        m = full_metrics(p_llm, lam)
        rows.append({'model': model, 'condition': tag, 'MSE': m['MSE'], 'MAE': m['MAE'],
                     'MVR_Wait': m['MVR_W'], 'MVR_Wait_total': m['MVR_W_tot'],
                     'MVR_SE': m['MVR_SE'], 'MVR_SE_total': m['MVR_SE_tot'], 'rho': m['rho']})
        print(f"  {tag:3s}: MSE={m['MSE']:.4f} MAE={m['MAE']:.4f} "
              f"MVR-W={100*m['MVR_W']/m['MVR_W_tot']:.1f}% MVR-SE={100*m['MVR_SE']/m['MVR_SE_tot']:.1f}% "
              f"rho={m['rho']:+.3f}")
pd.DataFrame(rows).to_csv(RESULTS / 'canonical_main_table1_clean.csv', index=False)
print('\nWrote canonical_main_table1_clean.csv')

# Qwen lambda-sweep 12 rows
pllm = pd.read_csv(files['Qwen2.5-72B-Instruct'])
if 'parse_success' in pllm.columns:
    pllm = pllm[pllm['parse_success'].astype(str).str.lower().isin(['true', '1', 'yes'])]
p_llm = pllm.groupby('state')['probability_0_1'].mean().sort_index().values
lams = [0.00, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
sweep = []
for lam in lams:
    m = full_metrics(p_llm, lam)
    sweep.append({'lambda': f'{lam:.2f}', 'main_spec': 'yes' if abs(lam-0.25) < 1e-9 else '',
                  'MSE': m['MSE'], 'MAE': m['MAE'],
                  'MVR_Wait': m['MVR_W']/m['MVR_W_tot'], 'MVR_SE': m['MVR_SE']/m['MVR_SE_tot'],
                  'Spearman_rho': m['rho'],
                  'note': 'Pure-DCE' if lam == 0 else ('Unconstrained LLM' if lam == 1 else '')})
pd.DataFrame(sweep).to_csv(RESULTS / 'lambda_sensitivity_frontier_clean.csv', index=False)
print('\n=== Qwen lambda-sweep (clean 6-param) ===')
print(pd.DataFrame(sweep).to_string(index=False))
