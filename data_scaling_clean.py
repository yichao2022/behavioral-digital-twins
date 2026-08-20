"""Data-scaling ablation with CLEAN 6-param frontier (wait=2 excluded).

Same logic as data_scaling_fixed_lambda025.py but:
- p_full = P_static_6_clean (correct formula P = sigmoid(U - ASC))
- subsample fits use the same 6-param clogit on respondent subsamples (wait=2 excluded)
- fixed lambda = 0.25, Qwen2.5-72B
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
import random, os

REPO = Path('/Users/cary/bdt_repo')
GRID = pd.read_csv(REPO / 'results/bdt_eval_grid_static_6param_clean.csv')
P_FULL = dict(zip(GRID['state'].astype(int), GRID['P_static_6_clean']))
SEEDS = [2026, 2027, 2028, 2029, 2030]

LLM = pd.read_csv(REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv')
if 'parse_success' in LLM.columns:
    LLM = LLM[LLM['parse_success'].astype(str).str.lower().isin(['true', '1', 'yes'])]
P_LLM = LLM.groupby('state')['probability_0_1'].mean().sort_index()

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

def p_static_from_coefs(b):
    w = GRID['wait'].astype(float).values; e = GRID['eff'].astype(float).values; s = GRID['se'].astype(float).values
    return expit(b[1]*w + b[2]*e + b[3]*s - b[5])

def mvr_wait(pi, p_ref):
    w = GRID['wait'].astype(float).values; e = GRID['eff'].astype(float).values; s = GRID['se'].astype(float).values
    v = tot = 0
    for i in range(len(pi)):
        for j in range(len(pi)):
            if w[j] > w[i] and e[j] == e[i] and s[j] == s[i]:
                tot += 1
                if pi[j] > pi[i]: v += 1
    return v / tot if tot else 0

def metrics(ps, target='full'):
    pi = 0.25 * P_LLM.values + 0.75 * np.array(ps)
    p_ref = np.array([P_FULL[s] for s in sorted(P_FULL)]) if target == 'full' else np.array(ps)
    mse = float(np.mean((pi - p_ref) ** 2))
    mvr = mvr_wait(pi, p_ref)
    rho = float(spearmanr(pi, p_ref).correlation)
    return mse, mvr, rho

dce = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
dce = dce[dce['WaitTime'] != 2]  # clean rule before everything
rids = sorted(dce['RespondentID'].unique())

rows = []
for share in [0.30, 0.50, 0.70, 1.00]:
    # full-sample benchmark: vs P_FULL
    if share >= 1.0 - 1e-9:
        ps = [P_FULL[s] for s in sorted(P_FULL)]
        mse, mvr, rho = metrics(ps)
        rows.append({'share': '100%', 'benchmark': 'Full-sample canonical empirical frontier', 'MSE': mse, 'MVR_Wait': mvr, 'rho': rho})
        rows.append({'share': '100%', 'benchmark': 'Subsample-reestimated frontier', 'MSE': mse, 'MVR_Wait': mvr, 'rho': rho})
        continue
    n = max(1, int(round(share * len(rids))))
    # full-sample benchmark: mean over seeds
    mse_l, mvr_l, rho_l = [], [], []
    # own-frontier benchmark
    mse_o, mvr_o, rho_o = [], [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        samp = set(rng.sample(rids, n))
        sub = dce[dce['RespondentID'].isin(samp)]
        b = fit_clogit(sub)
        ps_sub = p_static_from_coefs(b)
        m1, m2, m3 = metrics(ps_sub, 'full')  # vs full-sample frontier
        mse_l.append(m1); mvr_l.append(m2); rho_l.append(m3)
        m1, m2, m3 = metrics(ps_sub, 'own')   # vs own frontier
        mse_o.append(m1); mvr_o.append(m2); rho_o.append(m3)
    rows.append({'share': f'{int(share*100)}%', 'benchmark': 'Full-sample canonical empirical frontier', 'MSE': np.mean(mse_l), 'MVR_Wait': np.mean(mvr_l), 'rho': np.mean(rho_l)})
    rows.append({'share': f'{int(share*100)}%', 'benchmark': 'Subsample-reestimated frontier', 'MSE': np.mean(mse_o), 'MVR_Wait': np.mean(mvr_o), 'rho': np.mean(rho_o)})

out = pd.DataFrame(rows)
out.to_csv(REPO / 'results/data_scaling_clean_table.csv', index=False)
print(out.to_string(index=False))
