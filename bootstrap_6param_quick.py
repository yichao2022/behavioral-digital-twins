#!/usr/bin/env python3
"""Quick bootstrap for 6-param model (500 iterations)."""
from pathlib import Path
import numpy as np
import pandas as pd
import json
from scipy.optimize import minimize
from collections import defaultdict

WS = Path('/Users/cary/bdt_repo')
DCE = Path('/Users/cary/.openclaw/workspace/behavioral-digital-twins/analysis_output/dce_encoded.csv')
LLM_PATH = Path('/Users/cary/bdt_repo/llm_parsed_outputs_qwen72b_unconstrained.csv')

N_BOOT = 500
SEED = 2026

print(f"BOOTSTRAP: {N_BOOT} iterations")

# Load data
dce = pd.read_csv(DCE)
dce['alt_idx'] = dce['Alt'].map({'C': 0, 'A': 1, 'B': 2})
dce['rid_str'] = dce['RespondentID'].astype(str).str.zfill(5)
dce['set_id'] = dce['rid_str'] + '_' + dce['Choiceset'].astype(str)

# Load LLM
llm = pd.read_csv(LLM_PATH)
if 'parse_success' in llm.columns:
    llm = llm[llm['parse_success'].astype(str).str.lower().isin(['true', '1', 'yes'])]
p_llm = llm.groupby('state')['probability_0_1'].mean().to_dict()
states = sorted(p_llm.keys(), key=lambda s: int(s))
p_llm_arr = np.array([p_llm[s] for s in states])

# Load grid
grid = pd.read_csv(WS / 'bdt_eval_grid_static.csv')
state_meta = {r['state']: {'wait': float(r['wait']), 'eff': float(r['eff']), 'se': float(r['se'])}
              for _, r in grid.iterrows()}

# Build full design matrix
N_sets = dce['set_id'].nunique()
X_full = np.zeros((N_sets, 3, 6))
y_full = np.full(N_sets, -1, dtype=int)
sid_to_idx = {sid: i for i, sid in enumerate(sorted(dce['set_id'].unique()))}

for _, r in dce.iterrows():
    s = sid_to_idx[r['set_id']]
    j = r['alt_idx']
    X_full[s, j, 0] = r['VaccineOrigin']
    X_full[s, j, 1] = r['WaitTime']
    X_full[s, j, 2] = r['VaccineEfficacy']
    X_full[s, j, 3] = r['SideEffects']
    X_full[s, j, 4] = r['CashIncentives']
    X_full[s, j, 5] = 1 if r['Alt'] == 'C' else 0
    if r['Choice'] == 1:
        y_full[s] = j

rid_to_sets = defaultdict(list)
for sid, idx in sid_to_idx.items():
    rid = sid.split('_')[0]
    rid_to_sets[rid].append(idx)

coef_ref = np.array([0.244230, -0.059244, 0.413207, -0.036433, 0.001048, -0.213815])

def nll(b, X, y):
    utils = X @ b
    utils -= utils.max(axis=1, keepdims=True)
    expu = np.exp(utils)
    probs = expu / expu.sum(axis=1, keepdims=True)
    p_chosen = probs[np.arange(len(y)), y]
    return -np.log(p_chosen + 1e-12).sum()

def compute_p_static(coef, grid_df):
    U_vaccine = coef[0] * 0 + coef[1] * grid_df['wait'] + coef[2] * grid_df['eff'] + coef[3] * grid_df['se'] + coef[4] * 0
    U_optout = coef[5]
    exp_vaccine = np.exp(U_vaccine)
    exp_optout = np.exp(U_optout)
    return exp_vaccine / (exp_vaccine + exp_optout)

def spearman_rho(x, y):
    n = len(x)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    return float(1 - 6 * (d ** 2).sum() / (n * (n * n - 1)))

print(f"Running {N_BOOT} bootstrap iterations...")
boot_rhos = []
boot_efr = []

rids = list(rid_to_sets.keys())

for i in range(N_BOOT):
    if i % 100 == 0:
        print(f"  {i}/{N_BOOT}")
    
    np.random.seed(SEED + i)
    boot_rids = np.random.choice(rids, len(rids), replace=True)
    boot_sets = []
    for rid in boot_rids:
        boot_sets.extend(rid_to_sets[rid])
    boot_sets = list(set(boot_sets))
    
    Xb = X_full[boot_sets]
    yb = y_full[boot_sets]
    
    best = None
    for b0 in [np.zeros(6), coef_ref]:
        r = minimize(lambda b: nll(b, Xb, yb), b0, method='L-BFGS-B', options={'gtol': 1e-10, 'maxiter': 10000})
        if best is None or r.fun < best.fun:
            best = r
    
    p_static_boot = compute_p_static(best.x, grid)
    p_static_arr = np.array([p_static_boot[int(s)-1] for s in states])
    
    boot_rhos.append(spearman_rho(p_static_arr, p_llm_arr))
    boot_efr.append(p_static_boot.mean())

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

rho_ci = (np.percentile(boot_rhos, 2.5), np.percentile(boot_rhos, 97.5))
efr_ci = (np.percentile(boot_efr, 2.5), np.percentile(boot_efr, 97.5))

print(f"\nSpearman ρ: {np.mean(boot_rhos):.4f}")
print(f"  SD: {np.std(boot_rhos):.4f}")
print(f"  95% CI: [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]")

print(f"\nEFR (mean P_static): {np.mean(boot_efr):.4f}")
print(f"  SD: {np.std(boot_efr):.4f}")
print(f"  95% CI: [{efr_ci[0]:.4f}, {efr_ci[1]:.4f}]")

results = {
    'n_bootstrap': N_BOOT,
    'spearman_rho': {'mean': float(np.mean(boot_rhos)), 'std': float(np.std(boot_rhos)), 'ci95': [float(rho_ci[0]), float(rho_ci[1])]},
    'efr': {'mean': float(np.mean(boot_efr)), 'std': float(np.std(boot_efr)), 'ci95': [float(efr_ci[0]), float(efr_ci[1])]}
}

with open(WS / 'results' / 'bootstrap_6param_500.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {WS / 'results' / 'bootstrap_6param_500.json'}")
