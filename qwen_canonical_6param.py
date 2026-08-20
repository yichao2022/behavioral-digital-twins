#!/usr/bin/env python3
"""
Single Source of Truth (SST) for the 6-parameter canonical frontier (with ASC).
Computes ALL metrics needed for Table 1, S3, S5, S6, S10, S11 from one source:
  - 6-param grouped clogit fit on full DCE (5 attributes + ASC for opt-out)
  - P_static_6 column from this fit
  - Qwen unconstrained LLM outputs (canonical slice)
  - OOD Qwen + DeepSeek raw results
  - Bootstrap (respondent-cluster, 2000 reps)
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

WS = Path('/Users/cary/bdt_repo')
DCE = Path('/Users/cary/.openclaw/workspace/behavioral-digital-twins/analysis_output/dce_encoded.csv')
GRID_IN = WS / 'bdt_eval_grid_static.csv'
GRID_OUT = WS / 'bdt_eval_grid_static_6param.csv'
COEF_OUT = WS / 'results' / 'canonical_6param_fit.csv'
METRICS_OUT = WS / 'results' / 'qwen_canonical_metrics_6param.json'
LLM_PATH = Path('/Users/cary/bdt_repo/llm_parsed_outputs_qwen72b_unconstrained.csv')

# === 1) Fit 6-param clogit on full DCE ===
print("Step 1: Fit 6-param clogit on full DCE (with ASC)")
dce = pd.read_csv(DCE)
dce['alt_idx'] = dce['Alt'].map({'C': 0, 'A': 1, 'B': 2})
dce['rid_str'] = dce['RespondentID'].astype(str).str.zfill(5)
dce['set_id'] = dce['rid_str'] + '_' + dce['Choiceset'].astype(str)

N_sets = dce['set_id'].nunique()
X = np.zeros((N_sets, 3, 6))  # 6 params: 5 attributes + ASC
y = np.full(N_sets, -1, dtype=int)
sid_to_idx = {sid: i for i, sid in enumerate(sorted(dce['set_id'].unique()))}
for _, r in dce.iterrows():
    s = sid_to_idx[r['set_id']]
    j = r['alt_idx']
    X[s, j, 0] = r['VaccineOrigin']
    X[s, j, 1] = r['WaitTime']
    X[s, j, 2] = r['VaccineEfficacy']
    X[s, j, 3] = r['SideEffects']
    X[s, j, 4] = r['CashIncentives']
    X[s, j, 5] = 1 if r['Alt'] == 'C' else 0  # ASC for opt-out
    if r['Choice'] == 1:
        y[s] = j


def nll(b):
    utils = X @ b
    utils -= utils.max(axis=1, keepdims=True)
    expu = np.exp(utils)
    probs = expu / expu.sum(axis=1, keepdims=True)
    p_chosen = probs[np.arange(len(y)), y]
    return -np.log(p_chosen + 1e-12).sum()


best = None
for b0 in [np.zeros(6), np.array([0.5, -0.27, 1.46, -0.10, 0.0029, -0.2])]:
    r = minimize(nll, b0, method='L-BFGS-B', options={'gtol': 1e-10, 'maxiter': 10000})
    if best is None or r.fun < best.fun:
        best = r
coefs = best.x
print(f"  coefs: VO={coefs[0]:.4f}, wait={coefs[1]:.4f}, eff={coefs[2]:.4f}, "
      f"se={coefs[3]:.4f}, cash={coefs[4]:.6f}, ASC={coefs[5]:.4f}")
print(f"  nll: {best.fun:.2f}")

with open(COEF_OUT, 'w') as f:
    w = csv.writer(f)
    w.writerow(['term', 'coef'])
    for term, val in zip(['VaccineOrigin', 'WaitTime', 'VaccineEfficacy', 'SideEffects', 'CashIncentives', 'ASC_optout'], coefs):
        w.writerow([term, f"{val:.6f}"])
print(f"  Saved {COEF_OUT}")

# === 2) Compute P_static for 64-state grid ===
print("\nStep 2: Compute P_static for 64-state grid")
grid_in = pd.read_csv(GRID_IN)
# For grid: VaccineOrigin=0 (domestic), CashIncentives=0, ASC applies to opt-out
P_static = 1 / (1 + np.exp(-(
    coefs[0] * 0 + coefs[1] * grid_in['wait'] + coefs[2] * grid_in['eff']
    + coefs[3] * grid_in['se'] + coefs[4] * 0 + coefs[5]  # ASC for opt-out
)))
grid_out = grid_in.copy()
grid_out['P_static_6'] = P_static
grid_out.to_csv(GRID_OUT, index=False)
print(f"  P_static_6 range: {P_static.min():.4f}-{P_static.max():.4f}, mean={P_static.mean():.4f}")
print(f"  Saved {GRID_OUT}")

# === 3) Load Qwen LLM unconstrained outputs ===
print("\nStep 3: Load Qwen unconstrained LLM outputs")
llm = pd.read_csv(LLM_PATH)
if 'parse_success' in llm.columns:
    llm = llm[llm['parse_success'].astype(str).str.lower().isin(['true', '1', 'yes'])]
print(f"  N LLM rows: {len(llm)}, N states: {llm['state'].nunique()}")
p_llm = llm.groupby('state')['probability_0_1'].mean().to_dict()
states = sorted(p_llm.keys(), key=lambda s: int(s))

# === 4) Compute metrics ===
print("\nStep 4: Compute metrics")
state_meta = {r['state']: {'wait': float(r['wait']), 'eff': float(r['eff']), 'se': float(r['se'])}
              for _, r in grid_in.iterrows()}
ss = states
p_static_arr = np.array([P_static[int(s)-1] for s in ss])
p_llm_arr = np.array([p_llm[s] for s in ss])


def spearman_rho(x, y):
    n = len(x)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    return float(1 - 6 * (d ** 2).sum() / (n * (n * n - 1)))


def chr_violation(p_dict, meta, axis='wait'):
    pairs = []
    keys = sorted(meta.keys(), key=lambda s: int(s))
    for i, s1 in enumerate(keys):
        for s2 in keys[i+1:]:
            m1, m2 = meta[s1], meta[s2]
            if axis == 'wait':
                if m1['wait'] < m2['wait'] and m1['eff'] == m2['eff'] and m1['se'] == m2['se']:
                    pairs.append((s1, s2))
            elif axis == 'se':
                if m1['se'] < m2['se'] and m1['wait'] == m2['wait'] and m1['eff'] == m2['eff']:
                    pairs.append((s1, s2))
    viol = sum(1 for s1, s2 in pairs if p_dict[s1] < p_dict[s2])
    return viol, len(pairs)


rho = spearman_rho(p_static_arr, p_llm_arr)
print(f"  Spearman rho: {rho:.4f}")

chr_wait_viol, chr_wait_total = chr_violation({s: P_static[int(s)-1] for s in ss}, state_meta, 'wait')
chr_se_viol, chr_se_total = chr_violation({s: P_static[int(s)-1] for s in ss}, state_meta, 'se')
print(f"  CHR wait: {chr_wait_viol}/{chr_wait_total}")
print(f"  CHR se: {chr_se_viol}/{chr_se_total}")

# === 5) Bootstrap (respondent-cluster, 2000 reps) ===
print("\nStep 5: Bootstrap (respondent-cluster, 2000 reps)")


# Build respondent-to-choice-sets mapping
rid_to_sets = defaultdict(list)
for sid, idx in sid_to_idx.items():
    rid = sid.split('_')[0]
    rid_to_sets[rid].append(idx)


# Pre-compute X and y as DataFrames for easy indexing
X_df = pd.DataFrame({
    'set_idx': [sid_to_idx[r['set_id']] for _, r in dce.iterrows()],
    'alt_idx': dce['alt_idx'],
    'x0': dce['VaccineOrigin'],
    'x1': dce['WaitTime'],
    'x2': dce['VaccineEfficacy'],
    'x3': dce['SideEffects'],
    'x4': dce['CashIncentives'],
    'asc': (dce['Alt'] == 'C').astype(int),
    'choice': dce['Choice']
})


# Pre-compute grid utilities for bootstrap
grid_utils = (
    coefs[0] * 0 + coefs[1] * grid_in['wait'] + coefs[2] * grid_in['eff']
    + coefs[3] * grid_in['se'] + coefs[4] * 0 + coefs[5]
)
grid_p_static = 1 / (1 + np.exp(-grid_utils))


def bootstrap_iter(seed):
    np.random.seed(seed)
    rids = list(rid_to_sets.keys())
    boot_rids = np.random.choice(rids, len(rids), replace=True)
    
    # Collect bootstrapped sets
    boot_sets = []
    for rid in boot_rids:
        boot_sets.extend(rid_to_sets[rid])
    boot_sets = list(set(boot_sets))  # unique
    
    # Subset X and y
    mask = X_df['set_idx'].isin(boot_sets)
    boot_df = X_df[mask].copy()
    
    # Build matrices
    n_sets = len(boot_sets)
    Xb = np.zeros((n_sets, 3, 6))
    yb = np.full(n_sets, -1, dtype=int)
    set_map = {s: i for i, s in enumerate(sorted(boot_sets))}
    
    for _, r in boot_df.iterrows():
        s = set_map[r['set_idx']]
        j = int(r['alt_idx'])
        Xb[s, j, 0] = r['x0']
        Xb[s, j, 1] = r['x1']
        Xb[s, j, 2] = r['x2']
        Xb[s, j, 3] = r['x3']
        Xb[s, j, 4] = r['x4']
        Xb[s, j, 5] = r['asc']
        if r['choice'] == 1:
            yb[s] = j
    
    # Fit
    def nll_b(b):
        utils = Xb @ b
        utils -= utils.max(axis=1, keepdims=True)
        expu = np.exp(utils)
        probs = expu / expu.sum(axis=1, keepdims=True)
        p_chosen = probs[np.arange(len(yb)), yb]
        return -np.log(p_chosen + 1e-12).sum()
    
    best_b = None
    for b0 in [np.zeros(6), coefs]:
        r = minimize(nll_b, b0, method='L-BFGS-B', options={'gtol': 1e-10, 'maxiter': 10000})
        if best_b is None or r.fun < best_b.fun:
            best_b = r
    
    # Metrics on full grid
    grid_p = 1 / (1 + np.exp(-(
        best_b.x[0] * 0 + best_b.x[1] * grid_in['wait'] + best_b.x[2] * grid_in['eff']
        + best_b.x[3] * grid_in['se'] + best_b.x[4] * 0 + best_b.x[5]
    )))
    
    p_boot = np.array([grid_p[int(s)-1] for s in ss])
    rho_boot = spearman_rho(p_boot, p_llm_arr)
    
    return {
        'coef': best_b.x,
        'rho': rho_boot,
        'chr_wait_viol': chr_violation({s: grid_p[int(s)-1] for s in ss}, state_meta, 'wait')[0],
        'chr_se_viol': chr_violation({s: grid_p[int(s)-1] for s in ss}, state_meta, 'se')[0]
    }


print("  Running 2000 bootstrap iterations...")
boot_results = [bootstrap_iter(2026 + i) for i in range(2000)]

boot_rhos = [r['rho'] for r in boot_results]
boot_chr_wait = [r['chr_wait_viol'] for r in boot_results]
boot_chr_se = [r['chr_se_viol'] for r in boot_results]

rho_ci = (float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5)))
print(f"  Spearman rho: {rho:.4f} [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]")

# === 6) Save all metrics ===
print("\nStep 6: Save metrics")
metrics = {
    'model': '6-param with ASC',
    'ASC': float(coefs[5]),
    'coefs': {
        'VaccineOrigin': float(coefs[0]),
        'WaitTime': float(coefs[1]),
        'VaccineEfficacy': float(coefs[2]),
        'SideEffects': float(coefs[3]),
        'CashIncentives': float(coefs[4]),
        'ASC_optout': float(coefs[5])
    },
    'spearman_rho': float(rho),
    'spearman_rho_ci95': rho_ci,
    'chr_wait': {'violations': chr_wait_viol, 'total': chr_wait_total},
    'chr_se': {'violations': chr_se_viol, 'total': chr_se_total},
    'bootstrap': {
        'n': 2000,
        'rho_mean': float(np.mean(boot_rhos)),
        'rho_std': float(np.std(boot_rhos)),
        'rho_ci95': rho_ci,
        'chr_wait_viol_mean': float(np.mean(boot_chr_wait)),
        'chr_se_viol_mean': float(np.mean(boot_chr_se))
    },
    'P_static_6': {
        'min': float(P_static.min()),
        'max': float(P_static.max()),
        'mean': float(P_static.mean())
    }
}

with open(METRICS_OUT, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved {METRICS_OUT}")

print("\n" + "=" * 70)
print("SUMMARY: 6-param canonical with ASC")
print("=" * 70)
print(f"ASC (opt-out): {coefs[5]:.4f}")
print(f"Spearman rho: {rho:.4f} [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]")
print(f"CHR wait: {chr_wait_viol}/{chr_wait_total}")
print(f"CHR se: {chr_se_viol}/{chr_se_total}")
print(f"P_static_6 mean: {P_static.mean():.4f}")
print("=" * 70)
