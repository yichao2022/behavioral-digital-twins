"""Bootstrap CIs for frontier-alignment diagnostics on CLEAN 6-param frontier.

Mirrors the S11 table: 2000 state-level bootstrap replicates (with replacement)
for MSE and Spearman rho; MVR-Wait as point estimate. Uses P_static_6_clean
as the empirical reference; LLM probabilities grouped by state.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import os

REPO = Path('/Users/cary/bdt_repo')
GRID = pd.read_csv(REPO / 'results/bdt_eval_grid_static_6param_clean.csv')
P_REF = GRID['P_static_6_clean'].values
N_BOOT = 2000
RNG = np.random.default_rng(2026)

files = {
    'Qwen2.5-72B-Instruct': REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv',
    'DeepSeek V4': REPO / 'llm_parsed_outputs_deepseek_unconstrained.csv',
    'MiroThinker-1.7-mini': REPO / 'llm_parsed_outputs_mirothinker_unconstrained.csv',
}

def mvr_wait(pi):
    w = GRID['wait'].astype(float).values; e = GRID['eff'].astype(float).values; s = GRID['se'].astype(float).values
    v = tot = 0
    for i in range(len(pi)):
        for j in range(len(pi)):
            if w[j] > w[i] and e[j] == e[i] and s[j] == s[i]:
                tot += 1
                if pi[j] > pi[i]: v += 1
    return v / tot if tot else 0

rows = []
for model, path in files.items():
    pllm = pd.read_csv(path)
    if 'parse_success' in pllm.columns:
        pllm = pllm[pllm['parse_success'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    p_llm = pllm.groupby('state')['probability_0_1'].mean().sort_index().values
    for tag, lam in [('Unconstrained LLM', 1.0), ('Static-BDT Anchor', 0.25)]:
        pi_full = lam * p_llm + (1 - lam) * P_REF
        mse_full = float(np.mean((pi_full - P_REF) ** 2))
        rho_full = float(spearmanr(pi_full, P_REF).correlation)
        mvr_full = mvr_wait(pi_full)
        # bootstrap over states
        mse_b, rho_b = [], []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, 64, 64)
            pi_b = lam * p_llm[idx] + (1 - lam) * P_REF[idx]
            mse_b.append(np.mean((pi_b - P_REF[idx]) ** 2))
            rho_b.append(spearmanr(pi_b, P_REF[idx]).correlation)
        rows.append({'model': model, 'method': tag, 'metric': 'MSE',
                     'est': mse_full, 'lo': np.percentile(mse_b, 2.5), 'hi': np.percentile(mse_b, 97.5)})
        rows.append({'model': model, 'method': tag, 'metric': 'MVR_Wait',
                     'est': mvr_full, 'lo': np.nan, 'hi': np.nan})
        rows.append({'model': model, 'method': tag, 'metric': 'rho',
                     'est': rho_full, 'lo': np.percentile(rho_b, 2.5), 'hi': np.percentile(rho_b, 97.5)})
        print(f'{model} | {tag}: MSE={mse_full:.4f} [{np.percentile(mse_b,2.5):.4f}, {np.percentile(mse_b,97.5):.4f}] | '
              f'MVR-W={mvr_full:.4f} | rho={rho_full:.4f} [{np.percentile(rho_b,2.5):.4f}, {np.percentile(rho_b,97.5):.4f}]')

out = pd.DataFrame(rows)
out.to_csv(REPO / 'results/bootstrap_alignment_clean.csv', index=False)
print('\nWrote bootstrap_alignment_clean.csv')
