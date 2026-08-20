"""
Recompute held-out DCE matching metrics + S15 task-level multinomial
with INTENDED DESIGN SUPPORT (wait ∈ {0,1,3,6}, exclude wait=2 outlier).

Canonical seed=2026, train_frac=0.80.
"""
import pandas as pd
import numpy as np
import random
from scipy.optimize import minimize
from scipy.special import expit
from pathlib import Path

SEED = 2026
TRAIN_FRAC = 0.80
REPO = Path('/Users/cary/bdt_repo')

# ============================================================
# 1. Load + split
# ============================================================
df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED); shuffled = rids[:]; rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train]); test_ids = set(shuffled[n_train:])
heldout = df[df['RespondentID'].isin(test_ids)].copy()

# ============================================================
# 2. Apply INTENDED DESIGN FILTER (exclude wait=2 outlier)
# ============================================================
print('='*72)
print(f'[1] Intended design support: wait ∈ {{0,1,3,6}}; wait=2 outlier excluded')
print('='*72)
intended_dce = df[df['WaitTime'].isin({0,1,3,6})].copy()
heldout_intended = heldout[heldout['WaitTime'].isin({0,1,3,6})].copy()
print(f'Full DCE rows after excl wait=2: {len(intended_dce)} (was {len(df)})')
print(f'Held-out rows after excl wait=2: {len(heldout_intended)} (was {len(heldout)})')

# ============================================================
# 3. Matched alt-rows: 12-cell literal intersection
#    wait {0,6} × eff {0.5,0.7} × se {1,2,3} = 12 cells
#    (note: grid wait {0,2,4,6} ∩ intended {0,1,3,6} = {0,6})
# ============================================================
nonopt = heldout_intended[
    (heldout_intended['CashIncentives']>0) &
    (heldout_intended['VaccineEfficacy']>0) &
    (heldout_intended['SideEffects']>0)
].copy()

matched_alt = nonopt[
    nonopt['WaitTime'].isin({0, 6}) &
    nonopt['VaccineEfficacy'].isin({0.5, 0.7}) &
    nonopt['SideEffects'].isin({1.0, 2.0, 3.0})
].copy()
print(f'\\nMatched non-opt-out alt-rows (intended ∩ grid): {len(matched_alt)}')
cells = matched_alt[['WaitTime','VaccineEfficacy','SideEffects']].drop_duplicates()
print(f'Distinct (wait,eff,se) cells: {len(cells)}')
for _, c in cells.sort_values(['WaitTime','VaccineEfficacy','SideEffects']).iterrows():
    print(f'  ({c.WaitTime}, {c.VaccineEfficacy}, {c.SideEffects})')

# Tasks
tasks_2alt = matched_alt.groupby(['RespondentID','Choiceset']).filter(lambda g: len(g)==2)
n_tasks_2alt = tasks_2alt.groupby(['RespondentID','Choiceset']).ngroups
print(f'\\nTasks (2-alt matched): {n_tasks_2alt}')

# ============================================================
# 4. Refit conditional logit on training data (intended design only)
# ============================================================
print('\\n' + '='*72)
print('[2] Refit 5-param conditional logit on training data (intended design only)')
print('='*72)
# Train rows: non-optout with intended design support
train_dce = intended_dce[
    (intended_dce['RespondentID'].isin(train_ids)) &
    (intended_dce['CashIncentives']>0) &
    (intended_dce['VaccineEfficacy']>0) &
    (intended_dce['SideEffects']>0)
].copy()
print(f'Training non-optout rows (intended): {len(train_dce)}')

# Build alt-level long format with y
# Each row has y=Choice (already alt-level in dce_encoded.csv)
X_train = np.column_stack([
    np.ones(len(train_dce)),
    train_dce['WaitTime'].values,
    train_dce['VaccineEfficacy'].values,
    train_dce['SideEffects'].values,
    train_dce['CashIncentives'].values,
    train_dce['VaccineOrigin'].values,
])
y_train = train_dce['Choice'].values

# Cluster by RespondentID + Choiceset (each task has 3 alts)
def fit_mle_clustered(X, y, clusters, max_iter=50):
    """Binomial logit with cluster-robust SE (sandwich)."""
    from numpy.linalg import solve as la_solve
    n = len(X)
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
            beta_new = la_solve(A, b)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new; break
        beta = beta_new
    # Cluster-robust SE
    unique_clusters = sorted(set(clusters))
    score_resid = np.zeros((len(unique_clusters), X.shape[1]))
    for i, c in enumerate(clusters):
        idx = unique_clusters.index(c)
        score_resid[idx] += X[i] * (y[i] - expit(X[i] @ beta))
    bread = X.T @ (X * (expit(X @ beta) * (1 - expit(X @ beta)))[:,None])
    meat = score_resid.T @ score_resid
    try:
        V = np.linalg.inv(bread) @ meat @ np.linalg.inv(bread)
    except:
        V = np.linalg.inv(bread)
    return beta, V

clusters_train = list(zip(train_dce['RespondentID'], train_dce['Choiceset']))
beta_train, V_train = fit_mle_clustered(X_train, y_train, clusters_train)
names = ['const','wait','eff','se','cash','origin']
print(f'\\nFitted 5-param logit (no ASC; using origin instead of ASC for non-optout-only):')
for n, b, v in zip(names, beta_train, np.diag(V_train)**0.5):
    print(f'  {n}: {b:.4f} ± {v:.4f}')

# ============================================================
# 5. Predict heldout matched subset + compute metrics
# ============================================================
print('\\n' + '='*72)
print('[3] Held-out matched subset metrics (intended design)')
print('='*72)
X_match = np.column_stack([
    np.ones(len(matched_alt)),
    matched_alt['WaitTime'].values,
    matched_alt['VaccineEfficacy'].values,
    matched_alt['SideEffects'].values,
    matched_alt['CashIncentives'].values,
    matched_alt['VaccineOrigin'].values,
])
y_match = matched_alt['Choice'].values
p_match = expit(X_match @ beta_train)

# Binary log loss + Brier
def log_loss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

def brier(y, p):
    return np.mean((y - p)**2)

print(f'Matched n: {len(matched_alt)}')
print(f'Pure-DCE log loss: {log_loss(y_match, p_match):.4f}')
print(f'Pure-DCE Brier:    {brier(y_match, p_match):.4f}')

# LLM unconstrained: read from Qwen unconstrained
pllm_df = pd.read_csv(REPO / 'llm_parsed_outputs_qwen72b_unconstrained.csv')
# Mean P per state
pllm_by_state = pllm_df.groupby('state')['probability_0_1'].mean().to_dict()
# Map matched alt-rows to grid state
grid_df = pd.read_csv(REPO / 'bdt_eval_grid_static.csv')
def match_state(row):
    """Find grid state by (wait, eff, se) tuple."""
    matches = grid_df[(grid_df['wait']==row['WaitTime']) &
                      (grid_df['eff']==row['VaccineEfficacy']) &
                      (grid_df['se']==row['SideEffects'])]
    if len(matches) == 0:
        return None
    return matches.iloc[0]['state']
matched_alt['state'] = matched_alt.apply(match_state, axis=1)
matched_alt['p_llm'] = matched_alt['state'].map(pllm_by_state)
unmatched = matched_alt['p_llm'].isna().sum()
print(f'LLM-unmatched alt-rows: {unmatched}')
matched_alt_llm = matched_alt.dropna(subset=['p_llm'])
y_llm = matched_alt_llm['Choice'].values
p_llm = matched_alt_llm['p_llm'].values
print(f'LLM log loss: {log_loss(y_llm, p_llm):.4f}')
print(f'LLM Brier:    {brier(y_llm, p_llm):.4f}')

# Static-BDT (λ=0.25)
LAMBDA = 0.25
p_bdt = LAMBDA * p_llm + (1 - LAMBDA) * expit(np.column_stack([
    np.ones(len(matched_alt_llm)),
    matched_alt_llm['WaitTime'].values,
    matched_alt_llm['VaccineEfficacy'].values,
    matched_alt_llm['SideEffects'].values,
    matched_alt_llm['CashIncentives'].values,
    matched_alt_llm['VaccineOrigin'].values,
]) @ beta_train)
y_bdt = matched_alt_llm['Choice'].values
print(f'\\nStatic-BDT (λ=0.25) log loss: {log_loss(y_bdt, p_bdt):.4f}')
print(f'Static-BDT (λ=0.25) Brier:    {brier(y_bdt, p_bdt):.4f}')

# ============================================================
# 6. Task-level multinomial (S15): 2-alt matched tasks
# ============================================================
print('\\n' + '='*72)
print('[4] S15 task-level multinomial (intended design, 2-alt matched tasks)')
print('='*72)
print(f'Tasks (2-alt matched): {n_tasks_2alt}')

# Task-level log loss for binary A vs B
def task_log_loss(p_a, y_a, eps=1e-7):
    """Binary task log loss: -y_a*log(p_a) - (1-y_a)*log(1-p_a)."""
    p_a = np.clip(p_a, eps, 1-eps)
    return -np.mean(y_a * np.log(p_a) + (1-y_a) * np.log(1-p_a))

# Aggregate alt-rows to task level: each task has 2 alts (A, B) both in matched subset
# Per task: p_a (predicted) = model prob; y_a = chosen by respondent (binary)
task_data = []
for (resp, cs), grp in tasks_2alt.groupby(['RespondentID','Choiceset']):
    if len(grp) != 2:
        continue
    # Sort by Alt (A=0, B=1)
    grp = grp.sort_values('Alt')
    y_a = grp[grp['Alt']=='A']['Choice'].iloc[0] if 'A' in grp['Alt'].values else grp.iloc[0]['Choice']
    # For 2-alt tasks, "chosen A" = Choice==1 for A alt
    y_a = grp.iloc[0]['Choice']  # A's choice
    y_b = grp.iloc[1]['Choice']  # B's choice
    # Pure-DCE: p_a from logit
    p_a_dce = expit(X_match[matched_alt.index.get_loc(grp.index[0])] @ beta_train)
    # LLM
    p_a_llm = grp.iloc[0]['p_llm']
    # Static-BDT
    p_a_bdt = LAMBDA * p_a_llm + (1-LAMBDA) * p_a_dce
    task_data.append({
        'task_id': f'{resp}_{cs}',
        'y_a': int(y_a),
        'p_dce': float(p_a_dce),
        'p_llm': float(p_a_llm) if not pd.isna(p_a_llm) else 0.5,
        'p_bdt': float(p_a_bdt),
    })
task_df = pd.DataFrame(task_data)
print(f'Total 2-alt tasks with LLM: {(~task_df["p_llm"].isna()).sum()}')

# Log loss
ll_dce = task_log_loss(task_df['p_dce'].values, task_df['y_a'].values)
ll_llm = task_log_loss(task_df['p_llm'].values, task_df['y_a'].values)
ll_bdt = task_log_loss(task_df['p_bdt'].values, task_df['y_a'].values)
print(f'\\nTask-level log loss (binary 2-alt):')
print(f'  Pure-DCE:       {ll_dce:.4f}')
print(f'  Static-BDT (λ=0.25): {ll_bdt:.4f}')
print(f'  Unconstrained LLM:  {ll_llm:.4f}')

# Save
import json
results = {
    'n_matched_alt': int(len(matched_alt)),
    'n_distinct_cells': int(len(cells)),
    'n_tasks_2alt': int(n_tasks_2alt),
    'alt_level_log_loss': {
        'Pure-DCE': float(log_loss(y_match, p_match)),
        'Static-BDT_L025': float(log_loss(y_bdt, p_bdt)),
        'LLM': float(log_loss(y_llm, p_llm)),
    },
    'alt_level_brier': {
        'Pure-DCE': float(brier(y_match, p_match)),
        'Static-BDT_L025': float(brier(y_bdt, p_bdt)),
        'LLM': float(brier(y_llm, p_llm)),
    },
    'task_level_log_loss': {
        'Pure-DCE': float(ll_dce),
        'Static-BDT_L025': float(ll_bdt),
        'LLM': float(ll_llm),
    },
}
with open('/tmp/intended_design_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved to /tmp/intended_design_metrics.json')