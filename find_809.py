"""Find what gives 809 (or 73)."""
import pandas as pd
import numpy as np
import random
from pathlib import Path

SEED = 2026
TRAIN_FRAC = 0.80

df = pd.read_csv('/Users/cary/bdt_repo/analysis_output/dce_encoded.csv')
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED); shuffled = rids[:]; rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train]); test_ids = set(shuffled[n_train:])
heldout = df[df['RespondentID'].isin(test_ids)].copy()

# Read old predictions
preds = pd.read_csv('/Users/cary/bdt_repo/results/heldout_dce_predictions.csv')
print(f'Old preds rows: {len(preds)}')
print(f'Old preds cols: {preds.columns.tolist()}')
print(f'Old preds matched (matched_llm_state != ""): {(preds["matched_llm_state"].astype(str) != "").sum()}')

# Read grid
grid = pd.read_csv('/Users/cary/bdt_repo/bdt_eval_grid_static.csv')
grid_keys = set((float(r['wait']), float(r['eff']), float(r['se'])) for _, r in grid.iterrows())

# Match using OLD logic (no wait filter, just (wait, eff, se) tuple ∈ grid)
def match(row):
    return (float(row['WaitTime']), float(row['VaccineEfficacy']), float(row['SideEffects'])) in grid_keys
matched = heldout[heldout.apply(match, axis=1)].copy()
print(f'\\n[A] Match all by (wait,eff,se) ∈ grid, no filter: {len(matched)}')

# Old audit said N=809 — that must be non-optout + cash>0 filter + eff>0 + se>0
nonopt = matched[(matched['CashIncentives']>0) & (matched['VaccineEfficacy']>0) & (matched['SideEffects']>0)]
print(f'[B] + non-optout filter (cash>0, eff>0, se>0): {len(nonopt)}')

# Maybe further restrict eff to grid eff only
g_eff = {0.3, 0.5, 0.7, 0.9}
g_se = {0.0, 1.0, 2.0, 3.0}
g_wait = {0, 2, 4, 6}
nonopt_grid_eff = nonopt[nonopt['VaccineEfficacy'].isin(g_eff) & nonopt['SideEffects'].isin(g_se) & nonopt['WaitTime'].isin(g_wait)]
print(f'[C] + (wait, eff, se) all in grid values: {len(nonopt_grid_eff)}')

# Tasks
print(f'[A] 2-alt tasks: {(nonopt.groupby(["RespondentID","Choiceset"]).size()==2).sum()}')
print(f'[C] 2-alt tasks: {(nonopt_grid_eff.groupby(["RespondentID","Choiceset"]).size()==2).sum()}')

# alt-level per resp
print(f'\\n[A] Per Respondent alt-row count distribution:')
print(nonopt.groupby('RespondentID').size().describe())

# Check old preds filter on non-optout
preds_match = preds[preds['matched_llm_state'].astype(str) != ''].copy()
preds_match = preds_match[(preds_match['cash']>0) & (preds_match['eff']>0) & (preds_match['se']>0)]
print(f'\\nOld preds matched non-opt-out: {len(preds_match)}')

# Check if 809 = non-optout matched and (eff, se, wait) ALL ∈ grid
nonopt_strict = nonopt[nonopt['VaccineEfficacy'].isin(g_eff)]
print(f'\\n[D] non-opt + eff ∈ grid eff: {len(nonopt_strict)}')
nonopt_strict2 = nonopt_strict[nonopt_strict['SideEffects'].isin(g_se)]
print(f'[E] + se ∈ grid se: {len(nonopt_strict2)}')
nonopt_strict3 = nonopt_strict2[nonopt_strict2['WaitTime'].isin(g_wait)]
print(f'[F] + wait ∈ grid wait: {len(nonopt_strict3)}')
print(f'[F] 2-alt tasks: {(nonopt_strict3.groupby(["RespondentID","Choiceset"]).size()==2).sum()}')

# Check old preds vs new: which rows in [A] not in [F]?
old_keys = set(zip(preds_match['respondent_id'], preds_match['wait'], preds_match['eff'], preds_match['se']))
new_keys_F = set(zip(nonopt_strict3['RespondentID'], nonopt_strict3['WaitTime'], nonopt_strict3['VaccineEfficacy'], nonopt_strict3['SideEffects']))
print(f'\\n[A] but not in old preds: {len(set(zip(matched["RespondentID"], matched["WaitTime"], matched["VaccineEfficacy"], matched["SideEffects"])) - old_keys)}')
print(f'[F] but not in old preds: {len(new_keys_F - old_keys)}')