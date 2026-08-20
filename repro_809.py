"""Reproduce old 809 number exactly per heldout_dce_validation.py logic."""
import pandas as pd
import numpy as np
import random
from pathlib import Path

SEED = 2026
TRAIN_FRAC = 0.80

REPO = Path('/Users/cary/bdt_repo')

# Read encoded DCE
df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
print(f'Total DCE rows: {len(df)}')

# Respondent split
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED)
shuffled = rids[:]
rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
n_train = max(1, min(n_train, len(shuffled) - 1))
train_ids = set(shuffled[:n_train])
test_ids = set(shuffled[n_train:])
heldout = df[df['RespondentID'].isin(test_ids)].copy()
print(f'Held-out rows: {len(heldout)}')

# Read grid
grid = pd.read_csv(REPO / 'bdt_eval_grid_static.csv')
grid_keys = set((float(r['wait']), float(r['eff']), float(r['se'])) for _, r in grid.iterrows())
print(f'Grid states: {len(grid_keys)}, first 3: {list(grid_keys)[:3]}')

# Match all held-out rows
def match(row):
    return (float(row['WaitTime']), float(row['VaccineEfficacy']), float(row['SideEffects'])) in grid_keys

matched = heldout[heldout.apply(match, axis=1)].copy()
print(f'Total matched alt-rows (any cell): {len(matched)}')

# Restrict to non-optout
nonopt = matched[(matched['CashIncentives'] > 0) & (matched['VaccineEfficacy'] > 0) & (matched['SideEffects'] > 0)]
print(f'Matched non-opt-out alt-rows: {len(nonopt)}')

# Restricted to non-opt-out, only eff in {0.5, 0.7, 0.95} or grid eff
print(f'\\nBreakdown by WaitTime (matched non-optout):')
print(nonopt['WaitTime'].value_counts().sort_index().to_string())

# Count 2-alt tasks (which would give 73)
task_counts = nonopt.groupby(['RespondentID', 'Choiceset']).size()
print(f'\\nTasks: total={nonopt.groupby(["RespondentID","Choiceset"]).ngroups}')
print(f'  1-alt: {(task_counts==1).sum()}')
print(f'  2-alt: {(task_counts==2).sum()}')
print(f'  3-alt: {(task_counts==3).sum()}')

# Now restrict to intended design wait (excl wait=2)
intended = nonopt[nonopt['WaitTime'].isin([0,1,3,6])].copy()
print(f'\\nIntended design (wait {{0,1,3,6}}):')
print(f'  Matched non-opt-out alt-rows: {len(intended)}')
task_counts_intended = intended.groupby(['RespondentID','Choiceset']).size()
print(f'  Tasks: total={intended.groupby(["RespondentID","Choiceset"]).ngroups}')
print(f'  2-alt: {(task_counts_intended==2).sum()}, 3-alt: {(task_counts_intended==3).sum()}')