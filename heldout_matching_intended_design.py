"""
Held-out DCE validation matching — reproduce canonical 809/73 numbers with
INTENDED DESIGN SUPPORT (wait ∈ {0,1,3,6}; wait=2 excluded as data anomaly).

The 1-row wait=2 outlier in raw dce_encoded.csv is from RespondentID=1, Choiceset=1,
Alt=A, eff=0.5, se=2, cash=800. We exclude this row from matching because wait=2
is not an administered design level (the canonical design is wait ∈ {0,1,3,6}).

This script also reports what changes if we keep the row (for sensitivity).
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path

SEED = 2026
TRAIN_FRAC = 0.80
RANDOM_STATE = 2026

REPO = Path('/Users/cary/bdt_repo')

# Read encoded DCE
df = pd.read_csv(REPO / 'analysis_output/dce_encoded.csv')
print(f'Total DCE rows: {len(df)}')

# Respondent split (seed=2026, 80/20 train/test)
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED)
shuffled = rids[:]
rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
n_train = max(1, min(n_train, len(shuffled) - 1))
train_ids = set(shuffled[:n_train])
test_ids = set(shuffled[n_train:])
print(f'Train: {len(train_ids)} respondents; Test: {len(test_ids)} respondents')

heldout = df[df['RespondentID'].isin(test_ids)].copy()
print(f'Held-out rows: {len(heldout)}')

# Intended design wait support
DESIGN_WAITS = {0, 1, 3, 6}
# Grid levels for matching
GRID_WAITS = {0, 2, 4, 6}
GRID_EFFS = {0.3, 0.5, 0.7, 0.9}
GRID_SE = {0.0, 1.0, 2.0, 3.0}

# Non-opt-out criteria
NONOPT = lambda r: (r['CashIncentives'] > 0) and (r['VaccineEfficacy'] > 0) and (r['SideEffects'] > 0)

# A. With intended design support (wait ∈ {0,1,3,6}, wait=2 row excluded)
intended = heldout[heldout['WaitTime'].isin(DESIGN_WAITS)].copy()
print(f'\\n[A] Intended design support (wait ∈ {sorted(DESIGN_WAITS)}, wait=2 excluded):')
print(f'  Held-out rows after excluding wait=2: {len(intended)}')
print(f'  wait=2 rows in held-out: {(heldout["WaitTime"]==2).sum()}')

# Match against grid
matched_intended = intended[
    intended['WaitTime'].isin(GRID_WAITS) &
    intended['VaccineEfficacy'].isin(GRID_EFFS) &
    intended['SideEffects'].isin(GRID_SE)
]
matched_nonopt_intended = matched_intended[matched_intended.apply(NONOPT, axis=1)]
print(f'  Matched alt-rows (intended design × grid): {len(matched_intended)}')
print(f'  Matched non-opt-out alt-rows: {len(matched_nonopt_intended)}')
print(f'  Distinct (wait, eff, se) cells: {len(matched_nonopt_intended[["WaitTime","VaccineEfficacy","SideEffects"]].drop_duplicates())}')

# Tasks (alt-level rows grouped by (RespondentID, Choiceset))
tasks = matched_nonopt_intended.groupby(['RespondentID', 'Choiceset']).ngroups
print(f'  Tasks (RespondentID × Choiceset): {tasks}')

# 2-alt tasks (matched non-opt-out tasks that have exactly 2 non-opt-out alts)
task_alt_counts = matched_nonopt_intended.groupby(['RespondentID', 'Choiceset']).size()
tasks_2alt = (task_alt_counts == 2).sum()
tasks_3alt = (task_alt_counts == 3).sum()
print(f'  2-alt tasks: {tasks_2alt}; 3-alt tasks: {tasks_3alt}')

# B. Raw support (wait ∈ {0,1,2,3,6}) — for sensitivity
print(f'\\n[B] Raw observed support (wait ∈ {{0,1,2,3,6}} — wait=2 1-row outlier included):')
matched_raw = heldout[
    heldout['WaitTime'].isin(GRID_WAITS) &
    heldout['VaccineEfficacy'].isin(GRID_EFFS) &
    heldout['SideEffects'].isin(GRID_SE)
]
matched_nonopt_raw = matched_raw[matched_raw.apply(NONOPT, axis=1)]
print(f'  Matched non-opt-out alt-rows: {len(matched_nonopt_raw)}')
tasks_raw = matched_nonopt_raw.groupby(['RespondentID', 'Choiceset']).ngroups
print(f'  Tasks: {tasks_raw}')
task_alt_counts_raw = matched_nonopt_raw.groupby(['RespondentID', 'Choiceset']).size()
print(f'  2-alt tasks: {(task_alt_counts_raw == 2).sum()}; 3-alt tasks: {(task_alt_counts_raw == 3).sum()}')

# C. Population balance SMD
print(f'\\n[C] Covariate balance (intended-design matched vs full held-out pool):')
# SMD helper
def smd(matched, full, col):
    m = matched[col].mean()
    f = full[col].mean()
    s = np.sqrt((matched[col].var() + full[col].var()) / 2)
    if s == 0:
        return 0
    return (m - f) / s

print(f'  WaitTime SMD: {smd(matched_nonopt_intended, heldout, "WaitTime"):.4f}')
print(f'  VaccineEfficacy SMD: {smd(matched_nonopt_intended, heldout, "VaccineEfficacy"):.4f}')
print(f'  SideEffects SMD: {smd(matched_nonopt_intended, heldout, "SideEffects"):.4f}')
print(f'  CashIncentives SMD: {smd(matched_nonopt_intended, heldout, "CashIncentives"):.4f}')

# Save
matched_nonopt_intended.to_csv('/tmp/matched_intended_design.csv', index=False)
print(f'\\nSaved matched non-opt-out alt-rows to /tmp/matched_intended_design.csv')