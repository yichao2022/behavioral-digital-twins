"""Find what gives 809 — try only 12-cell literal intersection."""
import pandas as pd
import numpy as np
import random

SEED = 2026
TRAIN_FRAC = 0.80

df = pd.read_csv('/Users/cary/bdt_repo/analysis_output/dce_encoded.csv')
rids = sorted(df['RespondentID'].unique())
rng = random.Random(SEED); shuffled = rids[:]; rng.shuffle(shuffled)
n_train = int(round(TRAIN_FRAC * len(shuffled)))
train_ids = set(shuffled[:n_train]); test_ids = set(shuffled[n_train:])
heldout = df[df['RespondentID'].isin(test_ids)].copy()

# 12-cell literal intersection (per heldout_matched_audit.md + paper)
# wait {0,2,6} ∩ eff {0.5,0.7} ∩ se {1,2,3} → 18 cells, 12 populated
int_wait = {0, 2, 6}
int_eff = {0.5, 0.7}
int_se = {1.0, 2.0, 3.0}

# Non-optout
nonopt = heldout[(heldout['CashIncentives']>0) & (heldout['VaccineEfficacy']>0) & (heldout['SideEffects']>0)]
# Match to literal intersection
m12 = nonopt[
    nonopt['WaitTime'].isin(int_wait) &
    nonopt['VaccineEfficacy'].isin(int_eff) &
    nonopt['SideEffects'].isin(int_se)
]
print(f'12-cell literal intersection (raw 5 wait levels):')
print(f'  Matched alt-rows: {len(m12)}')
tasks = m12.groupby(['RespondentID', 'Choiceset'])
print(f'  Tasks: {tasks.ngroups}')
print(f'  2-alt: {(tasks.size()==2).sum()}, 3-alt: {(tasks.size()==3).sum()}')

# Intended design (wait {0,1,3,6}, exclude wait=2)
int_design_wait = {0, 1, 3, 6}
m12_intended = nonopt[
    nonopt['WaitTime'].isin(int_design_wait) &
    nonopt['VaccineEfficacy'].isin(int_eff) &
    nonopt['SideEffects'].isin(int_se)
]
print(f'\\n12-cell literal intersection (intended design wait {{0,1,3,6}}):')
print(f'  Matched alt-rows: {len(m12_intended)}')
tasks_intended = m12_intended.groupby(['RespondentID', 'Choiceset'])
print(f'  Tasks: {tasks_intended.ngroups}')
print(f'  2-alt: {(tasks_intended.size()==2).sum()}, 3-alt: {(tasks_intended.size()==3).sum()}')

# 16 cells (grid ∩ DCE non-optout se only, no wait restriction beyond grid)
# Maybe the literal overlap is more restrictive (se {1,2,3} from non-optout DCE)
# but eff {0.5, 0.7} from DCE non-optout eff
print(f'\\n--- DCE non-optout literal overlap: wait {{0,2,6}} ∩ eff {{0.5,0.7}} ∩ se {{1,2,3}} ---')
print(f'  DCE WaitTime unique (non-optout): {sorted(nonopt["WaitTime"].unique())}')
print(f'  DCE VaccineEfficacy unique (non-optout): {sorted(nonopt["VaccineEfficacy"].unique())}')
print(f'  DCE SideEffects unique (non-optout): {sorted(nonopt["SideEffects"].unique())}')
# Wait: 0,1,2,3,6 → ∩ grid {0,2,4,6} = {0,2,6}
# Eff: 0.5,0.7,0.95 → ∩ grid {0.3,0.5,0.7,0.9} = {0.5,0.7}
# SE: 1,2,3 → ∩ grid {0,1,2,3} = {1,2,3}