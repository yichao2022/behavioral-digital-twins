"""Find which 5 rows differ between old 809 and my 814."""
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
nonopt = heldout[(heldout['CashIncentives']>0) & (heldout['VaccineEfficacy']>0) & (heldout['SideEffects']>0)]
m12 = nonopt[
    nonopt['WaitTime'].isin({0, 2, 6}) &
    nonopt['VaccineEfficacy'].isin({0.5, 0.7}) &
    nonopt['SideEffects'].isin({1.0, 2.0, 3.0})
]
print(f'My matched: {len(m12)}')

# Distinct (wait,eff,se) cells
cells = m12[['WaitTime','VaccineEfficacy','SideEffects']].drop_duplicates()
print(f'Distinct cells: {len(cells)}')
print(cells.sort_values(['WaitTime','VaccineEfficacy','SideEffects']).to_string())

# Maybe old code filtered out cash = 800 only? Or one specific cell?
# Try: also exclude wait=2 admin row (1 alt)
m_no_w2 = m12[m12['WaitTime'] != 2]
print(f'\\nWithout wait=2 row: {len(m_no_w2)}')

# Maybe also filter cash=800? Or cash != some level?
print(f'\\nCash distribution in 814:')
print(m12['CashIncentives'].value_counts().sort_index().to_string())

# Try excluding cash=800 (only? or all non-balanced?)
m_no800 = m12[m12['CashIncentives'] != 800]
print(f'\\nExcl cash=800: {len(m_no800)}')

m_only50 = m12[m12['CashIncentives'] == 50]
m_only200 = m12[m12['CashIncentives'] == 200]
m_only800 = m12[m12['CashIncentives'] == 800]
print(f'\\ncash=50: {len(m_only50)}, cash=200: {len(m_only200)}, cash=800: {len(m_only800)}')

# Maybe cash=0 alt-rows are filtered differently? Try nonopt-with-cash>=50
m_cashge50 = m12[m12['CashIncentives'] >= 50]
print(f'cash>=50: {len(m_cashge50)}')

# Maybe filter out one specific (eff, se) cell?
for cell in [(0.5, 1), (0.5, 2), (0.5, 3), (0.7, 1), (0.7, 2), (0.7, 3)]:
    sub = m12[(m12['VaccineEfficacy']==cell[0]) & (m12['SideEffects']==cell[1])]
    print(f'cell (eff={cell[0]}, se={cell[1]}): {len(sub)}')