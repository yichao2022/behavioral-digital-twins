"""Try different held-out split seeds to see if 809 was from a different seed."""
import pandas as pd
import numpy as np
import random

df = pd.read_csv('/Users/cary/bdt_repo/analysis_output/dce_encoded.csv')
rids = sorted(df['RespondentID'].unique())
nonopt_template = lambda heldout: heldout[(heldout['CashIncentives']>0) & (heldout['VaccineEfficacy']>0) & (heldout['SideEffects']>0)]
m12_filter = lambda m: m[m['WaitTime'].isin({0,2,6}) & m['VaccineEfficacy'].isin({0.5,0.7}) & m['SideEffects'].isin({1.0,2.0,3.0})]

# Try seed 2026 with different frac
for seed in [2026, 1, 2, 3, 100, 200, 0, 42, 99]:
    for frac in [0.80]:
        rng = random.Random(seed)
        shuffled = rids[:]; rng.shuffle(shuffled)
        n_train = int(round(frac * len(shuffled)))
        test_ids = set(shuffled[n_train:])
        heldout = df[df['RespondentID'].isin(test_ids)].copy()
        m = m12_filter(nonopt_template(heldout))
        if len(m) in (808, 809, 810, 811, 812, 813, 814):
            print(f'seed={seed} frac={frac}: matched={len(m)}')

# Also try without seed (random)
for seed in range(50):
    rng = random.Random(seed)
    shuffled = rids[:]; rng.shuffle(shuffled)
    n_train = int(round(0.80 * len(shuffled)))
    test_ids = set(shuffled[n_train:])
    heldout = df[df['RespondentID'].isin(test_ids)].copy()
    m = m12_filter(nonopt_template(heldout))
    if len(m) == 809:
        print(f'>>> seed={seed}: matched=809')
        break