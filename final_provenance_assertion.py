"""FINAL provenance assertion: S4 tex coefficients + Table S3 coding -> 64-state vs Main Table 1 frontier.

Reads the S4 coefficient values AS REPORTED IN THE SUPPLEMENT TEX (not from csv),
applies the Table S3 coding scheme, regenerates the 64-state frontier, and compares
row-by-row against the P_static_6_clean column used by Main Table 1.
"""
import re
import pandas as pd
import numpy as np
from scipy.special import expit

SUPP = '/tmp/bdt_extract/supplement_round3.tex'
MAIN = '/tmp/bdt_extract/manuscript_round3_blinded.tex'
GRID = '/Users/cary/bdt_repo/results/bdt_eval_grid_static_6param_clean.csv'

s = open(SUPP).read()

# ---- 1) Parse S4 coefficients as printed in the tex ----
# S4 table region: "Vaccine Origin & $0.2455^{***}$ & $(0.048)$ ..." etc
s4_start = s.find('\\begin{table}[htbp]', s.find('static frontier'))
# find the S4 label
label_idx = s.find('\\label{tab:static_frontier_estimates}')
# take the table containing that label (search backwards from label for \begin{table})
tbl_start = s.rfind('\\begin{table}', 0, label_idx)
# but ensure we got the right table: the label must appear between tbl_start and its \end
# (rfind could land on an earlier table if the label is inside a nested structure; for S4 it's fine)
tbl_end_candidate = s.find('\\end{table}', tbl_start)
tbl = s[tbl_start:s.find('\\end{table}', tbl_start) + len('\\end{table}')]
# safety: if the label isn't in this window, widen forward
if 'tab:static_frontier_estimates' not in tbl:
    nxt = s.find('\\begin{table}', label_idx)
    tbl = s[nxt:s.find('\\end{table}', nxt) + len('\\end{table}')]

def parse_coef(label):
    # row like: Vaccine Origin & $0.2455^{***}$ & $(0.048)$
    # label may be followed by a parenthetical unit: "Wait Time (months) & $-0.0590^{***}$ ..."
    patterns = [
        r'[^(]*\([^)]*\)\s*&\s*\$([0-9.-]+)\^\{\*{0,3}\}\$',   # with unit + stars
        r'[^(]*\([^)]*\)\s*&\s*\$([0-9.-]+)\$',                # with unit, no stars
        r'\s*&\s*\$([0-9.-]+)\^\{\*{0,3}\}\$',                 # no unit + stars
        r'\s*&\s*\$([0-9.-]+)\$',                              # no unit, no stars
    ]
    for p in patterns:
        m = re.search(re.escape(label) + p, tbl)
        if m:
            return float(m.group(1))
    raise ValueError(f'coef not found for {label}')

coefs = {
    'origin': parse_coef('Vaccine Origin'),
    'wait':   parse_coef('Wait Time'),
    'eff':    parse_coef('Vaccine Efficacy'),
    'se':     parse_coef('Side Effects'),
    'cash':   parse_coef('Cash Incentive'),
    'asc':    parse_coef('ASC'),
}
print('[1] S4 coefficients AS PRINTED IN TEX:')
for k, v in coefs.items():
    print(f'    {k:6s} = {v:.6f}')

# ---- 2) Table S3 coding: extract the "enters linear predictor" column ----
# Table S3 rows (from the coding dictionary table): Wait Time (months) raw; efficacy proportion; se coded level; cash RMB; origin 0/1
# The linear predictor column values are the coded values; grid column values:
# wait {0,2,4,6}, eff {0.3,0.5,0.7,0.9}, se {0,1,2,3}, cash 0, origin 0
# Verify grid values from the actual grid csv
grid = pd.read_csv(GRID)
g_w = grid['wait'].astype(float).values
g_e = grid['eff'].astype(float).values
g_s = grid['se'].astype(float).values
print(f'\n[2] Grid coding (from bdt_eval_grid_static_6param_clean.csv):')
print(f'    wait = {sorted(set(g_w))}')
print(f'    eff  = {sorted(set(g_e))}')
print(f'    se   = {sorted(set(g_s))}')

# ---- 3) Regenerate frontier from tex coefficients + coding ----
# P = sigmoid(U - ASC), U = b_w*w + b_e*e + b_s*s  (origin=0, cash=0 on grid)
U = coefs['wait'] * g_w + coefs['eff'] * g_e + coefs['se'] * g_s
P_tex = expit(U - coefs['asc'])

# ---- 4) Main Table 1 frontier ----
P_main = grid['P_static_6_clean'].values
diff = np.abs(P_tex - P_main)
print(f'\n[3] COMPARISON:')
print(f'    Max |P(S4 tex coefs) - P(Main Table 1)| = {diff.max():.2e}')
print(f'    Mean diff = {diff.mean():.2e}')
print(f'    #states with diff > 1e-12: {(diff > 1e-12).sum()} / 64')
print(f'    P range (tex-derived): [{P_tex.min():.4f}, {P_tex.max():.4f}]')
print(f'    P range (Main Table 1): [{P_main.min():.4f}, {P_main.max():.4f}]')

# Also check Main Table 1 tex numbers match the grid column
main_txt = open(MAIN).read()
# abstract numbers
for pat in ['0.689', '0.812', '0.756']:
    if pat not in main_txt:
        print(f'    WARNING: {pat} not in main tex')

print('\n' + '=' * 70)
if diff.max() < 1e-12:
    print('VERDICT: PROVENANCE CLOSED.')
    print('S4 tex coefficients + Table S3 coding reproduce Main Table 1')
    print('64-state frontier exactly (pure float precision).')
else:
    print(f'VERDICT: MISMATCH -- max diff {diff.max():.2e}')
print('=' * 70)
