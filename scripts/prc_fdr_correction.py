#!/usr/bin/env python3
"""
PRC multiplicity correction (Benjamini-Hochberg FDR) for Table 4.
Computes BH-adjusted significance for all Model-Method PRC rows.

Usage: python scripts/prc_fdr_correction.py
Output: results/prc_fdr_correction.csv, results/prc_fdr_note.tex
"""

import csv
from scipy import stats

# Table 4 PRC data: (Model, Method, correct, total)
prc_rows = [
    ("Qwen2.5-72B", "Unconstrained LLM",  6, 15),
    ("Qwen2.5-72B", "Static-BDT Anchor",  13, 15),
    ("DeepSeek V4 Pro", "Unconstrained LLM", 5, 15),
    ("DeepSeek V4 Pro", "Static-BDT Anchor", 9, 15),
    ("MiroThinker-1.7-mini", "Unconstrained LLM", 8, 15),
    ("MiroThinker-1.7-mini", "Static-BDT Anchor", 12, 15),
]

# Compute raw one-sided binomial p-values (H0: PRC = 0.5)
rows = []
for model, method, k, n in prc_rows:
    p = float(stats.binomtest(k, n, p=0.5, alternative='greater').pvalue)
    rows.append((model, method, k, n, p))

# --- Print BH FDR results ---
for label, m, subset_filter in [
    ("BDT methods only (m=3)", 3, lambda r: "Static-BDT" in r[1]),
    ("All rows (m=6)", 6, None),
]:
    subset = [r for r in rows if subset_filter(r)] if subset_filter else rows
    subset.sort(key=lambda r: r[4])

    print(f"\n--- {label} ---")
    print(f"{'Model':<20} {'Method':<25} {'k/n':<8} {'p-value':<10} {'BH rank':<8} {'threshold':<10} {'Survives?':<10}")
    print("-" * 95)
    for i, (model, method, k, n, p) in enumerate(subset):
        rank = i + 1
        threshold = (rank / m) * 0.05
        survives = "✅" if p <= threshold else " "
        print(f"{model:<20} {method:<25} {k}/{n:<5} {p:<10.5f} {rank}/{m:<6} {threshold:<10.5f} {survives:<10}")

# --- Save CSV ---
# BH classification: which survive in each family
bdts = [r for r in rows if "Static-BDT" in r[1]]
bdt_sorted = sorted(bdts, key=lambda r: r[4])
all_sorted = sorted(rows, key=lambda r: r[4])

def bh_survives(p, rank, m):
    return p <= (rank / m) * 0.05

output = [["family", "model", "method", "correct", "total", "raw_p", "bh_sig_m3", "bh_sig_m6"]]
for model, method, k, n, p in rows:
    sig_m3 = "no"
    for rank, (_, _, _, _, bp) in enumerate(bdt_sorted):
        if abs(bp - p) < 1e-10 and bh_survives(p, rank + 1, 3):
            sig_m3 = "yes"

    sig_m6 = "no"
    for rank, (_, _, _, _, ap) in enumerate(all_sorted):
        if abs(ap - p) < 1e-10 and bh_survives(p, rank + 1, 6):
            sig_m6 = "yes"

    output.append(["PRC (Table 4)", model, method, str(k), str(n), f"{p:.5f}", sig_m3, sig_m6])

with open("results/prc_fdr_correction.csv", "w", newline="") as f:
    csv.writer(f).writerows(output)
print("\n✅ Saved results/prc_fdr_correction.csv")

# --- Generate LaTeX note ---
latex = (
    r"% PRC multiplicity correction note (for Table 4)"
    "\n"
    r"% One-sided binomial test against H0: PRC = 0.5"
    "\n"
    r"% BH FDR correction at q = 0.05"
    "\n\n"
    r"\emph{Note.} Raw one-sided binomial $p$-values (against $H_0$: PRC $= 0.5$):"
    "\n"
)
for model, method, k, n, p in rows:
    if "Static-BDT" in method:
        latex += f"\n{model} ({method}): $p = {p:.4f}$;\n"

latex += (
    "\n"
    "Under Benjamini--Hochberg FDR correction at $q = 0.05$ across the three BDT method comparisons, "
    "Qwen2.5-72B ($p = 0.0037$, critical threshold at rank 1: 0.017) and MiroThinker-1.7-mini "
    "($p = 0.0176$, critical threshold at rank 2: 0.033) remain significant. "
    "DeepSeek V4 Pro ($p = 0.3036$) does not significantly exceed chance-level PRC."
    "\n"
)

with open("results/prc_fdr_note.tex", "w") as f:
    f.write(latex)
print("✅ Saved results/prc_fdr_note.tex")
