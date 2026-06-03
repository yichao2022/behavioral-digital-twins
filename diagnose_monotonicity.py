#!/usr/bin/env python3
"""Diagnostic 1+2: monotonicity check + per-row smoke v2 printout"""

import csv
import os

CSV_PATH_UPDATED = "/Users/cary/.openclaw/workspace/bdt_eval_grid_updated.csv"
CSV_PATH_V2 = "/Users/cary/.openclaw/workspace/bdt_eval_grid_v2.csv"
PARSED_PATH = "/Users/cary/.openclaw/workspace/llm_parsed_outputs_smoke_v2.csv"

# =========================================================
# DIAGNOSTIC 1: P_dynamic_1 monotonicity
# =========================================================
print("=" * 70)
print("DIAGNOSTIC 1: P_dynamic_1 Monotonicity Check")
print("=" * 70)

with open(CSV_PATH_UPDATED, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"\nTotal rows: {len(rows)}")
print(f"Columns: {list(reader.fieldnames)}")

# Convert to typed list
typed = []
for r in rows:
    typed.append(dict(r, wait=float(r["wait"]), eff=float(r["eff"]), se=float(r["se"]),
                       V0=float(r["V0"]), V1=float(r["V1"]),
                       P_dynamic_1=float(r["P_dynamic_1"])))

# Check 1a: fixed eff, se → wait ↑ → P_dynamic_1 ↓
# Group by (eff, se)
from collections import defaultdict

groups_eff_se = defaultdict(list)
for r in typed:
    groups_eff_se[(r["eff"], r["se"])].append(r)

violations_wait = 0
total_pairs_wait = 0
for key, grp in groups_eff_se.items():
    grp_sorted = sorted(grp, key=lambda x: x["wait"])
    for i in range(len(grp_sorted) - 1):
        for j in range(i + 1, len(grp_sorted)):
            if grp_sorted[j]["wait"] > grp_sorted[i]["wait"]:
                total_pairs_wait += 1
                if grp_sorted[j]["P_dynamic_1"] > grp_sorted[i]["P_dynamic_1"]:
                    violations_wait += 1

print(f"\n  [Monotonicity 1a] Fixed (eff, se): wait ↑ → P_dynamic_1 ↓")
print(f"    Violations:     {violations_wait}")
print(f"    Total pairs:    {total_pairs_wait}")
print(f"    Violation rate: {violations_wait/max(total_pairs_wait,1)*100:.1f}%")

# Check 1b: fixed wait, se → eff ↑ → P_dynamic_1 ↑
groups_wait_se = defaultdict(list)
for r in typed:
    groups_wait_se[(r["wait"], r["se"])].append(r)

violations_eff = 0
total_pairs_eff = 0
for key, grp in groups_wait_se.items():
    grp_sorted = sorted(grp, key=lambda x: x["eff"])
    for i in range(len(grp_sorted) - 1):
        for j in range(i + 1, len(grp_sorted)):
            if grp_sorted[j]["eff"] > grp_sorted[i]["eff"]:
                total_pairs_eff += 1
                if grp_sorted[j]["P_dynamic_1"] < grp_sorted[i]["P_dynamic_1"]:
                    violations_eff += 1

print(f"\n  [Monotonicity 1b] Fixed (wait, se): eff ↑ → P_dynamic_1 ↑")
print(f"    Violations:     {violations_eff}")
print(f"    Total pairs:    {total_pairs_eff}")
print(f"    Violation rate: {violations_eff/max(total_pairs_eff,1)*100:.1f}%")

# Check 1c: fixed wait, eff → se ↑ → P_dynamic_1 ↓
groups_wait_eff = defaultdict(list)
for r in typed:
    groups_wait_eff[(r["wait"], r["eff"])].append(r)

violations_se = 0
total_pairs_se = 0
for key, grp in groups_wait_eff.items():
    grp_sorted = sorted(grp, key=lambda x: x["se"])
    for i in range(len(grp_sorted) - 1):
        for j in range(i + 1, len(grp_sorted)):
            if grp_sorted[j]["se"] > grp_sorted[i]["se"]:
                total_pairs_se += 1
                if grp_sorted[j]["P_dynamic_1"] > grp_sorted[i]["P_dynamic_1"]:
                    violations_se += 1

print(f"\n  [Monotonicity 1c] Fixed (wait, eff): se ↑ → P_dynamic_1 ↓")
print(f"    Violations:     {violations_se}")
print(f"    Total pairs:    {total_pairs_se}")
print(f"    Violation rate: {violations_se/max(total_pairs_se,1)*100:.1f}%")

# Summary table
print(f"\n{'─'*60}")
print(f"{'Monotonicity Check Summary':^60}")
print(f"{'─'*60}")
print(f"{'Dimension':<20} {'Violations':<12} {'Total Pairs':<12} {'Violation Rate':<14}")
print(f"{'─'*60}")
print(f"{'wait ↑ ⇒ P ↓':<20} {violations_wait:<12} {total_pairs_wait:<12} {violations_wait/max(total_pairs_wait,1)*100:<13.1f}%")
print(f"{'eff ↑ ⇒ P ↑':<20} {violations_eff:<12} {total_pairs_eff:<12} {violations_eff/max(total_pairs_eff,1)*100:<13.1f}%")
print(f"{'se ↑ ⇒ P ↓':<20} {violations_se:<12} {total_pairs_se:<12} {violations_se/max(total_pairs_se,1)*100:<13.1f}%")
print(f"{'─'*60}")


# =========================================================
# DIAGNOSTIC 2: Per-row smoke v2 results
# =========================================================
print(f"\n\n{'=' * 70}")
print("DIAGNOSTIC 2: Smoke Test v2 — Per-Row Results")
print("=" * 70)

with open(PARSED_PATH, newline="") as f:
    reader = csv.DictReader(f)
    parsed_rows = list(reader)

print(f"\n{'State':<6} {'Wait':<6} {'Eff':<6} {'Se':<6} {'V0':<10} {'V1':<10} {'P_dyn_1':<10} {'Condition':<30} {'Rep':<5} {'Predicted':<10}")
print(f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*30} {'─'*5} {'─'*10}")

# Focus on dynamic_bdt_rule rows for diagnosis
print(f"\n--- dynamic_bdt_rule rows (the 9 that gave Spearman ρ = -0.957) ---")
for r in parsed_rows:
    if r["condition"] == "prompt_dynamic_bdt_rule":
        pred = float(r["probability_0_1"]) if r["probability_0_1"] else None
        p_dyn = float(r["P_dynamic_1"])
        print(f"{r['state'].zfill(4):<6} {r['wait']:<6} {r['eff']:<6} {r['se']:<6} "
              f"{float(r['V0']):<10.4f} {float(r['V1']):<10.4f} {p_dyn:<10.4f} "
              f"{r['condition']:<30} {r['repeat']:<5} {pred:<10.4f}" if pred else f"{'N/A':<10}")

print(f"\n--- all 27 rows ---")
for r in parsed_rows:
    pred = float(r["probability_0_1"]) if r["probability_0_1"] else None
    p_dyn = float(r["P_dynamic_1"])
    cond_short = r["condition"].replace("prompt_", "")
    if pred is not None:
        print(f"{r['state'].zfill(4):<6} {r['wait']:<6} {r['eff']:<6} {r['se']:<6} "
              f"{float(r['V0']):<10.4f} {float(r['V1']):<10.4f} {p_dyn:<10.4f} "
              f"{cond_short:<30} {r['repeat']:<5} {pred:<10.4f}")
    else:
        print(f"{r['state'].zfill(4):<6} {r['wait']:<6} {r['eff']:<6} {r['se']:<6} "
              f"{float(r['V0']):<10.4f} {float(r['V1']):<10.4f} {p_dyn:<10.4f} "
              f"{cond_short:<30} {r['repeat']:<5} {'N/A':<10}")

print(f"\n\n{'=' * 70}")
print("DONE")
print("=" * 70)
