#!/usr/bin/env python3
"""
BDT Evaluation Grid — Benchmark Diagnostic (P_dynamic_1 Monotonicity)
=====================================================================
Reads: bdt_eval_grid_updated.csv
Outputs: textual diagnostic report (no file writes beyond stdout)
"""

import csv
import sys
from collections import defaultdict

# ── 1. Load ──────────────────────────────────────────────────────────
path = "/Users/cary/.openclaw/workspace/bdt_eval_grid_updated.csv"

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"{'='*70}")
print(f"  BDT Evaluation Grid — Benchmark Diagnostic")
print(f"  File: bdt_eval_grid_updated.csv")
print(f"{'='*70}\n")

# ── 2. Summary stats ────────────────────────────────────────────────
wait_vals = sorted(set(float(r["wait"]) for r in rows))
eff_vals  = sorted(set(float(r["eff"])  for r in rows))
se_vals   = sorted(set(float(r["se"])   for r in rows))

pdyn = [float(r["P_dynamic_1"]) for r in rows]

print(f"  Total rows:           {len(rows)}")
print(f"  Unique wait values:   {len(wait_vals)}  {wait_vals}")
print(f"  Unique eff values:    {len(eff_vals)}   {eff_vals}")
print(f"  Unique se values:     {len(se_vals)}   {se_vals}")
print(f"  P_dynamic_1 min:      {min(pdyn):.6f}")
print(f"  P_dynamic_1 max:      {max(pdyn):.6f}")
print()

# ── 3. Key grid layout (for diagnostics) ────────────────────────────
# Build indices for easy comparison
by_key = {}
for r in rows:
    key = (float(r["wait"]), float(r["eff"]), float(r["se"]))
    by_key[key] = float(r["P_dynamic_1"])

# A quick layout print
print("── Grid layout (wait × eff, fixed se=0.0) ──")
print(f"{'wait\\eff':>8}", end="")
for e in eff_vals:
    print(f"{e:>8.1f}", end="")
print()
for w in wait_vals:
    print(f"{w:>8.1f}", end="")
    for e in eff_vals:
        v = by_key.get((w, e, 0.0), None)
        if v is not None:
            print(f"{v:>8.4f}", end="")
        else:
            print(f"{'':>8}", end="")
    print()
print()

# ── 4. Monotonicity checks ──────────────────────────────────────────

def check_monotonicity(label, pairs, direction):
    """
    direction: +1 means x2 > x1 => y2 >= y1 expected,
               -1 means x2 > x1 => y2 <= y1 expected.
    pairs: list of ((x1, y1), (x2, y2))
    """
    total = len(pairs)
    violations = 0
    violation_details = []
    for (x1, y1), (x2, y2) in pairs:
        if direction == 1:   # x increases => y should increase or stay
            if y2 < y1 - 1e-9:
                violations += 1
                violation_details.append((x1, x2, y1, y2))
        else:                # x increases => y should decrease or stay
            if y2 > y1 + 1e-9:
                violations += 1
                violation_details.append((x1, x2, y1, y2))

    rate = violations / total if total else 0
    print(f"  {label}")
    print(f"    Total comparable pairs:  {total}")
    print(f"    Violation count:         {violations}")
    print(f"    Violation rate:          {rate:.4f} ({rate*100:.2f}%)")
    if violations > 0 and violations <= 10:
        for x1, x2, y1, y2 in violation_details:
            print(f"      [violation] ({x1} -> {x2}):  {y1:.6f} -> {y2:.6f}")
    elif violations > 10:
        print(f"      (showing first 10 of {violations} violations)")
        for x1, x2, y1, y2 in violation_details[:10]:
            print(f"      [violation] ({x1} -> {x2}):  {y1:.6f} -> {y2:.6f}")
    print()

# ---- 4a. Fixed eff, se: wait ↑ => P_dynamic_1 ↓ ────────────────────
pairs_wait = []
for (w1, e, s), v1 in by_key.items():
    for (w2, e2, s2), v2 in by_key.items():
        if e == e2 and s == s2 and w2 > w1:
            pairs_wait.append(((w1, v1), (w2, v2)))

check_monotonicity(
    "[A] Fixed eff, se:  wait ↑  ➔  P_dynamic_1 ↓ (monotone decreasing)",
    pairs_wait, -1
)

# ---- 4b. Fixed wait, se: eff ↑ => P_dynamic_1 ↑ ────────────────────
pairs_eff = []
for (w, e1, s), v1 in by_key.items():
    for (w2, e2, s2), v2 in by_key.items():
        if w == w2 and s == s2 and e2 > e1:
            pairs_eff.append(((e1, v1), (e2, v2)))

check_monotonicity(
    "[B] Fixed wait, se:  eff ↑  ➔  P_dynamic_1 ↑ (monotone increasing)",
    pairs_eff, 1
)

# ---- 4c. Fixed wait, eff: se ↑ => P_dynamic_1 ↓ ────────────────────
pairs_se = []
for (w, e, s1), v1 in by_key.items():
    for (w2, e2, s2), v2 in by_key.items():
        if w == w2 and e == e2 and s2 > s1:
            pairs_se.append(((s1, v1), (s2, v2)))

check_monotonicity(
    "[C] Fixed wait, eff:  se ↑  ➔  P_dynamic_1 ↓ (monotone decreasing)",
    pairs_se, -1
)

# ── 5. Detailed violation table (if any) ───────────────────────────
# Re-check and print a structured comparison table for wait monotonicity
print("── Detailed wait-monotonicity grid (eff=0.9, se=0.0) ──")
print(f"{'wait':>8}  {'P_dynamic_1':>12}  {'Δwait':>8}  {'ΔP':>12}  {'OK?':>6}")
prev = None
for w in sorted(wait_vals):
    v = by_key.get((w, 0.9, 0.0))
    if v is None:
        continue
    if prev is not None:
        dw = w - prev[0]
        dv = v - prev[1]
        ok = "✓" if dv <= 1e-9 else "✗ VIOLATION"
        print(f"{w:>8.1f}  {v:>12.6f}  {dw:>8.1f}  {dv:>+12.6f}  {ok:>6}")
    else:
        print(f"{w:>8.1f}  {v:>12.6f}")
    prev = (w, v)
print()

# Also show a se-fixed slice for wait-monotonicity (se=3.0, eff=0.9)
print("── Detailed wait-monotonicity grid (eff=0.9, se=3.0) ──")
print(f"{'wait':>8}  {'P_dynamic_1':>12}  {'Δwait':>8}  {'ΔP':>12}  {'OK?':>6}")
prev = None
for w in sorted(wait_vals):
    v = by_key.get((w, 0.9, 3.0))
    if v is None:
        continue
    if prev is not None:
        dw = w - prev[0]
        dv = v - prev[1]
        ok = "✓" if dv <= 1e-9 else "✗ VIOLATION"
        print(f"{w:>8.1f}  {v:>12.6f}  {dw:>8.1f}  {dv:>+12.6f}  {ok:>6}")
    else:
        print(f"{w:>8.1f}  {v:>12.6f}")
    prev = (w, v)
print()

print("── Detailed eff-monotonicity grid (wait=0.0, se=0.0) ──")
print(f"{'eff':>8}  {'P_dynamic_1':>12}  {'Δeff':>8}  {'ΔP':>12}  {'OK?':>6}")
prev = None
for e in sorted(eff_vals):
    v = by_key.get((0.0, e, 0.0))
    if v is None:
        continue
    if prev is not None:
        de = e - prev[0]
        dv = v - prev[1]
        ok = "✓" if dv >= -1e-9 else "✗ VIOLATION"
        print(f"{e:>8.1f}  {v:>12.6f}  {de:>8.1f}  {dv:>+12.6f}  {ok:>6}")
    else:
        print(f"{e:>8.1f}  {v:>12.6f}")
    prev = (e, v)
print()

print("── Detailed se-monotonicity grid (wait=0.0, eff=0.9) ──")
print(f"{'se':>8}  {'P_dynamic_1':>12}  {'Δse':>8}  {'ΔP':>12}  {'OK?':>6}")
prev = None
for s in sorted(se_vals):
    v = by_key.get((0.0, 0.9, s))
    if v is None:
        continue
    if prev is not None:
        ds = s - prev[0]
        dv = v - prev[1]
        ok = "✓" if dv <= 1e-9 else "✗ VIOLATION"
        print(f"{s:>8.1f}  {v:>12.6f}  {ds:>8.1f}  {dv:>+12.6f}  {ok:>6}")
    else:
        print(f"{s:>8.1f}  {v:>12.6f}")
    prev = (s, v)
print()

print(f"{'='*70}")
print(f"  Diagnostic complete.")
print(f"{'='*70}")
