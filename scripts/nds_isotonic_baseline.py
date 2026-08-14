#!/usr/bin/env python3
"""
NDS baseline: Isotonic regression on LLM OOD probabilities (Panel A, 6 paired groups only).
Forces monotonicity in wait_time WITHOUT pulling toward P_static.
This isolates the "pull-back-to-frontier" mechanism in BDT.

Panel A scenarios: 6 paired groups (low wait=2 vs high wait=6), 12 scenarios total.

Usage: python3 scripts/nds_isotonic_baseline.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.isotonic import IsotonicRegression

WORKSPACE = Path(__file__).resolve().parent.parent
LAMBDA = 0.25

# ── Load data ──────────────────────────────────────────────────────
def load_parsed_probs(path: Path) -> dict[str, dict[str, list[float]]]:
    by_model = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("parse_success", "").lower() in ("true", "1", "yes"):
                try:
                    prob = float(r["probability_0_1"])
                except (ValueError, TypeError):
                    continue
                by_model[r["model"]][r["scenario_id"]].append(prob)
    return by_model

def load_pstatic(path: Path) -> dict[str, float]:
    pstatic = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pstatic[r["scenario_id"]] = float(r["P_static"])
    return pstatic

# ── Metrics (from compute_out_of_design_metrics.py) ─────────────────
def mvr_wait_panel_a(preds: dict, scenarios: dict, panel_a_sids: list) -> float:
    """MVR-Wait over Panel A's 6 paired groups (low wait=2 vs high wait=6)."""
    violations = 0
    total = 0
    for s_low in panel_a_sids:
        # Find paired high-wait scenario
        s_high = s_low.replace("A", "B")  # A1<->B1, etc.
        if s_low not in preds or s_high not in preds:
            continue
        total += 1
        # Lower wait should have higher prob
        if preds[s_low] < preds[s_high]:
            violations += 1
    return violations / total if total > 0 else float('nan')

def spearmanr_safe(x, y):
    if len(x) < 3: return float('nan')
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except ImportError:
        return float('nan')

# ── Main ──────────────────────────────────────────────────────────
def main():
    parsed = load_parsed_probs(WORKSPACE / "results/out_of_design_parsed_probabilities.csv")
    pstatic_ood = load_pstatic(WORKSPACE / "results/out_of_design_pstatic.csv")
    scenarios = {}
    with open(WORKSPACE / "out_of_design_scenarios.csv", newline="") as f:
        for r in csv.DictReader(f):
            scenarios[r["scenario_id"]] = {
                "wait_time": float(r["wait_time"]),
                "group": r["expected_wait_order_group"],
            }

    # Panel A: scenarios A1-A6 (low wait=2), paired with B1-B6 (high wait=6)
    panel_a_low = [f"A{i}" for i in range(1, 7)]
    panel_a_high = [f"B{i}" for i in range(1, 7)]
    panel_a_sids = panel_a_low + panel_a_high

    print("=" * 80)
    print("NDS baseline: Isotonic on LLM OOD probabilities (Panel A only, 6 paired groups)")
    print("=" * 80)
    print(f"Models in parsed data: {sorted(parsed.keys())}")
    print(f"Panel A scenarios: {panel_a_sids}")
    print(f"P_static (Panel A): {[(s, pstatic_ood.get(s, 'NA')) for s in panel_a_sids]}")
    print()

    results = []
    for model in sorted(parsed.keys()):
        # Average LLM probability per scenario
        p_llm_mean = {sid: mean(parsed[model][sid]) for sid in panel_a_sids if parsed[model][sid]}

        if not p_llm_mean:
            print(f"[{model}] No data, skipping")
            continue

        # ── NDS: Isotonic on LLM OOD probabilities only (no P_static anchoring) ──
        # For each pair (Ai, Bi): force monotonicity w.r.t. wait_time
        # Apply isotonic globally: x = combined OOD probs, y = same
        # For 2-point pairs: predict = clip to [0, 1] with monotonic constraint
        p_nds = {}
        for sid_lo, sid_hi in zip(panel_a_low, panel_a_high):
            if sid_lo not in p_llm_mean or sid_hi not in p_llm_mean:
                continue
            # Isotonic: enforce wait decreasing => prob non-increasing
            x = [2.0, 6.0]  # wait times
            y = [p_llm_mean[sid_lo], p_llm_mean[sid_hi]]
            # Isotonic with wait as feature (decreasing)
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(x, y)
            p_nds[sid_lo] = float(iso.predict([2.0])[0])
            p_nds[sid_hi] = float(iso.predict([6.0])[0])

        # ── BDT (λ=0.25): pull toward P_static ──
        p_bdt = {}
        for sid in p_llm_mean:
            ps = pstatic_ood.get(sid, 0.5)
            p_bdt[sid] = LAMBDA * p_llm_mean[sid] + (1 - LAMBDA) * ps

        # ── Compute MVR-Wait ──
        mvr_llm = mvr_wait_panel_a(p_llm_mean, scenarios, panel_a_low)
        mvr_nds = mvr_wait_panel_a(p_nds, scenarios, panel_a_low)
        mvr_bdt = mvr_wait_panel_a(p_bdt, scenarios, panel_a_low)

        # ── Compute Spearman (with P_static) ──
        valid_sids = [s for s in panel_a_sids if s in p_llm_mean and s in pstatic_ood]
        if len(valid_sids) >= 3:
            ps = [pstatic_ood[s] for s in valid_sids]
            rho_llm = spearmanr_safe([p_llm_mean[s] for s in valid_sids], ps)
            rho_nds = spearmanr_safe([p_nds[s] for s in valid_sids], ps)
            rho_bdt = spearmanr_safe([p_bdt[s] for s in valid_sids], ps)
        else:
            rho_llm = rho_nds = rho_bdt = float('nan')

        # ── Policy Ranking Consistency (PRC) ──
        # For each pair: does model agree with P_static on which is higher?
        prc_llm = prc_nds = prc_bdt = 0
        prc_total = 0
        for sid_lo, sid_hi in zip(panel_a_low, panel_a_high):
            if sid_lo not in p_llm_mean or sid_hi not in p_llm_mean:
                continue
            if sid_lo not in pstatic_ood or sid_hi not in pstatic_ood:
                continue
            prc_total += 1
            # True ranking: lo < hi wait => P_static(lo) > P_static(hi)
            true_lo_higher = pstatic_ood[sid_lo] > pstatic_ood[sid_hi]
            if (p_llm_mean[sid_lo] > p_llm_mean[sid_hi]) == true_lo_higher:
                prc_llm += 1
            if (p_nds[sid_lo] > p_nds[sid_hi]) == true_lo_higher:
                prc_nds += 1
            if (p_bdt[sid_lo] > p_bdt[sid_hi]) == true_lo_higher:
                prc_bdt += 1
        prc_llm = prc_llm / prc_total if prc_total else float('nan')
        prc_nds = prc_nds / prc_total if prc_total else float('nan')
        prc_bdt = prc_bdt / prc_total if prc_total else float('nan')

        print(f"--- {model} ---")
        print(f"  Per-scenario LLM probs (Panel A):")
        for i, (sl, sh) in enumerate(zip(panel_a_low, panel_a_high), 1):
            if sl in p_llm_mean and sh in p_llm_mean:
                ps_lo = pstatic_ood.get(sl, 'NA')
                ps_hi = pstatic_ood.get(sh, 'NA')
                print(f"    Pair {i}: LLM {p_llm_mean[sl]:.3f}/{p_llm_mean[sh]:.3f}  P_static {ps_lo:.3f}/{ps_hi:.3f}  NDS {p_nds[sl]:.3f}/{p_nds[sh]:.3f}  BDT {p_bdt[sl]:.3f}/{p_bdt[sh]:.3f}")
        print()
        print(f"  Method           | MVR-Wait | Spearman ρ | PRC")
        print(f"  Unconstrained LLM|  {mvr_llm:.3f}   |   {rho_llm:+.3f}   | {prc_llm:.3f}")
        print(f"  NDS (isotonic)   |  {mvr_nds:.3f}   |   {rho_nds:+.3f}   | {prc_nds:.3f}")
        print(f"  BDT (λ=0.25)     |  {mvr_bdt:.3f}   |   {rho_bdt:+.3f}   | {prc_bdt:.3f}")
        print()
        results.append({
            "model": model,
            "llm_mvr": mvr_llm, "llm_rho": rho_llm, "llm_prc": prc_llm,
            "nds_mvr": mvr_nds, "nds_rho": rho_nds, "nds_prc": prc_nds,
            "bdt_mvr": mvr_bdt, "bdt_rho": rho_bdt, "bdt_prc": prc_bdt,
        })

    # Save
    out_path = WORKSPACE / "results/nds_isotonic_baseline.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
