#!/usr/bin/env python3
"""
Uniform Shrinkage Baseline — independent of existing EFR implementation.

Computes:
  pi_uniform = lambda * p_global + (1-lambda) * P_static

and produces a comparison table across all three LLMs.

Usage:
  python3 uniform_shrinkage_baseline.py
"""
from __future__ import annotations

import csv
import sys
import os
from collections import defaultdict
from statistics import mean
from pathlib import Path

# Ensure we can import from the same workspace
WORKSPACE = str(Path(__file__).resolve().parent)
sys.path.insert(0, WORKSPACE)

from table1_metrics_lib import (
    pi_uniform,
    row_metrics,
    chr_violation_rate,
    spearman_rho,
)

# ── Configuration ──────────────────────────────────────────────────────────────
LAMBDA = 0.25
P_GLOBAL = 0.48223759  # overall mean acceptance prob from DCE data

MODELS: list[dict] = [
    {
        "label": "Qwen2.5-72B",
        "parsed_path": os.path.join(WORKSPACE, "llm_parsed_outputs_qwen72b_unconstrained.csv"),
    },
    {
        "label": "DeepSeek V4 Pro",
        "parsed_path": os.path.join(WORKSPACE, "llm_parsed_outputs_deepseek_unconstrained.csv"),
    },
    {
        "label": "MiroThinker",
        "parsed_path": os.path.join(WORKSPACE, "llm_parsed_outputs_mirothinker_unconstrained.csv"),
    },
]

STATIC_GRID = os.path.join(WORKSPACE, "bdt_eval_grid_static.csv")

# ── Helpers ────────────────────────────────────────────────────────────────────


def load_static_grid(path: str) -> tuple[dict[str, float], dict[str, dict]]:
    """Return (p_static, state_meta)."""
    with open(path, newline="") as f:
        grid = list(csv.DictReader(f))
    p_static: dict[str, float] = {}
    state_meta: dict[str, dict] = {}
    for r in grid:
        s = r["state"]
        p_static[s] = float(r["P_static"])
        state_meta[s] = {
            "wait": r["wait"],
            "eff": r["eff"],
            "se": r["se"],
        }
    return p_static, state_meta


def load_llm_means(parsed_path: str, p_static: dict[str, float]) -> dict[str, float]:
    """Compute mean LLM probability per state from parsed outputs."""
    with open(parsed_path, newline="") as f:
        parsed = list(csv.DictReader(f))
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in parsed:
        if str(row.get("parse_success", "")).lower() in ("true", "1", "yes"):
            by_state[row["state"]].append(float(row["probability_0_1"]))
    states = sorted(p_static.keys(), key=lambda s: int(s))
    p_llm: dict[str, float] = {}
    for s in states:
        if s in by_state and by_state[s]:
            p_llm[s] = mean(by_state[s])
    return p_llm


def parse_rate(parsed_path: str) -> float:
    """Compute parse success rate."""
    with open(parsed_path, newline="") as f:
        parsed = list(csv.DictReader(f))
    if not parsed:
        return 0.0
    ok = sum(
        1 for r in parsed
        if str(r.get("parse_success", "")).lower() in ("true", "1", "yes")
    )
    return ok / len(parsed)


def p_static_metrics(
    p_static: dict[str, float], state_meta: dict[str, dict]
) -> dict:
    """Metrics for Pure-DCE (P_static vs itself — by construction perfect)."""
    states = sorted(p_static.keys(), key=lambda s: int(s))
    vals = [p_static[s] for s in states]
    # MSE/MAE vs itself = 0
    # CHR = monotonicity violations in P_static
    chr_w = chr_violation_rate(p_static, state_meta, axis="wait")
    return {
        "method": "Pure-DCE (P_static)",
        "MSE": "0.00000000",
        "MAE": "0.00000000",
        "MVR_wait": f"{chr_w:.4f}" if chr_w is not None else "N/A",
        "Spearman": "1.0000",
    }


def fmt_float(x: float) -> str:
    return f"{x:.6f}"


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"{'='*80}")
    print(f"  Uniform Shrinkage Baseline Evaluation  (λ = {LAMBDA}, p_global = {P_GLOBAL:.6f})")
    print(f"{'='*80}")
    print()

    p_static, state_meta = load_static_grid(STATIC_GRID)

    # ── Compute EFR metrics from existing saved outputs ────────────────────
    # (Re-compute from scratch for each model so we have consistent numbers)
    efr_metrics: dict[str, dict] = {}

    for model in MODELS:
        label = model["label"]
        parsed_path = model["parsed_path"]
        if not os.path.isfile(parsed_path):
            print(f"  ⚠  Skipping {label}: no parsed outputs at {parsed_path}")
            continue

        p_llm = load_llm_means(parsed_path, p_static)
        pr = parse_rate(parsed_path)
        states = sorted(p_static.keys(), key=lambda s: int(s))
        ss = [s for s in states if s in p_llm]

        # ── Unconstrained LLM ──────────────────────────────────────────
        uncon_row = row_metrics(label, "Unconstrained LLM", p_llm, p_static, state_meta, "", pr)

        # ── EFR (λ=0.25) ───────────────────────────────────────────────
        from table1_metrics_lib import pi_bdt
        pi_efr = pi_bdt(p_llm, p_static, LAMBDA)
        efr_row = row_metrics(label, f"EFR (λ={LAMBDA:.2f})", pi_efr, p_static, state_meta, f"{LAMBDA:.2f}", pr)

        # ── Uniform Shrinkage (λ=0.25) ─────────────────────────────────
        pi_us = pi_uniform(P_GLOBAL, p_static, LAMBDA)
        us_row = row_metrics(label, f"Uniform Shrinkage (λ={LAMBDA:.2f})", pi_us, p_static, state_meta, f"{LAMBDA:.2f}", 1.0)

        # ── Pure-DCE ───────────────────────────────────────────────────
        pure_mvr = chr_violation_rate(p_static, state_meta, axis="wait")
        pure_row = {
            "method": "Pure-DCE (P_static)",
            "MSE": "0.00000000",
            "MAE": "0.00000000",
            "CHR_Wait": f"{pure_mvr:.4f}" if pure_mvr is not None else "N/A",
            "Spearman_rho": "1.0000",
            "Pearson_r": "1.0000",
            "selected_lambda": "",
            "parse_success_rate": "1.0000",
        }

        efr_metrics[label] = {
            "pure": pure_row,
            "us": us_row,
            "efr": efr_row,
            "uncon": uncon_row,
        }

    # ── Print comparison tables ────────────────────────────────────────────
    for label, m in efr_metrics.items():
        print(f"{'─'*80}")
        print(f"  {label}")
        print(f"{'─'*80}")
        print(f"  {'Method':<35} {'MSE':>10} {'MAE':>10} {'MVR_wait':>10} {'Spearman':>10}")
        print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
        for key in ("pure", "us", "efr", "uncon"):
            r = m[key]
            print(f"  {r['method']:<35} {r.get('MSE', 'N/A'):>10} {r.get('MAE', 'N/A'):>10} "
                  f"{r.get('CHR_Wait', 'N/A'):>10} {r.get('Spearman_rho', 'N/A'):>10}")
        print()

    # ── Interpretation ─────────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"  INTERPRETATION")
    print(f"{'='*80}")
    print()

    # For each model, compare EFR vs Uniform Shrinkage
    for label, m in efr_metrics.items():
        efr_mse = float(m["efr"]["MSE"])
        us_mse = float(m["us"]["MSE"])
        efr_mae = float(m["efr"]["MAE"])
        us_mae = float(m["us"]["MAE"])
        efr_mvr = float(m["efr"]["CHR_Wait"]) if m["efr"]["CHR_Wait"] != "N/A" else None
        us_mvr = float(m["us"]["CHR_Wait"]) if m["us"]["CHR_Wait"] != "N/A" else None

        print(f"  ── {label} ──")
        print(f"    EFR MSE:            {efr_mse:.8f}")
        print(f"    Uniform Shrink MSE: {us_mse:.8f}")
        print(f"    Δ MSE:              {efr_mse - us_mse:+.8f}")
        print(f"    (negative = EFR better)")
        print()
        print(f"    EFR MAE:            {efr_mae:.8f}")
        print(f"    Uniform Shrink MAE: {us_mae:.8f}")
        print(f"    Δ MAE:              {efr_mae - us_mae:+.8f}")
        print()
        if efr_mvr is not None and us_mvr is not None:
            print(f"    EFR MVR:            {efr_mvr:.4f}")
            print(f"    Uniform Shrink MVR: {us_mvr:.4f}")
            print(f"    Δ MVR:              {efr_mvr - us_mvr:+.4f}")
            print()

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"  Q1: How much of EFR's improvement can be explained ")
    print(f"      by simple shrinkage toward P_static?")
    print(f"{'='*80}")
    print()
    for label, m in efr_metrics.items():
        uncon_mse = float(m["uncon"]["MSE"])
        us_mse = float(m["us"]["MSE"])
        efr_mse = float(m["efr"]["MSE"])
        # Total possible improvement from uncon to Pure-DCE
        total_improvement = uncon_mse  # since Pure-DCE MSE = 0
        # Improvement from uncon to Uniform Shrinkage
        improvement_from_shrinkage = uncon_mse - us_mse
        # Remaining gap from Uniform Shrinkage to Pure-DCE
        remaining_gap = us_mse  # since Pure-DCE MSE = 0
        # How much EFR captures beyond uniform
        efr_improvement = uncon_mse - efr_mse

        pct_shrinkage = (improvement_from_shrinkage / total_improvement * 100) if total_improvement > 0 else 0
        pct_llm = ((efr_improvement - improvement_from_shrinkage) / total_improvement * 100) if total_improvement > 0 else 0

        print(f"  {label}:")
        print(f"    Improvement from shrinkage alone:  {improvement_from_shrinkage:.6f} MSE ({pct_shrinkage:.1f}%)")
        print(f"    Incremental from LLM heterogeneity: {efr_improvement - improvement_from_shrinkage:.6f} MSE ({pct_llm:.1f}%)")
        print(f"    Unexplained gap to P_static:        {efr_mse:.6f} MSE")
        print()

    print(f"{'='*80}")
    print(f"  Q2: Does retaining state-specific LLM heterogeneity")
    print(f"      provide measurable benefit beyond uniform shrinkage?")
    print(f"{'='*80}")
    print()
    for label, m in efr_metrics.items():
        efr_mse = float(m["efr"]["MSE"])
        us_mse = float(m["us"]["MSE"])
        delta = efr_mse - us_mse  # negative = EFR better
        if delta < -1e-8:
            print(f"  {label}: YES — EFR (MSE={efr_mse:.6f}) < Uniform (MSE={us_mse:.6f})")
            print(f"          LLM provides heterogeneous information beyond scalar shrinkage.")
        elif delta > 1e-8:
            print(f"  {label}: NO — Uniform (MSE={us_mse:.6f}) < EFR (MSE={efr_mse:.6f})")
            print(f"          LLM heterogeneity adds noise, not signal.")
        else:
            print(f"  {label}: NEGLIGIBLE — EFR ≈ Uniform (Δ = {delta:.8f})")
            print(f"          LLM heterogeneity provides no measurable benefit.")
        print()

    print(f"{'='*80}")
    print(f"  Q3: Implications if Uniform ≈ EFR")
    print(f"{'='*80}")
    print()
    print(f"  If uniform shrinkage matches EFR performance, the paper's")
    print(f"  contribution shifts from 'LLMs improve behavioral simulation'")
    print(f"  to 'parametric shrinkage toward a DCE frontier produces")
    print(f"  behaviorally consistent estimates.' The LLM becomes a")
    print(f"  interchangeable component — any predictor with state-level")
    print(f"  heterogeneity would produce similar results.")
    print()

    print(f"{'='*80}")
    print(f"  Q4: Incremental value of LLM component (if EFR > Uniform)")
    print(f"{'='*80}")
    print()
    for label, m in efr_metrics.items():
        efr_mse = float(m["efr"]["MSE"])
        us_mse = float(m["us"]["MSE"])
        delta = us_mse - efr_mse  # positive = EFR better
        if delta > 1e-8:
            pct = delta / us_mse * 100 if us_mse > 0 else 0
            print(f"  {label}: EFR reduces MSE by {delta:.6f} ({pct:.1f}%) over uniform shrinkage.")
            print(f"          The LLM's state-specific variation contributes measurably")
            print(f"          beyond a homogeneous shrinkage baseline.")
        else:
            print(f"  {label}: No incremental value detected (Δ ≤ 0).")
        print()


if __name__ == "__main__":
    main()
