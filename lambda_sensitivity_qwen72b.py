from pathlib import Path
#!/usr/bin/env python3
"""Qwen2.5-72B λ-sensitivity table vs P_static (no API)."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from statistics import mean

from table1_metrics_lib import (
    chr_violation_rate,
    pearson_r,
    pi_bdt,
    spearman_rho,
)

WORKSPACE = str(Path(__file__).resolve().parent)
PARSED = os.path.join(WORKSPACE, "llm_parsed_outputs_qwen72b_unconstrained.csv")
STATIC_GRID = os.path.join(WORKSPACE, "bdt_eval_grid_static.csv")
OUT = os.path.join(WORKSPACE, "results", "lambda_sensitivity_qwen72b.csv")

MODEL_LABEL = "Qwen2.5-72B"
LAMBDAS = [0.25, 0.50, 0.75, 1.00]


def metrics_row(
    lam: float,
    preds: dict[str, float],
    p_static: dict[str, float],
    state_meta: dict[str, dict],
) -> dict:
    states = sorted(p_static.keys(), key=lambda s: int(s))
    ss = [s for s in states if s in preds]
    targets = [p_static[s] for s in ss]
    predv = [preds[s] for s in ss]
    mse_v = mean((p - t) ** 2 for p, t in zip(predv, targets))
    mae_v = mean(abs(p - t) for p, t in zip(predv, targets))
    chr_w = chr_violation_rate(preds, state_meta, axis="wait")
    chr_se = chr_violation_rate(preds, state_meta, axis="se")
    rho = spearman_rho(predv, targets)
    pr = pearson_r(predv, targets)
    return {
        "model": MODEL_LABEL,
        "lambda": f"{lam:.2f}",
        "MSE": f"{mse_v:.8f}",
        "MAE": f"{mae_v:.8f}",
        "CHR_Wait": f"{chr_w:.4f}" if chr_w is not None else "N/A",
        "CHR_SE": f"{chr_se:.4f}" if chr_se is not None else "N/A",
        "Spearman_rho": f"{rho:.4f}" if rho is not None else "N/A",
        "Pearson_r": f"{pr:.4f}" if pr is not None else "N/A",
    }


def latex_row(r: dict) -> str:
    lam = float(r["lambda"])
    return (
        f"{lam:.2f} & {float(r['MSE']):.4f} & {float(r['MAE']):.4f} & "
        f"{float(r['CHR_Wait']):.4f} & {float(r['CHR_SE']):.4f} & "
        f"{float(r['Spearman_rho']):.4f} \\\\"
    )


def main() -> None:
    with open(STATIC_GRID, newline="", encoding="utf-8") as f:
        grid = list(csv.DictReader(f))
    with open(PARSED, newline="", encoding="utf-8") as f:
        parsed = list(csv.DictReader(f))

    grid_lookup = {r["state"]: r for r in grid}
    p_static = {s: float(grid_lookup[s]["P_static"]) for s in grid_lookup}
    states = sorted(p_static.keys(), key=lambda s: int(s))

    by_state: dict[str, list[float]] = defaultdict(list)
    for row in parsed:
        if str(row.get("parse_success", "")).lower() in ("true", "1", "yes"):
            by_state[row["state"]].append(float(row["probability_0_1"]))

    p_llm_mean = {s: mean(by_state[s]) for s in states if s in by_state and by_state[s]}
    print(f"States with p_LLM_mean: {len(p_llm_mean)} / {len(states)}")

    state_meta = {
        s: {"wait": grid_lookup[s]["wait"], "eff": grid_lookup[s]["eff"], "se": grid_lookup[s]["se"]}
        for s in states
    }

    rows: list[dict] = []
    for lam in LAMBDAS:
        pi = pi_bdt(p_llm_mean, p_static, lam)
        rows.append(metrics_row(lam, pi, p_static, state_meta))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = [
        "model", "lambda", "MSE", "MAE", "CHR_Wait", "CHR_SE", "Spearman_rho", "Pearson_r",
    ]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT}\n")
    print("LaTeX rows (λ, MSE, MAE, CHR-Wait, CHR-SE, Spearman ρ):")
    print("-" * 72)
    for r in rows:
        print(latex_row(r))
    print("-" * 72)

    print("\nMarkdown check vs expected:")
    print(f"| {'λ':>4} | {'MSE':>8} | {'MAE':>8} | {'CHR-W':>7} | {'CHR-SE':>7} | {'ρ':>7} |")
    for r in rows:
        print(
            f"| {r['lambda']:>4} | {float(r['MSE']):>8.4f} | {float(r['MAE']):>8.4f} | "
            f"{r['CHR_Wait']:>7} | {r['CHR_SE']:>7} | {r['Spearman_rho']:>7} |"
        )


if __name__ == "__main__":
    main()
