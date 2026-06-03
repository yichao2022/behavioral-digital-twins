#!/usr/bin/env python3
"""
Appendix E.2 / Figure 3 — fixed λ=0.25 data-scaling table (Qwen2.5-72B, no API).

pi_BDT_s = 0.25 * p_LLM_mean + 0.75 * P_static_s
Metrics vs P_static_full from bdt_eval_grid_static.csv.
At 100% training share, P_static_s := P_static_full (full-sample MXL grid frontier).
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from figure3_data_scaling import (
    SHARES,
    SEEDS,
    filter_dce_by_respondents,
    fit_conditional_logit,
    load_dce_respondents,
    load_grid,
    load_p_llm_mean,
    predict_p_static,
)
from table1_metrics_lib import chr_violation_rate, spearman_rho

WORKSPACE = Path(__file__).resolve().parent
OUT = WORKSPACE / "results" / "data_scaling_fixed_lambda025_table.csv"
LAMBDA = 0.25
INTERMEDIATE = WORKSPACE / "results" / "data_scaling_pstatic_by_share.csv"


def metrics_fixed_lambda(
    p_llm: dict[str, float],
    p_static_s: dict[str, float],
    p_full: dict[str, float],
    state_meta: dict[str, dict],
) -> dict[str, float]:
    states = sorted(p_full.keys(), key=lambda s: int(s))
    pi = {
        s: LAMBDA * p_llm[s] + (1.0 - LAMBDA) * p_static_s[s]
        for s in states
        if s in p_llm and s in p_static_s
    }
    targets = [p_full[s] for s in states if s in pi]
    predv = [pi[s] for s in states if s in pi]
    mse = mean((p - t) ** 2 for p, t in zip(predv, targets))
    chr_w = chr_violation_rate(pi, state_meta, axis="wait")
    rho = spearman_rho(predv, targets)
    return {"MSE": mse, "CHR_Wait": chr_w, "Spearman_rho": rho}


def p_static_for_share(
    share: float,
    grid_rows: list[dict],
    all_rids: list[str],
    all_dce: list[dict],
    p_full: dict[str, float],
    seeds: list[int],
) -> tuple[dict[str, float], str]:
    """Return P_static_s (state -> prob) and description of source."""
    if share >= 1.0 - 1e-9:
        return dict(p_full), "full_sample_grid_P_static"

    by_seed: list[dict[str, float]] = []
    n_resp = len(all_rids)
    n_sample = max(1, int(round(share * n_resp)))
    for seed in seeds:
        rng = random.Random(seed)
        sampled = set(rng.sample(all_rids, n_sample))
        sub_dce = filter_dce_by_respondents(all_dce, sampled)
        coefs = fit_conditional_logit(sub_dce)
        by_seed.append(predict_p_static(coefs, grid_rows))

    # Mean frontier across seeds (same state keys)
    states = sorted(p_full.keys(), key=lambda s: int(s))
    p_mean = {
        s: mean(ps[s] for ps in by_seed)
        for s in states
    }
    return p_mean, f"conditional_logit_mean_over_{len(seeds)}_seeds"


def save_intermediate(rows: list[dict]) -> None:
    fields = ["share", "share_pct", "state", "P_static_s", "source"]
    with open(INTERMEDIATE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    grid_rows, p_full, state_meta = load_grid()
    p_llm = load_p_llm_mean()
    all_rids, all_dce = load_dce_respondents()

    table_rows: list[dict] = []
    intermediate: list[dict] = []

    for share in SHARES:
        p_static_s, source = p_static_for_share(
            share, grid_rows, all_rids, all_dce, p_full, SEEDS
        )
        pct = int(round(share * 100))
        for s, v in p_static_s.items():
            intermediate.append(
                {
                    "share": f"{share:.2f}",
                    "share_pct": pct,
                    "state": s,
                    "P_static_s": f"{v:.8f}",
                    "source": source,
                }
            )

        if share >= 1.0 - 1e-9:
            m = metrics_fixed_lambda(p_llm, p_static_s, p_full, state_meta)
        else:
            # Metrics per seed, then average (frontier already averaged above)
            # Also compute per-seed metrics for robustness
            seed_metrics = []
            n_resp = len(all_rids)
            n_sample = max(1, int(round(share * n_resp)))
            for seed in SEEDS:
                rng = random.Random(seed)
                sampled = set(rng.sample(all_rids, n_sample))
                sub_dce = filter_dce_by_respondents(all_dce, sampled)
                coefs = fit_conditional_logit(sub_dce)
                ps = predict_p_static(coefs, grid_rows)
                seed_metrics.append(metrics_fixed_lambda(p_llm, ps, p_full, state_meta))
            m = {
                "MSE": mean(x["MSE"] for x in seed_metrics),
                "CHR_Wait": mean(x["CHR_Wait"] for x in seed_metrics if x["CHR_Wait"] is not None),
                "Spearman_rho": mean(x["Spearman_rho"] for x in seed_metrics if x["Spearman_rho"] is not None),
            }

        table_rows.append(
            {
                "share_pct": pct,
                "share": f"{share:.2f}",
                "lambda": f"{LAMBDA:.2f}",
                "P_static_source": source,
                "MSE": f"{m['MSE']:.8f}",
                "CHR_Wait": f"{m['CHR_Wait']:.4f}",
                "Spearman_rho": f"{m['Spearman_rho']:.4f}",
            }
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["share_pct", "share", "lambda", "P_static_source", "MSE", "CHR_Wait", "Spearman_rho"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table_rows)

    save_intermediate(intermediate)

    print(f"Wrote {OUT}")
    print(f"Wrote {INTERMEDIATE} (P_static_s by share × state)\n")
    print("LaTeX rows (share\\% & MSE & CHR-Wait & Spearman $\\rho$):")
    print("-" * 56)
    for r in table_rows:
        print(
            f"{r['share_pct']}\\% & {float(r['MSE']):.4f} & "
            f"{float(r['CHR_Wait']):.4f} & {float(r['Spearman_rho']):.4f} \\\\"
        )
    print("-" * 56)
    print("\n| Share | MSE | CHR-Wait | Spearman ρ |")
    print("|-------|-----|----------|------------|")
    for r in table_rows:
        print(
            f"| {r['share_pct']}% | {float(r['MSE']):.4f} | "
            f"{float(r['CHR_Wait']):.4f} | {float(r['Spearman_rho']):.4f} |"
        )


if __name__ == "__main__":
    main()
