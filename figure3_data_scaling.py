#!/usr/bin/env python3
"""
Figure 3: Data-scaling learning curve (Static-BDT Anchor vs full-sample frontier).

No LLM calls. Fits conditional logit on respondent-sampled DCE subsets,
evaluates on fixed 64-state grid with Qwen unconstrained p_LLM_mean.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.optimize import minimize

from table1_metrics_lib import (
    chr_violation_rate,
    pearson_r,
    row_metrics,
    spearman_rho,
)

WORKSPACE = Path(__file__).resolve().parent
DCE_PATH = WORKSPACE / "analysis_output" / "dce_encoded.csv"
GRID_PATH = WORKSPACE / "bdt_eval_grid_static.csv"
PARSED_PATH = WORKSPACE / "llm_parsed_outputs_qwen72b_unconstrained.csv"
RESULTS_DIR = WORKSPACE / "results"
FIG_DIR = WORKSPACE / "figures"

SHARES = [0.30, 0.50, 0.70, 1.00]
SEEDS = [1, 2, 3, 4, 5]
ATTRS = ("WaitTime", "VaccineEfficacy", "SideEffects")
LAMBDA_GRID = [round(i * 0.05, 2) for i in range(21)]
MODEL_LABEL = "Qwen2.5-72B"


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def load_grid() -> tuple[list[dict], dict[str, float], dict[str, dict]]:
    rows = list(csv.DictReader(open(GRID_PATH, encoding="utf-8")))
    p_full = {r["state"]: float(r["P_static"]) for r in rows}
    meta = {
        r["state"]: {"wait": r["wait"], "eff": r["eff"], "se": r["se"]}
        for r in rows
    }
    return rows, p_full, meta


def load_p_llm_mean() -> dict[str, float]:
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in csv.DictReader(open(PARSED_PATH, encoding="utf-8")):
        if str(row.get("parse_success", "")).lower() not in ("true", "1", "yes"):
            continue
        by_state[row["state"]].append(float(row["probability_0_1"]))
    if len(by_state) < 64:
        print(f"WARNING: only {len(by_state)} states with parse_ok LLM outputs", file=sys.stderr)
    return {s: mean(v) for s, v in by_state.items()}


def load_dce_respondents() -> tuple[list[str], list[dict]]:
    """Return sorted respondent IDs and all DCE rows (long format)."""
    rows = list(csv.DictReader(open(DCE_PATH, encoding="utf-8")))
    rids = sorted({r["RespondentID"] for r in rows}, key=lambda x: int(x) if str(x).isdigit() else str(x))
    return rids, rows


def filter_dce_by_respondents(all_rows: list[dict], sampled_ids: set[str]) -> list[dict]:
    return [r for r in all_rows if r["RespondentID"] in sampled_ids]


def fit_conditional_logit(dce_rows: list[dict]) -> dict[str, float]:
    """MLE conditional logit: Choice ~ WaitTime + VaccineEfficacy + SideEffects."""
    by_chid: dict[str, list[dict]] = defaultdict(list)
    for r in dce_rows:
        try:
            w = float(r["WaitTime"])
            e = float(r["VaccineEfficacy"])
            s = float(r["SideEffects"])
            c = int(float(r["Choice"]))
        except (TypeError, ValueError):
            continue
        by_chid[r["chid"]].append({"x": (w, e, s), "choice": c})

    groups: list[tuple[np.ndarray, int]] = []
    for chid, alts in by_chid.items():
        if len(alts) < 2:
            continue
        chosen = [i for i, a in enumerate(alts) if a["choice"] == 1]
        if len(chosen) != 1:
            continue
        X = np.array([a["x"] for a in alts], dtype=float)
        groups.append((X, chosen[0]))

    if len(groups) < 20:
        raise ValueError(f"too few choice sets for clogit: {len(groups)}")

    def neg_ll(beta: np.ndarray) -> float:
        ll = 0.0
        for X, j in groups:
            v = X @ beta
            v = v - np.max(v)
            ev = np.exp(v)
            p = ev / ev.sum()
            ll += math.log(max(float(p[j]), 1e-300))
        return -ll

    x0 = np.array([-0.27, 1.46, -0.10])
    res = minimize(neg_ll, x0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"clogit failed: {res.message}")
    return dict(zip(ATTRS, res.x.tolist()))


def predict_p_static(coefs: dict[str, float], grid_rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in grid_rows:
        u = (
            coefs["WaitTime"] * float(r["wait"])
            + coefs["VaccineEfficacy"] * float(r["eff"])
            + coefs["SideEffects"] * float(r["se"])
        )
        out[r["state"]] = sigmoid(u)
    return out


def select_lambda_star(
    p_llm: dict[str, float], p_static_s: dict[str, float], p_full: dict[str, float]
) -> tuple[float, dict[str, float]]:
    states = sorted(p_full.keys(), key=lambda s: int(s))
    best_lam = 0.0
    best_mse = float("inf")
    best_pi: dict[str, float] = {}
    for lam in LAMBDA_GRID:
        pi = {s: lam * p_llm[s] + (1.0 - lam) * p_static_s[s] for s in states if s in p_llm}
        mse = mean((pi[s] - p_full[s]) ** 2 for s in states if s in pi)
        if mse < best_mse - 1e-12 or (abs(mse - best_mse) <= 1e-12 and lam > best_lam):
            best_mse = mse
            best_lam = lam
            best_pi = pi
    return best_lam, best_pi


def metrics_for_preds(
    preds: dict[str, float],
    p_full: dict[str, float],
    state_meta: dict[str, dict],
) -> dict[str, float | str]:
    states = sorted(p_full.keys(), key=lambda s: int(s))
    ss = [s for s in states if s in preds]
    targets = [p_full[s] for s in ss]
    predv = [preds[s] for s in ss]
    mse_v = mean((p - t) ** 2 for p, t in zip(predv, targets))
    mae_v = mean(abs(p - t) for p, t in zip(predv, targets))
    chr_w = chr_violation_rate(preds, state_meta, axis="wait")
    chr_se = chr_violation_rate(preds, state_meta, axis="se")
    rho = spearman_rho(predv, targets)
    pr = pearson_r(predv, targets)
    return {
        "MSE": mse_v,
        "MAE": mae_v,
        "CHR_Wait": chr_w,
        "CHR_SE": chr_se,
        "Spearman_rho": rho,
        "Pearson_r": pr,
    }


def run_scaling(
    *,
    seeds: list[int] = SEEDS,
    shares: list[float] = SHARES,
) -> tuple[list[dict], dict]:
    grid_rows, p_full, state_meta = load_grid()
    p_llm = load_p_llm_mean()
    all_rids, all_dce = load_dce_respondents()
    n_resp = len(all_rids)

    llm_metrics = metrics_for_preds(p_llm, p_full, state_meta)
    print("Unconstrained LLM baseline (constant across shares):")
    print(
        f"  MSE={llm_metrics['MSE']:.4f}  MAE={llm_metrics['MAE']:.4f}  "
        f"CHR-W={llm_metrics['CHR_Wait']:.4f}  CHR-SE={llm_metrics['CHR_SE']:.4f}  "
        f"rho={llm_metrics['Spearman_rho']:.4f}  r={llm_metrics['Pearson_r']:.4f}"
    )

    by_seed_rows: list[dict] = []
    for share in shares:
        n_sample = max(1, int(round(share * n_resp)))
        for seed in seeds:
            rng = random.Random(seed)
            sampled = set(rng.sample(all_rids, n_sample))
            sub_dce = filter_dce_by_respondents(all_dce, sampled)
            coefs = fit_conditional_logit(sub_dce)
            p_static_s = predict_p_static(coefs, grid_rows)
            lam_star, pi_star = select_lambda_star(p_llm, p_static_s, p_full)
            m = metrics_for_preds(pi_star, p_full, state_meta)
            by_seed_rows.append(
                {
                    "share": f"{share:.2f}",
                    "share_pct": int(round(share * 100)),
                    "seed": seed,
                    "model_type": "conditional_logit",
                    "n_respondents": n_sample,
                    "selected_lambda": f"{lam_star:.2f}",
                    "MSE": f"{m['MSE']:.8f}",
                    "MAE": f"{m['MAE']:.8f}",
                    "CHR_Wait": f"{m['CHR_Wait']:.4f}" if m["CHR_Wait"] is not None else "N/A",
                    "CHR_SE": f"{m['CHR_SE']:.4f}" if m["CHR_SE"] is not None else "N/A",
                    "Spearman_rho": f"{m['Spearman_rho']:.4f}" if m["Spearman_rho"] is not None else "N/A",
                    "Pearson_r": f"{m['Pearson_r']:.4f}" if m["Pearson_r"] is not None else "N/A",
                    "beta_WaitTime": f"{coefs['WaitTime']:.6f}",
                    "beta_VaccineEfficacy": f"{coefs['VaccineEfficacy']:.6f}",
                    "beta_SideEffects": f"{coefs['SideEffects']:.6f}",
                }
            )
            print(
                f"share={share:.0%} seed={seed} n={n_sample} "
                f"λ*={lam_star:.2f} MSE={m['MSE']:.4f} CHR-W={m['CHR_Wait']:.4f} ρ={m['Spearman_rho']:.4f}"
            )

    return by_seed_rows, llm_metrics


def summarize(by_seed: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in by_seed:
        groups[r["share"]].append(r)

    summary: list[dict] = []
    for share in sorted(groups.keys(), key=float):
        rows = groups[share]
        for metric in ("MSE", "MAE", "CHR_Wait", "CHR_SE", "Spearman_rho", "Pearson_r"):
            vals = [float(r[metric]) for r in rows if r[metric] != "N/A"]
            summary.append(
                {
                    "share": share,
                    "metric": metric,
                    "metric_mean": f"{mean(vals):.8f}",
                    "metric_sd": f"{pstdev(vals):.6f}" if len(vals) > 1 else "0.000000",
                }
            )
        lam_vals = [float(r["selected_lambda"]) for r in rows]
        summary.append(
            {
                "share": share,
                "metric": "selected_lambda",
                "metric_mean": f"{mean(lam_vals):.4f}",
                "metric_sd": f"{pstdev(lam_vals):.4f}" if len(lam_vals) > 1 else "0.0000",
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_figure(by_seed: list[dict], llm_metrics: dict) -> list[dict]:
    import matplotlib.pyplot as plt

    shares_pct = [30, 50, 70, 100]
    share_keys = [f"{s:.2f}" for s in SHARES]

    def series(metric: str) -> tuple[list[float], list[float]]:
        means, sds = [], []
        for sk in share_keys:
            vals = [float(r[metric]) for r in by_seed if r["share"] == sk and r[metric] != "N/A"]
            means.append(mean(vals))
            sds.append(pstdev(vals) if len(vals) > 1 else 0.0)
        return means, sds

    panels = [
        ("MSE", "MSE", llm_metrics["MSE"]),
        ("CHR_Wait", "CHR–Wait", llm_metrics["CHR_Wait"]),
        ("Spearman_rho", "Spearman ρ", llm_metrics["Spearman_rho"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.32, left=0.08, right=0.98)
    fig.suptitle(
        "Data-scaling validation of Static-BDT anchoring",
        fontsize=11,
        fontweight="normal",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Fixed 64-state policy grid; benchmark = full-sample mixed-logit frontier",
        ha="center",
        fontsize=9,
        color="#333333",
    )

    curve_rows: list[dict] = []
    for ax, (key, ylab, baseline) in zip(axes, panels):
        mus, sds = series(key)
        ax.errorbar(
            shares_pct,
            mus,
            yerr=sds,
            fmt="o-",
            color="#1F618D",
            linewidth=1.8,
            markersize=6,
            capsize=3,
            elinewidth=1.2,
        )
        if baseline is not None and not (isinstance(baseline, float) and math.isnan(baseline)):
            ax.axhline(float(baseline), color="#B03A2E", linestyle="--", linewidth=1.2, label="Unconstrained LLM")
        ax.set_xlabel("DCE training share (%)")
        ax.set_ylabel(ylab)
        ax.set_xticks(shares_pct)
        ax.set_xlim(25, 105)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for pct, sk, m, sd in zip(shares_pct, share_keys, mus, sds):
            curve_rows.append(
                {
                    "share_pct": pct,
                    "share": sk,
                    "metric": key,
                    "mean": m,
                    "sd": sd,
                    "llm_baseline": baseline,
                }
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.01))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig3_data_scaling_learning_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_data_scaling_learning_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    return curve_rows


def print_summary_table(by_seed: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("Figure 3 summary (Static-BDT Anchor, MSE-opt λ, vs P_static full)")
    print("=" * 80)
    print(f"| {'Share':>6} | {'MSE':>8} | {'CHR-W':>7} | {'ρ':>6} | {'λ* mean':>8} |")
    print(f"|{'-'*8}|{'-'*10}|{'-'*9}|{'-'*8}|{'-'*10}|")
    for share in ["0.30", "0.50", "0.70", "1.00"]:
        rows = [r for r in by_seed if r["share"] == share]
        mse = mean(float(r["MSE"]) for r in rows)
        chr_w = mean(float(r["CHR_Wait"]) for r in rows)
        rho = mean(float(r["Spearman_rho"]) for r in rows)
        lam = mean(float(r["selected_lambda"]) for r in rows)
        print(f"| {int(float(share)*100):>5}% | {mse:>8.4f} | {chr_w:>7.4f} | {rho:>6.4f} | {lam:>8.2f} |")
    print("=" * 80)


def main() -> None:
    import os

    seeds = SEEDS
    if os.environ.get("FIG3_QUICK") == "1":
        seeds = [1]
        print("FIG3_QUICK=1: using seed 1 only")

    by_seed, llm_metrics = run_scaling(seeds=seeds)
    summary = summarize(by_seed)

    write_csv(
        RESULTS_DIR / "data_scaling_metrics_by_seed.csv",
        by_seed,
        [
            "share", "seed", "model_type", "n_respondents", "selected_lambda",
            "MSE", "MAE", "CHR_Wait", "CHR_SE", "Spearman_rho", "Pearson_r",
            "beta_WaitTime", "beta_VaccineEfficacy", "beta_SideEffects",
        ],
    )
    write_csv(
        RESULTS_DIR / "data_scaling_metrics_summary.csv",
        summary,
        ["share", "metric", "metric_mean", "metric_sd"],
    )

    curve_rows = plot_figure(by_seed, llm_metrics)
    write_csv(
        FIG_DIR / "fig3_data_scaling_learning_curve_data.csv",
        curve_rows,
        ["share_pct", "share", "metric", "mean", "sd", "llm_baseline"],
    )

    print_summary_table(by_seed)
    print(f"\nWrote {RESULTS_DIR / 'data_scaling_metrics_by_seed.csv'}")
    print(f"Wrote {RESULTS_DIR / 'data_scaling_metrics_summary.csv'}")
    print(f"Wrote {FIG_DIR / 'fig3_data_scaling_learning_curve.png'}")
    print(f"Wrote {FIG_DIR / 'fig3_data_scaling_learning_curve.pdf'}")


if __name__ == "__main__":
    main()
