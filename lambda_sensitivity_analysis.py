#!/usr/bin/env python3
"""
Lambda sensitivity: frontier alignment (64-state) and held-out predictive (matched).

No LLM API calls. Qwen2.5-72B, main specification lambda=0.25.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from heldout_dce_validation import compute_metrics
from table1_metrics_lib import chr_violation_rate, spearman_rho

WORKSPACE = Path(__file__).resolve().parent
GRID = WORKSPACE / "bdt_eval_grid_static.csv"
PARSED = WORKSPACE / "llm_parsed_outputs_qwen72b_unconstrained.csv"
HELDOUT_PRED = WORKSPACE / "results" / "heldout_dce_predictions.csv"
OUT_FRONTIER = WORKSPACE / "results" / "lambda_sensitivity_frontier.csv"
OUT_HELDOUT = WORKSPACE / "results" / "lambda_sensitivity_heldout.csv"
MAIN_LAMBDA = 0.25
LAMBDAS = sorted({round(i * 0.10, 2) for i in range(11)} | {MAIN_LAMBDA})  # 0..1 step 0.1 + main


def load_grid_llm() -> tuple[dict[str, float], dict[str, float], dict[str, dict]]:
    grid = list(csv.DictReader(open(GRID, newline="", encoding="utf-8")))
    p_static = {r["state"]: float(r["P_static"]) for r in grid}
    state_meta = {
        r["state"]: {"wait": r["wait"], "eff": r["eff"], "se": r["se"]}
        for r in grid
    }
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in csv.DictReader(open(PARSED, newline="", encoding="utf-8")):
        if str(row.get("parse_success", "")).lower() not in ("true", "1", "yes"):
            continue
        by_state[row["state"]].append(float(row["probability_0_1"]))
    p_llm = {s: mean(v) for s, v in by_state.items() if s in p_static}
    return p_static, p_llm, state_meta


def frontier_row(
    lam: float,
    p_static: dict[str, float],
    p_llm: dict[str, float],
    state_meta: dict[str, dict],
) -> dict:
    states = sorted(p_static.keys(), key=lambda s: int(s))
    pi = {s: lam * p_llm[s] + (1.0 - lam) * p_static[s] for s in states}
    targets = [p_static[s] for s in states]
    predv = [pi[s] for s in states]
    mse = mean((p - t) ** 2 for p, t in zip(predv, targets))
    mae = mean(abs(p - t) for p, t in zip(predv, targets))
    mvr_w = chr_violation_rate(pi, state_meta, axis="wait")
    mvr_se = chr_violation_rate(pi, state_meta, axis="se")
    rho = spearman_rho(predv, targets)
    return {
        "lambda": f"{lam:.2f}",
        "main_spec": "yes" if abs(lam - MAIN_LAMBDA) < 1e-9 else "",
        "MSE": f"{mse:.8f}",
        "MAE": f"{mae:.8f}",
        "MVR_Wait": f"{mvr_w:.4f}" if mvr_w is not None else "NA",
        "MVR_SE": f"{mvr_se:.4f}" if mvr_se is not None else "NA",
        "Spearman_rho": f"{rho:.4f}" if rho is not None else "NA",
        "note": "Pure-DCE" if lam == 0 else ("Unconstrained LLM" if lam == 1 else ""),
    }


def load_matched_heldout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(HELDOUT_PRED, newline="", encoding="utf-8")))
    matched = [r for r in rows if str(r.get("matched_llm_state", "")).strip()]
    y = np.array([int(float(r["y"])) for r in matched], dtype=float)
    p_dce = np.array([float(r["p_dce"]) for r in matched], dtype=float)
    p_llm = np.array([float(r["p_llm"]) for r in matched], dtype=float)
    return y, p_dce, p_llm


def heldout_row(lam: float, y: np.ndarray, p_dce: np.ndarray, p_llm: np.ndarray) -> dict:
    p = lam * p_llm + (1.0 - lam) * p_dce
    m = compute_metrics(y, p)
    return {
        "lambda": f"{lam:.2f}",
        "main_spec": "yes" if abs(lam - MAIN_LAMBDA) < 1e-9 else "",
        "N": m["N"],
        "log_loss": f"{m['log_loss']:.6f}",
        "brier": f"{m['brier_score']:.6f}",
        "AUC": f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "NA",
        "calibration_intercept": f"{m['calibration_intercept']:.4f}"
        if not np.isnan(m["calibration_intercept"])
        else "NA",
        "calibration_slope": f"{m['calibration_slope']:.4f}"
        if not np.isnan(m["calibration_slope"])
        else "NA",
        "note": "Pure-DCE" if lam == 0 else ("Unconstrained LLM" if lam == 1 else ""),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_latex_frontier(rows: list[dict]) -> None:
    print("\n% Task A: Frontier-alignment sensitivity (vs P_static, 64 states)")
    print("\\begin{tabular}{c r r r r r l}")
    print("\\toprule")
    print("$\\lambda$ & MSE & MAE & MVR-Wait & MVR-SE & $\\rho$ & Note \\\\")
    print("\\midrule")
    for r in rows:
        star = "$^*$" if r["main_spec"] == "yes" else ""
        note = r.get("note", "")
        print(
            f"{r['lambda']}{star} & {float(r['MSE']):.4f} & {float(r['MAE']):.4f} & "
            f"{r['MVR_Wait']} & {r['MVR_SE']} & {r['Spearman_rho']} & {note} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("% $^*$ main specification ($\\lambda=0.25$)")


def print_latex_heldout(rows: list[dict]) -> None:
    print("\n% Task B: Held-out predictive sensitivity (matched N=809)")
    print("\\begin{tabular}{c r r r r r r l}")
    print("\\toprule")
    print("$\\lambda$ & $N$ & Log loss & Brier & AUC & Cal.\\ int. & Cal.\\ slope & Note \\\\")
    print("\\midrule")
    for r in rows:
        star = "$^*$" if r["main_spec"] == "yes" else ""
        note = r.get("note", "")
        print(
            f"{r['lambda']}{star} & {r['N']} & {r['log_loss']} & {r['brier']} & {r['AUC']} & "
            f"{r['calibration_intercept']} & {r['calibration_slope']} & {note} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("% $^*$ main specification ($\\lambda=0.25$); $\\lambda=0$: Pure-DCE ($p_{dce}$); $\\lambda=1$: LLM")


def main() -> None:
    p_static, p_llm, state_meta = load_grid_llm()
    if len(p_llm) != 64:
        print(f"WARNING: p_llm covers {len(p_llm)}/64 states")

    frontier_rows = [frontier_row(lam, p_static, p_llm, state_meta) for lam in LAMBDAS]
    write_csv(
        OUT_FRONTIER,
        ["lambda", "main_spec", "MSE", "MAE", "MVR_Wait", "MVR_SE", "Spearman_rho", "note"],
        frontier_rows,
    )

    y, p_dce, p_llm_h = load_matched_heldout()
    heldout_rows = [heldout_row(lam, y, p_dce, p_llm_h) for lam in LAMBDAS]
    write_csv(
        OUT_HELDOUT,
        [
            "lambda",
            "main_spec",
            "N",
            "log_loss",
            "brier",
            "AUC",
            "calibration_intercept",
            "calibration_slope",
            "note",
        ],
        heldout_rows,
    )

    print("=" * 72)
    print("Task A — Frontier alignment (64-state, benchmark = P_static)")
    print("=" * 72)
    print(f"{'λ':>5} {'MSE':>8} {'MAE':>8} {'MVR-W':>7} {'MVR-SE':>7} {'ρ':>7}  note")
    for r in frontier_rows:
        mark = " *" if r["main_spec"] else ""
        print(
            f"{r['lambda']:>5}{mark} {float(r['MSE']):>8.4f} {float(r['MAE']):>8.4f} "
            f"{r['MVR_Wait']:>7} {r['MVR_SE']:>7} {r['Spearman_rho']:>7}  {r.get('note','')}"
        )

    print("\n" + "=" * 72)
    print(f"Task B — Held-out predictive (matched N={len(y)}, p_bdt = λ*p_llm + (1-λ)*p_dce)")
    print("=" * 72)
    print(f"{'λ':>5} {'N':>5} {'log_loss':>9} {'brier':>8} {'AUC':>7}  note")
    for r in heldout_rows:
        mark = " *" if r["main_spec"] else ""
        print(
            f"{r['lambda']:>5}{mark} {r['N']:>5} {r['log_loss']:>9} {r['brier']:>8} "
            f"{r['AUC']:>7}  {r.get('note','')}"
        )

    print_latex_frontier(frontier_rows)
    print_latex_heldout(heldout_rows)

    print(f"\nWrote {OUT_FRONTIER}")
    print(f"Wrote {OUT_HELDOUT}")


if __name__ == "__main__":
    main()
