#!/usr/bin/env python3
"""Matched-sample held-out DCE validation tables (Panels A & B)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from heldout_dce_validation import compute_metrics

WORKSPACE = Path(__file__).resolve().parent
PREDICTIONS = WORKSPACE / "results" / "heldout_dce_predictions.csv"
OUT_DIR = WORKSPACE / "outputs" / "heldout_dce_validation"
LAMBDA = 0.25


def load_predictions() -> list[dict]:
    rows = list(csv.DictReader(open(PREDICTIONS, newline="", encoding="utf-8")))
    for r in rows:
        r["y"] = int(float(r["y"]))
        r["p_dce"] = float(r["p_dce"])
        if r.get("p_llm", "").strip():
            r["p_llm"] = float(r["p_llm"])
            r["p_bdt"] = float(r["p_bdt"])
        else:
            r["p_llm"] = None
            r["p_bdt"] = None
        r["matched"] = bool(str(r.get("matched_llm_state", "")).strip())
    return rows


def metrics_row(panel: str, method: str, y: np.ndarray, p: np.ndarray) -> dict:
    m = compute_metrics(y, p)
    return {
        "panel": panel,
        "method": method,
        "N": m["N"],
        "log_loss": f"{m['log_loss']:.6f}",
        "brier_score": f"{m['brier_score']:.6f}",
        "auc": f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "NA",
        "calibration_intercept": f"{m['calibration_intercept']:.4f}"
        if not np.isnan(m["calibration_intercept"])
        else "NA",
        "calibration_slope": f"{m['calibration_slope']:.4f}"
        if not np.isnan(m["calibration_slope"])
        else "NA",
        "mean_pred": f"{m['mean_pred']:.6f}",
        "observed_rate": f"{m['observed_rate']:.6f}",
        "_log_loss": m["log_loss"],
        "_brier": m["brier_score"],
    }


def fmt_delta(a: float, b: float) -> str:
    """a - b (Static-BDT relative to reference)."""
    return f"{a - b:+.6f}"


def write_tex(
    panel_a: list[dict],
    panel_b: list[dict],
    comparisons: list[tuple[str, str]],
    n_full: int,
    n_matched: int,
) -> None:
    path = OUT_DIR / "heldout_matched_table.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Held-out DCE validation: full test sample and LLM-matched subsample.}\n")
        f.write("\\label{tab:heldout-dce-matched}\n")
        f.write("\\small\n")

        f.write("\\textbf{Panel A: Full held-out test sample}\\\\\n")
        f.write("\\begin{tabular}{l r r r r r r}\n\\toprule\n")
        f.write("Method & $N$ & Log loss & Brier & AUC & Cal. int. & Cal. slope \\\\\n\\midrule\n")
        for r in panel_a:
            f.write(
                f"{r['method']} & {r['N']} & {r['log_loss']} & {r['brier_score']} & {r['auc']} & "
                f"{r['calibration_intercept']} & {r['calibration_slope']} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write("\\vspace{0.6em}\n")
        f.write("\\textbf{Panel B: LLM-matched held-out test sample (same $N$ for all methods)}\\\\\n")
        f.write("\\begin{tabular}{l r r r r r r}\n\\toprule\n")
        f.write("Method & $N$ & Log loss & Brier & AUC & Cal. int. & Cal. slope \\\\\n\\midrule\n")
        for r in panel_b:
            f.write(
                f"{r['method']} & {r['N']} & {r['log_loss']} & {r['brier_score']} & {r['auc']} & "
                f"{r['calibration_intercept']} & {r['calibration_slope']} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\n")

        f.write("\\vspace{0.4em}\n")
        f.write("\\begin{minipage}{0.92\\textwidth}\n\\footnotesize\n")
        f.write(
            f"Panel A uses all {n_full} held-out choice rows (20\\% respondent split, seed 2026). "
            f"Panel B restricts to {n_matched} rows whose $(w,e,s)$ attributes exactly match the "
            "64-state LLM policy grid; all three methods are evaluated on this same subsample. "
            "Static-BDT uses $\\lambda=0.25$. "
        )
        f.write(
            f"Matched-sample deltas: log-loss improvement over Unconstrained LLM = {comparisons[0][1]}; "
            f"Brier improvement = {comparisons[1][1]}; "
            f"log-loss minus Pure-DCE = {comparisons[2][1]}; "
            f"Brier minus Pure-DCE = {comparisons[3][1]}."
        )
        f.write("\n\\end{minipage}\n")
        f.write("\\end{table}\n")


def write_audit(n_full: int, n_matched: int, comparisons: list[tuple[str, str]]) -> None:
    path = OUT_DIR / "heldout_matched_audit.md"
    unmatched = n_full - n_matched
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Held-out DCE Matched-Sample Audit\n\n")
        f.write("## Why $N$ differs\n\n")
        f.write(
            f"- **Panel A (Pure-DCE):** $N={n_full}$ — every alternative-level row in the held-out "
            "20% respondent test split.\n"
        )
        f.write(
            f"- **Panel B (all methods):** $N={n_matched}$ — the subset of those rows whose "
            "`(wait, eff, se)` triple **exactly matches** an attribute combination in "
            "`bdt_eval_grid_static.csv` / Qwen unconstrained LLM outputs.\n"
        )
        f.write(
            f"- **Unmatched:** {unmatched} rows ({100*unmatched/n_full:.1f}%) lack an LLM probability "
            "because the DCE design includes attribute levels outside the 64-state grid.\n\n"
        )
        f.write("### DCE vs. 64-state grid\n\n")
        f.write("| Attribute | DCE (dce_encoded) | 64-state grid (LLM) |\n")
        f.write("|-----------|-------------------|---------------------|\n")
        f.write("| WaitTime | 0, 1, 2, 3, 6 | 0, 2, 4, 6 |\n")
        f.write("| Efficacy | 0, 0.5, 0.7, 0.95 | 0.3, 0.5, 0.7, 0.9 |\n")
        f.write("| SideEffects | 0, 1, 2, 3 | 0, 1, 2, 3 |\n\n")
        f.write(
            "Examples of unmatched DCE rows: `eff=0` or `0.95`, `wait=1` or `3`, or combinations "
            "that never appear in the grid. No nearest-neighbor imputation was used.\n\n"
        )
        f.write("## Files used\n\n")
        f.write(f"- `{PREDICTIONS}`\n")
        f.write("- `analysis_output/dce_encoded.csv` (source DCE, via prior run)\n")
        f.write("- `llm_parsed_outputs_qwen72b_unconstrained.csv` (LLM merge)\n\n")
        f.write("## Panel B matched-sample comparisons (Static-BDT, $\\lambda=0.25$)\n\n")
        for label, val in comparisons:
            f.write(f"- {label}: **{val}**\n")
        f.write("\n")
        f.write("Negative log-loss / Brier deltas vs. Unconstrained LLM mean Static-BDT is better on that metric.\n")
        f.write("Negative deltas vs. Pure-DCE mean Static-BDT beats the train-only DCE frontier on the matched subsample.\n")


def main() -> None:
    if not PREDICTIONS.is_file():
        raise SystemExit(f"Missing {PREDICTIONS}. Run heldout_dce_validation.py first.")

    rows = load_predictions()
    n_full = len(rows)
    matched = [r for r in rows if r["matched"]]
    n_matched = len(matched)

    y_full = np.array([r["y"] for r in rows], dtype=float)
    p_dce_full = np.array([r["p_dce"] for r in rows], dtype=float)

    y_m = np.array([r["y"] for r in matched], dtype=float)
    p_dce_m = np.array([r["p_dce"] for r in matched], dtype=float)
    p_llm_m = np.array([r["p_llm"] for r in matched], dtype=float)
    p_bdt_m = np.array([r["p_bdt"] for r in matched], dtype=float)

    panel_a = [metrics_row("A_full", "Pure-DCE", y_full, p_dce_full)]
    panel_b = [
        metrics_row("B_matched", "Pure-DCE", y_m, p_dce_m),
        metrics_row("B_matched", "Unconstrained LLM", y_m, p_llm_m),
        metrics_row("B_matched", f"Static-BDT Anchor ($\\lambda={LAMBDA:.2f}$)", y_m, p_bdt_m),
    ]

    bdt = panel_b[2]
    llm = panel_b[1]
    dce = panel_b[0]
    comparisons = [
        ("Static-BDT minus Unconstrained LLM (log loss)", fmt_delta(bdt["_log_loss"], llm["_log_loss"])),
        ("Static-BDT minus Unconstrained LLM (Brier)", fmt_delta(bdt["_brier"], llm["_brier"])),
        ("Static-BDT minus Pure-DCE (log loss)", fmt_delta(bdt["_log_loss"], dce["_log_loss"])),
        ("Static-BDT minus Pure-DCE (Brier)", fmt_delta(bdt["_brier"], dce["_brier"])),
    ]

    out_rows: list[dict] = []
    for r in panel_a + panel_b:
        out_rows.append({k: v for k, v in r.items() if not k.startswith("_")})
    for label, val in comparisons:
        row = {
            "panel": "B_comparisons",
            "method": label,
            "N": n_matched,
            "log_loss": "",
            "brier_score": "",
            "auc": "",
            "calibration_intercept": "",
            "calibration_slope": "",
            "mean_pred": "",
            "observed_rate": "",
        }
        if "log loss" in label.lower():
            row["log_loss"] = val
        if "brier" in label.lower():
            row["brier_score"] = val
        out_rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "panel",
        "method",
        "N",
        "log_loss",
        "brier_score",
        "auc",
        "calibration_intercept",
        "calibration_slope",
        "mean_pred",
        "observed_rate",
    ]
    with open(OUT_DIR / "heldout_matched_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    write_tex(panel_a, panel_b, comparisons, n_full, n_matched)
    write_audit(n_full, n_matched, comparisons)

    print("Panel A — Full held-out test (Pure-DCE only)")
    for r in panel_a:
        print(
            f"  {r['method']}: N={r['N']} log_loss={r['log_loss']} brier={r['brier_score']} "
            f"auc={r['auc']}"
        )
    print(f"\nPanel B — LLM-matched subsample (N={n_matched}, all methods)")
    for r in panel_b:
        print(
            f"  {r['method']}: N={r['N']} log_loss={r['log_loss']} brier={r['brier_score']} "
            f"auc={r['auc']}"
        )
    print("\nMatched-sample comparisons (Static-BDT):")
    for label, val in comparisons:
        print(f"  {label}: {val}")
    print(f"\nWrote {OUT_DIR}/heldout_matched_table.tex")
    print(f"Wrote {OUT_DIR}/heldout_matched_metrics.csv")
    print(f"Wrote {OUT_DIR}/heldout_matched_audit.md")


if __name__ == "__main__":
    main()
