#!/usr/bin/env python3
"""
Compute all out-of-design stress test metrics and save outputs.
Run after LLM generation: python3 scripts/compute_out_of_design_metrics.py
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

LAMBDA = 0.25
WORKSPACE = Path(__file__).resolve().parent.parent

# ── load data ──────────────────────────────────────────────────────
def load_scenarios(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(parsed_path: Path, pstatic_path: Path, scenarios_path: Path) -> list[dict]:
    scenarios = {r["scenario_id"]: r for r in load_scenarios(scenarios_path)}

    pstatic: dict[str, float] = {}
    with open(pstatic_path, newline="") as f:
        for r in csv.DictReader(f):
            pstatic[r["scenario_id"]] = float(r["P_static"])

    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(parsed_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("parse_success", "").lower() in ("true", "1", "yes"):
                try:
                    prob = float(r["probability_0_1"])
                except (ValueError, TypeError):
                    continue
                by_model[r["model"]][r["scenario_id"]].append(prob)

    scenario_ids = sorted(pstatic.keys())
    results = []

    for model in sorted(by_model.keys()):
        p_llm_mean: dict[str, float] = {}
        for sid in scenario_ids:
            vals = by_model[model].get(sid, [])
            if vals:
                p_llm_mean[sid] = mean(vals)

        # P_static as reference list
        p_static_list = [pstatic[s] for s in scenario_ids]
        p_llm_list = [p_llm_mean.get(s) for s in scenario_ids]

        # ── Unconstrained LLM metrics ──
        valid_llm = [(p, ps) for p, ps in zip(p_llm_list, p_static_list) if p is not None]
        llm_vals = [v[0] for v in valid_llm]
        ps_vals = [v[1] for v in valid_llm]
        n_valid = len(valid_llm)

        # MAD from P_static
        mad_llm = mean(abs(a - b) for a, b in zip(llm_vals, ps_vals))

        # Spearman with P_static
        rho_llm = _spearmanr(llm_vals, ps_vals)

        # MVR-Wait over the 6 paired groups
        mvr_llm = _mvr_wait(p_llm_mean, scenarios, scenario_ids)

        # ── BDT (lambda=0.25) metrics ──
        p_bdt = {
            s: LAMBDA * p_llm_mean[s] + (1 - LAMBDA) * pstatic[s]
            for s in scenario_ids if s in p_llm_mean
        }
        bdt_vals = [p_bdt[s] for s in scenario_ids if s in p_bdt]
        ps_bdt = [pstatic[s] for s in scenario_ids if s in p_bdt]
        mad_bdt = mean(abs(a - b) for a, b in zip(bdt_vals, ps_bdt))
        rho_bdt = _spearmanr(bdt_vals, ps_bdt)
        mvr_bdt = _mvr_wait(p_bdt, scenarios, scenario_ids)

        # ── Direction of extra-policy-description response ──
        desc_responses = []
        for gname in sorted(set(s["expected_wait_order_group"] for s in scenarios.values())):
            g_sids = [s for s in scenario_ids if scenarios[s]["expected_wait_order_group"] == gname]
            if len(g_sids) < 1:
                continue
            s_low = next((s for s in g_sids if scenarios[s]["wait_time"] == "2"), None)
            if s_low and s_low in p_llm_mean:
                diff = p_llm_mean[s_low] - pstatic[s_low]
                desc_responses.append({
                    "group": gname,
                    "scenario": s_low,
                    "description": scenarios[s_low]["extra_policy_description"],
                    "p_llm": round(p_llm_mean[s_low], 4),
                    "p_static": round(pstatic[s_low], 4),
                    "delta": round(diff, 4),
                })

        results.append({
            "model": model,
            "method": "Unconstrained LLM",
            "MVR_Wait": mvr_llm,
            "MAD": round(mad_llm, 6),
            "Spearman_rho": round(rho_llm, 4) if rho_llm is not None else None,
            "n_scenarios": n_valid,
            "desc_responses": desc_responses,
        })
        results.append({
            "model": model,
            "method": f"Static-BDT (lambda={LAMBDA})",
            "MVR_Wait": mvr_bdt,
            "MAD": round(mad_bdt, 6),
            "Spearman_rho": round(rho_bdt, 4) if rho_bdt is not None else None,
            "n_scenarios": len(bdt_vals),
            "desc_responses": [],
        })

    return results, scenarios, pstatic


def _mvr_wait(preds: dict[str, float], scenarios: dict, scenario_ids: list[str]) -> float | None:
    """Compute MVR-Wait over the 6 paired groups."""
    violations = 0
    total = 0
    for gname in sorted(set(s["expected_wait_order_group"] for s in scenarios.values())):
        g_sids = [s for s in scenario_ids if scenarios[s]["expected_wait_order_group"] == gname]
        if len(g_sids) != 2:
            continue
        s_low = next((s for s in g_sids if scenarios[s]["wait_time"] == "2"), None)
        s_high = next((s for s in g_sids if scenarios[s]["wait_time"] == "6"), None)
        if s_low is None or s_high is None:
            continue
        p_low = preds.get(s_low)
        p_high = preds.get(s_high)
        if p_low is None or p_high is None:
            continue
        total += 1
        if p_low < p_high:
            violations += 1
    if total == 0:
        return None
    return violations / total


def _spearmanr(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except ImportError:
        pass
    def rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def save_metrics_csv(results: list[dict], path: Path) -> None:
    fields = ["model", "method", "MVR_Wait", "MAD", "Spearman_rho", "n_scenarios"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {path}")


def save_stress_test_table_tex(results: list[dict], desc_responses: dict[str, list[dict]],
                                scenarios: dict, pstatic: dict, path: Path) -> None:
    """LaTeX table of out-of-design stress test results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Out-of-Design Scenario Stress Test Results}",
        r"\label{tab:out-of-design}",
        r"\footnotesize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Method & MVR-Wait $\downarrow$ & MAD $\downarrow$ & "
        r"Spearman $\rho$ & $N$ \\",
        r"\midrule",
    ]
    for r in results:
        mvr = f"{r['MVR_Wait']:.4f}" if r["MVR_Wait"] is not None else "---"
        mad = f"{r['MAD']:.4f}" if r["MAD"] else "---"
        rho = f"{r['Spearman_rho']:.4f}" if r["Spearman_rho"] is not None else "---"
        model_short = r["model"].replace("Qwen2.5-72B", "Qwen-72B").replace("DeepSeek V4 Pro", "DS-V4")
        lines.append(
            f"{model_short} & {r['method']} & {mvr} & {mad} & {rho} & "
            f"{r['n_scenarios']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\par\smallskip")
    lines.append(
        r"\textit{Note:} MVR-Wait is the monotonicity violation rate over "
        r"6 wait-time paired scenario groups. "
        r"MAD is the mean absolute deviation from $P_{\mathrm{static}}$. "
        r"Spearman $\rho$ measures rank correlation with $P_{\mathrm{static}}$. "
        r"All scenarios hold vaccine efficacy fixed at 70\% and side-effect "
        r"risk at low."
    )
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    parsed_path = WORKSPACE / "results" / "out_of_design_parsed_probabilities.csv"
    pstatic_path = WORKSPACE / "results" / "out_of_design_pstatic.csv"
    scenarios_path = WORKSPACE / "out_of_design_scenarios.csv"
    metrics_out = WORKSPACE / "results" / "out_of_design_bdt_metrics.csv"
    tex_out = WORKSPACE / "results" / "out_of_design_stress_test_table.tex"

    if not parsed_path.exists():
        print(f"ERROR: {parsed_path} not found. Run LLM generation first.")
        print("  python3 scripts/out_of_design_stress_test.py run-llm")
        return

    results, scenarios, pstatic = compute_metrics(parsed_path, pstatic_path, scenarios_path)
    save_metrics_csv(results, metrics_out)

    # Group desc responses
    desc_by_model = {}
    for r in results:
        if r["desc_responses"]:
            desc_by_model[r["model"]] = r["desc_responses"]

    save_stress_test_table_tex(results, desc_by_model, scenarios, pstatic, tex_out)

    # Print summary
    print(f"\n{'='*72}")
    print("OUT-OF-DESIGN STRESS TEST — SUMMARY")
    print(f"{'='*72}")
    for r in results:
        mvr = f"{r['MVR_Wait']:.4f}" if r["MVR_Wait"] is not None else "---"
        print(f"  {r['model']:<20} | {r['method']:<30} | "
              f"MVR-Wait={mvr} | MAD={r['MAD']:.4f} | ρ={r['Spearman_rho']}")
    print(f"\nFiles saved:")
    print(f"  {metrics_out}")
    print(f"  {tex_out}")


if __name__ == "__main__":
    main()
