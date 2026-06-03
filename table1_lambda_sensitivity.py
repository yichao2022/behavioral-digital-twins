from pathlib import Path
#!/usr/bin/env python3
"""Recompute Table 1 metrics with fixed λ and constrained λ grid (no API, λ≥0.25)."""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from statistics import mean

WORKSPACE = str(Path(__file__).resolve().parent)
PARSED = os.path.join(WORKSPACE, "llm_parsed_outputs_deepseek_unconstrained.csv")
STATIC_GRID = os.path.join(WORKSPACE, "bdt_eval_grid_static.csv")
OUT = os.path.join(WORKSPACE, "metrics_table1_deepseek_static_lambda_sensitivity.csv")

MODEL_LABEL = "DeepSeek V4 Pro"
LAMBDA_GRID = [round(0.25 + i * 0.05, 2) for i in range(16)]  # 0.25 .. 1.0


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _spearman(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except ImportError:
        def ranks(vals: list[float]) -> list[float]:
            order = sorted(range(n), key=lambda i: vals[i])
            r = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    r[order[k]] = avg_rank
                i = j + 1
            return r
        return _pearson(ranks(x), ranks(y))


def chr_violation_rate(
    state_preds: dict[str, float],
    state_meta: dict[str, dict],
    *,
    axis: str,
) -> float | None:
    state_list = [
        (s, float(state_meta[s]["wait"]), float(state_meta[s]["eff"]), float(state_meta[s]["se"]), state_preds[s])
        for s in state_preds
        if s in state_meta
    ]
    violations = total = 0
    for i, (s1, w1, e1, se1, p1) in enumerate(state_list):
        for j, (s2, w2, e2, se2, p2) in enumerate(state_list):
            if i >= j:
                continue
            if axis == "wait" and e1 == e2 and se1 == se2 and w1 != w2:
                total += 1
                if (w1 > w2 and p1 > p2) or (w1 < w2 and p1 < p2):
                    violations += 1
            elif axis == "se" and w1 == w2 and e1 == e2 and se1 != se2:
                total += 1
                if (se1 > se2 and p1 > p2) or (se1 < se2 and p1 < p2):
                    violations += 1
    return violations / total if total else None


def pi_bdt(p_llm: dict[str, float], p_static: dict[str, float], lam: float) -> dict[str, float]:
    return {s: lam * p_llm[s] + (1.0 - lam) * p_static[s] for s in p_llm}


def select_lambda_constrained(
    p_llm: dict[str, float], p_static: dict[str, float]
) -> tuple[float, dict[str, float]]:
    states = sorted(p_static.keys(), key=lambda s: int(s))
    best_lambda = 1.0
    best_mse = float("inf")
    best_pi: dict[str, float] = {}
    for lam in LAMBDA_GRID:
        pi = {s: lam * p_llm[s] + (1.0 - lam) * p_static[s] for s in states}
        mse = mean((pi[s] - p_static[s]) ** 2 for s in states)
        if mse < best_mse - 1e-12 or (abs(mse - best_mse) <= 1e-12 and lam > best_lambda):
            best_mse = mse
            best_lambda = lam
            best_pi = pi
    return best_lambda, best_pi


def row_metrics(
    method: str,
    preds: dict[str, float],
    p_static: dict[str, float],
    state_meta: dict[str, dict],
    lam: str,
    parse_rate: float,
) -> dict:
    states = sorted(p_static.keys(), key=lambda s: int(s))
    ss = [s for s in states if s in preds]
    targets = [p_static[s] for s in ss]
    predv = [preds[s] for s in ss]
    mse_v = mean((p - t) ** 2 for p, t in zip(predv, targets))
    mae_v = mean(abs(p - t) for p, t in zip(predv, targets))
    chr_w = chr_violation_rate(preds, state_meta, axis="wait")
    chr_se = chr_violation_rate(preds, state_meta, axis="se")
    rho = _spearman(predv, targets)
    pr = _pearson(predv, targets)
    return {
        "model": MODEL_LABEL,
        "method": method,
        "MSE": f"{mse_v:.8f}",
        "MAE": f"{mae_v:.8f}",
        "CHR_Wait": f"{chr_w:.4f}" if chr_w is not None else "N/A",
        "CHR_SE": f"{chr_se:.4f}" if chr_se is not None else "N/A",
        "Spearman_rho": f"{rho:.4f}" if rho is not None else "N/A",
        "Pearson_r": f"{pr:.4f}" if pr is not None else "N/A",
        "selected_lambda": lam,
        "parse_success_rate": f"{parse_rate:.4f}",
    }


def main() -> None:
    with open(STATIC_GRID, newline="") as f:
        grid = list(csv.DictReader(f))
    with open(PARSED, newline="") as f:
        parsed = list(csv.DictReader(f))

    grid_lookup = {r["state"]: r for r in grid}
    p_static = {s: float(grid_lookup[s]["P_static"]) for s in grid_lookup}
    states = sorted(p_static.keys(), key=lambda s: int(s))

    by_state: dict[str, list[float]] = defaultdict(list)
    for row in parsed:
        if str(row.get("parse_success", "")).lower() in ("true", "1", "yes"):
            by_state[row["state"]].append(float(row["probability_0_1"]))

    p_llm_mean = {s: mean(by_state[s]) for s in states if s in by_state and by_state[s]}
    parse_ok = sum(
        1 for r in parsed if str(r.get("parse_success", "")).lower() in ("true", "1", "yes")
    )
    parse_rate = parse_ok / len(parsed) if parsed else 0.0

    state_meta = {
        s: {"wait": grid_lookup[s]["wait"], "eff": grid_lookup[s]["eff"], "se": grid_lookup[s]["se"]}
        for s in states
    }

    rows = [
        row_metrics("Unconstrained LLM", p_llm_mean, p_static, state_meta, "", parse_rate),
    ]

    for lam in (0.25, 0.50, 0.75):
        pi = pi_bdt(p_llm_mean, p_static, lam)
        rows.append(
            row_metrics(
                f"Static-BDT Anchor (λ={lam:.2f})",
                pi,
                p_static,
                state_meta,
                f"{lam:.2f}",
                parse_rate,
            )
        )

    lam_star, pi_star = select_lambda_constrained(p_llm_mean, p_static)
    rows.append(
        row_metrics(
            f"Static-BDT Anchor (MSE-opt, λ={lam_star:.2f})",
            pi_star,
            p_static,
            state_meta,
            f"{lam_star:.2f}",
            parse_rate,
        )
    )

    fields = [
        "model", "method", "MSE", "MAE", "CHR_Wait", "CHR_SE",
        "Spearman_rho", "Pearson_r", "selected_lambda", "parse_success_rate",
    ]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT} ({len(rows)} rows)\n")
    print(f"{'Method':<42} {'MSE':>8} {'MAE':>8} {'CHR-W':>7} {'CHR-SE':>7} {'ρ':>7} {'r':>7}  λ")
    print("-" * 95)
    for r in rows:
        lam = r["selected_lambda"] or "—"
        print(
            f"{r['method']:<42} {float(r['MSE']):>8.4f} {float(r['MAE']):>8.4f} "
            f"{r['CHR_Wait']:>7} {r['CHR_SE']:>7} {r['Spearman_rho']:>7} {r['Pearson_r']:>7}  {lam}"
        )
    print(f"\nMSE-opt λ* (grid {LAMBDA_GRID[0]}–{LAMBDA_GRID[-1]}): {lam_star:.2f}")


if __name__ == "__main__":
    main()
