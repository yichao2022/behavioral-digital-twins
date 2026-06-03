"""Shared Table 1 static-frontier metrics (λ sensitivity, no API)."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from statistics import mean

LAMBDA_GRID = [round(0.25 + i * 0.05, 2) for i in range(16)]


def pearson_r(x: list[float], y: list[float]) -> float | None:
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


def spearman_rho(x: list[float], y: list[float]) -> float | None:
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
        return pearson_r(ranks(x), ranks(y))


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
    model_label: str,
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
    rho = spearman_rho(predv, targets)
    pr = pearson_r(predv, targets)
    return {
        "model": model_label,
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


def compute_lambda_sensitivity(
    *,
    parsed_path: str,
    static_grid_path: str,
    metrics_out: str,
    anchor_out: str,
    model_label: str,
    print_table: bool = True,
) -> tuple[list[dict], float]:
    with open(static_grid_path, newline="") as f:
        grid = list(csv.DictReader(f))
    with open(parsed_path, newline="") as f:
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

    lam_star, pi_star = select_lambda_constrained(p_llm_mean, p_static)
    pi_25 = pi_bdt(p_llm_mean, p_static, 0.25)
    pi_50 = pi_bdt(p_llm_mean, p_static, 0.50)
    pi_75 = pi_bdt(p_llm_mean, p_static, 0.75)

    anchor_fields = [
        "state", "wait", "eff", "se", "P_static", "p_LLM_mean",
        "pi_BDT_lam0.25", "pi_BDT_lam0.50", "pi_BDT_lam0.75", "pi_BDT_mseopt", "lambda_star",
    ]
    with open(anchor_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=anchor_fields)
        w.writeheader()
        for s in states:
            if s not in p_llm_mean:
                continue
            g = grid_lookup[s]
            w.writerow({
                "state": s,
                "wait": g["wait"],
                "eff": g["eff"],
                "se": g["se"],
                "P_static": g["P_static"],
                "p_LLM_mean": f"{p_llm_mean[s]:.8f}",
                "pi_BDT_lam0.25": f"{pi_25[s]:.8f}",
                "pi_BDT_lam0.50": f"{pi_50[s]:.8f}",
                "pi_BDT_lam0.75": f"{pi_75[s]:.8f}",
                "pi_BDT_mseopt": f"{pi_star[s]:.8f}",
                "lambda_star": f"{lam_star:.2f}",
            })

    metrics_rows = [
        row_metrics(model_label, "Unconstrained LLM", p_llm_mean, p_static, state_meta, "", parse_rate),
    ]
    for lam in (0.25, 0.50, 0.75):
        pi = pi_bdt(p_llm_mean, p_static, lam)
        metrics_rows.append(
            row_metrics(
                model_label,
                f"Static-BDT Anchor (λ={lam:.2f})",
                pi,
                p_static,
                state_meta,
                f"{lam:.2f}",
                parse_rate,
            )
        )
    metrics_rows.append(
        row_metrics(
            model_label,
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
    with open(metrics_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(metrics_rows)

    if print_table:
        print(f"Wrote {anchor_out}")
        print(f"Wrote {metrics_out} ({len(metrics_rows)} rows)\n")
        print_table1_candidates(model_label, metrics_rows, lam_star)

    return metrics_rows, lam_star


def print_table1_candidates(model_label: str, rows: list[dict], lam_star: float) -> None:
    uncon = next(r for r in rows if r["method"] == "Unconstrained LLM")
    anchor = next(r for r in rows if "MSE-opt" in r["method"])
    print("=" * 72)
    print(f"TABLE 1 candidates — {model_label} (vs P_static)")
    print("=" * 72)
    print(f"| {'Method':<28} | {'MSE':>8} | {'MAE':>8} | {'CHR-W':>7} | {'CHR-SE':>7} | {'ρ':>6} | {'r':>6} |")
    print(f"|{'-'*30}|{'-'*10}|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*8}|{'-'*8}|")
    for r in (uncon, anchor):
        short = "Unconstrained LLM" if "Unconstrained" in r["method"] else f"Static-BDT Anchor (λ*={lam_star:.2f})"
        print(
            f"| {short:<28} | {float(r['MSE']):>8.4f} | {float(r['MAE']):>8.4f} | "
            f"{r['CHR_Wait']:>7} | {r['CHR_SE']:>7} | {r['Spearman_rho']:>6} | {r['Pearson_r']:>6} |"
        )
    print(f"| parse_success_rate           | {uncon['parse_success_rate']}")
    print("=" * 72)
