#!/usr/bin/env python3
"""
Held-out DCE stated-choice validation (train-only frontier, fixed lambda=0.25).

Out-of-sample validation on held-out respondents' choice rows.
No LLM API calls.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.special import expit, logit as sp_logit
from scipy.stats import norm

WORKSPACE = Path(__file__).resolve().parent
RESULTS_DIR = WORKSPACE / "results"
GRID_PATH = WORKSPACE / "bdt_eval_grid_static.csv"
PARSED_LLM = WORKSPACE / "llm_parsed_outputs_qwen72b_unconstrained.csv"

SEED = 2026
TRAIN_FRAC = 0.80
LAMBDA = 0.25
CLIP_LO, CLIP_HI = 1e-6, 1.0 - 1e-6

RESPONDENT_ALIASES = ["respondentid", "respondent_id", "id", "subject_id"]
CHOICE_ALIASES = ["choice", "y", "chosen", "select", "decision"]
WAIT_ALIASES = ["waittime", "wait_time", "waiting_time", "wait"]
EFF_ALIASES = ["vaccineefficacy", "vaccine_efficacy", "efficacy", "eff"]
SE_ALIASES = ["sideeffects", "side_effects", "side_effect", "se"]
CASH_ALIASES = ["cashincentives", "cash_incentives", "cash", "incentive", "incentives"]


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _find_unique(columns: list[str], aliases: list[str], label: str) -> str:
    cols = {_norm(c): c for c in columns}
    hits = [cols[_norm(a)] for a in aliases if _norm(a) in cols]
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        print(f"\n❌ Could not find column for {label}.")
        print(f"   Available columns: {columns}")
        print(f"   Tried aliases: {aliases}")
        sys.exit(1)
    print(f"\n❌ Ambiguous columns for {label}: {hits}")
    print(f"   Available columns: {columns}")
    sys.exit(1)


def discover_dce_file() -> Path:
    patterns = [
        WORKSPACE / "analysis_output" / "dce_encoded.csv",
        WORKSPACE / "dce_repaired_clean.csv",
        WORKSPACE / "dce_repaired.csv",
        WORKSPACE / "analysis_output" / "dce_repaired.csv",
        WORKSPACE / "analysis_output" / "cleaned_dce_data_v2.csv",
    ]
    patterns += [Path(p) for p in glob.glob(str(WORKSPACE / "**/*dce*.csv"), recursive=True)]
    patterns += [Path(p) for p in glob.glob(str(WORKSPACE / "**/*choice*.csv"), recursive=True)]
    seen: set[str] = set()
    candidates: list[Path] = []
    for p in patterns:
        sp = str(p.resolve())
        if sp in seen or not p.is_file():
            continue
        seen.add(sp)
        if "synthetic" in sp.lower():
            continue
        candidates.append(p)

    scored: list[tuple[int, Path]] = []
    for path in candidates:
        try:
            rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        except OSError:
            continue
        if len(rows) < 1000:
            continue
        cols = list(rows[0].keys()) if rows else []
        score = 0
        ncol = {_norm(c) for c in cols}
        if any(_norm(a) in ncol for a in RESPONDENT_ALIASES):
            score += 2
        if any(_norm(a) in ncol for a in CHOICE_ALIASES):
            score += 2
        if any(_norm(a) in ncol for a in WAIT_ALIASES):
            score += 1
        if any(_norm(a) in ncol for a in EFF_ALIASES):
            score += 1
        if "dce_encoded" in path.name.lower():
            score += 5
        scored.append((score, path))

    if not scored:
        print("❌ No suitable DCE CSV found.")
        sys.exit(1)
    scored.sort(key=lambda x: (-x[0], -len(str(x[1]))))
    best = scored[0][1]
    print(f"Using DCE file: {best}")
    return best


def load_and_standardize_dce(path: Path) -> tuple[list[dict], dict[str, str]]:
    raw = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not raw:
        print(f"❌ Empty DCE file: {path}")
        sys.exit(1)
    columns = list(raw[0].keys())
    col_map = {
        "respondent_id": _find_unique(columns, RESPONDENT_ALIASES, "respondent_id"),
        "y": _find_unique(columns, CHOICE_ALIASES, "choice outcome (0/1)"),
        "wait": _find_unique(columns, WAIT_ALIASES, "WaitTime"),
        "eff": _find_unique(columns, EFF_ALIASES, "Efficacy"),
        "se": _find_unique(columns, SE_ALIASES, "SideEffects"),
    }
    cash_col = None
    ncol = {_norm(c) for c in columns}
    cash_hits = [c for c in columns if _norm(c) in {_norm(a) for a in CASH_ALIASES}]
    if len(cash_hits) == 1:
        cash_col = cash_hits[0]
        col_map["cash"] = cash_col
    elif len(cash_hits) > 1:
        print(f"\n❌ Ambiguous cash columns: {cash_hits}")
        sys.exit(1)

    rows: list[dict] = []
    dropped = 0
    for i, r in enumerate(raw):
        try:
            y = int(float(r[col_map["y"]]))
            if y not in (0, 1):
                dropped += 1
                continue
            wait = float(r[col_map["wait"]])
            eff = float(r[col_map["eff"]])
            se = float(r[col_map["se"]])
            cash = float(r[cash_col]) if cash_col else 0.0
            rid = str(r[col_map["respondent_id"]]).strip()
        except (TypeError, ValueError, KeyError):
            dropped += 1
            continue
        rows.append(
            {
                "row_id": i,
                "respondent_id": rid,
                "y": y,
                "wait": wait,
                "eff": eff,
                "se": se,
                "cash": cash,
            }
        )
    print(f"DCE rows kept: {len(rows)} (dropped {dropped})")
    print(f"Column mapping: {col_map}")
    return rows, col_map


def respondent_split(rows: list[dict]) -> tuple[set[str], set[str]]:
    rids = sorted({r["respondent_id"] for r in rows})
    rng = random.Random(SEED)
    shuffled = rids[:]
    rng.shuffle(shuffled)
    n_train = int(round(TRAIN_FRAC * len(shuffled)))
    n_train = max(1, min(n_train, len(shuffled) - 1))
    train_ids = set(shuffled[:n_train])
    test_ids = set(shuffled[n_train:])
    return train_ids, test_ids


def build_design(rows: list[dict], include_cash: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n = len(rows)
    names = ["const", "wait", "eff", "se"]
    X = np.ones((n, 4 if not include_cash else 5))
    y = np.array([r["y"] for r in rows], dtype=float)
    for i, r in enumerate(rows):
        X[i, 1] = r["wait"]
        X[i, 2] = r["eff"]
        X[i, 3] = r["se"]
        if include_cash:
            if i == 0:
                names.append("cash")
            X[i, 4] = r["cash"]
    return X, y, names


def fit_logit_mle(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binomial logit MLE via IRLS (equivalent to statsmodels.Logit / GLM Binomial)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        mu = expit(eta)
        w = np.clip(mu * (1 - mu), 1e-8, None)
        z_adj = eta + (y - mu) / w
        XtW = X.T * w  # each row weighted
        A = XtW @ X
        b = XtW @ z_adj
        try:
            beta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta_new, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    mu = expit(X @ beta)
    w = np.clip(mu * (1 - mu), 1e-8, None)
    XtW = X.T * w
    try:
        H = XtW @ X
        cov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        zvals = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        pvals = 2 * (1 - norm.cdf(np.abs(zvals)))
    except np.linalg.LinAlgError:
        se = np.full_like(beta, np.nan)
        pvals = np.full_like(beta, np.nan)
    return beta, se, pvals


def clip_prob(p: np.ndarray | float) -> np.ndarray | float:
    if isinstance(p, np.ndarray):
        return np.clip(p, CLIP_LO, CLIP_HI)
    return min(CLIP_HI, max(CLIP_LO, float(p)))


def load_llm_lookup() -> dict[tuple[float, float, float], tuple[float, str | None]]:
    """Exact (wait, eff, se) -> (p_llm_mean, state_id)."""
    grid_rows = list(csv.DictReader(open(GRID_PATH, newline="", encoding="utf-8")))
    by_state: dict[str, list[float]] = {}
    parsed = list(csv.DictReader(open(PARSED_LLM, newline="", encoding="utf-8")))
    for row in parsed:
        if str(row.get("parse_success", "")).lower() not in ("true", "1", "yes"):
            continue
        by_state.setdefault(row["state"], []).append(float(row["probability_0_1"]))

    lookup: dict[tuple[float, float, float], tuple[float, str | None]] = {}
    for g in grid_rows:
        st = g["state"]
        if st not in by_state:
            continue
        key = (float(g["wait"]), float(g["eff"]), float(g["se"]))
        lookup[key] = (mean(by_state[st]), st)
    return lookup


def attr_key(wait: float, eff: float, se: float) -> tuple[float, float, float]:
    return (float(wait), float(eff), float(se))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_prob(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    y_sorted = y[order]
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trap is None:
        return float("nan")
    return float(trap(tpr, fpr))


def calibration_params(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logit(y) ~ intercept + slope * logit(p). Returns (intercept, slope)."""
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lp = sp_logit(clip_prob(np.asarray(p, dtype=float)))
    X = np.column_stack([np.ones(len(y)), lp])
    try:
        beta, _, _ = fit_logit_mle(X, y)
        return float(beta[0]), float(beta[1])
    except RuntimeError:
        return float("nan"), float("nan")


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = clip_prob(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)
    intercept, slope = calibration_params(y, p)
    return {
        "N": len(y),
        "log_loss": log_loss(y, p),
        "brier_score": brier_score(y, p),
        "auc": roc_auc(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_pred": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    dce_path = discover_dce_file()
    rows, col_map = load_and_standardize_dce(dce_path)
    include_cash = "cash" in col_map

    train_ids, test_ids = respondent_split(rows)
    train_rows = [r for r in rows if r["respondent_id"] in train_ids]
    test_rows = [r for r in rows if r["respondent_id"] in test_ids]

    print(f"\nRespondent split (seed={SEED}):")
    print(f"  Train respondents: {len(train_ids)} | choice rows: {len(train_rows)}")
    print(f"  Test respondents:  {len(test_ids)} | choice rows: {len(test_rows)}")

    X_train, y_train, coef_names = build_design(train_rows, include_cash)
    beta, se, pvals = fit_logit_mle(X_train, y_train)

    coef_rows = []
    for name, b, s, p in zip(coef_names, beta, se, pvals):
        coef_rows.append(
            {
                "term": name,
                "coef": f"{b:.8f}",
                "std_err": f"{s:.8f}" if not math.isnan(s) else "NA",
                "p_value": f"{p:.8g}" if not math.isnan(p) else "NA",
            }
        )
    write_csv(
        RESULTS_DIR / "heldout_dce_frontier_coefficients.csv",
        ["term", "coef", "std_err", "p_value"],
        coef_rows,
    )
    print("\nTrain-only logit coefficients:")
    for r in coef_rows:
        print(f"  {r['term']:8s}  coef={r['coef']}  se={r['std_err']}  p={r['p_value']}")

    X_test, y_test, _ = build_design(test_rows, include_cash)
    p_dce = clip_prob(expit(X_test @ beta))

    llm_lookup = load_llm_lookup()
    prediction_rows: list[dict] = []
    unmatched_rows: list[dict] = []

    for i, r in enumerate(test_rows):
        key = attr_key(r["wait"], r["eff"], r["se"])
        matched = key in llm_lookup
        p_llm_val = ""
        p_bdt_val = ""
        matched_state = ""
        if matched:
            p_llm, st = llm_lookup[key]
            p_llm_c = float(clip_prob(p_llm))
            p_bdt_c = float(clip_prob(LAMBDA * p_llm_c + (1 - LAMBDA) * float(p_dce[i])))
            p_llm_val = f"{p_llm_c:.8f}"
            p_bdt_val = f"{p_bdt_c:.8f}"
            matched_state = st or ""
        else:
            unmatched_rows.append({**r, "p_dce": f"{float(p_dce[i]):.8f}"})

        prediction_rows.append(
            {
                "respondent_id": r["respondent_id"],
                "y": r["y"],
                "wait": r["wait"],
                "eff": r["eff"],
                "se": r["se"],
                "cash": r["cash"],
                "p_dce": f"{float(p_dce[i]):.8f}",
                "p_llm": p_llm_val,
                "p_bdt": p_bdt_val,
                "matched_llm_state": matched_state,
            }
        )

    n_test = len(test_rows)
    n_matched = sum(1 for r in prediction_rows if r["matched_llm_state"])
    print(f"\nLLM exact-match on held-out test rows: {n_matched}/{n_test} ({100*n_matched/n_test:.1f}%)")
    if n_matched < n_test:
        print(f"  Unmatched rows saved to results/heldout_dce_unmatched_rows.csv ({n_test - n_matched} rows)")

    y_all = np.array([r["y"] for r in test_rows], dtype=float)
    p_dce_all = np.asarray(p_dce, dtype=float)

    matched_idx = [i for i, r in enumerate(prediction_rows) if r["matched_llm_state"]]
    y_m = y_all[matched_idx]
    p_llm_m = np.array([float(prediction_rows[i]["p_llm"]) for i in matched_idx])
    p_bdt_m = np.array([float(prediction_rows[i]["p_bdt"]) for i in matched_idx])

    metrics_rows = []
    for method, p_vec, mask_note in [
        ("Pure-DCE", p_dce_all, "all test rows"),
        ("Unconstrained LLM", p_llm_m, f"matched rows only (n={len(matched_idx)})"),
        ("Static-BDT Anchor", p_bdt_m, f"matched rows only (n={len(matched_idx)})"),
    ]:
        m = compute_metrics(y_all if method == "Pure-DCE" else y_m, p_vec)
        metrics_rows.append(
            {
                "method": method,
                "N": m["N"],
                "log_loss": f"{m['log_loss']:.6f}",
                "brier_score": f"{m['brier_score']:.6f}",
                "auc": f"{m['auc']:.4f}" if not math.isnan(m["auc"]) else "NA",
                "calibration_intercept": f"{m['calibration_intercept']:.4f}"
                if not math.isnan(m["calibration_intercept"])
                else "NA",
                "calibration_slope": f"{m['calibration_slope']:.4f}"
                if not math.isnan(m["calibration_slope"])
                else "NA",
                "mean_pred": f"{m['mean_pred']:.6f}",
                "observed_rate": f"{m['observed_rate']:.6f}",
                "note": mask_note,
            }
        )

    write_csv(
        RESULTS_DIR / "heldout_dce_validation_metrics.csv",
        [
            "method",
            "N",
            "log_loss",
            "brier_score",
            "auc",
            "calibration_intercept",
            "calibration_slope",
            "mean_pred",
            "observed_rate",
            "note",
        ],
        metrics_rows,
    )
    write_csv(
        RESULTS_DIR / "heldout_dce_predictions.csv",
        [
            "respondent_id",
            "y",
            "wait",
            "eff",
            "se",
            "cash",
            "p_dce",
            "p_llm",
            "p_bdt",
            "matched_llm_state",
        ],
        prediction_rows,
    )
    if unmatched_rows:
        write_csv(
            RESULTS_DIR / "heldout_dce_unmatched_rows.csv",
            ["row_id", "respondent_id", "y", "wait", "eff", "se", "cash", "p_dce"],
            unmatched_rows,
        )

    print("\n" + "=" * 72)
    print("Held-out DCE validation metrics")
    print("=" * 72)
    for r in metrics_rows:
        print(
            f"{r['method']:<22} N={r['N']:<5} log_loss={r['log_loss']} brier={r['brier_score']} "
            f"auc={r['auc']} cal_int={r['calibration_intercept']} cal_slope={r['calibration_slope']}"
        )

    print("\nLaTeX table rows:")
    for r in metrics_rows:
        auc = r["auc"] if r["auc"] != "NA" else "--"
        print(
            f"{r['method']} & {r['N']} & {r['log_loss']} & {r['brier_score']} & {auc} & "
            f"{r['calibration_intercept']} & {r['calibration_slope']} \\\\"
        )

    print("\n" + "-" * 72)
    print("Interpretation:")
    print("- This is held-out stated-choice validation on 20% of respondents (seed=2026).")
    print("- Pure-DCE is the train-only logit frontier; it is the strong behavioral benchmark.")
    pure = metrics_rows[0]
    bdt = metrics_rows[2]
    pure_ll = float(pure["log_loss"])
    bdt_ll = float(bdt["log_loss"]) if bdt["N"] != "0" else float("inf")
    if bdt["N"] == "0" or math.isnan(bdt_ll):
        print("- Static-BDT could not be evaluated on matched rows (no exact LLM state match).")
    elif bdt_ll < pure_ll:
        print("- Static-BDT beats Pure-DCE on log loss here → LLM component adds predictive value on matched rows.")
    else:
        print(
            "- Static-BDT does not beat Pure-DCE on log loss here. "
            "That is not a failure: BDT mainly regularizes LLM synthetic agents toward the behavioral benchmark, "
            "rather than replacing the DCE model."
        )
    print("-" * 72)


if __name__ == "__main__":
    main()
