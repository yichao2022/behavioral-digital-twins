#!/usr/bin/env python3
"""Compute policy ATE and PRC relative to the Pure-DCE reference."""

from __future__ import annotations

import csv
import json
import math
import os
from itertools import combinations
from pathlib import Path
from statistics import mean


WORKSPACE = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE / "outputs" / "prc_analysis"

GRID_PATH = WORKSPACE / "bdt_eval_grid_static.csv"
MODEL_SPECS = [
    {
        "model": "Qwen2.5-72B",
        "parsed": WORKSPACE / "llm_parsed_outputs_qwen72b_unconstrained.csv",
        "anchor": WORKSPACE / "static_bdt_anchor_qwen72b.csv",
    },
    {
        "model": "DeepSeek V4 Pro",
        "parsed": WORKSPACE / "llm_parsed_outputs_deepseek_unconstrained.csv",
        "anchor": WORKSPACE / "static_bdt_anchor_deepseek.csv",
    },
    {
        "model": "MiroThinker-1.7-mini",
        "parsed": WORKSPACE / "llm_parsed_outputs_mirothinker_unconstrained.csv",
        "anchor": WORKSPACE / "static_bdt_anchor_mirothinker.csv",
    },
]

WAIT_ALIASES = ["wait", "wait_time", "waiting_time", "waittime", "wait time", "w"]
EFF_ALIASES = [
    "eff",
    "efficacy",
    "vaccine efficacy",
    "vaccine_efficacy",
    "vaccineefficacy",
    "e",
]
SE_ALIASES = [
    "side",
    "side_effects",
    "sideeffects",
    "side effects",
    "s",
    "se",
]
PSTATIC_ALIASES = [
    "p_static",
    "static_prob",
    "dce_prob",
    "frontier_prob",
    "empirical_frontier",
    "p_frontier",
]
LLM_ALIASES = ["pi_llm", "llm_prob", "raw_prob", "probability", "prob", "p_llm", "p_llm_mean"]
BDT_ALIASES = ["pi_bdt", "bdt_prob", "anchored_prob", "static_bdt", "pi_bdt_lam0.25"]

POLICIES = [
    {
        "policy_id": "p1",
        "policy_label": "WaitTime 6 -> 2",
        "target_attribute": "WaitTime",
        "from_value": 6.0,
        "to_value": 2.0,
    },
    {
        "policy_id": "p2",
        "policy_label": "WaitTime 4 -> 0",
        "target_attribute": "WaitTime",
        "from_value": 4.0,
        "to_value": 0.0,
    },
    {
        "policy_id": "p3",
        "policy_label": "Efficacy 0.50 -> 0.70",
        "target_attribute": "Efficacy",
        "from_value": 0.50,
        "to_value": 0.70,
    },
    {
        "policy_id": "p4",
        "policy_label": "Efficacy 0.70 -> 0.90",
        "target_attribute": "Efficacy",
        "from_value": 0.70,
        "to_value": 0.90,
    },
    {
        "policy_id": "p5",
        "policy_label": "SideEffects 3 -> 1",
        "target_attribute": "SideEffects",
        "from_value": 3.0,
        "to_value": 1.0,
    },
    {
        "policy_id": "p6",
        "policy_label": "SideEffects 2 -> 0",
        "target_attribute": "SideEffects",
        "from_value": 2.0,
        "to_value": 0.0,
    },
]

EPSILON = 0.005
EXPECTED_WAIT = {0.0, 2.0, 4.0, 6.0}
EXPECTED_EFF = {0.30, 0.50, 0.70, 0.90}
EXPECTED_SE = {0.0, 1.0, 2.0, 3.0}


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_alias(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_norm_name(c): c for c in columns}
    for alias in aliases:
        match = normalized.get(_norm_name(alias))
        if match:
            return match
    return None


def _to_probability(x: str | float | int | None) -> float:
    if x is None:
        raise ValueError("missing probability")
    val = float(x)
    if 0.0 <= val <= 1.0:
        return val
    if 0.0 <= val <= 100.0:
        return val / 100.0
    raise ValueError(f"probability out of range: {val}")


def _state_key(wait: float, eff: float, se: float) -> tuple[float, float, float]:
    return (round(wait, 6), round(eff, 6), round(se, 6))


def inspect_csv(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    columns = rows[0].keys() if rows else []
    columns_list = list(columns)
    wait_col = _find_alias(columns_list, WAIT_ALIASES)
    eff_col = _find_alias(columns_list, EFF_ALIASES)
    se_col = _find_alias(columns_list, SE_ALIASES)
    p_static_col = _find_alias(columns_list, PSTATIC_ALIASES)
    llm_col = _find_alias(columns_list, LLM_ALIASES)
    bdt_col = _find_alias(columns_list, BDT_ALIASES)
    return {
        "path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "columns": columns_list,
        "row_count": len(rows),
        "first_5_rows": rows[:5],
        "contains_wait_eff_se": bool(wait_col and eff_col and se_col),
        "contains_p_static": p_static_col is not None,
        "contains_llm_probability": llm_col is not None,
        "contains_bdt_probability": bdt_col is not None,
        "column_mapping_guess": {
            "WaitTime": wait_col,
            "Efficacy": eff_col,
            "SideEffects": se_col,
            "P_static": p_static_col,
            "pi_llm": llm_col,
            "pi_bdt": bdt_col,
        },
    }


def load_grid() -> tuple[dict[str, dict], dict[tuple[float, float, float], str], dict]:
    rows = list(csv.DictReader(open(GRID_PATH, newline="", encoding="utf-8")))
    grid_by_state: dict[str, dict] = {}
    state_lookup: dict[tuple[float, float, float], str] = {}
    for row in rows:
        wait = float(row["wait"])
        eff = float(row["eff"])
        se = float(row["se"])
        state = row["state"]
        grid_by_state[state] = {
            "state": state,
            "WaitTime": wait,
            "Efficacy": eff,
            "SideEffects": se,
            "P_static": _to_probability(row["P_static"]),
        }
        state_lookup[_state_key(wait, eff, se)] = state

    missing_states = [
        _state_key(w, e, s)
        for w in sorted(EXPECTED_WAIT)
        for e in sorted(EXPECTED_EFF)
        for s in sorted(EXPECTED_SE)
        if _state_key(w, e, s) not in state_lookup
    ]
    completeness = {
        "grid_rows": len(rows),
        "expected_states": 64,
        "missing_states": missing_states,
    }
    return grid_by_state, state_lookup, completeness


def load_p_llm_mean(parsed_path: Path) -> tuple[dict[str, float], dict]:
    rows = list(csv.DictReader(open(parsed_path, newline="", encoding="utf-8")))
    columns = list(rows[0].keys()) if rows else []
    state_col = _find_alias(columns, ["state"])
    prob_col = _find_alias(columns, ["probability_0_1", "probability", "prob", "p_llm"])
    parse_col = _find_alias(columns, ["parse_success"])
    if not (state_col and prob_col and parse_col):
        raise ValueError(f"Required parsed columns missing in {parsed_path}")

    by_state: dict[str, list[float]] = {}
    parse_ok = 0
    for row in rows:
        if str(row[parse_col]).lower() not in ("true", "1", "yes"):
            continue
        parse_ok += 1
        by_state.setdefault(row[state_col], []).append(_to_probability(row[prob_col]))

    p_llm_mean = {state: mean(vals) for state, vals in by_state.items() if vals}
    parse_info = {
        "rows_total": len(rows),
        "rows_parse_success": parse_ok,
        "states_with_means": len(p_llm_mean),
        "probability_column": prob_col,
        "parse_column": parse_col,
        "probability_scale_conversion": "none (already 0-1)" if "0_1" in prob_col else "auto-detected",
    }
    return p_llm_mean, parse_info


def validate_existing_anchor(anchor_path: Path, grid_by_state: dict[str, dict], p_llm_mean: dict[str, float]) -> dict:
    rows = list(csv.DictReader(open(anchor_path, newline="", encoding="utf-8")))
    columns = list(rows[0].keys()) if rows else []
    bdt_col = (
        _find_alias(columns, ["pi_BDT_lam0.25"])
        or _find_alias(columns, ["pi_BDT"])
        or _find_alias(columns, BDT_ALIASES)
    )
    if not bdt_col:
        return {
            "path": str(anchor_path),
            "matched_formula": False,
            "checked_column": None,
            "max_abs_diff": None,
            "note": "no pi_bdt-like column found",
        }

    diffs = []
    for row in rows:
        state = row["state"]
        if state not in grid_by_state or state not in p_llm_mean:
            continue
        expected = 0.25 * p_llm_mean[state] + 0.75 * grid_by_state[state]["P_static"]
        observed = _to_probability(row[bdt_col])
        diffs.append(abs(observed - expected))

    max_diff = max(diffs) if diffs else None
    matched = max_diff is not None and max_diff <= 1e-8
    return {
        "path": str(anchor_path),
        "matched_formula": matched,
        "checked_column": bdt_col,
        "max_abs_diff": max_diff,
        "note": "used for validation only",
    }


def build_state_dataframe() -> tuple[list[dict], dict]:
    grid_by_state, state_lookup, grid_info = load_grid()
    all_rows: list[dict] = []
    completeness: dict[str, dict] = {"grid": grid_info, "models": {}}
    anchor_validation: dict[str, dict] = {}
    file_inspections: list[dict] = [inspect_csv(GRID_PATH)]

    for spec in MODEL_SPECS:
        model = spec["model"]
        parsed_path = spec["parsed"]
        anchor_path = spec["anchor"]
        file_inspections.append(inspect_csv(parsed_path))
        file_inspections.append(inspect_csv(anchor_path))

        p_llm_mean, parse_info = load_p_llm_mean(parsed_path)
        missing_states = sorted(set(grid_by_state) - set(p_llm_mean), key=int)
        anchor_validation[model] = validate_existing_anchor(anchor_path, grid_by_state, p_llm_mean)

        completeness["models"][model] = {
            "parse_info": parse_info,
            "missing_states": missing_states,
            "complete_64_states": len(p_llm_mean) == 64 and not missing_states,
        }

        for state, grid_row in sorted(grid_by_state.items(), key=lambda kv: int(kv[0])):
            if state not in p_llm_mean:
                continue
            p_static = grid_row["P_static"]
            pi_llm = p_llm_mean[state]
            pi_bdt = 0.25 * pi_llm + 0.75 * p_static
            all_rows.append(
                {
                    "model": model,
                    "state": state,
                    "WaitTime": grid_row["WaitTime"],
                    "Efficacy": grid_row["Efficacy"],
                    "SideEffects": grid_row["SideEffects"],
                    "P_static": p_static,
                    "pi_llm": pi_llm,
                    "pi_bdt": pi_bdt,
                }
            )

    return all_rows, {
        "completeness": completeness,
        "anchor_validation": anchor_validation,
        "file_inspections": file_inspections,
        "drive_search": "No synced Google Drive / CloudStorage folder with stronger candidates was found outside the workspace.",
        "state_lookup": {str(k): v for k, v in state_lookup.items()},
    }


def save_clean_dataframe(rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "clean_analysis_dataframe.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "model",
            "state",
            "WaitTime",
            "Efficacy",
            "SideEffects",
            "P_static",
            "pi_llm",
            "pi_bdt",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def method_probability(row: dict, method: str) -> float:
    if method == "Pure-DCE":
        return row["P_static"]
    if method == "Unconstrained LLM":
        return row["pi_llm"]
    if method == "Static-BDT Anchor":
        return row["pi_bdt"]
    raise ValueError(f"Unknown method: {method}")


def compute_policy_ates(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    ate_rows: list[dict] = []
    for model, model_rows in by_model.items():
        state_map = {
            _state_key(r["WaitTime"], r["Efficacy"], r["SideEffects"]): r
            for r in model_rows
        }
        for method in ("Pure-DCE", "Unconstrained LLM", "Static-BDT Anchor"):
            for policy in POLICIES:
                deltas = []
                eligible = 0
                for row in model_rows:
                    wait = row["WaitTime"]
                    eff = row["Efficacy"]
                    se = row["SideEffects"]
                    if policy["target_attribute"] == "WaitTime" and wait != policy["from_value"]:
                        continue
                    if policy["target_attribute"] == "Efficacy" and eff != policy["from_value"]:
                        continue
                    if policy["target_attribute"] == "SideEffects" and se != policy["from_value"]:
                        continue

                    cf_wait, cf_eff, cf_se = wait, eff, se
                    if policy["target_attribute"] == "WaitTime":
                        cf_wait = policy["to_value"]
                    elif policy["target_attribute"] == "Efficacy":
                        cf_eff = policy["to_value"]
                    else:
                        cf_se = policy["to_value"]

                    cf_row = state_map.get(_state_key(cf_wait, cf_eff, cf_se))
                    if not cf_row:
                        continue
                    eligible += 1
                    deltas.append(method_probability(cf_row, method) - method_probability(row, method))

                ate = mean(deltas) if deltas else float("nan")
                ate_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "policy_id": policy["policy_id"],
                        "policy_label": policy["policy_label"],
                        "target_attribute": policy["target_attribute"],
                        "from_value": policy["from_value"],
                        "to_value": policy["to_value"],
                        "eligible_states": eligible,
                        "ATE": f"{ate:.8f}",
                    }
                )
    return ate_rows


def sign_epsilon(x: float, eps: float = EPSILON) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def compute_prc(ate_rows: list[dict]) -> list[dict]:
    by_model_method: dict[tuple[str, str], dict[str, float]] = {}
    for row in ate_rows:
        by_model_method.setdefault((row["model"], row["method"]), {})[row["policy_id"]] = float(row["ATE"])

    prc_rows: list[dict] = []
    policy_pairs = list(combinations([p["policy_id"] for p in POLICIES], 2))
    for model in [spec["model"] for spec in MODEL_SPECS]:
        pure = by_model_method[(model, "Pure-DCE")]
        for method in ("Unconstrained LLM", "Static-BDT Anchor"):
            alt = by_model_method[(model, method)]
            matching = 0
            for p, q in policy_pairs:
                s_ref = sign_epsilon(pure[p] - pure[q])
                s_alt = sign_epsilon(alt[p] - alt[q])
                if s_ref == s_alt:
                    matching += 1
            prc = matching / len(policy_pairs)
            prc_rows.append(
                {
                    "model": model,
                    "method": method,
                    "num_policy_pairs": len(policy_pairs),
                    "num_matching_pairs": matching,
                    "PRC_vs_PureDCE": f"{prc:.4f}",
                }
            )
    return prc_rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(prc_rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "prc_table.tex"
    model_order = [spec["model"] for spec in MODEL_SPECS]
    method_order = {"Unconstrained LLM": 0, "Static-BDT Anchor": 1}
    ordered = sorted(
        prc_rows,
        key=lambda r: (model_order.index(r["model"]), method_order[r["method"]]),
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Policy Ranking Consistency relative to the Pure-DCE reference.}\n")
        f.write("\\label{tab:prc}\n")
        f.write("\\begin{tabular}{l l c c c}\n")
        f.write("\\toprule\n")
        f.write("Model & Method & Policy pairs & Matching pairs & PRC vs. Pure-DCE \\\\\n")
        f.write("\\midrule\n")
        for row in ordered:
            f.write(
                f"{row['model']} & {row['method']} & {row['num_policy_pairs']} & "
                f"{row['num_matching_pairs']} & {float(row['PRC_vs_PureDCE']):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{0.4em}\n")
        f.write("\\begin{minipage}{0.90\\textwidth}\n")
        f.write("\\small\n")
        f.write(
            "\\textbf{Note.} PRC measures the share of policy pairs ranked in the same order as the Pure-DCE simulator. "
            "The Pure-DCE simulator uses \\(P_{\\mathrm{static}}\\) directly and serves as the ranking reference. "
            "Pairwise rankings use a tolerance of \\(\\epsilon=0.005\\).\n"
        )
        f.write("\\end{minipage}\n")
        f.write("\\end{table}\n")
    return path


def write_audit_report(metadata: dict, used_files: list[Path], output_files: list[Path]) -> Path:
    path = OUTPUT_DIR / "prc_audit_report.md"
    completeness = metadata["completeness"]
    anchor_validation = metadata["anchor_validation"]
    file_inspections = metadata["file_inspections"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# PRC Audit Report\n\n")
        f.write("## Files Used\n")
        for file_path in used_files:
            f.write(f"- `{file_path}`\n")
        f.write("\n")

        f.write("## External Search\n")
        f.write(f"- {metadata['drive_search']}\n\n")

        f.write("## Candidate File Inspection\n")
        for info in file_inspections:
            f.write(f"### `{info['path']}`\n")
            f.write(f"- File type: `{info['file_type']}`\n")
            f.write(f"- Rows: {info['row_count']}\n")
            f.write(f"- Columns: `{', '.join(info['columns'])}`\n")
            f.write(f"- Contains WaitTime / Efficacy / SideEffects: `{info['contains_wait_eff_se']}`\n")
            f.write(f"- Contains P_static or equivalent: `{info['contains_p_static']}`\n")
            f.write(f"- Contains raw LLM probability: `{info['contains_llm_probability']}`\n")
            f.write(f"- Contains BDT probability: `{info['contains_bdt_probability']}`\n")
            f.write(f"- Detected column mappings: `{json.dumps(info['column_mapping_guess'], ensure_ascii=False)}`\n")
            f.write("- First 5 rows:\n\n")
            f.write("```json\n")
            f.write(json.dumps(info["first_5_rows"], indent=2, ensure_ascii=False))
            f.write("\n```\n\n")

        f.write("## Detected Column Mappings Used in Analysis\n")
        f.write("- Grid: `wait -> WaitTime`, `eff -> Efficacy`, `se -> SideEffects`, `P_static -> P_static`\n")
        for model, model_info in completeness["models"].items():
            parse_info = model_info["parse_info"]
            f.write(
                f"- {model}: `state -> state`, `{parse_info['probability_column']} -> pi_llm`, "
                f"`{parse_info['parse_column']} -> parse_success`\n"
            )
        f.write("\n")

        f.write("## Probability Scale Conversions\n")
        f.write("- `P_static` in `bdt_eval_grid_static.csv` was already in 0-1 scale.\n")
        f.write("- Parsed LLM probabilities used `probability_0_1`, so no 0-100 conversion was needed.\n")
        f.write("- Existing anchor files were inspected in 0-1 scale for validation only.\n\n")

        f.write("## Completeness Checks\n")
        f.write(
            f"- Fixed grid rows: {completeness['grid']['grid_rows']} (expected 64 states); "
            f"missing grid states: {completeness['grid']['missing_states']}\n"
        )
        for model, model_info in completeness["models"].items():
            parse_info = model_info["parse_info"]
            f.write(
                f"- {model}: parse-success rows = {parse_info['rows_parse_success']} / {parse_info['rows_total']}; "
                f"states with `pi_llm` means = {parse_info['states_with_means']} / 64; "
                f"complete = {model_info['complete_64_states']}; "
                f"missing states = {model_info['missing_states']}\n"
            )
        f.write("\n")

        f.write("## Existing `pi_bdt` Validation\n")
        for model, info in anchor_validation.items():
            max_diff = "None" if info["max_abs_diff"] is None else f"{info['max_abs_diff']:.10f}"
            f.write(
                f"- {model}: checked `{info['checked_column']}` in `{info['path']}`; "
                f"matches `0.25*pi_llm + 0.75*P_static` = `{info['matched_formula']}`; "
                f"max abs diff = `{max_diff}`.\n"
            )
        f.write("- DeepSeek's existing anchor file did not represent the fixed `lambda=0.25` formula, so `pi_bdt` was recomputed from `pi_llm` and `P_static`.\n\n")

        f.write("## Outputs\n")
        for file_path in output_files:
            f.write(f"- `{file_path}`\n")
        f.write("\n")

    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state_rows, metadata = build_state_dataframe()
    missing_any = any(
        not info["complete_64_states"]
        for info in metadata["completeness"]["models"].values()
    ) or bool(metadata["completeness"]["grid"]["missing_states"])
    if missing_any:
        raise SystemExit(
            "Data insufficient for PRC analysis; see completeness checks in prc_audit_report.md after rerun."
        )

    clean_df_path = save_clean_dataframe(state_rows)
    ate_rows = compute_policy_ates(state_rows)
    prc_rows = compute_prc(ate_rows)

    ate_path = OUTPUT_DIR / "policy_ate_results.csv"
    prc_path = OUTPUT_DIR / "policy_ranking_prc.csv"
    write_csv(
        ate_path,
        [
            "model",
            "method",
            "policy_id",
            "policy_label",
            "target_attribute",
            "from_value",
            "to_value",
            "eligible_states",
            "ATE",
        ],
        ate_rows,
    )
    write_csv(
        prc_path,
        ["model", "method", "num_policy_pairs", "num_matching_pairs", "PRC_vs_PureDCE"],
        prc_rows,
    )
    tex_path = write_tex(prc_rows)
    audit_path = write_audit_report(
        metadata,
        [GRID_PATH] + [spec["parsed"] for spec in MODEL_SPECS] + [spec["anchor"] for spec in MODEL_SPECS],
        [clean_df_path, ate_path, prc_path, tex_path],
    )

    print(f"Wrote {clean_df_path}")
    print(f"Wrote {ate_path}")
    print(f"Wrote {prc_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {audit_path}\n")

    print("Main-text LaTeX table rows:")
    model_order = [spec["model"] for spec in MODEL_SPECS]
    method_order = {"Unconstrained LLM": 0, "Static-BDT Anchor": 1}
    for row in sorted(prc_rows, key=lambda r: (model_order.index(r["model"]), method_order[r["method"]])):
        print(
            f"{row['model']} & {row['method']} & {row['num_policy_pairs']} & "
            f"{row['num_matching_pairs']} & {float(row['PRC_vs_PureDCE']):.3f} \\\\"
        )


if __name__ == "__main__":
    main()
