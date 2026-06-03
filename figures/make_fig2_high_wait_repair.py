#!/usr/bin/env python3
"""Figure 2: High-wait region repair (Qwen2.5-72B local, static benchmark)."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).resolve().parents[1]
GRID = WORKSPACE / "bdt_eval_grid_static.csv"
PARSED = WORKSPACE / "llm_parsed_outputs_qwen72b_unconstrained.csv"
ANCHOR = WORKSPACE / "static_bdt_anchor_qwen72b.csv"
OUT_DIR = WORKSPACE / "figures"
OUT_PNG = OUT_DIR / "fig2_high_wait_repair.png"
OUT_PDF = OUT_DIR / "fig2_high_wait_repair.pdf"
OUT_DATA = OUT_DIR / "fig2_high_wait_repair_data.csv"

LAMBDA = 0.25
PANELS = [
    {"panel": "A", "eff": 0.7, "se": 1.0, "title": r"eff = 0.7, side effect = 1"},
    {"panel": "B", "eff": 0.9, "se": 2.0, "title": r"eff = 0.9, side effect = 2"},
]
WAITS = [0, 2, 4, 6]


def _find_file(candidates: list[Path], patterns: list[str]) -> Path:
    for p in candidates:
        if p.is_file():
            return p
    for p in WORKSPACE.glob("*.csv"):
        name = p.name.lower()
        if all(x in name for x in patterns):
            return p
    raise FileNotFoundError(f"No file matching {patterns}")


def load_merged() -> list[dict]:
    grid_path = GRID if GRID.is_file() else _find_file([GRID], ["bdt", "static"])
    parsed_path = (
        PARSED
        if PARSED.is_file()
        else _find_file([PARSED, WORKSPACE / "llm_parsed_outputs_qwen72b_unconstrained_partial.csv"], ["qwen", "parsed"])
    )
    anchor_path = (
        ANCHOR
        if ANCHOR.is_file()
        else _find_file([ANCHOR], ["qwen", "static_bdt_anchor"])
    )

    grid = {r["state"]: r for r in csv.DictReader(open(grid_path, encoding="utf-8"))}
    anchor = {r["state"]: r for r in csv.DictReader(open(anchor_path, encoding="utf-8"))}

    llm_vals: dict[str, list[float]] = defaultdict(list)
    for r in csv.DictReader(open(parsed_path, encoding="utf-8")):
        if str(r.get("parse_success", "")).lower() not in ("true", "1", "yes"):
            continue
        try:
            llm_vals[r["state"]].append(float(r["probability_0_1"]))
        except (TypeError, ValueError):
            pass

    rows: list[dict] = []
    for state, g in grid.items():
        if state not in llm_vals:
            continue
        a = anchor.get(state, {})
        lam_col = f"pi_BDT_lam{LAMBDA:.2f}"
        pi = a.get(lam_col) or a.get("pi_BDT_mseopt") or a.get("p_bdt_anchor")
        if pi is None or pi == "":
            p_llm = sum(llm_vals[state]) / len(llm_vals[state])
            pi = LAMBDA * p_llm + (1 - LAMBDA) * float(g["P_static"])
        rows.append(
            {
                "state": state,
                "wait": int(float(g["wait"])),
                "eff": float(g["eff"]),
                "se": float(g["se"]),
                "P_static": float(g["P_static"]),
                "p_llm_mean": sum(llm_vals[state]) / len(llm_vals[state]),
                "pi_BDT": float(pi),
            }
        )
    return rows


def filter_slice(rows: list[dict], eff: float, se: float) -> list[dict]:
    sub = [r for r in rows if abs(r["eff"] - eff) < 1e-9 and abs(r["se"] - se) < 1e-9]
    return sorted(sub, key=lambda r: r["wait"])


def wait_monotonicity_violation(series: list[dict], key: str) -> bool:
    vals = [r[key] for r in series]
    return any(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def plot_figure(rows: list[dict]) -> list[dict]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    fig.suptitle("High-wait region repair", fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.96,
        f"Qwen2.5-72B (local), Static-BDT Anchor λ = {LAMBDA:.2f}",
        ha="center",
        fontsize=10,
        color="#444444",
    )

    plot_rows: list[dict] = []
    line_styles = [
        ("P_static", "Empirical frontier ($P_{static}$)", "#333333", "--", "o"),
        ("p_llm_mean", "Unconstrained LLM", "#C0392B", "-", "s"),
        ("pi_BDT", f"Static-BDT Anchor ($\\lambda$={LAMBDA:.2f})", "#2471A3", "-", "^"),
    ]

    for ax, spec in zip(axes, PANELS):
        sub = filter_slice(rows, spec["eff"], spec["se"])
        if len(sub) != 4:
            print(f"WARNING: panel {spec['panel']} has {len(sub)} points (expected 4)", file=sys.stderr)

        for r in sub:
            plot_rows.append(
                {
                    "panel": spec["panel"],
                    "eff": spec["eff"],
                    "se": spec["se"],
                    "wait": r["wait"],
                    "P_static": r["P_static"],
                    "p_llm_mean": r["p_llm_mean"],
                    "pi_BDT": r["pi_BDT"],
                    "state": r["state"],
                }
            )

        xs = [r["wait"] for r in sub]
        for key, label, color, ls, marker in line_styles:
            ys = [r[key] for r in sub]
            ax.plot(xs, ys, linestyle=ls, color=color, marker=marker, linewidth=2, markersize=7, label=label)

        ax.set_title(spec["title"], fontsize=11)
        ax.set_xlabel("Wait time (months)")
        ax.set_xticks(WAITS)
        ax.set_xlim(-0.3, 6.3)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Acceptance probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    return plot_rows


def write_data_csv(plot_rows: list[dict]) -> None:
    fields = ["panel", "eff", "se", "wait", "state", "P_static", "p_llm_mean", "pi_BDT"]
    with open(OUT_DATA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(plot_rows)


def print_report(rows: list[dict], plot_rows: list[dict]) -> None:
    print("=" * 72)
    print("Figure 2 — High-wait region repair (Qwen2.5-72B local)")
    print("=" * 72)
    print(f"Inputs: {GRID.name}, {PARSED.name}, {ANCHOR.name}")
    print(f"λ = {LAMBDA}")
    print(f"Outputs: {OUT_PNG.name}, {OUT_PDF.name}, {OUT_DATA.name}")
    print()

    for spec in PANELS:
        sub = filter_slice(rows, spec["eff"], spec["se"])
        print(f"Panel {spec['panel']}: eff = {spec['eff']}, se = {spec['se']}")
        print(f"  {'wait':>4}  {'P_static':>10}  {'p_llm_mean':>12}  {'pi_BDT':>10}")
        for r in sub:
            print(
                f"  {r['wait']:4d}  {r['P_static']:10.4f}  {r['p_llm_mean']:12.4f}  {r['pi_BDT']:10.4f}"
            )
        llm_v = wait_monotonicity_violation(sub, "p_llm_mean")
        bdt_v = wait_monotonicity_violation(sub, "pi_BDT")
        st_v = wait_monotonicity_violation(sub, "P_static")
        print(f"  P_static monotonicity violation:        {st_v}")
        print(f"  Unconstrained LLM violation:            {llm_v}")
        print(f"  Static-BDT Anchor violation:            {bdt_v}")
        print()

    print(f"Plotted rows: {len(plot_rows)}")


def main() -> None:
    rows = load_merged()
    if len(rows) < 64:
        print(f"WARNING: only {len(rows)} states merged (expected 64)", file=sys.stderr)
    plot_rows = plot_figure(rows)
    write_data_csv(plot_rows)
    print_report(rows, plot_rows)
    print("Done.")


if __name__ == "__main__":
    main()
