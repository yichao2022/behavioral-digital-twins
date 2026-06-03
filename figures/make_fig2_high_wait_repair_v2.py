#!/usr/bin/env python3
"""Figure 2 v2 — paper style, from fig2_high_wait_repair_data.csv only."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent
DATA = FIG_DIR / "fig2_high_wait_repair_data.csv"
OUT_PNG = FIG_DIR / "fig2_high_wait_repair_v2.png"
OUT_PDF = FIG_DIR / "fig2_high_wait_repair_v2.pdf"

SUGGESTED_CAPTION = (
    "Figure 2. Repairing wait-time monotonicity violations. The figure plots "
    "acceptance probabilities across waiting-time states for two fixed "
    "efficacy–side-effect slices using the local Qwen2.5-72B model. The "
    "unconstrained LLM produces weakly monotone or non-monotone wait-time "
    "responses, especially in high-wait regions. Static-BDT anchoring with "
    "λ = 0.25 pulls generated probabilities toward the mixed-logit empirical "
    "frontier and restores wait-time consistency. The benchmark is the static "
    "mixed-logit frontier."
)

PANEL_TITLES = {
    "A": "Efficacy = 0.70, side-effect level = 1",
    "B": "Efficacy = 0.90, side-effect level = 2",
}

SERIES = [
    ("P_static", "Empirical frontier", "#222222", "--", "o"),
    ("p_llm_mean", "Unconstrained LLM", "#B03A2E", "-", "s"),
    ("pi_BDT", "Static-BDT Anchor", "#1F618D", "-", "^"),
]


def load_data(path: Path) -> dict[str, list[dict]]:
    if not path.is_file():
        sys.exit(f"Missing data file: {path}")
    by_panel: dict[str, list[dict]] = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        panel = row["panel"]
        by_panel.setdefault(panel, []).append(
            {
                "panel": panel,
                "eff": float(row["eff"]),
                "se": float(row["se"]),
                "wait": int(float(row["wait"])),
                "P_static": float(row["P_static"]),
                "p_llm_mean": float(row["p_llm_mean"]),
                "pi_BDT": float(row["pi_BDT"]),
            }
        )
    for panel in by_panel:
        by_panel[panel] = sorted(by_panel[panel], key=lambda r: r["wait"])
    return by_panel


def monotonicity_violation(rows: list[dict], key: str) -> bool:
    vals = [r[key] for r in rows]
    return any(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def plot(by_panel: dict[str, list[dict]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.28, left=0.09, right=0.98)

    fig.suptitle(
        "Repairing wait-time monotonicity violations",
        fontsize=11,
        fontweight="normal",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Qwen2.5-72B local, Static-BDT Anchor, λ = 0.25",
        ha="center",
        fontsize=9,
        color="#333333",
    )

    for ax, panel in zip(axes, ["A", "B"]):
        rows = by_panel[panel]
        xs = [r["wait"] for r in rows]
        for key, label, color, ls, marker in SERIES:
            ys = [r[key] for r in rows]
            ax.plot(
                xs,
                ys,
                linestyle=ls,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=5.5,
                markerfacecolor="white" if ls == "--" else color,
                markeredgewidth=1.2,
                label=label,
            )

        ax.set_title(PANEL_TITLES[panel], fontsize=10, pad=8)
        ax.set_xlabel("Wait time (months)")
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xlim(-0.25, 6.25)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45, color="#888888")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Acceptance probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.4,
        handletextpad=0.5,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def print_report(by_panel: dict[str, list[dict]]) -> None:
    print("=" * 76)
    print("Figure 2 v2 — paper version")
    print("=" * 76)
    print(f"Data: {DATA.name}")
    print(f"Outputs: {OUT_PNG.name}, {OUT_PDF.name}")
    print()
    print("Suggested caption:")
    print(SUGGESTED_CAPTION)
    print()

    for panel in ["A", "B"]:
        rows = by_panel[panel]
        eff, se = rows[0]["eff"], rows[0]["se"]
        print(f"Panel {panel}: efficacy = {eff:.2f}, side-effect level = {se:g}")
        print(f"  {'wait':>4}  {'Empirical':>12}  {'Unconstrained':>14}  {'Static-BDT':>12}")
        for r in rows:
            print(
                f"  {r['wait']:4d}  {r['P_static']:12.4f}  {r['p_llm_mean']:14.4f}  {r['pi_BDT']:12.4f}"
            )
        llm_v = monotonicity_violation(rows, "p_llm_mean")
        bdt_v = monotonicity_violation(rows, "pi_BDT")
        print(f"  Unconstrained LLM wait monotonicity violation: {llm_v}")
        print(f"  Static-BDT Anchor wait monotonicity violation:   {bdt_v}")
        print()


def main() -> None:
    by_panel = load_data(DATA)
    plot(by_panel)
    print_report(by_panel)
    print("Done.")


if __name__ == "__main__":
    main()
