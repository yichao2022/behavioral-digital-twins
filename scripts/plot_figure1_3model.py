"""
Figure 1 (3-model version, v5): 2-panel layout (Spearman rho + MAD).
Reads:
  - Qwen2.5-72B-Instruct OOD from results/out_of_design_parsed_probabilities.csv
  - DeepSeek V3 OOD from results/out_of_design_parsed_probabilities.csv
  - MiroThinker-1.7-mini OOD from c_half_validation/MiroThinker-1.7-mini_n200_n40.csv

The 12 pre-registered OOD scenarios are loaded from out_of_design_scenarios.csv
and the empirical P_static values from results/out_of_design_pstatic.csv.

Run from repo root:
    python3 scripts/plot_figure1_3model.py
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from scipy.stats import spearmanr
import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAMBDA = 0.25
REPO = Path(__file__).resolve().parent.parent
PARSED_REPO = REPO / 'results' / 'out_of_design_parsed_probabilities.csv'
PSTATIC_FILE = REPO / 'results' / 'out_of_design_pstatic.csv'
SCENARIOS_FILE = REPO / 'out_of_design_scenarios.csv'
MIROTHINKER_C_HALF = REPO / 'c_half_validation' / 'MiroThinker-1.7-mini_n200_n40.csv'

# Load P_static (12 pre-registered scenarios A-F)
pstatic = {}
with open(PSTATIC_FILE, newline='') as f:
    for r in csv.DictReader(f):
        pstatic[r['scenario_id']] = float(r['P_static'])

scenarios = {}
with open(SCENARIOS_FILE, newline='') as f:
    for r in csv.DictReader(f):
        if r['scenario_id'] in pstatic:
            scenarios[r['scenario_id']] = r

groups = sorted(set(scenarios[s]['expected_wait_order_group'] for s in scenarios))


def load_repo_llm(model_name):
    """Load P_LLM by scenario from the pre-registered 12-scenario OOD file."""
    by = defaultdict(list)
    with open(PARSED_REPO, newline='') as f:
        for r in csv.DictReader(f):
            if r['model'] != model_name:
                continue
            if r.get('parse_success', '').lower() not in ('true', '1', 'yes'):
                continue
            try:
                by[r['scenario_id']].append(float(r['probability_0_1']))
            except (ValueError, KeyError):
                continue
    return by


def load_mirothinker_c_half():
    """Load MiroThinker P_LLM by scenario from the c-half validation set."""
    by = defaultdict(list)
    with open(MIROTHINKER_C_HALF, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('parse_success', '').lower() not in ('true', '1', 'yes'):
                continue
            try:
                by[r['scenario_id']].append(float(r['probability_0_1']))
            except (ValueError, KeyError):
                continue
    return by


def nds_isotonic(by_scen_raw):
    """PAVA-style isotonic regression fitted on all 12 OOD scenarios.

    The published NDS metric in Figure 1 panel (a) uses a single PAVA
    fit across all 12 pre-registered OOD scenarios rather than a per-group
    fit. With only 2 scenarios per wait-time group, a per-group fit reduces
    to a linear map and is uninformative. The across-all fit captures
    the overall rank agreement between P_LLM and P_static after monotone
    projection.
    """
    sids = sorted([s for s in by_scen_raw if s in pstatic and len(by_scen_raw[s]) > 0])
    if not sids:
        return {}
    X = np.array([np.mean(by_scen_raw[s]) for s in sids])
    y = np.array([pstatic[s] for s in sids])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
    iso.fit(X, y)
    out = {}
    for s in by_scen_raw:
        if s in pstatic and len(by_scen_raw[s]) > 0:
            p_llm_s = float(np.mean(by_scen_raw[s]))
            out[s] = float(np.clip(iso.predict([p_llm_s])[0], 0.0, 1.0))
    return out


def metrics(p_dict):
    sids = sorted([s for s in p_dict if s in pstatic])
    psl = [pstatic[s] for s in sids]
    pll = [p_dict[s] for s in sids]
    if len(pll) >= 3:
        rho = spearmanr(pll, psl)[0]
    else:
        rho = 0.0
    mad = mean(abs(a - b) for a, b in zip(pll, psl))
    return {'rho': float(rho) if rho is not None else 0.0, 'mad': mad}


def compute_full(by_scen_raw):
    p_llm_mean = {s: mean(v) for s, v in by_scen_raw.items() if s in pstatic and v}
    nds = nds_isotonic(by_scen_raw)
    p_bdt = {s: LAMBDA * p_llm_mean[s] + (1 - LAMBDA) * pstatic[s]
             for s in p_llm_mean}
    return {
        'Unconstrained LLM': metrics(p_llm_mean),
        'NDS (isotonic)': metrics(nds),
        'BDT (lambda=0.25)': metrics(p_bdt),
    }


if __name__ == '__main__':
    sources = [
        ('Qwen2.5-72B-Instruct', load_repo_llm('Qwen2.5-72B')),
        ('MiroThinker-1.7-mini', load_mirothinker_c_half()),
        ('DeepSeek V3', load_repo_llm('DeepSeek V3')),
    ]
    results = {name: compute_full(by) for name, by in sources}

    model_order = [n for n, _ in sources]
    x = np.arange(len(model_order))
    w = 0.27

    colors = {
        'Unconstrained LLM': '#d62728',
        'NDS (isotonic)': '#ff7f0e',
        'BDT (lambda=0.25)': '#2ca02c',
    }
    methods = ['Unconstrained LLM', 'NDS (isotonic)', 'BDT (lambda=0.25)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Panel (a): Spearman rho
    ax = axes[0]
    for i, m in enumerate(methods):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, [results[mo][m]['rho'] for mo in model_order], w,
                      color=colors[m], label=m, edgecolor='black', linewidth=0.6)
        for j, b in enumerate(bars):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + (0.02 if h >= 0 else -0.04),
                    f"{h:.2f}", ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({'repo' if m != 'MiroThinker-1.7-mini' else 'c-half'})"
                        for m in model_order], fontsize=8)
    ax.set_ylabel(r"Spearman $\rho$ with $P_{\mathrm{static}}$")
    ax.set_title("(a) Spearman ρ: 12 narrative OOD scenarios", fontsize=10)
    ax.set_ylim(-0.7, 1.05)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Panel (b): MAD
    ax = axes[1]
    for i, m in enumerate(methods):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, [results[mo][m]['mad'] for mo in model_order], w,
                      color=colors[m], label=m, edgecolor='black', linewidth=0.6)
        for j, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({'repo' if m != 'MiroThinker-1.7-mini' else 'c-half'})"
                        for m in model_order], fontsize=8)
    ax.set_ylabel(r"MAD from $P_{\mathrm{static}}$")
    ax.set_title("(b) Mean absolute deviation", fontsize=10)
    ax.set_ylim(0, 0.35)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Figure 1. Out-of-design performance: 3 LLMs × 3 methods",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = REPO / 'figures' / 'figure_1_ood_performance.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    # Mirror to /tmp
    mirror = Path('/tmp/bdt_vih_src/figures/figure_1_ood_performance.png')
    mirror.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(mirror, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    for name, r in results.items():
        print(f"\n{name}:")
        for m, vals in r.items():
            print(f"  {m:20s}  rho={vals['rho']:+.3f}  mad={vals['mad']:.3f}")
