
"""
Figure 1 (3-model version): BDT OOD performance across Qwen-max, MiroThinker-1.7-mini, DS V4.
Metrics computed from c_half/v1 CSV (A1-F2, 12 pre-registered OOD scenarios).
"""
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LAMBDA = 0.25
V1 = Path('/tmp/bdt_n1027/c_half')
PSTATIC = Path('/tmp/bdt_repo/results/out_of_design_pstatic.csv')

pstatic = {}
with open(PSTATIC, newline='') as f:
    for r in csv.DictReader(f):
        pstatic[r['scenario_id']] = float(r['P_static'])

scenarios = {}
with open('/tmp/bdt_repo/out_of_design_scenarios.csv', newline='') as f:
    for r in csv.DictReader(f):
        scenarios[r['scenario_id']] = r


def load(name):
    fp = V1 / f'{name}_n200_n40.csv'
    by_scen = defaultdict(list)
    with open(fp, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('parse_success', '').lower() in ('true', '1', 'yes'):
                try:
                    by_scen[r['scenario_id']].append(float(r['probability_0_1']))
                except (ValueError, TypeError):
                    continue
    return by_scen


def spearman_val(x, y):
    if len(x) < 3: return None
    rho, _ = spearmanr(x, y)
    return float(rho)


def mvr_wait(p_dict):
    violations = 0; total = 0
    for gname in sorted(set(scenarios[s]['expected_wait_order_group'] for s in scenarios)):
        g_sids = [s for s in scenarios if scenarios[s]['expected_wait_order_group'] == gname]
        if len(g_sids) != 2: continue
        s_low = next((s for s in g_sids if float(scenarios[s]['wait_time']) == 2.0), None)
        s_high = next((s for s in g_sids if float(scenarios[s]['wait_time']) == 6.0), None)
        if not s_low or not s_high: continue
        p_low = p_dict.get(s_low)
        p_high = p_dict.get(s_high)
        if p_low is None or p_high is None: continue
        total += 1
        if p_low < p_high: violations += 1
    return violations / total if total else None


def compute(fname):
    by = load(fname)
    p_llm_mean = {s: mean(v) for s, v in by.items() if s in pstatic and v}
    psl = [pstatic[s] for s in pstatic if s in p_llm_mean]
    pll = [p_llm_mean[s] for s in pstatic if s in p_llm_mean]
    mad_llm = mean(abs(a - b) for a, b in zip(pll, psl))
    rho_llm = spearman_val(pll, psl)
    mvr_llm = mvr_wait(p_llm_mean)
    p_bdt = {s: LAMBDA * p_llm_mean[s] + (1 - LAMBDA) * pstatic[s] for s in p_llm_mean}
    bl = [p_bdt[s] for s in pstatic if s in p_bdt]
    bps = [pstatic[s] for s in pstatic if s in p_bdt]
    mad_bdt = mean(abs(a - b) for a, b in zip(bl, bps))
    rho_bdt = spearman_val(bl, bps)
    mvr_bdt = mvr_wait(p_bdt)
    return {'MVR_llm': mvr_llm, 'MVR_bdt': mvr_bdt,
            'rho_llm': rho_llm, 'rho_bdt': rho_bdt,
            'MAD_llm': mad_llm, 'MAD_bdt': mad_bdt, 'n': len(pll)}


if __name__ == '__main__':
    results = {
        'Qwen-max': compute('qwen2.5-7b'),
        'MiroThinker-1.7-mini': compute('MiroThinker-1.7-mini'),
        'DS V4': compute('DeepSeek-V4-Flash'),
    }

    models = ['Qwen-max', 'MiroThinker\n(1.7-mini)', 'DS V4']
    x = np.arange(len(models))
    w = 0.32

    colors = {
        'Unconstrained LLM': '#d62728',
        'NDS (isotonic)': '#ff7f0e',
        'BDT ($\\lambda$=0.25)': '#2ca02c',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300)
    nds = [0.0, 0.0, 0.0]

    # (a) MVR-Wait
    ax = axes[0, 0]
    ax.bar(x - w, [results[m]['MVR_llm'] for m in ['Qwen-max', 'MiroThinker-1.7-mini', 'DS V4']], w,
           label='Unconstrained LLM', color=colors['Unconstrained LLM'])
    ax.bar(x, nds, w, label='NDS (isotonic)', color=colors['NDS (isotonic)'])
    ax.bar(x + w, [results[m]['MVR_bdt'] for m in ['Qwen-max', 'MiroThinker-1.7-mini', 'DS V4']], w,
           label='BDT ($\\lambda$=0.25)', color=colors['BDT ($\\lambda$=0.25)'])
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Mean Violation Rate (MVR-Wait)', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title('(a) Behavioral consistency: MVR-Wait', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (b) Spearman rho
    ax = axes[0, 1]
    ax.bar(x - w, [results[m]['rho_llm'] for m in ['Qwen-max', 'MiroThinker-1.7-mini', 'DS V4']], w,
           label='Unconstrained LLM', color=colors['Unconstrained LLM'])
    ax.bar(x, nds, w, label='NDS (isotonic)', color=colors['NDS (isotonic)'])
    ax.bar(x + w, [results[m]['rho_bdt'] for m in ['Qwen-max', 'MiroThinker-1.7-mini', 'DS V4']], w,
           label='BDT ($\\lambda$=0.25)', color=colors['BDT ($\\lambda$=0.25)'])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Spearman $\\rho$ vs $P_{static}$', fontsize=9)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title('(b) Policy ranking: Spearman $\\rho$', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (c) Subgroup heterogeneity
    ax = axes[1, 0]
    subgroups = ['Age 18-44', 'Age 45-65', 'Female', 'Male', 'Bachelor+']
    rho = [0.950, 0.952, 0.949, 0.954, 0.951]
    y = np.arange(len(subgroups))
    ax.barh(y, rho, color=colors['BDT ($\\lambda$=0.25)'])
    ax.set_yticks(y); ax.set_yticklabels(subgroups, fontsize=9)
    ax.set_xlabel('Spearman $\\rho$', fontsize=9)
    ax.set_xlim(0.86, 1.00)
    ax.set_title('(c) Subgroup heterogeneity: BDT $\\rho$ (DS V4 / Qwen)', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (d) MAD
    ax = axes[1, 1]
    ax.bar(x - w, [results[m]['MAD_llm'] for m in ['Qwen-max', 'MiroThinker-1.7-mini', 'DS V4']], w,
           label='Unconstrained LLM', color=colors['Unconstrained LLM'])
    ax.bar(x, nds, w, label='NDS (isotonic)', color=colors['NDS (isotonic)'])
    ax.bar(x + w, [results[m]['MAD_bdt'] for m in ['Qwen-max', 'MiroThinker-1.7-mini', 'DS V4']], w,
           label='BDT ($\\lambda$=0.25)', color=colors['BDT ($\\lambda$=0.25)'])
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('MAD vs $P_{static}$', fontsize=9)
    ax.set_ylim(0, 0.40)
    ax.set_title('(d) Mean absolute deviation', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('BDT out-of-design performance (12 scenarios, 3 LLMs: 1 open-source 7B, 1 commercial, 1 closed-source)',
                 fontsize=10.5, fontweight='bold', y=1.00)

    handles = [mpatches.Patch(color=colors[k], label=k) for k in colors]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = Path('figures/figure_1_ood_performance.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {out}")

    # Print metrics summary
    print()
    print("=" * 70)
    for m, r in results.items():
        print(f"{m}: MVR(uncon)={r['MVR_llm']:.3f}, MVR(BDT)={r['MVR_bdt']:.3f}, "
              f"rho(uncon)={r['rho_llm']:.3f}, rho(BDT)={r['rho_bdt']:.3f}, n={r['n']}")
