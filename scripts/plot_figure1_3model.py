"""
Figure 1 (3-model version, v4): 2-panel layout (Spearman rho + MAD).
Dropped: Panel (a) MVR-Wait (NDS/BDT are 0 by construction; MiroThinker
violation already in caption), Panel (c) subgroup heterogeneity (MiroThinker
subgroup data not collected; subgroup results in Appendix H.3).
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
import matplotlib.patches as mpatches

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
        if r['scenario_id'] in pstatic:
            scenarios[r['scenario_id']] = r

groups = sorted(set(scenarios[s]['expected_wait_order_group'] for s in scenarios))


def load(name):
    fp = V1 / f'{name}_n200_n40.csv'
    by = defaultdict(list)
    with open(fp, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('parse_success', '').lower() in ('true', '1', 'yes'):
                try: by[r['scenario_id']].append(float(r['probability_0_1']))
                except: continue
    return by


def nds_isotonic(by_scen_raw):
    out = {}
    for gname in groups:
        g_sids = sorted([s for s in scenarios if scenarios[s]['expected_wait_order_group'] == gname],
                        key=lambda s: float(scenarios[s]['wait_time']))
        g_sids = [s for s in g_sids if s in by_scen_raw and len(by_scen_raw[s]) > 0]
        if not g_sids: continue
        X = np.array([pstatic[s] for s in g_sids])
        y = np.array([np.mean(by_scen_raw[s]) for s in g_sids])
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
        y_iso = iso.fit_transform(X, y)
        for s, v in zip(g_sids, y_iso):
            out[s] = float(v)
    return out


def metrics(p_dict):
    sids = sorted([s for s in p_dict if s in pstatic])
    psl = [pstatic[s] for s in sids]
    pll = [p_dict[s] for s in sids]
    rho = spearmanr(pll, psl)[0] if len(pll) >= 3 else None
    mad = mean(abs(a-b) for a, b in zip(pll, psl))
    return {'rho': float(rho) if rho is not None else 0, 'mad': mad}


def compute_full(fname):
    by = load(fname)
    p_llm_mean = {s: mean(v) for s, v in by.items() if s in pstatic and v}
    nds = nds_isotonic(by)
    p_bdt = {s: LAMBDA * p_llm_mean[s] + (1 - LAMBDA) * pstatic[s] for s in p_llm_mean}
    return {
        'Unconstrained': metrics(p_llm_mean),
        'NDS (isotonic)': metrics(nds),
        'BDT (lambda=0.25)': metrics(p_bdt),
    }


if __name__ == '__main__':
    results = {m: compute_full(f) for m, f in [
        ('Qwen-max', 'qwen2.5-7b'),
        ('MiroThinker-1.7-mini', 'MiroThinker-1.7-mini'),
        ('DS V4', 'DeepSeek-V4-Flash'),
    ]}

    model_order = list(results.keys())
    x = np.arange(len(model_order))
    w = 0.27

    colors = {
        'Unconstrained': '#d62728',
        'NDS (isotonic)': '#ff7f0e',
        'BDT (lambda=0.25)': '#2ca02c',
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # (a) Spearman rho
    ax = axes[0]
    for i, m in enumerate(['Unconstrained', 'NDS (isotonic)', 'BDT (lambda=0.25)']):
        offset = (i - 1) * w
        ax.bar(x + offset, [results[mo][m]['rho'] for mo in model_order], w,
               color=colors[m], label=m)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(['Qwen-max', 'MiroThinker\n(1.7-mini)', 'DS V4'], fontsize=10)
    ax.set_ylabel('Spearman $\\rho$ vs $P_{static}$', fontsize=10)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title('(a) Policy ranking: Spearman $\\rho$', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.annotate('MiroThinker:\nUnconstrained $\\rho=-0.509$\nNDS $\\rho=0.073$ (weak rescue)\nBDT $\\rho=0.871$ (full restoration)',
                xy=(1, -0.509), xytext=(1.6, -0.7), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', lw=0.5))

    # (b) MAD
    ax = axes[1]
    for i, m in enumerate(['Unconstrained', 'NDS (isotonic)', 'BDT (lambda=0.25)']):
        offset = (i - 1) * w
        ax.bar(x + offset, [results[mo][m]['mad'] for mo in model_order], w,
               color=colors[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(['Qwen-max', 'MiroThinker\n(1.7-mini)', 'DS V4'], fontsize=10)
    ax.set_ylabel('MAD vs $P_{static}$', fontsize=10)
    ax.set_ylim(0, 0.40)
    ax.set_title('(b) Mean absolute deviation', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('BDT out-of-design performance: 3 LLMs x 3 methods on 12 pre-registered OOD scenarios',
                 fontsize=11, fontweight='bold', y=1.02)

    handles = [mpatches.Patch(color=colors[k], label=k) for k in colors]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = Path('figures/figure_1_ood_performance.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')
