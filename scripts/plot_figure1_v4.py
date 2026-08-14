"""
Redraw Figure 1: BDT OOD performance (4 panels) with corrected model name.
Original: 'DS V3' → 'DS V4' (unified DeepSeek V4 per paper main text).
Saves /tmp/bdt_vih_latex/figures/figure_1_ood_performance.png at 300 dpi.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from Table 2 (Panel A: 6 paired wait-time groups, MVR/MAD/rho; Panel B: PRC)
# Qwen-max and DeepSeek V4 (was V3, now unified per paper revision)

fig, axes = plt.subplots(2, 2, figsize=(11.9, 8.9), dpi=300)

# Model names (corrected)
models = ['Qwen-max', 'DS V4']  # was 'DS V3'
x = np.arange(len(models))
w = 0.25

colors = {'Unconstrained LLM': '#d62728', 'NDS (isotonic)': '#ff7f0e', 'BDT ($\\lambda$=0.25)': '#2ca02c'}

# Panel (a): MVR-Wait (Panel A, 6 pairs)
ax = axes[0, 0]
unconstrained = [0.333, 0.667]  # 4/6 for V4
nds = [0.0, 0.0]
bdt = [0.0, 0.0]
ax.bar(x - w, unconstrained, w, label='Unconstrained LLM', color=colors['Unconstrained LLM'])
ax.bar(x, nds, w, label='NDS (isotonic)', color=colors['NDS (isotonic)'])
ax.bar(x + w, bdt, w, label='BDT ($\\lambda$=0.25)', color=colors['BDT ($\\lambda$=0.25)'])
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Mean Violation Rate (MVR-Wait)', fontsize=9)
ax.set_ylim(0, 0.8)
ax.set_title('(a) Behavioral consistency: MVR-Wait by method', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel (b): Spearman rho
ax = axes[0, 1]
unconstrained = [0.12, -0.07]
nds = [0.0, -0.943]
bdt = [0.87, 0.87]
ax.bar(x - w, unconstrained, w, label='Unconstrained LLM', color=colors['Unconstrained LLM'])
ax.bar(x, nds, w, label='NDS (isotonic)', color=colors['NDS (isotonic)'])
ax.bar(x + w, bdt, w, label='BDT ($\\lambda$=0.25)', color=colors['BDT ($\\lambda$=0.25)'])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Spearman $\\rho$ (policy rank)', fontsize=9)
ax.set_ylim(-1.0, 1.0)
ax.set_title('(b) Policy ranking: Spearman $\\rho$ by method', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel (c): Subgroup heterogeneity (BDT only, 5 subgroups, ~0.95 rho)
ax = axes[1, 0]
subgroups = ['Age 18-44', 'Age 45-65', 'Female', 'Male', 'Bachelor+']
rho = [0.950, 0.952, 0.949, 0.954, 0.951]
y = np.arange(len(subgroups))
ax.barh(y, rho, color=colors['BDT ($\\lambda$=0.25)'])
ax.set_yticks(y)
ax.set_yticklabels(subgroups, fontsize=9)
ax.set_xlabel('Spearman $\\rho$', fontsize=9)
ax.set_xlim(0.86, 1.00)
ax.set_title('(c) Subgroup heterogeneity: BDT $\\rho$ (809-respondent matched subset)', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel (d): PRC (20 OOD scenarios)
ax = axes[1, 1]
unconstrained = [0.33, 0.40]
nds = [0.0, 0.0]
bdt = [0.87, 0.60]
ax.bar(x - w, unconstrained, w, label='Unconstrained LLM', color=colors['Unconstrained LLM'])
ax.bar(x, nds, w, label='NDS (isotonic)', color=colors['NDS (isotonic)'])
ax.bar(x + w, bdt, w, label='BDT ($\\lambda$=0.25)', color=colors['BDT ($\\lambda$=0.25)'])
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Policy Ranking Consistency', fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title('(d) PRC: 20 OOD scenarios', fontsize=10, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Main title
fig.suptitle('BDT Out-of-design performance (20 OOD scenarios, 6 paired wait-time groups, 5 subgroups)',
             fontsize=11, fontweight='bold', y=0.995)

# Single legend at bottom
handles = [mpatches.Patch(color=colors[k], label=k) for k in colors]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('/tmp/bdt_vih_latex/figures/figure_1_ood_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved figure_1_ood_performance.png (DS V3 → DS V4)")
