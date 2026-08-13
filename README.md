# Empirical-Frontier Regularization for LLM Synthetic Agents

Replication code and outputs for the empirical-frontier regularization pipeline: mixed-logit empirical choice frontier (`P_static`), unconstrained LLM policy simulation on a 64-state grid, and convex anchoring \\(\pi_{\text{EFR}} = \lambda \bar{\pi}_{\text{LLM}} + (1-\lambda) P_{\text{static}}\\) (main specification \\(\lambda = 0.25\\)). The framework is designed as an empirical correction layer for LLM synthetic agents, not as a replacement for DCE-based choice modeling.

## Requirements

- Python 3.11+
- `numpy`, `scipy`, `certifi` (see `requirements.txt`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

| File | Description |
|------|-------------|
| `bdt_eval_grid_static.csv` | 64-state grid with `P_static` |
| `analysis_output/dce_encoded.csv` | Encoded DCE choice data (held-out validation) |
| `analysis_output/mxl_coefs.csv` | Mixed logit coefficients for frontier construction |
| `llm_parsed_outputs_*_unconstrained.csv` | Parsed LLM runs (64×10 per model) |

## Main scripts

| Script | Purpose |
|--------|---------|
| `table1_qwen72b_static.py` | Qwen Table 1 + λ sensitivity |
| `table1_deepseek_static.py` | DeepSeek Table 1 |
| `table1_mirothinker_static.py` | MiroThinker Table 1 |
| `heldout_dce_validation.py` | Train-only logit + held-out metrics |
| `heldout_dce_matched_comparison.py` | Full vs matched panels |
| `heldout_auc_orientations.py` | AUC orientation diagnostics |
| `lambda_sensitivity_analysis.py` | λ grid (frontier + held-out) |
| `prc_analysis.py` | Policy ranking consistency |
| `extract_runtime_summary.py` | Runtime from checkpoint CSVs |

Set `DEEPSEEK_API_KEY` (and optional `DASHSCOPE_API_KEY` / `OPENROUTER_API_KEY`) for API backends; local vLLM uses `http://127.0.0.1:8000/v1` by default.

## Results (precomputed)

Under `results/` and `outputs/`: λ sensitivity tables, held-out predictions, PRC, runtime summary, Figure 2/3 data.

## Out-of-Design Stress Test

| Script | Purpose |
|--------|---------|
| `scripts/out_of_design_stress_test.py` | Generate prompts, compute P_static, run LLM (OpenRouter) |
| `scripts/compute_out_of_design_metrics.py` | Compute MVR-Wait, MAD, Spearman from LLM outputs |

```bash
# Step-by-step
python3 scripts/out_of_design_stress_test.py gen-prompts
python3 scripts/out_of_design_stress_test.py compute-pstatic
python3 scripts/out_of_design_stress_test.py run-llm "Qwen2.5-72B"
python3 scripts/out_of_design_stress_test.py run-llm "DeepSeek V3"
python3 scripts/compute_out_of_design_metrics.py
```

### Inputs
- `out_of_design_scenarios.csv` — 12 scenarios (6 wait-time pairs × 2 levels)
- `analysis_output/mxl_coefs.csv` — Mixed-logit coefficients for P_static

### Outputs
- `results/out_of_design_parsed_probabilities.csv` — Parsed LLM probabilities
- `results/out_of_design_bdt_metrics.csv` — MVR-Wait, MAD, Spearman per model/method
- `results/out_of_design_stress_test_table.tex` — LaTeX table
- `section_out_of_design_stress_test.tex` — LaTeX subsection for paper

Under `results/` and `outputs/`: λ sensitivity tables, held-out predictions, PRC, runtime summary, Figure 2/3 data.

## Revision-Stage Stress Test (2026-08-13)

A complementary stress test on a representative subset of eight Panel B out-of-design scenarios (G1–N1) was added during the Value in Health revision cycle. The test uses three additional open-source models (Qwen2.5-7B-Instruct, DeepSeek-V4-Flash, MiroThinker-1.7-mini) on 4,800 prompts and surfaces two failure modes that the original 20-scenario Panel B evaluation did not detect:

- **Zero-variance default-answer signature** — Qwen2.5-7B produced constant outputs in {0.75, 0.85} across all 1,600 prompts, indicating a default-answer mode rather than prompt-grounded simulation.
- **Complete parsing failure on waiver-language scenario N1** — DeepSeek-V4-Flash achieved 0% parse on scenario N1 (a complex waiver-language description) despite 100% parse on the in-design 64-state grid.

These observations support, rather than weaken, the EFR framework: the empirical anchoring layer is necessary precisely because unconstrained LLM behavior under realistic policy narratives is not uniformly reliable. Replication scripts and per-scenario CSVs are available at the `stress_test/` subdirectory (Qwen2.5-7B, DeepSeek-V4-Flash, MiroThinker-1.7-mini).

## Citation

If you use this repository, cite the associated Value in Health submission:

> Jin, Y. (2026). *Anchoring Large Language Models to Empirical Behavioral Evidence: A Hybrid Approach for Vaccination Uptake Simulation*. Manuscript submitted to Value in Health (Special Issue on LLM/DCE Integration). University of Texas at Dallas.

## License

Code: MIT. Research data files may carry separate usage restrictions—do not redistribute `dce_encoded.csv` without permission from the original study.
