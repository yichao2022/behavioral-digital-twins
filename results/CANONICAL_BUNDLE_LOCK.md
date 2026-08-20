# 64-state evaluation grid — Canonical Results Bundle (LOCK 2026-08-20)

**Status: LOCKED. Do not modify without re-deriving all downstream tables.**

## Reproducibility
- Grid: `bdt_eval_grid_static.csv` (5-param refit, 320 lines = 64 states × 5 repeats, mtime 7/28)
- Frontier estimator: `lambda_sensitivity_analysis.py` + `data_scaling_fixed_lambda025.py`
- 6-param MXL reference: `bdt_eval_grid_static_6param.csv` (P_static_6 column, MXL frontier refit)
- LLM parsed outputs: `llm_parsed_outputs_qwen72b_unconstrained.csv` (mtime 7/28), `llm_parsed_outputs_deepseek_*.csv`, `llm_parsed_outputs_mirothinker_*.csv`

## Canonical result files (md5-locked 2026-08-20)
- `results/lambda_sensitivity_frontier.csv` — Qwen 12-row λ-sweep (5-param grid + 6-param MXL reference, main bundle)
  - md5: `578440e598b8c52a7267f6fd3cd85bdf`
  - λ=0.25 (main anchor): MSE 0.0030 / MAE 0.0469 / MVR-Wait 0.0417 / MVR-SE 0.2708 / ρ 0.9520
  - λ=1.00 (raw unconstrained LLM): MSE 0.0482 / MAE 0.1878 / MVR-Wait 0.6250 / MVR-SE 0.5521 / ρ -0.0238
- `results/lambda_sensitivity_qwen72b.csv` — Qwen 12-row λ-sweep (Qwen-specific, identical numerical content)
  - md5: `fb2d70bddb118ed152d13dc61cf716dd`
- `results/data_scaling_fixed_lambda025_table.csv` — 4 rows (30/50/70/100%) for full-sample benchmark at fixed λ=0.25
  - 30%: MSE 0.0286, MVR-W 0.2458, ρ 0.6239
  - 50%: 0.0279, 0.2437, 0.6293
  - 70%: 0.0277, 0.2479, 0.6241
  - 100%: 0.0030, 0.0417, 0.9520 (matches main λ=0.25 anchor)

## Stale files (DO NOT USE)
- `results/STALE_8d15_DO_NOT_USE_lambda_sensitivity_qwen72b_6param_full.csv` — Aug 15 15:53 output from **deleted** `bdt_eval_grid_updated.csv` (now removed from `bdt_repo/`). Numbers (λ=0.25: MSE 0.0016 / MVR-W 0.3021 / ρ 0.6908; λ=1.00: MSE 0.0258 / MVR-W 0.6250 / ρ 0.0608) are **not reproducible from current code**. Renamed with `STALE_*` prefix to make accidental use impossible. Move to `trash/` after ViH submission.

## Paper main bundle (reproducible, locked 2026-08-20)
| Model | Raw MSE | Raw MAE | Raw MVR-Wait | Raw ρ | EFR (λ=0.25) MSE | EFR MAE | EFR MVR-Wait | EFR ρ |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | 0.0482 | 0.1878 | 62.5% | -0.024 | 0.0030 | 0.0469 | 4.2% | 0.952 |
| DeepSeek V4 Pro | 0.0807 | 0.2331 | 34.4% | 0.245 | 0.0050 | 0.0583 | 0.0% | 0.973 |
| MiroThinker-1.7-mini | 0.0669 | 0.2034 | 10.4% | 0.459 | 0.0042 | 0.0509 | 0.0% | 0.924 |

## Forensics
- **Why two bundles existed**: Aug 15 production run used `bdt_eval_grid_updated.csv` (a 5-param grid with a different P_static column for the 6-param MXL reference); this grid was deleted on Aug 17 when the production pipeline migrated to the canonical `bdt_eval_grid_static.csv` 5-param refit + 6-param MXL frontier reference. The Aug 15 `lambda_sensitivity_qwen72b_6param_full.csv` output was preserved in the supplement but is not reproducible from current code.
- **Verification**: Re-running `lambda_sensitivity_qwen72b.py` on the canonical grid reproduces the main bundle (0.0030/0.0417/0.9520) byte-for-byte. Re-running on the deleted grid is impossible.
- **Action taken**: All stale 0.0016/0.3021/0.6908 numbers in the supplement (Tables S5, S7, S11) and main text (L64 narrative, L329 Conclusion) have been swept to the canonical 0.0030/0.0417/0.9520 numbers.

## DCE matching (also locked 2026-08-20)
- 12-cell populated intersection (per `analysis_output/dce_encoded.csv` + `heldout_matched_audit.md`)
- 73 task-level A-versus-B matched tasks (canonical full 3-alt matched subset is empty)
- Task-level log loss (n=73): Pure-DCE 0.4502 / Static-BDT (λ=0.25) 0.5244 / unconstrained LLM 0.7896
- Alt-level binary log loss (n=809): Pure-DCE 0.6882 / Static-BDT 0.7034 / unconstrained LLM 0.8149
