# Out-of-Design Stress Test — Run Instructions

## Status

| Model | OpenRouter | Status |
|-------|-----------|--------|
| Qwen2.5-72B | `qwen/qwen-2.5-72b-instruct` | ✅ Done (120 calls) |
| DeepSeek V4 Pro | `deepseek/deepseek-v4-pro` | ✅ Done (120 calls) |
| MiroThinker-1.7-mini | Not available on OpenRouter | ❌ Requires local vLLM |

## How to Run

### Prerequisites
```bash
pip install numpy scipy
```

### Step 1: Generate prompts
```bash
python3 scripts/out_of_design_stress_test.py gen-prompts
```

### Step 2: Compute P_static
```bash
python3 scripts/out_of_design_stress_test.py compute-pstatic
```

### Step 3: Run LLM generation

**Qwen2.5-72B via OpenRouter:**
```bash
export OPENROUTER_API_KEY=***
python3 scripts/out_of_design_stress_test.py run-llm "Qwen2.5-72B"
```

**DeepSeek V4 Pro via OpenRouter:**
```bash
export OPENROUTER_API_KEY=***
python3 scripts/out_of_design_stress_test.py run-llm "DeepSeek V4 Pro"
```

**MiroThinker-1.7-mini (local vLLM):**
Requires a running local vLLM server at http://127.0.0.1:8000/v1.
The existing table1 scripts (e.g., `table1_mirothinker_static.py`) can be adapted.

1. Start server: see `table1/start_local_vllm_mirothinker.sh`
2. Run generation:
```bash
python3 -c "
from scripts.out_of_design_stress_test import run_llm_model
run_llm_model('MiroThinker-1.7-mini', 'no-api-key')
"
```
Wait, this uses OpenRouter by default. For local vLLM, the existing LLM scripts in the repo need to be referenced.
See `table1_mirothinker_static.py` for the local inference pattern.

### Step 4: Compute metrics
```bash
python3 scripts/compute_out_of_design_metrics.py
```

### Output files
- `results/out_of_design_raw_outputs.jsonl` - Raw LLM responses
- `results/out_of_design_parsed_probabilities.csv` - Parsed probabilities per repeat
- `results/out_of_design_bdt_metrics.csv` - MVR-Wait, MAD, Spearman by model/method
- `results/out_of_design_stress_test_table.tex` - LaTeX table
- `section_out_of_design_stress_test.tex` - LaTeX subsection for paper
