# Table 1 — Qwen2.5-72B (local vllm-mlx)

Paper label: **Qwen2.5-72B**. Local weights: **Qwen2-72B-4bit** on ORICO.

## 1. Start local model (required)

```bash
~/.openclaw/workspace/table1/start_local_vllm_qwen.sh
```

## 2. Run Table 1 (local only — no OpenRouter fallback)

```bash
cd ~/.openclaw/workspace
export TABLE1_QWEN_BACKEND=local
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

python3 table1_qwen72b_static.py --preflight   # must print backend=local
python3 table1_qwen72b_static.py run-llm
python3 table1_qwen72b_static.py metrics
```

Preflight must show:

```text
Using backend=local url=http://127.0.0.1:8000/v1/chat/completions model=/Volumes/ORICO/mlx-models/Qwen2-72B-4bit
```

## 3. If you already ran on OpenRouter (~127 rows)

Those rows are **not** from local 72B. Archive before a clean local run:

```bash
cd ~/.openclaw/workspace
mv llm_parsed_outputs_qwen72b_unconstrained_partial.csv \
   llm_parsed_outputs_qwen72b_openrouter_partial.csv.bak 2>/dev/null || true
mv llm_raw_outputs_qwen72b_unconstrained_partial.csv \
   llm_raw_outputs_qwen72b_openrouter_partial.csv.bak 2>/dev/null || true
```

Then start local vllm and run `run-llm` again (640 calls, several hours).

## 4. Cloud only (explicit opt-in)

```bash
export TABLE1_QWEN_BACKEND=openrouter
```
