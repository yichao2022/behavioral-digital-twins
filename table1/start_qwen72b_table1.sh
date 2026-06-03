#!/bin/bash
# Table 1 Qwen — LOCAL vllm-mlx only. Run in Terminal.app.
set -euo pipefail
cd "$HOME/.openclaw/workspace"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy 2>/dev/null || true
export TABLE1_QWEN_BACKEND=local

echo "=== Step 0: local vllm-mlx ==="
"$HOME/.openclaw/workspace/table1/start_local_vllm_qwen.sh"

echo "=== Step 1: static grid ==="
python3 table1_qwen72b_static.py build-grid

echo "=== Preflight (must show backend=local) ==="
python3 table1_qwen72b_static.py --preflight

echo "=== Step 2: LLM (64×10, resume-safe) ==="
python3 table1_qwen72b_static.py run-llm --no-preflight

echo "=== Steps 3–4: anchor + metrics ==="
python3 table1_qwen72b_static.py metrics
