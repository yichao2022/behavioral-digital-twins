#!/bin/bash
# Run Table 1 DeepSeek V4 Pro in Terminal (avoids Cursor proxy).
set -euo pipefail
cd "$HOME/.openclaw/workspace"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy 2>/dev/null || true

echo "=== Step 1: static grid ==="
python3 table1_deepseek_static.py build-grid

echo "=== Preflight ==="
python3 table1_deepseek_static.py --preflight

echo "=== Step 2: LLM (64×10, resume-safe) ==="
python3 table1_deepseek_static.py run-llm --no-preflight

echo "=== Steps 3–4: anchor + metrics ==="
python3 table1_deepseek_static.py metrics
