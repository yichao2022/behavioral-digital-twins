#!/bin/bash
# Start MiroThinker 1.7-mini on port 8001 (vllm-mlx).
set -euo pipefail
MODEL="/Users/cary/models/MiroThinker-1.7-mini"
PLIST="$HOME/Library/LaunchAgents/ai.openclaw.vllm-mlx-mirothinker.plist"
LOG_DIR="$HOME/.openclaw/logs"
mkdir -p "$LOG_DIR"

if [[ ! -e "$MODEL" ]]; then
  echo "❌ Model not found: $MODEL"
  exit 1
fi

echo "Stopping Qwen Table-1 runner (if any) ..."
"$HOME/.openclaw/workspace/table1/stop_qwen72b.sh" || true

echo "Stopping Qwen vllm on :8000 (free RAM for MiroThinker) ..."
launchctl bootout "gui/$(id -u)/ai.openclaw.vllm-mlx-qwen2" 2>/dev/null || true

echo "Starting vllm-mlx-mirothinker on :8001 ..."
launchctl bootout "gui/$(id -u)/ai.openclaw.vllm-mlx-mirothinker" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/ai.openclaw.vllm-mlx-mirothinker"

echo "Waiting for http://127.0.0.1:8001/v1/models ..."
for i in $(seq 1 90); do
  if curl -sf -m 5 -H "Authorization: Bearer not-needed" http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "✅ MiroThinker API up (~${i}s)"
    curl -sS -H "Authorization: Bearer not-needed" http://127.0.0.1:8001/v1/models | head -c 400
    echo ""
    exit 0
  fi
  sleep 2
done
echo "❌ Timed out. tail -50 $LOG_DIR/vllm-mlx-mirothinker.stderr.log"
exit 1
