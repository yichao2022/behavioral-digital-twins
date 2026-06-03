#!/bin/bash
# Start local Qwen 72B (vllm-mlx) for Table 1. Run in Terminal.app.
set -euo pipefail

PLIST="$HOME/Library/LaunchAgents/ai.openclaw.vllm-mlx-qwen2.plist"
MODEL="/Volumes/ORICO/mlx-models/Qwen2-72B-4bit"

if [[ ! -d "$MODEL" ]]; then
  echo "❌ Model not found: $MODEL"
  echo "   Plug in ORICO drive and wait for /Volumes/ORICO to mount."
  exit 1
fi

if [[ ! -f "$PLIST" ]]; then
  echo "❌ Missing LaunchAgent: $PLIST"
  exit 1
fi

echo "Starting vllm-mlx-qwen2 via launchd..."
launchctl bootout "gui/$(id -u)/ai.openclaw.vllm-mlx-qwen2" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/ai.openclaw.vllm-mlx-qwen2"

echo "Waiting for http://127.0.0.1:8000/v1/models ..."
for i in $(seq 1 120); do
  if curl -sf -m 5 -H "Authorization: Bearer not-needed" http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "✅ Local Qwen API is up (after ~${i}s)"
    curl -sS -H "Authorization: Bearer not-needed" http://127.0.0.1:8000/v1/models | head -c 300
    echo ""
    exit 0
  fi
  sleep 5
done

echo "❌ Timed out. Check logs:"
echo "   tail -50 ~/.openclaw/logs/vllm-mlx-qwen2.stderr.log"
exit 1
