#!/bin/bash
# Table 1 MiroThinker — local vllm-mlx :8001, background.
set -euo pipefail
cd "$HOME/.openclaw/workspace"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy __CURSOR_SANDBOX_ENV_RESTORE
export TABLE1_MIRO_BACKEND=local

LOG="$HOME/.openclaw/workspace/table1/mirothinker_local_run.log"
PIDFILE="$HOME/.openclaw/workspace/table1/mirothinker_local_run.pid"

"$HOME/.openclaw/workspace/table1/stop_qwen72b.sh" || true

if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE")
  if kill -0 "$pid" 2>/dev/null; then
    echo "Already running PID $pid — tail -f $LOG"
    exit 0
  fi
fi

if ! curl -sf -m 5 -H "Authorization: Bearer not-needed" http://127.0.0.1:8001/v1/models >/dev/null; then
  "$HOME/.openclaw/workspace/table1/start_local_vllm_mirothinker.sh"
fi

python3 table1_mirothinker_static.py --preflight
nohup python3 -u table1_mirothinker_static.py run-llm --no-preflight >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "Started MiroThinker Table-1 PID $(cat "$PIDFILE")"
echo "  tail -f $LOG"
