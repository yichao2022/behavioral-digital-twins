#!/bin/bash
# Archive failed Miro partial (think-only outputs) before a clean restart.
set -euo pipefail
WS="$HOME/.openclaw/workspace"
ts=$(date +%Y%m%d_%H%M%S)
"$WS/table1/stop_mirothinker.sh"
for f in llm_parsed_outputs_mirothinker_unconstrained_partial.csv \
         llm_raw_outputs_mirothinker_unconstrained_partial.csv; do
  if [[ -f "$WS/$f" ]]; then
    mv "$WS/$f" "$WS/${f%.csv}.failed_${ts}.csv.bak"
    echo "Archived $f"
  fi
done
echo "Ready to rerun: ./table1/run_mirothinker_local_background.sh"
