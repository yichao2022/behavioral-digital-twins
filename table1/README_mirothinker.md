# Table 1 — MiroThinker-1.7-mini (local)

MiroThinker is a **reasoning/agent** model. For Table 1 you must disable thinking or it will output `<think>` meta-commentary instead of the three-line format.

## Fix (required)

Every API call uses:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

`table1_mirothinker_static.py` sets this automatically. Optional server default is in `ai.openclaw.vllm-mlx-mirothinker.plist`.

## Run (Terminal.app)

```bash
cd ~/.openclaw/workspace
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

./table1/stop_mirothinker.sh
./table1/reset_mirothinker_partial.sh   # archive failed think-only partial

./table1/start_local_vllm_mirothinker.sh   # :8001, stop Qwen :8000 first

PYTHONPATH=. python3 table1/test_miro_one.py   # must print OK
PYTHONPATH=. python3 table1_mirothinker_static.py --preflight

export TABLE1_MIRO_BACKEND=local
./table1/run_mirothinker_local_background.sh
tail -f table1/mirothinker_local_run.log
```

## If still failing

- Log shows `The user asks` or `<think>` → thinking still on; rerun preflight after restart.
- Batch aborts at 15 calls with >50% errors → same fix.
- Do **not** run 640 calls with the old partial; reset first.

## Outputs

- `llm_parsed_outputs_mirothinker_unconstrained*.csv`
- `static_bdt_anchor_mirothinker.csv`
- `metrics_table1_mirothinker_static_lambda_sensitivity.csv`
