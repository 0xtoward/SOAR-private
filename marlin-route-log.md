# Marlin Route Log

## Environment Note

- Local draft/package work:
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft`
- Remote runtime/checkpoint targets:
  - `rtx6000-2:/root/autodl-tmp`
- Reusable skill logic is outside this repo:
  - `/Users/ql/.codex/skills`

## 2026-04-09

- Route status:
  - technically promising as a `gptq_marlin` runtime fallback
  - not yet a strong submission candidate
- Why:
  - local notes already show `SGLang + gptq_marlin` can load and run MiniCPM-SALA
  - the currently known public GPTQ checkpoint is much too inaccurate for SOAR
  - self-quantization is still blocked by `gptqmodel` / `transformers` compatibility
- Current draft cleanup:
  - `prepare_env.sh` now matches a Marlin route instead of the old NVFP4 scaffold
  - `prepare_model.sh` now calls `preprocess_model.py`
  - `preprocess_model.py` now prefers:
    1. already-GPTQ input
    2. `SOAR_GPTQ_LOCAL_PATH`
    3. fallback Hugging Face GPTQ repo
- Next validation target:
  - confirm the draft is internally consistent and can materialize a GPTQ checkpoint from `SOAR_GPTQ_LOCAL_PATH=/root/autodl-tmp/models-gptq-w4a16-v19` without invoking ModelOpt
- Highest-priority blocker:
  - checkpoint quality, not runtime kernel support

## 2026-04-09 Champion-Route Handoff

- Exact route name:
  - `W4A16 + GPTQ + Marlin (local GPTQ checkpoint first, no public fallback in first measurement)`
- Why this matches the champion notes better than current NVFP4:
  - Week 4's main route is explicitly `W4A16 + GPTQ + Marlin`, chosen to target decode-side bandwidth pressure rather than generic low-bit quantization
  - Week 3 also stresses calibration/data quality and stable local filtering, but it does not override Week 4's final primary route choice
  - current `NVFP4 MLP-only` is still a serious Blackwell-specific candidate, but it is not the same route family as the Week 4 winner
- Exact local draft/files that should be prepared next:
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/prepare_env.sh`
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/prepare_model.sh`
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/preprocess_model.py`
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/README.md`
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/RESULTS.md`
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/SHA256SUMS.txt`
- Structural judgment on the current draft:
  - not close enough as-is
  - it needs a structural change from a `runtime fallback package with public-download escape hatch` into a `local-GPTQ-first champion-route package`
  - specifically:
    - `prepare_env.sh` should be hardened like the NVFP4 draft with `set -euo pipefail` and post-install verification
    - the first-path materialization should prefer `SOAR_GPTQ_LOCAL_PATH=/root/autodl-tmp/models-gptq-w4a16-v19` and treat public download only as a last-resort debug fallback
    - unrelated NVFP4 leftovers in this draft such as `quantize_modelopt_nvfp4_conservative.py`, `nvidia_modelopt`/`pulp` baggage, and local `__pycache__` should be removed so the package is clearly one route
- Exact first measurement once the GPU is free:
  - first use the existing local GPTQ checkpoint, not the weak public fallback repo
  - first measurement should be a smoke comparison on the stable runtime path with no `fuse_topk`
  - exact measurement flow:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true && nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server --model-path /root/autodl-tmp/models-gptq-w4a16-v19 --trust-remote-code --host 127.0.0.1 --port 30001 --dtype float16 --quantization gptq_marlin --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --disable-radix-cache --disable-cuda-graph --dense-as-sparse > /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_server.log 2>&1 &'`
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 bash fast_test.sh --eval-only | tee /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_eval.log'`

## 2026-04-09 Draft Structural Cleanup

- Draft tightened from `mixed fallback package` into `local-GPTQ-first champion-route package`
- Key changes:
  - `prepare_env.sh` now uses `set -euo pipefail`, verifies wheel/index installs, and verifies editable `sglang` import
  - `preprocess_model.py` now prefers:
    1. already-GPTQ input
    2. `SOAR_GPTQ_LOCAL_PATH`
    3. public fallback download only when `SOAR_ALLOW_GPTQ_DOWNLOAD=1`
  - `prepare_model.sh` now prints the local checkpoint path and whether fallback download is enabled
  - removed obvious NVFP4 baggage:
    - `quantize_modelopt_nvfp4_conservative.py`
    - `nvidia_modelopt` / `pulp` wheels
    - local `__pycache__`
    - unused calibration files
- Fresh cleaned artifact:
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft.tar.gz`
  - sha256: `9dd4530b7476a0ee572cce577492d32a8f01d4775aafc2dbba08841d2feaf4b2`

## 2026-04-09 Next Quant Experiment After Stock-Base Control

- Queued recommendation:
  - run the champion-aligned `local GPTQ first` `W4A16 + GPTQ + Marlin` smoke as the next quantization experiment after the current stock-base control run finishes
- Why this is the next best quant move:
  - current NVFP4 has already been demoted to fallback after a weak `32%` smoke
  - Week 4 explicitly prioritizes `W4A16 + GPTQ + Marlin` as the main route
  - this first measurement is cheap because it reuses the existing local GPTQ checkpoint instead of launching a new quantization pass
- Gate before running:
  - only launch this if the stock-base control run is technically valid
  - if stock-base control shows harness/runtime invalidity, hold this experiment and fix the control path first
- Exact first measurement command when the current remote baseline test is done:
  - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true && nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server --model-path /root/autodl-tmp/models-gptq-w4a16-v19 --trust-remote-code --host 127.0.0.1 --port 30001 --dtype float16 --quantization gptq_marlin --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --disable-radix-cache --disable-cuda-graph --dense-as-sparse > /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_server.log 2>&1 &'`
  - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 bash fast_test.sh --eval-only | tee /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_eval.log'`

## 2026-04-09 Post-Official-Eval Lightweight Preflight

- Execute this first after the official stock-base eval finishes, before launching `gptq_marlin` smoke:
  - `ssh rtx6000-2 'python3 - <<'"'"'PY'"'"'\nfrom pathlib import Path\np = Path(\"/root/autodl-tmp/models-gptq-w4a16-v19\")\nrequired = [\"config.json\", \"quantize_config.json\", \"tokenizer_config.json\"]\nmissing = [name for name in required if not (p / name).exists()]\nweights = sorted([x.name for x in p.glob(\"*.safetensors\")])\nprint({\"path\": str(p), \"missing\": missing, \"num_safetensors\": len(weights), \"sample_weights\": weights[:3]})\nraise SystemExit(0 if not missing and weights else 1)\nPY'`
- Why this preflight:
  - no GPU work
  - verifies that the local GPTQ checkpoint we intend to use for the champion route is structurally present before spending the next free GPU window on serve + smoke

## 2026-04-09 Champion-Route Full-Eval Priority

- Full-eval rank among quant routes:
  - `#1`
- Why this route deserves the first official quant eval slot:
  - it matches the Week 4 champion mainline directly
  - it attacks the decode-bandwidth bottleneck the champion notes focused on
  - it does not require a fresh heavy quantization pass before first quality judgment because the local GPTQ checkpoint already exists
- Gate before spending a full official eval slot:
  - stock-base official eval must finish first
  - local GPTQ checkpoint structure must pass the lightweight preflight above
  - first stable-runtime smoke should not show immediate route collapse
- What should not preempt this route:
  - public fallback GPTQ download path
  - plain GPTQ without Marlin
  - another speculative NVFP4 variant beyond the current bounded fallback plan

## 2026-04-09 Post-Official-Eval Marlin Gate

- This route stays paused until the official stock-base eval is complete and judged technically healthy.
- Health read should start with this lightweight log scan:
  - `ssh rtx6000-2 'RUN_DIR=$(cat /root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_latest.txt) && echo RUN_DIR=$RUN_DIR && rg -n "Average Score|Total Duration|Overall TPS|Request failed|ERROR|HTTP" "$RUN_DIR/eval_model.log" "$RUN_DIR/server.log" || true && tail -n 20 "$RUN_DIR/eval_model.log"'`
- If that official stock-base eval is technically healthy:
  - keep `W4A16 + GPTQ + Marlin` as the next primary candidate
  - run the existing GPTQ checkpoint structure preflight above
  - then launch exactly:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true && nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server --model-path /root/autodl-tmp/models-gptq-w4a16-v19 --trust-remote-code --host 127.0.0.1 --port 30001 --dtype float16 --quantization gptq_marlin --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --disable-radix-cache --disable-cuda-graph --dense-as-sparse > /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_server.log 2>&1 &'`
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 bash fast_test.sh --eval-only | tee /root/autodl-tmp/SOAR-Toolkit/test_results/w4a16_marlin_local_smoke_eval.log'`
- If that official stock-base eval is not technically healthy:
  - do not launch `gptq_marlin`
  - next stronger stock-base quality-control check is:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true && nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server --model-path /root/autodl-tmp/models --trust-remote-code --host 127.0.0.1 --port 30001 --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --disable-radix-cache --disable-cuda-graph --dense-as-sparse > /root/autodl-tmp/SOAR-Toolkit/test_results/stock_base_control_server.log 2>&1 & sleep 20 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 MINI_PER_TASK=4 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 bash mini_test.sh | tee /root/autodl-tmp/SOAR-Toolkit/test_results/stock_base_control_mini.log'`

## 2026-04-09 Marlin Route Paused By Reachability, Not Route Logic

- This pass did not change route priority.
- Current reason the Marlin route did not advance:
  - local DNS cannot currently resolve `connect.bjb1.seetacloud.com`
  - this blocks `ssh rtx6000-2` from this machine even though the SSH alias/config is still correct
- Therefore:
  - the route remains queued as the first quant candidate after the stock official eval
  - no route-level demotion or fallback decision should be inferred from this pass

## 2026-04-09 Marlin Route Still Blocked By Reachability

- No new route-level evidence was gathered in this pass.
- The only new result was another direct SSH failure on DNS resolution for:
  - `connect.bjb1.seetacloud.com`
- Therefore:
  - `gptq_marlin` remains the first post-official quant route
  - but it stays blocked on reading the official stock result and regaining reachability

## 2026-04-09 Marlin Route Decision Rule

- Fresh quant-side route rule in this pass:
  - if `gptq_marlin` finishes cleanly with no material pathology and:
    - `Average Score >= 0.90 * stock_base_avg`
  - then keep `gptq_marlin` as the first official quant route
  - else promote:
    - `NVFP4 curated16k v2`
- This pass did not demote the Marlin route on quality grounds.
- This pass also did not launch the route, because the active official full eval remains the gating long run.

## 2026-04-09 Marlin Route Still First Under Sandbox Blocker

- Fresh quant delivery in this pass reaffirmed:
  - keep `gptq_marlin` as the first quant route
  - do not promote the prepared `NVFP4 32K64` fallback ahead of it from this pass
- New blocker in this pass:
  - fresh direct remote SSH from the current sandbox failed with:
    - `Operation not permitted`
- Therefore:
  - no new route-level remote evidence was gathered in this pass

## 2026-04-09 Marlin Fast-Proxy Gate Refined

- Fresh quant follow-up delivery in this pass:
  - exact post-official eval gate command:
    - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 bash fast_test.sh --eval-only | tee /root/autodl-tmp/SOAR-Toolkit/test_results/gptq_marlin_fast_eval.log'`
- Route decision rule refined in this pass:
  - keep `gptq_marlin` first only if the first post-official proxy gate:
    - finishes cleanly
    - shows no broad technical pathology
    - and reaches at least the base fast `avg_score`
  - otherwise:
    - do not spend a larger window on Marlin first
    - hand off to the prepared `NVFP4 mlp_weight_only + bf16 + 32K + 64` route
- New blocker in this pass:
  - fresh direct-IP SSH from this sandbox failed with:
    - `Operation not permitted`
- Therefore:
  - this pass further tightened the Marlin gate
  - but it did not launch the route

## 2026-04-09 Marlin Route Preflight Tightened Around Local GPTQ

- Fresh quant delivery in this pass promoted:
  - `W4A16 + GPTQ + Marlin (local-GPTQ-first)` as the first precision route to try after the stock official run
- New concrete handoff:
  - `ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && export SOAR_GPTQ_LOCAL_PATH=/root/autodl-tmp/models-gptq-w4a16-v19 && bash /root/autodl-tmp/codex-drafts/w4a16-marlin-draft/prepare_model.sh --input /root/autodl-tmp/models --output /root/autodl-tmp/models-gptq-marlin-local-prepared'`
- Cheap preflight completed in this pass:
  - confirmed remote local GPTQ checkpoint exists and is structurally complete:
    - `/root/autodl-tmp/models-gptq-w4a16-v19`
    - includes `quantize_config.json`
    - includes two `.safetensors` shards
  - confirmed remote draft path exists:
    - `/root/autodl-tmp/codex-drafts/w4a16-marlin-draft`
  - confirmed remote `prepare_model.sh` syntax OK
  - confirmed remote `preprocess_model.py` `py_compile` OK
- Submission/runtime risk tightened in this pass:
  - before:
    - `preprocess_model.py` mostly depended on one external `SOAR_GPTQ_LOCAL_PATH`
  - after:
    - it auto-detects a small ordered set of known local GPTQ candidate paths
    - it prints the searched paths on failure
    - `prepare_model.sh` enables `/etc/network_turbo` and `HF_ENDPOINT=https://hf-mirror.com` before any fallback download path
- Synced files:
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/preprocess_model.py`
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/prepare_model.sh`
  - deployed to:
    - `/root/autodl-tmp/codex-drafts/w4a16-marlin-draft/`
- Therefore:
  - the Marlin route is better prepared for the first post-official precision pass
  - but it is still gated on the active stock official eval finishing

## 2026-04-09 Marlin Route Blocked By Lost Run-State, Not By Route Logic

- Fresh quant delivery in this pass reaffirmed:
  - `gptq_marlin` remains the first precision route
- New blocker in this pass:
  - the expected `tmux` windows:
    - `stock-official-eval`
    - `quant-v2-queue`
  - were both missing on the remote host
- Therefore:
  - this pass could not confirm whether the stock official run ended cleanly
  - this pass could not confirm whether the post-official queue already moved
  - the Marlin route was not launched
- Fresh quant follow-up delivery in this pass:
  - first rediscovery command before any Marlin action:
    - `ssh rtx6000-2 'cd /root/autodl-tmp/SOAR-Toolkit/test_results && ls -td official_stock_base_* gptq_marlin_* nvfp4_* candidate_benches/* 2>/dev/null | head -n 20'`

## 2026-04-09 Marlin Route Held Ready After Runner Preflight

- Fresh quant delivery in this pass:
  - keep the first Marlin validation as:
    - `python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py --label gptq-marlin-fast-v1 --model-path /root/autodl-tmp/models-gptq-marlin-local-prepared --quantization gptq_marlin --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --profile fast-eval --disable-radix-cache --disable-cuda-graph --dense-as-sparse`
- Cheap preflight completed in this pass:
  - local runner `py_compile` passed
- Therefore:
  - this pass did not change route order
  - it only marked the runner patch as locally runnable for the next live Marlin proxy gate
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Marlin Route Frozen Pending Read-Only Metadata Inspection

- Fresh quant delivery in this pass:
  - keep the prepared route untouched:
    - `/root/autodl-tmp/models-gptq-marlin-local-prepared`
- Cheap preflight attempted in this pass:
  - read-only inspection of:
    - `config.json`
    - `quantize_config.json`
    - `model.safetensors.index.json`
  - goal:
    - distinguish `serve-arg mismatch` from `checkpoint/schema mismatch`
- Result:
  - blocked before execution by sandbox SSH restrictions:
    - `Operation not permitted`
- Therefore:
  - do not relaunch `Marlin`
  - do not patch metadata blindly
  - next Marlin action stays:
    - one read-only metadata/schema check after trusted SSH is available again
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Marlin Route Still Frozen While Medium-Eval State Is Untrusted

- Fresh quant delivery in this pass:
  - no change to the `Marlin` route:
    - keep `/root/autodl-tmp/models-gptq-marlin-local-prepared` untouched
    - keep the next route action as one read-only metadata/schema inspection only
- Cheap preflight attempted in this pass:
  - read the active `medium-eval` state before any more route action
- Result:
  - blocked before execution by sandbox SSH restrictions:
    - `Operation not permitted`
- Therefore:
  - `Marlin` remains frozen from this pass
  - no route re-ranking or metadata patching happened
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Marlin Route Remains Frozen After Local Readiness-Only Pass

- This pass did not touch remote Marlin artifacts.
- The next Marlin action is still exactly one read-only inspection on:
  - `/root/autodl-tmp/models-gptq-marlin-local-prepared`
- Files to inspect when trusted SSH returns:
  - `config.json`
  - `quantize_config.json`
  - `model.safetensors.index.json`
- This route was not reranked in this pass because:
  - the only preflight spent in this pass was the local `medium-eval` readiness patch
  - no trusted remote schema read occurred
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes
