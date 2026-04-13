# Clone Server Handoff

Date: 2026-04-09

Note: if you see older aliases for the quick-eval gate in historical notes, treat them all as the current single `fast` gate at `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`.

## Current Live Handoff (2026-04-13 19:30 CST)

This section supersedes older notes below for the current active quantization run.

### Which server is current

- Current working clone host is not the old `rtx6000-2` note below.
- Current active SSH target:
  - `ssh -p 14968 root@connect.bjb1.seetacloud.com`
- Local alias `rtx6000-1` was repointed to this clone host.

### Current live quantization run

- tmux session:
  - `codex-soar`
- tmux window:
  - `sel-noattn-hybrid16g`
- log:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/selective_noattn_skipdown151617_hybrid16g_20260413_194500_quant.log`
- output target:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-hybrid16g-v1`
- active command:
  - `python -X faulthandler /root/autodl-tmp/codex-drafts/quantize_gptq_w4a16.py --model-path /root/autodl-tmp/models --calibration-path /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_noattn_strongerclose49k_v1.jsonl --output-path /root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-hybrid16g-v1 --num-samples 100 --dtype float16 --dynamic-config-path /root/autodl-tmp/codex-drafts/configs/selective_marlin/noattn-skipdown151617-gs128.json --desc-act --damp-percent 0.10 --offload-to-disk --vram-strategy exclusive --gc-mode on_stage_end --gpu-cache-max-gib 16`

### Why this run matters

- Earlier full-CPU output cache solved GPU OOM but pushed host RSS above `107 GiB`, then the process was killed with `SIGKILL`.
- This new route uses a hybrid replay cache:
  - keep a bounded subset of replay outputs on GPU
  - spill the remainder to CPU
- The implementation is local and already synced to remote:
  - `/Users/ql/cursor/openbmb/scripts/quantize_gptq_w4a16.py`
  - `/Users/ql/cursor/openbmb/scripts/remote_quant_gptq_py310.sh`
  - remote copies live under `/root/autodl-tmp/codex-drafts/`

### Current health of the hybrid16g run

- While this handoff was written, the process was still alive and had progressed through layers `0..5`.
- Latest visible progress:
  - `Quantizing layer 5 of 31`
- Key heartbeats:
  - `19:19:53`: GPU `0.84 GiB`, RSS `41.65 GiB`, cgroup current `42.68 GB`
  - `19:20:53`: GPU `23.55 GiB`, RSS `30.27 GiB`, cgroup current `31.71 GB`
  - `19:21:53`: GPU `52.99 GiB`, RSS `31.00 GiB`, cgroup current `32.62 GB`
  - `19:22:54`: GPU `50.71 GiB`, RSS `51.27 GiB`, cgroup current `48.72 GB`
- This is much healthier than the previous all-CPU-cache run:
  - GPU is actually being used
  - host RAM has not re-entered the previous `100+ GiB` kill zone

### Stronger calibration currently in use

- local artifact:
  - `/Users/ql/cursor/openbmb/tmp/calibration_audit/calibration_gptq_w4a16_noattn_strongerclose49k_v1.jsonl`
- remote artifact:
  - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_noattn_strongerclose49k_v1.jsonl`
- audit summary:
  - `/Users/ql/cursor/openbmb/tmp/calibration_audit/noattn_strongerclose49k_v1.summary.json`
- composition:
  - total `100`
  - `soar_public = 64`
  - `soar_chat_close = 24`
  - `open_long_semantic_tail = 12`
- length buckets:
  - `0-4K = 20`
  - `16K-32K = 30`
  - `32K-128K = 50`

### Dynamic / no-attn config in use

- local:
  - `/Users/ql/cursor/openbmb/configs/selective_marlin/noattn-skipdown151617-gs128.json`
- remote:
  - `/root/autodl-tmp/codex-drafts/configs/selective_marlin/noattn-skipdown151617-gs128.json`
- effective idea:
  - skip all attention linear quantization
  - skip `down_proj` on layers `15/16/17`
  - quantize remaining MLP modules at `W4A16 gs=128`

### Critical script changes already made

1. `quantize_gptq_w4a16.py`
- supports:
  - `--gpu-cache-max-gib`
  - `--offload-to-disk-path`
- still supports:
  - `--cpu-cache-layer-outputs`
  - `--desc-act`
  - `--damp-percent`
  - `--dynamic-config-path`

2. `remote_quant_gptq_py310.sh`
- now records:
  - `python_pid`
  - `python_exit_code`
  - `python_exit_signal`
  - per-minute heartbeats
  - `memory.current`
- also supports:
  - `OFFLOAD_TO_DISK_PATH`

### System disk issue and cleanup

- The clone server root overlay is only `30G`.
- GPTQModel auto-created relative offload dirs under:
  - `/root/gptqmodel_offload`
- That directory had grown to about `16 GiB`, pushing `/` to `81%`.
- During this conversation, stale offload dirs were deleted and only the active one was kept:
  - active dir:
    - `/root/gptqmodel_offload/vjjihpmw-yxejlwbe`
- Result:
  - root usage dropped from `81%` to `33%`
- Long-term fix is already prepared locally:
  - future runs can pass an explicit offload root on `/root/autodl-tmp`
  - example target:
    - `/root/autodl-tmp/gptqmodel_offload`

### Previous failure mode to remember

- The prior stronger-calibration no-attn run without hybrid GPU cache died late around layer `28/29`.
- Improved wrapper proved:
  - exit code `137`
  - signal `9`
  - `oom=0`
  - `oom_kill=0`
- Best interpretation:
  - external/platform kill under host memory pressure
  - not Python exception
  - not CUDA OOM

### Most important files to read in a fresh conversation

- `/Users/ql/cursor/openbmb/worklog.md`
- `/Users/ql/cursor/openbmb/quantization-log.md`
- `/Users/ql/cursor/openbmb/clone-server-handoff.md`
- `/Users/ql/cursor/openbmb/tmp/calibration_audit/noattn_strongerclose49k_v1.summary.json`
- remote live log:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/selective_noattn_skipdown151617_hybrid16g_20260413_194500_quant.log`

### Best next actions for the new conversation

1. First check whether the hybrid16g run is still alive or has finished:
   - `ssh -p 14968 root@connect.bjb1.seetacloud.com`
   - `tmux attach -t codex-soar`
   - `select-window -t sel-noattn-hybrid16g`

2. If it is still running:
   - do not launch another heavy GPU job
   - just monitor:
     - `python_exit_code`
     - latest layer number
     - heartbeat GPU / RSS / `memory.current`

3. If it finishes successfully:
   - verify:
     - `saved_quantize_config True`
     - `done`
     - output dir exists
   - then run only a cheap serve / fast smoke before any packaging

4. If it fails:
   - inspect the last heartbeats and exit code first
   - do not assume OOM without checking:
     - `python_exit_signal`
     - `memory.events`
     - `memory.current`

5. If another run is needed:
   - keep hybrid cache
   - set explicit offload root on `/root/autodl-tmp`
   - do not go back to full CPU output cache unless necessary

## Environment Split

- Local Mac handoff/scripts/logs:
  - `/Users/ql/cursor/openbmb`
- Codex skill home:
  - `/Users/ql/.codex/skills`
- Remote GPU workspace:
  - `rtx6000-2:/root/autodl-tmp`

Read `/Users/ql/cursor/openbmb/environment-map.md` first before acting on any path in this note.

## Hard Rule: Quant Env And Eval Env Must Stay Split

- Do not reuse one Python environment for both GPTQ quantization and SGLang serving/eval.
- Keep these roles separate on `rtx6000-2`:
  - quantization env:
    - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env`
    - purpose:
      - `transformers 5.5.0`
      - `gptqmodel`
      - `flash_attn`
      - `infllm_v2`
      - GPU-side GPTQ only
  - eval env:
    - `/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env`
    - purpose:
      - `transformers 4.57.1`
      - `sglang`
      - `flash_attn`
      - `infllm_v2`
      - serving / `fast` / `medium` eval only
- Why this is mandatory:
  - `gptqmodel` currently wants `transformers >= 5.2`
  - the evaluation machine base stack and SGLang runtime are built around `transformers 4.57.1`
  - trying to collapse both into one env repeatedly caused runtime or save/load incompatibilities
- Operational rule:
  - `prepare_model` / local GPTQ experiments -> use the quant env
  - `launch_server`, `mini_test.sh`, `eval_model.py`, `fast`, `medium` -> use the eval env

## Current Remote State

- Remote host:
  - preferred direct SSH:
    - `ssh -p 50884 -i /Users/ql/.ssh/id_ed25519 root@106.120.183.117`
  - alias `ssh rtx6000-2` exists locally, but DNS has been flaky
- Remote workspace root:
  - `/root/autodl-tmp/SOAR-Toolkit`
- Shared env entry:
  - `source /root/autodl-tmp/use_soar_uv.sh`
- Current official eval script on remote:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_model.py`
  - content was checked against upstream `OpenBMB/SOAR-Toolkit` and matched
  - remote file sha256:
    - `ba2d7c6f5e13d6dc70ba8a95b25fd8a8da9e9aa0be0de20afd789cee4abb4d7b`

## Medium Eval

- New medium subset:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.jsonl`
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.meta.json`
- Shape:
  - `50` samples total
  - `10` per task:
    - `mcq`, `qa`, `niah`, `fwe`, `cwe`
  - bucket mix:
    - `0-4K`: `10`
    - `16K-32K`: `12`
    - `32K-128K`: `28`
- Remote launcher script:
  - `/root/autodl-tmp/SOAR-Toolkit/run_medium_eval.sh`
- Local source of that launcher:
  - `/Users/ql/cursor/openbmb/scripts/run_medium_eval_remote.sh`
- tmux target:
  - session: `codex-soar`
  - window: `medium-eval`
- Current run dir:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/medium_eval_stock_base_20260409_104835`

## Local Automation

- The old LaunchAgent was local-Mac orchestration only.
- It ran `codex exec resume ...` against the same conversation every 6 minutes.
- It wrote logs under:
  - `/Users/ql/cursor/openbmb/automation/logs/`
- It is now stopped and disabled:
  - `/Users/ql/Library/LaunchAgents/com.ql.codex.soar.resume30m.plist.disabled`

## Most Important Operator Mistakes To Avoid

These were real mistakes made in this thread and should be treated as known failure modes.

1. Do not inline multi-line remote Python or shell through nested `ssh 'python - <<PY ...'` unless absolutely necessary.
   - Local shell expansion repeatedly corrupted quotes and paths.
   - Safer pattern:
     - write a local script
     - `scp` it to remote
     - run it remotely

2. Do not trust `tmux new-window` success by itself.
   - Verify both:
     - window exists in `tmux list-windows`
     - expected run dir or log file exists on disk

3. Always capture stderr for eval runs.
   - Use:
     - `... 2>&1 | tee <log>`
   - Earlier full official eval only logged stdout, so if `eval_model.py` threw on stderr, the log hid it.

4. Do not run `py_compile` on shell scripts.
   - This happened with:
     - `/Users/ql/cursor/openbmb/soar-patches/mini_test.sh`
   - Use:
     - `bash -n` for shell
     - `python3 -m py_compile` only for Python

5. Be careful with local expansion inside quoted remote commands.
   - Repeated bad pattern:
     - `$(...)` expanded locally when it was intended for remote
   - Prefer:
     - `scp` + remote script
     - or here-doc with no local interpolation

6. Do not assume `official` finished just because the server kept decoding.
   - The earlier stock official full eval never wrote summary files.
   - The output dir stayed empty.
   - The most likely cause was that the client process was interrupted before score/save.

## Key Project Insights

1. `fast_test.sh` is smoke only.
   - It is not a precision source of truth.
   - Earlier low scores on base and quant were not enough to decide route quality.

2. `fast` is the intended local directional gate.
   - It is derived from the validated medium-eval subset.
   - Dataset:
     - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
   - Runner:
     - `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`

3. `fast` must keep task-aware output caps.
   - Earlier quick-eval variants inherited runaway output budgets.
   - Current intended caps:
     - `mcq=64`
     - `qa=128`
     - `niah=128`
     - `fwe=256`
     - `cwe=256`

4. `gptq_marlin` remains the highest-upside precision route.
   - Prepared model path:
     - `/root/autodl-tmp/models-gptq-marlin-local-prepared`
   - Local GPTQ source path:
     - `/root/autodl-tmp/models-gptq-w4a16-v19`

5. Current Marlin blocker is runtime/config compatibility, not just missing files.
   - Observed blockers:
     - `ValueError: Quantization method specified in the model config (gptq) does not match ... (gptq_marlin)`
     - with `--quantization gptq`, later:
       - `KeyError: 'model.layers.1.self_attn.z_proj.weight'`
   - Interpretation:
     - `gptq_marlin` is not yet proven runnable end-to-end on current remote stack
     - next useful step is metadata / schema inspection, not blind reruns

6. Prepared fallback route stays:
   - `NVFP4 mlp_weight_only + bf16 + 32K + 64`

## Precision Route Order

1. `gptq_marlin`
2. `NVFP4 mlp_weight_only + bf16 + 32K + 64`
3. only then consider more conservative fallback

## Full Official Eval Status

- Earlier stock full eval run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- It did not produce a valid final result.
- Why:
  - `eval_model.log` stopped before score/save
  - output dir stayed empty:
    - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260409_034351`
  - server kept decoding until roughly `2026-04-09 04:44:51`
  - most likely the client process was interrupted before finishing
- Time estimate for full `150`:
  - treat as `80-90 min`
  - reserve `2h` isolated window

## Recommended Next Commands On A Fresh Conversation / Clone Server

1. Verify medium eval is alive:
   - `ssh rtx6000-2 -t 'tmux attach -t codex-soar \; select-window -t medium-eval'`
   - or:
   - `ssh rtx6000-2 'tail -f /root/autodl-tmp/SOAR-Toolkit/test_results/medium_eval_stock_base_20260409_104835/eval_model.log'`

2. If medium eval finishes cleanly, use it as the local control anchor before any new quant run.

3. For any new remote workflow longer than one line:
   - edit local script under `/Users/ql/cursor/openbmb/scripts/`
   - `scp` it to remote
   - execute it from `tmux`

4. When re-running official or medium eval:
   - always log with `2>&1 | tee`
   - do not start any second heavy GPU job

## Useful Local Files

- `/Users/ql/cursor/openbmb/worklog.md`
- `/Users/ql/cursor/openbmb/quantization-log.md`
- `/Users/ql/cursor/openbmb/calibration-handoff.md`
- `/Users/ql/cursor/openbmb/marlin-route-log.md`
- `/Users/ql/cursor/openbmb/sgl-optimization-log.md`
- `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
- `/Users/ql/cursor/openbmb/scripts/run_fast_eval_remote.sh`
- `/Users/ql/cursor/openbmb/scripts/run_quant_fast_queue.sh`
- `/Users/ql/cursor/openbmb/scripts/run_medium_eval_remote.sh`
- `/Users/ql/cursor/openbmb/soar-patches/mini_test.sh`
- `/Users/ql/cursor/openbmb/environment-map.md`

## Current Package And Read

- Current local draft tarball:
  - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft.tar.gz`
  - sha256:
    - `1ed9ec0e90fad4f3c6cbe27e0370d672bf1c3bda9640b6b027dca00200efe61d`
- Packaging note:
  - reused the local `flash_attn` wheel cache
  - did not redownload `flash_attn`
- Current route read:
  - `gptq_marlin` is slightly better than stock on bounded fast and medium accuracy
  - raw TPS is about flat
  - the main remaining weakness is stopping / over-generation on uncapped medium

## Submission Packaging State

- `w4a16-marlin-draft` was repaired after a failed submission that still depended on a dev-box-only GPTQ checkpoint.
- The draft now quantizes on the evaluation GPU from `/models/MiniCPM-SALA`.
- Bundled calibration files inside the draft:
  - `calibration_gptq_w4a16_true160k_v1.jsonl`
  - `calibration_gptq_w4a16_pg19_v2.jsonl`
- The intended attempt order is:
  1. `true160k_fp16_32`
  2. `pg19_32k_fp16_64`

## Remote Runtime Cleanup

- `rtx6000-2` no longer keeps a stray `uv` binary under `/root/miniconda3/bin/uv`.
- `/root/autodl-tmp/use_soar_uv.sh` was simplified and now only exposes:
  - `soar_python`
  - `soar_pip`
- Treat the shared runtime as a normal Python venv helper, not as a uv-managed helper.
- The local Codex network skill `autodl-network-turbo` was intentionally removed.

## Minimal Eval-Machine Probe

- A deliberately failing minimal submission now exists at:
  - `/Users/ql/cursor/openbmb/submission-drafts/eval-env-probe`
- Packaged tar:
  - `/Users/ql/cursor/openbmb/submission-drafts/eval-env-probe.tar.gz`
- sha256:
  - `1dc99bf0fd8fa08472aaf20023d530fc43158ed254ea6815b1a382a5222771f1`
- Purpose:
  - print targeted evaluation-machine environment details into the last 50 log lines
  - then fail intentionally in `prepare_model.sh`
- Use this before further GPTQ submission debugging if the official environment remains ambiguous.
