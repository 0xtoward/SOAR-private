# SGL Optimization Log

Note: historical `fast`, `fast`, and `v2` labels below are obsolete. The only active quick gate is `fast`, backed by `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`.

## Environment Note

- Local runner scripts / notes:
  - `/Users/ql/cursor/openbmb`
- Remote SGLang runtime under test:
  - `rtx6000-2:/root/autodl-tmp/sglang`
- Remote benchmark artifacts:
  - `rtx6000-2:/root/autodl-tmp/SOAR-Toolkit/test_results`

## Scope

- Focus: SGLang-side runtime and kernel-path optimization for MiniCPM-SALA.
- Goal: benchmark and compare multiple optimization ideas beyond the current NVFP4 calibration work.
- Rule: each attempt should record the exact server args, code/path changes, benchmark command, and outcome.

## Baseline Facts

- Remote repo: `/root/autodl-tmp/sglang`
- Current branch: `soar-pre-awq-backup-20260325`
- Current working tree diff is minimal:
  - `python/sglang/srt/utils/common.py` has a local warning-only version-check patch
  - `python/sglang/srt/models/minicpm.py` already matches the previously noted SOAR custom branch state
- Relevant history seen in `git log`:
  - `backup MiniCPM SOAR patches before AWQ`
  - `dense_as_sparse`
  - multiple flashinfer / graph / sparse cache fixes

## Bench Policy

- `fast_test.sh` bench half is only a smoke signal and is not enough for serious speed decisions.
- Prefer `bench_serving.sh` style measurements with a fixed local proxy speed dataset.
- Keep local benchmark settings and datasets stable across attempts so relative comparisons remain meaningful.
- Current local bench tiers:
- `smoke`: 5 requests, very fast, useful for daily regression only
- `proxy11`: 11 requests, built from the existing smoke+mini sets, intended to be more representative without becoming as harsh as the full mini tail
- `mini`: 15 requests, stronger stress test, but currently harsher than the official public distribution and not cheap enough for every iteration

## Attempt Log

### Attempt 0: Audit Current Fusion State

- Status: confirmed
- What I checked:
  - `git log` for `minicpm.py`, `minicpm_backend.py`, `layernorm.py`
  - working-tree diff for likely fusion-related files
- Current read:
  - the main MiniCPM SOAR performance patches appear to already be baked into branch history, not pending in the worktree
  - this suggests we should treat the current code as an already-fused baseline, then test deltas on top
- Concrete diff vs `minicpm.py.orig`:
  - removed the `q.float()/k.float()` RoPE round-trip and cast-back path
  - switched decoder blocks to fused `residual + RMSNorm` usage
  - final norm also consumes the carried residual path

### Attempt 0.1: Curated Calibration Input For Quant Experiments

- Status: confirmed
- New calibration file:
  - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_v1.jsonl`
- Mix:
  - `96` SOAR public-style prompts
  - `16` `cnn_dailymail`
  - `16` `wikitext`
- Task balance:
  - `mcq=20`
  - `qa=30`
  - `niah=30`
  - `fwe=8`
  - `cwe=8`
  - `open_text=32`
- Length policy:
  - token-aware slicing with `max_sample_tokens=8192`
  - long samples use `head + middle + tail`
  - tail region is intentionally weighted more heavily to preserve the actual question/instruction region

## Planned Attempt Slots

1. Baseline current server args and benchmark.
2. Attention backend variant.
3. Chunked prefill tuning.
4. CUDA graph toggle validation.
5. `dense-as-sparse` validation.
6. Radix cache / cache-path validation.
7. Additional fusion or kernel-path change if still uncovered.

### Attempt 1: Base FP16 + `--fuse-topk` Smoke A/B

- Status: blocked before launch
- Why this was chosen:
  - `sgl-optimization-candidates.md` ranks `--fuse-topk` as the first low-cost, high-signal runtime candidate
  - it is a launch-arg-only delta with no code edits
  - it can be tested on the base FP16 route to separate SGL runtime effects from NVFP4 calibration noise
- Planned server delta vs current stable base serve:
  - keep:
    - `--model-path /root/autodl-tmp/models`
    - `--dtype float16`
    - `--attention-backend minicpm_flashinfer`
    - `--chunked-prefill-size 8192`
    - `--disable-radix-cache`
    - `--disable-cuda-graph`
    - `--dense-as-sparse`
  - add:
    - `--fuse-topk`
- Planned benchmark command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --profile smoke
```

- Preflight blocker:
  - remote reachability failed before GPU inspection
  - `ssh rtx6000-2` returned:

```text
ssh: Could not resolve hostname connect.bjb1.seetacloud.com: -65563
```

  - because of that, I could not:
    - inspect `/root/autodl-tmp/SOAR-Toolkit/test_results`
    - verify `nvidia-smi` idleness
    - check whether the earlier quantization tmux job had already exited
- Outcome:
  - no benchmark was launched
  - no new speed or smoke-quality metrics were collected
  - the single next experiment remains `--fuse-topk` on base FP16 once remote access works again

### Attempt 1.1: Bench Harness Upgrade

- Status: ready
- What changed:
  - `run_remote_candidate_bench.py` now starts the server with the explicit uv-environment `python3`
  - this avoids the earlier failure mode where `bash -lc` resolved `python` to `/root/miniconda3/bin/python` instead of the shared uv runtime
  - added `proxy` profile backed by a deterministic remote dataset:
    - `/root/autodl-tmp/SOAR-Toolkit/bench_speed_v2_proxy11.jsonl`
- Why `proxy11` exists:
  - current `smoke` is only `5` rows
  - current `mini` is `15` rows but contains an over-harsh tail, including prompts beyond the official `160K` public length cap
  - with the current eligible local pool, `11` rows is a better fit than `12` because the official `128K-160K` bucket only has one valid candidate
  - `proxy11` targets input-bucket counts of `3/1/2/4/1` for `0-4K / 4-16K / 16-32K / 32-128K / 128-160K`
  - deterministic row selection:
    - smoke rows `1,3,4,5`
    - mini rows `1,3,4,6,8,11,14`

### Attempt 1.2: Base FP16 `speed-smoke` Harness Check

- Status: running, first hard result confirmed
- Command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-speed-smoke-check \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile speed-smoke
```

- Confirmed facts:
  - server now launches under the explicit uv python:
    - `/root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3`
  - `bench_serving.sh` no longer falls back to `/usr/bin/python3`
  - `S1` is no longer `0`
- First live result:
  - `S1 benchmark duration = 189.13s`
- Current read:
  - bench is now genuinely usable for speed comparisons
  - the remaining work is about tier design and runtime comparison quality, not basic harness breakage

### Attempt 1.3: Remote Reachability Retest and SSH Retry Visibility

- Status: no remote run launched
- Why this step:
  - before spending GPU time on `--fuse-topk`, I needed to re-check whether the remote host was actually reachable and idle
  - earlier runs were blocked by resolver failures against `connect.bjb1.seetacloud.com`
- What I observed:
  - remote reachability is flapping
  - one direct SSH probe succeeded and returned the expected host:

```text
connected
autodl-container-u6thunte83-2b3090a0
Thu Apr  9 02:49:30 CST 2026
```

  - immediately after that, the same host alias failed again with:

```text
ssh: Could not resolve hostname connect.bjb1.seetacloud.com: -65563
```

- Harness hardening:
  - `run_remote_candidate_bench.py` already had transient SSH retries
  - I expanded the retry matching to include macOS resolver text:
    - `Name or service not known`
    - `nodename nor servname provided`
  - I also added retry logging so automation output now clearly distinguishes transient SSH/DNS failure from model or benchmark failure
- Validation:
  - `python3 -m py_compile /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - light probe through the runner's own `ssh()` helper produced:

```text
[bench] ssh retry 1/4 after transient failure: ssh: Could not resolve hostname connect.bjb1.seetacloud.com: nodename nor servname provided, or not known
[bench] ssh retry 2/4 after transient failure: ssh: Could not resolve hostname connect.bjb1.seetacloud.com: nodename nor servname provided, or not known
[bench] ssh retry 3/4 after transient failure: ssh: Could not resolve hostname connect.bjb1.seetacloud.com: nodename nor servname provided, or not known
returncode=255
```

- Decision:
  - do not launch `base-fp16-fuse-topk-smoke` yet
  - the harness is ready, but remote idleness cannot be re-verified while DNS is unstable
- Next step:
  - as soon as SSH stabilizes, run the planned `base-fp16-fuse-topk-smoke` candidate unchanged

### Attempt 1.3: Base FP16 `--fuse-topk` smoke preflight and SSH hardening

- Status: benchmark launch blocked by intermittent local DNS failure
- Chosen command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --profile smoke
```

- What was verified before launch:
  - remote host was reachable once
  - `nvidia-smi` reported the GPU idle at that moment:
    - `NVIDIA RTX PRO 6000 Blackwell Server Edition, 0 MiB, 0 %`
  - no active `sglang.launch_server`, `bench_serving`, `eval_model.py`, or `fast_test.sh` jobs were found
- Harness fix applied before the retry:
  - `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py` now wraps every SSH call with:
    - `ConnectTimeout=10`
    - up to `4` attempts
    - retry only on clear transport-resolution failures such as `Could not resolve hostname`
  - local syntax check passed:
    - `python3 -m py_compile /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
- Actual blocker on retry:
  - the local machine then lost DNS resolution for `connect.bjb1.seetacloud.com`
  - repeated direct checks returned:

```text
ssh: Could not resolve hostname connect.bjb1.seetacloud.com: nodename nor servname provided, or not known
```

  - because the SSH transport never became stable again, the runner failed during `start_server()` before any remote server, eval, or bench logs were created
- Result:
  - no runtime metric was collected for `--fuse-topk`
  - no second heavy GPU task was started
  - the next runtime experiment remains the exact same `--fuse-topk` smoke command above
- Next action:
  - rerun the same command as soon as DNS resolution for `rtx6000-2` is healthy again

### Attempt 1.4: Preflight-Only Path and Transport Classification

- Status: ready and validated
- Why this change:
  - the runtime candidate itself is still the same `base-fp16-fuse-topk-smoke`
  - the main blocker was transport ambiguity and the lack of a cheap first-step check before launching a heavy run
- Harness hardening:
  - `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py` now classifies SSH/transport failures into explicit categories such as:
    - `dns_resolution`
    - `connect_timeout`
    - `connection_reset`
    - `auth_or_hostkey`
    - `transport_other`
    - `remote_command_failure`
  - the top-level runner now prints:
    - `transport_classification=...`
    - `transport_detail=...`
  - the runner now performs a cheap remote preflight before any heavy run starts
  - a dedicated `--preflight-only` mode now exists so the transport and remote-idle check can be run by itself
- Preflight behavior:
  - verifies SSH reachability
  - checks remote hostname and timestamp
  - checks GPU memory usage and utilization
  - checks for active heavy jobs matching:
    - `sglang.launch_server`
    - `bench_serving`
    - `eval_model.py`
    - `fast_test.sh`
  - marks the host idle only when:
    - GPU memory used is `<= 256 MiB`
    - GPU utilization is `<= 5%`
    - no heavy jobs are active
- Validation:
  - `python3 -m py_compile /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - validated cheap preflight command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke-preflight \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --profile smoke \
  --preflight-only
```

- Validation result:

```text
[bench] preflight transport=ok remote_idle=True host=autodl-container-u6thunte83-2b3090a0 time=2026-04-09_03:03:18 gpu_mem_mib=0 gpu_util_pct=0 active_jobs=0
```

- Exact next runtime command after preflight succeeds:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --profile smoke
```

### Attempt 1.5: `fuse_topk` backend guard and explicit batch-cap handoff

- Status: runtime-side fix prepared locally
- Concrete blocker from real run:
  - the actual `base-fp16-fuse-topk-smoke` run reached model load and failed inside `minicpm_backend.py`
  - with `fuse_topk=True`, the backend tried to JIT fused kernels with:

```python
for bs in range(1, model_runner.server_args.max_running_requests + 1):
```

  - but `model_runner.server_args.max_running_requests` was `None`, causing:

```text
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
```

- Why the backend should be fixed:
  - `max_running_requests=None` is legal on the normal path in SGLang
  - `server_args.py` only force-fills `48` for some speculative modes, not for the standard MiniCPM-SALA path
  - so `minicpm_backend.py` must not assume the field is always populated
- Runtime-side fix applied:
  - both local SGLang draft copies of `minicpm_backend.py` now:
    - read `max_running_requests` into `self.max_fused_batch_size`
    - if `fuse_topk=True` but `max_running_requests is None`, print a warning and disable `fuse_topk` instead of crashing
    - only JIT fused kernels when `self.fuse_topk` is still enabled
  - the bench runner now accepts and forwards an explicit:
    - `--max-running-requests`
- Why both layers matter:
  - backend fix prevents a hard crash and makes the runtime code safe
  - explicit `--max-running-requests 48` is still needed when we actually want to benchmark the fused path rather than silently falling back to non-fused execution
- Validation:
  - `python3 -m py_compile /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - `python3 -m py_compile /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`
  - `python3 -m py_compile /Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`
- Exact rerun command after fix:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --max-running-requests 48 \
  --profile smoke
```

### Attempt 1.6: Second `fuse_topk` blocker and runtime decision

- Status: do not demote yet; patch the local callsite drift and give `fuse_topk` one final bounded retry
- New blocker from real run:
  - after passing `--max-running-requests 48`, the server got deeper into MiniCPM sparse backend init and failed in the TileLang JIT path with:

```text
TypeError: fused_attn_pooling_online_topk_prefill() missing 2 required positional arguments: 'actual_max_seqlen_q' and 'actual_max_seqlen_k'
```

- Analysis:
  - this is a local callsite/signature mismatch, not evidence that the optimization idea itself is invalid
  - in the local `minicpm_fuse_kernel.py`, `actual_max_seqlen_q` and `actual_max_seqlen_k` appear only in the function signature and docstring
  - they are not referenced in the current kernel body
  - that makes this a bounded local bug worth patching once
- Runtime decision:
  - keep `fuse_topk` as the immediate next runtime candidate for one more retry
  - if the next run fails again inside deeper TileLang/JIT logic, stop spending time on it and demote `fuse_topk`
  - at that point, move to `split_stage1` as the next clean launch-arg candidate
- Local fix state:
  - both local SGLang draft copies of `minicpm_backend.py` now pass the required signature arguments for `fused_attn_pooling_online_topk_prefill`
  - the values wired in are:
    - `actual_max_seqlen_q=model_runner.server_args.chunked_prefill_size`
    - `actual_max_seqlen_k=max_cache_len`
  - added a clarifying comment that these are currently signature-only in the local TileLang fork
- Exact next runtime experiment after this decision:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke-mrr48 \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --max-running-requests 48 \
  --profile smoke
```

- Fallback if this still fails:
  - immediately demote `fuse_topk`
  - next runtime candidate becomes:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-split-stage1-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --split-stage1 \
  --profile smoke
```

### Attempt 1.6: `fuse_topk` prefill JIT call-path patch

- Decision:
  - choose `1) patch the fused call path if the fix is local and bounded`
  - do not demote `fuse_topk` yet
- Why this is still patchable:
  - the second blocker is not a transport issue and not a broad kernel-correctness failure yet
  - it is a local call-site/signature mismatch:
    - `fused_attn_pooling_online_topk_prefill(...)` now requires:
      - `actual_max_seqlen_q`
      - `actual_max_seqlen_k`
    - `minicpm_backend.py` was still calling it without those arguments
  - from the current local code, those two static parameters are declared in `minicpm_fuse_kernel.py` but are not yet referenced inside the kernel body
  - that makes this a bounded compatibility patch rather than an open-ended algorithm rewrite
- Runtime-side fix applied:
  - in both local draft copies of `minicpm_backend.py`, the fused prefill JIT call now passes:
    - `actual_max_seqlen_q=model_runner.server_args.chunked_prefill_size`
    - `actual_max_seqlen_k=max_cache_len`
  - these are the most sensible static upper-bound values available at init time for the current prebuild path
- Validation:
  - `python3 -m py_compile /Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`
  - `python3 -m py_compile /Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/sglang/python/sglang/srt/layers/attention/minicpm_backend.py`
- Exact next runtime command after syncing this runtime patch into the active remote SGLang code:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke-mrr48 \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --max-running-requests 48 \
  --profile smoke
```

- Demotion threshold:
  - if the next real run gets past this signature mismatch and immediately hits a deeper TileLang/kernel correctness issue, `fuse_topk` should then be demoted behind `--split-stage1`

### Attempt 1.7: Post-smoke runtime handoff decision

- Current condition:
  - a separate heavy GPU job is still running, so runtime work stays in prep mode only
- Decision:
  - do **not** demote `fuse_topk` yet
  - there is still exactly one bounded local patch worth consuming first:
    - the prefill JIT call-path compatibility patch already prepared in local `minicpm_backend.py`
- Why this is still the right order:
  - the latest blocker is still a local call-site mismatch, not yet evidence that the fused kernels themselves are fundamentally broken on this runtime
  - the added arguments are required by the current TileLang JIT entrypoint and were simply omitted by the caller
  - from the local kernel source, those two new static parameters are declared but not yet used in the kernel body, so the patch is low-risk and bounded
- Exact next runtime candidate after the current smoke finishes:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke-mrr48 \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --max-running-requests 48 \
  --profile smoke
```

- Required local prep before that run:
  - no additional new code changes are needed beyond the already-prepared `minicpm_backend.py` patch
  - the same patch must be present in the active SGLang runtime copy used by the remote benchmark before launching the command above
- Immediate fallback if this still fails:
  - demote `fuse_topk` behind `--split-stage1`
  - then the next runtime candidate becomes:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-split-stage1-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --split-stage1 \
  --profile smoke
```

## 2026-04-09 Attempt 1.4 base FP16 fuse_topk smoke

- Closed-loop step reached:
  - worker delivery collected
  - one real experiment launched
  - blocker fed back to workers for resumed follow-up
- Exact command run:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --profile smoke
```

- Run directory:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_030449_base-fp16-fuse-topk-smoke`
- Outcome:
  - remote transport was healthy enough to launch
  - model weights loaded successfully
  - failure was in the runtime code path before `fast_eval` or `bench_smoke` produced results
- Concrete blocker from `server.log`:

```text
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
```

- Root cause:
  - with `--fuse-topk`, `MiniCPMSparseBackend` in `minicpm_backend.py` assumes `model_runner.server_args.max_running_requests` is an integer
  - current server args leave it as `None`
  - the code does `range(1, max_running_requests + 1)` and crashes during backend initialization
- Immediate next action:
  - patch the runtime side to handle `max_running_requests=None` safely, and consider whether the runner should also set an explicit fallback value for `--max-running-requests`

## 2026-04-09 Attempt 1.5 base FP16 fuse_topk smoke with explicit max_running_requests

- Exact command run:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke-mrr48 \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --max-running-requests 48 \
  --profile smoke
```

- Run directory:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_030942_base-fp16-fuse-topk-smoke-mrr48`
- Outcome:
  - remote transport was healthy
  - service got past the `max_running_requests=None` failure
  - model weights loaded successfully
  - fused JIT path still failed before `fast_eval` or `bench_smoke` produced results
- Concrete blocker from `server.log`:

```text
TypeError: fused_attn_pooling_online_topk_prefill() missing 2 required positional arguments: 'actual_max_seqlen_q' and 'actual_max_seqlen_k'
```

- Current read:
  - this is no longer a launch-arg issue
  - `fuse_topk` now looks blocked by a deeper TileLang/callsite mismatch in the local fused kernel path
  - this likely needs either a code fix in the `fuse_topk` callsite or deprioritization of `fuse_topk` in favor of the next runtime candidate

### Attempt 1.8: explicit post-bench runtime ordering

- Current condition:
  - another heavy GPU bench is still active, so this pass stays in lightweight prep mode only
- Final decision for ordering:
  - keep `fuse_topk` ahead of `split-stage1` for exactly one more rerun
  - this is justified only because the remaining blocker is still a bounded local compatibility patch, not yet a demonstrated fused-kernel correctness failure
  - if the next `fuse_topk` rerun fails again anywhere in TileLang/kernel init, demote it immediately behind `--split-stage1`
- Exact next runtime candidate after the current bench finishes:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-fuse-topk-smoke-mrr48 \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --fuse-topk \
  --max-running-requests 48 \
  --profile smoke
```

- Local prep required before that rerun:
  - no new code edits beyond the already-prepared `minicpm_backend.py` call-path patch
  - the active remote SGLang runtime copy must include that same patch before launch
- Concrete fallback if that rerun fails:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-split-stage1-smoke \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --split-stage1 \
  --profile smoke
```

### Attempt 1.9: stock-base control audit and harness gate

- Goal:
  - prepare the right next step once the near-stock base control smoke/proxy result lands
  - avoid misclassifying a bad tiny-smoke result as a runtime regression
- Current audit of local bench tiers:
  - `smoke`
    - useful for service reachability, empty-output detection, obvious stop-behavior collapse, and large regressions
    - not representative enough for quality conclusions
    - it takes only one shortest sample per task, so it is highly sensitive to sample-specific formatting quirks
  - `proxy11`
    - useful for relative speed comparison
    - closer to the official input-length distribution than `smoke`
    - still not a quality benchmark; it should not be used to explain a low score
  - `speed-smoke`
    - useful only to verify that speed measurement is alive and nonzero
    - not representative enough to choose a final runtime path by itself
- Harness symptoms to watch if stock-base smoke is also weak:
  - failures concentrated in `mcq` while the outputs are non-empty and look semantically correct
  - many short samples ending with `finish_reason=length`
  - stock-base and experimental routes both failing the same tiny smoke subset
  - healthy speed path and server stability, but poor tiny-smoke quality
- Why those are harness-shaped signals:
  - `fast_test.sh` uses a very small sample set and a narrow MCQ extractor
  - current MCQ extraction heavily prefers:
    - `ANSWER: A`
    - boxed forms like `\boxed{A}`
  - a semantically correct but differently formatted stock output can therefore look like a fail
- Most valuable next action if stock-base smoke is weak or ambiguous:
  - do not jump straight into another runtime A/B
  - first run a stock-base `mini_test.sh` quality cross-check on the same stable serve path
- Exact command prepared for that case:

```bash
ssh rtx6000-2 'source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true && nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server --model-path /root/autodl-tmp/models --trust-remote-code --host 127.0.0.1 --port 30001 --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --disable-radix-cache --disable-cuda-graph --dense-as-sparse > /root/autodl-tmp/SOAR-Toolkit/test_results/stock_base_control_server.log 2>&1 & sleep 20 && cd /root/autodl-tmp/SOAR-Toolkit && API_BASE=http://127.0.0.1:30001 TIMEOUT=0 MINI_PER_TASK=4 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 bash mini_test.sh | tee /root/autodl-tmp/SOAR-Toolkit/test_results/stock_base_control_mini.log'
```

- Why this is the best post-control step:
  - it resolves whether low stock smoke is a harness/sample artifact or a real runtime problem
  - it is still cheaper than a full 150-question public eval
  - it gives a better quality read than `smoke` without mixing in a new runtime variable

## 2026-04-09 Stock-Base Control Smoke Result

- Control setup:
  - remote editable `sglang` was switched to `/root/autodl-tmp/sglang-stock/python`
  - that stock tree was created from `git archive HEAD` and then had `minicpm.py.orig` restored
- Fast smoke result from:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_033456_stock-base-fp16-smoke-control/fast_eval.log`

```text
avg_score=28.00%
pass=1
part=1
fail=3
```

- Per-task signals:
  - `cwe=0.40`
  - `fwe=1.00`
  - `mcq=0.00`
  - `niah=0.00`
  - `qa=0.00`
- Runtime-side interpretation:
  - this matches the earlier working-tree `base-fp16` smoke headline score exactly
  - therefore the weak smoke result is not explained by our current fused `minicpm.py` delta alone
  - smoke remains too small and too format-sensitive to use as the only quality gate
  - since both stock and working-tree smoke are similarly weak, the next runtime move should not be another model-file rollback experiment
- Runtime ordering update:
  - if the ongoing stock speed smoke does not expose a transport/runtime failure, the next quality-control experiment should be the queued stock-base `mini_test.sh` command already recorded above
  - only after that control-quality read should we resume runtime A/Bs like `fuse_topk` or `split-stage1`

## 2026-04-09 Official Eval Gate After Stock Control

- Official quality script confirmed in toolkit:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_model.py`
- Relevant official-eval properties:
  - dataset-driven, not 5-sample smoke-driven
  - `/v1/chat/completions`
  - `concurrency=8`
  - `max_out_len=65536`
  - `mode='mid'`
  - `--num_samples` optional; unset means full dataset
- Runtime conclusion:
  - with stock-base `fast_test.sh` also landing at `28.00%`, `fast_test.sh` must be treated as a smoke gate only
  - for route ranking, `eval_model.py` is the quality source of truth, not another `smoke` rerun

## 2026-04-09 stock-base official eval in flight

- Current state:
  - the official stock-base eval is already running in tmux window:
    - `stock-official-eval`
  - log directory:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Runtime policy while it runs:
  - do not start any other heavy GPU task
  - treat this run as the current quality truth source for calibrating local harnesses

## 2026-04-09 local harness fix after stock-base control

- Most important lightweight local change:
  - add a `mini-eval` quality mode to `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - this allows local quality cross-checking with `mini_test.sh` through the same preflight/serve/teardown path
  - it avoids overloading `fast_test.sh --eval-only` as if it were a quality benchmark
- Why this is the right fix:
  - it is local-only and does not interfere with the current long-running official eval
  - it upgrades the local harness from:
    - `smoke only`
  - to:
    - `smoke for quick health`
    - `mini-eval` for lightweight quality cross-check
    - `official` for actual quality truth
- Validation:
  - `python3 -m py_compile /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`

## 2026-04-09 post-official lightweight cross-check

- Best next lightweight cross-validation after the official stock-base eval finishes:
  - run stock-base through the new `mini-eval` mode
  - compare the local `mini-eval` score and failure pattern with the official stock-base result
  - use that to judge whether local `mini-eval` is predictive enough for future runtime A/B triage
- Exact command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label stock-base-fp16-mini-eval \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

## 2026-04-09 Official Stock Eval Running

- Official stock-base eval is now in flight in remote `tmux`:
  - session/window: `codex-soar:stock-official-eval`
- Run directory:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Runtime implication while this run is active:
  - no new runtime A/B should be launched
  - the next runtime decision should be made only after the official stock-base quality signal lands
  - if the official stock-base eval still looks weak, prioritize a control-quality cross-check such as stock-base `mini_test.sh` before resuming `fuse_topk` or `split-stage1`

## 2026-04-09 Official Eval Running State

- Active long-running quality source of truth:
  - remote tmux window: `codex-soar:stock-official-eval`
  - run dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Confirmed launch state:
  - stock-base server is up on `127.0.0.1:30001`
  - `eval_model.py` is running on the full `150`-sample public dataset
  - `concurrency=8`
  - current policy is `no second heavy GPU job until this run ends`
- Runtime implication while it is running:
  - do not start `mini_test.sh`
  - do not resume `fuse_topk` / `split-stage1`
  - keep runtime work limited to log review and post-run preparation only
- First lightweight runtime preflight immediately after official eval completes:

```bash
ssh rtx6000-2 'RUN_DIR=$(cat /root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_latest.txt) && echo RUN_DIR=$RUN_DIR && rg -n "Average Score|Total Duration|Overall TPS|Request failed|ERROR|HTTP" "$RUN_DIR/eval_model.log" "$RUN_DIR/server.log" || true && tail -n 20 "$RUN_DIR/eval_model.log"'
```

- Why this is the right next lightweight check:
  - it tells us whether the full official eval actually completed cleanly
  - it extracts the official quality number before we spend another GPU minute
  - it cleanly gates the next branch:
    - if official stock quality is materially healthier than smoke, keep `fast_test.sh` in the smoke-only bucket and move on to route comparison
    - if official stock quality is still weak, prefer another quality-control step before runtime A/B

## 2026-04-09 Runtime Hold While Official Stock Eval Runs

- Active official-quality run:
  - `tmux` window:
    - `codex-soar:stock-official-eval`
  - run directory:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318`
- Live-status preflight during this pass:
  - `server ready`
  - `eval_model.py` started with:
    - `150` samples
    - `concurrency=8`
  - visible progress at check time:
    - `8/150`
- Runtime policy during this run:
  - do not start `mini_test.sh`
  - do not resume `fuse_topk` / `split-stage1`
  - keep runtime work limited to log review and post-run preparation only
- Immediate post-run gate remains:
  - first inspect the official-eval logs with the lightweight grep/tail preflight already recorded above
  - only after that decide between:
    - stock-base `mini_test.sh`
    - runtime A/B resumption

## 2026-04-09 Post-Official-Eval Runtime Gate

- Decision split after the official stock eval finishes:
  - if the official stock eval is technically healthy:
    - clean completion
    - `Average Score` and `Total Duration` are present
    - no obvious `Request failed`, `HTTP`, or server-side error patterns dominate the log
  - if the official stock eval is not technically healthy:
    - missing summary lines
    - request/pathology errors
    - or another obviously invalid completion state

- If technically healthy, first light verification before resuming route comparison:
  - establish one cheap control proxy on the same stable stock-base serve path using our local mini harness
  - exact command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label stock-base-fp16-mini-control \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

  - why this branch exists:
    - it gives us a cheaper, repeatable control signal after the official baseline is known-good
    - only after this mini-control lands should route comparison resumes for candidates like `gptq_marlin` or runtime A/B

- If not technically healthy, next quality-control check before any runtime A/B:
  - rerun an official-style stock-base check on a small subsample with reduced concurrency to isolate whether the problem is the full official path itself
  - exact command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label stock-base-fp16-official-subsample-c1 \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile official \
  --official-data-path eval_dataset/perf_public_set.jsonl \
  --official-max-seq-len 524288 \
  --official-concurrency 1 \
  --official-num-samples 12
```

  - why this branch exists:
    - it keeps the quality-control path close to the official script
    - it separates “official eval path itself is unhealthy” from “model baseline is genuinely weak”
    - no runtime A/B should resume until this subsample official check is understood

## 2026-04-09 full-eval queue policy for quantization-first phase

- Main-line change:
  - runtime speed work is no longer the priority
  - the main line is now:
    - quantization-route review
    - official full-quality evals for quantization candidates
  - the current stock-base official eval keeps running and must not be interrupted

- Why one SGL server should not carry two formal full official evals at the same time:
  - `eval_model.py` already drives the server with multiple concurrent requests
  - mixing two formal eval clients on one server destroys attribution:
    - latency no longer belongs to one eval run
    - failures and retries become ambiguous
    - server-side logs cannot be cleanly mapped back to one candidate
  - the request streams interfere at the scheduler and KV-cache level, so the observed duration is no longer a valid official-comparison signal
  - if one run hits a pathological sample, it can back up the shared server and distort the other run's timing and completion behavior

- Why a serialized queue is the safer choice under current memory conditions:
  - the current official eval already occupies one long-lived server process plus concurrent generation workload
  - forcing a second full eval means either:
    - reusing the same loaded server and mixing request streams, which invalidates the measurement
    - or launching a second server/model process, which is the riskier option for memory pressure and instability
  - with long-context MiniCPM-SALA, instability usually shows up as:
    - OOM or allocator pressure
    - degraded throughput from cache contention
    - confusing timeout / retry / partial-output behavior
  - a serialized queue gives cleaner logs, more stable memory behavior, and a trustworthy mapping from one candidate to one official result

- Smallest local cross-check after an official eval finishes:
  - do one lightweight local `mini-eval` on the exact same stable serve path for that candidate
  - do not start another full official eval before that small cross-check is inspected
  - exact command template:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label candidate-post-official-mini-eval \
  --model-path /root/autodl-tmp/models \
  --dtype float16 \
  --quantization none \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

- Purpose of that cross-check:
  - confirm the local harness still tracks the official result directionally
  - catch gross post-eval serve/path regressions without paying for another full run
  - keep the queue moving while preserving one clean official result per candidate

## 2026-04-09 Runtime Gate Kept Current

- Heavy GPU work remains paused while the official stock eval is still running.
- The exact post-run branch commands above are the current source of truth:
  - healthy official eval:
    - run `stock-base-fp16-mini-control`
  - technically unhealthy or weak official eval:
    - run `stock-base-fp16-official-subsample-c1`
- No runtime A/B should be scheduled before one of those two branch commands is chosen from the post-run grep/tail preflight.

## 2026-04-09 Post-Official Minimal Cross-Check Rules

- Treat the just-finished official full eval as the quality truth; do not use `fast_test.sh --eval-only` to overrule it.
- Before starting the next official full eval, run exactly one local `mini-eval` on the same stable serve path for the same candidate.
- Compare only direction and obvious pathologies: score order, empty outputs, repeated `finish_reason=length`, and major task collapse.
- If local `mini-eval` disagrees sharply with the official result, debug the local harness first; do not schedule another runtime A/B from `smoke`.

## 2026-04-09 Minimal Local `mini-eval` Template

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label <LABEL> \
  --model-path <MODEL_PATH> \
  --quantization <none|modelopt|gptq> \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

## 2026-04-09 Direct `mini-eval` Commands For Post-Official Cross-Checks

`gptq_marlin`

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label gptq-marlin-mini-eval \
  --model-path /root/autodl-tmp/models-gptq-w4a16-v19 \
  --quantization gptq \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

`nvfp4 curated16k128`

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label nvfp4-curated16k128-mini-eval \
  --model-path /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128 \
  --quantization modelopt \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

## 2026-04-09 Serve-Flag Accuracy Audit And Minimal Precision Control

- Current stable serve path used for official/mini style checks:
  - `--dtype float16`
  - `--attention-backend minicpm_flashinfer`
  - `--chunked-prefill-size 8192`
  - `--disable-radix-cache`
  - `--disable-cuda-graph`
  - `--dense-as-sparse`
  - plus route-specific quantization:
    - base: `--quantization none`
    - NVFP4: `--quantization modelopt`
    - GPTQ/Marlin: `--quantization gptq`

- Flags most likely to affect correctness, not just speed:
  - `--quantization`
    - primary source of accuracy change
  - `--dtype`
    - correctness-relevant for MiniCPM-SALA; earlier base runs showed non-fp16 loading/runtime issues
  - `--attention-backend`
    - backend-specific numerical/runtime path; keep fixed across control and candidate
  - `--chunked-prefill-size`
    - should mostly be a runtime knob, but can surface long-context/path bugs if the prefill path is unstable
  - `--dense-as-sparse`
    - not a pure speed flag here; it changes which execution path handles requests, so keep it fixed during precision control

- Flags currently treated as lower correctness risk on the stable path:
  - `--disable-cuda-graph`
    - mainly speed/stability
  - `--disable-radix-cache`
    - mainly speed/cache behavior

- Minimal precision-control experiment after the official queue:
  - run a paired local `mini-eval` A/B with identical stable serve flags
  - change only:
    - `--model-path`
    - `--quantization`
  - interpretation:
    - if stock/base is healthy but the quantized route drops, blame quantization first
    - if both are similarly depressed, inspect serve path or local harness before blaming quantization

- Exact command pair template:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-mini-eval-control \
  --model-path /root/autodl-tmp/models \
  --quantization none \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label <CANDIDATE_LABEL>-mini-eval-control \
  --model-path <CANDIDATE_MODEL_PATH> \
  --quantization <none|modelopt|gptq> \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

- Rule for this control:
  - do not change `attention-backend`, `chunked-prefill-size`, `dense-as-sparse`, or `dtype` inside the control pair
  - otherwise the experiment no longer isolates quantization from serve-path effects

## 2026-04-09 Quant-First Runtime Support State

- Runtime/support role in this pass:
  - speed optimization remains paused
  - official full eval remains the quality source-of-truth
  - local validation remains subordinate to the official path
- Why we are still not sharing one server across two formal full evals:
  - the active official eval already drives the server concurrently
  - a second formal client would blur attribution for latency, failures, and output pathologies
  - for different quantized candidates, the loaded model itself differs, so one shared formal server is not a valid comparison setup
- Current runtime-support expectation after the stock-base official eval:
  - do exactly one same-route local `mini-eval`
  - use it only as a directional cross-check
  - if it disagrees sharply with the official result, repair the local harness before any new official comparison

## 2026-04-09 Mid-Run Runtime Refresh

- Official stock-base eval is still the only heavy run.
- This pass kept runtime heavy work paused and used only a cheap health preflight.
- Fresh runtime delivery in this pass:
  - `stock-base-fp16-mini-control` remains the healthy-branch local cross-check
  - bounded fallback remains an `official` subsample run with `--official-num-samples 12`
  - runner help was checked and confirmed to support:
    - `--profile official`
    - `--official-data-path`
    - `--official-max-seq-len`
    - `--official-concurrency`
    - `--official-num-samples`
- New small blocker captured:
  - remote host lacks `rg`
  - runtime post-run scans must use `grep -E`
- Follow-up assigned in this pass:
  - runtime worker must provide one unified `grep -E` post-run summary command for official-eval completion

## 2026-04-09 Remote DNS Blocker

- Fresh runtime delivery in this pass:
  - the next runtime-side step is still the same-route post-official `mini-eval`
  - it is not runnable until remote reachability from this machine recovers
- Cheap preflight result from this pass:
  - `ssh -G rtx6000-2` is correct
  - local DNS lookup for `connect.bjb1.seetacloud.com` currently fails with:
    - `gaierror(8, 'nodename nor servname provided, or not known')`
- Runtime follow-up after this result:
  - keep the post-official same-route `mini-eval` paused
  - do not schedule a new runtime-side run from this machine until DNS resolution is restored
- Worker bookkeeping for this pass:
  - runtime worker delivered:
    - yes
  - runtime worker received a concrete follow-up:
    - yes

## 2026-04-09 Runtime Hold: Official Eval Still Running At 50/150

- Fresh runtime delivery in this pass:
  - while the stock-base official eval is still running, runtime should only do a `tmux capture-pane` style status check
  - after the official eval finishes, runtime should immediately run the shared `grep -E` plus tail summary scan
- Current official-eval state seen by the orchestrator:
  - `50/150`
  - continued `200 OK`
  - no newly observed `ERROR`
  - no newly observed `Request failed`
- Follow-up assigned in this pass:
  - keep the post-official runtime gate current in:
    - `/Users/ql/cursor/openbmb/sgl-optimization-log.md`

## 2026-04-09 Runtime Comparison Versus Original Weights

- Fresh runtime delivery in this pass:
  - original weights and working-tree base both land at:
    - `28.00%` on the current local smoke gate
  - this means the low local smoke score is not enough evidence that our modified runtime path broke quality
- Current interpretation:
  - the next decisive metric must be the official `Average Score`
  - until that lands, runtime should keep guarding against over-reading `smoke`
- Follow-up assigned in this pass:
  - keep the post-official runtime guardrail fixed on:
    - healthy official result -> stock-base mini-control first
    - unhealthy or contradictory official/local picture -> repair local harness before route comparison

## 2026-04-09 Runtime Branches Ready, Still Waiting On Official Stock Result

- Fresh runtime delivery in this pass:
  - healthy official stock result:
    - run `stock-base-fp16-mini-control`
  - weak or abnormal official stock result:
    - run `stock-base-fp16-official-subsample-c1`
- Current orchestrator read:
  - official stock eval is still running at about `71/150`
  - no post-run summary exists yet, so runtime quality branching is still blocked on the official result
- Follow-up assigned in this pass:
  - keep the two post-official runtime branches current
  - do not start GPU work before the official stock result lands

## 2026-04-09 Runtime Hold Under DNS Blocker

- This pass did not launch any runtime-side run.
- The single gating preflight for this pass was a direct SSH reachability check, and it failed on DNS resolution for:
  - `connect.bjb1.seetacloud.com`
- Fresh runtime delivery in this pass:
  - blocker report from the replacement runtime worker
  - local alias expansion still passes, so the blocker is narrowed to DNS/reachability rather than SSH config text
- Follow-up assigned in this pass:
  - runtime side stays paused until DNS/reachability recovers or the official stock result becomes readable again

## 2026-04-09 Runtime Third Possibility, Kept Bounded

- Fresh runtime delivery in this pass:
  - third possibility:
    - one single-point sanity A/B using the existing runner:
      - `stock-base-fp16-mini-control-nodense`
    - this changes only:
      - `--no-dense-as-sparse`
- Trigger condition:
  - only if the main post-official runtime branches remain weak or ambiguous
  - do not expand into a wider runtime sweep
- Follow-up assigned in this pass:
  - keep this third possibility bounded as fallback-only

## 2026-04-09 Runtime Blocker Reconfirmed

- Direct remote runtime follow-up remained blocked in this pass by local DNS failure for:
  - `connect.bjb1.seetacloud.com`
- No runtime-side GPU work was started.
- Runtime implication:
  - keep the bounded post-official branches as-is
  - do not broaden runtime exploration until reachability and the official stock result are both available

## 2026-04-09 Runtime Minimal Support During Quant Focus

- Current official stock eval is still healthy, but one step has become noticeably slower.
- Runtime interpretation in this pass:
  - treat this first as a possible long-sample effect
  - do not promote it into a new runtime regression claim while the run still shows `200 OK` and no new hard failures
- Follow-up assigned in this pass:
  - stay in minimal-support mode only

## 2026-04-09 Runtime Replacement Worker: SSH Alias Gate

- Latest no-GPU runtime gate:
  - `ssh -G rtx6000-2 | sed -n '1,20p'`
- Result:
  - alias expansion is correct
  - `host` resolves to `connect.bjb1.seetacloud.com`
  - `port` resolves to `50884`
- Interpretation:
  - the blocker is still DNS/reachability to the remote host, not local SSH alias config
  - the runtime-side post-official A/B is not runnable from this machine until that blocker clears

## 2026-04-09 Post-Official Minimal Precision Attribution

- Ground truth for this phase:
  - the official full eval is still running
  - the public set actually exercised here has only these populated input buckets:
    - `0-4K`
    - `16K-32K`
    - `32K-128K`
  - so post-official attribution should focus on quality consistency, not on trying to explain missing `4K-16K` or `128K-160K` behavior from this run

- Minimal attribution flow after the official result lands:
  - first read the official result as the quality truth for the route
  - then run one local serial control pair with identical stable serve flags
  - in that pair, change only:
    - `--model-path`
    - `--quantization`
  - if base control is healthy and the quantized route drops, blame quantization first
  - if both are similarly weak, inspect serve flags or local harness before blaming quantization

- First control experiment is fixed as a serial A/B:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label base-fp16-mini-eval-control \
  --model-path /root/autodl-tmp/models \
  --quantization none \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label <CANDIDATE_LABEL>-mini-eval-control \
  --model-path <CANDIDATE_MODEL_PATH> \
  --quantization <none|modelopt|gptq> \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile mini-eval
```

- Rule for this control pair:
  - do not change `dtype`, `attention-backend`, `chunked-prefill-size`, `dense-as-sparse`, `disable-radix-cache`, or `disable-cuda-graph`
  - otherwise the first attribution step is no longer isolating quantization from serve-path effects

## 2026-04-09 Post-Official Fast Proxy Eval

- Added runner support for a post-official directional quality proxy:
  - profile:
    - `fast-eval`
  - file:
    - `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
- Design goal:
  - faster and more representative than the old `fast_test.sh` smoke
  - still explicitly weaker than official full eval
  - safe to run only after the current official/full-eval chain finishes
- Fixed subset plan:
  - `mcq`:
    - `4` rows from `0-4K`
  - `qa / niah / fwe / cwe`:
    - each `1` row from `16K-32K`
    - each `2` rows from `32K-128K`
  - total:
    - `16` rows
- Why this is preferable to the current smoke:
  - it keeps all five tasks
  - it includes both mid-long and long buckets that actually exist in the public set on disk
  - it stays small enough for fast post-official iteration
- Remote subset is already materialized:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
  - `16` rows
  - task counts:
    - `mcq: 4`
    - `qa: 3`
    - `niah: 3`
    - `fwe: 3`
    - `cwe: 3`
- Launch helper prepared:
  - `/Users/ql/cursor/openbmb/scripts/run_fast_eval_remote.sh`
  - usage:
    - `bash /Users/ql/cursor/openbmb/scripts/run_fast_eval_remote.sh <label> <model_path> <quantization>`

## 2026-04-09 Official Full-Eval Result Extraction Template

```bash
LOG_DIR=/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318 && \
grep -E "Average Score:|Total Duration:|Request failed|ERROR|Traceback" "$LOG_DIR"/*.log 2>/dev/null || true
```

```bash
LOG_DIR=/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318 && \
tail -n 80 "$LOG_DIR"/*.log 2>/dev/null
```

- Live check in this pass:
  - official stock eval is still running
  - no `Average Score` / `Total Duration` / `Request failed` / `ERROR` / `Traceback` summary lines are present yet
  - latest visible signal is healthy in-flight decode activity from `server.log`

## 2026-04-09 Runtime Minimal-Support Follow-Up

- This pass did not open any new runtime A/B branch.
- Fresh runtime delivery in this pass:
  - keep the first post-official control fixed as:
    - `base-fp16-mini-eval-control`
- Fresh runtime follow-up delivery in this pass:
  - exact first official-result extraction command:

```bash
LOG_DIR=/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318 && (grep -E "Average Score:|Total Duration:|Request failed|ERROR|Traceback" "$LOG_DIR"/*.log 2>/dev/null || true) && tail -n 80 "$LOG_DIR"/*.log 2>/dev/null
```

- Worker delivery in this pass:
  - yes
- Worker follow-up assigned in this pass:
  - yes

## 2026-04-09 Runtime Rediscovery Rule After Lost Tmux State

- Fresh runtime delivery in this pass:
  - if Marlin proxy is low after healthy stock official and base fast, the next control remains:
    - `base-fp16-mini-eval-control`
- Cheap preflight blocker in this pass:
  - remote `tmux` windows expected for:
    - `stock-official-eval`
    - `quant-v2-queue`
  - were both missing
- Therefore:
  - do not start `fast` or `mini` from this pass
  - first recover authoritative run directories
- Fresh runtime follow-up delivery in this pass:
  - first rediscovery command:
    - `ssh rtx6000-2 'cd /root/autodl-tmp/SOAR-Toolkit/test_results && ls -td official_stock_base_* quant_* 2>/dev/null | head -n 10'`
  - runtime-side reason:
    - re-establish the latest result directories without depending on tmux survival

## 2026-04-09 Fast-Proxy Budget Fix Landed

- Fresh runtime delivery in this pass:
  - the current `fast` budget path is wrong for proxy gating
  - fix proposal:
    - use the patched `mini_test.sh`
    - enforce task-aware caps:
      - `mcq=64`
      - `qa/niah=128`
      - `fwe/cwe=256`
- Cheap preflight confirmed in this pass:
  - `run_remote_candidate_bench.py` already had:
    - `FAST_EVAL_TASK_CAPS`
    - `task_caps` plumbing into `run_mini_eval(...)`
  - but `run_mini_eval(...)` still invoked the stock remote `mini_test.sh`
- Fix completed in this pass:
  - updated `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - new behavior:
    - when `task_caps` is set, the runner uploads `/Users/ql/cursor/openbmb/soar-patches/mini_test.sh`
    - then runs that patched script remotely
    - so `fast-eval` will finally honor `EVAL_TASK_MAX_TOKENS`
- Why this matters:
  - the previous path let a short-MCQ proxy item inherit `completion_tokens=2767`
  - that turned a cheap sanity gate into a multi-thousand-token generation test

## 2026-04-09 Runtime Follow-Up: Post-Official Sanity Sequence Tightened

- Fresh runtime delivery in this pass:
  - keep the post-official quality cross-check strictly serial:
    1. `base fast-eval`
    2. `quant fast-eval`
    3. only if both are technically healthy, `quant mini-eval`
- Additional runtime follow-up delivery in this pass:
  - if:
    - `base fast-eval` is technically healthy
    - but `quant fast-eval` is weak
  - then the lowest-cost runtime-side sanity check is:
    - same serve flags
    - only change `model-path/quantization`
    - run:
      - `base mini-eval`
      - `quant mini-eval`
  - concrete command family:
    - `python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py --label <label> --model-path <path> --quantization <none|modelopt|gptq> --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --disable-radix-cache --disable-cuda-graph --dense-as-sparse --profile mini-eval`
- Harness hardening completed in this pass:
  - updated `/Users/ql/cursor/openbmb/scripts/run_quant_fast_queue.sh`
  - new queue behavior:
    - abort the quant queue if `base fast` has technical failures or no summary
    - skip `mini` if `quant fast` shows technical failures, missing summary, or non-zero `empty=`
- synced deployed copy to:
  - `/root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/run_quant_fast_queue.sh`

## 2026-04-09 Runtime Bug Fixes After Official Exit

- New abnormal stock result in this pass:
  - the stock official process ended, but:
    - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260409_034351`
    - is empty
  - so this stock official run cannot be treated as a valid quality anchor yet
- Queue bug found and fixed in this pass:
  - `run_quant_fast_queue.sh` kept waiting even after official exited
  - root cause:
    - `pgrep -af` wait expressions could self-match the queue shell context
  - fix:
    - replaced the wait loop with:
      - `official_eval_active()`
      - `stock_server_active()`
    - both now use `ps ... | awk` and explicitly ignore the queue script itself
  - synced deployed copy:
    - `/root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/run_quant_fast_queue.sh`
- `fast` harness bug found and fixed in this pass:
  - the first rerun of:
    - `stock-base-fast-eval`
  - failed with:
    - `NameError: name 'k' is not defined`
  - root cause:
    - the remote helper script builder in `run_remote_candidate_bench.py` embedded a dict comprehension inside an outer f-string
  - fix:
    - rewrote the JSON serialization line to avoid brace interpolation
    - `py_compile` passed after the patch
- New active runtime control run:
  - label:
    - `stock-base-fast-eval`
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_045057_stock-base-fast-eval`
  - live evidence after the fix:
    - `fast_eval.log` shows the fixed 16-row proxy set selection
    - `server.log` shows stock server load complete and requests started

## 2026-04-09 Runtime Gate Fix: Task-Aware Caps For `fast`

- New proxy-quality bug confirmed in this pass:
  - even after the remote-script `NameError` fix, `stock-base-fast-eval` still gave short `mcq` samples:
    - `req_out=2767`
  - that made the proxy effectively behave like an unintended long-generation test
- Fix completed in this pass:
  - patched `/Users/ql/cursor/openbmb/soar-patches/mini_test.sh`
    - added optional `EVAL_TASK_MAX_TOKENS`
    - task-aware caps:
      - `mcq=64`
      - `qa=128`
      - `niah=128`
      - `fwe=256`
      - `cwe=256`
  - patched `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
    - `fast-eval` now exports those caps only for that profile
  - patched `/Users/ql/cursor/openbmb/scripts/run_quant_fast_queue.sh`
    - both base and quant `fast` phases now use the same caps
  - synced remote copies:
    - `/root/autodl-tmp/SOAR-Toolkit/mini_test.sh`
    - `/root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/run_quant_fast_queue.sh`
- New capped control run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_045846_stock-base-fast-eval-capped`
- First observed capped rows:
  - `mcq` rows now request `64`, not `2767`
  - no server error is present in the observed prefix
  - therefore:
    - `fast` is now usable as a directionality gate
    - but the stock base quality on this proxy still appears low

## 2026-04-09 Post-Official Fast Proxy Eval Subset Proposal

- Suggested fixed proxy subset size:
  - `20` questions total
- Suggested task/length quota:
  - `mcq`: `4` from `0-4K`
  - `qa`: `2` from `16K-32K`, `4` from `32K-128K`
  - `niah`: `2` from `16K-32K`, `4` from `32K-128K`
  - `fwe`: `1` from `16K-32K`, `1` from `32K-128K`
  - `cwe`: `1` from `16K-32K`, `1` from `32K-128K`
- Why this quota:
  - it keeps `mcq` present but prevents the cheap `0-4K` bucket from dominating the proxy
  - it preserves the observed public-set shape where the long-context burden is mainly in `qa/niah/fwe/cwe`
  - it intentionally over-weights `qa/niah` because they are both common and sensitive to long-context quality loss
  - it keeps `fwe/cwe` in the proxy so formatting/retrieval regressions still show up, but at low enough count that they do not dominate runtime cost
  - `20` total is still much cheaper than full official eval, but materially more representative than the current `smoke`

## 2026-04-09 Post-Official `fast-eval` Command

- Fixed remote dataset:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
- Exact post-official command:

```bash
python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label <LABEL>-fast-eval \
  --model-path <MODEL_PATH> \
  --quantization <none|modelopt|gptq> \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile fast-eval
```

- Usage boundary:
  - `fast-eval` is only for directional post-official judgment
  - it can help rank nearby candidates quickly after an official result lands
  - it cannot replace or overrule the official full eval
- Current execution strategy for this thread:
  - because the user explicitly asked for `eval` to be followed by a non-disruptive `fast_eval` before continuing quantization, the active remote queue is allowed to auto-run one base `fast-eval` immediately after the stock official eval finishes
  - treat that as a single bounded cross-check, not a general precedent for inserting more proxy runs ahead of official work
- Queue policy lock:
  - historical default was to keep post-official `fast-eval` manual-only
  - for the current thread, this is superseded by the active remote queue because the user explicitly wants:
    - official eval -> bounded non-disruptive `fast_eval` -> continue quantization
  - local sandbox SSH preflight is currently unavailable, so no new remote-state assumption should be made from that older pass alone
- One-line post-official execution rule:
  - stock official healthy end -> base `fast-eval` -> quant `fast-eval` -> if both are technically healthy, then `mini-eval`; if any step is technically abnormal, fix harness/serve first and do not directly call the model route worse

## 2026-04-09 Post-Official Summary Template

- official: `Average Score=<...>%`, `Total Duration=<...>s`
- technical health: `healthy` / `abnormal` with top error keyword
- base fast: `<score/status>`
- quant fast: `<score/status>`
- next action: `mini-eval` / `fix harness/serve` / `resume quant queue`

## 2026-04-09 One-Line Official Result Extraction

```bash
LOG_DIR=/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318 && ((grep -E "Average Score:|Total Duration:|Request failed|ERROR|Traceback" "$LOG_DIR"/*.log 2>/dev/null || true) && echo "__TAIL__" && tail -n 80 "$LOG_DIR"/*.log 2>/dev/null)
```

## 2026-04-09 Speed Gate

- 只有当某条量化路线已经拿到技术健康的 official full eval，且本地 `fast-eval` / `mini-eval` 与 official 方向一致时，才允许切到速度优化。
- 只要 official 未落地、存在技术异常，或本地交叉验证与 official 明显冲突，就继续留在精度线，先修量化或 harness/serve。
- 速度优化只能发生在“精度结论已基本稳定”之后，不能反过来替代精度判断。

## 2026-04-09 Queue Policy Reaffirmed Under SSH Blocker

- Fresh runtime delivery in this pass:
  - keep post-official `fast-eval` manual-only
  - do not auto-insert it ahead of the main quant sequence
- Reason recorded in this pass:
  - it is directional only
  - it adds latency and another failure point without changing the official quality conclusion
- Cheap preflight blocker in this pass:
  - fresh direct remote SSH from the current sandbox failed with:
    - `Operation not permitted`
- Therefore:
  - do not infer new remote queue state from this pass
  - keep the queue policy unchanged

## 2026-04-09 Runtime Blocker Corrected: `fast-eval` Already Exists

- Fresh runtime delivery in this pass initially claimed:
  - `run_remote_candidate_bench.py` did not implement `fast-eval`
- Follow-up assigned in this pass:
  - re-check the current local file state and either prove the blocker or correct it
- Fresh runtime follow-up delivery in this pass corrected the blocker:
  - `fast-eval` already exists in `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - cited evidence:
    - argparse profile choice
    - dataset constant:
      - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
    - helper:
      - `ensure_fast_eval_dataset()`
    - dispatch branch
    - execution branch
- Correct next runtime handoff remains:
  - `python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py --label stock-base-fast-eval --model-path /root/autodl-tmp/models --quantization none --dtype float16 --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --disable-radix-cache --disable-cuda-graph --dense-as-sparse --profile fast-eval`
- New blocker in this pass:
  - fresh direct-IP remote reachability check from this sandbox failed with:
    - `Operation not permitted`
- Therefore:
  - do not treat `fast-eval` support as a blocker anymore
  - the active blocker for this pass is sandbox SSH reachability, not runtime runner wiring
- Worker delivery in this pass:
  - yes
- Worker follow-up assigned in this pass:
  - yes

## 2026-04-09 Attempt: Direct-IP `gptq` Marlin Smoke Gate

- Status:
  - blocked before any stable smoke eval could run
- Why this was chosen:
  - quant logs kept `W4A16 + GPTQ + Marlin` as the next champion-aligned route
  - the local GPTQ checkpoint already exists, so the first read should be a cheap smoke gate, not a new quantization pass
- Remote access used in this pass:
  - DNS for `rtx6000-2` remained flaky, so I reused the previously recovered direct fallback:
    - `root@106.120.183.117:50884`
- Stock official-eval gate:
  - the earlier full stock-base eval at `/root/autodl-tmp/SOAR-Toolkit/test_results/official_stock_base_20260409_034318` was technically healthy in flight
  - it continued to show `200 OK` with no visible `Request failed`, `Traceback`, or `ERROR`
  - to free the single serialized slot for this pass, I terminated that older full-eval/server pair after capturing that health read
- First launch attempt:
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/manual_marlin_smoke_ip_20260409`
  - command intent:
    - local GPTQ checkpoint
    - stock import path
    - `--quantization gptq_marlin`
  - result:
    - immediate argument/config mismatch
  - blocker:
    - `ValueError: Quantization method specified in the model config (gptq) does not match the quantization method specified in the quantization argument (gptq_marlin).`
- Second launch attempt:
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/manual_marlin_smoke_ip_gptq_20260409`
  - corrected only:
    - `--quantization gptq`
  - result:
    - launch progressed into weight loading, then failed
  - blocker:
    - `KeyError: 'model.layers.1.self_attn.z_proj.weight'`
- Import-path isolation:
  - `/root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3` imports:
    - default: `/root/autodl-tmp/sglang-stock/python/sglang/launch_server.py`
    - with `PYTHONPATH=/root/autodl-tmp/sglang/python`: `/root/autodl-tmp/sglang/python/sglang/launch_server.py`
- Third launch attempt:
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/manual_marlin_smoke_ip_gptq_live_20260409`
  - corrected only:
    - forced the live custom tree with `PYTHONPATH=/root/autodl-tmp/sglang/python`
    - kept `--quantization gptq`
  - result:
    - same load-time failure
  - blocker:
    - `KeyError: 'model.layers.1.self_attn.z_proj.weight'`
- Prepared-model handoff:
  - existing quant-side materialization step succeeded:
    - `bash /root/autodl-tmp/codex-drafts/w4a16-marlin-draft/prepare_model.sh --input /root/autodl-tmp/models --output /root/autodl-tmp/models-gptq-marlin-local-prepared`
  - output path:
    - `/root/autodl-tmp/models-gptq-marlin-local-prepared`
- Fourth launch attempt:
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/manual_marlin_smoke_ip_prepared_20260409`
  - corrected only:
    - prepared model path plus live custom tree plus `--quantization gptq`
  - result:
    - server still died during load
  - blocker:
    - `KeyError: 'model.layers.1.self_attn.z_proj.weight'`
- Runtime conclusion from this pass:
  - the active Marlin blocker is now narrowed to checkpoint/model-schema compatibility
  - it is not primarily:
    - the remote SSH path
    - `gptq_marlin` vs `gptq` launch arg naming
    - stock-vs-live SGLang import path
- Metrics collected:
  - no stable smoke correctness metric was collected because no launch survived long enough to run `fast_test.sh --eval-only`
  - strongest in-flight runtime signal came from the earlier stock-base full eval before it was stopped:
    - sustained `200 OK`
    - no visible hard failure markers
- End-of-pass note:
  - final read-only check showed a new base-model `mini_test.sh` plus base `sglang.launch_server` on port `30001`
  - that means the remote GPU was no longer clean for another serialized run at the end of this pass
- Next runtime action:
  - inspect why the GPTQ checkpoint family lacks the loader-expected `z_proj` weights
  - if the checkpoint naming is recoverable, add the required remap/materialization fix before retrying this same Marlin smoke gate

## 2026-04-09 Runtime Follow-Up After Fast-Proxy Runner Patch

- Fresh runtime delivery in this pass:
  - accept the patched `fast` runner path as runnable
  - concrete evidence:
    - `run_remote_candidate_bench.py` defines `FAST_EVAL_TASK_CAPS`
    - when `task_caps` is set, it uploads and runs the patched remote `mini_test.sh`
    - the `fast-eval` branch passes `task_caps=FAST_EVAL_TASK_CAPS`
    - local `py_compile` for the runner passed
- Cheap preflight completed in this pass:
  - `python3 -m py_compile /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
  - result:
    - passed
- Non-blocking side observation:
  - `py_compile` on `/Users/ql/cursor/openbmb/soar-patches/mini_test.sh` failed only because the file is a shell script
- First runtime proof check after this pass:
  - after the next base `fast-eval`, inspect `fast_eval.log`
  - confirm short `mcq` rows use capped `req_out`, not the old `2767`
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Runtime Hold Rule After Sandbox-Blocked Marlin Metadata Check

- Fresh runtime delivery in this pass:
  - do not relaunch `Marlin` with either:
    - `--quantization gptq_marlin`
    - `--quantization gptq`
  - reason:
    - `gptq_marlin` is already blocked at the quantization-method config check
    - `gptq` already got past that check before and then died later on:
      - `KeyError: 'model.layers.1.self_attn.z_proj.weight'`
- Cheap preflight attempted in this pass:
  - read-only remote metadata inspection for the prepared `Marlin` checkpoint
- Result:
  - blocked at the local sandbox layer:
    - `ssh: connect to host 106.120.183.117 port 50884: Operation not permitted`
- Therefore:
  - runtime retries are frozen from this pass
  - the next useful move remains on the quant/materialization side after trusted SSH returns
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Runtime Hold Rule After Medium-Eval Status Check Failed At Sandbox Layer

- Fresh runtime delivery in this pass:
  - after `medium-eval`, do not launch any next runtime A/B until there is one trusted read of:
    - `eval_model.log`
    - output / summary files
  - if SSH is still blocked with `Operation not permitted`, treat runtime state as unknown and keep runtime on no-launch hold
- Cheap preflight attempted in this pass:
  - one direct remote status read for the active `medium-eval`
- Result:
  - blocked at the sandbox layer:
    - `ssh: connect to host 106.120.183.117 port 50884: Operation not permitted`
- Replacement worker note:
  - one replacement runtime worker later produced an untrusted destructive claim about remote directory deletion
  - because this pass had no trusted SSH read, that claim was discarded and the replacement worker was closed
- Therefore:
  - runtime stays on no-launch hold from this pass
  - the next valid runtime action is still one trusted post-run read of `medium-eval` artifacts
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Runtime Delivery After Local Medium-Eval Readiness Patch

- Fresh runtime delivery in this pass:
  - yes
- One cheap preflight executed:
  - patched:
    - `/Users/ql/cursor/openbmb/scripts/run_medium_eval_remote.sh`
  - replaced fixed sleep with readiness polling against `/v1/models`
  - if readiness never arrives, the launcher now kills the started server and exits
  - local validation:
    - `bash -n /Users/ql/cursor/openbmb/scripts/run_medium_eval_remote.sh` passed
- Runtime handoff after that patch:
  - keep the next local guardrail narrowed to one change only:
    - persist the successful `/v1/models` payload to `$RUN_DIR/model_probe.json`
    - require a non-empty model id before launching `eval_model.py`
- No broader runtime branch was opened in this pass.
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Runtime Trust Gate Before Next Medium-Eval Launch

- Fresh runtime delivery in this pass:
  - the local `medium-eval` launcher is already patched from fixed sleep to readiness polling and passes `bash -n`
- Exactly one next runtime follow-up:
  - before any new `medium-eval`, make the launcher fail closed unless `/v1/models` returns at least one non-empty model id, and use that returned id for the eval invocation instead of trusting model auto-detect
- Why this is the trust gate:
  - it covers both prior launch-failure signatures in one cheap preflight:
    - `127.0.0.1:30001 ... connection refused`
    - `Could not auto-detect model name`
  - a bare readiness `200 OK` is weaker than a verified non-empty model-id gate
- Heavy run status in this pass:
  - none
- Follow-up assigned in this pass:
  - yes
- Follow-up acknowledgement received in this pass:
  - yes

## 2026-04-09 Stock Base Medium-Eval Result And Runtime Caveat

- Completed run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/medium_eval_stock_base_20260409_104835`
- Output:
  - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260409_104905`
- Final summary:
  - `Original Accuracy: 50.40%`
  - `Normalized Accuracy: 63.0%`
  - `Total Duration: 2405.74 s`
  - `TPS: 178.88`
- Runtime conclusion:
  - the launcher/readiness fix worked; this run completed cleanly
  - however, this medium-eval is not a clean speed benchmark because generation was effectively uncapped for some tasks
- Evidence of duration pollution:
  - sample `39` output tokens:
    - `65545`
  - sample `45` output tokens:
    - `65545`
  - combined:
    - `131090 / 430343` output tokens
    - about `30.5%` of all generated output tokens came from `2/50` samples
- Practical implication:
  - use this run as a correctness/control anchor
  - do not use its duration directly to compare runtime optimizations
  - for runtime comparisons, prefer capped `fast` or dedicated bench datasets

## 2026-04-09 Runtime Bench Policy Shift

- Old `fast` artifacts were removed from remote and are no longer the active quick gate.
- New quick correctness gate:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
- Why:
  - it is a direct subset of the validated `medium_eval` set
  - it uses the official `eval_model.py` path instead of the patched mini-test path
  - it keeps exact task balance:
    - `4` samples per task
- New intended interpretation:
  - `fast-eval`:
    - quick correctness direction
  - `medium-eval`:
    - stronger correctness anchor
  - dedicated bench datasets:
    - speed judgment

## 2026-04-09 Runtime Interpretation Of The Failed BF16 NVFP4 Route

- The first fresh `NVFP4 mlp_weight_only + bfloat16 + 32K + 64` route should not be treated as a clean accuracy datapoint.
- Why:
  - quantization/export completed
  - but the first online eval batch crashed the server
  - fast and medium logs therefore read as:
    - `0.00%`
    - because the server died, not because the model cleanly answered every row wrong
- Runtime warning sequence worth remembering:
  - `DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0`
  - `Casting torch.bfloat16 to torch.float16`
  - then:
    - `CUDA illegal memory access`
- Runtime implication:
  - the most plausible next runtime probe is not a new broad sweep
  - it is a narrow retry with:
    - `float16` export
    - `float16` serve dtype
    - `DeepGemm disabled`

## 2026-04-09 Fast-Eval Gate Corrected To Avoid Runaway Outputs

- The first medium-derived `fast-eval v2` still had a design flaw:
  - it kept the original sample `completion_tokens`
  - so a supposedly quick gate could still get trapped in long runaway generations
- Corrected policy:
  - keep the same medium-derived rows
  - keep `4` rows per task
  - cap completion tokens per task:
    - `mcq=64`
    - `qa=128`
    - `niah=128`
    - `fwe=256`
    - `cwe=256`
- Intended use:
  - `fast-eval` becomes a real quick gate again
  - `medium-eval` stays the stronger quality anchor

## 2026-04-09 Runtime/Harness Naming Cleanup

- Active quick gate is now named only:
  - `fast`
- Active dataset:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl`
- Active launcher helper:
  - `/Users/ql/cursor/openbmb/scripts/run_fast_eval_remote.sh`
- Active quant queue helper:
  - `/Users/ql/cursor/openbmb/scripts/run_quant_fast_queue.sh`
- Repaired stock-base control now running on the unified `fast` gate:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_175739_stock-base-fast-eval`
- Runtime state at log time:
  - live server on original weights
  - live `eval_model.py`
  - GPU busy rather than idle
  - so the current `fast` harness is behaving like a real bounded eval, not a dead launcher

## 2026-04-09 GPTQ Fast Tail Attribution

- For the live `gptq_marlin` `fast` run, the tail currently looks `decode-heavy`, not `prefill-heavy`.
- Why:
  - progress already reached `8/9` in `eval_model.log`
  - sampling kwargs still allow `max_tokens=65536`
  - the run has over-length prompt warnings but no request-time stats that would indicate a prefill stall
  - GPU utilization around `38%` is consistent with batch collapse onto one or two long decode requests
- Caveat:
  - current server args still have `enable_request_time_stats_logging=False`
  - so the existing `server.log` is not sufficient for an exact per-request prefill/decode breakdown

## 2026-04-09 Fast Runtime Visibility + Cap Fix

- Root cause of the overly slow `fast` gate:
  - the active `fast-eval` path was still routed through official `eval_model.py`
  - official `eval_model.py` hardcodes `model.generate(inputs, max_out_len=65536)`
  - so curated per-task caps in `perf_public_fast_eval.jsonl` were not taking effect
- Runtime-facing fix:
  - `fast-eval` in `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py`
    now uses patched `/Users/ql/cursor/openbmb/soar-patches/mini_test.sh`
    with:
    - `EVAL_USE_ALL_ROWS=1`
    - task-aware `EVAL_TASK_MAX_TOKENS`
  - `fast-eval` now auto-adds:
    - `--enable-request-time-stats-logging`
- Resulting expectation for future `fast` runs:
  - real capped output lengths
  - finer-grained request-time stats from SGLang
  - much less risk of a single long decode turning `fast` into a 10-minute pseudo-medium run

## 2026-04-09 Stock Fast Rerun Validation

- A fresh stock-base `fast` rerun was launched after the harness fix:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260409_200504_stock-base-fast-eval-rerun`
- Early live output already validates the fix:
  - `mcq` is now truly capped at `req_out=64`
  - `qa` is now truly capped at `req_out=128`
  - `fwe` is now truly capped at `req_out=256`
- Early latencies are correspondingly bounded:
  - `2.49s`, `6.17s`, `8.88s`
- The stock server is also running with:
  - `--enable-request-time-stats-logging`
  so later server-side timing attribution should be possible without another harness rewrite.

## 2026-04-10 MiniCPM Sparse Bypass For Py310 GPTQ Eval

- Confirmed by source inspection and remote repro:
  - `flash_attn` alone does not replace `sparse_kernel_extension` for MiniCPM-SALA
  - both `minicpm_flashattn` and `minicpm_flashinfer` still route through `MiniCPMSparseBackend`
  - `MiniCPMSparseBackend` imports `sparse_kernel_extension` and `tilelang` unconditionally
- Confirmed a viable runtime bypass for eval:
  - use dense backend instead of sparse MiniCPM backend
  - launcher change:
    - from `--attention-backend minicpm_flashinfer --dense-as-sparse`
    - to `--attention-backend flashinfer --force-dense-minicpm`
- Dense path still requires `SimpleGLAAttnBackend` support for lightning layers:
  - copied `fla` from the py310 GPTQ env into the py310 eval env
- Dense `flashinfer` path also required:
  - `ninja` available on `PATH`
  - corrected harness env names for the patched `mini_test.sh`
- After these changes, py310 GPTQ fast eval progressed past:
  - sparse-kernel import failures
  - `fla` missing
  - `ninja` missing
  - wrong fast dataset env wiring
- Latest active run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_fastcheck_20260410_130050`
- Verified live progress:
  - server ready
  - fast subset = `9` items
  - first request completed with request-time stats logged by SGLang

## 2026-04-10 Dense-Fallback GPTQ Fast Completed

- The dense `flashinfer + force_dense_minicpm` fallback completed the full `9`-sample bounded fast gate.
- Final run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_fastcheck_20260410_142000`
- Final score:
  - `avg_score=46.67%`
- This validates a practical py310 evaluation route that does not depend on:
  - `MiniCPMSparseBackend`
  - `sparse_kernel_extension`
  - `tilelang`
- Important caveat:
  - this is a compatibility-first fallback, not a proof that dense runtime is the final best competition route
  - for now it is the cleanest path to submission environment validation

## 2026-04-10 Over-Generation Runtime Follow-Up On Dense Fallback

- Scope rule for this pass:
  - left `eval_model.py` unchanged
  - restricted changes to serve-side experiment plumbing only
- Verified on the working dense-fallback route:
  - launch path:
    - `--attention-backend flashinfer --force-dense-minicpm`
  - live eval sampling kwargs still were:
    - `temperature=0.0`
    - `max_tokens=65536`
    - `stop=['<|im_end|>', '</s>']`
  - tokenizer decode confirms those stop strings are correct
- Runtime conclusion from that verification:
  - dense fallback does not remove the stopping pathology
  - so current over-generation is not primarily a sparse-backend bug
- Fresh tail inspection from the completed dense-fallback medium run showed:
  - `cwe/fwe` enter low-entropy repeated loops
  - `mcq` more often spends thousands of tokens on long reasoning before the final answer line
- Runner improvements for runtime-only A/B:
  - updated:
    - `/Users/ql/cursor/openbmb/scripts/remote_medium_eval_gptq_py310.sh`
    - `/Users/ql/cursor/openbmb/scripts/remote_fast_eval_gptq_py310.sh`
  - both now support:
    - `PREFERRED_SAMPLING_PARAMS`
    - shorthand `REPETITION_PENALTY`
  - this allows serve-side anti-repeat ablations without changing the eval client
- Bounded runtime ablation:
  - focus subset:
    - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_overgen8_eval_v1.jsonl`
  - run:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_overgen8_rp105_20260410_172432`
  - serve delta only:
    - `REPETITION_PENALTY=1.05`
- Why this subset was chosen:
  - these `8` rows alone accounted for:
    - `323311 / 330508` output tokens
    - about `97.82%` of the earlier `25`-row dense-fallback medium output
- Important caveat:
  - I stopped the run after the server had emitted request-time stats for all `8` requests, but before the eval client wrote final outputs / score files
  - therefore this pass only supports token-length conclusions, not accuracy conclusions
- Server-side request-time stats still gave a high-signal result:
  - total subset output length fell:
    - `323311 -> 142756`
    - about `-55.8%`
  - strongest reductions:
    - `input len=127738`:
      - `65541 -> 688`
    - `input len=115313`:
      - `65545 -> 1124`
    - `input len=31433`:
      - `65538 -> 43752`
    - `input len=31697`:
      - `65544 -> 43752`
  - `mcq` behavior stayed mixed:
    - `27922 -> 13694`
    - `19748 -> 18591`
    - `6812 -> 5829`
    - but one case worsened:
      - `6661 -> 15326`
- Runtime read:
  - a light repetition penalty is strong enough to shrink the worst extraction-style loops
  - but not strong enough to be treated as a safe blanket runtime fix
  - `mcq` remains the main caution against simply turning this on globally
- Next runtime recommendation:
  - keep the dense-fallback path as the clean compatibility route
  - do not spend the next runtime window on sparse-kernel surgery for this problem
  - if score attribution on the subset is needed, rerun the same `8` rows and let the client finish writing outputs
  - otherwise hand the main fix back to the quantization side:
    - calibration should better preserve short assistant-closing states instead of only long continuation states

## 2026-04-10 Stop-Aligned Calibration Follow-Up On Dense Fallback

- Scope rule for this pass:
  - left `eval_model.py` unchanged
  - left dense-fallback serve route unchanged:
    - `--attention-backend flashinfer --force-dense-minicpm`
  - changed the quantization calibration distribution instead of the eval client
- New calibration asset:
  - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_stopaligned64k_v1.jsonl`
- What changed in the calibration builder:
  - added a third quota family:
    - `--chat-close-quotas`
  - new rows are rendered with the model chat template:
    - `user -> assistant short answer`
  - for these rows:
    - `skip_special_tokens=False`
    - this preserves closing markers such as `<|im_end|>` in the calibration text
  - added a sidecar summary file:
    - `*.meta.json`
    - records source mix, task mix, and token-length stats
- Stop-aligned v1 composition:
  - `64` rows total
  - source mix:
    - `soar_public=36`
    - `soar_chat_close=16`
    - `pg19_local=12`
  - task mix:
    - `mcq=12`
    - `qa=16`
    - `niah=16`
    - `fwe=4`
    - `cwe=4`
    - `open_text=12`
  - length summary:
    - `calib_tokens p50=63193`
    - `calib_tokens p90=65536`
    - `calib_tokens max=65536`
- New quantized checkpoint:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1`
- Quantization note:
  - `quant_log.csv` shows `221` module-level `rtn failsafe` rows
  - this is a real quant-side risk marker, but the final evaluation behavior still improved materially
- Tokenizer startup note:
  - the checkpoint initially failed eval-env tokenizer load because GPTQ save output did not preserve a compatible tokenizer asset set
  - copying tokenizer assets from the source model into the output checkpoint unblocked eval-env startup
  - this was a checkpoint packaging compatibility repair, not the reason for the behavioral score gain
- Fast eval after tokenizer repair:
  - run:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_stopaligned64k_v1_fast_retry_20260410_184336`
  - result:
    - `avg_score=58.89%`
  - compared with the earlier py310 GPTQ dense-fallback fast baseline:
    - `46.67% -> 58.89%`
    - `+12.22`
  - request-time stats show several rows now stopping at short lengths such as:
    - `63`
    - `64`
    - `128`
    - `256`
  - however some `mcq` / `niah` cases still reached `length`
- Medium eval final completed rerun:
  - run:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_stopaligned64k_v1_medium_retry_20260410_191105`
  - outputs:
    - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260410_191120`
  - final result:
    - `Average Score=84.40%`
    - `Total Duration=1206.01s`
    - `Overall TPS (Output)=171.76`
    - `Total Tokens: In=1494045, Out=207140`
- Compared with the earlier py310 GPTQ dense-fallback medium baseline on the same `25` rows:
  - score:
    - `72.00% -> 84.40%`
  - duration:
    - `1256.88s -> 1206.01s`
  - output tokens:
    - `330508 -> 207140`
  - output TPS:
    - `262.96 -> 171.76`
- Runtime interpretation:
  - the win did not come from faster per-token decoding
  - the win came from generating fewer tokens overall on many rows
  - dense fallback remains slower per token than desired, but no longer dominates the total wall time once runaway generation is reduced
- Remaining failure pattern from request-time stats:
  - two long-context rows still hit `65536`
  - several short-prompt `mcq` rows still over-generated into the `3k-21k` range
  - `cwe` remains the clearest residual weakness
- Direct overlap against the previously saved stock/original-weight outputs on shared `index` values shows:
  - `fwe` and `niah` improved the most and usually stopped much earlier
  - `mcq` improved in score but still has unstable output length tails
  - `qa` stayed roughly flat
  - `cwe` still has cases where output length explodes even when score is acceptable
- Current runtime takeaway:
  - dense fallback is still the valid serve/eval route
  - sparse-vs-dense is not the main lever for the current gap
  - stop-aligned calibration is the first change that clearly improved both score and total duration without touching `eval_model.py`

## 2026-04-10 Stop-Aligned + `repetition_penalty=1.03` Runtime Probe

- Scope:
  - left `eval_model.py` unchanged
  - kept the same dense-fallback serve route
  - only changed serve-side default sampling via:
    - `--preferred-sampling-params {"repetition_penalty":1.03}`
- Checkpoint under test:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1`
- Subset under test:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_overgen8_eval_v1.jsonl`
- Run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_stopaligned64k_v1_overgen8_rp103_20260410_195537`
- Server confirmed the new default was active:
  - `preferred_sampling_params={'repetition_penalty': 1.03}`
- Current read from the completed `7/8` request-time stats:
  - extraction-style long rows:
    - `31433 -> 808`
    - `115313 -> 924`
    - `31697 -> 1277`
  - short-prompt `mcq` rows:
    - `103 -> 6210`
    - `167 -> 21525`
    - `344 -> 19226`
    - `589 -> 19505`
- Compared with the stop-aligned no-penalty baseline on the same `7` rows:
  - baseline total output tokens:
    - `67525`
  - `rp=1.03` total output tokens:
    - `69475`
  - delta:
    - `+1950`
- Per-row comparison against the stop-aligned baseline:
  - `cwe/fwe`:
    - `813 -> 808`
    - `1188 -> 924`
    - `842 -> 1277`
  - `mcq`:
    - `4747 -> 6210`
    - `19992 -> 21525`
    - `21050 -> 19226`
    - `18893 -> 19505`
- Runtime interpretation:
  - on top of stop-aligned quantization, a light repetition penalty is no longer a clear win
  - it still helps some extraction tails
  - but the net effect across the first `7` finished rows is slightly worse because several `mcq` outputs lengthen
  - this is milder than the earlier `1.05` diagnostic on the older checkpoint, but still not clean enough to promote as a default runtime fix
- Open state at log time:
  - the final heaviest row was still running
  - likely the longest-context `cwe` case

## 2026-04-10 Completed Penalty Sweep On Stop-Aligned Checkpoint

- Checkpoint under test:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1`
- Baseline for comparison:
  - stop-aligned no-penalty outputs on the `7` mutable rows
  - total output tokens:
    - `67525`
  - average score:
    - `0.9857`

### `repetition_penalty=1.03`

- Run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_stopaligned64k_v1_overgen8_rp103_20260410_195537`
- Full `8`-row subset:
  - total output tokens:
    - `135057`
  - average score:
    - `0.85`
- Mutable `7`-row slice:
  - total output tokens:
    - `69512`
  - average score:
    - `0.8429`

### `frequency_penalty=0.05`

- Run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/stopalign_fp005_overgen8_fixed_20260410_201922`
- Full `8`-row subset:
  - total output tokens:
    - `135057`
  - average score:
    - `0.85`
- Key result:
  - it matched `rp=1.03` exactly on this test slice

### `presence_penalty=0.05`

- Run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/stopalign_pp005_overgen7_20260410_203904`
- Mutable `7`-row subset:
  - total output tokens:
    - `120462`
  - average score:
    - `0.70`
- Per-row read:
  - `103 mcq`:
    - `4747 -> 5628`
    - score stayed:
      - `1`
  - `167 mcq`:
    - `19992 -> 16522`
    - score fell:
      - `1 -> 0`
  - `344 mcq`:
    - `21050 -> 8805`
    - score stayed:
      - `1`
  - `589 mcq`:
    - `18893 -> 21493`
    - score fell:
      - `1 -> 0`
  - `31433 cwe`:
    - `813 -> 65542`
    - score stayed:
      - `0.9`
  - `31697 cwe`:
    - `842 -> 1284`
    - score stayed:
      - `1.0`
  - `115313 fwe`:
    - `1188 -> 1188`
    - score stayed:
      - `1.0`

### Runtime Conclusion

- No tested penalty improved on the stop-aligned no-penalty baseline.
- Ranking:
  - best:
    - no penalty
  - tied worse:
    - `repetition_penalty=1.03`
    - `frequency_penalty=0.05`
  - worst:
    - `presence_penalty=0.05`
- Decision:
  - do not promote any of these penalties into the default runtime route
  - do not spend another `fast` eval on penalty tuning for now
  - return focus to quantization/calibration work

## 2026-04-10 Stop-Aligned Full `medium` Rerun

- Run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/stopaligned64k_v1_medium_full_20260410_210533`
- Output:
  - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260410_210548`
- Result:
  - `Average Score: 76.40%`
  - `Total Duration: 1222.97 s`
  - `Total Tokens: In=1494045, Out=208484`
  - `Overall TPS (Output): 170.47`
- Compared with the earlier stop-aligned medium retry:
  - score:
    - `84.40% -> 76.40%`
  - output tokens:
    - `207140 -> 208484`
  - duration:
    - `1206.01 -> 1222.97`
- Per-row drift is concentrated in `mcq`:
  - `167 mcq`:
    - `19992 -> 27617`
    - score:
      - `1 -> 0`
  - `589 mcq`:
    - `18893 -> 22575`
    - score:
      - `1 -> 0`
  - `344 mcq`:
    - `21050 -> 9667`
    - score stayed:
      - `1`
  - `103 mcq`:
    - `4747 -> 6466`
    - score stayed:
      - `1`
- Runtime interpretation:
  - the rerun did not fail because the model suddenly emitted far more total tokens
  - it failed because a few short `mcq` rows are still unstable enough to flip answers between runs
  - the stubborn `cwe` tail remains, but it was already present and does not explain most of the new score drop by itself

## 2026-04-11 Selective Marlin focus gate infrastructure

- Added local runtime-side helpers to keep the selective loop reproducible:
  - `scripts/build_focus_eval_subset.py`
  - `scripts/summarize_eval_predictions.py`
  - `scripts/launch_gptq_medium_eval_remote.sh`
- New remote focused dataset:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_selective_focus10_v1.jsonl`
- Focus gate definition in practice:
  - `mcq` prompt tokens:
    - `103, 167, 202, 344, 589`
  - `cwe` prompt tokens:
    - `31433, 31697, 63430, 127483, 127738`
- Important baseline recovery:
  - no extra baseline eval was needed
  - the same focus rows were extracted from the existing stop-aligned `medium` predictions
  - latest rerun focus baseline:
    - `avg_score=0.71`
    - `total_output_tokens=202859`

## 2026-04-11 Candidate A `odown-gs64` focused eval

- Checkpoint:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1-odown-gs64`
- Eval run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/selA-odown64-focus_20260411_120315`
- Output:
  - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260411_120337`
- Summary:
  - `avg_score=0.49`
  - `total_output_tokens=267821`
- Key row outcomes:
  - `103 mcq`:
    - `score=1`
    - `out=3362`
  - `167 mcq`:
    - `score=0`
    - `out=26430`
  - `202 mcq`:
    - `score=1`
    - `out=2762`
  - `344 mcq`:
    - `score=0`
    - `out=13658`
  - `589 mcq`:
    - `score=0`
    - `out=24125`
  - `31433 cwe`:
    - `score=0.8`
    - `out=65199`
  - `31697 cwe`:
    - `score=1.0`
    - `out=718`
  - `63430 cwe`:
    - `score=0.2`
    - `out=65545`
  - `127483 cwe`:
    - `score=0.1`
    - `out=65545`
  - `127738 cwe`:
    - `score=0.8`
    - `out=477`
- Runtime interpretation:
  - serve compatibility was fine:
    - dense fallback launched cleanly
    - stop tokens still propagated as:
      - `['</s>', '<|im_end|>']`
  - behavior was not fine:
    - `167` and `589` remained wrong
    - `344` regressed from correct to wrong
    - `31433 cwe` exploded from a short healthy tail to `65199`
    - two long `cwe` rows still pinned at `65545`
- Gate decision:
  - Candidate A fails the focused gate and must not proceed to full `medium`

## 2026-04-11 Runtime next step after Candidate A

- Do not spend more eval time on Candidate A.
- Candidate B was queued immediately:
  - quant window:
    - `codex-soar:sel-skip-o-down64`
- Why:
  - Candidate A's biggest new runtime regression came from the route that still quantized the two suspect `o_proj` layers, just with `gs=64`
  - Candidate B keeps the `down_proj -> gs64` half and removes those `o_proj` layers from quantization entirely

## 2026-04-11 Runtime handoff into Candidate B focus gate

- Quant side delivered a runnable Candidate B checkpoint:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1-skip-o-down64`
- Remote GPU was idle after quant completion, so runtime work can proceed immediately without queue contention.
- Gate to run next:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_selective_focus10_v1.jsonl`
- Gate rule remains unchanged:
  - at least one of `167 mcq` or `589 mcq` must recover
  - `31433 cwe` and `31697 cwe` must stay short-tail and healthy
- If Candidate B passes:
  - promote straight to the full `25`-row `medium` eval
- If Candidate B fails:
  - move to Candidate C instead of spending more runtime on B

## 2026-04-11 Candidate B focus outcome and runtime decision

- Focus result:
  - `avg_score=0.66`
  - `total_output_tokens=285992`
- Runtime read:
  - positive:
    - `31433 cwe` stayed short:
      - `out=679`
    - `127738 cwe` also stayed short:
      - `out=884`
    - `589 mcq` recovered to:
      - `score=1.0`
      - `out=15361`
  - negative:
    - `167 mcq` still failed:
      - `score=0.0`
      - `out=48540`
    - `344 mcq` also failed:
      - `score=0.0`
      - `out=16106`
    - `31697 cwe` became a new long-tail failure:
      - `out=65537`
- Gate decision:
  - Candidate B does **not** proceed to full `medium`
  - the route is not runtime-safe enough because it trades one repaired `mcq` for a newly exploded `cwe`

## 2026-04-11 Runtime handoff into Candidate C

- Quant side delivered a new next candidate by immediately launching:
  - `Candidate C = odown-kq-gs64`
- Until that quant finishes, runtime stays idle by design to avoid overlapping heavy jobs.
- Next runtime action after save:
  - same `focus10` gate
  - same pass/fail rule

## 2026-04-11 Candidate C runtime blocker

- Candidate C never reached generation.
- Server load failed immediately inside the MiniCPM QKV loading path with:
  - `AssertionError: param_data.shape=torch.Size([32, 512]), loaded_weight.shape=torch.Size([64, 512])`
- Practical meaning:
  - this is not a subtle score regression
  - it is a route-level incompatibility between the widened `k/q -> gs64` selective config and the current SGLang `gptq_marlin` MiniCPM loader
- Runtime decision:
  - reject Candidate C without spending any more eval time on it
  - feed the compatibility failure back into the candidate ordering

## 2026-04-11 Runtime handoff into Candidate D

- Candidate D was launched immediately after Candidate C failed to load.
- Runtime expectation for D:
  - it should remain serve-safe because it only excludes modules
  - if quant saves cleanly, run the same `focus10` gate next

## 2026-04-11 Candidate D runtime gate started

- Quant side delivered a runnable Candidate D checkpoint:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1-skip-o-down`
- A first scripted launch attempt hit the usual local SSH/DNS flake, but a manual retry succeeded and the runtime gate is now live.
- Focus run:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/selD-skip-o-down-focus_manual_20260411_151312`
- Output dir:
  - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260411_151329`
- Confirmed runtime state:
  - server booted cleanly under the same dense fallback path:
    - `--quantization gptq_marlin`
    - `--attention-backend flashinfer`
    - `--force-dense-minicpm`
  - `eval_model.py` reached generation on the `10`-row focus gate
  - progress snapshot:
    - `Generating: 20%|██        | 2/10`

## 2026-04-11 Candidate D focus outcome and runtime decision

- Focus result:
  - `avg_score=0.52`
  - `total_output_tokens=151758`
- Runtime read:
  - positive:
    - this route remained serve-safe through the whole run
    - healthy short-tail `cwe` rows stayed short:
      - `31433 -> 543`
      - `31697 -> 1256`
      - `63430 -> 709`
  - negative:
    - both unstable `mcq` rows still failed:
      - `167 -> score=0.0, out=40721`
      - `589 -> score=0.0, out=23504`
    - `103 mcq` also regressed to:
      - `score=0.0`
- Gate decision:
  - Candidate D does **not** proceed to full `medium`
  - pure skip-on-`o/down` is runtime-safe, but it is not accuracy-safe enough to beat the stop-aligned baseline

## 2026-04-11 Runtime note after Candidate D rejection

- The next quant-side experiment is a revised Candidate C:
  - `odown-qkvall-gs64`
- Runtime rationale:
  - the first Candidate C failed before serving with an asymmetric `k/q` selective config
  - the revised route makes `q/k/v` uniform on the same targeted layers, which is the smallest useful test of whether the packed MiniCPM QKV path wants symmetric QKV quant settings
- Runtime remains idle while this new quant is in flight.

## 2026-04-11 Candidate C2 runtime gate

- Quant side delivered a new checkpoint:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1-odown-qkvall-gs64`
- Focus run launched:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/selC2-odown-qkvall-focus_20260411_213413`
- Most important read so far:
  - this revised symmetric `q/k/v -> gs64` route has not reproduced the old Candidate C load-time QKV shape assertion
  - it is already past the previous blocker and inside normal SGLang startup / weight loading under:
    - `--quantization gptq_marlin`
    - `--attention-backend flashinfer`
    - `--force-dense-minicpm`

## 2026-04-11 Candidate C2 runtime failure

- Final outcome:
  - no focus score
  - server exited during model load
- Concrete error:
  - `AssertionError: param_data.shape=torch.Size([32, 512]), loaded_weight.shape=torch.Size([64, 512])`
- Runtime read:
  - the symmetric `q/k/v -> gs64` variant got slightly farther into startup than the original Candidate C
  - but it still fails on the same underlying MiniCPM packed-QKV loader assumption
- Runtime decision:
  - reject Candidate C2
  - do not spend more eval time on QKV `gs64` variants unless we are willing to patch the loader path itself

## 2026-04-11 MiniCPM fused-QKV loader analysis update

- Refined read after tracing model and quant code:
  - dynamic GPTQ overrides in SGLang are matched against the runtime layer prefix
  - MiniCPM runtime layer prefix is fused `...self_attn.qkv_proj`, not separate `q_proj/k_proj/v_proj`
  - therefore a checkpoint whose `dynamic` only names `q_proj/k_proj/v_proj` can still miss the override at runtime even if the saved shard tensors are already `gs64`
- Why the failure shape is `32 vs 64`:
  - fused runtime `qkv_proj` likely allocates `qzeros/scales` with global `group_size=128`
  - selected checkpoint Q/K/V shards carry `gs64` tensor shapes
  - `load_qkv_weight` compares the narrowed fused target buffer to the loaded checkpoint shard and asserts on mismatch
- Local mitigation prepared on the quant side:
  - `/Users/ql/cursor/openbmb/scripts/quantize_gptq_w4a16.py` now synthesizes `qkv_proj` runtime alias rules for symmetric QKV dynamic patterns
  - this is meant to validate whether the blocker is metadata naming rather than deeper Marlin packing logic
- Runtime recommendation:
  - rerun one symmetric QKV selective checkpoint with the alias-enabled quant script before editing SGLang fused-QKV loader internals

## 2026-04-11 Fused-QKV dynamic fallback patch

- Alias-only metadata patch on the existing C2 checkpoint was not enough:
  - even with explicit `qkv_proj` alias in `quantize_config.json`, the server still failed at:
    - `load_qkv_weight`
    - `param_data.shape=[32,512]`
    - `loaded_weight.shape=[64,512]`
- Runtime patch applied:
  - `/root/autodl-tmp/sglang/python/sglang/srt/layers/quantization/utils.py`
  - local draft mirror:
    - `/Users/ql/cursor/openbmb/submission-drafts/w4a16-marlin-draft/sglang/python/sglang/srt/layers/quantization/utils.py`
- Patch behavior:
  - keep direct dynamic regex matching as-is
  - add fused-QKV fallback:
    - when `layer_name.endswith(".qkv_proj")` has no direct match
    - probe sibling shard names `q_proj/k_proj/v_proj`
    - if at least two matched shard overrides agree, reuse that override for fused `qkv_proj`
- Why this is the right level:
  - it does not depend on guessing the exact runtime alias to inject into checkpoint JSON
  - it reuses the shard rules that quantization already used
  - it targets the exact fusion asymmetry unique to MiniCPM runtime
- Result:
  - existing checkpoint `/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1-odown-qkvall-gs64` now loads successfully under:
    - `--quantization gptq_marlin`
    - `--attention-backend flashinfer`
    - `--force-dense-minicpm`
  - server reached ready state
  - focus10 eval started normal generation instead of crashing during model load
- Current status:
  - run:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/selC2rtfix-odown-qkvall-focus_20260411_220634`
  - last progress seen:
    - `7/10`

## 2026-04-11 Selective C runtime patch isolated into its own submission folder

- New packaging target:
  - `/Users/ql/cursor/openbmb/submission-w4a16-marlin-c-qkvall-gs64`
- This folder intentionally keeps the fused-QKV runtime fallback in:
  - `sglang/python/sglang/srt/layers/quantization/utils.py`
- Old draft intentionally does not keep that patch anymore:
  - `submission-drafts/w4a16-marlin-draft/.../utils.py` was reverted
- Reason:
  - keep stop-aligned draft and selective `C` draft separate
  - make packaging handoff unambiguous for the submission thread

## 2026-04-12 Selective C full-run status

- Full public-set check launched in tmux for the runtime-patched selective `C` checkpoint:
  - tmux:
    - `codex-soar:eval-selC2rtfix-full`
  - run dir:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/selC2rtfix_full_20260412_012318`
  - output dir:
    - `/root/autodl-tmp/SOAR-Toolkit/outputs/20260412_012501`
- Status at first live check:
  - server ready
  - generation active on `150` samples
  - GPU in use
  - no final `predictions.jsonl` yet, so this check confirms landing rather than final score

## 2026-04-14 Runtime KV probe on isolated fusion package

- Probe package:
  - `/Users/ql/cursor/openbmb/submission-w4a16-marlin-fusion-kvcache-probe-v1`
- Intention:
  - runtime-only KV exploration on the existing W4A16 weight
  - no winning-package edits
  - no fusion broadening
- New runtime entrypoints:
  - `run_probe_server.sh`
    - probe-only launcher
    - validates probe-local `sglang.launch_server`
    - writes normalized `launch_context.json`
  - `run_kv_probe_smoke.sh`
    - fixed smoke wrapper for:
      - baseline `flashinfer + auto`
      - fp8 `flashinfer + fp8_e4m3`
      - fp4 `triton + fp4_e2m1`
    - writes per-case `summary.json` with failure-stage and first-token fields
- Remote probe root:
  - `/root/autodl-tmp/codex-drafts/w4a16-marlin-fusion-kvcache-probe-v1`
- Checkpoint used for all three cases:
  - `/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-hybrid16g-v1`
- Smoke outcomes:
  - baseline:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/kv_probe_20260414_014421_baseline`
    - `success`
    - first token:
      - `<think>`
    - `ttft_ms = 840.485`
    - `request_latency_ms = 1075.93`
    - KV dtype line:
      - `torch.float16`
  - fp8:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/kv_probe_20260414_014450_fp8`
    - `success`
    - first token:
      - `<think>`
    - `ttft_ms = 856.822`
    - `request_latency_ms = 1088.187`
    - KV dtype line:
      - `torch.float8_e4m3fn`
  - fp4:
    - `/root/autodl-tmp/SOAR-Toolkit/test_results/kv_probe_20260414_014318_fp4`
    - `failed`
    - failure stage:
      - `server_boot`
    - precise blocker after `Load weight end`:
      - `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'`
      - site:
        - `sglang/srt/mem_cache/memory_pool.py`
- Runtime takeaway:
  - `flashinfer + auto` and `flashinfer + fp8_e4m3` are both valid same-weight runtime paths in the isolated probe package
  - the current `triton + fp4_e2m1` route is blocked specifically by FP4 KV buffer initialization, so the next runtime step is a targeted SGLang backend/memory-pool fix rather than more environment work

## 2026-04-14 fp8 fast gate result on the same isolated probe package

- Probe package:
  - `/Users/ql/cursor/openbmb/submission-w4a16-marlin-fusion-kvcache-probe-v1`
- Runtime args held fixed at:
  - `flashinfer + kv_cache_dtype=fp8_e4m3`
  - `gptq_marlin + float16`
- Discarded first attempt:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/fp8_fast_eval_20260414_015913`
  - reason:
    - it went through official `eval_model.py`
    - log again showed `max_tokens=65536`
    - so it reproduced the old uncapped fast-harness bug and is not a valid bounded comparison
- Valid bounded fast gate:
  - `/root/autodl-tmp/SOAR-Toolkit/test_results/fp8_fast_gate_20260414_020423`
  - harness:
    - `/root/autodl-tmp/SOAR-Toolkit/.codex_mini_eval_task_caps.sh`
    - full curated fast file with all rows enabled
    - task caps:
      - `mcq=64`
      - `qa=128`
      - `niah=128`
      - `fwe=256`
      - `cwe=256`
- Outcome:
  - `avg_score=32.22%`
  - `pass=2`
  - `part=2`
  - `fail=5`
  - `empty=0/9`
- Task pattern:
  - short `qa` and short `niah` passed
  - long `qa` / `niah` failed at their `128` caps
  - both `cwe` rows were partial at `256`
  - both `fwe` rows failed, including one shorter row that stopped early
  - `mcq` also failed at `64`
- Runtime judgement:
  - fp8 KV remains a good feasibility result
  - but on the bounded fast gate it does not improve quality and currently lands near stock-base territory rather than the stronger earlier GPTQ fast anchors
- Next useful comparison if this line continues:
  - run the exact same bounded fast gate under `kv_cache_dtype=auto` in the same probe package for a clean A/B
- 2026-04-14 03:37 CST: Native MiniCPM backend probe on `rtx6000-2` advanced materially after a probe-only compat patch. `minicpm_flashinfer` with exact winning checkpoint `/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-v1` no longer dies at fused QKV load. The fix was to force attention modules unquantized when dynamic config marks all attention shards skipped, matching the no-attn checkpoint semantics. Native boot now reaches `Load weight end`, `Using KV cache dtype: torch.float16`, and allocates both mamba and KV memory pools before failing on a new blocker: `ModuleNotFoundError: No module named 'tilelang'` in `sglang/srt/layers/attention/minicpm_fuse_kernel.py`. This narrows the native-backend path blocker from checkpoint structure to missing runtime dependency/kernel path.
- 2026-04-14 03:55 CST: Installed `tilelang==0.1.8` into the probe eval env on `rtx6000-2` and retried native MiniCPM backends using checkpoint `/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-hybrid16g-v1`. Both `minicpm_flashattn` and `minicpm_flashinfer` now boot all the way to `The server is fired up and ready to roll!`, return `GET /v1/models` successfully, and only fail on the first short chat request. The new shared blocker is in `sglang/python/sglang/srt/layers/attention/minicpm_backend.py`, where `update_batch_for_sparse()` computes `seqlen_q_sparse_tensor.max()` on an empty tensor and crashes with `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0`. This strongly suggests the remaining native-backend issue is a sparse metadata / empty-prefill edge case, not kernel availability, flashinfer-vs-flashattn choice, or no-attn checkpoint incompatibility.
- 2026-04-14 10:30-10:46 CST: Continued runtime-side A/B on `rtx6000-2` using the exact winning checkpoint `/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-v1` and the isolated probe package `/Users/ql/cursor/openbmb/submission-w4a16-marlin-fusion-kvcache-probe-v1`.
  - `flash_attn` viability:
    - our packaged wheel `flash_attn-2.8.3+cu128sm120` still fails to import on this machine with `libstdc++.so.6: version 'CXXABI_1.3.15' not found`
    - so `flash_attn` is not a practical runtime candidate here without a newer host libstdc++ or a rebuilt wheel/runtime stack
  - native MiniCPM sparse extension:
    - rebuilt `/root/autodl-tmp/sglang/3rdparty/sparse_kernel` with `sm_120`
    - the previous site-packages `.so` only exposed `compute_90/sm_90`
    - after rebuilding with `nvcc` visible in `PATH`, the new `.so` exposes `compute_120/sm_120`
    - this removed the earlier `no kernel image is available for execution on the device` blocker for native sparse decode
  - probe-only runtime patches:
    - `sglang/python/sglang/srt/mem_cache/common.py`
      - skip zero-length sparse writes
    - `sglang/python/sglang/srt/mem_cache/memory_pool.py`
      - early-return on zero-sized sparse write buffers
    - `sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
      - only pass `forward_batch=` into replay metadata init when the backend signature supports it
  - exact long-prompt A/B (`prompt_chars=230999`, longest public `cwe`):
    - exact dense baseline (`flashinfer` + explicit `--force-dense-minicpm`):
      - `11359.280 ms`
      - `/tmp/codex_ab3_flashinfer_dense.log`
    - native `minicpm_flashinfer`:
      - `18307.803 ms`
      - `/tmp/codex_ab3_minicpm_flashinfer.log`
    - conclusion:
      - native MiniCPM route is now functionally correct on the exact winning checkpoint, but is about `61%` slower on this representative long prompt
  - exact dense `chunked_prefill_size` sweep on the same prompt:
    - `4096`: `11862.052 ms`
    - `8192`: `11320.653 ms`
    - `16384`: `11637.716 ms`
    - conclusion:
      - current winning/default `8192` still looks best
  - exact dense `triton` after the replay-signature compat patch:
    - `14516.726 ms`
    - `/tmp/triton_dense_exact_patch.log`
    - conclusion:
      - `triton` is no longer crashing, but is about `28%` slower than exact dense `flashinfer` on the same long prompt
  - overall runtime takeaway:
    - for the current winning checkpoint, the exact dense `flashinfer` route still dominates the native MiniCPM path and the current `triton` dense path on representative long prompts
    - near-term runtime wins are unlikely to come from native sparse backend or generic `triton` unless a deeper backend change lands
- 2026-04-14 10:57-11:15 CST: Tightened the runtime A/B so it really matches the winning serve flags instead of the lighter probe defaults.
  - exact strict dense vs native (`--disable-radix-cache --chunked-prefill-size 8192 --skip-server-warmup --disable-cuda-graph --quantization gptq_marlin --dtype float16`, same longest public `cwe` prompt):
    - `flashinfer_dense_exactfull`:
      - `11900.422 ms`
      - `/tmp/flashinfer_dense_exactfull.log`
    - `minicpm_flashinfer_exactfull`:
      - `18535.713 ms`
      - `/tmp/minicpm_flashinfer_exactfull.log`
    - conclusion:
      - native MiniCPM stays slower even under exact winning flags, by roughly `56%`
  - probe env cleanup:
    - found a broken local `torch 2.11` in the probe eval env shadowing the inherited official `torch 2.9.1+cu128`
    - child processes then failed with `libtorch_global_deps.so` missing
    - removed the local `torch/torchgen` tree so the probe env again resolves torch through `soar_official_base_env.pth`
  - strict exact `triton` after env cleanup:
    - `triton_dense_exactfull`:
      - `48600.792 ms`
      - `/tmp/triton_dense_exactfull.log`
    - conclusion:
      - with the real winning serve flags, `triton` is over `4x` slower than exact dense `flashinfer`
  - strict CUDA graph knob on exact dense `flashinfer`:
    - `flashinfer_dense_cudagraph_on` (omit `--disable-cuda-graph`):
      - `42482.739 ms`
      - `/tmp/flashinfer_dense_cudagraph_on.log`
    - `flashinfer_dense_cudagraph_off`:
      - `12064.137 ms`
      - `/tmp/flashinfer_dense_cudagraph_off.log`
    - conclusion:
      - for the current winning checkpoint and long-prompt workload, keeping `--disable-cuda-graph` is strongly preferred; enabling CUDA graph is dramatically slower
- 2026-04-14 12:02-12:04 CST: Measured a true fusion delta instead of another backend/flag no-op. In the probe `sglang` tree, added an env-gated switch in `sglang/python/sglang/srt/models/minicpm.py` so `SGLANG_DISABLE_MINICPM_ADD_RMSNORM_FUSION=1` forces the MiniCPM decoder block to split `residual add` and `RMSNorm` instead of using the current fused `self.input_layernorm(hidden_states, residual)` / `self.post_attention_layernorm(hidden_states, residual)` path. With the exact winning checkpoint, exact winning serve flags, and the longest public `cwe` prompt (`230999` chars):
  - fused baseline:
    - result dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/fusion_add_rmsnorm_20260414_120234/baseline_fused`
    - `ttft_ms = 11583.803`
    - `request_latency_ms = 11703.271`
  - unfused add+rmsnorm:
    - result dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/fusion_add_rmsnorm_20260414_120234/unfused_add_rmsnorm`
    - `ttft_ms = 11668.396`
    - `request_latency_ms = 11798.423`
  - conclusion:
    - fused add+RMSNorm is real and already active on the winning runtime path
    - removing it only hurts by about `0.7%` on TTFT and `0.8%` on total request latency on this representative long prompt
    - so this Week 2 / Week 4 fusion point is already basically exhausted and is not the missing `48.2 -> 50+` lever by itself
- 2026-04-14 12:11-12:14 CST: Measured the fused `SiluAndMul` (SwiGLU) MLP path on the same exact-winning route. In `/Users/ql/cursor/openbmb/submission-w4a16-marlin-fusion-kvcache-probe-v1/sglang/python/sglang/srt/models/minicpm.py`, added an env-gated switch so `SGLANG_DISABLE_MINICPM_SILU_AND_MUL_FUSION=1` replaces `self.act_fn(gate_up)` with native `F.silu(gate_up[..., :d]) * gate_up[..., d:]`. Same exact winning checkpoint, same exact winning serve flags, same longest public `cwe` prompt (`230999` chars):
  - fused baseline:
    - result dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/fusion_silu_and_mul_20260414_121125/baseline_fused`
    - `ttft_ms = 11539.797`
    - `request_latency_ms = 12075.750`
  - unfused silu+mul:
    - result dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/fusion_silu_and_mul_20260414_121125/unfused_silu_and_mul`
    - `ttft_ms = 11630.933`
    - `request_latency_ms = 12164.006`
  - conclusion:
    - fused `SiluAndMul` is also already active and correct on the winning runtime path
    - removing it hurts by only about `0.8%` on TTFT and about `0.7%` on total request latency
    - so the obvious MLP fusion point is also already basically exhausted
- 2026-04-14 12:27-12:30 CST: Tested the native MiniCPM hint `--dense-as-sparse` on `rtx6000-2` using the exact winning checkpoint, native `minicpm_flashinfer`, and the same longest public `cwe` prompt. Result dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/dense_as_sparse_20260414_122833`
  - native default:
    - `ttft_ms = 18591.756`
    - `request_latency_ms = 19352.760`
  - native + `--dense-as-sparse`:
    - `ttft_ms = 18318.924`
    - `request_latency_ms = 19071.291`
  - conclusion:
    - forcing short/native batches down the sparse path gives a small native-only win (~`1.5%`)
    - but it does not close the much larger gap to exact winning dense `flashinfer`
- 2026-04-14 16:20-16:45 CST: Closed the gap between the disappointing local FP8-KV bench story and the new official evaluation result for the exact-winning sibling package with `full fp8 KV`.
  - New official result reported by the user for `submission-w4a16-marlin-publichybrid-noattn-skipdown151617-fp8full-v1`:
    - `acc = 99.14`
    - `acc_ori = 79.31`
    - `final_score = 48.8`
    - `benchmark_duration = {S1: 571.16, S8: 695.67, Smax: 1160.96}`
  - Interpretation:
    - the runtime-only `--kv-cache-dtype fp8_e4m3` sibling is real and slightly faster on the official machine
    - the earlier local negative result should therefore be treated as a bench-mismatch warning, not as proof that full-FP8 KV is useless
  - Root-cause analysis of the local mismatch:
    - `proxy11` was always a quick relative A/B proxy, not an official-like speed harness
    - more importantly, `perf_public_set.jsonl` lacks `model_response`, while official `bench_serving.sh` derives decode length from the assistant text inside the speed JSONL
    - therefore any local speed bench that reused raw public eval rows without synthesizing assistant text length was suppressing decode pressure and could miss KV-cache wins
  - Harness fix implemented:
    - `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py` now has a new `public-speed` profile
    - it generates `/root/autodl-tmp/SOAR-Toolkit/bench_speed_public_full.jsonl` from `perf_public_set.jsonl`
    - it uses the target model tokenizer to synthesize deterministic assistant text whose token length matches each row’s public `completion_tokens`
    - it also reorders rows in a stable round-robin over `mcq / qa / niah / fwe / cwe` after sorting each task by prompt length, so `S8/Smax` are no longer accidentally front-loaded by one task family
    - optional `--public-speed-completion-cap` was added for bounded debug runs
  - Updated guidance:
    - use `proxy11` only for cheap relative smoke/triage
    - use `public-speed` for submission-facing runtime A/B, especially for KV-cache or scheduler experiments
- 2026-04-14 17:00-18:55 CST: Tightened the new `public-speed` harness after actually trying to run it on the exact winning route.
  - Exact-winning runtime launch path:
    - `flashinfer + force-dense-minicpm + gptq_marlin + float16 + disable-radix-cache + disable-cuda-graph`
    - server must be launched from the probe/editable env, not the stock shared env, otherwise the fused-QKV checkpoint fails to load with:
      - `KeyError: 'model.layers.0.self_attn.qkv_proj.weight'`
  - Runner upgrade:
    - `/Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py` now honors:
      - `SOAR_REMOTE_HOST`
      - `SOAR_REMOTE_SOAR_ROOT`
      - `SOAR_REMOTE_SGLANG_ROOT`
      - `SOAR_REMOTE_PYTHON`
      - `SOAR_REMOTE_ENV`
    - this makes the speed harness usable with isolated probe/submission envs instead of only the shared stock env
  - First public-speed reality check:
    - using all 150 public rows with raw public `completion_tokens` generated:
      - `prompt_tokens_total = 8,644,166`
      - `completion_tokens_bench_actual_total = 246,942`
    - on the exact-winning route, that immediately looked wrong for an official-like speed proxy:
      - `S1` decode sat around `~60 tok/s`
      - uncapped `S1` would therefore drift toward hour-scale runtime
    - conclusion:
      - direct reuse of raw public `completion_tokens` overweights decode and is not a faithful hidden-speed proxy
  - Second harness upgrade:
    - added `--public-speed-per-task-bucket`
    - the public-speed generator can now stably subsample one or more rows per `(task, prompt-bucket)` instead of always keeping all public rows
    - together with `--public-speed-completion-cap 128`, this yields a much smaller public-derived speed proxy:
      - `completion_tokens_bench_actual_total = 18,424`
      - much closer to the official `S1≈571s` scale than the uncapped 246k-output version
  - Operational blocker:
    - repeated SSH transport resets (`kex_exchange_identification: read: Connection reset by peer`) hit both `rtx6000-2` and `rtx6000-1`
    - so the final bounded public-proxy A/B (`fp16` vs `full fp8`) did not complete in this pass
  - Current recommendation:
    - for the next stable window, run:
      - exact winning route
      - `--profile public-speed`
      - `--public-speed-completion-cap 128`
      - `--public-speed-per-task-bucket 1`
      - server launched with probe-env `SOAR_REMOTE_PYTHON`
- 2026-04-14 19:01-19:04 CST: Ran the requested quick “most explosive” local check on `rtx6000-2` after the public-speed harness fixes.
  - Setup:
    - exact winning runtime route
    - server launched from the probe editable env
    - `Smax` only
    - public-derived proxy with:
      - `--public-speed-completion-cap 128`
      - `--public-speed-per-task-bucket 1`
    - resulting speed set metadata:
      - `count = 9`
      - `prompt_tokens_total = 608,903`
      - `completion_tokens_bench_actual_total = 1,115`
  - Results:
    - `fp16_auto`
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260414_190147_fp16_auto_publicspeed_smax_quick`
      - `Smax = 51.04s`
    - `fp8_full`
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/candidate_benches/20260414_190339_fp8_full_publicspeed_smax_quick`
      - `Smax = 51.95s`
  - Interpretation:
    - this quick local `Smax` probe still does **not** reproduce the official `fp8_full` win
    - local delta is small but negative for FP8 (`+0.91s`, about `+1.8%`)
    - so the mismatch remains real:
      - official machine + hidden speed set favors `fp8_full`
      - our current public-derived local proxy still slightly disfavors it
- 2026-04-14 19:09-19:23 CST: Verified the stronger KV-pressure condition requested by the user using the probe-local `kv_pressure_smax` path on `rtx6000-2`.
  - This route is distinct from `bench_serving.sh`: it polls live `/get_load` and is meant to answer “did we really push live KV/token load past the intended threshold?”
  - FP16 exact-winning route:
    - `flashinfer + force-dense-minicpm + gptq_marlin + float16 + disable-radix-cache + disable-cuda-graph`
    - parsed `max_total_num_tokens = 6,301,509`
  - First attempt:
    - `oversubscribe_ratio = 2.1`
    - `min_prompt_tokens = 100000`
    - `output_cap = 64`
    - observed live `/get_load` plateau only around `~12.50M` tokens, about `1.98x` the FP16 capacity reference
    - diagnosis: the active request mix still included enough `112K / 119K` prompts that the live average prompt size was slightly too low
  - Tightened attempt:
    - `oversubscribe_ratio = 2.1`
    - `min_prompt_tokens = 127500`
    - `output_cap = 64`
    - observed live `/get_load` on the FP16 route:
      - `num_tokens = 12,757,786`
      - ratio vs FP16 capacity reference: `2.02456x`
    - so this pass really did exceed `2x` the FP16 KV capacity line
  - FP8 cross-check with the same FP16-based target:
    - `fp8 max_total_num_tokens = 12,603,019`
    - launched with `--capacity-ref-tokens 6301509` so the workload target remained tied to the FP16 reference
    - observed live `/get_load`:
      - `num_tokens = 12,753,083`
      - ratio vs FP16 reference: `2.02381x`
      - ratio vs FP8 server capacity itself: about `1.0119x`
  - Updated lesson:
    - to truly push beyond `2x` the FP16 capacity reference, raising `oversubscribe_ratio` alone was not enough
    - the workload also had to be restricted to the very top `127.5K+` prompt band so the live active set stayed dense enough
- 2026-04-14 19:54-20:26 CST: Corrected the interpretation of the earlier KV-pressure metric and finished a stronger synthetic serving A/B on `rtx6000-2`.
  - Source-level correction:
    - `Scheduler.get_load()` adds waiting-queue `req.seqlen` values to the current token count.
    - therefore the earlier `>2x fp16 capacity` result was a **queued live-load** observation, not a proof that resident KV occupancy itself exceeded `2x`.
  - `bench_one_batch` attempt:
    - exact winning route with `flashinfer`, `160k input`, `256 output`, `batch=39` on FP16
    - failed at FlashInfer prefill-planner workspace allocation before any real KV capwall comparison
    - bumping `SGLANG_FLASHINFER_WORKSPACE_SIZE` to `2GB` did not remove that planner overflow
  - Switched to a harsher server-style synthetic benchmark:
    - `bench_serving --backend sglang --dataset-name random-ids --num-prompts 256 --random-input-len 32768 --random-output-len 2048 --random-range-ratio 0 --max-concurrency 256 --disable-tqdm`
    - kept the exact winning server route except for `kv_cache_dtype`
    - uncovered one more harness nuance:
      - without `--tokenize-prompt`, the `random-ids` path decodes ids back to text before sending them to the server
      - this means the actual server-side prompt lengths were lower than the nominal `32K / 2K`
      - the FP16 vs FP8 comparison is still fair, but the next stricter rerun should add `--tokenize-prompt`
  - FP16 result:
    - dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/smax_synth32k2kfix_20260414_200911_fp16`
    - duration `384.0669s`
    - total throughput `10835.05 tok/s`
    - peak log state:
      - full token usage `0.68`
      - mamba usage `0.50`
      - running req `255`
      - queue req `229`
  - Full-FP8 result:
    - dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/smax_synth32k2kfix_20260414_202010_fp8full`
    - duration `373.5966s`
    - total throughput `11138.71 tok/s`
    - peak log state:
      - full token usage `0.34`
      - mamba usage `0.50`
      - running req `255`
      - queue req `238`
  - Interpretation:
    - `full fp8 KV` did win under this stronger synthetic serving load:
      - duration improved by `10.47s` (`-2.73%`)
      - total throughput improved by `+2.80%`
    - the hybrid bottleneck split is now much clearer:
      - FP8 relieves the **full-attention KV** side substantially
      - but it does **not** relieve the **mamba/lightning-state** side
    - this is a plausible reason the local gain often looked smaller/noisier than the official machine even though the official submission still improved.
- 2026-04-15 21:30-21:35 CST: `fp4_e2m1` KV cache resumed on `rtx6000-2` with the plain `99+` winning checkpoint.
  - Reproduced the old failure exactly:
    - hybrid MiniCPM used `HybridLinearKVPool -> MHATokenToKVPool`
    - that path allocates `torch.zeros(..., dtype=torch.float4_e2m1fn_x2)` and fails at boot with:
      - `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'`
  - Minimal runtime patch in the isolated probe tree:
    - `HybridLinearKVPool` now routes FP4-dtype hybrid models to the existing `MHATokenToKVPoolFP4 / MLATokenToKVPoolFP4` implementations
  - New behavior after the patch:
    - default/uncapped FP4 boot succeeds
    - but first decode still OOMs because `get_key_buffer/get_value_buffer` dequantize the entire FP4 pool into `bf16`
  - Important working smoke runs:
    - `max_total_tokens=4194304`:
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/fp4_caprun_20260415_213351`
      - HTTP 200 short decode succeeded
    - `max_total_tokens=8388608`:
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/fp4_caprun8m_20260415_213454`
      - HTTP 200 short decode also succeeded
  - Bottom line:
    - FP4 KV is now startup-feasible on the hybrid route
    - the remaining blocker is not init anymore; it is the whole-buffer dequant path in the Triton full-attention backend
- 2026-04-15 22:04-22:05 CST: isolated `SimpleGLAAttnBackend` on idle `rtx6000-1`.
  - Backend identity:
    - this MiniCPM lightning path is Triton-backed, not flashinfer and not a standalone custom CUDA kernel path
    - `SimpleGLAAttnBackend` calls `fused_recurrent_simple_gla` or `chunk_simple_gla`
    - the vendored `fla.ops.simple_gla.parallel`, `fla.ops.simple_gla.chunk`, and `fla.ops.common.fused_recurrent` are Triton implementations
  - MiniCPM-like microbench (`H=32`, `K=128`, `V=128`, `float16`):
    - recurrent `B=32,T=1`: `0.091 ms` kernel-only, `0.123 ms` with simulated state gather/writeback
    - recurrent `B=8,T=32`: `0.091 ms` kernel-only, `0.117 ms` with simulated state gather/writeback
    - chunk `B=8,T=64`: `0.132 ms` kernel-only, `0.149 ms` with simulated state gather/writeback
    - chunk `B=4,T=256`: `0.118 ms` kernel-only, `0.148 ms` with simulated state gather/writeback
    - chunk `B=1,T=8192`: `0.471 ms` kernel-only, `0.485 ms` with simulated state gather/writeback
  - Reading:
    - isolated lightning kernels are not the obvious bottleneck
    - short recurrent calls show noticeable but not dominant state-IO overhead
    - for longer chunked prefill-style calls, the state-IO tax becomes very small relative to kernel time
- 2026-04-20 10:55-11:05 CST: explored `SimpleGLAAttnBackend` more aggressively on `rtx6000-1` and landed a first probe-side optimization patch.
  - Added reusable harness:
    - `/Users/ql/cursor/openbmb/submission-w4a16-marlin-fusion-kvcache-probe-v1/bench_simple_gla_variants.py`
  - Remote benchmark artifacts:
    - `simple_gla_state_b8_random_20260420.json`
    - `simple_gla_state_b32_random_20260420.json`
    - `simple_gla_state_b32_contig_20260420.json`
    - `simple_gla_full_b8_t64_random_20260420.json`
    - `simple_gla_full_b8_t256_ragged_random_20260420.json`
    - `simple_gla_threshold_b8_uniform_random_20260420.json`
    - `simple_gla_threshold_b8_ragged_random_20260420.json`
  - Bench conclusions:
    - `state[idx]` already returns contiguous tensors in the tested patterns; `.contiguous()` is effectively redundant here
    - `index_copy_` is consistently faster than advanced-index assignment for temporal-state writeback
    - the best chunk-path combination in the full-path microbench is `index_select + index_copy_`
    - Blackwell crossover for `fused_recurrent_simple_gla` vs `chunk_simple_gla` is around `seq_len ~= 80`
      - `64`: fused wins
      - `72`: fused still wins narrowly
      - `80+`: chunk wins
    - `cu_seqlens_cpu` on the chunk path did not produce meaningful gains in this setup
  - Probe patch in `hybrid_linear_attn_backend.py`:
    - cache and reuse `mamba_indices`
    - gather initial state with `index_select`
    - write final state back with `index_copy_`
    - map negative/padding indices to the reserved extra state slot
    - bump fused/chunk cutoff from `<64` to `<80`
  - Validation:
    - first smoke exposed an `index_copy_` `int64` index requirement; fixed by promoting safe indices to `torch.int64`
    - second smoke succeeded:
      - `kv_probe_20260420_110420_linear-opt-smoke2`
      - `flashinfer + auto KV`, server boot + short decode OK
  - Working hypothesis:
    - the linear backend’s near-term headroom is mostly in state bookkeeping and dispatch threshold tuning rather than Triton kernel surgery
- 2026-04-20 11:35-11:52 CST: validated the linear-backend patch with a proper detached extreme A/B on `rtx6000-1`.
  - Shared serving setup:
    - weight: `/root/autodl-tmp/models-gptq-w4a16-py310-attnqkv-allowlist-v1-gate0-fixorder`
    - backend: `flashinfer`
    - flags: `--force-dense-minicpm --chunked-prefill-size 8192 --disable-radix-cache --disable-cuda-graph --cuda-graph-max-bs=1`
    - workload: `random-ids`, `256 x 32k/2k`, `max_concurrency=256`
  - Baseline:
    - tree: `...probe-v1-linear-baseline`
    - run: `extreme_attnqkv-gate0-linear-baseline_20260420_113533`
    - `duration=384.01s`
    - `total tok/s=10836.62`
    - `output tok/s=659.41`
  - Patched:
    - tree: `...probe-v1`
    - run: `extreme_attnqkv-gate0-linear-patched2_20260420_114449`
    - `duration=385.20s`
    - `total tok/s=10803.04`
    - `output tok/s=657.36`
  - Interpretation:
    - end-to-end result is effectively flat, with patched slightly behind baseline (`~0.31%` lower total throughput)
    - the microbench improvements in state gather/writeback and the `80`-token cutoff shift do not materially move the whole serving stack on this workload
    - likely reasons:
      - linear backend is not the dominant bottleneck in this production-shaped case
      - wins in per-layer bookkeeping are drowned out by full-attn, scheduler, and macro-level request dynamics
    - recommendation:
      - do not upstream this patch as a serving default
      - keep the harness around for future experiments, but shift optimization attention back to higher-leverage bottlenecks
- 2026-04-20 12:11-12:12 CST: validated a lighter hybrid FP8 KV route on `rtx6000-2`.
  - Remote probe tree was partially stale, so I synced the minimal mixed-KV files from local:
    - `run_probe_server.sh`
    - `run_kv_probe_smoke.sh`
    - `server_args.py`
    - `model_runner_kv_cache_mixin.py`
    - `flashinfer_backend.py`
  - Smoke config:
    - weight: `/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-v1`
    - backend: `flashinfer`
    - `kv_cache_dtype=fp8_e4m3`
    - `kv_cache_protect_layers=29,30,31`
    - plus dense-winning flags: `--force-dense-minicpm --chunked-prefill-size 8192 --disable-radix-cache --disable-cuda-graph`
  - Result:
    - run dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/kv_probe_20260420_121134_fp8_tail3`
    - boot + `/v1/models` + short decode all succeeded
    - `ttft_ms=10109.626`
    - `request_latency_ms=10226.461`
  - Server-side confirmation:
    - mixed KV path was active
    - requested protected layers `[29, 30, 31]`
    - effective protected full-attn layers `[29, 30, 31]`
    - no ignored layers
  - Interpretation:
    - a lighter tail-only BF16 protect set is viable on top of `full fp8 KV`
    - this is a cleaner next candidate than the earlier heavier `0,29,30,31` protect set
- 2026-04-20 12:07-12:35 CST: completed the first Nsight-guided pass over the MiniCPM `SimpleGLA` backend on `rtx6000-1`.
  - Tooling status:
    - usable `nsys` binary: `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`
    - `ncu` launch works, but counter collection is blocked by `ERR_NVGPUCTRPERM`; this machine cannot currently provide occupancy / stall / L2 metrics from Nsight Compute
    - extended local tooling with:
      - `bench_simple_gla_variants.py` support for `fused_chunk_simple_gla`
      - `profile_simple_gla_case.py` for isolated NVTX-instrumented runs
      - detached `nsys` / `ncu` helper scripts
    - fixed the detached `nsys` runner so it:
      - handles `.nsys-rep`
      - exits cleanly when `nsys` kills the traced server after a bounded capture window
  - System-level `nsys` on the baseline serving tree with attnqkv-gate0 weight:
    - prefill-heavy window:
      - run: `nsys_extreme_attnqkv-gate0-nsys-prefill_20260420_120743`
      - total GPU kernel time: `44.288s`
      - `SimpleGLA` share:
        - `chunk_fwd_kernel_o = 596.8ms`
        - `chunk_fwd_kernel_h = 409.8ms`
        - combined `~2.27%`
      - conclusion: prefill-side `SimpleGLA` is not a first-order bottleneck
    - decode-heavy window:
      - run: `nsys_extreme_attnqkv-gate0-nsys-decode_20260420_121605`
      - total GPU kernel time: `43.566s`
      - `fused_recurrent_fwd_kernel = 7.458s` (`~17.1%`)
      - neighboring large decode-window GPU ops:
        - `index_elementwise_kernel = 7.874s`
        - `vectorized_gather_kernel = 7.025s`
      - conclusion: decode-side `SimpleGLA` matters, but state-indexing GPU work is similarly large
  - Isolated NVTX-instrumented `nsys` runs:
    - decode-shaped fused recurrent:
      - run: `nsys_simple_gla_fusedio_20260420_122753`
      - `variant=fused`, `batch=128`, `seq_len=1`, `include_state_io=true`
      - `~1.09ms/iter`
      - GPU-kernel split over 25 profiled iterations:
        - `vectorized_gather_kernel ~= 9.02ms`
        - `fused_recurrent_fwd_kernel ~= 8.62ms`
        - `index_copy/index_elementwise ~= 9.17ms`
      - read: wrapper-side state gather/writeback is as important as the recurrent kernel in decode mode
    - chunk, medium-length batched prefill shape:
      - run: `nsys_simple_gla_chunkio_20260420_122832`
      - `variant=chunk`, `batch=128`, `seq_len=64`, `include_state_io=true`
      - `~1.61ms/iter`
      - split:
        - `chunk_fwd_kernel_h ~= 13.87ms`
        - `chunk_fwd_kernel_o ~= 7.22ms`
        - `state_gather ~= 8.85ms`
        - `state_writeback ~= 8.35ms`
      - read: in this regime, `h` is heavier than `o` and state IO remains material
    - chunk, real `8192` prefill chunk shape:
      - run: `nsys_simple_gla_chunkio_b1t8192_20260420_123053`
      - `variant=chunk`, `batch=1`, `seq_len=8192`, `include_state_io=true`
      - `~0.54ms/iter`
      - split:
        - `chunk_fwd_kernel_o ~= 3.58ms`
        - `chunk_fwd_kernel_h ~= 2.23ms`
        - `state_gather + state_writeback < 0.10ms`
      - read: for the actual `8192` prefill chunk, `o` is the heavier kernel and state IO is basically irrelevant
  - Threshold sweep at `batch=128`:
    - output: `simple_gla_threshold_b128_20260420_122557.json`
    - winners by sequence length:
      - `fused` at `<=8`
      - `fused_chunk` around `16-32`
      - `chunk/chunk_cpu` from `48+`
    - implication:
      - `fused_chunk_simple_gla` is not a universal replacement; it only helps in a narrow mid-length band
  - Optimization direction after profiling:
    - stop treating prefill-side `chunk_simple_gla` as the main target
    - if continuing on this backend, focus on decode-side `fused_recurrent` plus state gather/writeback together
    - without `ncu` counter access, the current evidence ceiling on this machine is `nsys + isolated NVTX`
- 2026-04-20 12:43-13:58 CST: validated `mamba_ssm_dtype=bfloat16` on the current `attnW+mlpW gate0-fixorder` baseline with a real official full eval on `rtx6000-1`.
  - Patched the attnQKV tree with the same minimal dtype-cast writeback fix already proven on the winning route:
    - cast `final_state` to `layer_cache.temporal.dtype` before saving it in `SimpleGLAAttnBackend`
  - Run:
    - dir: `/root/autodl-tmp/SOAR-Toolkit/test_results/official_attnqkv_gate0_fixorder_mamba_bf16_20260420_124628`
    - weight: `/root/autodl-tmp/models-gptq-w4a16-py310-attnqkv-allowlist-v1-gate0-fixorder`
  - Memory/capacity effect vs float32 gate0 baseline:
    - float32:
      - `ssm_state size: 24.05GB`
      - `max_total_num_tokens=6369605`
    - bf16:
      - `ssm_state size: 12.02GB`
      - `max_total_num_tokens=8004353`
    - net:
      - `~50%` lower state memory
      - `~25.7%` higher token capacity
  - Final metrics:
    - `Average Score: 81.69%` vs baseline `81.11%`
    - `Total Duration: 4270.89 s` vs baseline `3767.77 s`
    - `Total Tokens Out: 1493022` vs baseline `1193331`
    - `Overall TPS (Output): 349.58` vs baseline `316.72`
  - Read:
    - unlike the older winning-route bf16 experiment, this one did not collapse quality
    - the tradeoff here is different:
      - better state memory / token capacity
      - slightly better score
      - better output TPS
      - but clearly worse wall-clock duration because the model generated many more output tokens / tails
  - Current position:
    - `mamba_ssm_dtype=bfloat16` is now a legitimate runtime variant for the attnQKV gate0 line
    - it should not be promoted to default yet without a strategy for controlling the longer generations it induces
- 2026-04-20 14:10-14:50 CST: tested whether decode-side state locality can be improved just by sorting `SimpleGLA` requests by `mamba_indices` before gather/recurrent/writeback.
  - Implementation in probe tree:
    - extended `/Users/ql/cursor/openbmb/submission-w4a16-marlin-fusion-kvcache-probe-v1/bench_simple_gla_variants.py` with `clustered` and `sorted_random` index patterns
    - added a probe-only env flag in `/Users/ql/cursor/openbmb/submission-w4a16-marlin-fusion-kvcache-probe-v1/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`:
      - `SGLANG_SIMPLE_GLA_SORT_DECODE_BY_STATE=1`
      - behavior:
        - sort decode batch by `mamba_indices`
        - reorder `q/k/v`
        - gather/writeback in sorted slot order
        - invert the permutation on `o` before returning
  - Microbench result on `rtx6000-1` (decode-shaped `batch=128, seq_len=1`):
    - output dir:
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/simplegla_locality_20260420`
    - state gather/write and full-path timings were effectively tied across `random`, `sorted_random`, `clustered`, and `contiguous`
    - no meaningful locality win appeared at the isolated-kernel level
  - End-to-end extreme A/B on the current best attnQKV gate0 weight:
    - weight:
      - `/root/autodl-tmp/models-gptq-w4a16-py310-attnqkv-allowlist-v1-gate0-fixorder`
    - baseline:
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/extreme_attnqkv-gate0-sgla-base_20260420_143449`
      - `duration=384.75 s`
      - `total tok/s=10815.85`
      - `output tok/s=658.14`
    - sorted decode:
      - `/root/autodl-tmp/SOAR-Toolkit/test_results/extreme_attnqkv-gate0-sgla-sortdecode_20260420_144234`
      - `duration=389.29 s`
      - `total tok/s=10689.76`
      - `output tok/s=650.47`
    - net:
      - `~-1.17%` total throughput
      - `~-1.17%` output throughput
      - `~+1.18%` duration
  - Read:
    - the decode-side bottleneck is real, but naive slot-order sorting is not the right fix
    - the next credible step, if continuing, would be a deeper change such as fused indexed recurrent/state IO rather than wrapper-level batch reordering
