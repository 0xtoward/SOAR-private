# SGL Optimization Candidates

Current MiniCPM-SALA baseline already includes:
- fused `residual + RMSNorm`
- removed RoPE FP32 round-trip
- residual carried into final norm

Do not spend more time redoing those.

## Control-Group Gate

- If the near-stock base control also scores poorly on `smoke`, do not immediately interpret that as a runtime regression.
- Treat these patterns as likely eval-harness artifacts first:
  - failures concentrated in `mcq` while outputs are non-empty and look semantically correct
  - many `finish_reason=length` cases on otherwise short prompts
  - smoke score looks poor, but speed path is healthy and no server/runtime errors appear
  - the control and experimental route both fail the same tiny smoke subset
- In that case, the next action is not another runtime A/B.
- `official eval_model.py` is the quality source of truth.
- The lightweight local follow-up after official eval should be a stock-base `mini-eval` cross-check, not another `fast_test` smoke.

## Ranked Candidates

1. `--fuse-topk`
- Type: launch arg only
- Expected upside: high
- Risk: low-to-medium stability risk
- Landing area: `server_args.py`, `minicpm_backend.py`
- Status: one final post-patch attempt only
- Note: current fused path is FP16-oriented and has already hit two real runtime blockers
- Gate:
  - only one more rerun is justified, after syncing the bounded local call-path patch
  - if that rerun still fails in TileLang or fused-kernel init, demote immediately behind `--split-stage1`

2. `--split-stage1`
- Type: launch arg only
- Expected upside: medium-to-high on long prefill
- Risk: low accuracy risk, medium backend stability risk
- Landing area: `server_args.py`, `minicpm_backend.py`

3. `--chunked-prefill-size` sweep
- Type: launch arg only
- Try: `8192 -> 16384 -> 32768` only if memory is clean
- Expected upside: medium
- Risk: no accuracy risk, can increase memory / regress sparse preprocessing

4. Re-enable CUDA graph
- Type: launch arg only for first pass
- Change: remove `--disable-cuda-graph`
- Expected upside: medium on decode
- Risk: replay path still does `begin_forward()` + `torch.cuda.synchronize()`, so real gain may be small

5. `--dense-as-sparse` A/B
- Type: launch arg only first, code follow-up optional
- Expected upside: medium, workload-dependent
- Risk: no accuracy risk
- Note: with this enabled, dense-friendly requests still go through sparse path

6. Code-level backend cleanup
- Type: code edit
- Targets:
  - Python-side dense/sparse reshuffle loops in `minicpm_backend.py`
  - flashinfer CUDA-graph replay metadata update overhead
- Expected upside: medium
- Risk: medium correctness risk
- Cost: medium-to-high

## Benchmark Order

1. `--fuse-topk` exactly one more time, only after the bounded local call-path patch is synced
2. `--split-stage1` immediately after that if `--fuse-topk` still fails
3. `--fuse-topk --split-stage1` only if the standalone `--fuse-topk` rerun actually boots cleanly
4. best surviving candidate + `chunked-prefill-size` sweep
5. best surviving candidate + CUDA graph on/off
6. best surviving candidate + `dense-as-sparse` on/off
7. only then decide whether code edits are worth doing

## Control-Group Next Step

- If stock-base smoke is clearly healthy, continue with the current runtime candidate order.
- If stock-base smoke is unexpectedly weak or ambiguous and no official eval is running, use a stock-base `mini-eval` cross-check before any more runtime A/Bs.
- If official eval is already running, wait for it and then run this lightweight stock-base cross-check:

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

## Key Gotchas

- `fuse_topk` is no longer a pure launch-arg candidate in practice; it already needed bounded local runtime fixes.
- The remaining `fuse_topk` rerun is a last compatibility check, not an open-ended kernel debugging branch.
- `split_stage1` and CUDA-graph paths exist, but branch history suggests they were not trivial to stabilize.
- CUDA graph should be treated as an A/B measurement, not assumed to help.
- The next likely wins are in sparse-topk, prefill chunking, and graph replay behavior, not in `minicpm.py` block fusion.
