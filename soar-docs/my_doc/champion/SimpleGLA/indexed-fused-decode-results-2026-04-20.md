# Indexed Fused Decode Results (2026-04-20)

## Scope

These notes summarize the decode-side `SimpleGLA` work on `rtx6000-1` using the current best `attnW + mlpW` weight:

- model: `/root/autodl-tmp/models-gptq-w4a16-py310-attnqkv-allowlist-v1-gate0-fixorder`
- route:
  - `flashinfer`
  - `--force-dense-minicpm`
  - `--chunked-prefill-size 8192`
  - `--disable-radix-cache`
  - `--disable-cuda-graph`
  - `--cuda-graph-max-bs 1`

## What Was Tried

### 1. Wrapper-level locality experiments

- Added `clustered` / `sorted_random` index patterns to the microbench.
- Added a decode-only sort-by-state wrapper experiment with:
  - `SGLANG_SIMPLE_GLA_SORT_DECODE_BY_STATE=1`
- Idea:
  - sort the decode batch by `mamba_indices`
  - gather/update/writeback in a more locality-friendly order
  - then restore output order

### 2. Decode-only indexed fused recurrent/state-io

- Added `SGLANG_SIMPLE_GLA_INDEXED_FUSED_DECODE=1`
- Instead of:
  - gather state into a temporary tensor
  - run recurrent kernel
  - scatter final state back
- The new fast path lets one Triton kernel:
  - read state from `layer_cache.temporal` using `mamba_indices`
  - do the recurrent update
  - write final state back in place
  - write the current output token

## Key Results

### Locality / sorting

Microbench had almost no signal.

`batch=128, seq_len=1`:

- `random`: `1.0860 ms`
- `sorted_random`: `1.0868 ms`
- `clustered`: `1.0845 ms`
- `contiguous`: `1.0851 ms`

Extreme end-to-end benchmark got worse:

- baseline:
  - `384.75 s`
  - `10815.85 total tok/s`
  - `658.14 output tok/s`
- sort-by-state:
  - `389.29 s`
  - `10689.76 total tok/s`
  - `650.47 output tok/s`

Conclusion:

- wrapper-level sort/reorder did not help
- likely not the right level to attack this bottleneck

### Indexed fused decode with float32 state

- baseline:
  - `384.75 s`
  - `10815.85 total tok/s`
  - `658.14 output tok/s`
  - `mean TTFT 100481 ms`
- indexed fused:
  - `366.90 s`
  - `11341.91 total tok/s`
  - `690.15 output tok/s`
  - `mean TTFT 103546 ms`

Delta:

- `+4.86%` total throughput
- `+4.86%` output throughput
- `-4.64%` duration
- TTFT slightly worse

### Indexed fused decode with bfloat16 state

- bf16 baseline:
  - `377.32 s`
  - `11028.92 total tok/s`
  - `671.11 output tok/s`
  - `mean TTFT 100781 ms`
- bf16 indexed fused:
  - `351.44 s`
  - `11840.88 total tok/s`
  - `720.52 output tok/s`
  - `mean TTFT 100876 ms`

Delta:

- `+7.36%` total throughput
- `+7.36%` output throughput
- `-6.86%` duration
- TTFT effectively flat

## Current Interpretation

- The useful decode bottleneck is real, but it is not solved by wrapper-level reordering.
- The bottleneck responds to removing explicit gather/scatter around recurrent update.
- The best current signal is:
  - `indexed fused decode`
  - together with `bfloat16` runtime state

## Practical Recommendation

- If continuing this line, carry the indexed fused decode fast path forward.
- Prefer evaluating it with `bfloat16` runtime state first.
- Keep `nsys` in the loop for system-level checks.
- Do not spend more time on prefill-side `SimpleGLA` tuning unless the global bottleneck picture changes.
