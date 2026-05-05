# all-QKV189 bf16full115 runtime-bf16 overlay

This directory is a review-only overlay for the current SOAR MiniCPM-SALA
submission experiment. It contains only the non-SGLang files that explain the
submission behavior. It is not a full runnable submission package by itself.

## Purpose

The experiment tests whether the bf16-clean calibration path can support a
larger W4A16 GPTQ-Marlin quantization range without immediately changing the
SGLang source branch.

The package built locally from this overlay was:

```text
submission-w4a16-marlin-attnqkv-allqkv189-bf16full115-runtimebf16-v1.tar.gz
sha256: cf758adb8e0f5d342f486927a252604285486c74024ebe1a8449f331603fbc87
```

The generated tarball is intentionally **not** committed.

## What Changed

### Runtime

`prepare_env.sh` sets the serving route to:

```text
--attention-backend flashinfer
--force-dense-minicpm
--chunked-prefill-size 8192
--disable-radix-cache
--disable-cuda-graph
--cuda-graph-max-bs 1
--quantization gptq_marlin
--dtype bfloat16
```

No CUDA Graph, FP8 KV, FP4 KV, or 32K output cap is included in this overlay.

### Quantization

`prepare_model.sh` runs one primary quantization attempt:

```text
calibration: calibration_gptq_w4a16_publichybrid_semanticend49k_reduced_v1.jsonl
samples: 115
dtype: bfloat16
bits: 4
group_size: 128
batch_size: 1
dynamic config: configs/selective_marlin/attnqkv-allqkv189-bf16full115-mse24.json
memory flags: --offload-to-disk --vram-strategy exclusive --gc-mode on_stage_end --cpu-cache-layer-outputs
```

The calibration JSONL is not committed here because it is large and already
exists in the runnable local package.

### Dynamic Quantization Range

`configs/selective_marlin/attnqkv-allqkv189-bf16full115-mse24.json` targets:

```text
MLP: all gate_proj/up_proj/down_proj, except layers 15/16/17 down_proj
Attention: all 32 layers self_attn q_proj/k_proj/v_proj
Skipped dense: all self_attn o_proj
Scale search: mse = 2.4 on positive targets
```

Expected module count:

```text
MLP: 93 modules
QKV: 32 layers * 3 = 96 modules
Total: 189 W4A16 GPTQ-Marlin modules
```

### GPTQ Hygiene Patch

`quantize_gptq_w4a16.py` includes the current MiniCPM-SALA GPTQ compatibility
logic plus a finite-row filter around GPTQ Hessian construction. The filter is
a safety net:

```text
if a token row has NaN/Inf activation, drop the whole row before X^T X
do not convert NaN to zero
if bf16 calibration is clean, this path should record no drops
```

This was added because earlier fp16 calibration traces produced NaNs at
`layers.0.self_attn.output`; bf16 + flash_attention_2 was observed clean on the
48-sample source trace.

## Current Evidence

Known before this overlay:

```text
fp16 + flash_attention_2 source trace: 32/48 bad
bfloat16 + flash_attention_2 source trace: 0/48 bad
117-control bf16full115 hard gate: 117/117 true GPTQ, 0 RTN failsafe
117-control float16 runtime full eval: Average Score 79.73%
117-control bfloat16 runtime smoke: /v1/models 200, short generation 200
```

The all-QKV189 + runtime-bf16 package was prepared for external evaluator
submission; it had not been locally full-evaluated at branch creation time.

## Relationship To The SGLang Branch

SGLang source changes are intentionally kept out of this branch. Review them at:

```text
OpenBMB/sglang:minicpm_sala
  ...
0xtoward/sglang:codex/minicpm-sala-clean-diff-20260505
```

This overlay branch is for submission/package logic only.
