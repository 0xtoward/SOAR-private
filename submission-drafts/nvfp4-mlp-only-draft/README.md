# nvfp4-mlp-only-draft

This is a local draft submission package for the current conservative ModelOpt NVFP4 path.

## Route

- Runtime style: same as `v3-fusion-flashinfer`, with local editable `sglang/python`
- Quantization path: `nvidia-modelopt`
- Config: `NVFP4_MLP_ONLY_CFG`
- Calibration: packaged local JSONL, `16K` first
- Serve path: `--quantization modelopt --dtype float16`

## Why this package exists

This package is not the final answer yet.

The goal is to preserve the current working state in a submission-shaped bundle so we can:

- stop re-downloading large wheels
- iterate on `prepare_env.sh` / `prepare_model.sh`
- compare future quantization branches against a stable draft

## Current local smoke status

Same-harness `fast_test.sh --eval-only` smoke results:

- Base model with `--dtype float16`: `28.00%`
- Current NVFP4 MLP-only export: `36.00%`

These are only local smoke results, not official final evaluation.

## Package contents

- `prepare_env.sh`
- `prepare_model.sh`
- `calibration/calibration_curated_16k_v2.jsonl`
- `RESULTS.md`
- `SHA256SUMS.txt`
- `quantize_modelopt_nvfp4_conservative.py`
- `calibration/calibration_data_merged.jsonl`
- `wheels/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`
- `wheels/nvidia_modelopt-0.42.0-py3-none-any.whl`
- `wheels/pulp-3.2.2-py3-none-any.whl`
- `sglang/python/`

## Important caveats

- The bundled `flash_attn` wheel is the same style used by `v3-fusion-flashinfer`, and is CPython 3.10 specific.
- This package assumes the official SOAR base image already provides the common SGLang runtime dependencies, and only adds the explicit extras we currently need.
- `prepare_model.sh` now tries `16K bf16` first, then `8K bf16`, then a known-safer `2048 fp16` fallback so the submission does not fail at model-preparation time.
- The current route should still be treated as a draft until it passes stronger local mini/full evaluation.
