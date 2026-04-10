# Local Results Snapshot

This draft preserves the current locally reproducible status of the conservative ModelOpt NVFP4 route.

## Harness

- Eval script: patched `fast_test.sh --eval-only`
- Purpose: local smoke only
- Caveat: not equivalent to the official 150-question public evaluation

## Current numbers

- Base model, `--dtype float16`: `28.00%`
- Current NVFP4 MLP-only export: `36.00%`

## Notes

- Base model needed `--dtype float16` to avoid a runtime dtype mismatch in the current environment.
- The current NVFP4 route is runnable, but several smoke samples still end with `finish_reason=length`, so stop behavior and accuracy both still need work.
- The previous official submission failed in `prepare_model.sh` because `pulp` was missing from the environment. This draft now bundles `pulp` and installs it conditionally in `prepare_env.sh`.
- A `16K bf16` sanity quantization with `8` curated samples completed successfully on `2026-04-09`, so the long-calibration route is now at least minimally verified before fallback.
- `prepare_model.sh` is now retry-based: `16K bf16` first, then `8K bf16`, then a safer `2048 fp16` fallback.
- These numbers are preserved here so future draft packages can be compared against the same local smoke baseline.
