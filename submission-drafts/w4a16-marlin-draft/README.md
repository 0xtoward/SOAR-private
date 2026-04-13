# w4a16-marlin-draft

This is the current local draft submission package for the GPTQ W4A16 + Marlin route.

This draft is now organized as a formal dual-env submission:

- `prepare_env.sh` builds and exports the eval/serve env
- `prepare_model.sh` creates a separate quant env and runs online GPTQ there
- the only handoff between the two is the prepared model written to the platform `--output` path

## Route

- Runtime style: same as `v3-fusion-flashinfer`, with local editable `sglang/python`
- Quantization path: GPU-side `GPTQ` on the evaluation machine from `/models/MiniCPM-SALA`
- Runtime kernel: `gptq_marlin`
- Serve path: `--quantization gptq_marlin --dtype float16`
- Current eval fallback: `--attention-backend flashinfer --force-dense-minicpm`
- Submission env split:
  - eval env: `runtime_envs/eval_py310_env`
  - quant env: `runtime_envs/quant_py310_env`
- Current materialization strategy:
  1. try bundled `stop-aligned 64K` GPTQ calibration first
  2. if that fails, fall back to bundled `160K` GPTQ calibration
  3. if that still fails, fall back to bundled `32K PG19` GPTQ calibration
  4. always quantize on GPU during `prepare_model.sh`

## Why this package exists

The goal is to preserve a champion-aligned local-GPTQ-first package so we can:

- stop re-solving the Marlin packaging problem from scratch
- iterate on `prepare_env.sh` / `prepare_model.sh` without mixing in NVFP4 tooling
- compare a `gptq_marlin` candidate against stock base under the repaired `fast` and `medium` harnesses
- keep the submission self-contained instead of depending on a checkpoint that only exists on our own server

## Current measured status

Latest bounded `fast` status:

- stock base `fast`: `32.59%`
- `gptq_marlin` `fast` on the new py310 dense fallback eval env: `46.67%`

Current `medium` anchor:

- stock base `medium`:
  - `Original Accuracy: 50.40%`
  - `Normalized Accuracy: 63.0%`
  - `Total Duration: 2405.74 s`
  - `TPS: 178.88`
- `gptq_marlin` `medium`:
  - `Original Accuracy: 52.80%`
  - `Normalized Accuracy: 66.0%`
  - `Total Duration: 3640.09 s`
  - `TPS: 180.26`

Interpretation:

- this GPTQ route is currently a clear quality win over stock on the repaired bounded fast gate
- raw token throughput is roughly flat
- end-to-end duration is worse because the quantized model over-generates on uncapped `medium`
- for submission purposes, the important part is that py310 serve/eval now has a validated fallback route that can boot the quantized checkpoint

## Bundled Calibration

- `calibration_gptq_w4a16_stopaligned64k_v1.jsonl`
  - `64` rows
  - `36` SOAR public rows + `16` SOAR chat-close rows + `12` PG19 local rows
  - target:
    - preserve assistant-closing / `<|im_end|>` states without changing eval logic
- `calibration_gptq_w4a16_true160k_v1.jsonl`
  - `64` rows
  - `48` SOAR public rows + `16` PG19 long-form rows
  - token stats:
    - `p50=63410`
    - `p90=160000`
    - `max=160000`
- `calibration_gptq_w4a16_pg19_v2.jsonl`
  - `64` rows
  - `48` SOAR public rows + `16` PG19 long-form rows
  - token stats:
    - `p50=32768`
    - `p90=32768`
    - `max=32768`

## Package contents

- `prepare_env.sh`
- `prepare_model.sh`
- `submission_env_common.sh`
- `quantize_gptq_w4a16.py`
- `RESULTS.md`
- `SHA256SUMS.txt`
- `calibration_gptq_w4a16_true160k_v1.jsonl`
- `calibration_gptq_w4a16_stopaligned64k_v1.jsonl`
- `calibration_gptq_w4a16_pg19_v2.jsonl`
- `wheels/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`
- `wheels/gptqmodel-5.8.0+cu128torch2.9-cp310-cp310-linux_x86_64.whl`
- `vendor/fla/`
- `third_party/infllmv2_cuda_impl/`
- `sglang/python/`

## Important caveats

- The bundled `flash_attn` wheel is reused from the local cache and is not redownloaded during packaging. It is the same style used by `v3-fusion-flashinfer`, and is CPython 3.10 specific.
- The bundled `gptqmodel` wheel is also shipped locally because the domestic index did not provide the same build we validated.
- The dual-env scripts intentionally reuse the official evaluation image's built-in `torch` instead of rebundling it. This avoids both upload bloat and CUDA runtime mismatch risk.
- The top-level `fla/` package is bundled directly because the py310 eval path needed it but relying on online `flash-linear-attention` resolution was less stable than just carrying the already validated package tree.
- `prepare_env.sh` now creates a probe-like py310 eval env with `--system-site-packages`, installs only the serve/runtime delta, and exports that env for the later SGLang launch.
- `prepare_model.sh` now creates a separate py310 quant env with `--system-site-packages`, overlays the GPTQ-specific stack there, and runs online GPTQ without polluting the eval env.
- The current runtime patch keeps MiniCPM-SALA-specific `o_gate` and `z_proj` unquantized under `gptq` / `gptq_marlin`, which is required for the current local checkpoint to load in SGLang.
- `prepare_model.sh` now quantizes on the evaluation GPU instead of looking for a prebuilt checkpoint from our own server.
- Quantization-time model loading still uses the MiniCPM-SALA-specific `transformers 5.5.0` compatibility shims we validated locally, including the FA2 dispatch bridge and tied-weight save shim.
- The current primary submission candidate is the stop-aligned calibration route because it improved local py310 GPTQ results materially, but it is still a candidate rather than a fully closed final route.
- `infllm_v2` is bundled in two forms:
  - source tree for rebuild fallback
  - cp310 `.so` when available, so `prepare_model.sh` can try import-first before paying the full compile cost
- The currently validated py310 eval route is still a dense fallback (`flashinfer + force_dense-minicpm`) rather than the original `minicpm_flashinfer` sparse path.
- This draft intentionally avoids bundling NVFP4 / ModelOpt artifacts so the route is easier to reason about.
