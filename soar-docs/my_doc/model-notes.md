# Model Notes

## SALA Architecture

- SALA is a hybrid model with both standard softmax attention layers and Lightning Attention / Simple GLA linear attention layers.
- SGLang wraps this with `HybridLinearAttnBackend`.
- SALA-specific linear paths include `o_gate` and `z_proj`, which matter for quantization and optimization coverage.

## Backend Trade-Off

| Backend | Behavior | Speed | Risk |
|---|---|---|---|
| `minicpm_flashinfer` | Sparse TopK behavior for standard attention | Baseline | Safest |
| `flashinfer` | Full attention on standard attention layers | Usually faster | May hurt accuracy |

## Confirmed Code-Level Optimizations

The most direct low-risk optimization points are:

1. Fuse residual add and RMSNorm using SGLang existing fused kernels.
2. Remove the extra FP32 round-trip around rotary embedding.

Current implementation target:

- File actually used in local SGLang source: `sglang/python/sglang/srt/models/minicpm.py`
- Submission copy: `submissions/v3-fusion-flashinfer/sglang/python/sglang/srt/models/minicpm.py`

## Quantization Notes

### RTN

- Very easy to try.
- Failed for us in v1 due to severe accuracy collapse.
- Latest selective RTN mixed-precision prototype can now quantize, load, and start a server, but local ultra-short eval still collapses to empty outputs and `0.00%` score.
- Conclusion: RTN was useful for proving the `prepare_model.sh` route, but is currently not a promising final direction.

### GPTQ

- Real calibration-based path.
- Needs 128-256 calibration samples.
- Likely stronger than RTN for preserving accuracy.
- Usually requires `--quantization gptq --dtype float16 --disable-cuda-graph`.

### AWQ

- More promising than RTN for the next serious attempt.
- Should be treated as the main post-RTN direction for `v4`.
- Likely needs calibration samples, selective protection of sensitive layers, and an offline conversion path that still fits the `prepare_model.sh` submission constraint.
- If AWQ tooling cannot load MiniCPM-SALA directly, prefer model-specific shard/tensor conversion over falling back to generic HF quantization flows.

### NVFP4

- Strong fit for Blackwell hardware.
- Potentially the strongest long-term path, but requires more source adaptation.

### Mixed Precision

- Keep sensitive layers in higher precision.
- Quantize the rest aggressively.
- `o_gate` and `z_proj` are likely sensitivity hotspots.
- First / last layers are also likely worth protecting.
- Recent RTN prototype suggests that “loadable” is much easier than “still produces valid text”, so mixed precision must be judged by output quality, not only by startup success.

## Champion Week 2 Takeaways

- LayerNorm + residual fusion is worth doing first because it is localized and low risk.
- Rotary FP32 removal is another targeted micro-optimization with little logic change.
- AI workflow works best when the planner and the executor are separated:
  one agent suggests directions, another one edits code and runs tests.

## Working Rules For Future Changes

- Do not change multiple high-risk dimensions at once.
- For backend changes, always re-run fast eval before trusting bench gains.
- When changing quantization or backend, keep `--disable-radix-cache` unless there is a very explicit reason not to.
- A quantized path that returns mostly empty strings should be treated as failed even if the server is healthy and latency looks good.
