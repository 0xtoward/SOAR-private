# spec_engine CUDA-graph capture fix on Blackwell sm120 (2026-05-25)

**File patched:** `spec_engine/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
**Patch:** `cudagraph_capture_fix_sm120_20260525.patch` (unified diff vs pre-fix backup)

## What it enables
Before: spec (EAGLE3) on MiniCPM-SALA could **not** capture a CUDA graph on Blackwell sm120
(`cudaErrorStreamCaptureInvalidated`) → spec forced to run eager. After: **topk=1 spec+cudagraph
captures `--cuda-graph-bs 1 8 32` fine** (verified: server comes up + serves S1/S8).

## The fix (topk=1 — 3 host-sync sites in the SimpleGLA decode/verify path)
All three were data-dependent host syncs illegal during graph capture; guarded with the
non-syncing query `torch.cuda.is_current_stream_capturing()`:
1. `if is_decode() or has_initial_state.any():` → add `or torch.cuda.is_current_stream_capturing()`
   so the `.any()` short-circuits (branch always entered during capture, which is what we want).
2. & 3. `all_valid = bool(valid_mask.all().item())` (two sites) → `False if is_current_stream_capturing()
   else bool(...)` → during capture force the **general `torch.where`-based path** (always correct;
   the `if all_valid:` fast path is only a sort optimization).

## topk>1 (tree-verify) — PARTIAL
`_build_tree_parent_tokens` rewritten to fixed-shape `gather`/`scatter_`/`where` (no dynamic
boolean-mask indexing). This is the GLA sibling-state fork (engine implements it correctly).
**Status: 1 of ≥2 capture syncs fixed — topk>1 capture still fails** (another sync remains; iterative).

## Findings this session (why spec still doesn't help the SOAR score)
- **CUDA graph is the dominant speed lever: 3.45× at S1** (no-spec cudagraph 92.9 vs eager 26.9 tok/s).
- **spec ≈ eager** per-forward; spec's win comes only from accept_len>break-even.
- **verify ≈ decode at batch=1** (weight-load-bound; K draft tokens ~free). `T_cycle≈16ms = 1 verify
  (≈1 decode) + 3 draft(~5.6ms)` → **break-even accept_len ≈ 1.5**.
- **v3 head is good on short-ctx reasoning: accept_len ≈ 2.95** (6k/8k/12k all ~2.95 → plateaued by
  ~8k; pick epoch_2_step_8000). At accept 2.95, spec+cg would win ~1.9× at S1.
- **BUT the SOAR eval is long-context (RULER 60k median) → accept_len → 1.0 → spec net-negative**
  (eval50: spec+cg acc 65.93/TPS 63 vs no-spec 67.40/78.54). So spec helps only short-ctx workloads.
- **Verdict: NVFP4 4o6 W4A4 quant (acc 98.44 / score 27.09) remains the optimal submission.**
  spec viability for a future target hinges entirely on that target's eval context length.
