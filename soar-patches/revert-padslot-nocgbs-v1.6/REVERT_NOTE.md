# Revert pad-slot GLA-state patch → official OpenBMB minicpm_sala (v1.6, no-cgbs)

## What was reverted
The submission bundle's `sglang/.../layers/attention/hybrid_linear_attn_backend.py` carried a **custom cuda-graph-compat GLA-state patch** (NOT in official `OpenBMB/sglang@minicpm_sala`). Reverted to pristine official (HEAD `d29fb13`). The patch removed:
- `_indexed_simple_gla_decode_kernel` — custom Triton kernel reading/writing GLA recurrent state by per-request `slot` (`p_s = state_ptr + slot*SSN + ...`).
- `_replay_metadata` **dedicated-pad-slot** routing: `req_pool_indices[bs-num_padding:]=0; mamba_indices[bs-num_padding:]=-1` (comment: *"dedicated pad slot and keep the captured graph shape static"*).
- `fused_recurrent_max_seq_len = 80` (official uses `seq_len < 64`).

Origin: our commit `9ff1a7d "Add NVFP4 SALA spec_engine CUDA-graph capture fix (Blackwell sm120)"`.

The official reverted file is included here: `hybrid_linear_attn_backend.official.py`.

## Why (root cause of the cgbs→acc bug)
Clean platform A/B (byte-identical packages, only `--cuda-graph-bs` differs):
- WITH `--cuda-graph-bs 1 8 32`: acc **98.44**, durations S1/S8/Smax = 672/827/1857.
- WITHOUT the flag (default larger `cuda_graph_max_bs`): acc **95.39 (−3) → final_score 0**, durations 668/820/1830 (**speed unchanged**).

The pad-slot/state indexing is numerically correct **only at the captured `max_bs` it was tuned for = 32** (i.e. `--cuda-graph-bs 1 8 32`). Off that (default cgbs → larger max_bs), the `-1`/pad-slot assumption breaks → padding rows' garbage GLA state writes into real request slots → ~3% of requests corrupted. **It moves acc, not speed.**

## v1.6 package
`submission-nvfp4-4o6-v1.6-nocgbs-revert.tar.gz` = fulllencalib 98.44 recipe + this revert + **NO `--cuda-graph-bs` flag** (default cuda-graph). Hypothesis: official path is cgbs-robust → keeps ~98 acc at no-cgbs (+ marginally faster Smax via more cuda-graph coverage). To be confirmed on platform.

## NGRAM / spec-decode relevance (READ before merging)
The official file (now in v1.6) **STILL has** `if forward_mode.is_target_verify() and spec_info.topk > 1:` (~lines 255/515/576) — the exact line that crashed NGRAM: `'NgramVerifyInput' object has no attribute 'topk'`. So NGRAM on this base **still needs** the topk-safe `_needs_tree_custom_attn_mask(spec_info)` handling, which the spec branch (`codex/eagle-spec-minimal` / `tmp/sglang-openbmb-minicpm-sala`) **already has**.

⚠️ Do NOT blindly merge this revert onto the spec branch — it would drop the spec-verify state machinery (`update_simple_gla_state_after_target_verify`, mamba `SpeculativeState`, `_needs_tree_custom_attn_mask`). This revert is for the **non-spec W4A4 submission bundle only**.
