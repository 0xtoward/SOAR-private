# NGRAM spec v2 — verify==decode, batch-gate, and the bs=32 crash (MiniCPM-SALA NVFP4, 2026-05-26)

Builds on [[ngram-gla-state-fix]] (commit 5a2b209, `SPEC_NGRAM_FIX_GLA=1`, acc 22→77).
This note covers the two further fixes that make NGRAM spec **lossless + crash-free at
all concurrencies**, plus the final v2 package config.

Package: `submission-nvfp4-official4o6-gateup-spec-ngram-v2` (lean tar ~193M, mirrors the
v1 fulllencalib package; only difference vs v1 = engine swapped to spec_engine + spec args).
Quant/serving is **identical to v1** (same 4o6 model, `modelopt_fp4 --dtype float16
--force-dense-minicpm --cuda-graph-bs 1 8 32`); the 4o6 win is in the weights, engine-independent.

## Fix 1 — verify GLA kernel call == decode (`verify_match_decode.patch`)

Env: `SPEC_NGRAM_VERIFY_MATCH_DECODE=1`. File: `hybrid_linear_attn_backend.py`.

verify ran the GLA kernel in a *batched* layout (`[bs, step, H, D]`); decode runs it
*per-step varlen* (`[1, bs, H, D]` + `cu_seqlens=arange(bs+1)`). Same kernel, different
reduction order → small per-step diffs that **compound** along the recurrent state →
accepted-token hidden states drift from decode → ~6pt loss (verify 77 vs decode 83 @cap32k).
Fix: verify calls `fused_recurrent_simple_gla` per step in the identical varlen layout, then
reshapes back. Recovered ~77 → ~82 ≈ decode 83 (lossless within noise).
(This patch also carries the write-back capture guard at the post-verify state commit:
`if valid_mask is None or all_valid or torch.cuda.is_current_stream_capturing():` —
keeps the boolean-mask `index_copy_` legal under cudagraph capture.)

## Fix 2 — batch gate (`batch_gate.patch`)

Env: `SPEC_NGRAM_BATCH_GATE=8`. File: `ngram_worker.py`.

`forward_batch_generation`: if `batch_gate >= 0 and not is_extend() and batch_size() > gate`,
run `_gated_plain_decode(batch)` (sets `spec_algorithm=NONE`, `prepare_for_decode()`, runs the
target worker in plain DECODE, appends 1 tok/req) **instead of** spec-verify.

### Why it's needed — the bs=32 spec-verify crash
`--speculative-num-draft-tokens 12` gives the best S1, but spec **verify crashes at bs=32**
under long context:
`hybrid...init_forward_metadata_replay_cuda_graph → flashinfer_backend...call_begin_forward →
BatchPrefillWithPagedKVCacheWrapper.plan() → CUDA illegal memory access`.
verify uses flashinfer's **prefill** wrapper; its cudagraph replay metadata at bs=32 × 62K
context overflows. Plain decode (the **decode** wrapper, = v1's path) at bs=32 is fine — so the
bug is verify-specific. Gate → bs>8 never enters verify → no crash. (Spec helps most at low
batch / memory-bound; at high batch the GPU is compute-bound and spec barely helps anyway.)

### Note on the gated high-bs path (eager, but acc-safe)
In NGRAM mode `cuda_graph_runner` only captures TARGET_VERIFY graphs, so the gated `bs>8`
plain decode runs **eager**. Measured: this is NOT an acc regression here — eager decode at
conc=32 (78.67) is ≥ the no-spec **cudagraph** decode baseline (74.00). On this model the exact
recurrent kernel is, if anything, slightly more accurate than the bs=32 cudagraph approximation.
(If a future need wants high-bs on cudagraph decode, the engine would have to capture both
verify AND decode graphs — deep change in cuda_graph_runner; not needed today.)

## Final config (`prepare_env.sh`)

```
export SPEC_NGRAM_FIX_GLA=1
export SPEC_NGRAM_VERIFY_MATCH_DECODE=1
export SPEC_NGRAM_BATCH_GATE=8
SGLANG_SERVER_ARGS += --speculative-algorithm NGRAM --speculative-num-steps 3 \
  --speculative-num-draft-tokens 12 --speculative-ngram-branch-length 20 \
  --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 1
```
`--cuda-graph-bs 1 8 32` mandatory (accuracy-load-bearing). No `--max-running-requests` (model
self-caps ~48 concurrent: float32 GLA recurrent state is a fixed per-request cost; bs>~48 OOMs).

## Validation (local eval30 = 30 balanced RULER samples, greedy, cap32k)

| conc | path | acc | speed | crash |
|---|---|---|---|---|
| 1 (S1) | spec | 81.67 (decode ~83) | 6.23× vs no-spec, accept 11.97 | no |
| 8 | spec | 81.33 | 384 tok/s, accept 4.50 | no |
| 32 | gate→plain decode | 78.67 (≥ no-spec 74.00) | 477 s (no-spec 521 s) | no |
| 64 (bs→48) | gate→plain decode | — | 64/64 ok, 800 tok/s | no |

spec ≈/≥ no-spec at every concurrency (lossless). conc=32 raw drop vs conc=1 is intrinsic to
high-concurrency eval (hits spec AND no-spec equally), not a spec regression. Caveat: local
proxy (platform uses 150 samples / cap65536); conc=32/64 are single noisy runs (direction holds).

## Reproduce the package
`build_v2.sh` (extract lean v1 tar → swap sglang/python to spec_engine → pull v2 config/docs) +
`patch_v2_final.sh` (draft 8→12, add `SPEC_NGRAM_BATCH_GATE=8`). The two `.patch` files here are
the isolated engine diffs (vs pristine spec_engine `.bak` backups). C++ ngram trie JIT-compiles
at first import (needs g++ `-std=c++20` + ninja); `prepare_env.sh` warms it up fail-fast.
